#!/usr/bin/env python3
"""Make every figure with one command.

This script checks your setup, runs each step in the right order, skips any
step that can't run yet, and ends with a list of what is still left to do.

    python run_all.py                    check setup, then make every figure it can
    python run_all.py --check            only check setup and list what's missing
    python run_all.py --explore          exploration figures (build_features + fig_*.py)
    python run_all.py --pipeline         pipeline report figures (exposure.py ... report.py)
    python run_all.py --notebook         run data_exploration.ipynb -> reports/data_exploration.html
    python run_all.py fig_event_timing   run just the named step(s); see --list
    python run_all.py --rebuild          redo build_features.py even if its outputs exist
    python run_all.py --add-missing-columns
                                         if a clinical file lacks a column the scripts need
                                         (e.g. FACETS_WGD), fetch just that column from
                                         cBioPortal and add it (original kept as *.orig)

You can combine flags (e.g. --explore --notebook). You can run it from any
folder, because every path is taken from the folder this file is in.

ONE-TIME SETUP
--------------
1. Keep this file next to loaders.py, viz.py, common.py, the build_* / fig_*
   scripts and data_exploration.ipynb.
2. Unzip the three cBioPortal studies into
       raw/msk_met_2021/   raw/msk_impact_50k_2026/   raw/msk_chord_2024/
3. python -m pip install "pandas>=2.2" "matplotlib>=3.6" numpy pyarrow
       pipeline also needs: pyyaml scikit-learn
       notebook also needs: scipy lifelines upsetplot nbconvert ipykernel
4. Pipeline only: a config/ folder holding pipeline.yaml, drug_classes.yaml,
   target_variants.yaml, gene_aliases.csv and site_map.csv. Copy these from the
   original project; this script can't create them. If raw/panels/IMPACT*.json
   is missing, it is downloaded from cBioPortal automatically.

HOW LONG IT TAKES
-----------------
Exploration figures take under a minute once build_features has run. A full
pipeline run (exposure -> build_cohort -> build_alterations -> preprocess ->
report) takes about 10 minutes. The very first pipeline run also builds the
parquet cache in raw/cache/, which adds about 2 minutes.

OUTPUT
------
Figures, tables and the notebook HTML are written to reports/. Everything the
steps print is also saved to reports/run_all.log; send that file if a step fails.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
LOG_PATH = REPORTS / "run_all.log"
PY = sys.executable

STUDIES = {"raw/msk_met_2021": "MSK-MET", "raw/msk_impact_50k_2026": "MSK-IMPACT 50K",
           "raw/msk_chord_2024": "MSK-CHORD"}
PANELS = ["IMPACT341", "IMPACT410", "IMPACT468", "IMPACT505"]
API = "https://www.cbioportal.org/api"
PANEL_URL = API + "/gene-panels/{}"

# (import name, pip name, minimum version)
PACKAGES = {
    "explore": [("pandas", "pandas", (2, 2)), ("numpy", "numpy", None),
                ("matplotlib", "matplotlib", (3, 6)), ("pyarrow", "pyarrow", None)],
    "pipeline": [("yaml", "pyyaml", None), ("sklearn", "scikit-learn", None)],
    "notebook": [("scipy", "scipy", None), ("lifelines", "lifelines", None),
                 ("upsetplot", "upsetplot", None), ("nbconvert", "nbconvert", None),
                 ("ipykernel", "ipykernel", None)],
}


def met(*f): return [f"raw/msk_met_2021/{x}" for x in f]
def imp(*f): return [f"raw/msk_impact_50k_2026/{x}" for x in f]
def chord(*f): return [f"raw/msk_chord_2024/{x}" for x in f]
def cfg(*f): return [f"config/{x}" for x in f]


PANEL_FILES = [f"raw/panels/{p}.json" for p in PANELS]
EVENT_NEEDS = (met("data_clinical_sample.txt", "data_timeline_sequencing.txt",
                   "data_timeline_diagnosis.txt")
               + imp("data_clinical_sample.txt")
               + chord("data_timeline_specimen_surgery.txt", "data_timeline_diagnosis.txt",
                       "data_timeline_treatment.txt"))


@dataclass
class Step:
    name: str
    group: str                       # explore | pipeline | notebook
    cmd: list[str]
    code: list[str]                  # local scripts it runs or imports
    needs: list[str]                 # data / config files it reads
    makes: list[str]                 # files it must produce
    after: list[str] = field(default_factory=list)
    once: bool = False               # skip if outputs already exist (unless --rebuild)


STEPS = [
    # ---- exploration figures -------------------------------------------------
    Step("build_features", "explore", [PY, "build_features.py"],
         ["build_features.py", "loaders.py"],
         met("data_clinical_sample.txt", "data_clinical_patient.txt",
             "data_timeline_diagnosis.txt", "data_timeline_sequencing.txt")
         + imp("data_clinical_sample.txt", "data_mutations.txt", "data_armlevel_cna.txt")
         + chord("data_timeline_treatment.txt"),
         ["derived/het_features.parquet", "derived/cohort.parquet",
          "derived/treatment.parquet"], once=True),
    Step("fig_cohort_overview", "explore", [PY, "fig_cohort_overview.py"],
         ["fig_cohort_overview.py", "loaders.py", "viz.py"],
         met("data_clinical_sample.txt") + imp("data_clinical_sample.txt")
         + chord("data_clinical_sample.txt", "data_timeline_treatment.txt",
                 "data_timeline_radiation.txt"),
         ["reports/fig_cohort_overview.png", "reports/fig_metastatic_spread.png"],
         after=["build_features"]),
    *[Step(n, "explore", [PY, f"{n}.py"],
           list(dict.fromkeys([f"{n}.py", "fig_event_order.py", "loaders.py", "viz.py"])),
           EVENT_NEEDS, [f"reports/{n}.png"])
      for n in ("fig_event_order", "fig_event_sankey", "fig_event_timing")],

    # ---- pipeline report figures (each stage reads the previous one) --------
    Step("exposure", "pipeline", [PY, "exposure.py"],
         ["exposure.py", "common.py", "loaders.py"],
         chord("data_timeline_treatment.txt", "data_clinical_patient.txt",
               "data_clinical_sample.txt", "data_timeline_diagnosis.txt",
               "data_timeline_tumor_sites.txt", "data_timeline_specimen_surgery.txt")
         + cfg("pipeline.yaml", "drug_classes.yaml"),
         ["derived/exposure.parquet"]),
    Step("build_cohort", "pipeline", [PY, "build_cohort.py"],
         ["build_cohort.py", "exposure.py", "common.py", "loaders.py"],
         chord("data_timeline_specimen_surgery.txt", "data_clinical_sample.txt",
               "data_clinical_patient.txt", "data_timeline_treatment.txt")
         + imp("data_clinical_sample.txt", "data_clinical_patient.txt",
               "data_gene_panel_matrix.txt", "data_cna.txt", "data_mutations.txt")
         + cfg("pipeline.yaml", "drug_classes.yaml"),
         ["derived/samples.parquet", "derived/samples_A-on.parquet"], after=["exposure"]),
    Step("build_alterations", "pipeline", [PY, "build_alterations.py"],
         ["build_alterations.py", "build_cohort.py", "exposure.py", "common.py", "loaders.py"],
         imp("data_gene_panel_matrix.txt", "data_clinical_sample.txt", "data_mutations.txt",
             "data_cna.txt", "data_CNA.ascna.txt", "data_cna_hg19.seg", "data_sv.txt",
             "data_mutational_signatures_contribution_v2.txt")
         + PANEL_FILES
         + cfg("pipeline.yaml", "drug_classes.yaml", "target_variants.yaml", "gene_aliases.csv"),
         ["derived/alteration_matrix.parquet", "derived/genomic_features.parquet"],
         after=["build_cohort"]),
    Step("preprocess", "pipeline", [PY, "preprocess.py"],
         ["preprocess.py", "common.py", "loaders.py"],
         cfg("pipeline.yaml", "site_map.csv", "target_variants.yaml"),
         ["derived/model_table_pooled.parquet", "derived/feature_dictionary.csv"],
         after=["build_alterations"]),
    Step("report", "pipeline", [PY, "report.py"],
         ["report.py", "exposure.py", "build_alterations.py", "common.py", "loaders.py", "viz.py"],
         chord("data_timeline_treatment.txt") + PANEL_FILES
         + cfg("pipeline.yaml", "drug_classes.yaml", "site_map.csv"),
         ["reports/F1_attrition.png", "reports/F2_sampling_time.png",
          "reports/F4_covariate_balance.png", "reports/F5_panel_confounding.png",
          "reports/F6_controls.png", "reports/README.md"],
         after=["preprocess"]),

    # ---- notebook, executed to HTML so its figures can be viewed anywhere ----
    Step("notebook", "notebook",
         [PY, "-m", "nbconvert", "--to", "html", "--execute",
          "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=python3",
          "--output-dir", "reports", "data_exploration.ipynb"],
         ["data_exploration.ipynb", "loaders.py", "viz.py"],
         list(STUDIES), ["reports/data_exploration.html"], after=["build_features"]),
]
BY_NAME = {s.name: s for s in STEPS}

# Columns the scripts select by name from the clinical files. They are checked
# against each file's header line before anything runs, so an older download
# fails in seconds instead of minutes into the pipeline.
_FACETS = ["SAMPLE_ID", "FACETS_QC", "FACETS_PURITY", "FACETS_PLOIDY", "FACETS_WGD",
           "TMB_SCORE", "MSI_SCORE", "MSI_TYPE"]
REQUIRED_COLUMNS = {
    "raw/msk_impact_50k_2026/data_clinical_sample.txt": {
        "build_features": _FACETS + ["GENE_PANEL", "CANCER_TYPE", "SAMPLE_TYPE", "DISEASE_STATUS"],
        "build_cohort": _FACETS,
        "build_alterations": _FACETS,
        **{n: ["SAMPLE_ID", "PATIENT_ID"] for n in
           ("fig_cohort_overview", "fig_event_order", "fig_event_sankey", "fig_event_timing")},
    },
    "raw/msk_impact_50k_2026/data_clinical_patient.txt": {"build_cohort": ["PATIENT_ID", "SEX"]},
    "raw/msk_chord_2024/data_clinical_sample.txt": {
        "build_cohort": ["SAMPLE_ID", "PATIENT_ID", "CANCER_TYPE", "SAMPLE_TYPE",
                         "METASTATIC_SITE", "PRIMARY_SITE", "GENE_PANEL"]},
    "raw/msk_chord_2024/data_clinical_patient.txt": {
        "build_cohort": ["PATIENT_ID", "GENDER", "HR", "HER2", "PRIOR_MED_TO_MSK",
                         "CURRENT_AGE_DEID"]},
}

HINTS = [
    ("CHECKLIST FAILED", "One of the script's built-in data checks failed, so it stopped "
                         "on purpose instead of writing bad data. The CHECKLIST FAILED line "
                         "says which check."),
    ("ModuleNotFoundError", "A package is missing. Install the module named in the error with "
                            "python -m pip install <name>."),
    ("MemoryError", "The computer ran out of memory. Close other programs and try again; the "
                    "large MSK-IMPACT files need several GB of RAM."),
    ("Unmapped agents", "Add the drug names listed in the error to config/drug_classes.yaml."),
    ("config/site_map.csv", "Add the biopsy sites listed in the error to config/site_map.csv."),
    ("FileNotFoundError", "A file is missing, or its name differs from what loaders.py expects. "
                          "Check the path printed in the error."),
    ("not in index", "Your data file doesn't have a column the script needs (named in the "
                     "error). Your copy of the study is probably an older cBioPortal release. "
                     "Download the study again, then run: python run_all.py --rebuild"),
    ("KeyError", "The data is missing a field the script expects. Your copy of the study may "
                 "be a different cBioPortal release from the one the scripts were written for."),
    ("formatargspec", "An old copy of the 'wrapt' package doesn't work with your Python version. "
                      "Run: python -m pip install --upgrade wrapt"),
    ("NoSuchKernel", "Run: python -m ipykernel install --user"),
]


def hints_for(text: str) -> list[str]:
    return list(dict.fromkeys(h for key, h in HINTS if key in text))


def summarize_failure(lines: list[str]) -> list[str]:
    """Pick out where it failed (the user's own script or notebook cell) and the error."""
    err = next((ln.strip() for ln in reversed(lines)
                if re.match(r"\s*([\w.]+(Error|Exception|Exit)|CHECKLIST FAILED)\b", ln)), None)
    here = str(HERE).lower()
    where = []
    for i in range(len(lines) - 1, -1, -1):
        ln = lines[i]
        m = re.search(r'File "([^"]+)", line (\d+)', ln)
        if m and m.group(1).lower().startswith(here):
            code = lines[i + 1].strip() if i + 1 < len(lines) else ""
            where = [f"at {Path(m.group(1)).name}, line {m.group(2)}:  {code[:110]}"]
            break
        m = re.search(r"Cell In\[(\d+)\], line (\d+)", ln)
        if m:
            code = next((re.sub(r"^\d+\s+", "", l.split("---->", 1)[1].strip())
                         for l in lines[i + 1:i + 6] if "---->" in l), "")
            where = [f"in notebook cell [{m.group(1)}], line {m.group(2)}:  {code[:110]}"]
            break
    out = where + ([err] if err else [])
    return out or [ln for ln in lines[-8:]]

_log = None


def say(msg: str = "") -> None:
    print(msg, flush=True)
    if _log:
        _log.write(msg + "\n")
        _log.flush()


def exists(rel: str) -> bool:
    return (HERE / rel).exists()


def fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


# --------------------------------------------------------------------------
# Setup checks
# --------------------------------------------------------------------------

def check_packages(groups: set[str]) -> dict[str, tuple[str, str]]:
    """Return {pip spec: (kind, message)} for every missing, too-old or broken package.

    Each package is actually imported, because "installed" is not enough: an
    old dependency can make a package fail on import (e.g. lifelines -> wrapt).
    """
    bad = {}
    for g in ["explore", *sorted(groups - {"explore"})]:     # every group needs the base set
        for mod, pip, minv in PACKAGES[g]:
            spec = f"{pip}>={'.'.join(map(str, minv))}" if minv else pip
            if importlib.util.find_spec(mod) is None:
                bad[spec] = ("missing", "not installed")
                say(f"  [!!] {pip:<13} not installed")
                continue
            ver = ""
            try:
                ver = importlib.metadata.version(pip)
            except importlib.metadata.PackageNotFoundError:
                pass
            have = tuple(int(x) for x in re.findall(r"\d+", ver)[:2])
            if minv and have and have < minv:
                bad[spec] = ("old", f"version {ver} is too old")
                say(f"  [!!] {pip:<13} {ver} is too old, need >= {'.'.join(map(str, minv))}")
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    importlib.import_module(mod)
            except Exception as e:                               # broken install / dependency
                msg = f"{type(e).__name__}: {e}".splitlines()[0]
                bad[spec] = ("broken", msg)
                say(f"  [!!] {pip:<13} {ver} is installed but won't load: {msg}")
                continue
            say(f"  [ok] {pip:<13} {ver}")
    return bad


def get_json(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "run_all.py"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def download_panels() -> None:
    missing = [p for p in PANELS if not exists(f"raw/panels/{p}.json")]
    if not missing:
        return
    say("Downloading gene-panel lists from cBioPortal:")
    for p in missing:
        try:
            data = get_json(PANEL_URL.format(p))
            if not data.get("genes"):
                raise ValueError("the response had no gene list")
            out = HERE / "raw" / "panels" / f"{p}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(data), encoding="utf-8")
            say(f"  [ok] raw/panels/{p}.json ({len(data['genes'])} genes)")
        except Exception as e:                                   # network, proxy, bad JSON
            say(f"  [!!] {p}: {e}")


def add_missing_columns(bad_cols: dict[str, dict[str, list[str]]]) -> None:
    """Fetch columns missing from a clinical file from cBioPortal and append them.

    Only the missing columns are downloaded (the study id is the folder name).
    The original file is kept next to it as <name>.orig the first time.
    """
    todo: dict[str, set[str]] = {}
    for per_file in bad_cols.values():
        for rel, cols in per_file.items():
            todo.setdefault(rel, set()).update(cols)
    for rel, cols in todo.items():
        path = HERE / rel
        study = path.parent.name
        patient = "patient" in path.name
        id_col = "PATIENT_ID" if patient else "SAMPLE_ID"
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        n_meta = next(i for i, ln in enumerate(lines) if not ln.startswith("#"))
        header = lines[n_meta].split("\t")
        if id_col not in header:
            say(f"  [!!] {rel} has no {id_col} column, so values can't be matched; skipped")
            continue
        idx, width = header.index(id_col), len(header)
        added = []
        for attr in sorted(cols):
            say(f"  fetching {attr} for {study} from cBioPortal (can take a minute) ...")
            try:
                recs = get_json(f"{API}/studies/{study}/clinical-data?clinicalDataType="
                                f"{'PATIENT' if patient else 'SAMPLE'}&attributeId={attr}"
                                f"&projection=SUMMARY&pageSize=10000000", timeout=600)
                meta = get_json(f"{API}/studies/{study}/clinical-attributes/{attr}")
            except Exception as e:
                say(f"  [!!] could not fetch {attr}: {e}")
                continue
            vals = {r["patientId" if patient else "sampleId"]: str(r["value"]) for r in recs}
            if not vals:
                say(f"  [!!] cBioPortal has no {attr} values for {study}; not added")
                continue
            head = [meta.get("displayName", attr), meta.get("description", attr),
                    meta.get("datatype", "STRING"), str(meta.get("priority", 1))]
            for i in range(n_meta):
                lines[i] += "\t" + (head[i] if i < len(head) else "")
            lines[n_meta] += "\t" + attr
            rows = filled = 0
            for j in range(n_meta + 1, len(lines)):
                if not lines[j].strip():
                    continue
                f = lines[j].split("\t")
                f += [""] * (width - len(f))                 # keep the new column aligned
                v = vals.get(f[idx], "")
                lines[j] = "\t".join(f) + "\t" + v
                rows += 1
                filled += bool(v)
            width += 1
            added.append(attr)
            say(f"  [ok] {attr}: values for {filled:,} of {rows:,} rows "
                f"({len(vals):,} in cBioPortal's current release)")
        if added:
            backup = path.with_name(path.name + ".orig")
            if not backup.exists():
                backup.write_bytes(path.read_bytes())
                say(f"  original kept as {rel}.orig")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def header_columns(rel: str) -> list[str]:
    """Column names of a cBioPortal clinical file: the first line not starting with '#'."""
    with open(HERE / rel, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                return [c.strip() for c in line.rstrip("\r\n").split("\t")]
    return []


def check_columns(steps: list[Step]) -> dict[str, dict[str, list[str]]]:
    """Return {step: {file: [missing columns]}} and print one line per file."""
    names = {s.name for s in steps}
    missing: dict[str, dict[str, list[str]]] = {}
    for rel, per_step in REQUIRED_COLUMNS.items():
        need = sorted({c for n, cols in per_step.items() if n in names for c in cols})
        if not need or not exists(rel):
            continue
        have = header_columns(rel)
        gone = [c for c in need if c not in have]
        if not gone:
            say(f"  [ok] {rel}")
            continue
        for n, cols in per_step.items():
            lost = [c for c in cols if c in gone]
            if n in names and lost:
                missing.setdefault(n, {})[rel] = lost
        for c in gone:
            token = c.split("_")[-1].upper()
            similar = list(dict.fromkeys(difflib.get_close_matches(c, have, n=3, cutoff=0.6)
                                         + [h for h in have if token in h.upper()]))
            say(f"  [!!] {rel} has no {c} column; "
                + (f"similar columns it does have: {', '.join(similar[:6])}" if similar
                   else "no similarly named column"))
    return missing


def blockers(step: Step, bad_pkgs: dict[str, tuple[str, str]], status: dict[str, str],
             selected: set[str], ignore: frozenset[str] = frozenset(),
             bad_cols: dict[str, dict[str, list[str]]] | None = None) -> dict[str, list[str]]:
    """Everything that stops `step` from running right now (empty dict = ready)."""
    out = {}
    cols = [f"{Path(f).name} ({', '.join(c)})"
            for f, c in (bad_cols or {}).get(step.name, {}).items()]
    code = [f for f in step.code if not exists(f)]
    files = [f for f in step.needs if not exists(f) and f not in ignore]
    groups = {"explore", step.group}
    pkgs = [spec for g in groups for _, pip, _ in PACKAGES[g]
            for spec in bad_pkgs if spec.split(">=")[0] == pip]
    deps = []
    for d in step.after:
        if status.get(d) in ("ok", "built", "ready"):
            continue
        if d not in selected and all(exists(m) for m in BY_NAME[d].makes):
            continue
        deps.append(d)
    for k, v in [("code", code), ("files", files), ("packages", pkgs), ("columns", cols),
                 ("after", deps)]:
        if v:
            out[k] = v
    return out


def describe(b: dict[str, list[str]]) -> str:
    parts = []
    if "packages" in b:
        parts.append("needs packages " + ", ".join(b["packages"]))
    if "code" in b:
        parts.append("missing " + ", ".join(b["code"]))
    if "files" in b:
        f = b["files"]
        parts.append(f"missing {len(f)} input file{'s' * (len(f) > 1)} (e.g. {f[0]})")
    if "columns" in b:
        parts.append("data lacks columns: " + "; ".join(b["columns"]))
    if "after" in b:
        parts.append("needs " + ", ".join(b["after"]) + " to finish first")
    return "; ".join(parts)


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def run(step: Step) -> tuple[int, list[str]]:
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
           "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}
    tail: deque[str] = deque(maxlen=400)       # enough to find the traceback's own frames
    p = subprocess.Popen(step.cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", bufsize=1)
    try:
        for line in p.stdout:
            line = ANSI.sub("", line.rstrip("\n"))   # nbconvert colours its tracebacks
            print("    " + line, flush=True)
            if _log:
                _log.write("    " + line + "\n")
            if line.strip():
                tail.append(line)
        return p.wait(), list(tail)
    except KeyboardInterrupt:
        p.kill()
        raise


def main() -> int:
    global _log
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    if sys.version_info < (3, 9):
        print(f"Python {sys.version.split()[0]} is too old: install Python 3.10 or newer.")
        return 1

    ap = argparse.ArgumentParser(description="Make every figure in order and list what's left to do.")
    ap.add_argument("steps", nargs="*", help="run only these steps (see --list)")
    ap.add_argument("--check", action="store_true", help="only check setup, run nothing")
    ap.add_argument("--explore", action="store_true", help="exploration figures")
    ap.add_argument("--pipeline", action="store_true", help="pipeline report figures")
    ap.add_argument("--notebook", action="store_true", help="execute the notebook to HTML")
    ap.add_argument("--rebuild", action="store_true", help="redo build_features.py")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    ap.add_argument("--add-missing-columns", action="store_true",
                    help="fetch clinical columns your data files lack from cBioPortal and add them")
    a = ap.parse_args()

    if a.list:
        for s in STEPS:
            print(f"  {s.name:<20} {s.group:<9} -> {', '.join(Path(m).name for m in s.makes)}")
        return 0
    unknown = [n for n in a.steps if n.removesuffix(".py") not in BY_NAME]
    if unknown:
        ap.error(f"unknown step(s) {unknown}; choose from {list(BY_NAME)}")
    named = {n.removesuffix(".py") for n in a.steps}
    groups = {g for g in ("explore", "pipeline", "notebook") if getattr(a, g)}
    if named:
        steps = [s for s in STEPS if s.name in named]
    else:
        steps = [s for s in STEPS if s.group in (groups or {"explore", "pipeline", "notebook"})]
    selected = {s.name for s in steps}
    sel_groups = {s.group for s in steps}

    if not a.check:
        REPORTS.mkdir(exist_ok=True)
        _log = open(LOG_PATH, "w", encoding="utf-8")
        say(f"run_all.py  {datetime.now():%Y-%m-%d %H:%M}  args: {' '.join(sys.argv[1:]) or '(none)'}")

    # ---- setup check -----------------------------------------------------------
    say(f"\n== Setup check ==\nPython {sys.version.split()[0]}  ({PY})\nPackages:")
    bad_pkgs = check_packages(sel_groups)
    say("Data folders:")
    for folder, label in STUDIES.items():
        say(f"  [{'ok' if exists(folder) else '!!'}] {folder:<26} {label}")
    if "pipeline" in sel_groups:
        if not a.check and any(f in PANEL_FILES for s in steps for f in s.needs):
            download_panels()
        cfg_missing = [f for f in cfg("pipeline.yaml", "drug_classes.yaml", "target_variants.yaml",
                                      "gene_aliases.csv", "site_map.csv") if not exists(f)]
        say(f"  [{'!!' if cfg_missing else 'ok'}] config/  "
            + (f"missing {len(cfg_missing)} of 5 files" if cfg_missing else "all 5 files present"))
        pmiss = [p for p in PANEL_FILES if not exists(p)]
        say(f"  [{'!!' if pmiss else 'ok'}] raw/panels/  "
            + (f"missing {len(pmiss)} of 4 panel files"
               + (" (a normal run downloads them)" if a.check else "") if pmiss
               else "all 4 panel files present"))
    # In --check mode the panels aren't downloaded yet, so don't count them as blockers.
    ignore = frozenset(PANEL_FILES) if a.check else frozenset()
    say("Data columns:")
    bad_cols = check_columns(steps)
    if bad_cols and a.add_missing_columns and not a.check:
        say("Adding the missing columns from cBioPortal:")
        add_missing_columns(bad_cols)
        say("Data columns, re-checked:")
        bad_cols = check_columns(steps)

    # ---- plan / run ------------------------------------------------------------
    status, why, tails, took = {}, {}, {}, {}
    t_start = time.time()
    say(f"\n== {'Plan (nothing is run in --check mode)' if a.check else 'Running'} ==")
    if "pipeline" in sel_groups and not a.check:
        first = not (HERE / "raw" / "cache").exists()
        say("The pipeline steps take about 10 minutes in total"
            + (" (plus about 2 minutes this first time, to build the parquet cache)" if first
               else "") + ".")
    for i, s in enumerate(steps, 1):
        head = f"[{i}/{len(steps)}] {s.name}"
        # Existing outputs win over blockers, so later steps can still use them.
        if s.once and not a.rebuild and s.name not in named and all(exists(m) for m in s.makes):
            status[s.name] = "built"
            say(f"{head}: already built (use --rebuild to redo)")
            continue
        b = blockers(s, bad_pkgs, status, selected, ignore, bad_cols)
        if b:
            status[s.name], why[s.name] = "skipped", b
            say(f"{head}: SKIPPED - {describe(b)}")
            continue
        if a.check:
            status[s.name] = "ready"
            say(f"{head}: ready")
            continue
        say(f"{head}: running  ({' '.join(Path(c).name if c == PY else c for c in s.cmd)})")
        t0 = time.time()
        try:
            code, tail = run(s)
        except KeyboardInterrupt:
            say("\nStopped by Ctrl+C.")
            return 130
        took[s.name] = time.time() - t0
        fresh = all(exists(m) and (HERE / m).stat().st_mtime >= t0 - 2 for m in s.makes)
        if code == 0 and fresh:
            status[s.name] = "ok"
            say(f"{head}: done in {fmt_time(took[s.name])}")
        else:
            status[s.name] = "failed"
            tails[s.name] = tail if code else tail + ["(finished without error but did not "
                                                      "write all its output files)"]
            say(f"{head}: FAILED after {fmt_time(took[s.name])} (exit code {code})")

    # ---- summary ---------------------------------------------------------------
    figs = [m for s in steps for m in s.makes if m.endswith((".png", ".html"))]
    if not a.check:
        say("\n== Summary ==")
        label = {"ok": "ok", "built": "ok", "skipped": "skipped", "failed": "FAILED"}
        for s in steps:
            st = status[s.name]
            note = {"ok": fmt_time(took.get(s.name, 0)), "built": "already built",
                    "skipped": describe(why.get(s.name, {})), "failed": "see below"}[st]
            say(f"  {label[st]:<8} {s.name:<20} {note}")
        if figs:
            say("\nFigures in reports/  (new = made by this run, old = left over from an earlier "
                "run and not updated):")
        for f in figs:
            p = HERE / f
            tag = ("new" if p.stat().st_mtime >= t_start - 2 else "old") if p.exists() else "missing"
            say(f"  [{tag:<7}] {Path(f).name}")

    # ---- what's left -----------------------------------------------------------
    todo = []
    install = sorted(s for s, (kind, _) in bad_pkgs.items() if kind != "broken")
    if install:
        todo.append("Install the missing packages (with the same Python you run this script with):\n"
                    "     python -m pip install " + " ".join(f'"{p}"' for p in install))
    for spec, (kind, msg) in bad_pkgs.items():
        if kind == "broken":
            fix = hints_for(msg) or [f"Reinstall it: python -m pip install --upgrade "
                                     f"--force-reinstall {spec}"]
            todo.append(f"{spec} is installed but won't load ({msg}).\n     Fix: {fix[0]}")
    code_missing = sorted({f for b in why.values() for f in b.get("code", [])})
    if code_missing:
        todo.append("Put these files in the same folder as run_all.py:\n     "
                    + ", ".join(code_missing))
    files = sorted({f for b in why.values() for f in b.get("files", [])})
    for folder, lab in STUDIES.items():
        if not exists(folder) and any(f.startswith(folder) or f == folder for f in files):
            todo.append(f"Download the {lab} study from cBioPortal (study id "
                        f"'{Path(folder).name}', Download button on the study page) and unzip it "
                        f"into {folder}/")
        else:
            inside = [f for f in files if f.startswith(folder + "/")]
            if inside:
                todo.append(f"These files are missing from {folder}/ (re-download the study, or "
                            f"check the file names):\n     "
                            + ", ".join(Path(f).name for f in inside))
    cfgs = [f for f in files if f.startswith("config/")]
    if cfgs:
        todo.append("Copy the pipeline's config files into config/ from the original project. They "
                    "hold the drug-class, time-window and site mappings and can't be generated here:\n"
                    "     " + ", ".join(Path(f).name for f in cfgs))
    lacking: dict[str, set[str]] = {}
    for per_file in bad_cols.values():
        for rel, cols in per_file.items():
            lacking.setdefault(rel, set()).update(cols)
    for rel, cols in lacking.items():
        folder = str(Path(rel).parent.as_posix())
        todo.append(f"{rel} has no {', '.join(sorted(cols))} column (see 'Data columns' above). "
                    f"Your copy of the {STUDIES.get(folder, 'study')} study is probably an older "
                    f"cBioPortal release. Either:\n"
                    f"     a) fetch just the missing column (quick, keeps a backup of the file):\n"
                    f"          python run_all.py --add-missing-columns\n"
                    f"     b) or download the whole study again (study id '{Path(folder).name}'), "
                    f"unzip it over {folder}/,\n"
                    f"        then run: python run_all.py --rebuild")
    pans = [f for f in files if f.startswith("raw/panels/")]
    if pans:
        todo.append("The automatic panel download didn't work. Open each link below in a browser "
                    "and save the page as raw/panels/<NAME>.json:\n"
                    + "\n".join(f"     {PANEL_URL.format(Path(p).stem)}" for p in pans))
    for s in steps:
        b = why.get(s.name, {})
        for d in b.get("after", []):
            if d not in selected:
                cmd = "--pipeline" if BY_NAME[d].group == "pipeline" else d
                todo.append(f"{s.name} needs {d}'s output first. Run: python run_all.py {cmd}")
    for name, tail in tails.items():
        hints = hints_for("\n".join(tail))
        todo.append(f"{name} failed\n"
                    + "\n".join(f"       {ln}" for ln in summarize_failure(tail))
                    + "".join(f"\n     Hint: {h}" for h in hints[:1])
                    + f"\n     Full output: {LOG_PATH.relative_to(HERE)}")

    say("\n== What's left to do ==")
    if todo:
        for i, t in enumerate(todo, 1):
            say(f"  {i}. {t}")
        say("\nAfter fixing these, run python run_all.py again (python run_all.py --check "
            "re-checks without running anything). build_features is skipped if its output "
            "already exists; the other steps run again.")
    elif a.check:
        say("  Nothing. Setup is complete; run python run_all.py to make the figures.")
    else:
        say(f"  Nothing. All {len(figs)} figures are in reports/." if figs else "  Nothing.")
    if _log:
        _log.close()
    return 1 if any(v == "failed" for v in status.values()) else 0


if __name__ == "__main__":
    sys.exit(main())