"""Stage 2: treatment exposure (data_cleaning_plan.md Section 6).

Writes derived/exposure.parquet, one row per patient x cohort class:
class_start, merged episodes, prior / concurrent classes, subclass, companion
flags, primary diagnosis, first distant metastasis, adjuvant_start and
follow-up. Per-sample exposure (t_days, cumulative days on class, ...) needs a
collection date and is computed by :func:`sample_exposure`, which
build_cohort.py calls.

All dates are integer day offsets on the MSK-CHORD axis (0 = first sequencing).

Run: ``python exposure.py``
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

import common as C
import loaders as L

W = C.pipeline_cfg()["windows"]


# --------------------------------------------------------------------------
# Agent -> class mapping (checklist 1)
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def agent_class_map() -> dict[str, str]:
    m = {}
    for cls, agents in C.config("drug_classes")["agent_classes"].items():
        for a in agents:
            if a in m:
                raise ValueError(f"agent {a} listed under both {m[a]} and {cls}")
            m[a] = cls
    return m


@functools.lru_cache(maxsize=None)
def treatment() -> pd.DataFrame:
    """MSK-CHORD treatment rows with a `cls` column. Raises on unknown agents."""
    tx = L.load_chord_timeline("treatment")[
        ["PATIENT_ID", "START_DATE", "STOP_DATE", "SUBTYPE", "AGENT"]].copy()
    amap = agent_class_map()
    unknown = sorted(set(tx.AGENT) - set(amap))
    if unknown:
        raise ValueError(f"Unmapped agents in treatment timeline (add to "
                         f"config/drug_classes.yaml): {unknown}")
    tx["cls"] = tx.AGENT.map(amap)
    C.check(tx.cls.notna().all(), "1. every treatment agent maps to a class or 'other'")
    # STOP_DATE is never missing in this release; guard anyway.
    tx["STOP_DATE"] = tx.STOP_DATE.fillna(tx.START_DATE)
    tx["systemic"] = tx.cls != "non_systemic"
    return tx


# --------------------------------------------------------------------------
# Patient-level clinical anchors
# --------------------------------------------------------------------------

def primary_dx(cohort: str) -> pd.DataFrame:
    """Earliest primary diagnosis of this cohort's cancer per patient.

    The MSK-CHORD diagnosis timeline has only SUBTYPE == "Primary" rows; the
    cancer is identified by the ICD-O topography code in DX_DESCRIPTION.
    """
    sites = C.config("drug_classes")["cohorts"][cohort]["icd_o_sites"]
    dx = L.load_chord_timeline("diagnosis")
    dx = dx[dx.SUBTYPE == "Primary"].copy()
    dx["site"] = dx.DX_DESCRIPTION.str.extract(r"\| (C\d\d)\d?\)")[0]
    dx = dx[dx.site.isin(sites)].sort_values("START_DATE")
    first = dx.groupby("PATIENT_ID").first()
    return pd.DataFrame({"dx_date": first.START_DATE,
                         "stage_at_dx": first.STAGE_CDM_DERIVED})


@functools.lru_cache(maxsize=None)
def _tumor_sites() -> pd.DataFrame:
    return L.load_chord_timeline("tumor_sites")[["PATIENT_ID", "START_DATE", "TUMOR_SITE"]]


def first_distant_met(cohort: str, dx: pd.DataFrame) -> pd.DataFrame:
    """Earliest distant-metastasis day per patient (config/pipeline.yaml)."""
    distant = C.pipeline_cfg()["distant_sites"][cohort]
    ts = _tumor_sites()
    nlp = ts[ts.TUMOR_SITE.isin(distant)].groupby("PATIENT_ID").START_DATE.min()

    stage4 = dx.loc[dx.stage_at_dx == "Stage 4", "dx_date"]

    ss = L.load_chord_timeline("specimen_surgery")[["SAMPLE_ID", "START_DATE"]]
    cs = L.load_chord_sample()[["SAMPLE_ID", "PATIENT_ID", "CANCER_TYPE",
                                "SAMPLE_TYPE", "METASTATIC_SITE"]]
    ctype = C.config("drug_classes")["cohorts"][cohort]["cancer_type"]
    met_s = cs[(cs.CANCER_TYPE == ctype) & (cs.SAMPLE_TYPE == "Metastasis")
               & ~L.normalize_site(cs.METASTATIC_SITE).str.contains("Lymph", na=False)]
    met_s = met_s.merge(ss, on="SAMPLE_ID").groupby("PATIENT_ID").START_DATE.min()

    src = pd.DataFrame({"stage4_dx": stage4, "nlp_site": nlp, "met_biopsy": met_s})
    out = pd.DataFrame({"met_date": src.min(axis=1),
                        "met_source": src.idxmin(axis=1, skipna=True)})
    return out


@functools.lru_cache(maxsize=None)
def last_event_day() -> pd.Series:
    """Last recorded MSK-CHORD event per patient, across every timeline."""
    days = []
    for f in sorted(L.CHORD.glob("data_timeline_*.txt")):
        t = pd.read_csv(f, sep="\t", usecols=["PATIENT_ID", "START_DATE", "STOP_DATE"],
                        low_memory=False)
        d = pd.concat([pd.to_numeric(t.START_DATE, errors="coerce"),
                       pd.to_numeric(t.STOP_DATE, errors="coerce")], axis=1).max(axis=1)
        days.append(d.groupby(t.PATIENT_ID).max())
    return pd.concat(days, axis=1).max(axis=1).rename("last_event_day")


# --------------------------------------------------------------------------
# Episodes
# --------------------------------------------------------------------------

def merge_episodes(starts: np.ndarray, stops: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Merge [start, stop] intervals whose gaps are <= `gap` days."""
    order = np.argsort(starts, kind="stable")
    eps: list[list[int]] = []
    for s, e in zip(starts[order], stops[order]):
        e = max(e, s)
        if eps and s - eps[-1][1] <= gap:
            eps[-1][1] = max(eps[-1][1], e)
        else:
            eps.append([int(s), int(e)])
    return [tuple(x) for x in eps]


def _agent_set(spec: dict, rows: pd.DataFrame) -> pd.Series:
    """Boolean mask of `rows` matching a {classes: [...], agents: [...]} spec."""
    return rows.cls.isin(spec.get("classes", [])) | rows.AGENT.isin(spec.get("agents", []))


def _subclass(cfg: dict, start_rows: pd.DataFrame) -> str:
    for name, members in cfg["subclass"].items():
        for m in members:
            cls, _, agent = m.partition(":")
            hit = start_rows.cls.eq(cls) if not agent else start_rows.AGENT.eq(agent)
            if hit.any():
                return name
    return "other"


def build_exposure() -> pd.DataFrame:
    tx = treatment()
    dcfg = C.config("drug_classes")["cohorts"]
    patient = L.load_chord_patient().set_index("PATIENT_ID")
    last = last_event_day()
    sample = L.load_chord_sample()

    rows, log_rows = [], []
    for cohort, cfg in dcfg.items():
        pts = sample.loc[sample.CANCER_TYPE == cfg["cancer_type"], "PATIENT_ID"].unique()
        dx = primary_dx(cohort).reindex(pts)
        met = first_distant_met(cohort, dx).reindex(pts)
        t = tx[tx.PATIENT_ID.isin(pts)].copy()
        t["dx_date"] = t.PATIENT_ID.map(dx.dx_date)
        # 6.2 step 2: drop rows before this cancer's primary dx - 30 d. Patients
        # with no matching primary dx row keep all rows (flagged dx_missing).
        before_dx = t.START_DATE < t.dx_date - W["dx_buffer_days"]
        anchor = t.cls.isin(cfg["anchor_classes"])
        n_anchor_dropped = int((anchor & before_dx).sum())
        t = t[~before_dx]
        anchor = t.cls.isin(cfg["anchor_classes"])
        companion_mask = pd.Series(False, index=t.index)
        for spec in cfg["companions"].values():
            companion_mask |= _agent_set(spec, t)

        for pid, a in t[anchor].groupby("PATIENT_ID"):
            p = t[t.PATIENT_ID == pid]
            cs = int(a.START_DATE.min())
            eps = merge_episodes(a.START_DATE.to_numpy(), a.STOP_DATE.to_numpy(),
                                 W["episode_gap_days"])
            lw = W["line_window_days"]
            sysrows = p[p.systemic]
            other = sysrows[~sysrows.cls.isin(cfg["anchor_classes"])]
            prior = other[other.START_DATE < cs - lw]
            concurrent = other[(other.START_DATE >= cs - lw) & (other.START_DATE <= cs + lw)
                               & ~companion_mask.reindex(other.index)]
            start_rows = p[(p.START_DATE >= cs - lw) & (p.START_DATE <= cs + lw)]
            first_ep_end = eps[0][1]
            rec = {
                "PATIENT_ID": pid, "cohort": cohort,
                "dx_date": dx.dx_date.get(pid), "stage_at_dx": dx.stage_at_dx.get(pid),
                "dx_missing": pd.isna(dx.dx_date.get(pid)),
                "class_start": cs, "class_end": int(max(e for _, e in eps)),
                "n_episodes": len(eps),
                "episode_starts": [s for s, _ in eps], "episode_stops": [e for _, e in eps],
                "first_systemic_start": int(sysrows.START_DATE.min()),
                "prior_classes": sorted(prior.cls.unique()),
                "concurrent_classes": sorted(concurrent.cls.unique()),
                "subclass": _subclass(cfg, start_rows),
                "met_date": met.met_date.get(pid), "met_source": met.met_source.get(pid),
                "followup_days": float(last.get(pid, np.nan)) - cs,
            }
            # Companion flags: given alongside the class in its first episode
            # (class_start - 30 .. end of episode 1). Defined the same way for
            # pre- and post-treatment samples, so they cannot leak the label.
            for name, spec in cfg["companions"].items():
                c = p[_agent_set(spec, p)]
                rec[name] = bool(((c.START_DATE >= cs - lw) & (c.START_DATE <= first_ep_end)).any())
            # Later-line flag start days; turned into "exposed before collection"
            # per sample in sample_exposure().
            for name, spec in cfg["later_line_flags"].items():
                c = p[_agent_set(spec, p)]
                rec[f"{name}_first_day"] = float(c.START_DATE.min()) if len(c) else np.nan
            rows.append(rec)
        log_rows.append({"cohort": cohort, "patients_with_cancer": len(pts),
                         "patients_with_class": int(t[anchor].PATIENT_ID.nunique()),
                         "anchor_rows_dropped_pre_dx": n_anchor_dropped,
                         "no_matching_primary_dx": int(dx.dx_date.isna().sum())})

    ex = pd.DataFrame(rows)
    ex["strict_first_line"] = ex.class_start <= ex.first_systemic_start + W["line_window_days"]
    ex["n_prior_classes"] = ex.prior_classes.str.len()
    # adjuvant_start (6.3): class started before the first distant metastasis,
    # or no metastasis is ever recorded.
    ex["adjuvant_start"] = ex.met_date.isna() | (ex.class_start < ex.met_date)
    ex["PRIOR_MED_TO_MSK"] = ex.PATIENT_ID.map(patient.PRIOR_MED_TO_MSK)
    ex["prior_med_to_msk"] = ex.PRIOR_MED_TO_MSK.eq("Prior medications to MSK")

    # ---- checklist 2 and 3 -------------------------------------------------
    has_dx = ex.dx_date.notna()
    C.check((ex.loc[has_dx, "class_start"] >= ex.loc[has_dx, "dx_date"] - W["dx_buffer_days"]).all(),
            "2. no class exposure before primary dx - 30 d is used")
    ok3 = all(st[0] == cs and all(b > a for a, b in zip(sp[:-1], st[1:]))
              and all(s <= e for s, e in zip(st, sp))
              for cs, st, sp in zip(ex.class_start, ex.episode_starts, ex.episode_stops))
    C.check(ok3, "3. class_start <= every episode start; merged episodes do not overlap")

    out = L.DERIVED / "exposure.parquet"
    ex.to_parquet(out, index=False)
    log = pd.DataFrame(log_rows).set_index("cohort")
    summ = ex.groupby("cohort").agg(
        patients=("PATIENT_ID", "size"), strict_first_line=("strict_first_line", "mean"),
        adjuvant_start=("adjuvant_start", "mean"), has_met=("met_date", lambda s: s.notna().mean()),
        prior_med_to_msk=("prior_med_to_msk", "mean"), median_episodes=("n_episodes", "median"))

    sub = ex.groupby(["cohort", "subclass"]).size().unstack(fill_value=0)
    C.write_log("exposure", f"""
Stage 2 (exposure.py). Read {len(tx):,} MSK-CHORD treatment rows ({tx.AGENT.nunique()} agents,
all mapped to a class). Wrote {len(ex):,} patient x cohort-class rows to `derived/exposure.parquet`.
Class rows dated before the cohort cancer's primary diagnosis - {W['dx_buffer_days']} d were dropped
(belong to an earlier cancer). Episodes merge gaps <= {W['episode_gap_days']} d. First distant
metastasis is the earliest of Stage 4 at diagnosis, an NLP radiology mention of a distant organ,
or a non-nodal metastasis biopsy (config/pipeline.yaml).
""", {"Input / filter counts": log, "Per-cohort summary (shares)": summ, "Subclass counts": sub,
      "First-metastasis source": ex.groupby(["cohort", "met_source"]).size().unstack(fill_value=0)})
    print(f"wrote {out} ({len(ex):,} rows)")
    return ex


# --------------------------------------------------------------------------
# Per-sample exposure (Section 6.3)
# --------------------------------------------------------------------------

def sample_exposure(s: pd.DataFrame, ex: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Exposure variables for samples `s` (PATIENT_ID, collect_day) in `cohort`.

    Returns a frame indexed like `s`. Patients without the class get NaN.
    """
    cfg = C.config("drug_classes")["cohorts"][cohort]
    lw, grace = W["line_window_days"], W["on_class_grace_days"]
    e = ex[ex.cohort == cohort].set_index("PATIENT_ID")
    tx = treatment()
    tx = tx[tx.PATIENT_ID.isin(e.index) & tx.systemic]
    companion = pd.Series(False, index=tx.index)
    for spec in cfg["companions"].values():
        companion |= _agent_set(spec, tx)
    # A companion agent is part of the class regimen while a class episode is
    # running (+/- line window); started outside every episode (e.g.
    # capecitabine maintenance after FOLFOX ended) it is an intervening line.
    in_episode = pd.Series(False, index=tx.index)
    for pid, g in tx[companion].groupby("PATIENT_ID"):
        r = e.loc[pid]
        for a, b in zip(r.episode_starts, r.episode_stops):
            in_episode[g.index[(g.START_DATE >= a - lw) & (g.START_DATE <= b + lw)]] = True
    other = tx[~tx.cls.isin(cfg["anchor_classes"]) & ~(companion & in_episode)]
    other_by_pt = {pid: g for pid, g in other.groupby("PATIENT_ID")}

    recs = []
    for idx, pid, day in zip(s.index, s.PATIENT_ID, s.collect_day):
        if pid not in e.index:
            recs.append({"_idx": idx})
            continue
        r = e.loc[pid]
        cs = r.class_start
        t = day - cs
        eps = list(zip(r.episode_starts, r.episode_stops))
        if t <= 0:
            cum, on, since, between = 0.0, False, np.nan, []
        else:
            # Days on class from class_start up to collection (inclusive ends).
            cum = float(sum(max(0, min(b, day) - a + 1) for a, b in eps if a <= day))
            on = any(a <= day <= b + grace for a, b in eps)
            past = [b for a, b in eps if a <= day]
            since = np.nan if on else float(day - max(past))
            g = other_by_pt.get(pid)
            between = [] if g is None else sorted(
                g.loc[(g.START_DATE > cs + lw) & (g.START_DATE <= day), "cls"].unique())
        rec = {"_idx": idx, "t_days": float(t), "cum_days_on_class": cum,
               "on_class_at_collection": on, "days_since_class_stop": since,
               "classes_between": between, "n_lines_between": len(between),
               "metastatic_at_collection": bool(pd.notna(r.met_date) and r.met_date <= day)}
        for name in cfg["later_line_flags"]:
            fd = r[f"{name}_first_day"]
            rec[f"{name}_before_collection"] = bool(pd.notna(fd) and fd <= day)
        recs.append(rec)
    out = pd.DataFrame(recs).set_index("_idx")
    out.index.name = s.index.name
    return out.reindex(s.index)


if __name__ == "__main__":
    build_exposure()
