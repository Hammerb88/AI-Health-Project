"""Stages 3-4: sample cohort, exclusions and labels (data_cleaning_plan.md 7-8).

Writes
  derived/samples.parquet            one row per sample (main label variant)
  derived/samples_<variant>.parquet  other label variants (A-on by default)
  derived/attrition.csv              Table T1 source: counts at every step
  derived/plan_reproduction.csv      Section 2.1 table recomputed
  derived/controls.csv               Table T4 source (raw-mutation stage)
  derived/logs/build_cohort.md

Run: ``python build_cohort.py [--label-variant A A-on A-30 A-180 A-progression]``
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import common as C
import exposure as E
import loaders as L

W = C.pipeline_cfg()["windows"]
F = C.pipeline_cfg()["filters"]
DCFG = C.config("drug_classes")["cohorts"]

LABEL_VARIANTS = {
    # name: (post threshold days, restriction on post samples)
    "A": (90, None),
    "A-on": (90, "on_class"),
    "A-30": (30, None),
    "A-180": (180, None),
    "A-progression": (90, "progression"),
}

# Section 2.1 / 2.2 values from the plan, used for the M1 reproduction check.
PLAN_2_1 = pd.DataFrame({
    "samples": [3992, 1314, 2696, 3513], "t_le_0": [2638, 974, 1852, 2815],
    "t_0_90": [285, 72, 242, 86], "t_gt_90": [1069, 268, 602, 612],
    "t_gt_90_on_class": [615, 202, 451, 57], "strict_t_le_0": [827, 339, 717, 1580],
    "strict_t_gt_90": [236, 115, 297, 391], "strict_t_gt_90_no_between": [116, 80, 111, 187],
}, index=C.COHORTS)
PLAN_2_2 = {  # (cohort, control) -> (pre %, post %, expected direction)
    ("breast", "ESR1 mutation"): (2.5, 14.3, "up"),
    ("nsclc", "EGFR T790M"): (1.5, 26.9, "up"),
    ("nsclc", "EGFR C797S"): (0.3, 1.5, "up"),
    ("nsclc", "MET amplification"): (0.8, 7.1, "up"),
    ("prostate", "AR amplification"): (1.3, 29.1, "up"),
    ("prostate", "AR mutation"): (0.8, 8.5, "up"),
    ("crc", "KRAS mutation"): (35.7, 37.9, "flat"),
}


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def base_samples() -> pd.DataFrame:
    """Every dated MSK-CHORD sample with its clinical + IMPACT sample fields."""
    ss = L.load_chord_timeline("specimen_surgery")[["SAMPLE_ID", "PATIENT_ID", "START_DATE", "SEQ_DATE"]]
    ss = ss.rename(columns={"START_DATE": "collect_day", "SEQ_DATE": "seq_day"})
    cs = L.load_chord_sample()[["SAMPLE_ID", "CANCER_TYPE", "SAMPLE_TYPE", "METASTATIC_SITE",
                                "PRIMARY_SITE", "GENE_PANEL"]]
    imp = L.load_impact_sample()[["SAMPLE_ID", "FACETS_QC", "FACETS_PURITY", "FACETS_PLOIDY",
                                  "FACETS_WGD", "TMB_SCORE", "MSI_SCORE", "MSI_TYPE"]]
    pm = L.load_gene_panel_matrix().set_index("SAMPLE_ID")
    s = ss.merge(cs, on="SAMPLE_ID", how="left").merge(imp, on="SAMPLE_ID", how="left")
    s["in_impact"] = s.SAMPLE_ID.isin(pm.index) & s.SAMPLE_ID.isin(L.cna_profiled_samples())
    s["panel_version"] = s.SAMPLE_ID.map(pm["mutations"])
    s["facets_pass"] = s.FACETS_QC.eq("Pass")
    pt = L.load_chord_patient().set_index("PATIENT_ID")
    ipt = L.load_impact_patient().set_index("PATIENT_ID")
    s["sex"] = s.PATIENT_ID.map(pt.GENDER).replace("Unknown", np.nan)
    s["sex"] = s.sex.fillna(s.PATIENT_ID.map(ipt.SEX))
    for c in ["HR", "HER2", "PRIOR_MED_TO_MSK", "CURRENT_AGE_DEID"]:
        s[c] = s.PATIENT_ID.map(pt[c])
    return s


def egfr_sensitizing(mut: pd.DataFrame) -> pd.Series:
    """Per-mutation flag: classic EGFR-sensitizing alteration.

    ex19del = in-frame deletion / delins starting in codons 729-761; L858R;
    plus the other established sensitizing hotspots (G719X, L861Q, S768I,
    E709X) used for the >= 95% cohort check (6.5). EGFR_sensitizing the
    *feature* (9.3) is ex19del / L858R only.
    """
    e = mut[mut.Hugo_Symbol == "EGFR"]
    pos = pd.to_numeric(e.Protein_position.str.extract(r"^(\d+)")[0], errors="coerce")
    hg = e.HGVSp_Short.fillna("")
    ex19 = e.Variant_Classification.isin(["In_Frame_Del", "In_Frame_Ins"]) & pos.between(729, 761) \
        & hg.str.contains("del")
    classic = hg.eq("p.L858R")
    other = hg.str.match(r"^p\.(G719[A-Z]|L861Q|S768I|E709[A-Z])$")
    return pd.DataFrame({"SAMPLE_ID": e.SAMPLE_ID, "ex19_l858r": ex19 | classic,
                         "any_sensitizing": ex19 | classic | other})


def progression_days() -> pd.DataFrame:
    p = L.load_chord_timeline("progression")
    p = p[(p.PROGRESSION == "Y") & (p.NLP_PROGRESSION_PROBABILITY >= 0.8)]
    return p[["PATIENT_ID", "START_DATE"]]


# --------------------------------------------------------------------------
# Cohort build
# --------------------------------------------------------------------------

def label_frame(s: pd.DataFrame, variant: str) -> pd.Series:
    """Target A for `variant`: 1 / 0 / NaN (excluded)."""
    thr, restrict = LABEL_VARIANTS[variant]
    lab = pd.Series(np.nan, index=s.index)
    lab[s.t_days <= 0] = 0
    post = s.t_days > thr
    if restrict == "on_class":
        post &= s.on_class_at_collection.astype(bool)
    elif restrict == "progression":
        pr = progression_days()
        m = s[["PATIENT_ID", "collect_day", "class_start"]].reset_index().merge(pr, on="PATIENT_ID")
        hit = m[(m.START_DATE > m.class_start) & (m.START_DATE <= m.collect_day)
                & (m.collect_day <= m.START_DATE + 90)]["index"].unique()
        post &= s.index.isin(hit)
    lab[post] = 1
    return lab


def pick_one_per_patient(d: pd.DataFrame) -> pd.DataFrame:
    """Largest t_days among post samples, else the latest pre sample (7)."""
    d = d.sort_values(["PATIENT_ID", "label_A", "t_days"], ascending=[True, False, False])
    return d.drop_duplicates("PATIENT_ID", keep="first")


def build(variants: list[str]) -> dict[str, pd.DataFrame]:
    ex = pd.read_parquet(L.DERIVED / "exposure.parquet")
    base = base_samples()
    mut = L.load_mutations()
    sens = egfr_sensitizing(mut).groupby("SAMPLE_ID")[["ex19_l858r", "any_sensitizing"]].any()

    attr_rows, repro_rows, ctrl_rows, dedup_log = [], [], [], []
    per_variant: dict[str, list[pd.DataFrame]] = {v: [] for v in variants}
    step0 = len(base)

    for cohort, cfg in DCFG.items():
        def rec(step, desc, d, lab=None):
            r = {"cohort": cohort, "step": step, "description": desc, "samples": len(d),
                 "patients": d.PATIENT_ID.nunique()}
            if lab is not None:
                r["label_0"] = int((lab == 0).sum()); r["label_1"] = int((lab == 1).sum())
            attr_rows.append(r)

        attr_rows.append({"cohort": cohort, "step": "0", "description":
                          "MSK-CHORD samples with a collection date", "samples": step0,
                          "patients": base.PATIENT_ID.nunique()})
        d = base[base.CANCER_TYPE == cfg["cancer_type"]].copy()
        rec("1", "sample cancer type matches cohort", d)

        e = ex[ex.cohort == cohort]
        d = d[d.PATIENT_ID.isin(e.PATIENT_ID)].copy()
        d = d.join(E.sample_exposure(d, ex, cohort))
        d = d.merge(e[["PATIENT_ID", "class_start", "strict_first_line", "prior_classes",
                       "n_prior_classes", "concurrent_classes", "subclass", "adjuvant_start",
                       "met_date", "dx_date", "stage_at_dx", "followup_days",
                       *[k for k in cfg["companions"]]]], on="PATIENT_ID", how="left")
        d["cohort"] = cohort
        d["prior_med_to_msk"] = d.PRIOR_MED_TO_MSK.eq("Prior medications to MSK")
        d["strict_patient"] = d.strict_first_line & ~d.prior_med_to_msk
        rec("2", "patient received the class after primary diagnosis", d)

        # ---- Section 2.1 reproduction (loose population, before IMPACT filter)
        t = d.t_days
        strict = d.strict_patient & d.facets_pass
        repro_rows.append({"cohort": cohort, "samples": len(d), "t_le_0": int((t <= 0).sum()),
                           "t_0_90": int(((t > 0) & (t <= 90)).sum()), "t_gt_90": int((t > 90).sum()),
                           "t_gt_90_on_class": int(((t > 90) & d.on_class_at_collection.astype(bool)).sum()),
                           "strict_t_le_0": int((strict & (t <= 0)).sum()),
                           "strict_t_gt_90": int((strict & (t > 90)).sum()),
                           "strict_t_gt_90_no_between": int((strict & (t > 90) & (d.n_lines_between == 0)).sum()),
                           "median_t_gt_90": float(t[t > 90].median())})

        # ---- Section 2.2 controls, raw calls, loose cohort. The plan's values
        # used every loose sample as denominator (unprofiled samples counted
        # as unaltered); recompute that for the reproduction check, and the
        # correct profiled-only denominator for reporting.
        ctrl_rows += raw_controls(cohort, d, mut, stage="raw_plan_denominator")
        d = d[d.in_impact].copy()
        rec("3", "sample in MSK-IMPACT 50K with mutation + CNA data", d)
        ctrl_rows += raw_controls(cohort, d, mut, stage="raw")

        # ---- step 4: cohort-specific rules ---------------------------------
        d["egfr_sensitizing_any"] = d.SAMPLE_ID.map(sens.any_sensitizing).fillna(False).astype(bool)
        if cohort == "breast":
            d = d[d.HR == "Yes"].copy()
            rec("4", "breast: HR-positive (HR == 'Yes')", d)
        elif cohort == "nsclc":
            share = d.egfr_sensitizing_any.mean()
            print(f"  nsclc EGFR-sensitizing share before restriction: {share:.1%}")
            dedup_log.append(f"NSCLC: {share:.1%} of samples carry an EGFR-sensitizing "
                             f"alteration before restriction")
            d = d[d.egfr_sensitizing_any].copy()
            rec("4", "nsclc: EGFR-sensitizing alteration (ex19del, L858R, G719X, L861Q, S768I, E709X)", d)
        else:
            rec("4", "no cohort-specific rule", d)

        d["hypermutated"] = d.MSI_TYPE.eq("Instable") | (d.TMB_SCORE > F["hypermutated_tmb"])
        d["low_purity"] = d.FACETS_PURITY < F["low_purity"]
        d["archived_tissue"] = d.collect_day < d.seq_day - W["archived_tissue_days"]

        for v in variants:
            dv = d.copy()
            dv["label_variant"] = v
            dv["label_A"] = label_frame(dv, v)
            rec_v = (lambda step, desc, x: rec(step, f"[{v}] {desc}", x, x.label_A)) if v != "A" \
                else (lambda step, desc, x: rec(step, desc, x, x.label_A))
            rec_v("4L", "labelled (all t)", dv)
            dv = dv[dv.label_A.notna()].copy()
            rec_v("5", f"exclude 0 < t_days <= {LABEL_VARIANTS[v][0]} (and unlabelled)", dv)
            dv = dv[~((dv.label_A == 0) & dv.prior_med_to_msk)].copy()
            rec_v("5b", "exclude label-0 samples with prior medications to MSK", dv)
            n_before = len(dv)
            dv = pick_one_per_patient(dv)
            if v == "A":
                dedup_log.append(f"{cohort}: one-sample-per-patient removed {n_before - len(dv)} samples")
            rec_v("5c", "one sample per patient", dv)
            dv["label_A"] = dv.label_A.astype(int)
            dv["cum_days_on_class"] = np.where(dv.label_A == 0, 0.0, dv.cum_days_on_class)
            dv["label_B"] = dv.cum_days_on_class
            dv["label_B_log1p"] = np.log1p(dv.cum_days_on_class)
            dv["tier_relaxed"] = True
            dv["tier_strict"] = dv.strict_patient & dv.facets_pass
            dv["tier_strict_noint"] = dv.tier_strict & (dv.n_lines_between == 0)
            rec_v("6", "strict tier: class first-line, no prior meds to MSK, FACETS QC pass",
                  dv[dv.tier_strict])
            rec_v("6b", "strict, no intervening therapy", dv[dv.tier_strict_noint])
            rec_v("7", "relaxed tier with FACETS QC pass (clonality available)", dv[dv.facets_pass])
            per_variant[v].append(dv)

    out = {v: pd.concat(parts, ignore_index=True) for v, parts in per_variant.items()}
    attrition = pd.DataFrame(attr_rows)
    repro = pd.DataFrame(repro_rows).set_index("cohort")
    controls = pd.DataFrame(ctrl_rows)
    validate(out["A"], ex)
    m1_reproduction(repro, controls)

    for v, df in out.items():
        name = "samples.parquet" if v == "A" else f"samples_{v}.parquet"
        df.to_parquet(L.DERIVED / name, index=False)
    attrition.to_csv(L.DERIVED / "attrition.csv", index=False)
    repro.to_csv(L.DERIVED / "plan_reproduction.csv")
    controls.to_csv(L.DERIVED / "controls.csv", index=False)

    s = out["A"]
    by = s.groupby(["cohort", "label_A"]).size().unstack()
    tiers = s.groupby("cohort")[["tier_relaxed", "tier_strict", "tier_strict_noint"]].sum()
    C.write_log("build_cohort", f"""
Stages 3-4 (build_cohort.py). Started from {step0:,} dated MSK-CHORD samples and
{len(ex):,} patient x class exposure rows. Wrote {len(s):,} samples (one per patient per
cohort) to `derived/samples.parquet` for label variant A, and variants
{', '.join(v for v in variants if v != 'A') or 'none'} to `samples_<variant>.parquet`.
Label-0 samples from patients with prior medications to MSK are removed in every tier.
{' '.join(dedup_log)}
""", {"Label counts (variant A)": by, "Tier sizes (variant A)": tiers,
      "Section 2.1 reproduction (computed)": repro,
      "Section 2.2 controls (raw mutations, loose cohort)": controls.set_index(["cohort", "control"])})
    return out


def raw_controls(cohort: str, d: pd.DataFrame, mut: pd.DataFrame, stage: str) -> list[dict]:
    """Section 2.2: % altered pre (t <= 0) vs post (t > 90), all raw calls."""
    cna = L.load_cna()
    pre = d.loc[d.t_days <= 0, "SAMPLE_ID"]
    post = d.loc[d.t_days > 90, "SAMPLE_ID"]
    def mset(gene, hgvsp=None):
        m = mut[mut.Hugo_Symbol == gene]
        if hgvsp:
            m = m[m.HGVSp_Short == hgvsp]
        return set(m.SAMPLE_ID)
    def cset(gene, val):
        c = cna[(cna.gene == gene) & (cna.value == val)]
        return set(c.SAMPLE_ID.astype(str))
    defs = {"breast": {"ESR1 mutation": mset("ESR1")},
            "nsclc": {"EGFR T790M": mset("EGFR", "p.T790M"), "EGFR C797S": mset("EGFR", "p.C797S"),
                      "MET amplification": cset("MET", 2)},
            "prostate": {"AR amplification": cset("AR", 2), "AR mutation": mset("AR")},
            "crc": {"KRAS mutation": mset("KRAS")}}[cohort]
    rows = []
    for name, hit in defs.items():
        a, b = pre.isin(hit), post.isin(hit)
        rows.append({"cohort": cohort, "control": name, "stage": stage,
                     "n_pre": len(pre), "n_post": len(post),
                     "pct_pre": 100 * a.mean(), "pct_post": 100 * b.mean(),
                     "plan_pct_pre": PLAN_2_2[(cohort, name)][0],
                     "plan_pct_post": PLAN_2_2[(cohort, name)][1],
                     "expected": PLAN_2_2[(cohort, name)][2]})
    return rows


# --------------------------------------------------------------------------
# Checklist items 4-7 and the M1 reproduction gate
# --------------------------------------------------------------------------

def validate(s: pd.DataFrame, ex: pd.DataFrame) -> None:
    C.check(not ((s.t_days > 0) & (s.t_days <= 90)).any(),
            "4a. label A has no samples with 0 < t_days <= 90")
    C.check(((s.label_B == 0) == (s.t_days <= 0)).all(),
            "4b. label B is 0 exactly for t_days <= 0 (and > 0 for every post sample)")

    # 5. Spot-check 20 patients with prior other therapy: recompute t_days from
    # the raw treatment timeline using only the cohort's anchor agents.
    tx = E.treatment()
    rng = np.random.default_rng(0)
    pool = s[s.n_prior_classes > 0]
    pick = pool.iloc[rng.choice(len(pool), size=min(20, len(pool)), replace=False)]
    ok, first_any_differs = True, 0
    for _, r in pick.iterrows():
        cfg = DCFG[r.cohort]
        t = tx[tx.PATIENT_ID == r.PATIENT_ID]
        t = t[t.START_DATE >= (r.dx_date - W["dx_buffer_days"] if pd.notna(r.dx_date) else -np.inf)]
        cs = t.loc[t.cls.isin(cfg["anchor_classes"]), "START_DATE"].min()
        ok &= (r.collect_day - cs) == r.t_days
        first_any_differs += int(t.loc[t.systemic, "START_DATE"].min() < cs)
    C.check(ok and first_any_differs == len(pick),
            f"5. labels are relative to the cohort class ({len(pick)} patients with prior "
            f"therapy spot-checked; first therapy of any class precedes class_start in all)")

    C.check(not s.duplicated(["cohort", "PATIENT_ID"]).any(), "6a. one sample per patient per cohort")
    C.check(not ((s.label_A == 0) & s.prior_med_to_msk).any(),
            "6b. no label-0 sample has PRIOR_MED_TO_MSK == 'Prior medications to MSK' (any tier)")

    C.check((s.loc[s.cohort == "breast", "HR"] == "Yes").all(), "7a. breast cohort is 100% HR-positive")
    C.check(s.loc[s.cohort == "nsclc", "egfr_sensitizing_any"].mean() >= 0.95,
            "7b. NSCLC cohort >= 95% EGFR-sensitizing")
    for c in ["adjuvant_start", "metastatic_at_collection", "sex"]:
        C.check(s[c].notna().all(), f"7c. {c} is non-null for every sample")


def m1_reproduction(repro: pd.DataFrame, controls: pd.DataFrame) -> None:
    """Compare the recomputed Section 2.1 counts / 2.2 prevalences to the plan."""
    cmp = repro[PLAN_2_1.columns].div(PLAN_2_1) - 1
    print("\nSection 2.1 counts, relative difference vs plan:")
    print((100 * cmp).round(1).to_string())
    big = cmp[["samples", "t_le_0", "t_gt_90"]].abs().max().max()
    C.check(big <= 0.025, f"M1: Section 2.1 loose counts reproduce within 2.5% (max {big:.1%})")
    print("\nSection 2.2 controls (raw calls):")
    print(controls[["cohort", "control", "stage", "n_pre", "n_post", "pct_pre", "plan_pct_pre",
                    "pct_post", "plan_pct_post"]].round(1).to_string(index=False))
    pdn = controls[controls.stage == "raw_plan_denominator"]
    dev = np.maximum((pdn.pct_pre - pdn.plan_pct_pre).abs(), (pdn.pct_post - pdn.plan_pct_post).abs())
    C.check((dev <= 2).all(), f"M1: Section 2.2 prevalences reproduce within 2 percentage points "
            f"with the plan's denominator (max {dev.max():.2f} pp)")
    # Direction on the correct (profiled) denominator is the scientific gate.
    for _, r in controls[controls.stage == "raw"].iterrows():
        if r.expected == "up":
            C.check(r.pct_post > 2 * r.pct_pre and r.pct_post - r.pct_pre > 1,
                    f"2.2 control {r.control} ({r.cohort}) enriched post-treatment "
                    f"({r.pct_pre:.1f}% -> {r.pct_post:.1f}%)")
        else:
            C.check(abs(r.pct_post - r.pct_pre) < 5,
                    f"2.2 negative control {r.control} ({r.cohort}) not enriched "
                    f"({r.pct_pre:.1f}% -> {r.pct_post:.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-variant", nargs="+", default=["A", "A-on"],
                    choices=list(LABEL_VARIANTS))
    args = ap.parse_args()
    variants = list(dict.fromkeys(["A", *args.label_variant]))
    build(variants)
