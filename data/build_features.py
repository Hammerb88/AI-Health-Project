"""Build the derived per-sample tables the exploration notebook reads.

Writes three parquet files to data/derived/:
  het_features.parquet  one row per sequenced sample, heterogeneity features
  cohort.parquet        the core cohort with cross-sectional + prospective labels
  treatment.parquet     per-patient MSK-CHORD exposure, split pre/post sequencing

Run once: ``python build_features.py``. Everything downstream reads the parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import loaders as L

PERI = 30  # days either side of sequencing treated as "present at biopsy"


def build_het_features() -> pd.DataFrame:
    """Per-sample heterogeneity features from the FACETS ASCN block.

    CCF is the fraction of *cancer cells* carrying a mutation, already
    corrected by FACETS for purity, local copy number and mutation
    multiplicity. Clonal (CCF~1) means the mutation predates diversification;
    subclonal means it sits on a branch. The spread of CCF within a sample is
    our proxy for how much branching the panel can see.
    """
    print("reading mutations (190 MB, ASCN columns only)...")
    mut = L.load_ascn_mutations()
    mut = mut.rename(columns={"Tumor_Sample_Barcode": "SAMPLE_ID"})

    ccf = pd.to_numeric(mut["ASCN.CCF_M_COPIES"], errors="coerce")
    mut["ccf"] = ccf
    clonal = mut["ASCN.CLONAL"]

    # MATH-style dispersion: MAD of CCF scaled by its median. Robust to the
    # long tail and comparable across samples with different mutation counts.
    def math_score(x: pd.Series) -> float:
        x = x.dropna()
        if len(x) < 2:
            return np.nan
        med = x.median()
        if med == 0:
            return np.nan
        return 100.0 * (x - med).abs().median() / med

    g = mut.groupby("SAMPLE_ID")
    feat = pd.DataFrame({
        "n_mutations": g.size(),
        "n_ccf_called": g["ccf"].count(),
        "n_clonal": g.apply(lambda d: (d["ASCN.CLONAL"] == "CLONAL").sum(),
                            include_groups=False),
        "n_subclonal": g.apply(lambda d: (d["ASCN.CLONAL"] == "SUBCLONAL").sum(),
                               include_groups=False),
        "n_indeterminate": g.apply(
            lambda d: (d["ASCN.CLONAL"] == "INDETERMINATE").sum(),
            include_groups=False),
        "ccf_mean": g["ccf"].mean(),
        "ccf_median": g["ccf"].median(),
        "ccf_std": g["ccf"].std(),
        "ccf_min": g["ccf"].min(),
        "math_score": g["ccf"].apply(math_score),
        "ascn_purity": g["ASCN.PURITY"].first(),
    })

    # Denominator is determinate calls only: INDETERMINATE means the panel
    # could not tell, which is missing information rather than evidence of
    # clonality. Samples with no determinate call get NaN, not 0.
    determinate = feat["n_clonal"] + feat["n_subclonal"]
    feat["n_determinate"] = determinate
    feat["subclonal_fraction"] = np.where(
        determinate > 0, feat["n_subclonal"] / determinate, np.nan)

    # Granularity of subclonal_fraction: with k determinate calls the statistic
    # can only take k+1 distinct values. Carried forward so the notebook can
    # show how coarse the feature is at low burden.
    feat["subclonal_fraction_resolution"] = np.where(
        determinate > 0, 1.0 / determinate, np.nan)

    feat = feat.reset_index()

    # Sample-level FACETS fields live in the clinical file, not the MAF.
    imp = L.load_impact_sample()
    keep = ["SAMPLE_ID", "FACETS_PURITY", "FACETS_PLOIDY", "FACETS_WGD",
            "FACETS_QC", "TMB_SCORE", "MSI_SCORE", "MSI_TYPE", "GENE_PANEL",
            "CANCER_TYPE", "SAMPLE_TYPE", "DISEASE_STATUS"]
    feat = feat.merge(imp[keep], on="SAMPLE_ID", how="left")

    feat = add_armlevel_diversity(feat)
    return feat


def add_armlevel_diversity(feat: pd.DataFrame) -> pd.DataFrame:
    """Count altered chromosome arms per sample.

    Chromosomal instability generates heterogeneity continuously, so arm-level
    burden is a cause rather than a measure of it -- kept separate from the
    CCF-derived features for exactly that reason.
    """
    print("reading arm-level CNA...")
    arm = pd.read_csv(L.IMPACT / "data_armlevel_cna.txt", sep="\t",
                      low_memory=False)
    arm = arm.drop(columns=["ENTITY_STABLE_ID", "DESCRIPTION"]).set_index("NAME")
    gain = (arm == "Gain").sum()
    loss = (arm == "Loss").sum()
    called = arm.notna().sum()
    out = pd.DataFrame({
        "arms_gain": gain, "arms_loss": loss,
        "arms_altered": gain + loss, "arms_called": called,
    })
    out["arm_altered_fraction"] = np.where(
        out.arms_called > 0, out.arms_altered / out.arms_called, np.nan)
    out.index.name = "SAMPLE_ID"
    return feat.merge(out.reset_index(), on="SAMPLE_ID", how="left")


def build_cohort() -> pd.DataFrame:
    """Core cohort: MSK-MET labels + MSK-IMPACT FACETS, one row per sample.

    Carries both label families:
      cross-sectional  MET_SITE_COUNT and the 21 DMETS_DX_* organ flags
      prospective      per-organ time-to-first-event after sequencing, with a
                       censoring indicator, on the sequencing-day-zero axis
    """
    met, imp = L.load_met_sample(), L.load_impact_sample()
    j = met.merge(imp, on="SAMPLE_ID", suffixes=("", "_imp"))
    j = j[j.FACETS_QC == "Pass"].copy()
    j = j.drop(columns=[c for c in j.columns if c.endswith("_imp")])

    mp = L.load_met_patient()
    j = j.merge(mp, on="PATIENT_ID", how="left")

    # Follow-up: sequencing to last contact or death, in days.
    j["followup_days"] = (j["AGE_AT_LAST_CONTACT"].fillna(j["AGE_AT_DEATH"])
                          - j["AGE_AT_SEQUENCING"]) * 365.25

    j = add_prospective_labels(j)
    return j


def add_prospective_labels(cohort: pd.DataFrame) -> pd.DataFrame:
    """Time to first metastasis per organ, measured from sequencing.

    Events at or before sequencing describe disease already present at biopsy;
    predicting those would leak. Only events more than PERI days after
    sequencing count, and each patient's follow-up sets the censoring time.
    """
    dx = L.load_met_timeline("diagnosis")
    dx = L.anchor_met_to_sequencing(dx)
    dx["organ"] = dx["SUBTYPE"].map(L.TIMELINE_SUBTYPE_TO_ORGAN)
    dx = dx[dx.organ.notna()]

    post = dx[dx.DAYS_FROM_SEQ > PERI]
    first = post.groupby(["PATIENT_ID", "organ"]).DAYS_FROM_SEQ.min().unstack()

    fu = cohort.set_index("PATIENT_ID")["followup_days"]
    out = cohort.set_index("PATIENT_ID").copy()

    for organ in sorted(L.TIMELINE_SUBTYPE_TO_ORGAN.values()):
        t = first[organ].reindex(out.index) if organ in first else pd.Series(
            np.nan, index=out.index)
        event = t.notna()
        # Censor non-events at the end of observed follow-up.
        out[f"T_{organ}"] = np.where(event, t, fu.reindex(out.index))
        out[f"E_{organ}"] = event.astype(int)

    any_t = first.reindex(out.index).min(axis=1)
    out["T_ANY_MET"] = np.where(any_t.notna(), any_t, fu.reindex(out.index))
    out["E_ANY_MET"] = any_t.notna().astype(int)
    return out.reset_index()


def build_treatment() -> pd.DataFrame:
    """Per-patient MSK-CHORD treatment exposure, split around sequencing.

    CHORD dates are already days since sequencing, so no re-anchoring is
    needed here -- but anything joined against MSK-MET must be re-anchored
    first (see loaders.anchor_met_to_sequencing).
    """
    tx = L.load_chord_timeline("treatment")
    tx["window"] = np.where(tx.START_DATE < -PERI, "pre",
                            np.where(tx.START_DATE > PERI, "post", "peri"))

    counts = (tx.groupby(["PATIENT_ID", "window", "SUBTYPE"]).size()
              .unstack(["window", "SUBTYPE"], fill_value=0))
    counts.columns = [f"n_{w}_{s}".replace(" ", "_").lower()
                      for w, s in counts.columns]

    summary = pd.DataFrame({
        "n_treatments": tx.groupby("PATIENT_ID").size(),
        "n_agents": tx.groupby("PATIENT_ID").AGENT.nunique(),
        "n_classes": tx.groupby("PATIENT_ID").SUBTYPE.nunique(),
        "first_tx_day": tx.groupby("PATIENT_ID").START_DATE.min(),
        "last_tx_day": tx.groupby("PATIENT_ID").START_DATE.max(),
    })
    pre = tx[tx.window == "pre"].groupby("PATIENT_ID")
    summary["n_pre_treatments"] = pre.size()
    summary["days_last_pre_tx_to_seq"] = -pre.START_DATE.max()
    summary["any_pre_treatment"] = summary.n_pre_treatments.notna().astype(int)
    summary = summary.fillna({"n_pre_treatments": 0, "any_pre_treatment": 0})

    return summary.join(counts, how="left").fillna(0).reset_index()


if __name__ == "__main__":
    het = build_het_features()
    het.to_parquet(L.DERIVED / "het_features.parquet", index=False)
    print(f"het_features.parquet: {het.shape}")

    cohort = build_cohort()
    cohort.to_parquet(L.DERIVED / "cohort.parquet", index=False)
    print(f"cohort.parquet: {cohort.shape}")

    tx = build_treatment()
    tx.to_parquet(L.DERIVED / "treatment.parquet", index=False)
    print(f"treatment.parquet: {tx.shape}")
