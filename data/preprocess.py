"""Stages 7-9: nuisance covariates, missing data, CV folds and model tables
(data_cleaning_plan.md Sections 11-13).

Writes
  derived/model_table_<cohort>.parquet  one per cohort + `pooled`: ids, labels,
      tier / analysis flags, covariates, genomic features, alteration features,
      and strict-tier fold columns fold_r00..fold_r19
  derived/folds.parquet                 long: cohort x tier x variant x repeat x sample -> fold
  derived/feature_dictionary.csv        one row per feature (T9 source)
  derived/controls.csv                  (appended) T4 in every tier + S3 survivorship
  derived/logs/preprocess.md

Covariate roles. The plan's Section 11 lists the exposure context
(on_class_at_collection, n_lines_between, companion / later-line flags) as
covariates. Several of these are *defined only after the class starts* and are
0 / NaN for every label-0 sample, so they would leak Target A perfectly. They
are therefore split:
  covariate_A  usable for Target A and Target B (defined the same way pre/post)
  covariate_B  Target B dose model among post-treatment samples only (8.2)
A leakage check fails the stage if any covariate_A column separates the label
almost perfectly on its own.

All transforms that learn from data (imputation, scaling, one-hot, prevalence
filter) are *not* applied here. `make_preprocessor()` builds them for fitting
inside each training fold (13.3).

Run: ``python preprocess.py``
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import common as C
import loaders as L

P = C.pipeline_cfg()
F = P["filters"]
CV = P["cv"]
PRIMARY_ORGAN = {"breast": "Breast", "nsclc": "Lung", "prostate": "Prostate", "crc": "Colorectal"}
TIERS = ["strict", "strict_noint", "relaxed"]
ID_COLS = ["SAMPLE_ID", "PATIENT_ID", "cohort"]
LABEL_COLS = ["label_A", "label_B", "label_B_log1p", "t_days"]

# Covariate blocks (Section 11). `cat` = categorical (one-hot in fold),
# `bin` = 0/1, `num` = continuous (median-imputed + scaled in fold).
COVARIATES_A = {
    "sample_type": "cat", "biopsy_organ": "cat", "biopsy_organ_coarse": "cat",
    "metastatic_at_collection": "bin", "sex_male": "bin", "adjuvant_start": "bin",
    "n_mutations_oncogenic_log1p": "num", "panel_version": "cat", "log_patient_id": "num",
    "FACETS_PURITY": "num", "FACETS_PURITY_missing": "bin", "low_purity": "bin",
    "archived_tissue": "bin", "age_at_collection": "num", "stage_at_dx": "cat",
    "n_prior_classes": "num", "subclass": "cat", "HER2": "cat",
    "companion_cdk46": "bin", "ovarian_suppression": "bin", "fluoropyrimidine_backbone": "bin",
    "drug_class": "cat",
}
COVARIATES_B = {
    "on_class_at_collection": "bin", "days_since_class_stop": "num",
    "days_since_class_stop_missing": "bin", "n_lines_between": "num",
    "cdk46_any_before_collection": "bin", "pi3k_mtor_before_collection": "bin",
    "ar_inhibitor_exposure_before_collection": "bin", "irinotecan_before_collection": "bin",
    "anti_egfr_mab_before_collection": "bin", "anti_vegf_before_collection": "bin",
}
GENOMIC = {"tmb_log1p": "num", "FGA": "num", "FGA_missing": "bin", "MSI_SCORE": "num",
           "MSI_SCORE_missing": "bin", "FACETS_PLOIDY": "num", "FACETS_WGD": "bin",
           "FACETS_missing": "bin"}
FLAG_COLS = ["tier_strict", "tier_strict_noint", "tier_relaxed", "hypermutated", "facets_pass",
             "main_analysis", "s3_survivorship_keep", "label_A_on", "followup_days"]
COHORT_ONLY = {"sex_male": ["nsclc", "crc"], "HER2": ["breast"],
               "companion_cdk46": ["breast"], "ovarian_suppression": ["breast"],
               "fluoropyrimidine_backbone": ["crc"], "cdk46_any_before_collection": ["breast"],
               "pi3k_mtor_before_collection": ["breast"],
               "ar_inhibitor_exposure_before_collection": ["prostate"],
               "irinotecan_before_collection": ["crc"], "anti_egfr_mab_before_collection": ["crc"],
               "anti_vegf_before_collection": ["crc"], "drug_class": ["pooled"]}
CONTROL_FEATURES = {  # 13.6 positive / negative controls (feature, subclass restriction)
    "breast": [("ESR1_LBD", "AI"), ("ESR1_mut", None)],
    "nsclc": [("EGFR_T790M", None), ("MET_amp", None), ("EGFR_C797X", None)],
    "prostate": [("AR_amp", None), ("AR_LBD", None), ("AR_mut", None)],
    "crc": [("KRAS_any", None)],
}
GATED_CONTROLS = {"ESR1_LBD", "EGFR_T790M", "MET_amp", "AR_amp", "AR_LBD", "KRAS_any"}


# --------------------------------------------------------------------------
# Stage 7: covariates
# --------------------------------------------------------------------------

def biopsy_organ(s: pd.DataFrame) -> pd.Series:
    site_map = pd.read_csv(C.CONFIG / "site_map.csv").set_index("site").organ_group
    met = L.normalize_site(s.METASTATIC_SITE)
    use_met = met.notna() & ~met.isin(["Not Applicable", "Not Available", "Unknown", "Na"])
    raw = met.where(use_met, L.normalize_site(s.PRIMARY_SITE))
    org = raw.map(site_map)
    unmapped = raw[org.isna() & raw.notna()].unique()
    if len(unmapped):
        raise ValueError(f"biopsy sites missing from config/site_map.csv: {list(unmapped)[:20]}")
    return org.fillna("Unknown")


def build_covariates(s: pd.DataFrame, gf: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    g = gf.reindex(s.SAMPLE_ID)
    g.index = s.index
    s["sample_type"] = s.SAMPLE_TYPE.fillna("Unknown")
    s["biopsy_organ_full"] = biopsy_organ(s)
    # groups < 30 samples within the cohort -> Other
    small = s.groupby(["cohort", "biopsy_organ_full"]).SAMPLE_ID.transform("size") < F["site_min_samples"]
    s["biopsy_organ"] = s.biopsy_organ_full.mask(small, "Other")
    prim = s.cohort.map(PRIMARY_ORGAN)
    s["biopsy_organ_coarse"] = np.select(
        [s.biopsy_organ_full == prim, s.biopsy_organ_full.isin(["Liver", "Lung", "Bone"])],
        ["primary_organ", s.biopsy_organ_full.str.lower()], "other")
    s["sex_male"] = s.sex.map({"Male": 1.0, "Female": 0.0})
    s["log_patient_id"] = np.log(s.PATIENT_ID.str[2:].astype(int))
    s["n_mutations_oncogenic_log1p"] = g.n_mutations_oncogenic_log1p
    s["FACETS_PURITY"] = g.FACETS_PURITY
    s["FACETS_PURITY_missing"] = s.FACETS_PURITY.isna().astype(float)
    s["low_purity"] = (s.FACETS_PURITY < F["low_purity"]).astype(float)
    # Approximate: CURRENT_AGE_DEID is age at the last record.
    last = s.followup_days + s.class_start
    s["age_at_collection"] = pd.to_numeric(s.CURRENT_AGE_DEID, errors="coerce") - (last - s.collect_day) / 365.25
    s["stage_at_dx"] = s.stage_at_dx.fillna("Unknown")
    s["HER2"] = s.HER2.fillna("Unknown")
    s["drug_class"] = s.cohort
    s["days_since_class_stop_missing"] = s.days_since_class_stop.isna().astype(float)
    for c in ["tmb_log1p", "FGA", "MSI_SCORE", "FACETS_PLOIDY", "FACETS_WGD"]:
        s[c] = g[c]
    s["FGA_missing"] = s.FGA.isna().astype(float)
    s["MSI_SCORE_missing"] = s.MSI_SCORE.isna().astype(float)
    s["FACETS_missing"] = s.FACETS_PLOIDY.isna().astype(float)
    for c in ["metastatic_at_collection", "adjuvant_start", "archived_tissue", "on_class_at_collection",
              "companion_cdk46", "ovarian_suppression", "fluoropyrimidine_backbone",
              *[c for c in COVARIATES_B if c.endswith("_before_collection")]]:
        if c in s:
            s[c] = s[c].astype("float").astype("float32")
    return s


# --------------------------------------------------------------------------
# Stage 9 helpers
# --------------------------------------------------------------------------

class PrevalenceFilter(BaseEstimator, TransformerMixin):
    """Keep alteration features altered in >= min_n samples and >= min_frac of
    the *training* rows (9.6). Variant-level target features are always kept."""

    def __init__(self, min_n: int = F["min_altered_n"], min_frac: float = F["min_altered_frac"],
                 always_keep: tuple[str, ...] = ()):
        self.min_n, self.min_frac, self.always_keep = min_n, min_frac, always_keep

    def fit(self, X: pd.DataFrame, y=None):
        n = (X == 1).sum()
        frac = n / X.notna().sum().replace(0, np.nan)
        keep = (n >= self.min_n) & (frac >= self.min_frac)
        self.columns_ = [c for c in X.columns if keep.get(c, False) or c in self.always_keep]
        return self

    def transform(self, X: pd.DataFrame):
        return X[self.columns_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_)


class FoldPreprocessor(BaseEstimator, TransformerMixin):
    """In-fold transforms for M3 models: fit on training rows only (13.3).

    Alteration columns pass through a training-fold prevalence filter (no
    imputation: main-analysis features have no NA). Continuous covariates are
    median-imputed and standardized; categoricals one-hot encoded (levels
    unseen in training are dropped); binaries mode-imputed. Written in pandas
    because sklearn.compose / sklearn.impute are blocked by an Application
    Control policy on the development machine.
    """

    def __init__(self, alteration_cols: list[str], covariate_types: dict[str, str],
                 always_keep: tuple[str, ...] = ()):
        self.alteration_cols, self.covariate_types, self.always_keep = alteration_cols, covariate_types, always_keep

    def fit(self, X: pd.DataFrame, y=None):
        t = self.covariate_types
        self.num_ = [c for c, k in t.items() if k == "num"]
        self.cat_ = [c for c, k in t.items() if k == "cat"]
        self.bin_ = [c for c, k in t.items() if k == "bin"]
        self.prev_ = PrevalenceFilter(always_keep=self.always_keep).fit(X[self.alteration_cols])
        num = X[self.num_].astype(float)
        self.median_ = num.median()
        filled = num.fillna(self.median_)
        self.mean_, self.std_ = filled.mean(), filled.std(ddof=0).replace(0, 1.0)
        self.levels_ = {c: sorted(X[c].astype(str).unique()) for c in self.cat_}
        self.mode_ = X[self.bin_].astype(float).mode().iloc[0] if self.bin_ else pd.Series(dtype=float)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = [self.prev_.transform(X[self.alteration_cols]).add_prefix("alt__")]
        num = (X[self.num_].astype(float).fillna(self.median_) - self.mean_) / self.std_
        parts.append(num.add_prefix("num__"))
        for c in self.cat_:
            v = X[c].astype(str)
            parts.append(pd.DataFrame({f"cat__{c}_{lv}": (v == lv).astype(float) for lv in self.levels_[c]},
                                      index=X.index))
        parts.append(X[self.bin_].astype(float).fillna(self.mode_).add_prefix("bin__"))
        return pd.concat(parts, axis=1)


def make_preprocessor(alteration_cols: list[str], covariate_types: dict[str, str],
                      always_keep: tuple[str, ...] = ()) -> FoldPreprocessor:
    return FoldPreprocessor(alteration_cols, covariate_types, always_keep)


def assign_folds(d: pd.DataFrame, strat: pd.Series, seed: int) -> pd.DataFrame:
    """5 folds x 20 repeats, grouped by patient, stratified by `strat` (13.1-2)."""
    out = {}
    # Strata smaller than n_folds fall back to the label alone.
    y = strat.astype(str)
    lab = y.str.split("_").str[0]
    y = y.where(y.map(y.value_counts()) >= CV["n_folds"], lab + "_rare").to_numpy()
    for r in range(CV["n_repeats"]):
        f = np.full(len(d), -1)
        sgk = StratifiedGroupKFold(n_splits=CV["n_folds"], shuffle=True, random_state=seed + r)
        for k, (_, te) in enumerate(sgk.split(d, y, groups=d.PATIENT_ID)):
            f[te] = k
        out[f"fold_r{r:02d}"] = f
    return pd.DataFrame(out, index=d.index)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build() -> None:
    s = pd.read_parquet(L.DERIVED / "samples.parquet")
    s_on = pd.read_parquet(L.DERIVED / "samples_A-on.parquet")
    gf = pd.read_parquet(L.DERIVED / "genomic_features.parquet")
    alt = pd.read_parquet(L.DERIVED / "alteration_matrix.parquet")
    afeat = pd.read_csv(L.DERIVED / "alteration_features.csv").set_index("feature")
    tv = C.config("target_variants")["cohorts"]

    s = build_covariates(s, gf)
    s["label_A_on"] = s.SAMPLE_ID.isin(s_on.loc[s_on.label_A == 1, "SAMPLE_ID"]).astype(float)
    s.loc[s.label_A == 0, "label_A_on"] = 0.0
    s["main_analysis"] = s.tier_strict & ~s.hypermutated
    # S3 survivorship: label-0 kept only if followed >= cohort's median post t_days.
    med_post = s[s.label_A == 1].groupby("cohort").t_days.median()
    s["s3_survivorship_keep"] = (s.label_A == 1) | (s.followup_days >= s.cohort.map(med_post))

    # ---- feature blocks ------------------------------------------------------
    base_alt = [c for c in alt.columns if re.search(r"_(mut|amp|del|sv|loh)$", c) and not afeat.loc[c, "variant_level"]]
    clon = [c for c in alt.columns if re.search(r"_mut_(clonal|subclonal|clonality_unknown)$", c)]
    ccf = [c for c in alt.columns if c.endswith("_mut_max_ccf")]
    variant_cols = [c for c in alt.columns if afeat.loc[c, "variant_level"]]

    # ---- checklist 11 / 13 ---------------------------------------------------
    alt_block = base_alt + clon + ccf + variant_cols
    bad = [c for c in alt_block if re.search(r"(?i)signature|level_|oncokb|treatment|n_mutations|highest_", c)]
    C.check(not bad, "11. no signature / OncoKB evidence / treatment field / n_mutations_oncogenic in the alteration block")

    # ---- leakage check: Target A covariates ---------------------------------
    leak_rows = []
    for cohort in C.COHORTS:
        d = s[(s.cohort == cohort) & s.main_analysis]
        for c, t in COVARIATES_A.items():
            if c not in d or c in COHORT_ONLY and cohort not in COHORT_ONLY[c]:
                continue
            x = d[c]
            levels = x.astype(str).unique() if t == "cat" else [None]
            for lv in levels:
                v = (x.astype(str) == lv).astype(float) if lv is not None else x.astype(float).fillna(x.astype(float).median())
                if v.nunique() < 2:
                    continue
                auc = roc_auc_score(d.label_A, v)
                leak_rows.append({"cohort": cohort, "covariate": c if lv is None else f"{c}={lv}",
                                  "auc": max(auc, 1 - auc)})
    leak = pd.DataFrame(leak_rows).sort_values("auc", ascending=False)
    C.check(leak.auc.max() < 0.9, f"leakage: no Target-A covariate separates the label alone "
            f"(max single-covariate AUC {leak.auc.max():.2f}: {leak.iloc[0].covariate}, {leak.iloc[0].cohort})")
    # Why the B block is Target-B-only: share of label-0 vs label-1 samples with
    # a non-zero / non-missing value (near 0 for label 0 by construction).
    bguard = pd.DataFrame([{"covariate": c,
                            "label0_nonzero_pct": 100 * s.loc[s.label_A == 0, c].fillna(0).ne(0).mean(),
                            "label1_nonzero_pct": 100 * s.loc[s.label_A == 1, c].fillna(0).ne(0).mean()}
                           for c in COVARIATES_B if c in s and c != "days_since_class_stop_missing"]).set_index("covariate")
    C.check(not set(COVARIATES_A) & set(COVARIATES_B), "leakage guard: post-start exposure variables are kept out of the Target-A covariates")

    # ---- controls in every tier + S3 (T4; checklist 17 data-level) ----------
    ctrl = tier_controls(s, alt)
    old = pd.read_csv(L.DERIVED / "controls.csv")
    old = old[~old.stage.isin(ctrl.stage.unique())]
    pd.concat([old, ctrl], ignore_index=True).to_csv(L.DERIVED / "controls.csv", index=False)
    # Checklist 17 gates every tier and S3; label variants (A-on) are reported only
    # (CRC A-on has ~26 post samples, so KRAS swings by sampling noise alone).
    gated = ctrl.feature.isin(GATED_CONTROLS) & ~ctrl.stage.str.startswith("A-on")
    for _, r in ctrl[gated].iterrows():
        if r.feature == "KRAS_any":
            ok = abs(r.pct_post - r.pct_pre) < 10
            C.check(ok, f"17. KRAS not enriched [{r.stage}] ({r.pct_pre:.1f}% -> {r.pct_post:.1f}%)")
        else:
            ok = r.pct_post > r.pct_pre
            C.check(ok, f"17. {r.control} enriched post [{r.stage}] ({r.pct_pre:.1f}% -> {r.pct_post:.1f}%, "
                        f"n={r.n_pre}/{r.n_post})")

    # ---- model tables + folds ------------------------------------------------
    pg_341 = set(afeat.loc[afeat.on_IMPACT341, "gene"])
    fold_rows, dict_rows, written = [], {}, {}
    seed = CV["seed"]
    for cohort in [*C.COHORTS, "pooled"]:
        d = s if cohort == "pooled" else s[s.cohort == cohort]
        d = d.reset_index(drop=True)
        A = alt.reindex(d.SAMPLE_ID).reset_index(drop=True)
        own_tv = [c for c in variant_cols if cohort == "pooled" or c in tv.get(cohort, {})]
        alt_cols = [c for c in base_alt + clon + ccf if (A[c] == 1).any() or (c in ccf and A[c].notna().any())]
        feat_cols = alt_cols + own_tv
        main_cols = [c for c in feat_cols if afeat.loc[c, "gene"] in pg_341 and c not in ccf]
        # LOH is NA where FACETS gave no allele-specific call for the gene (often
        # X-linked genes in males), not because of the panel. Such LOH features
        # go to the secondary analysis; nothing else may be dropped this way.
        main_rows = d.main_analysis.to_numpy()
        na_main = [c for c in main_cols if A.loc[main_rows, c].isna().any()]
        C.check(all(c.endswith("_loh") for c in na_main),
                f"[{cohort}] only LOH features have unmeasured cells among IMPACT341 genes "
                f"({len(na_main)} LOH features moved to the secondary analysis)")
        main_cols = [c for c in main_cols if c not in na_main]
        cov_a = [c for c in COVARIATES_A if c not in COHORT_ONLY or cohort in COHORT_ONLY[c]]
        cov_b = [c for c in COVARIATES_B if c not in COHORT_ONLY or cohort in COHORT_ONLY[c]]
        cov_a = [c for c in cov_a if c in d and not (cohort != "pooled" and c == "drug_class")]
        cov_b = [c for c in cov_b if c in d]

        # checklist 9: main-analysis features are all on IMPACT341, so no NA.
        mains = A.loc[main_rows, main_cols]
        C.check(mains.notna().all().all(),
                f"9. [{cohort}] main-analysis alteration features are all on IMPACT341 (no NA)")

        table = pd.concat([d[ID_COLS + LABEL_COLS + FLAG_COLS + ["subclass", "panel_version"]]
                           .rename(columns={"subclass": "_subclass", "panel_version": "_panel_version"}),
                           d[[*dict.fromkeys(cov_a + cov_b + list(GENOMIC))]], A[feat_cols]], axis=1)
        table = table.drop(columns=["_subclass", "_panel_version"])
        # Folds per tier (and A-on for the strict tier).
        for tier in TIERS:
            m = d[f"tier_{tier}"].to_numpy() & ~d.hypermutated.to_numpy() if tier != "relaxed" \
                else np.ones(len(d), bool)
            sub = d[m]
            strat = sub.label_A.astype(str) + "_" + (sub.cohort if cohort == "pooled" else sub.subclass)
            fd = assign_folds(sub, strat, seed)
            long = fd.assign(SAMPLE_ID=sub.SAMPLE_ID.values).melt("SAMPLE_ID", var_name="repeat", value_name="fold")
            fold_rows.append(long.assign(cohort=cohort, tier=tier, label_variant="A"))
            if tier == "strict":
                table = table.join(fd)            # aligned on index; NaN outside the main analysis
        sub = d[d.main_analysis & ((d.label_A == 0) | (d.label_A_on == 1))]
        fd = assign_folds(sub, sub.label_A.astype(str), seed)
        fold_rows.append(fd.assign(SAMPLE_ID=sub.SAMPLE_ID.values).melt("SAMPLE_ID", var_name="repeat", value_name="fold")
                         .assign(cohort=cohort, tier="strict", label_variant="A-on"))

        # Grouping check: a patient never spans folds within a repeat.
        fr = table[["PATIENT_ID", "fold_r00"]].dropna()
        C.check((fr.groupby("PATIENT_ID").fold_r00.nunique() <= 1).all(),
                f"[{cohort}] folds are patient-grouped")

        out = L.DERIVED / f"model_table_{cohort}.parquet"
        table.to_parquet(out, index=False)
        written[cohort] = table.shape

        # Firth gene list: full cohort, main analysis, single list per cohort (9.6).
        dm = d.main_analysis.to_numpy()
        for c in feat_cols:
            x = A.loc[:, c]
            row = dict_rows.setdefault(c, {"feature": c})
            for lab, nm in [(0, "pre"), (1, "post")]:
                v = x[(d.label_A == lab).to_numpy() & dm]
                row[f"{cohort}_pct_{nm}"] = 100 * (v == 1).sum() / max(v.notna().sum(), 1) if c not in ccf else np.nan
            row[f"{cohort}_coverage"] = x.notna().mean()
            if c not in ccf:
                n = int((x[dm] == 1).sum())
                frac = n / max(x[dm].notna().sum(), 1)
                row[f"{cohort}_n_altered_main"] = n
                row[f"{cohort}_kept_firth"] = bool(c in own_tv or (n >= F["min_altered_n"] and frac >= F["min_altered_frac"]))
                row[f"{cohort}_main_analysis_feature"] = c in main_cols

        # Smoke test of the in-fold preprocessor on one split (checklist 15 groundwork).
        if cohort != "pooled":
            mt = table[table.main_analysis]
            tr = mt[mt.fold_r00 != 0]
            alt_main = main_cols
            types = {**{c: COVARIATES_A[c] for c in cov_a}, **GENOMIC}
            pre = make_preprocessor(alt_main, types, tuple(own_tv)).fit(tr)
            Xte = pre.transform(mt[mt.fold_r00 == 0])
            C.check(Xte.notna().all().all(), f"[{cohort}] in-fold preprocessor fits on training rows "
                    f"and yields a complete design matrix ({Xte.shape[1]} columns)")

    pd.concat(fold_rows, ignore_index=True).to_parquet(L.DERIVED / "folds.parquet", index=False)

    # ---- feature dictionary --------------------------------------------------
    fdict = pd.DataFrame(dict_rows.values()).set_index("feature")
    meta = afeat.reindex(fdict.index)
    fdict.insert(0, "block", np.select(
        [fdict.index.isin(variant_cols), fdict.index.str.endswith("_max_ccf"),
         fdict.index.str.contains(r"_mut_(?:clonal|subclonal|clonality_unknown)$")],
        ["variant_target", "ccf (post-only / GB only)", "clonality"], "alteration"))
    fdict.insert(1, "gene", meta.gene)
    fdict.insert(2, "data_type", meta.data_type)
    fdict.insert(3, "type", fdict.index.str.extract(r"_(mut|amp|del|sv|loh)", expand=False))
    fdict.insert(4, "source", meta.data_type.map({"mutations": "data_mutations.txt", "gistic": "data_cna.txt / data_CNA.ascna.txt",
                                                  "structural_variants": "data_sv.txt"}))
    fdict.insert(5, "transform", np.where(fdict.index.str.endswith("_max_ccf"), "max CCF, NA if not mutated",
                                          "binary, NA off-panel"))
    fdict.insert(6, "on_IMPACT341", meta.on_IMPACT341)
    fdict.insert(7, "oncogenicity_source", "rules_v1")
    other = []
    for blk, spec in [("covariate_A", COVARIATES_A), ("covariate_B (Target B only)", COVARIATES_B), ("genomic", GENOMIC)]:
        for c, t in spec.items():
            other.append({"feature": c, "block": blk, "type": t,
                          "transform": {"num": "median-impute + scale in fold", "cat": "one-hot in fold, NaN -> 'Unknown'",
                                        "bin": "0/1"}[t],
                          "cohorts": ",".join(COHORT_ONLY.get(c, C.COHORTS))})
    fdict = pd.concat([fdict.reset_index(), pd.DataFrame(other)], ignore_index=True)
    fdict.to_csv(L.DERIVED / "feature_dictionary.csv", index=False)

    # ---- checklist 13 --------------------------------------------------------
    # v1 outputs kept for the exploration notebook (plan Section 3) are not v3 outputs.
    v1 = {"cohort.parquet", "het_features.parquet", "treatment.parquet"}
    hits = [p.name for p in L.DERIVED.glob("*.parquet")
            if p.name not in v1 and "arm_altered_fraction" in pq.read_schema(p).names]
    C.check(not hits, "13. arm_altered_fraction does not exist in any output")

    # ---- log -------------------------------------------------------------------
    sizes = s.groupby("cohort").agg(relaxed=("tier_relaxed", "sum"), strict=("tier_strict", "sum"),
                                    strict_noint=("tier_strict_noint", "sum"), main=("main_analysis", "sum"),
                                    hypermutated=("hypermutated", "sum"))
    main_lab = s[s.main_analysis].groupby(["cohort", "label_A"]).size().unstack()
    nfirth = {c: int(fdict.get(f"{c}_kept_firth", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
              for c in [*C.COHORTS, "pooled"]}
    C.write_log("preprocess", f"""
Stages 7-9 (preprocess.py). {len(s):,} samples in; model tables written per cohort and pooled
({', '.join(f'{k}: {v[0]:,} x {v[1]:,}' for k, v in written.items())}). Main analysis = strict tier
minus hypermutated samples. Folds: StratifiedGroupKFold {CV['n_folds']} x {CV['n_repeats']} repeats, grouped by
patient, stratified by label x subclass (label x cohort for pooled), for every tier plus A-on.
Firth gene lists (full-cohort prevalence >= {F['min_altered_n']} and >= {F['min_altered_frac']:.0%}, main analysis,
variant-level targets always kept): {nfirth}. Exposure variables defined only after class start
(on_class_at_collection, days_since_class_stop, n_lines_between, *_before_collection) are Target-B-only
covariates: they are zero for >= 99.8% of label-0 samples by construction (table below).
""", {"Tier sizes": sizes, "Main analysis labels": main_lab,
      "Top single-covariate AUCs (Target A, main analysis)": leak.head(12).set_index(["cohort", "covariate"]),
      "Target-B-only exposure variables: % non-zero by label": bguard,
      "Controls by tier (T4)": ctrl.set_index(["cohort", "control", "stage"])[["n_pre", "n_post", "pct_pre", "pct_post"]]})
    print("model tables:", written)


def tier_controls(s: pd.DataFrame, alt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stages = {"strict": s.tier_strict & ~s.hypermutated, "strict_noint": s.tier_strict_noint & ~s.hypermutated,
              "relaxed": s.tier_relaxed, "S3_survivorship_strict": s.tier_strict & ~s.hypermutated & s.s3_survivorship_keep,
              "A-on_strict": s.tier_strict & ~s.hypermutated & ((s.label_A == 0) | (s.label_A_on == 1))}
    for stage, m in stages.items():
        for cohort, feats in CONTROL_FEATURES.items():
            for feat, sub in feats:
                d = s[m & (s.cohort == cohort)]
                if sub:
                    d = d[d.subclass == sub]
                v = pd.Series(alt.reindex(d.SAMPLE_ID)[feat].to_numpy(), index=d.index)
                pre, post = v[d.label_A == 0].dropna(), v[d.label_A == 1].dropna()
                rows.append({"cohort": cohort, "control": feat + (f" ({sub} subclass)" if sub else ""),
                             "feature": feat, "stage": stage, "n_pre": len(pre), "n_post": len(post),
                             "pct_pre": 100 * pre.mean(), "pct_post": 100 * post.mean(),
                             "expected": "flat" if feat == "KRAS_any" else "up"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    build()
