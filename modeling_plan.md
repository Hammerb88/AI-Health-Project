# Modeling, SHAP and Visualization Plan (M3)

**Project:** Pseudo-longitudinal resistance signatures: which genomic alterations are enriched
in tumors sampled after, and longer into, first-line therapy
**Builds on:** `data_cleaning_plan.md` (v3, milestones M1–M2 complete), `data/derived/`
(see `data/derived/DATA_DICTIONARY.md` for every column)
**Replaces:** Section 13 and the M3 rows of `data_cleaning_plan.md` where they conflict.
**Revision:** v2, 2026-11-06. Section 11 lists what changed from v1 and why.

All counts below come from the current `data/derived/` outputs and were re-checked against
them on 2026-11-06.

---

## 0. Decisions this plan makes

| # | Decision | Choice | Why |
|---|---|---|---|
| P1 | Training data | **Relaxed tier minus hypermutated samples** | Roughly doubles post-treatment samples in breast and NSCLC (Section 1). **Reverses D1** of the cleaning plan for the predictive models only. |
| P2 | Confirmation data | **Strict tier** (`main_analysis`) | Per-gene effects found in relaxed data are confirmed with Firth regression on the clean strict tier (Section 7). |
| P3 | Targets | **Two-part:** (A) pre vs post classifier; (B) among post samples, `log1p(cum_days_on_class)` regression | Time *before* treatment is not modeled (Section 10). |
| P4 | Models | Elastic-net logistic regression, random forest, gradient-boosted trees (XGBoost), plus baselines | Section 4 explains why ridge classifier, SVM, naive Bayes, a single tree and a neural network are dropped. |
| P5 | Primary metric | Out-of-fold **ROC-AUC**, with PR-AUC, Brier score and calibration. Computed on the pooled OOF predictions of each repeat (20 estimates); the fold-level spread is reported as variability, not as a confidence interval | Section 5.1 |
| P6 | Model selection | Highest mean ROC-AUC on the full feature set; a more complex model must beat the simpler one by > 0.01 **paired** AUC in ≥ 80% of folds, otherwise the simpler model wins | Fold estimates in repeated CV are correlated, so a naive standard error is too small; the paired rule needs no distributional assumption (Section 5.2) |
| P7 | SHAP | Out-of-fold SHAP from the selected model, all 20 repeats; **alteration features plotted, covariates reported separately** | Section 6 |
| P8 | Cohorts | Breast, NSCLC, prostate, CRC separately, plus pooled (drug class as a covariate) | Drug-target genes need per-cohort models; shared genes gain power when pooled. |
| P9 | Stratified analysis | **Metastasis-only** (and primary-only) models and a stratified enrichment table are planned, not optional | Post samples are 84–87% metastases in breast and prostate versus 17–18% of pre samples. The positive controls survive restriction to metastatic biopsies, so the stratum is the cleanest test that a signal is not sampling site (Sections 5.5, 6.5). |
| P10 | Target B rows | Post samples with < 30 recorded days on the class are excluded from the dose model and reported | 12–16% of breast and NSCLC post samples have ≤ 14 days on class despite a median of 1.5–3 years since the first dose; these are treatment-record artifacts, not exposure (Section 1.1). |

---

## 1. Data

**Rows.** For each cohort, `model_table_<cohort>.parquet` rows with `tier_relaxed == 1` and
`hypermutated == 0`.

| Cohort | Pre (label 0) | Post (label 1) | % post |
|---|---:|---:|---:|
| Breast / endocrine | 1,777 | 795 | 31% |
| NSCLC / EGFR-TKI | 559 | 191 | 25% |
| Prostate / ADT | 991 | 489 | 33% |
| CRC / FOLFOX-CAPOX | 1,778 | 478 | 21% |
| Pooled | 5,105 | 1,953 | 28% |

**Folds.** Reuse the `relaxed` folds in `derived/folds.parquet` (5 folds × 20 repeats,
grouped by patient), dropping hypermutated rows. Patient grouping is preserved. Target B
uses the same folds restricted to post samples.

**Hold-out policy.** There is no separate test set: every sample is held out exactly once
per repeat, and all metrics are out-of-fold (OOF). With 191–795 post samples per cohort, a
single split would rank models by chance.

**Pooled table.** 7,293 rows but 7,289 patients: four patients are in two cohorts. The pooled
folds are grouped by `PATIENT_ID`, so both rows of such a patient land in the same fold
(assert 3 covers this).

### 1.1 Target B label quality

`label_B` (`cum_days_on_class`) is a sum of treatment-episode lengths, and 10% of all
episodes in `exposure.parquet` are 0–1 day long (single-day administration records). Among
post samples in the training data:

| Cohort | Post | `label_B` < 30 d | ≤ 14 d | Median `t_days` | Median `label_B` |
|---|---:|---:|---:|---:|---:|
| Breast | 795 | 112 (14%) | 93 | 1,139 | 484 |
| NSCLC | 191 | 30 (16%) | 24 | 533 | 259 |
| Prostate | 489 | 21 (4%) | 18 | 1,041 | 564 |
| CRC | 478 | 31 (6%) | 13 | 404 | 99 |

A sample taken three years after the first dose with eight recorded days on the drug is a
record-keeping gap, not a short exposure. **Rule (P10):** rows with `label_B` < 30 are
excluded from the Target B models and counted in MT5; a sensitivity run includes them. For
the three cohorts on continuous therapy (breast, NSCLC, prostate), `log1p(t_days)` (time
since the first dose, which does not depend on stop dates) is a **secondary Target B
outcome**, reported beside the primary. CRC keeps only the primary outcome, because FOLFOX is
a fixed course and time since start is not exposure there (cleaning plan, Section 2.3).

---

## 2. Step 0: prerequisites

1. **Fix the target-column bug in `preprocess.py`.** `ERBB2_amp`, `MET_amp`, `AR_amp` and
   `AR_sv` are named like gene × type features but are kept only in the cohort where each
   is a variant-level target, because `base_alt` in `preprocess.build()` drops every
   column flagged `variant_level` in `alteration_features.csv`. As a result, ERBB2
   amplification (10.8% of breast samples, 2.8% of CRC; confirmed from
   `alteration_matrix.parquet`) is missing from those model tables. Fix: treat the four
   CNA/SV target columns as ordinary alteration features in every cohort (subject to the
   prevalence filter) and pass them to `always_keep` only in their target cohort. The
   definitions coincide: ERBB2, MET and AR are all `amp_driver` in
   `oncogenicity_rules.parquet`, so the target rule (CNA = 2) and the pattern rule agree.
   Note that ERBB2 amplification is **not** enriched post-treatment in breast (11.0% pre,
   10.2% post); the fix is about completeness, not an expected finding. Rerun
   `preprocess.py` and confirm the controls still pass.
2. **Environment (checked 2026-11-06: Python 3.14.3, scikit-learn 1.9.0, statsmodels
   0.14.6).** `pip install xgboost shap`: a dry run resolves `xgboost` 3.4.1 and `shap`
   0.54.1 wheels for this interpreter, so both are expected to install. Then confirm they
   import under the Windows Application Control policy. Fallbacks:
   - If `xgboost` is blocked: scikit-learn `HistGradientBoostingClassifier` / `Regressor`
     (installed; native missing-value handling, built-in early stopping, supported by
     SHAP's TreeExplainer). Prefer it over `GradientBoostingClassifier`, which is slower and
     cannot take NaN inputs (needed for F3).
   - If `shap` is blocked: exact linear SHAP computed directly (coefficient × centred
     feature) for logistic regression, and scikit-learn permutation importance for trees.
     Report that SHAP was unavailable.
   - `firthlogist` **cannot be installed**: every release requires Python < 3.11. The
     NumPy Firth implementation in Section 7.2 is therefore the only path, and it needs the
     validation test described there.
   - `sklearn.calibration`, `sklearn.compose` and `sklearn.impute` all import on this
     machine now (the earlier statement that they were blocked is stale, as is the note in
     the `FoldPreprocessor` docstring). `calibration_curve` may be used; the hand-written
     `FoldPreprocessor` stays because it already works and is tested.
3. **Leakage re-check on the relaxed tier.** Done during planning: the largest
   single-covariate AUC is 0.85 (breast `sample_type`), so no covariate encodes the label.
   `prior_med_to_msk` is 0 for every pre sample in the relaxed tier, but it is not in the
   model tables and must never be added. The training script asserts this.

---

## 3. Exact model inputs

### 3.1 Columns never used as inputs

Identifiers and labels: `SAMPLE_ID`, `PATIENT_ID`, `cohort`, `label_A`, `label_A_on`,
`label_B`, `label_B_log1p`, `t_days`.
Analysis flags: `tier_*`, `main_analysis`, `s3_survivorship_keep`, `hypermutated`,
`facets_pass`, `fold_r*`.
Survivorship: `followup_days` (post patients survive longer by construction).
Redundant: `biopsy_organ_coarse` (kept for the prognostic score in Section 7 only),
`low_purity` (a threshold of `FACETS_PURITY`).
Classifier only: every Target-B-only exposure variable (Section 3.5).
Feature-set choice: `<GENE>_loh`, `<GENE>_mut_clonal`, `<GENE>_mut_subclonal`,
`<GENE>_mut_clonality_unknown`, `<GENE>_mut_max_ccf` (Section 3.6).

### 3.2 Block C: covariates (Target A and B)

| Covariate | Type | Cohorts |
|---|---|---|
| `sample_type` | categorical (Primary / Metastasis / Local Recurrence / Unknown) | all |
| `biopsy_organ` | categorical (organ group; < 30 samples → Other) | all |
| `metastatic_at_collection` | binary | all |
| `adjuvant_start` | binary | all |
| `stage_at_dx` | categorical (Stage 1-3 / Stage 4 / Unknown) | all |
| `age_at_collection` | continuous | all |
| `panel_version` | categorical (IMPACT341/410/468/505) | all |
| `log_patient_id` | continuous (sequencing era) | all |
| `archived_tissue` | binary | all |
| `n_mutations_oncogenic_log1p` | continuous | all |
| `FACETS_PURITY`, `FACETS_PURITY_missing` | continuous + indicator | all |
| `n_prior_classes` | count (non-zero now that first-line is not required) | all |
| `subclass` | categorical | all |
| `sex_male` | binary | NSCLC, CRC |
| `HER2`, `companion_cdk46`, `ovarian_suppression` | categorical / binary | breast |
| `fluoropyrimidine_backbone` | binary | CRC |
| `drug_class` | categorical | pooled |

### 3.3 Block G: sample-level genomic features

`tmb_log1p`, `FGA`, `FGA_missing`, `MSI_SCORE`, `MSI_SCORE_missing`, `FACETS_PLOIDY`,
`FACETS_WGD`, `FACETS_missing`. These are interpretable features, not nuisance terms, but
they are shown in a separate SHAP panel from the gene-level alterations.

### 3.4 Block X: alteration features

**Rule.** Binary `<GENE>_mut`, `<GENE>_amp`, `<GENE>_del` and `<GENE>_sv` for genes on
IMPACT341 (checked: none of the 607 such columns has a missing cell among relaxed rows),
kept if altered in **≥ 10 samples and ≥ 2%** of the **training fold** (recomputed inside
each fold). This block also
includes the cohort's variant-level targets, which are always kept.

**Expected list.** Computed on the full relaxed tier; individual folds may add or drop
borderline features. ERBB2_amp joins breast and CRC once the Step 0 fix is in.

| Cohort | Mutations | Amplifications | Deletions | SVs | Always-kept targets |
|---|---|---|---|---|---|
| Breast (51 + targets) | AKT1, ARID1A, CBFB, CDH1, ESR1, FOXA1, GATA3, KMT2C, MAP2K4, MAP3K1, NCOR1, NF1, PIK3CA, PTEN, RUNX1, SF3B1, SPEN, TBX3, TP53 | AURKA, AXIN2, BRIP1, CCND1, CD79B, CDK12, FGF19, FGF3, FGF4, FGFR1, FOXA1, GNAS, IGF1R, IKBKE, IL10, MCL1, MDM2, MDM4, MYC, NBN, PAK1, PRKAR1A, RAD51C, RARA, RECQL4, RNF43, RPS6KB2, SOX9, SPOP (+ ERBB2) | CDKN2A, CDKN2B, MAP2K4 | — | ESR1_LBD, ESR1_fusion |
| NSCLC (35 + targets) | APC, CDKN2A, CTNNB1, EGFR, MED12, PIK3CA, PTEN, RB1, RBM10, SMAD4, TP53 | AKT2, CARD11, CCND1, CCNE1, CDK4, EGFR, ETV1, FOXA1, IKZF1, IL7R, MCL1, MDM2, MYC, NKX2-1, PMS2, RAC1, RECQL4, RICTOR, SDHA, TERT | CDKN2A, CDKN2B, NKX3-1, RB1 | — | EGFR_T790M, EGFR_C797X, EGFR_sensitizing, MET_amp, ERBB2_amp |
| Prostate (25 + targets) | APC, AR, BRCA2, CDK12, CTNNB1, FOXA1, KDM6A, KMT2C, PIK3CA, PTEN, SPOP, TP53 | CCND1, FOXA1, MYC, NBN, RECQL4, TBX3 | MAP3K1, NKX3-1, PTEN, RB1, TP53 | ERG, TMPRSS2 | AR_LBD, AR_amp, AR_sv |
| CRC (35 + targets) | AMER1, APC, ARID1A, ATM, BRAF, ERBB2, FBXW7, KRAS, NRAS, PIK3CA, PIK3R1, PTEN, RNF43, SOX9, TP53 | ASXL1, AURKA, BCL2L1, CCND2, CDK12, CDK8, DNMT3B, FGFR1, FLT1, FLT3, GNAS, IRS2, MYC, PTPRT, RECQL4, SRC, TOP1 (+ ERBB2) | NKX3-1, SMAD4 | — | KRAS_any, NRAS_any, BRAF_V600E |
| Pooled (44 + targets) | AKT1, APC, ARID1A, BRAF, CDH1, EGFR, ESR1, FBXW7, FOXA1, GATA3, KMT2C, KRAS, MAP3K1, PIK3CA, PTEN, SMAD4, SOX9, SPOP, TP53 | AURKA, BCL2L1, BRIP1, CCND1, CDK12, EGFR, FGF19, FGF3, FGF4, FGFR1, FOXA1, GNAS, MDM2, MYC, NBN, PAK1, RAD51C, RECQL4, RPS6KB2 | CDKN2A, CDKN2B, NKX3-1, PTEN | ERG, TMPRSS2 | all 13 targets |

Some targets duplicate gene-level features (e.g. `KRAS_any` ≈ `KRAS_mut`, `AR_mut` ⊃
`AR_LBD`). Keep both, because the target is the pre-specified hypothesis. SHAP credit
between the pair is reported as a pair (Section 6.4).

### 3.5 Block B: Target-B-only covariates (regression model only)

`on_class_at_collection`, `days_since_class_stop`, `days_since_class_stop_missing`,
`n_lines_between`, plus the cohort's `*_before_collection` flags
(breast: `cdk46_any_`, `pi3k_mtor_`; prostate: `ar_inhibitor_exposure_`; CRC:
`irinotecan_`, `anti_egfr_mab_`, `anti_vegf_`). They are 0 or missing for every
pre-treatment sample, so they never enter the classifier.

### 3.6 Feature sets compared

| Set | Blocks | Purpose |
|---|---|---|
| **F0** | C | Covariate-only baseline: how much sampling context alone predicts the label |
| **F1** | X | Alterations only |
| **F2** | C + G + X | **Full model; used for model selection and SHAP** |
| F3 (secondary) | C + G + X + LOH + clonality triples, restricted to FACETS-pass rows, trees only | Do LOH or clonality add signal? Clonality triples replace `<GENE>_mut` (never both, because they sum to it). |

The headline result is **ΔAUC = AUC(F2) − AUC(F0)**, the predictive value the genome adds
beyond sampling context, computed fold by fold.

### 3.7 In-fold transforms

Fitted on the training rows of each fold only, using `preprocess.FoldPreprocessor`:

| Step | Applies to | Detail |
|---|---|---|
| Prevalence filter | X | ≥ 10 samples and ≥ 2% altered |
| Median imputation | C/G continuous | — |
| Standardization | C/G continuous | **Logistic regression only**; trees use unscaled values |
| One-hot encoding | C categorical | Levels unseen in training are dropped |
| Missing indicators | — | Already columns; not refitted |

---

## 4. Models

### 4.1 Models trained

| Model | Implementation | In-fold tuning grid | Notes |
|---|---|---|---|
| Baseline: prevalence | predict training-fold % post | — | Floor for PR-AUC and Brier |
| Baseline: covariate-only | ridge logistic regression on F0 | C ∈ {0.01, 0.1, 1} | Also produces the prognostic score (Section 7) |
| **Elastic-net logistic regression** | `LogisticRegression(solver="saga", penalty="elasticnet", max_iter=5000)` | C ∈ {0.01, 0.03, 0.1, 0.3, 1}; l1_ratio ∈ {0, 0.5, 1} | Covariates are approximately unpenalized by scaling their columns by 10 after standardization, as the cleaning plan asks. This weakens the L2 part of their penalty 100× but the L1 part only 10×, so at l1_ratio = 1 covariates can still be zeroed; record how often a covariate coefficient is exactly 0 |
| **Random forest** | `RandomForestClassifier(n_estimators=500)` | max_features ∈ {sqrt, 0.3}; min_samples_leaf ∈ {5, 20} | Robust default for sparse binary data |
| **Gradient-boosted trees** | `XGBClassifier(n_estimators≤500, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)`; fallback `HistGradientBoostingClassifier(learning_rate=0.05, max_iter≤500)` | max_depth ∈ {2, 3, 4}; min_child_weight ∈ {1, 5}; reg_lambda ∈ {1, 5} (HGB: max_depth, min_samples_leaf ∈ {5, 20}, l2_regularization) | Shallow trees: ~750–2,600 rows per cohort. Early stopping runs on the inner validation fold; the refit on the full outer training fold uses the median best iteration across the three inner folds |

**Tuning** is nested. Inside each outer training fold, 3-fold `StratifiedGroupKFold` (by
patient) picks hyperparameters by **log-loss**, a proper scoring rule that rewards
calibrated probabilities. The best setting is refit on the whole outer training fold. No
class weighting (21–33% post is mild imbalance); calibration is checked instead.

### 4.2 Models dropped, and why

| Model | Reason |
|---|---|
| Ridge classifier | Same linear model as L2 logistic regression (already in the elastic-net grid, l1_ratio = 0) but without probabilities. Calibration would need `sklearn.calibration`, which does not import here. |
| SVM (linear or RBF) | A linear SVM duplicates logistic regression. An RBF kernel has little to exploit in ~60–100 mostly binary features; it needs Platt scaling and Kernel SHAP, which is slow and approximate. |
| Naive Bayes (Bernoulli) | The independence assumption is badly violated (co-amplified clusters such as CCND1/FGF3/FGF4/FGF19; CDKN2A/CDKN2B). Probabilities are poorly calibrated, and mixing in continuous covariates is awkward. |
| Single decision tree | Unstable with a few hundred post samples; dominated by the random forest and boosting. |
| Neural network (MLP) | With ~750–2,600 rows and ~100 features it is unlikely to beat logistic regression. SHAP would need the Kernel or Deep explainer, and `torch` is not installed. Revisit only if the learning curve (Section 5.4) is still rising steeply for the tree models. |

### 4.3 Target B regression (post samples only)

Outcome: `label_B_log1p` = `log1p(cum_days_on_class)`. Inputs: F2 + block B. Rows: post
samples with `label_B` ≥ 30 (Section 1.1); secondary outcome `log1p(t_days)` for breast,
NSCLC and prostate.

| Model | Tuning |
|---|---|
| Baseline: training-fold median | — |
| Baseline: covariate-only ridge regression (C + B) | alpha ∈ {0.1, 1, 10} |
| Elastic-net regression | alpha ∈ {0.01, 0.1, 1}; l1_ratio ∈ {0, 0.5, 1} |
| Random forest regressor | as in 4.1 |
| XGBoost regressor | as in 4.1, squared-error objective |

Post samples per cohort after the 30-day rule: 683 / 161 / 468 / 447. Expect **little
signal in CRC**, because FOLFOX is a fixed course (`cum_days_on_class` is short and
uniform). Report it anyway and say so.

**NSCLC is exploratory for Target B.** 161 rows give ~32 test rows per fold and ~110 rows
per inner training split, too few for nested tuning to be meaningful. NSCLC uses fixed
settings (ridge alpha = 1; elastic net alpha = 0.1, l1_ratio = 0.5; random forest
min_samples_leaf = 20; boosting max_depth = 2) and is labelled exploratory in MT5. The other
three cohorts use the nested grids above.

---

## 5. Evaluation

### 5.1 Metrics

**Aggregation.** The primary estimate of every metric is computed on the **pooled OOF
predictions of one repeat** (all samples, each predicted once), then averaged over the 20
repeats. This is more stable than per-fold values for PR-AUC, Brier score and calibration,
which are noisy with ~38 post samples per NSCLC fold. Per-fold values (100 per model) are
also stored and their 2.5th–97.5th percentiles are shown as the **spread across folds**.
That spread is variability of the resampling, not a confidence interval: the 100 estimates
share training data and are strongly correlated, so a standard error computed from them is
too small. Where a significance statement is needed (MT1 footnote), use the Nadeau–Bengio
corrected resampled t-test on paired fold differences, and say that it is approximate.

| Target | Primary | Secondary |
|---|---|---|
| A (classifier) | ROC-AUC | PR-AUC (average precision; compare with % post), Brier score (compare with the prevalence baseline), log-loss, calibration intercept and slope, ΔAUC vs F0 |
| B (regression) | Spearman ρ (predicted vs observed days) | MAE in days (back-transformed with `expm1`), R² on the log scale, ΔMAE vs covariate-only |

### 5.2 Model selection

1. **Phase 1, screening:** all models × feature sets F0–F2, repeats r00–r04
   (25 outer fits per model and cohort).
2. **Selection:** per cohort, start from the simplest model (logistic regression < random
   forest < XGBoost). Move to a more complex model only if its **paired** gain over the
   current choice, computed fold by fold on the same held-out rows, averages > 0.01 ROC-AUC
   **and** is positive in ≥ 80% of the 25 folds. Report every model and the paired
   differences (M1 inset), not just the winner.
3. **Phase 2, full run:** the selected model and elastic-net logistic regression (always
   kept as the interpretable reference) on all 20 repeats.

### 5.3 Evaluation on strict rows

The strict tier is a subset of the relaxed training rows, so the Phase 2 OOF predictions
can also be scored on strict rows only. This shows whether a model trained on looser data
still separates the cleanest samples.

### 5.4 Learning curve

For the selected model and logistic regression, retrain on 25%, 50%, 75% and 100% of each
training fold (repeats r00–r04), always testing on the full held-out fold. Plot ROC-AUC
and top-20 SHAP overlap against training size. A still-rising curve, likely for NSCLC and
prostate, means more data (e.g. lever C) is worth pursuing; a flat curve means it is not.

### 5.5 Sensitivity analyses (selected model only, Phase-1 repeats)

| Variant | Change |
|---|---|
| **Metastasis-only (P9)** | Rows restricted to `sample_type == "Metastasis"`. Pre / post: breast 325 / 693, NSCLC 261 / 102, prostate 164 / 412, CRC 294 / 296. F0, F2 and ΔAUC are all recomputed in the stratum, which removes the single strongest covariate (sample type) and most of the organ confounding |
| Primary-only (P9) | Rows restricted to `sample_type == "Primary"`. Post counts are small (95 / 84 / 75 / 178), so report AUC and the control gate only, no SHAP ranking |
| S3 survivorship | Pre rows restricted to `s3_survivorship_keep` |
| A-on | Post rows restricted to `label_A_on == 1` |
| Hypermutated included | Trees only; mutation features restricted to hotspot calls |
| Strict-trained | Train and evaluate on strict rows only, for comparison with 5.3 |
| Target B, all post rows | Target B without the 30-day exclusion of Section 1.1 |

The stratified variants reuse the relaxed folds (a stratum is a subset of each fold), so no
new fold assignment is needed. The prevalence filter is refit on the stratum's training rows.

---

## 6. SHAP

### 6.1 Computation

| Model | Explainer | Output scale |
|---|---|---|
| Elastic-net logistic regression | `shap.LinearExplainer`, background = training fold (interventional) | log-odds |
| XGBoost | `shap.TreeExplainer` (tree path-dependent) | log-odds (margin) |
| Random forest | `shap.TreeExplainer` | probability |

### 6.2 Out-of-fold aggregation

- Each fold's model explains only its **held-out** rows, so every sample gets one SHAP
  vector per repeat (20 in total).
- **Per-sample SHAP** = mean over the 20 repeats.
- A feature removed by the prevalence filter in a fold contributes 0 for that fold; record
  how often this happens.
- One-hot columns are summed back to their original variable (e.g. all `biopsy_organ=*`
  → `biopsy_organ`).
- **Global importance** = mean |SHAP| across samples.

### 6.3 Stability

For each of the 100 fold-models, rank features by mean |SHAP| on its held-out rows.

- **Selection frequency** = share of fold-models with the feature in the top 20.
- Features at **≥ 70%** are reported as robust.

### 6.4 Correlated features

Before reading individual gene ranks, group features whose pairwise phi correlation is
> 0.7 in the training data into clusters, for example:

- 11q13: CCND1, FGF3, FGF4, FGF19 amplification
- 9p21: CDKN2A, CDKN2B deletion
- a target and its gene-level twin (`KRAS_any` / `KRAS_mut`)

Report summed cluster SHAP next to member SHAP, because SHAP splits credit arbitrarily
inside a cluster.

### 6.5 Confounding checks

- **Covariate vs alteration share:** % of total |SHAP| from blocks C, G and X per cohort.
  This mirrors the ΔAUC result.
- **Stratified enrichment (all models, primary site check):** for every alteration feature
  in a cohort's top 20, tabulate % altered pre vs post overall, within metastases only and
  within primaries only, with the Mantel–Haenszel OR stratified by `sample_type` next to
  the crude OR. Flag a feature as **site-confounded** if the stratified log-OR is less than
  half the crude log-OR, or if the direction reverses in the metastasis-only stratum. This
  check does not depend on the model and is the one used for the agreement categories in
  Section 7.3. Expected values from the training data: ESR1_LBD 4.3% → 17.0% within breast
  metastases, AR_amp 4.3% → 39.8% within prostate metastases, EGFR_T790M 1.9% → 41.2%
  within NSCLC metastases, so the controls pass this check.
- **Site and era interaction (XGBoost only, secondary):** SHAP interaction values between
  the top 10 alteration features and `biopsy_organ` / `panel_version`. Report the share of
  each feature's |SHAP| that comes from the `biopsy_organ` interaction; use it to explain a
  stratified-enrichment flag, not as a flag on its own.
- **Control gate (stop and debug if it fails):**
  - Among each cohort's alteration features, ESR1_LBD (breast), EGFR_T790M and MET_amp
    (NSCLC), and AR_amp and AR_LBD (prostate) rank in the top 10 by mean |SHAP|, and
    push predictions toward post.
  - KRAS_any (CRC) is not in the top 10 and its mean signed SHAP is near 0.
  - The same positive controls hold in the metastasis-only variant. KRAS is **not** gated
    there: within CRC metastases it is 50.3% pre vs 39.2% post, a swing that is within
    sampling noise for 294 / 296 rows but outside the 10-point tolerance used in
    `preprocess.py`. Report it, do not gate on it.

---

## 7. Confirmation on the strict tier

### 7.1 What is confirmed

Every robust alteration feature from Section 6.3 (≥ 70% top-20 frequency), plus the
cohort's Firth-eligible list (`<cohort>_kept_firth` in `feature_dictionary.csv`), on
`main_analysis` rows.

### 7.2 Method

Per the cleaning plan, Section 13.4:

1. **Prognostic score:** a covariate-only ridge logistic regression on strict rows using
   the coarse 5-level `biopsy_organ_coarse`. Its **out-of-fold** predicted logit is
   `prog_score`.
2. **Per-feature Firth logistic regression:** `label ~ feature + prog_score` gives the
   adjusted OR, and `label ~ feature` the crude OR. Report 95% CIs, p-values and
   Benjamini–Hochberg q-values per cohort.
   - Firth is implemented directly: penalized Newton–Raphson with the Jeffreys-prior
     adjustment, about 40 lines in NumPy, with profile-likelihood CIs (Wald CIs if a
     profile fails to converge; the CI type is recorded per feature). `firthlogist` has no
     release for Python ≥ 3.11, so this is the only path (Section 2).
   - **Validation test, run before any result is written:** (a) on a non-separated
     simulated dataset with n = 5,000 the Firth estimates agree with `statsmodels` `Logit`
     maximum likelihood within 1%; (b) on a toy dataset with complete separation the Firth
     estimate is finite and the profile CI is finite on both sides; (c) on the
     endometrial-cancer example shipped with R `logistf` (coefficients published in its
     documentation) the estimates agree to two decimals. Store the test in
     `tests/test_firth.py`.
3. **Full-covariate check:** for the top 20 by q-value, also fit
   `label ~ feature + all covariates`, and flag ORs that move materially.

### 7.3 Agreement categories (relaxed SHAP vs strict Firth)

| Category | Rule |
|---|---|
| **Confirmed** | Same direction, strict q < 0.10 |
| Directional | Same direction, q ≥ 0.10 (strict is underpowered) |
| Discordant | Opposite direction |
| Site-confounded | Flagged by the stratified enrichment check in 6.5, regardless of the above |

---

## 8. Visualizations

All figures use `data/viz.py`. Colors are fixed: pre = slot 1 (blue), post = slot 2 (orange).
Each model keeps one color in every figure: covariate-only baseline in muted gray, logistic
regression slot 3, random forest slot 4, XGBoost slot 5. Static PNGs go to
`data/reports/models/`, with a `README.md` embedding them and the tables.

| ID | Figure | Form |
|---|---|---|
| M1 | Model comparison | Dot + interval plot: mean ROC-AUC (2.5–97.5% across folds) per model × feature set, one panel per cohort; F0 shown as a reference band |
| M2 | ΔAUC from the genome | Per-cohort distribution of fold-wise AUC(F2) − AUC(F0) |
| M3 | ROC and PR curves | Selected model, OOF, averaged across repeats, with the covariate-only curve and the prevalence line |
| M4 | Calibration | Predicted vs observed % post in 10 bins, selected model, per cohort |
| M5 | SHAP beeswarm | Top 20 **alteration** features per cohort (log-odds scale); positive controls labeled |
| M6 | SHAP importance + stability | Mean \|SHAP\| bars with top-20 selection frequency as a dot; 70% line; correlated clusters bracketed |
| M7 | Block share | Stacked bar of total \|SHAP\| from covariates / genomic features / alterations per cohort |
| M8 | Site interaction | For flagged features: SHAP dependence colored by `biopsy_organ` |
| M9 | Strict-tier confirmation | Forest plot of adjusted Firth ORs (95% CI) for robust SHAP features, colored by agreement category, crude OR as a hollow marker |
| M10 | Learning curve | ROC-AUC (and top-20 overlap) vs training fraction, selected model vs logistic regression |
| M11 | Target B | Predicted vs observed cumulative days (log axes), post samples, per cohort, with Spearman ρ; SHAP beeswarm for the dose model |
| M12 | Stratified results | Per cohort: ROC-AUC of F0 and F2 overall vs metastasis-only vs primary-only (dot + spread), and a paired dot plot of % altered pre vs post within metastases for the top 20 alteration features, site-confounded features marked |

### Tables

| ID | Table |
|---|---|
| MT1 | All models × feature sets × cohorts: ROC-AUC, PR-AUC, Brier, log-loss, calibration (mean and interval) |
| MT2 | Selected model per cohort and selection rationale |
| MT3 | SHAP ranking: mean \|SHAP\|, signed mean, selection frequency, cluster, site-confounding flag |
| MT4 | Strict-tier Firth results: crude / adjusted / full-covariate OR, CI, CI type, q, agreement category |
| MT5 | Target B regression metrics |
| MT6 | Sensitivity analyses (Section 5.5) |
| MT7 | Stratified enrichment: crude and Mantel–Haenszel OR by `sample_type`, % pre / post per stratum, site-confounded flag, for every top-20 feature per cohort |

---

## 9. Implementation

### 9.1 Files

```
data/
  train_models.py      # Steps 0-5: folds, nested tuning, OOF predictions, metrics
  explain.py           # Section 6: OOF SHAP, stability, clusters, interactions
  confirm_strict.py    # Section 7: prog_score, Firth, agreement
  report_models.py     # Section 8: figures + tables + reports/models/README.md
  derived/models/
    oof_predictions.parquet   # cohort x target x model x feature_set x repeat x sample
    metrics_folds.parquet     # one row per outer fold
    best_params.parquet       # tuned hyperparameters per outer fold
    shap_oof.parquet          # sample x feature mean SHAP (+ per-repeat file)
    shap_stability.csv
    firth_strict.csv
```

### 9.2 Asserts (run inside the scripts, as in the cleaning plan)

1. No column from Section 3.1 is in X; `prior_med_to_msk`, `followup_days`, `t_days` and
   `label_*` are absent.
2. Every transform is fitted on training rows only (unit test: the fold's preprocessor
   never sees held-out SAMPLE_IDs).
3. A patient never appears in both the training and held-out rows of a fold.
4. Block B features are absent from every classifier.
5. `prog_score` is out-of-fold.
6. Every Firth fit records its CI type.
7. The positive / negative control gate in Section 6.5 passes before any figure is written.
8. Target B training rows all have `label_B` ≥ 30 unless the run is the "all post rows"
   sensitivity; the excluded count per cohort matches Section 1.1.
9. The Firth validation test (Section 7.2) passes before `confirm_strict.py` writes output.
10. In the stratified variants, every training and held-out row has the stratum's
    `sample_type`; the fold assignment is unchanged from the relaxed folds.

### 9.3 Milestones

| Step | Done when |
|---|---|
| M3a | Step 0 fix applied; `xgboost` / `shap` import (or fallbacks chosen); assert 1 passes |
| M3b | Phase 1 screening complete; MT1 and M1 written; model selected per cohort |
| M3c | Phase 2 run; ROC/PR/calibration (M3, M4) and learning curve (M10) |
| M3d | OOF SHAP, stability, clusters, interactions; control gate passes; M5–M8 |
| M3e | Strict-tier Firth confirmation (MT4, M9) |
| M3f | Target B models (MT5, M11); sensitivity analyses (MT6); `reports/models/README.md` |

**Compute:** logistic regression and random forest are fast. XGBoost nested tuning is 12
grid points × 3 inner folds = 36 inner fits per outer fold (plus the refit); Phase 1 is
about 4,500 small fits across cohorts and feature sets, roughly an hour or two on this
machine. Phase 2 adds about 4× that for one model. The two stratified variants roughly
double Phase-1 cost for the selected model only. SHAP for trees is seconds per fold.

---

## 10. Out of scope / deferred

- **Time before treatment** (negative `t_days`). It reflects referral and clinical course,
  not resistance; the cleaning plan excludes it. If wanted, it would be a separate,
  explicitly exploratory regression among pre samples.
- **Lever C** (adding samples that are in MSK-CHORD but not MSK-IMPACT 50K, using CHORD's
  own genomic files). Pursue only if the learning curve is still rising, and after a batch
  check on overlapping samples.
- **Neural network / SVM**: see Section 4.2.
- **OncoKB annotation**: still the rule-based fallback (`rules_v1`).

---

## 11. Changes in this revision (v2, 2026-11-06)

Everything here was checked against `data/derived/` and the installed Python environment.

| # | Change | Why |
|---|---|---|
| 1 | Metastasis-only and primary-only variants added as **planned** analyses (P9, 5.5, 6.5, M12, MT7) | In breast and prostate, 84–87% of post samples are metastatic biopsies versus 17–18% of pre samples, so the pre/post classifier is largely a primary/metastasis classifier. The ΔAUC design already accounts for this on average, but it cannot show that a specific gene's enrichment is not sampling site. The strata are large (e.g. 325 pre / 693 post breast metastases) and the positive controls survive restriction to them, so the stratified analysis is both feasible and the most direct confounding check. |
| 2 | Site-confounding is decided by **stratified enrichment** (Mantel–Haenszel OR by sample type), with SHAP interaction values demoted to a supporting diagnostic | The SHAP-interaction rule worked only for XGBoost and had an arbitrary 25% threshold. The stratified table works for every model, is readable without SHAP, and is the standard epidemiological check for a confounder. |
| 3 | Target B excludes post samples with < 30 recorded days on class; `log1p(t_days)` added as a secondary dose outcome for continuous-therapy cohorts (P10, 1.1, 4.3) | 14% of breast and 16% of NSCLC post samples have ≤ 14 recorded days despite a median of 1.5–3 years since the first dose; 10% of treatment episodes are 0–1 day records. These rows would dominate the low end of a log-scale regression with values that reflect record granularity, not exposure. |
| 4 | NSCLC Target B uses fixed hyperparameters and is labelled exploratory (4.3) | 161 usable rows cannot support nested tuning; a tuned result would be noise presented as a model choice. |
| 5 | Metrics are computed on each repeat's pooled OOF predictions; fold spread is described as resampling variability, not a confidence interval (5.1) | Per-fold PR-AUC and calibration with ~38 positives are unstable, and fold estimates in repeated CV are correlated, so a naive SE understates uncertainty. |
| 6 | Model selection uses a paired fold-wise rule (> 0.01 AUC gain in ≥ 80% of folds) instead of "within one standard error" (P6, 5.2) | Same reason: the standard error across correlated folds is not well defined. The paired rule compares models on identical held-out rows and needs no distributional assumption. |
| 7 | Step 0 rewritten with verified facts: `xgboost` 3.4.1 and `shap` 0.54.1 wheels resolve for Python 3.14; `firthlogist` cannot be installed at all; `sklearn.calibration`, `compose` and `impute` import fine; `HistGradientBoosting` replaces `GradientBoostingClassifier` as the fallback (2) | The v1 environment notes were partly stale. The NumPy Firth implementation is now mandatory, so it gets an explicit validation test (7.2, assert 9). |
| 8 | Step 0 fix documented precisely: the bug is the `variant_level` filter on `base_alt` in `preprocess.build()`; the target and pattern definitions agree because ERBB2, MET and AR are amp-dominant; ERBB2 amplification is not expected to be enriched (2) | So the fix can be applied without re-deriving it, and so nobody reads the restored ERBB2 column as a finding. |
| 9 | Elastic-net note corrected: the ×10 column scaling weakens the L2 penalty 100× but the L1 penalty only 10× (4.1) | The v1 text claimed ~100× for the whole elastic-net penalty, which is wrong at l1_ratio > 0. |
| 10 | Early-stopping refit rule stated for boosting (4.1); inner-fit count corrected from 24 to 36 (9.3) | The v1 plan did not say what `n_estimators` the refit on the full outer training fold should use. |
| 11 | Pooled-table note: 7,293 rows, 7,289 patients (1) | Four patients appear in two cohorts; the patient-grouped folds already handle it, but it should be known. |

**Not changed, and why.** P1 (train on the relaxed tier) stays: it doubles post samples, and
its cost, that relaxed post samples mix in prior and intervening therapies, is addressed by
the strict-tier confirmation (Section 7), the strict-trained and A-on sensitivities, and the
stratified analyses. The model set (P4) stays: nothing in the data argues for a model that
was dropped. The 90-day window, the prevalence filter and the fold scheme are inherited from
the cleaning plan and were not re-opened. One limitation to state in the write-up: therapy
exposure is inseparable from time in the disease course in this design, because any
"time since diagnosis" covariate would be nearly collinear with the label.

### 11.1 Implementation notes (M3 run, 2026-11-06)

| # | Deviation from the text above | Why |
|---|---|---|
| I1 | The elastic-net logistic regression is solved with L-BFGS-B on a split positive/negative parameterization (`m3_common.PenalizedLR`) instead of `saga`, and block-C covariates get **zero penalty** (plus a 0.01 ridge guard) instead of ×10 column scaling | `saga` did not converge in 5,000 iterations on this design (coefficients off by up to 0.6 at l1_ratio = 0) and took ~1 minute per repeat. The new solver matches a tightly converged `saga` fit to < 1e-5 in predicted probability, runs 20–100× faster, and implements the cleaning plan's "covariates enter unpenalized" exactly. The ×10 scaling is kept only for the Target B elastic-net regression, where coordinate descent handles it. |
| I2 | Firth validation check (c) uses the exact 2×2 identity (Firth = log OR with 0.5 added to every cell) instead of the logistf endometrial dataset | The endometrial data are not available offline; the 2×2 identity is an exact analytic result and tests the same code path. |
| I3 | XGBoost SHAP and interaction values come from the booster's `pred_contribs` / `pred_interactions`; linear SHAP is computed as coef × (x − training mean) | Both are the same algorithms as `shap.TreeExplainer` (path-dependent) and `shap.LinearExplainer` (interventional), without the per-call overhead. `shap` is used for the random forest. |
| I4 | Inner tuning refits the in-fold transforms on every inner training split | Keeps hyperparameter choice free of inner-validation leakage. |
| I5 | Gate rule for KRAS "mean signed SHAP near 0": mean SHAP among KRAS-altered samples below 10% of the largest such value among the top-10 alteration features | The plan did not define "near 0"; this makes it scale-free across model types. |
| I6 | Site-confounded rule (6.5) applies only to features with crude \|log-OR\| ≥ 0.1; full-covariate Firth ORs "move materially" when they differ from the adjusted OR by more than a factor of 1.5 | Both thresholds were unspecified. |

