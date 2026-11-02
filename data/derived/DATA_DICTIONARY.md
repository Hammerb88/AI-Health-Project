# Data dictionary: `data/derived/`

Every column in every file in this folder, defined **once**. The **Files** column lists
where a column appears, using the short names in the file index below. Where a column
name means something slightly different in one file, the difference is stated in the
same entry.

Produced by the v3 pipeline (`exposure.py` → `build_cohort.py` → `build_alterations.py`
→ `preprocess.py`), run by `run_all.py`. Section 6 covers the three legacy files from the
earlier (v1) design.

## Contents

1. [File index](#1-file-index)
2. [Conventions](#2-conventions)
3. [Sample, cohort and analysis columns](#3-sample-cohort-and-analysis-columns)
   - 3.1 Identifiers · 3.2 Time and treatment exposure · 3.3 Labels and analysis flags ·
     3.4 Clinical and sample covariates · 3.5 Sample-level genomic features
4. [Alteration columns](#4-alteration-columns)
   - 4.1 Gene-level matrix (pattern columns) · 4.2 Variant-level target features ·
     4.3 Alteration events (`alt_long`) · 4.4 Oncogenicity rules
5. [Summary and bookkeeping tables](#5-summary-and-bookkeeping-tables)
6. [Legacy v1 files](#6-legacy-v1-files)

---

## 1. File index

| Short name | File | One row per | Built by |
|---|---|---|---|
| `exposure` | `exposure.parquet` | patient × cohort drug class | `exposure.py` |
| `samples` | `samples.parquet` (label variant A) and `samples_A-on.parquet` (variant A-on); same columns | sample (one per patient per cohort) | `build_cohort.py` |
| `attrition` | `attrition.csv` | cohort × exclusion step (× label variant) | `build_cohort.py` |
| `plan_repro` | `plan_reproduction.csv` | cohort | `build_cohort.py` |
| `controls` | `controls.csv` | cohort × control alteration × pipeline stage | `build_cohort.py`, `build_alterations.py`, `preprocess.py` |
| `alt_long` | `alterations_long.parquet` | sample × alteration event | `build_alterations.py` |
| `alt_matrix` | `alteration_matrix.parquet` | sample (wide: gene × alteration-type features) | `build_alterations.py` |
| `alt_gene` | `alteration_matrix_gene.parquet` | sample (wide: one collapsed feature per gene) | `build_alterations.py` |
| `alt_features` | `alteration_features.csv` | feature in `alt_matrix` | `build_alterations.py` |
| `onco_rules` | `oncogenicity_rules.parquet` | gene | `build_alterations.py` |
| `hotspots` | `hotspots_rules.csv` | hotspot residue | `build_alterations.py` |
| `genomic` | `genomic_features.parquet` | sample | `build_alterations.py` |
| `signatures` | `signature_checks.csv` | cohort × signature check | `build_alterations.py` |
| `model` | `model_table_breast/nsclc/prostate/crc/pooled.parquet` | sample | `preprocess.py` |
| `folds` | `folds.parquet` | cohort × tier × label variant × repeat × sample | `preprocess.py` |
| `feat_dict` | `feature_dictionary.csv` | model feature or covariate | `preprocess.py` |
| `cohort_v1` | `cohort.parquet` | sample (v1 design) | `build_features.py` |
| `het_v1` | `het_features.parquet` | sample (v1 design) | `build_features.py` |
| `treatment_v1` | `treatment.parquet` | patient (v1 design) | `build_features.py` |

`logs/*.md` are per-stage run logs (row counts in and out), not tables.

## 2. Conventions

- **Day axis.** Every `*_day`, `*_date` and `*_start` column is an integer day offset on the
  MSK-CHORD axis, where **day 0 is the patient's first sequencing**. There are no calendar
  dates anywhere. Differences between two such columns are durations in days.
- **Time zero for the study** is `class_start`, the first dose of the cohort's drug class.
  `t_days` is measured from it.
- **Cohorts.** `breast` = HR-positive breast cancer / endocrine therapy; `nsclc` =
  EGFR-mutant non-small cell lung cancer / EGFR TKI; `prostate` = prostate cancer / ADT;
  `crc` = colorectal cancer / oxaliplatin (FOLFOX/CAPOX); `pooled` = all four together.
- **Pre / post.** Pre-treatment = label 0 (`t_days` ≤ 0). Post-treatment = label 1
  (`t_days` > 90). Samples with 0 < `t_days` ≤ 90 are excluded.
- **Missing values.** In alteration features, NA means **not measured** (the gene is not on
  the sample's panel for that data type, or no FACETS call exists). It never means "not
  altered"; a measured absence is 0. Booleans are stored as `True`/`False` in `samples` and
  `exposure` and as `1.0`/`0.0` in `model`.
- **Panels.** MSK-IMPACT panel versions are nested: IMPACT341 ⊂ IMPACT410 ⊂ IMPACT468 ⊂
  IMPACT505 (341, 410, 468 and 505 genes).

---

## 3. Sample, cohort and analysis columns

### 3.1 Identifiers

| Column | Files | Definition |
|---|---|---|
| `SAMPLE_ID` | all sample-level files, `alt_long`, `folds` | MSK-IMPACT tumor sample ID, `P-XXXXXXX-TNN-IMN`. The `IM3`/`IM5`/`IM6`/`IM7` suffix is the panel (341/410/468/505). |
| `PATIENT_ID` | `exposure`, `samples`, `model`, `cohort_v1`, `treatment_v1` | Patient ID, `P-XXXXXXX`. Patient numbers were assigned sequentially, so they track sequencing era. |
| `cohort` | `exposure`, `samples`, `model`, `folds`, `attrition`, `plan_repro`, `controls`, `signatures` | Cohort name (Section 2). `pooled` appears only in `folds` and `model_table_pooled`. |

### 3.2 Time and treatment exposure

| Column | Files | Definition |
|---|---|---|
| `collect_day` | `samples` | Day the tumor tissue was collected (biopsy or surgery), from MSK-CHORD `specimen_surgery.START_DATE`. |
| `seq_day` | `samples` | Day the sample was sequenced (`SEQ_DATE`). Usually after `collect_day`; a large gap means archived tissue. |
| `dx_date` | `exposure`, `samples` | Day of the earliest primary diagnosis of this cohort's cancer, identified by ICD-O topography (C50 breast, C34 lung, C61 prostate, C18–C20 colorectal). |
| `dx_missing` | `exposure` | True if no matching primary-diagnosis row exists; for these patients no pre-diagnosis treatment rows could be dropped. |
| `stage_at_dx` | `exposure`, `samples`, `model` | Stage at that primary diagnosis (`STAGE_CDM_DERIVED`): `Stage 1-3` or `Stage 4`. Missing in `exposure`/`samples`; recoded `Unknown` in `model`. |
| `class_start` | `exposure`, `samples` | **Study time zero.** Day of the first dose of any anchor drug in the cohort's class (breast: aromatase inhibitor, tamoxifen/toremifene, fulvestrant/elacestrant; NSCLC: osimertinib, erlotinib, gefitinib, afatinib, dacomitinib; prostate: GnRH agonist/antagonist or bicalutamide/nilutamide/flutamide; CRC: oxaliplatin). Rows more than 30 days before `dx_date` are ignored as belonging to an earlier cancer. |
| `class_end` | `exposure` | Last day of the last merged treatment episode of the class. |
| `n_episodes` | `exposure` | Number of treatment episodes after merging anchor-drug intervals separated by ≤ 60 days. |
| `episode_starts`, `episode_stops` | `exposure` | Lists of each episode's first and last day, in order. |
| `first_systemic_start` | `exposure` | Day of the earliest systemic anticancer drug of any class after `dx_date` − 30 (bone agents, topicals, ophthalmics, liver-directed Y-90 and intravesical BCG are not systemic). |
| `strict_first_line` | `exposure`, `samples` | True if the cohort's class was the first systemic therapy for this cancer: `class_start` ≤ `first_systemic_start` + 30. |
| `prior_classes` | `exposure`, `samples` | List of other systemic drug classes started between `dx_date` − 30 and `class_start` − 30. Class names come from `config/drug_classes.yaml`. |
| `n_prior_classes` | `exposure`, `samples`, `model` | Length of `prior_classes`. 0 by definition in the strict tier. |
| `concurrent_classes` | `exposure`, `samples` | List of other systemic classes started within ±30 days of `class_start`, excluding companion drugs (e.g. chemotherapy started with an EGFR TKI). |
| `subclass` | `exposure`, `samples`, `model` | Which drugs the class started with (agents begun within 30 days of `class_start`). Breast: `AI`, `tamoxifen`, `fulvestrant`. NSCLC: `osimertinib`, `1st/2nd-gen TKI`. Prostate: `GnRH`, `antiandrogen`. CRC: `FOLFOX` (with 5-FU), `CAPOX` (with capecitabine), `other` (oxaliplatin without either at start). |
| `companion_cdk46` | `exposure`, `samples`, `model` (breast) | Breast: a CDK4/6 inhibitor (palbociclib, abemaciclib, ribociclib) was started between `class_start` − 30 and the end of the first episode. Defined identically for pre and post samples. |
| `ovarian_suppression` | `exposure`, `samples`, `model` (breast) | Breast: leuprolide, goserelin or triptorelin was started in the same window. |
| `fluoropyrimidine_backbone` | `exposure`, `samples`, `model` (CRC) | CRC: 5-FU, capecitabine or leucovorin was started in the same window. |
| `cdk46_any_first_day`, `pi3k_mtor_first_day`, `ar_inhibitor_exposure_first_day`, `irinotecan_first_day`, `anti_egfr_mab_first_day`, `anti_vegf_first_day` | `exposure` | First day of each later-line drug group after `dx_date` − 30, NaN if never given. Groups: any CDK4/6 inhibitor; alpelisib/everolimus (breast); abiraterone/enzalutamide/apalutamide/darolutamide (prostate); irinotecan, cetuximab/panitumumab/necitumumab, bevacizumab/ramucirumab/regorafenib (CRC). |
| `<group>_before_collection` (`cdk46_any_…`, `pi3k_mtor_…`, `ar_inhibitor_exposure_…`, `irinotecan_…`, `anti_egfr_mab_…`, `anti_vegf_…`) | `samples`, `model` (cohort-specific) | True/1 if the matching `<group>_first_day` is on or before `collect_day`. Nearly always 0 for pre-treatment samples, so **Target B covariate only**. |
| `met_date` | `exposure`, `samples` | Day of the first distant metastasis: the earliest of (a) `dx_date` if `stage_at_dx` = Stage 4, (b) the first NLP radiology mention of a distant organ for this cancer (`config/pipeline.yaml`), (c) collection of a non-lymph-node metastasis biopsy. NaN if none recorded. |
| `met_source` | `exposure` | Which source set `met_date`: `stage4_dx`, `nlp_site`, `met_biopsy`; missing if no metastasis. |
| `adjuvant_start` | `exposure`, `samples`, `model` | True if `class_start` precedes `met_date`, or no metastasis is ever recorded (class started in the non-metastatic setting). Over-calls in NSCLC, where lung-only recurrence is not counted as distant. |
| `followup_days` | `exposure`, `samples`, `model`, `cohort_v1` | Last recorded MSK-CHORD event of any kind minus `class_start`. *In `cohort_v1`:* days from sequencing to last contact or death, from MSK-MET ages. |
| `t_days` | `samples`, `model` | `collect_day` − `class_start`: days from first dose to tissue collection. ≤ 0 = pre-treatment. |
| `cum_days_on_class` | `samples` | Days actually on the class between `class_start` and collection: the sum of episode days up to `collect_day`, counted inclusively. Set to 0 for every pre-treatment sample. |
| `on_class_at_collection` | `samples`, `model` | True if `collect_day` falls inside an episode or within 30 days after one ends. Always False/0 for pre-treatment samples (**Target B only**). |
| `days_since_class_stop` | `samples`, `model` | Post-treatment samples off the drug: `collect_day` − end of the last episode started before collection. NaN for pre-treatment samples and for samples on class (**Target B only**). |
| `days_since_class_stop_missing` | `model` | 1 where `days_since_class_stop` is NaN. |
| `classes_between` | `samples` | List of other systemic classes started after `class_start` + 30 and on or before `collect_day` (intervening lines). Companion drugs count only if started outside every class episode. Empty for pre-treatment samples. |
| `n_lines_between` | `samples`, `model` | Length of `classes_between` (**Target B only**). |

### 3.3 Labels and analysis flags

| Column | Files | Definition |
|---|---|---|
| `label_variant` | `samples`, `folds` | Label definition used: `A` (main) or `A-on`. |
| `label_A` | `samples`, `model` | **Target A** (classification). 0 = collected on or before the first dose (`t_days` ≤ 0); 1 = collected more than 90 days after it. In `samples_A-on.parquet`, label 1 additionally requires `on_class_at_collection`. |
| `label_A_on` | `model` | A-on variant within the main table: 1 = post-treatment **and** on class at collection; 0 for every other row, including post samples that were off the drug. For an A-on analysis use only rows with `label_A` = 0 or `label_A_on` = 1. |
| `label_B` | `samples`, `model` | **Target B** (dose). Equal to `cum_days_on_class`: 0 for pre-treatment, cumulative days on the class for post-treatment. |
| `label_B_log1p` | `samples`, `model` | log(1 + `label_B`), the modelling scale for Target B. |
| `in_impact` | `samples` | Sample has MSK-IMPACT 50K mutation and CNA data. True for every row, because the cohort was filtered on it. |
| `strict_patient` | `samples` | `strict_first_line` and no prior medications to MSK. |
| `tier_strict` | `samples`, `model` | **Main tier:** `strict_patient` and FACETS QC pass. |
| `tier_strict_noint` | `samples`, `model` | `tier_strict` and `n_lines_between` = 0 (no intervening therapy). Sensitivity S2. |
| `tier_relaxed` | `samples`, `model` | True for every row: all samples left after exclusions. Sensitivity S1. |
| `hypermutated` | `samples`, `model` | MSI-high (`MSI_TYPE` = Instable) or `TMB_SCORE` > 20 mutations/Mb. Excluded from the main analysis. |
| `main_analysis` | `model` | **Rows for the main analysis:** `tier_strict` and not `hypermutated`. |
| `s3_survivorship_keep` | `model` | Survivorship sensitivity S3: True for post samples, and for pre samples whose `followup_days` ≥ the cohort's median post-treatment `t_days`. |
| `egfr_sensitizing_any` | `samples` | Sample carries an EGFR-sensitizing mutation: exon 19 deletion (in-frame deletion starting at codons 729–761), L858R, G719X, L861Q, S768I or E709X. NSCLC was restricted to True. |
| `fold_r00` … `fold_r19` | `model` | Cross-validation fold (0–4) for repeats 0–19 of 5-fold patient-grouped CV, stratified by label × subclass (label × cohort for pooled). Assigned to `main_analysis` rows only; NaN elsewhere. |

### 3.4 Clinical and sample covariates

| Column | Files | Definition |
|---|---|---|
| `CANCER_TYPE` | `samples`, `het_v1`, `cohort_v1` | Sample cancer type (MSK-CHORD in `samples`, MSK-IMPACT in `het_v1`, MSK-MET in `cohort_v1`). |
| `SAMPLE_TYPE` | `samples`, `het_v1`, `cohort_v1` | Raw sample type: Primary, Metastasis, Local Recurrence, Unknown. |
| `sample_type` | `model` | `SAMPLE_TYPE` with missing → `Unknown`. Nuisance covariate. |
| `PRIMARY_SITE` | `samples`, `cohort_v1` | Raw free-text organ sub-division of the primary tumor (e.g. Sigmoid Colon). |
| `METASTATIC_SITE` | `samples`, `cohort_v1` | Raw free-text site of the sequenced metastasis; `Not Applicable` for primaries. |
| `biopsy_organ` | `model` | Organ group of the biopsied tissue: `METASTATIC_SITE` if given, otherwise `PRIMARY_SITE`, mapped by `config/site_map.csv` (Breast, Lung, Prostate, Colorectal, Liver, Lymph node, Bone, Pleura, Brain/CNS, Peritoneum/abdomen, Soft tissue/skin, …). Groups with < 30 samples in the cohort → `Other`. |
| `biopsy_organ_coarse` | `model` | Five levels: `primary_organ` (biopsy of the cohort's own primary organ), `liver`, `lung`, `bone`, `other`. |
| `metastatic_at_collection` | `samples`, `model` | A distant metastasis was recorded on or before `collect_day` (`met_date` ≤ `collect_day`). |
| `GENE_PANEL` | `samples`, `het_v1`, `cohort_v1` | Raw sequencing assay (IMPACT341/410/468/505). |
| `panel_version` | `samples`, `model` | MSK-IMPACT panel used for the sample's mutation calls (from `data_gene_panel_matrix.txt`). Sequencing-era covariate. |
| `log_patient_id` | `model` | Natural log of the numeric patient ID. Continuous sequencing-era covariate, since IDs were assigned over time. |
| `archived_tissue` | `samples`, `model` | Tissue collected more than 365 days before sequencing (`collect_day` < `seq_day` − 365). |
| `sex` | `samples` | MSK-CHORD `GENDER` (MSK-IMPACT `SEX` where unknown): Female / Male. |
| `sex_male` | `model` (nsclc, crc) | 1 = male, 0 = female. Omitted for the single-sex cohorts. |
| `CURRENT_AGE_DEID` | `samples` | Patient's current age from MSK-CHORD (capped at 89), i.e. age at last record. |
| `age_at_collection` | `model` | Approximate age at tissue collection: `CURRENT_AGE_DEID` − (last recorded event − `collect_day`)/365.25. |
| `HR` | `samples` | Hormone-receptor status (MSK-CHORD patient): Yes / No; breast only. The breast cohort is all Yes. |
| `HER2` | `samples`, `model` (breast) | HER2 status (MSK-CHORD patient): Yes / No; `Unknown` in `model` when missing. |
| `PRIOR_MED_TO_MSK` | `exposure`, `samples` | Raw MSK-CHORD field: `Prior medications to MSK`, `No prior medications`, `Unknown`. |
| `prior_med_to_msk` | `exposure`, `samples` | True if `PRIOR_MED_TO_MSK` = "Prior medications to MSK". Such patients are never pre-treatment controls and are excluded from the strict tier. |
| `drug_class` | `model` (pooled) | Cohort of the row, used as the drug-class covariate in the pooled model. |

### 3.5 Sample-level genomic features

| Column | Files | Definition |
|---|---|---|
| `TMB_SCORE` | `samples`, `genomic`, `het_v1`, `cohort_v1` | MSK-IMPACT tumor mutational burden, mutations per megabase (already panel-normalized). |
| `tmb_log1p` | `genomic`, `model` | log(1 + `TMB_SCORE`). |
| `MSI_SCORE` | `samples`, `genomic`, `model`, `het_v1`, `cohort_v1` | MSIsensor microsatellite-instability score. Raw in `samples`/`het_v1`/`cohort_v1`; in `genomic`/`model`, set to NaN unless `MSI_TYPE` is Stable or Instable. |
| `MSI_TYPE` | `samples`, `genomic`, `het_v1`, `cohort_v1` | MSI call: Stable, Instable, Indeterminate, Do not report. In `genomic`, only Stable/Instable are kept (others → NaN). |
| `MSI_SCORE_missing` | `model` | 1 where `MSI_SCORE` is NaN. |
| `FACETS_QC` | `samples`, `het_v1`, `cohort_v1` | FACETS copy-number QC: Pass / Fail. |
| `facets_pass` | `samples`, `model` | `FACETS_QC` = Pass. Required for clonality, CCF and LOH. |
| `FACETS_PURITY` | `samples`, `genomic`, `model`, `het_v1`, `cohort_v1` | FACETS tumor purity (fraction of cells that are tumor). Raw in `samples`/`het_v1`/`cohort_v1`; in `genomic`/`model`, NaN when FACETS QC failed. |
| `FACETS_PURITY_missing` | `model` | 1 where `FACETS_PURITY` is NaN. |
| `low_purity` | `samples`, `model` | `FACETS_PURITY` < 0.2. Flagged, not excluded. |
| `FACETS_PLOIDY` | `samples`, `genomic`, `model`, `het_v1`, `cohort_v1` | FACETS genome ploidy estimate. Same QC rule as `FACETS_PURITY`. |
| `FACETS_WGD` | `samples`, `genomic`, `model`, `het_v1`, `cohort_v1` | FACETS whole-genome doubling. Boolean in `samples`/`het_v1`/`cohort_v1`; 1/0 (NaN if QC failed) in `genomic`/`model`. |
| `FACETS_missing` | `model` | 1 where FACETS ploidy is missing (QC failed). |
| `FGA` | `genomic`, `model`, `cohort_v1` | Fraction of genome altered: share of autosomal length in copy-number segments with \|log2 ratio\| > 0.2 (from `data_cna_hg19.seg`). Correlates r = 0.996 with MSK-MET's value. *In `cohort_v1`:* MSK-MET's own FGA field. |
| `FGA_missing` | `model` | 1 where `FGA` is NaN (no segmentation). |
| `n_mutations_total` | `genomic` | All mutation records for the sample in the MSK-IMPACT MAF, any classification. |
| `n_mutations_oncogenic` | `genomic` | Mutations called oncogenic by the rule-based fallback (Section 4.4). Not an alteration feature. |
| `n_mutations_oncogenic_log1p` | `genomic`, `model` | log(1 + `n_mutations_oncogenic`). Nuisance covariate. |

---

## 4. Alteration columns

### 4.1 Gene-level matrix (pattern columns)

These columns follow a naming pattern, with `<GENE>` an HGNC symbol (legacy aliases mapped
via `config/gene_aliases.csv`). Each pattern is defined once here.

| Column pattern | Files | Definition |
|---|---|---|
| `<GENE>_mut` | `alt_matrix`, `model` | 1 = at least one oncogenic mutation in the gene (rule-based calls; nonsynonymous variants plus TERT promoter); 0 = none; NA = gene not on the sample's mutation panel. |
| `<GENE>_amp` | `alt_matrix`, `model` | 1 = high-level amplification (discrete CNA = 2) in a gene whose amplifications outnumber deep deletions at least 2:1 across MSK-IMPACT 50K. NA = gene off the CNA panel. |
| `<GENE>_del` | `alt_matrix`, `model` | 1 = homozygous deep deletion (CNA = −2) in a gene whose deep deletions outnumber amplifications at least 2:1. NA as above. |
| `<GENE>_sv` | `alt_matrix`, `model` | 1 = structural variant/fusion with this gene as either partner, where a partner is a fusion driver or oncogene and the event is not antisense. NA = gene off the SV panel. |
| `<GENE>_loh` | `alt_matrix`, `model` | Tumor-suppressor genes only. 1 = loss of heterozygosity (FACETS minor copy number 0, total ≥ 1). NA if FACETS QC failed or FACETS made no call for the gene (often X-linked genes in males). LOH features with any NA among main-analysis rows are left out of the main analysis. |
| `<GENE>_mut_clonal` | `alt_matrix`, `model` | 1 = the gene's best-called oncogenic mutation is clonal: FACETS-QC-passed and `ASCN.CLONAL` = CLONAL or CCF lower bound ≥ 0.8. |
| `<GENE>_mut_subclonal` | `alt_matrix`, `model` | 1 = no clonal call, but at least one QC-passed mutation is SUBCLONAL. |
| `<GENE>_mut_clonality_unknown` | `alt_matrix`, `model` | 1 = gene mutated but clonality indeterminate, missing or FACETS QC failed. The three clonality columns are mutually exclusive and sum to `<GENE>_mut`. |
| `<GENE>_mut_max_ccf` | `alt_matrix`, `model` | Highest cancer-cell fraction among the gene's oncogenic, QC-passed mutations; NA if not mutated. For post-treatment-only and gradient-boosting analyses only. |
| `<GENE>_any` | `alt_gene` | Collapsed gene-level feature: max of `<GENE>_mut`, `_amp`, `_del`, `_sv`. LOH and variant-level features excluded. Alternative matrix (sensitivity S9), not modelled by default. |

`model` tables contain only the pattern columns altered in at least one sample of that cohort.

### 4.2 Variant-level target features

Always kept, regardless of prevalence. The `model` table for each cohort carries its own
targets; `model_table_pooled` carries all of them.

| Column | Files | Definition |
|---|---|---|
| `EGFR_T790M` | `alt_matrix`, `model` (nsclc) | EGFR p.T790M (first/second-generation TKI resistance). |
| `EGFR_C797X` | `alt_matrix`, `model` (nsclc) | Any EGFR missense at C797 (osimertinib resistance). |
| `EGFR_sensitizing` | `alt_matrix`, `model` (nsclc) | EGFR exon 19 deletion or L858R. |
| `MET_amp` | `alt_matrix`, `model` | MET discrete CNA = 2 (TKI bypass); NSCLC target. |
| `ERBB2_amp` | `alt_matrix`, `model` | ERBB2 discrete CNA = 2 (bypass); NSCLC target. |
| `ESR1_LBD` | `alt_matrix`, `model` (breast) | ESR1 missense at an E380, V422, S463, L536, Y537 or D538 hotspot (endocrine resistance). |
| `ESR1_fusion` | `alt_matrix`, `model` (breast) | Any structural variant involving ESR1. |
| `AR_LBD` | `alt_matrix`, `model` (prostate) | AR missense at L702, W742, H875, F877 or T878 (ADT / AR-inhibitor resistance). |
| `AR_amp` | `alt_matrix`, `model` | AR discrete CNA = 2; prostate target. |
| `AR_sv` | `alt_matrix`, `model` | Any structural variant involving AR (the AR enhancer itself is not captured by the panel); prostate target. |
| `KRAS_any` | `alt_matrix`, `model` (crc) | Any oncogenic KRAS mutation (negative control). |
| `NRAS_any` | `alt_matrix`, `model` (crc) | Any oncogenic NRAS mutation. |
| `BRAF_V600E` | `alt_matrix`, `model` (crc) | BRAF p.V600E. |

`MET_amp`, `ERBB2_amp`, `AR_amp` and `AR_sv` match the pattern names in 4.1, but use the
target definition above (no amp/deletion-dominance or driver-partner filter).

### 4.3 Alteration events (`alt_long`)

| Column | Files | Definition |
|---|---|---|
| `gene` | `alt_long`, `onco_rules`, `hotspots`, `alt_features`, `feat_dict` | HGNC gene symbol (legacy aliases mapped). In `alt_features`/`feat_dict`, the gene a feature refers to. |
| `alt_type` | `alt_long` | Event type: `mut` (mutation), `amp`, `del` (homozygous deletion), `sv` (structural variant, one row per partner gene), `loh` (TSG loss of heterozygosity). |
| `variant` | `alt_long` | Event detail: protein change (`HGVSp_Short`, else the variant class) for mutations; `AMP`/`HOMDEL` for CNAs; `partner:<GENE>` or `partner:intergenic` for SVs; `LOH tcn=<total copy number>` for LOH. |
| `Variant_Classification` | `alt_long` | MAF variant class (Missense_Mutation, Nonsense_Mutation, Frame_Shift_Del, …); mutations only. |
| `oncogenic` | `alt_long` | Event passes the rule-based oncogenicity filter. Non-oncogenic mutations, CNAs and SVs are kept with False for traceability. |
| `onco_rule` | `alt_long` | Rule that made the event oncogenic: `hotspot_missense`, `hotspot_inframe`, `truncating_in_tsg`, `tert_promoter`, `met_ex14`, `amp_dominant_gene`, `homdel_dominant_gene`, `fusion_driver_partner`, `loh_tsg`; empty if not oncogenic. |
| `gene_role` | `alt_long` | The gene's `role` from `onco_rules` (mutations only). |
| `ccf` | `alt_long` | FACETS cancer-cell fraction of the mutation (`ASCN.CCF_M_COPIES`); NaN unless the mutation passed FACETS QC. |
| `clonality` | `alt_long` | Mutation clonality: `clonal`, `subclonal` or `unknown`, with the rules in 4.1. |
| `mut_qc_pass` | `alt_long` | Mutation's `ASCN.FACETS_QC_PASSED` is True. |
| `oncogenicity_source` | `alt_long`, `feat_dict` | Annotation rule set used: `rules_v1` (rule-based fallback; OncoKB not used). |
| `panel` | `alt_long` | The sample's panel for this event's data type (mutations, CNA or SV). |
| `on_panel` | `alt_long` | Gene is on that panel. Off-panel events are not used in the matrix. |

### 4.4 Oncogenicity rules

These are derived from all MSK-IMPACT 50K samples, with hypermutated samples excluded
from the mutation counts. No labels are used.

| Column | Files | Definition |
|---|---|---|
| `n` | `onco_rules` | Number of nonsynonymous mutations in the gene. |
| `trunc` | `onco_rules` | Fraction of those that are truncating (nonsense, frameshift, splice site, start/stop loss). |
| `hot` | `onco_rules` | Fraction at hotspot residues (see `hotspots`). |
| `n_amp`, `n_homdel` | `onco_rules` | Number of amplification (CNA = 2) and deep-deletion (CNA = −2) calls in the gene. |
| `role` | `onco_rules` | `TSG` (≥ 20 mutations and `trunc` ≥ 0.20, the 20/20 rule); `oncogene` (≥ 20 mutations, `trunc` < 0.20 and `hot` ≥ 0.20, or override: MET); `oncogene_cna` / `TSG_cna` (too few mutations; amplifications or deep deletions dominate ≥ 2:1 with ≥ 20 calls); `unclassified`. |
| `amp_driver` | `onco_rules` | `n_amp` ≥ 2 × `n_homdel`: amplifications of the gene count as oncogenic. |
| `del_driver` | `onco_rules` | `n_homdel` ≥ 2 × `n_amp`: deep deletions count as oncogenic. |
| `codon` | `hotspots` | Hotspot residue (codon number). |
| `kind` | `hotspots` | `missense`: residue mutated in ≥ 10 samples and ≥ 5× the gene's per-residue background. `inframe`: in-frame indels starting at that codon in ≥ 5 samples. |

---

## 5. Summary and bookkeeping tables

| Column | Files | Definition |
|---|---|---|
| `feature` | `alt_features`, `feat_dict`, `controls` | Column name of a model feature or covariate. In `controls`, the alteration feature used at the tier stages. |
| `data_type` | `alt_features`, `feat_dict` | Panel-matrix data type used for masking: `mutations`, `gistic` (CNA and LOH), `structural_variants`. |
| `variant_level` | `alt_features` | Feature is a variant-level target (Section 4.2). |
| `on_IMPACT341` | `alt_features`, `feat_dict` | The feature's gene is on IMPACT341, so it is measured in every sample. Main-analysis features must be. |
| `block` | `feat_dict` | Feature group: `alteration`, `variant_target`, `clonality`, `ccf (post-only / GB only)`, `genomic`, `covariate_A` (usable for Targets A and B), `covariate_B (Target B only)` (defined only after treatment starts). |
| `type` | `feat_dict` | Alteration type (`mut`, `amp`, `del`, `sv`, `loh`) for alteration features; `num`, `cat` or `bin` for covariates. |
| `source` | `feat_dict` | Raw MSK-IMPACT file the feature comes from. |
| `transform` | `feat_dict` | Encoding / in-fold transform: binary with NA off-panel; max CCF; 0/1; median-impute + standardize within each training fold; one-hot within fold with NaN → Unknown. |
| `cohorts` | `feat_dict` | Covariates only: cohorts whose model table includes the covariate. |
| `<cohort>_pct_pre`, `<cohort>_pct_post` | `feat_dict` | % of main-analysis pre / post samples with the feature = 1, among those measured. |
| `<cohort>_coverage` | `feat_dict` | Share of the cohort's samples (all tiers) where the feature is measured (not NA). |
| `<cohort>_n_altered_main` | `feat_dict` | Number of main-analysis samples with the feature = 1. |
| `<cohort>_kept_firth` | `feat_dict` | On the cohort's per-gene Firth list: altered in ≥ 10 samples and ≥ 2% of main-analysis samples, or a variant-level target. |
| `<cohort>_main_analysis_feature` | `feat_dict` | Feature is in the main-analysis matrix: gene on IMPACT341, no missing cells among main-analysis rows, and not a `_max_ccf` column. |
| `repeat` | `folds` | CV repeat: `fold_r00` … `fold_r19`. |
| `fold` | `folds` | Fold (0–4) the sample is held out in for that repeat. |
| `tier` | `folds` | Rows the folds were assigned over: `strict` (= `main_analysis`), `strict_noint` (minus hypermutated), `relaxed` (all rows). Variant A-on uses `strict` rows with `label_A` = 0 or `label_A_on` = 1. |
| `step` | `attrition` | Exclusion step code: `0` all dated MSK-CHORD samples; `1` cancer type; `2` received the class; `3` in MSK-IMPACT 50K; `4` cohort rule (breast HR+, NSCLC EGFR-sensitizing); `4L` labelled; `5` early window excluded; `5b` pre-treatment samples with prior meds dropped; `5c` one sample per patient; `6` strict tier; `6b` strict, no intervening therapy; `7` relaxed with FACETS pass. |
| `description` | `attrition` | Text of the step; rows prefixed `[A-on]` are the A-on label variant. |
| `samples`, `patients` | `attrition` | Samples / patients remaining after the step. In `plan_repro`, `samples` is the loose cohort size. |
| `label_0`, `label_1` | `attrition` | Pre- / post-treatment samples remaining (from step `4L` on). |
| `t_le_0`, `t_0_90`, `t_gt_90` | `plan_repro` | Loose-cohort samples with `t_days` ≤ 0, in (0, 90], and > 90. Recomputes plan Section 2.1. |
| `t_gt_90_on_class` | `plan_repro` | Of `t_gt_90`, those on the class at collection. |
| `strict_t_le_0`, `strict_t_gt_90` | `plan_repro` | Same counts restricted to strict-tier criteria. |
| `strict_t_gt_90_no_between` | `plan_repro` | Strict post-treatment samples with no intervening therapy. |
| `median_t_gt_90` | `plan_repro` | Median `t_days` of the post-treatment samples. |
| `control` | `controls` | Positive or negative control alteration, e.g. "EGFR T790M", "ESR1_LBD (AI subclass)". |
| `stage` | `controls` | Pipeline point at which the control was measured: `raw_plan_denominator` (plan's denominator, including unprofiled samples), `raw` (all raw calls, profiled only), `onco_filter`, `panel_mask`, then tiers `relaxed`, `strict`, `strict_noint`, `S3_survivorship_strict`, `A-on_strict`. |
| `n_pre`, `n_post` | `controls`, `signatures` | Pre- / post-treatment samples in the denominator. |
| `pct_pre`, `pct_post` | `controls` | % of pre / post samples with the control alteration. |
| `plan_pct_pre`, `plan_pct_post` | `controls` | Value reported in the plan's Section 2.2, for comparison. |
| `expected` | `controls` | Expected direction: `up` (resistance alteration) or `flat` (KRAS negative control). |
| `signature` | `signatures` | COSMIC v2 signature check: `Signature 17 (5-FU)` in CRC, `APOBEC (2+13)` in breast. Samples with ≥ 10 mutations only. |
| `mean_pre`, `mean_post` | `signatures` | Mean signature contribution (0–1) in pre / post samples. |
| `pct_dominant_pre`, `pct_dominant_post` | `signatures` | % of samples in which the signature contributes > 0.2. |

---

## 6. Legacy v1 files

`cohort.parquet`, `het_features.parquet` and `treatment.parquet` come from the earlier v1
metastasis design (`build_features.py`). They are kept for `data_exploration.ipynb` and
are **not inputs to the v3 pipeline**. Columns already defined above are not repeated.

### `cohort_v1`: MSK-MET ∩ MSK-IMPACT samples with FACETS QC pass

| Column | Definition |
|---|---|
| `ORGAN_SYSTEM`, `SUBTYPE`, `SUBTYPE_ABBREVIATION`, `ONCOTREE_CODE`, `CANCER_TYPE_DETAILED` | MSK-MET tumor classification: organ system, subtype (name and abbreviation), OncoTree code, detailed cancer type. |
| `TUMOR_PURITY` | Pathologist-estimated tumor purity (%). |
| `SAMPLE_COVERAGE` | Sequencing coverage. |
| `IS_DIST_MET_MAPPED` | MSK-MET flag: patient has a mapped distant metastasis. |
| `MET_COUNT`, `MET_SITE_COUNT` | Number of metastasis records and of distinct metastatic organ sites (MSK-MET). |
| `DMETS_DX_<ORGAN>` (21 columns, e.g. `DMETS_DX_LIVER`) | MSK-MET curated flag, Yes/No: distant metastasis to that organ diagnosed at any point. |
| `TMB_NONSYNONYMOUS` | MSK-MET nonsynonymous TMB. |
| `DISEASE_STATUS` | MSK-IMPACT disease status of the sample (Primary / Metastasis). Also in `het_v1`. |
| `SOMATIC_STATUS` | Whether a matched normal was used (Matched / Unmatched). |
| `FACETS_SUITE_VERSION` | FACETS software version. |
| `PURITY_ESTIMATE_FROM_MUTATIONS` | Purity was estimated from mutation allele frequencies rather than FACETS. |
| `HLA_<A/B/C><1/2>_LOH` (6 columns) | Somatic loss of heterozygosity of each HLA class I allele. |
| `SEX`, `RACE` | Patient sex and self-reported race (MSK-MET). |
| `OS_MONTHS`, `OS_STATUS` | Overall survival in months from sequencing, and vital status (0:LIVING / 1:DECEASED). |
| `AGE_AT_EVIDENCE_OF_METS`, `AGE_AT_SEQUENCING`, `AGE_AT_DEATH`, `AGE_AT_SURGERY`, `AGE_AT_LAST_CONTACT` | Patient age (years) at first metastasis diagnosis, sequencing, death, surgery and last contact. |
| `T_<ORGAN>` (22 organs + `T_ANY_MET`) | Days from sequencing to the first metastasis in that organ more than 30 days after sequencing; censored at `followup_days` when no event. |
| `E_<ORGAN>` (22 organs + `E_ANY_MET`) | Event indicator for `T_<ORGAN>`: 1 = metastasis observed, 0 = censored. |

### `het_v1`: intratumor heterogeneity features (all MSK-IMPACT samples)

| Column | Definition |
|---|---|
| `n_mutations` | All mutation records in the sample. |
| `n_ccf_called` | Mutations with a FACETS CCF value. |
| `n_clonal`, `n_subclonal`, `n_indeterminate` | Mutations with `ASCN.CLONAL` = CLONAL, SUBCLONAL, INDETERMINATE. |
| `n_determinate` | `n_clonal` + `n_subclonal`. |
| `ccf_mean`, `ccf_median`, `ccf_std`, `ccf_min` | Summary statistics of mutation CCF in the sample. |
| `math_score` | MATH-style heterogeneity: 100 × median absolute deviation of CCF / median CCF (NaN with < 2 CCF values). |
| `ascn_purity` | FACETS purity from the mutation file's ASCN block. |
| `subclonal_fraction` | `n_subclonal` / `n_determinate`; NaN when there are no determinate calls. |
| `subclonal_fraction_resolution` | 1 / `n_determinate`: smallest step `subclonal_fraction` can take. |
| `arms_gain`, `arms_loss`, `arms_altered`, `arms_called` | Number of chromosome arms gained, lost, altered (gain + loss) and called (`data_armlevel_cna.txt`). |
| `arm_altered_fraction` | `arms_altered` / `arms_called`. Dropped from v3 because it duplicates FGA. |

### `treatment_v1`: per-patient MSK-CHORD treatment counts around first sequencing

| Column | Definition |
|---|---|
| `n_treatments` | Treatment-timeline rows for the patient. |
| `n_agents` | Distinct agents. |
| `n_classes` | Distinct MSK-CHORD treatment subtypes (Chemo, Hormone, Targeted, …). |
| `first_tx_day`, `last_tx_day` | Earliest and latest treatment start day. |
| `n_pre_treatments` | Treatment rows starting more than 30 days before first sequencing. |
| `days_last_pre_tx_to_seq` | Days from the last such pre-sequencing start to sequencing. |
| `any_pre_treatment` | 1 if `n_pre_treatments` > 0. |
| `n_<window>_<subtype>` (e.g. `n_pre_chemo`, `n_peri_hormone`, `n_post_targeted`) | Treatment rows of that MSK-CHORD subtype starting in that window relative to first sequencing: `pre` (> 30 days before), `peri` (within ±30 days), `post` (> 30 days after). Subtypes: chemo, hormone, targeted, biologic, immuno, investigational, bone_treatment, other. |
