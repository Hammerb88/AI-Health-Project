# preprocess

_Written 2026-10-13 00:20_

Stages 7-9 (preprocess.py). 7,293 samples in; model tables written per cohort and pooled
(breast: 2,607 x 1,617, nsclc: 753 x 1,002, prostate: 1,515 x 1,420, crc: 2,418 x 1,738, pooled: 7,293 x 2,150). Main analysis = strict tier
minus hypermutated samples. Folds: StratifiedGroupKFold 5 x 20 repeats, grouped by
patient, stratified by label x subclass (label x cohort for pooled), for every tier plus A-on.
Firth gene lists (full-cohort prevalence >= 10 and >= 2%, main analysis,
variant-level targets always kept): {'breast': 286, 'nsclc': 268, 'prostate': 262, 'crc': 279, 'pooled': 302}. Exposure variables defined only after class start
(on_class_at_collection, days_since_class_stop, n_lines_between, *_before_collection) are Target-B-only
covariates: they are zero for >= 99.8% of label-0 samples by construction (table below).

## Tier sizes

| cohort | relaxed | strict | strict_noint | main | hypermutated |
|---|---:|---:|---:|---:|---:|
| breast | 2,607 | 1,040 | 944 | 1,028 | 35 |
| crc | 2,418 | 1,991 | 1,853 | 1,852 | 162 |
| nsclc | 753 | 432 | 406 | 432 | 3 |
| prostate | 1,515 | 1,026 | 852 | 996 | 35 |

## Main analysis labels

| cohort | 0 | 1 |
|---|---:|---:|
| breast | 800 | 228 |
| crc | 1,479 | 373 |
| nsclc | 323 | 109 |
| prostate | 704 | 292 |

## Top single-covariate AUCs (Target A, main analysis)

| cohort | covariate | auc |
|---|---|---:|
| prostate | biopsy_organ_coarse=primary_organ | 0.843 |
| prostate | biopsy_organ=Prostate | 0.843 |
| prostate | sample_type=Primary | 0.841 |
| prostate | sample_type=Metastasis | 0.838 |
| breast | sample_type=Primary | 0.79 |
| breast | sample_type=Metastasis | 0.783 |
| breast | biopsy_organ_coarse=primary_organ | 0.783 |
| breast | biopsy_organ=Breast | 0.783 |
| breast | metastatic_at_collection | 0.765 |
| crc | sample_type=Primary | 0.745 |
| crc | sample_type=Metastasis | 0.741 |
| crc | biopsy_organ=Colorectal | 0.733 |

## Target-B-only exposure variables: % non-zero by label

| covariate | label0_nonzero_pct | label1_nonzero_pct |
|---|---:|---:|
| on_class_at_collection | 0 | 52.5 |
| days_since_class_stop | 0 | 47.5 |
| n_lines_between | 0 | 38.9 |
| cdk46_any_before_collection | 0.0946 | 8.02 |
| pi3k_mtor_before_collection | 0 | 3.78 |
| ar_inhibitor_exposure_before_collection | 0.17 | 14.1 |
| irinotecan_before_collection | 0.189 | 5.23 |
| anti_egfr_mab_before_collection | 0.0946 | 1.34 |
| anti_vegf_before_collection | 0.0568 | 3.34 |

## Controls by tier (T4)

| cohort | control | stage | n_pre | n_post | pct_pre | pct_post |
|---|---|---|---:|---:|---:|---:|
| breast | ESR1_LBD (AI subclass) | strict | 588 | 129 | 1.53 | 27.1 |
| breast | ESR1_mut | strict | 800 | 228 | 1.62 | 23.7 |
| nsclc | EGFR_T790M | strict | 323 | 109 | 1.86 | 34.9 |
| nsclc | MET_amp | strict | 323 | 109 | 1.24 | 11.9 |
| nsclc | EGFR_C797X | strict | 323 | 109 | 0 | 1.83 |
| prostate | AR_amp | strict | 704 | 292 | 0.71 | 31.8 |
| prostate | AR_LBD | strict | 704 | 292 | 0.142 | 8.22 |
| prostate | AR_mut | strict | 704 | 292 | 0.142 | 7.53 |
| crc | KRAS_any | strict | 1,479 | 373 | 41.8 | 39.7 |
| breast | ESR1_LBD (AI subclass) | strict_noint | 588 | 74 | 1.53 | 24.3 |
| breast | ESR1_mut | strict_noint | 800 | 134 | 1.62 | 15.7 |
| nsclc | EGFR_T790M | strict_noint | 323 | 83 | 1.86 | 34.9 |
| nsclc | MET_amp | strict_noint | 323 | 83 | 1.24 | 13.3 |
| nsclc | EGFR_C797X | strict_noint | 323 | 83 | 0 | 1.2 |
| prostate | AR_amp | strict_noint | 704 | 121 | 0.71 | 16.5 |
| prostate | AR_LBD | strict_noint | 704 | 121 | 0.142 | 6.61 |
| prostate | AR_mut | strict_noint | 704 | 121 | 0.142 | 4.13 |
| crc | KRAS_any | strict_noint | 1,479 | 241 | 41.8 | 43.2 |
| breast | ESR1_LBD (AI subclass) | relaxed | 1,251 | 462 | 1.28 | 18.2 |
| breast | ESR1_mut | relaxed | 1,789 | 818 | 1.4 | 16.3 |
| nsclc | EGFR_T790M | relaxed | 561 | 192 | 1.78 | 37.5 |
| nsclc | MET_amp | relaxed | 561 | 192 | 0.891 | 9.9 |
| nsclc | EGFR_C797X | relaxed | 561 | 192 | 0 | 2.08 |
| prostate | AR_amp | relaxed | 1,014 | 501 | 0.789 | 34.9 |
| prostate | AR_LBD | relaxed | 1,014 | 501 | 0.296 | 8.58 |
| prostate | AR_mut | relaxed | 1,014 | 501 | 0.394 | 8.18 |
| crc | KRAS_any | relaxed | 1,921 | 497 | 40.9 | 41.9 |
| breast | ESR1_LBD (AI subclass) | S3_survivorship_strict | 425 | 129 | 1.65 | 27.1 |
| breast | ESR1_mut | S3_survivorship_strict | 585 | 228 | 1.54 | 23.7 |
| nsclc | EGFR_T790M | S3_survivorship_strict | 240 | 109 | 1.25 | 34.9 |
| nsclc | MET_amp | S3_survivorship_strict | 240 | 109 | 1.25 | 11.9 |
| nsclc | EGFR_C797X | S3_survivorship_strict | 240 | 109 | 0 | 1.83 |
| prostate | AR_amp | S3_survivorship_strict | 493 | 292 | 0.406 | 31.8 |
| prostate | AR_LBD | S3_survivorship_strict | 493 | 292 | 0.203 | 8.22 |
| prostate | AR_mut | S3_survivorship_strict | 493 | 292 | 0.203 | 7.53 |
| crc | KRAS_any | S3_survivorship_strict | 1,249 | 373 | 40.4 | 39.7 |
| breast | ESR1_LBD (AI subclass) | A-on_strict | 588 | 76 | 1.53 | 27.6 |
| breast | ESR1_mut | A-on_strict | 800 | 132 | 1.62 | 25 |
| nsclc | EGFR_T790M | A-on_strict | 323 | 83 | 1.86 | 28.9 |
| nsclc | MET_amp | A-on_strict | 323 | 83 | 1.24 | 15.7 |
| nsclc | EGFR_C797X | A-on_strict | 323 | 83 | 0 | 1.2 |
| prostate | AR_amp | A-on_strict | 704 | 226 | 0.71 | 36.7 |
| prostate | AR_LBD | A-on_strict | 704 | 226 | 0.142 | 8.85 |
| prostate | AR_mut | A-on_strict | 704 | 226 | 0.142 | 7.96 |
| crc | KRAS_any | A-on_strict | 1,479 | 26 | 41.8 | 57.7 |
