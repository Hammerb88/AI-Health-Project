# exposure

_Written 2026-10-13 00:12_

Stage 2 (exposure.py). Read 134,943 MSK-CHORD treatment rows (245 agents,
all mapped to a class). Wrote 11,498 patient x cohort-class rows to `derived/exposure.parquet`.
Class rows dated before the cohort cancer's primary diagnosis - 30 d were dropped
(belong to an earlier cancer). Episodes merge gaps <= 60 d. First distant
metastasis is the earliest of Stage 4 at diagnosis, an NLP radiology mention of a distant organ,
or a non-nodal metastasis biopsy (config/pipeline.yaml).

## Input / filter counts

| cohort | patients_with_cancer | patients_with_class | anchor_rows_dropped_pre_dx | no_matching_primary_dx |
|---|---:|---:|---:|---:|
| breast | 5,368 | 3,988 | 137 | 2 |
| nsclc | 7,809 | 1,313 | 5 | 7 |
| prostate | 3,211 | 2,696 | 3 | 0 |
| crc | 5,543 | 3,501 | 16 | 26 |

## Per-cohort summary (shares)

| cohort | patients | strict_first_line | adjuvant_start | has_met | prior_med_to_msk | median_episodes |
|---|---:|---:|---:|---:|---:|---:|
| breast | 3,988 | 0.551 | 0.554 | 0.722 | 0.201 | 1 |
| crc | 3,501 | 0.868 | 0.388 | 0.793 | 0.124 | 1 |
| nsclc | 1,313 | 0.682 | 0.171 | 0.912 | 0.181 | 1 |
| prostate | 2,696 | 0.916 | 0.263 | 0.898 | 0.244 | 1 |

## Subclass counts

| cohort | 1st/2nd-gen TKI | AI | CAPOX | FOLFOX | GnRH | antiandrogen | fulvestrant | osimertinib | other | tamoxifen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| breast | 0 | 2,743 | 0 | 0 | 0 | 0 | 275 | 0 | 0 | 970 |
| crc | 0 | 0 | 439 | 2,716 | 0 | 0 | 0 | 0 | 346 | 0 |
| nsclc | 545 | 0 | 0 | 0 | 0 | 0 | 0 | 768 | 0 | 0 |
| prostate | 0 | 0 | 0 | 0 | 2,461 | 235 | 0 | 0 | 0 | 0 |

## First-metastasis source

| cohort | met_biopsy | nlp_site | stage4_dx |
|---|---:|---:|---:|
| breast | 303 | 1,845 | 733 |
| crc | 77 | 1,165 | 1,536 |
| nsclc | 17 | 397 | 783 |
| prostate | 52 | 1,097 | 1,272 |
