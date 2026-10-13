# build_cohort

_Written 2026-10-13 00:13_

Stages 3-4 (build_cohort.py). Started from 25,040 dated MSK-CHORD samples and
11,498 patient x class exposure rows. Wrote 7,293 samples (one per patient per
cohort) to `derived/samples.parquet` for label variant A, and variants
A-on to `samples_<variant>.parquet`.
Label-0 samples from patients with prior medications to MSK are removed in every tier.
breast: one-sample-per-patient removed 0 samples NSCLC: 87.0% of samples carry an EGFR-sensitizing alteration before restriction nsclc: one-sample-per-patient removed 0 samples prostate: one-sample-per-patient removed 0 samples crc: one-sample-per-patient removed 0 samples

## Label counts (variant A)

| cohort | 0 | 1 |
|---|---:|---:|
| breast | 1,789 | 818 |
| crc | 1,921 | 497 |
| nsclc | 561 | 192 |
| prostate | 1,014 | 501 |

## Tier sizes (variant A)

| cohort | tier_relaxed | tier_strict | tier_strict_noint |
|---|---:|---:|---:|
| breast | 2,607 | 1,040 | 944 |
| crc | 2,418 | 1,991 | 1,853 |
| nsclc | 753 | 432 | 406 |
| prostate | 1,515 | 1,026 | 852 |

## Section 2.1 reproduction (computed)

| cohort | samples | t_le_0 | t_0_90 | t_gt_90 | t_gt_90_on_class | strict_t_le_0 | strict_t_gt_90 | strict_t_gt_90_no_between | median_t_gt_90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| breast | 3,988 | 2,655 | 287 | 1,046 | 615 | 847 | 245 | 145 | 1.16e+03 |
| nsclc | 1,313 | 975 | 72 | 266 | 200 | 346 | 117 | 87 | 556 |
| prostate | 2,696 | 1,853 | 242 | 601 | 450 | 727 | 299 | 125 | 1.01e+03 |
| crc | 3,501 | 2,819 | 84 | 598 | 57 | 1,604 | 387 | 249 | 350 |

## Section 2.2 controls (raw mutations, loose cohort)

| cohort | control | stage | n_pre | n_post | pct_pre | pct_post | plan_pct_pre | plan_pct_post | expected |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| breast | ESR1 mutation | raw_plan_denominator | 2,655 | 1,046 | 2.64 | 14.3 | 2.5 | 14.3 | up |
| breast | ESR1 mutation | raw | 2,329 | 835 | 3.01 | 18 | 2.5 | 14.3 | up |
| nsclc | EGFR T790M | raw_plan_denominator | 975 | 266 | 1.54 | 27.1 | 1.5 | 26.9 | up |
| nsclc | EGFR C797S | raw_plan_denominator | 975 | 266 | 0.205 | 1.5 | 0.3 | 1.5 | up |
| nsclc | MET amplification | raw_plan_denominator | 975 | 266 | 0.821 | 7.14 | 0.8 | 7.1 | up |
| nsclc | EGFR T790M | raw | 771 | 211 | 1.95 | 34.1 | 1.5 | 26.9 | up |
| nsclc | EGFR C797S | raw | 771 | 211 | 0.259 | 1.9 | 0.3 | 1.5 | up |
| nsclc | MET amplification | raw | 771 | 211 | 1.04 | 9 | 0.8 | 7.1 | up |
| prostate | AR amplification | raw_plan_denominator | 1,853 | 601 | 1.3 | 29.1 | 1.3 | 29.1 | up |
| prostate | AR mutation | raw_plan_denominator | 1,853 | 601 | 0.809 | 8.49 | 0.8 | 8.5 | up |
| prostate | AR amplification | raw | 1,398 | 501 | 1.72 | 34.9 | 1.3 | 29.1 | up |
| prostate | AR mutation | raw | 1,398 | 501 | 1.07 | 10.2 | 0.8 | 8.5 | up |
| crc | KRAS mutation | raw_plan_denominator | 2,819 | 598 | 35.7 | 38.3 | 35.7 | 37.9 | flat |
| crc | KRAS mutation | raw | 2,213 | 497 | 45.5 | 46.1 | 35.7 | 37.9 | flat |
