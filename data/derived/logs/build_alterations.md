# build_alterations

_Written 2026-10-13 00:16_

Stages 5-6 (build_alterations.py). Oncogenicity source: **rules_v1** (rule-based fallback;
data/raw/oncokb/ is empty). Rules derived from all 51,191 MSK-IMPACT 50K
samples minus 3,585 hypermutated ones: 571 missense hotspot residues
(>= 10 samples and >= 5x gene background),
213 in-frame indel hotspots, gene roles by the 20/20 rule
(167 TSG, 44 oncogene,
50 TSG by deep deletion, 155 oncogene by amplification).
For the 7,293 cohort samples: 50,851 nonsynonymous mutations in, 21,340
oncogenic; 17,417 of 21,164 amp/homdel calls kept; 928 of
2,070 SVs kept; 332,501 TSG LOH calls. Legacy gene symbols mapped via config/gene_aliases.csv.
Matrix: 7,293 samples x 2,095 features (NA off-panel). FGA vs MSK-MET: r = 0.995 on 4,425 cohort samples, r = 0.996 on all 25,037 overlapping samples.

## Gene roles for reference genes

| gene | n | trunc | hot | n_amp | n_homdel | role |
|---|---:|---:|---:|---:|---:|---|
| TP53 | 2.42e+04 | 0.332 | 0.455 | 16 | 644 | TSG |
| PTEN | 3.57e+03 | 0.579 | 0.264 | 9 | 1.26e+03 | TSG |
| RB1 | 2.46e+03 | 0.825 | 0.0159 | 56 | 1e+03 | TSG |
| APC | 6.89e+03 | 0.889 | 0 | 33 | 137 | TSG |
| CDKN2A | 2.28e+03 | 0.618 | 0.191 | 32 | 5.38e+03 | TSG |
| KRAS | 8.38e+03 | 0.00131 | 0.947 | 868 | 22 | oncogene |
| PIK3CA | 7.04e+03 | 0.00852 | 0.886 | 461 | 13 | oncogene |
| EGFR | 3.42e+03 | 0.0219 | 0.799 | 1.66e+03 | 34 | oncogene |
| ESR1 | 952 | 0.0599 | 0.604 | 175 | 100 | oncogene |
| AR | 789 | 0.0963 | 0.2 | 472 | 169 | oncogene |
| MET | 751 | 0.325 | 0.206 | 468 | 19 | oncogene |
| ERBB2 | 1.35e+03 | 0.0311 | 0.667 | 2.05e+03 | 29 | oncogene |
| MYC | 228 | 0.0789 | 0.149 | 2.95e+03 | 8 | oncogene_cna |
| CCND1 | 165 | 0.164 | 0.345 | 2.62e+03 | 37 | oncogene |
| BRAF | 2.19e+03 | 0.0182 | 0.835 | 126 | 14 | oncogene |
| SMAD4 | 2.27e+03 | 0.397 | 0.443 | 25 | 524 | TSG |
| ARID1A | 4.03e+03 | 0.798 | 0.0124 | 11 | 197 | TSG |
| BRCA2 | 1.17e+03 | 0.377 | 0.00854 | 155 | 182 | TSG |
| FGFR1 | 372 | 0.0914 | 0.228 | 1.74e+03 | 323 | oncogene |
| MDM2 | 169 | 0.107 | 0 | 2e+03 | 2 | oncogene_cna |

## Oncogenic calls by rule

| alt_type | onco_rule | n |
|---|---|---:|
| amp | amp_dominant_gene | 14,481 |
| del | homdel_dominant_gene | 2,936 |
| loh | loh_tsg | 332,501 |
| mut | hotspot_inframe | 755 |
| mut | hotspot_missense | 7,323 |
| mut | met_ex14 | 1 |
| mut | tert_promoter | 55 |
| mut | truncating_in_tsg | 13,206 |
| sv | fusion_driver_partner | 1,606 |

## Controls (T4 stages)

| cohort | control | stage | n_pre | n_post | pct_pre | pct_post |
|---|---|---|---:|---:|---:|---:|
| breast | ESR1 mutation | onco_filter | 1,789 | 818 | 1.4 | 16.3 |
| breast | ESR1 mutation | panel_mask | 1,789 | 818 | 1.4 | 16.3 |
| breast | ESR1_LBD (AI subclass) | panel_mask | 1,251 | 462 | 1.28 | 18.2 |
| nsclc | EGFR T790M | onco_filter | 561 | 192 | 1.78 | 37.5 |
| nsclc | EGFR T790M | panel_mask | 561 | 192 | 1.78 | 37.5 |
| nsclc | EGFR C797S | onco_filter | 561 | 192 | 0 | 2.08 |
| nsclc | EGFR C797S | panel_mask | 561 | 192 | 0 | 2.08 |
| nsclc | MET amplification | onco_filter | 561 | 192 | 0.891 | 9.9 |
| nsclc | MET amplification | panel_mask | 561 | 192 | 0.891 | 9.9 |
| prostate | AR amplification | onco_filter | 1,014 | 501 | 0.789 | 34.9 |
| prostate | AR amplification | panel_mask | 1,014 | 501 | 0.789 | 34.9 |
| prostate | AR mutation | onco_filter | 1,014 | 501 | 0.394 | 8.18 |
| prostate | AR mutation | panel_mask | 1,014 | 501 | 0.394 | 8.18 |
| crc | KRAS mutation | onco_filter | 1,921 | 497 | 40.9 | 41.9 |
| crc | KRAS mutation | panel_mask | 1,921 | 497 | 40.9 | 41.9 |

## Signature checks

| cohort | signature | n_pre | n_post | mean_pre | mean_post | pct_dominant_pre | pct_dominant_post |
|---|---|---:|---:|---:|---:|---:|---:|
| crc | Signature 17 (5-FU) | 421 | 109 | 0.0095 | 0.0534 | 1.43 | 10.1 |
| breast | APOBEC (2+13) | 77 | 124 | 0.57 | 0.584 | 83.1 | 79.8 |
