# AI-Health-Project

## Datasets

- [IMPACT dataset](https://www.cbioportal.org/study/summary?id=msk_impact_50k_2026)
- [MET dataset](https://www.cbioportal.org/study/summary?id=msk_met_2021)

## Order of events

These are the 12,095 patients who are in both MSK-MET (metastasis) and MSK-CHORD (treatment) and have a primary diagnosis date. That means the five MSK-CHORD cancer types only, because no other cancer type has treatment dates.

**Abbreviations:** Dx = primary diagnosis; Met = first distant metastasis; Tx = first systemic therapy; Sample = collection of the MSK-MET sample.

Events on the same day are ordered Dx, Met, Tx, Sample.

| Order of events | Patients | % | ≥2 samples sequenced (dates unknown) | Second sample with a date | Dated sample after Tx | Dated before + after Tx pair |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dx → Met → Sample → Tx | 3,227 | 26.7 | 289 | 6 | 4 | 4 |
| Dx → Sample → Met → Tx | 2,250 | 18.6 | 206 | 8 | 7 | 7 |
| Dx → Met → Tx → Sample | 1,653 | 13.7 | 179 | 2 | 1 | 1 |
| Dx → Sample → Tx | 1,051 | 8.7 | 72 | 2 | 0 | 0 |
| Dx → Sample → Tx → Met | 760 | 6.3 | 91 | 1 | 1 | 1 |
| Dx → Tx → Met → Sample | 746 | 6.2 | 64 | 0 | 0 | 0 |
| Dx → Met → Sample (no Tx) | 703 | 5.8 | 20 | 1 | 0 | 0 |
| Dx → Sample (no Tx, no Met) | 615 | 5.1 | 17 | 1 | 0 | 0 |
| Dx → Sample → Met (no Tx) | 181 | 1.5 | 9 | 2 | 0 | 0 |
| Met → Dx → Sample → Tx | 168 | 1.4 | 17 | 0 | 0 | 0 |
| Met → Tx → Dx → Sample | 144 | 1.2 | 13 | 0 | 0 | 0 |
| Tx → Dx → Met → Sample | 115 | 1.0 | 13 | 1 | 1 | 0 |
| Dx → Tx → Sample → Met | 101 | 0.8 | 20 | 1 | 1 | 0 |
| Dx → Tx → Sample (no Met) | 95 | 0.8 | 7 | 0 | 0 | 0 |
| 18 smaller groups | 386 | 3.2 | 30 | 1 | 1 | 0 |
| **Total** | **12,095** | **100** | **1,047 (8.7%)** | **26** | **16** | **13** |


## Preprocessing run order, from data/: 
exposure.py → build_cohort.py → build_alterations.py → preprocess.py → report.py. The first run builds the parquet cache (about 2 minutes); a full run takes about 10 minutes.