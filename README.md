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

 
- Tracking which subclones survive treatment across two samples is not supported: 13 dated pairs, at most 689 unverified ones.
- Predicting response from a pre-treatment sample is well supported. 62% of patients (about 7,500) had their sample collected before systemic therapy started, so their genome is a clean pre-treatment picture. You can predict how they fare on the following therapy, for example time to progression, using MSK-CHORD's text-mined progression timeline (the radiology-report outcome from Q10).
- Predicting where the cancer will spread is better than I estimated. 16% of patients develop their first distant metastasis after the sample, and many more develop new organ sites later. 47% are diagnosed with metastasis within 30 days of the sample, which suggests the biopsy is often the procedure that diagnosed the metastasis.