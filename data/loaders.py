"""Loaders for the MSK-MET / MSK-IMPACT 50K / MSK-CHORD cBioPortal exports.

Three things about these files drive every design choice here:

1. cBioPortal clinical files carry a 5-line header. Rows 1-4 are metadata
   (display name, description, datatype, priority) and row 5 is the real
   column header. Read them with ``skiprows=4``.
2. ``data_mutations.txt`` (190 MB) and ``data_CNA.ascna.txt`` (103 MB) are too
   large to read whole. Always pass ``usecols``.
3. Timeline dates use two different origins. MSK-MET counts days since birth;
   MSK-CHORD counts days since the patient's first sequencing. Anything that
   compares the two must go through :func:`anchor_met_to_sequencing` first.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
RAW = DATA_DIR / "raw"
DERIVED = DATA_DIR / "derived"

MET = RAW / "msk_met_2021"
IMPACT = RAW / "msk_impact_50k_2026"
CHORD = RAW / "msk_chord_2024"

DERIVED.mkdir(exist_ok=True)

# Columns 46-60 of the IMPACT mutations file: the FACETS allele-specific copy
# number block that CCF is derived from, plus the identifiers needed to join.
ASCN_COLS = [
    "Hugo_Symbol",
    "Tumor_Sample_Barcode",
    "Variant_Classification",
    "t_ref_count",
    "t_alt_count",
    "ASCN.CLONAL",
    "ASCN.CCF_M_COPIES",
    "ASCN.CCF_M_COPIES_LOWER",
    "ASCN.CCF_M_COPIES_UPPER",
    "ASCN.PURITY",
    "ASCN.TOTAL_COPY_NUMBER",
    "ASCN.MINOR_COPY_NUMBER",
    "ASCN.EXPECTED_ALT_COPIES",
    "ASCN.FACETS_QC_PASSED",
]

# MSK-MET's 21 per-organ "distant metastasis at diagnosis" flags. These, not
# METASTATIC_SITE, are the metastasis labels: METASTATIC_SITE only records
# where the sequenced biopsy was taken from.
DMETS_COLS = [
    "DMETS_DX_UNSPECIFIED", "DMETS_DX_DIST_LN", "DMETS_DX_LUNG",
    "DMETS_DX_PLEURA", "DMETS_DX_MEDIASTINUM", "DMETS_DX_LIVER",
    "DMETS_DX_BILIARY_TRACT", "DMETS_DX_INTRA_ABDOMINAL", "DMETS_DX_BOWEL",
    "DMETS_DX_CNS_BRAIN", "DMETS_DX_BONE", "DMETS_DX_PNS",
    "DMETS_DX_ADRENAL_GLAND", "DMETS_DX_KIDNEY", "DMETS_DX_BLADDER_UT",
    "DMETS_DX_FEMALE_GENITAL", "DMETS_DX_OVARY", "DMETS_DX_MALE_GENITAL",
    "DMETS_DX_SKIN", "DMETS_DX_HEAD_NECK", "DMETS_DX_BREAST",
]

# MSK-MET's controlled METASTATIC_SITE vocabulary is the normalization target
# for the free-text site fields elsewhere. Maps the SUBTYPE values used in
# data_timeline_diagnosis.txt onto the DMETS_DX_* organ names.
TIMELINE_SUBTYPE_TO_ORGAN = {
    "Lung": "LUNG", "Liver": "LIVER", "Bone": "BONE",
    "Dist Lymph": "DIST_LN", "Regional Lymph": "DIST_LN", "Lymph": "DIST_LN",
    "Peritoneum": "INTRA_ABDOMINAL", "Pleura": "PLEURA",
    "CNS Brain": "CNS_BRAIN", "Bowel": "BOWEL",
    "Biliary Tract": "BILIARY_TRACT", "Adrenal Gland": "ADRENAL_GLAND",
    "Genital Female": "FEMALE_GENITAL", "Genital Male": "MALE_GENITAL",
    "Bladder Or Urinary Tract": "BLADDER_UT", "Ovary": "OVARY",
    "Peripheral Nervous System": "PNS", "Skin": "SKIN", "Kidney": "KIDNEY",
    "Mediastinum": "MEDIASTINUM", "Head And Neck": "HEAD_NECK",
    "Breast": "BREAST", "Other": "UNSPECIFIED",
}


def read_clinical(path: Path) -> pd.DataFrame:
    """Read a cBioPortal clinical file, skipping its 4 metadata rows."""
    return pd.read_csv(path, sep="\t", skiprows=4, low_memory=False)


def read_timeline(path: Path) -> pd.DataFrame:
    """Read a cBioPortal timeline file (ordinary single-row header)."""
    return pd.read_csv(path, sep="\t", low_memory=False)


@functools.lru_cache(maxsize=None)
def load_met_sample() -> pd.DataFrame:
    return read_clinical(MET / "data_clinical_sample.txt")


@functools.lru_cache(maxsize=None)
def load_met_patient() -> pd.DataFrame:
    return read_clinical(MET / "data_clinical_patient.txt")


@functools.lru_cache(maxsize=None)
def load_impact_sample() -> pd.DataFrame:
    return read_clinical(IMPACT / "data_clinical_sample.txt")


@functools.lru_cache(maxsize=None)
def load_impact_patient() -> pd.DataFrame:
    return read_clinical(IMPACT / "data_clinical_patient.txt")


@functools.lru_cache(maxsize=None)
def load_chord_sample() -> pd.DataFrame:
    return read_clinical(CHORD / "data_clinical_sample.txt")


@functools.lru_cache(maxsize=None)
def load_chord_patient() -> pd.DataFrame:
    return read_clinical(CHORD / "data_clinical_patient.txt")


@functools.lru_cache(maxsize=None)
def load_met_timeline(name: str) -> pd.DataFrame:
    """MSK-MET timeline: one of diagnosis, status, surgery, sequencing.

    START_DATE is days since birth.
    """
    return read_timeline(MET / f"data_timeline_{name}.txt")


@functools.lru_cache(maxsize=None)
def load_chord_timeline(name: str) -> pd.DataFrame:
    """MSK-CHORD timeline, e.g. treatment, progression, tumor_sites, specimen.

    START_DATE is days since the patient's first sequencing (day 0), so values
    before the biopsy are negative.
    """
    return read_timeline(CHORD / f"data_timeline_{name}.txt")


def load_ascn_mutations(usecols: list[str] | None = None) -> pd.DataFrame:
    """Read the IMPACT mutations file (190 MB) restricted to the ASCN block."""
    return pd.read_csv(
        IMPACT / "data_mutations.txt",
        sep="\t",
        usecols=usecols or ASCN_COLS,
        low_memory=False,
    )


@functools.lru_cache(maxsize=None)
def met_sequencing_dates() -> pd.Series:
    """Each MSK-MET patient's first sequencing date, in days since birth."""
    seq = load_met_timeline("sequencing")
    return seq.groupby("PATIENT_ID")["START_DATE"].min()


def anchor_met_to_sequencing(df: pd.DataFrame) -> pd.DataFrame:
    """Re-express MSK-MET timeline dates as days since sequencing.

    MSK-MET stores days since birth; MSK-CHORD stores days since sequencing.
    Subtracting each patient's sequencing date puts both studies on the same
    day-zero axis so their events can be compared. Adds DAYS_FROM_SEQ and
    drops patients with no sequencing date (259 of 25,776).
    """
    seq = met_sequencing_dates()
    out = df.copy()
    out["SEQ_DATE"] = out["PATIENT_ID"].map(seq)
    out = out[out["SEQ_DATE"].notna()].copy()
    out["DAYS_FROM_SEQ"] = out["START_DATE"] - out["SEQ_DATE"]
    return out


def normalize_site(s: pd.Series) -> pd.Series:
    """Collapse the free-text site fields (IMPACT has 364 distinct values,
    including both 'Lymph Node' and 'Lymph node') to title-cased, trimmed
    strings so that case-only duplicates merge."""
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
    )


def cohorts() -> dict[str, set[str]]:
    """Patient-ID sets for each study, for overlap accounting."""
    return {
        "met": set(load_met_sample()["PATIENT_ID"]),
        "impact": set(load_impact_sample()["PATIENT_ID"]),
        "chord": set(load_chord_sample()["PATIENT_ID"]),
    }
