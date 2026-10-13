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


# --------------------------------------------------------------------------
# Stage 0 cache (data_cleaning_plan.md Section 0.3 / 4). The three large IMPACT
# text files are parsed once into data/raw/cache/*.parquet; every later read
# goes through the parquet. Wide gene x sample matrices are cached long.
# --------------------------------------------------------------------------

CACHE = RAW / "cache"

# Mutation columns used anywhere in the v3 pipeline.
MUT_COLS = [
    "Hugo_Symbol", "Tumor_Sample_Barcode", "Chromosome", "Start_Position",
    "End_Position", "Reference_Allele", "Tumor_Seq_Allele2", "Consequence",
    "Variant_Classification", "Variant_Type", "Mutation_Status",
    "t_ref_count", "t_alt_count", "HGVSp_Short", "Protein_position",
    "ASCN.CLONAL", "ASCN.CCF_M_COPIES", "ASCN.CCF_M_COPIES_LOWER",
    "ASCN.CCF_M_COPIES_UPPER", "ASCN.PURITY", "ASCN.TOTAL_COPY_NUMBER",
    "ASCN.MINOR_COPY_NUMBER", "ASCN.FACETS_QC_PASSED",
]
SEG_COLS = ["ID", "chrom", "loc.start", "loc.end", "num.mark", "seg.mean"]


def _cached(name: str, build) -> pd.DataFrame:
    path = CACHE / f"{name}.parquet"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"[cache] building {path.name} (one-time parse of raw text)")
        build().to_parquet(path, index=False)
    return pd.read_parquet(path)


def _build_mutations() -> pd.DataFrame:
    m = pd.read_csv(IMPACT / "data_mutations.txt", sep="\t", usecols=MUT_COLS,
                    low_memory=False)
    for c in ["t_ref_count", "t_alt_count", "Start_Position", "End_Position",
              "ASCN.CCF_M_COPIES", "ASCN.CCF_M_COPIES_LOWER",
              "ASCN.CCF_M_COPIES_UPPER", "ASCN.PURITY",
              "ASCN.TOTAL_COPY_NUMBER", "ASCN.MINOR_COPY_NUMBER"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    for c in ["Chromosome", "Protein_position", "ASCN.FACETS_QC_PASSED"]:
        m[c] = m[c].astype("string")
    return m.rename(columns={"Tumor_Sample_Barcode": "SAMPLE_ID"})


def _melt_wide(path: Path, chunk: int = 4000) -> pd.DataFrame:
    """Gene x sample matrix -> long (SAMPLE_ID, gene, value), NaN dropped."""
    wide = pd.read_csv(path, sep="\t", index_col=0, dtype=str)
    wide.index.name = "gene"
    parts = []
    for i in range(0, wide.shape[1], chunk):
        s = wide.iloc[:, i:i + chunk].stack(future_stack=True).dropna()
        parts.append(s.rename("value").reset_index().rename(
            columns={"level_1": "SAMPLE_ID"}))
    out = pd.concat(parts, ignore_index=True)
    for c in ["SAMPLE_ID", "gene"]:
        out[c] = out[c].astype("category")
    return out


def _build_cna() -> pd.DataFrame:
    # Discrete GISTIC-style calls; only non-zero calls are kept (zero is the
    # measured default on a covering panel, see panel masking in Stage 5).
    long = _melt_wide(IMPACT / "data_cna.txt")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long[long["value"].fillna(0) != 0].reset_index(drop=True)


def _build_ascna() -> pd.DataFrame:
    # Cells are "total;minor" integer copy numbers from FACETS.
    long = _melt_wide(IMPACT / "data_CNA.ascna.txt")
    tm = long.pop("value").str.split(";", n=1, expand=True)
    long["tcn"] = pd.to_numeric(tm[0], errors="coerce").astype("Int16")
    long["mcn"] = pd.to_numeric(tm[1], errors="coerce").astype("Int16")
    return long


def _build_seg() -> pd.DataFrame:
    return pd.read_csv(IMPACT / "data_cna_hg19.seg", sep="\t", usecols=SEG_COLS,
                       dtype={"chrom": str})


@functools.lru_cache(maxsize=None)
def load_mutations() -> pd.DataFrame:
    """IMPACT MAF (somatic), cached. Sample column renamed to SAMPLE_ID."""
    return _cached("mutations", _build_mutations)


@functools.lru_cache(maxsize=None)
def load_cna() -> pd.DataFrame:
    """IMPACT discrete CNA, long, non-zero calls only (value in {-2,-1.5,-1,1,2})."""
    return _cached("cna_long", _build_cna)


@functools.lru_cache(maxsize=None)
def load_ascna() -> pd.DataFrame:
    """IMPACT allele-specific CN, long: SAMPLE_ID, gene, tcn, mcn."""
    return _cached("ascna_long", _build_ascna)


@functools.lru_cache(maxsize=None)
def load_seg() -> pd.DataFrame:
    """IMPACT hg19 segmentation (log2 ratio seg.mean)."""
    return _cached("seg", _build_seg)


@functools.lru_cache(maxsize=None)
def load_sv() -> pd.DataFrame:
    """IMPACT structural variants (small; read directly)."""
    sv = pd.read_csv(IMPACT / "data_sv.txt", sep="\t", low_memory=False)
    return sv.rename(columns={"Sample_Id": "SAMPLE_ID"})


@functools.lru_cache(maxsize=None)
def load_gene_panel_matrix() -> pd.DataFrame:
    """Per-sample panel for mutations / gistic (CNA) / structural_variants."""
    return pd.read_csv(IMPACT / "data_gene_panel_matrix.txt", sep="\t")


@functools.lru_cache(maxsize=None)
def cna_profiled_samples() -> frozenset[str]:
    """Samples that are columns of data_cna.txt (i.e. have CNA data)."""
    with open(IMPACT / "data_cna.txt") as fh:
        return frozenset(fh.readline().rstrip("\n").split("\t")[1:])


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
