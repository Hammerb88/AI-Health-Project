"""Stages 5-6: alteration matrix and sample-level genomic features
(data_cleaning_plan.md Sections 9-10).

Writes
  derived/oncogenicity_rules.parquet    gene roles + hotspot residues (rule set)
  derived/alterations_long.parquet      one row per sample x alteration event
  derived/alteration_matrix.parquet     wide gene x type matrix, NA off-panel,
                                        clonality + variant-level features
  derived/alteration_matrix_gene.parquet collapsed <gene>_any matrix (S9)
  derived/genomic_features.parquet      TMB, FGA, MSI, FACETS, n_mutations_oncogenic
  derived/signature_checks.csv          COSMIC v2 signature sanity checks (9.7)
  derived/controls.csv                  (appended) T4 after oncogenicity filter / panel mask
  derived/logs/build_alterations.md

Oncogenicity (D4): OncoKB-annotated files are used only if data/raw/oncokb/
is populated and config says `oncokb`; otherwise the rule-based fallback
(Section 9.1) below. OncoKB evidence levels / treatment fields are never used.

Run: ``python build_alterations.py``
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

import common as C
import loaders as L

PANELS = ["IMPACT341", "IMPACT410", "IMPACT468", "IMPACT505"]
RULESET = "rules_v1"

NONSYN = {"Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
          "Splice_Site", "In_Frame_Del", "In_Frame_Ins", "Translation_Start_Site",
          "Nonstop_Mutation"}
TRUNCATING = {"Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Splice_Site",
              "Translation_Start_Site", "Nonstop_Mutation"}
INFRAME = {"In_Frame_Del", "In_Frame_Ins"}

# Hotspot / role thresholds for the rule-based fallback, computed over the
# whole MSK-IMPACT 50K MAF excluding hypermutated samples (no labels used).
HOTSPOT_MIN_SAMPLES = 10        # missense residue recurrently hit
HOTSPOT_FOLD_OVER_UNIFORM = 5   # ... and >= 5x the gene's per-residue background
INFRAME_HOTSPOT_MIN = 5         # in-frame indels starting at the same codon
ROLE_MIN_MUTATIONS = 20         # 20/20 rule needs some mutations to classify
ROLE_FRACTION = 0.20            # Vogelstein et al. 2013 "20/20 rule"
CNA_DOMINANCE = 2.0             # amp:homdel (or reverse) ratio for CNA-only genes

# Documented exceptions to the 20/20 rule. MET exon-14 skipping variants are
# splice-site / in-frame changes that activate MET, so MET looks "truncated"
# but is an oncogene; its exon-14 variants are oncogenic via `met_ex14`.
ROLE_OVERRIDES = {"MET": "oncogene"}

# Fusion drivers: kinases / oncogenes whose rearrangements are oncogenic.
FUSION_DRIVERS = {"ALK", "ROS1", "RET", "NTRK1", "NTRK2", "NTRK3", "FGFR1", "FGFR2", "FGFR3",
                  "BRAF", "RAF1", "MET", "EGFR", "ERBB2", "NRG1", "ERG", "ETV1", "ETV4", "ETV5",
                  "ESR1", "AR", "PIK3CA", "NOTCH1", "NOTCH2", "MAP3K1", "PRKACA", "PDGFRA",
                  "PDGFRB", "KIT", "ABL1", "JAK2", "BCL2", "MYC", "NUTM1", "TFE3", "EWSR1"}


# --------------------------------------------------------------------------
# Panels and gene symbols
# --------------------------------------------------------------------------

def panel_genes() -> dict[str, frozenset[str]]:
    out = {}
    for p in PANELS:
        f = L.RAW / "panels" / f"{p}.json"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing: download from https://www.cbioportal.org/api/gene-panels/{p}")
        out[p] = frozenset(g["hugoGeneSymbol"] for g in json.loads(f.read_text())["genes"])
    return out


def alias_map() -> dict[str, str]:
    a = pd.read_csv(C.CONFIG / "gene_aliases.csv")
    return dict(zip(a.alias, a.symbol))


def canon(genes: pd.Series) -> pd.Series:
    return genes.astype(str).replace(alias_map())


# --------------------------------------------------------------------------
# Rule-based oncogenicity (Section 9.1 fallback)
# --------------------------------------------------------------------------

def _codon(pp: pd.Series) -> pd.Series:
    return pd.to_numeric(pp.astype("string").str.extract(r"^(\d+)")[0], errors="coerce")


def oncogenicity_rules(mut: pd.DataFrame, cna: pd.DataFrame, hyper: set[str]) -> dict:
    """Derive gene roles and hotspot residues from the full IMPACT cohort."""
    m = mut[mut.Variant_Classification.isin(NONSYN) & ~mut.SAMPLE_ID.isin(hyper)].copy()
    m["codon"] = _codon(m.Protein_position)

    mis = m[m.Variant_Classification == "Missense_Mutation"]
    per_res = mis.groupby(["Hugo_Symbol", "codon"]).SAMPLE_ID.nunique().rename("n").reset_index()
    gene_mis = mis.groupby("Hugo_Symbol").SAMPLE_ID.size()
    # Protein_position holds the codon only (no "/length"); the highest codon
    # mutated in 50K samples is a close proxy for protein length.
    plen = m.groupby("Hugo_Symbol").codon.max()
    per_res["background"] = per_res.Hugo_Symbol.map(gene_mis / plen)
    hot = per_res[(per_res.n >= HOTSPOT_MIN_SAMPLES)
                  & (per_res.n >= HOTSPOT_FOLD_OVER_UNIFORM * per_res.background)]
    hot_mis = set(zip(hot.Hugo_Symbol, hot.codon))

    inf = m[m.Variant_Classification.isin(INFRAME)]
    per_inf = inf.groupby(["Hugo_Symbol", "codon"]).SAMPLE_ID.nunique()
    hot_inf = set(per_inf[per_inf >= INFRAME_HOTSPOT_MIN].index)

    m["is_trunc"] = m.Variant_Classification.isin(TRUNCATING)
    m["is_hot"] = [(g, c) in hot_mis or (g, c) in hot_inf
                   for g, c in zip(m.Hugo_Symbol, m.codon)]
    g = m.groupby("Hugo_Symbol").agg(n=("is_trunc", "size"), trunc=("is_trunc", "mean"),
                                     hot=("is_hot", "mean"))
    c = cna.assign(gene=canon(cna.gene)).groupby(["gene", "value"]).size().unstack(fill_value=0)
    c = c.rename(columns={2: "n_amp", -2: "n_homdel"}).reindex(columns=["n_amp", "n_homdel"], fill_value=0)
    roles = g.join(c, how="outer").fillna({"n": 0, "n_amp": 0, "n_homdel": 0})
    roles["role"] = "unclassified"
    enough = roles.n >= ROLE_MIN_MUTATIONS
    roles.loc[enough & (roles.trunc >= ROLE_FRACTION), "role"] = "TSG"
    roles.loc[enough & (roles.trunc < ROLE_FRACTION) & (roles.hot >= ROLE_FRACTION), "role"] = "oncogene"
    # CNA-driven genes rarely mutated (MYC, CCND1, CDKN2A, ...): role from the
    # direction their copy-number changes take.
    unc = roles.role == "unclassified"
    roles.loc[unc & (roles.n_amp >= CNA_DOMINANCE * roles.n_homdel) & (roles.n_amp >= 20), "role"] = "oncogene_cna"
    roles.loc[unc & (roles.n_homdel >= CNA_DOMINANCE * roles.n_amp) & (roles.n_homdel >= 20), "role"] = "TSG_cna"
    for g, r in ROLE_OVERRIDES.items():
        roles.loc[g, "role"] = r
    # CNA direction is decided by the CNA data itself, not the mutation role
    # (e.g. FOXA1 is truncated *and* amplified; only the amp is the driver).
    roles["amp_driver"] = roles.n_amp >= CNA_DOMINANCE * roles.n_homdel
    roles["del_driver"] = roles.n_homdel >= CNA_DOMINANCE * roles.n_amp
    roles.index.name = "gene"
    return {"roles": roles, "hot_mis": hot_mis, "hot_inf": hot_inf}


def annotate_mutations(m: pd.DataFrame, rules: dict) -> pd.DataFrame:
    roles = rules["roles"].role
    m = m.copy()
    m["codon"] = _codon(m.Protein_position)
    role = m.Hugo_Symbol.map(roles).fillna("unclassified")
    tsg = role.str.startswith("TSG")
    key = list(zip(m.Hugo_Symbol, m.codon))
    hot_mis = np.array([k in rules["hot_mis"] for k in key])
    hot_inf = np.array([k in rules["hot_inf"] for k in key])
    vc = m.Variant_Classification
    tert_promoter = (m.Hugo_Symbol == "TERT") & (vc == "5'Flank")
    met_ex14 = (m.Hugo_Symbol == "MET") & (vc.isin({"Splice_Site", "In_Frame_Del"}))         & m.codon.between(960, 1015)
    rule = np.select(
        [tert_promoter, met_ex14,
         vc.isin(TRUNCATING) & tsg,
         (vc == "Missense_Mutation") & hot_mis,
         vc.isin(INFRAME) & hot_inf],
        ["tert_promoter", "met_ex14", "truncating_in_tsg", "hotspot_missense", "hotspot_inframe"],
        default="")
    m["onco_rule"] = rule
    m["oncogenic"] = rule != ""
    m["gene_role"] = role.values
    return m


def oncokb_available() -> bool:
    d = L.RAW / "oncokb"
    return d.exists() and any(d.iterdir())


# --------------------------------------------------------------------------
# Matrix construction
# --------------------------------------------------------------------------

def clonality_calls(m: pd.DataFrame) -> pd.DataFrame:
    """Per oncogenic mutation: clonal / subclonal / unknown (Section 9.4).

    Only mutations with ASCN.FACETS_QC_PASSED == True can be clonal or
    subclonal (checklist 10); everything else is `unknown`.
    """
    qc = m["ASCN.FACETS_QC_PASSED"].astype("string").str.lower().eq("true")
    clonal = qc & (m["ASCN.CLONAL"].eq("CLONAL") | (m["ASCN.CCF_M_COPIES_LOWER"] >= 0.8))
    sub = qc & ~clonal & m["ASCN.CLONAL"].eq("SUBCLONAL")
    out = m.assign(mut_qc_pass=qc.values,
                   clonality=np.select([clonal, sub], ["clonal", "subclonal"], "unknown"),
                   ccf=m["ASCN.CCF_M_COPIES"].where(qc))
    return out


def variant_features(samples: pd.Index, mut_all: pd.DataFrame, mut_onc: pd.DataFrame,
                     cna: pd.DataFrame, sv: pd.DataFrame) -> pd.DataFrame:
    """Section 9.3 variant-level features (config/target_variants.yaml)."""
    cfg = C.config("target_variants")["cohorts"]
    out = pd.DataFrame(index=samples)
    for cohort, feats in cfg.items():
        for name, spec in feats.items():
            g, t = spec["gene"], spec["type"]
            mg = mut_all[mut_all.Hugo_Symbol == g]
            hg = mg.HGVSp_Short.fillna("")
            if t == "hgvsp":
                hit = mg[hg.isin(spec["values"])]
            elif t == "hgvsp_regex":
                hit = mg[hg.str.match(spec["pattern"])]
            elif t == "egfr_sensitizing":
                pos = _codon(mg.Protein_position)
                ex19 = mg.Variant_Classification.isin(INFRAME) & pos.between(729, 761) & hg.str.contains("del")
                hit = mg[ex19 | hg.eq("p.L858R")]
            elif t == "missense_at":
                res = "|".join(spec["residues"])
                hit = mg[(mg.Variant_Classification == "Missense_Mutation")
                         & hg.str.match(rf"^p\.({res})[A-Z]$")]
            elif t == "cna":
                hit = cna[(cna.gene == g) & (cna.value == spec["value"])]
            elif t == "sv_partner":
                hit = sv[(sv.Site1_Hugo_Symbol == g) | (sv.Site2_Hugo_Symbol == g)]
            elif t == "mut_any":
                hit = mut_onc[mut_onc.Hugo_Symbol == g]
            else:
                raise ValueError(f"unknown target-variant type {t}")
            out[name] = samples.isin(set(hit.SAMPLE_ID)).astype("float32")
            out.attrs.setdefault("gene", {})[name] = g
            out.attrs.setdefault("dtype", {})[name] = {"cna": "gistic", "sv_partner": "structural_variants"}.get(t, "mutations")
            out.attrs.setdefault("cohort", {})[name] = cohort
    return out


def fga_from_seg(seg: pd.DataFrame, samples: pd.Index, thr: float) -> pd.Series:
    s = seg[seg.ID.isin(samples) & ~seg.chrom.isin(["X", "Y", "23", "24"])].copy()
    s["len"] = (s["loc.end"] - s["loc.start"]).clip(lower=0)
    s["alt"] = s.len.where(s["seg.mean"].abs() > thr, 0)
    g = s.groupby("ID")[["len", "alt"]].sum()
    return (g.alt / g.len).rename("FGA")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build() -> None:
    F = C.pipeline_cfg()["filters"]
    src = C.pipeline_cfg()["oncogenicity_source"]
    if src == "oncokb" and not oncokb_available():
        print("  oncogenicity_source=oncokb but data/raw/oncokb/ is empty -> using rules")
        src = "rules"
    if src == "oncokb":
        raise NotImplementedError("OncoKB-annotated input not wired yet; data/raw/oncokb/ is empty in this checkout")

    samples = pd.read_parquet(L.DERIVED / "samples.parquet")
    sids = pd.Index(sorted(samples.SAMPLE_ID.unique()), name="SAMPLE_ID")
    pm = L.load_gene_panel_matrix().set_index("SAMPLE_ID").loc[sids]
    pg = panel_genes()
    imp = L.load_impact_sample().set_index("SAMPLE_ID")
    hyper_all = set(imp.index[(imp.MSI_TYPE == "Instable") | (imp.TMB_SCORE > F["hypermutated_tmb"])])

    # ---- raw inputs (cache) -------------------------------------------------
    mut = L.load_mutations().copy()
    mut["Hugo_Symbol"] = canon(mut.Hugo_Symbol)
    cna = L.load_cna().copy()
    cna["gene"] = canon(cna.gene)
    cna = cna[~cna.gene.str.startswith("CDKN2Ap")]          # isoform rows; CDKN2A row covers them
    cna["SAMPLE_ID"] = cna.SAMPLE_ID.astype(str)
    cna = cna.groupby(["SAMPLE_ID", "gene"], as_index=False).value.agg(lambda v: v.loc[v.abs().idxmax()])
    sv = L.load_sv().copy()
    for c in ["Site1_Hugo_Symbol", "Site2_Hugo_Symbol"]:
        sv[c] = sv[c].where(sv[c].isna(), canon(sv[c]))

    rules = oncogenicity_rules(mut, cna, hyper_all)
    roles = rules["roles"]
    roles.reset_index().to_parquet(L.DERIVED / "oncogenicity_rules.parquet", index=False)
    pd.DataFrame(sorted(rules["hot_mis"]), columns=["gene", "codon"]).assign(kind="missense") \
        .pipe(lambda d: pd.concat([d, pd.DataFrame(sorted(rules["hot_inf"]), columns=["gene", "codon"]).assign(kind="inframe")])) \
        .to_csv(L.DERIVED / "hotspots_rules.csv", index=False)

    # ---- restrict to cohort samples ----------------------------------------
    m = mut[mut.SAMPLE_ID.isin(sids)]
    m = m[m.Variant_Classification.isin(NONSYN) | ((m.Hugo_Symbol == "TERT") & (m.Variant_Classification == "5'Flank"))]
    m = clonality_calls(annotate_mutations(m, rules))
    cna_s = cna[cna.SAMPLE_ID.isin(sids) & cna.value.isin([2, -2])].copy()
    amp_ok = cna_s.gene.map(roles.amp_driver).fillna(False).astype(bool)
    del_ok = cna_s.gene.map(roles.del_driver).fillna(False).astype(bool)
    cna_s["alt_type"] = np.where(cna_s.value == 2, "amp", "del")
    cna_s["oncogenic"] = np.where(cna_s.value == 2, amp_ok, del_ok)
    cna_s["onco_rule"] = np.where(cna_s.oncogenic, np.where(cna_s.value == 2, "amp_dominant_gene",
                                                            "homdel_dominant_gene"), "")

    sv_s = sv[sv.SAMPLE_ID.isin(sids)].copy()
    antisense = sv_s.Event_Info.fillna("").str.contains("antisense", case=False)
    drivers = FUSION_DRIVERS | set(roles.index[roles.role.str.startswith("oncogene")])
    p1, p2 = sv_s.Site1_Hugo_Symbol, sv_s.Site2_Hugo_Symbol
    involves_driver = p1.isin(drivers) | p2.isin(drivers)
    sv_s["oncogenic"] = involves_driver & ~antisense
    sv_long = pd.concat([
        sv_s.assign(gene=p1, partner=p2), sv_s.assign(gene=p2, partner=p1)[p2.notna() & (p2 != p1)]
    ])
    sv_long = sv_long[sv_long.gene.notna()]
    sv_long["onco_rule"] = np.where(sv_long.oncogenic, "fusion_driver_partner", "")

    # LOH in TSGs, FACETS QC pass samples only.
    asc = L.load_ascna()
    asc = asc[asc.SAMPLE_ID.isin(sids)].copy()
    asc["SAMPLE_ID"] = asc.SAMPLE_ID.astype(str)
    asc["gene"] = canon(asc.gene)
    facets_ok = set(imp.index[imp.FACETS_QC == "Pass"])
    tsgs = set(roles.index[roles.role.str.startswith("TSG")])
    loh = asc[(asc.mcn == 0) & (asc.tcn >= 1) & asc.gene.isin(tsgs) & asc.SAMPLE_ID.isin(facets_ok)]
    asc_measured = asc[asc.SAMPLE_ID.isin(facets_ok) & asc.tcn.notna()]

    # ---- alterations_long ----------------------------------------------------
    long = pd.concat([
        m.assign(alt_type="mut", variant=m.HGVSp_Short.fillna(m.Variant_Classification))[
            ["SAMPLE_ID", "Hugo_Symbol", "alt_type", "variant", "Variant_Classification", "oncogenic",
             "onco_rule", "gene_role", "ccf", "clonality", "mut_qc_pass"]].rename(columns={"Hugo_Symbol": "gene"}),
        cna_s.assign(variant=cna_s.value.map({2: "AMP", -2: "HOMDEL"}))[
            ["SAMPLE_ID", "gene", "alt_type", "variant", "oncogenic", "onco_rule"]],
        sv_long.assign(alt_type="sv", variant=sv_long.partner.fillna("intergenic").radd("partner:"))[
            ["SAMPLE_ID", "gene", "alt_type", "variant", "oncogenic", "onco_rule"]],
        loh.assign(alt_type="loh", variant=loh.tcn.astype(str).radd("LOH tcn="), oncogenic=True,
                   onco_rule="loh_tsg")[["SAMPLE_ID", "gene", "alt_type", "variant", "oncogenic", "onco_rule"]],
    ], ignore_index=True)
    long["oncogenicity_source"] = RULESET
    dtype_of = {"mut": "mutations", "amp": "gistic", "del": "gistic", "loh": "gistic", "sv": "structural_variants"}
    panel_of = {dt: pm[dt] for dt in ["mutations", "gistic", "structural_variants"]}
    long["panel"] = [panel_of[dtype_of[t]][s] for s, t in zip(long.SAMPLE_ID, long.alt_type)]
    long["on_panel"] = [g in pg[p] for g, p in zip(long.gene, long.panel)]
    long.to_parquet(L.DERIVED / "alterations_long.parquet", index=False)

    # ---- wide matrix with panel masking -------------------------------------
    onc = long[long.oncogenic & long.on_panel]
    onc = onc.assign(feature=onc.gene + "_" + onc.alt_type)
    wide = pd.crosstab(onc.SAMPLE_ID, onc.feature).clip(upper=1).reindex(sids, fill_value=0).astype("float32")
    # Clonality per mutated gene (mutually exclusive, sum = <gene>_mut).
    om = onc[onc.alt_type == "mut"]
    rank = om.clonality.map({"clonal": 0, "subclonal": 1, "unknown": 2})
    best = om.assign(r=rank).sort_values("r").drop_duplicates(["SAMPLE_ID", "gene"])
    for k in ["clonal", "subclonal", "unknown"]:
        b = best[best.clonality == k]
        x = pd.crosstab(b.SAMPLE_ID, b.gene + f"_mut_{'clonality_unknown' if k == 'unknown' else k}")
        wide = wide.join(x.clip(upper=1).astype("float32"), how="left")
    maxccf = om.groupby(["SAMPLE_ID", "gene"]).ccf.max().unstack()
    maxccf.columns = [f"{g}_mut_max_ccf" for g in maxccf.columns]
    wide = wide.join(maxccf.astype("float32"), how="left")
    clon_cols = [c for c in wide.columns if re.search(r"_mut_(clonal|subclonal|clonality_unknown)$", c)]
    wide[clon_cols] = wide[clon_cols].fillna(0)

    # LOH is only measurable where FACETS passed and the gene has an ASCN call.
    loh_cols = [c for c in wide.columns if c.endswith("_loh")]
    measured = set(zip(asc_measured.SAMPLE_ID, asc_measured.gene))
    for c in loh_cols:
        g = c[:-4]
        ok = np.array([(s, g) in measured for s in wide.index])
        wide.loc[~ok, c] = np.nan

    # Variant-level target features.
    vf = variant_features(sids, m, m[m.oncogenic], cna_s, sv_s)
    # Variant-level definitions win where names coincide (AR_amp, ERBB2_amp, AR_sv, ...).
    wide = wide.drop(columns=[c for c in vf.columns if c in wide.columns]).join(vf)

    # Panel mask: NA where the gene is off the sample's panel for that data type.
    feat_gene, feat_dtype = {}, {}
    for c in wide.columns:
        if c in vf.columns:
            feat_gene[c], feat_dtype[c] = vf.attrs["gene"][c], vf.attrs["dtype"][c]
            continue
        g, t = re.match(r"^(.+?)_(mut|amp|del|sv|loh)(?:_.*)?$", c).groups()
        feat_gene[c], feat_dtype[c] = g, dtype_of[t]
    for dt in ["mutations", "gistic", "structural_variants"]:
        for p in PANELS:
            rows = wide.index[pm[dt].reindex(wide.index).eq(p)]
            off = [c for c in wide.columns if feat_dtype[c] == dt and feat_gene[c] not in pg[p]]
            if len(rows) and off:
                wide.loc[rows, off] = np.nan

    # ---- checklist 8 and 10 --------------------------------------------------
    viol = 0
    for dt in ["mutations", "gistic", "structural_variants"]:
        for p in PANELS:
            rows = wide.index[pm[dt].reindex(wide.index).eq(p)]
            off = [c for c in wide.columns if feat_dtype[c] == dt and feat_gene[c] not in pg[p]]
            viol += int(wide.loc[rows, off].notna().to_numpy().sum()) if len(rows) and off else 0
    C.check(viol == 0, "8. no alteration feature is non-missing where the gene is off the sample's panel")
    cl = long[(long.alt_type == "mut") & long.clonality.isin(["clonal", "subclonal"])]
    C.check(cl.mut_qc_pass.all() and long.loc[long.alt_type == "mut", "ccf"].where(~long.mut_qc_pass).isna().all(),
            "10. clonality / CCF use only mutations with ASCN.FACETS_QC_PASSED == True")

    wide.to_parquet(L.DERIVED / "alteration_matrix.parquet")
    pd.DataFrame({"feature": list(feat_gene), "gene": list(feat_gene.values()),
                  "data_type": [feat_dtype[c] for c in feat_gene],
                  "variant_level": [c in vf.columns for c in feat_gene],
                  "on_IMPACT341": [feat_gene[c] in pg["IMPACT341"] for c in feat_gene]}
                 ).to_csv(L.DERIVED / "alteration_features.csv", index=False)

    # Collapsed gene-level matrix (D3 alternative; S9).
    base_cols = [c for c in wide.columns if re.search(r"_(mut|amp|del|sv)$", c) and c not in vf.columns]
    genes = sorted({feat_gene[c] for c in base_cols})
    coll = pd.DataFrame({f"{g}_any": wide[[c for c in base_cols if feat_gene[c] == g]].max(axis=1, skipna=True)
                         for g in genes}, index=wide.index)
    coll.to_parquet(L.DERIVED / "alteration_matrix_gene.parquet")

    # ---- Stage 6: sample-level genomic features -----------------------------
    seg = L.load_seg()
    gf = pd.DataFrame(index=sids)
    i = imp.reindex(sids)
    gf["TMB_SCORE"] = i.TMB_SCORE
    gf["tmb_log1p"] = np.log1p(i.TMB_SCORE)
    gf["FGA"] = fga_from_seg(seg, sids, F["fga_abs_seg_mean"]).reindex(sids)
    gf["MSI_SCORE"] = i.MSI_SCORE.where(i.MSI_TYPE.isin(["Stable", "Instable"]))
    gf["MSI_TYPE"] = i.MSI_TYPE.where(i.MSI_TYPE.isin(["Stable", "Instable"]))
    gf["FACETS_PURITY"] = i.FACETS_PURITY.where(i.FACETS_QC == "Pass")
    gf["FACETS_PLOIDY"] = i.FACETS_PLOIDY.where(i.FACETS_QC == "Pass")
    gf["FACETS_WGD"] = i.FACETS_WGD.where(i.FACETS_QC == "Pass").map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    n_onc = m[m.oncogenic].groupby("SAMPLE_ID").size().reindex(sids, fill_value=0)
    gf["n_mutations_oncogenic"] = n_onc
    gf["n_mutations_oncogenic_log1p"] = np.log1p(n_onc)
    gf["n_mutations_total"] = mut[mut.SAMPLE_ID.isin(sids)].groupby("SAMPLE_ID").size().reindex(sids, fill_value=0)
    gf.to_parquet(L.DERIVED / "genomic_features.parquet")

    # FGA cross-check against MSK-MET (checklist 12).
    met_dir = L.MET if L.MET.exists() else L.RAW / "msk_met"
    if met_dir.exists():
        ms = L.read_clinical(met_dir / "data_clinical_sample.txt")[["SAMPLE_ID", "FGA"]].dropna()
        ov = gf[["FGA"]].join(ms.set_index("SAMPLE_ID").FGA.rename("FGA_met"), how="inner").dropna()
        all_ov = fga_from_seg(seg, pd.Index(ms.SAMPLE_ID), F["fga_abs_seg_mean"]).to_frame() \
            .join(ms.set_index("SAMPLE_ID").FGA.rename("FGA_met"), how="inner").dropna()
        r_coh = ov.FGA.corr(ov.FGA_met) if len(ov) > 2 else np.nan
        r_all = all_ov.FGA.corr(all_ov.FGA_met)
        fga_msg = (f"FGA vs MSK-MET: r = {r_coh:.3f} on {len(ov):,} cohort samples, "
                   f"r = {r_all:.3f} on all {len(all_ov):,} overlapping samples")
        print("  " + fga_msg)
        C.check(r_all > 0.95, f"12. computed FGA correlates r > 0.95 with MSK-MET FGA ({fga_msg})")
    else:
        fga_msg = "MSK-MET absent: FGA cross-check skipped (checklist 12 logged, not run)"
        print("  WARNING " + fga_msg)

    # ---- Signature checks (9.7) ---------------------------------------------
    sig_rows = signature_checks(samples, gf)

    # ---- T4: controls after oncogenicity filter and panel mask --------------
    ctrl = controls_after(samples, m, cna_s, wide)
    old = pd.read_csv(L.DERIVED / "controls.csv")
    old = old[~old.stage.isin(ctrl.stage.unique())]
    pd.concat([old, ctrl], ignore_index=True).to_csv(L.DERIVED / "controls.csv", index=False)
    for _, r in ctrl[ctrl.stage == "panel_mask"].iterrows():
        if r.expected == "up":
            C.check(r.pct_post > 2 * r.pct_pre and r.pct_post - r.pct_pre > 1,
                    f"14. {r.control} ({r.cohort}) enriched post after onco filter + panel mask "
                    f"({r.pct_pre:.1f}% -> {r.pct_post:.1f}%)")
        else:
            C.check(abs(r.pct_post - r.pct_pre) < 5,
                    f"14. {r.control} ({r.cohort}) not enriched after onco filter + panel mask "
                    f"({r.pct_pre:.1f}% -> {r.pct_post:.1f}%)")

    # ---- log -----------------------------------------------------------------
    known = ["TP53", "PTEN", "RB1", "APC", "CDKN2A", "KRAS", "PIK3CA", "EGFR", "ESR1", "AR", "MET",
             "ERBB2", "MYC", "CCND1", "BRAF", "SMAD4", "ARID1A", "BRCA2", "FGFR1", "MDM2"]
    nonsyn = mut[mut.SAMPLE_ID.isin(sids) & mut.Variant_Classification.isin(NONSYN)]
    C.write_log("build_alterations", f"""
Stages 5-6 (build_alterations.py). Oncogenicity source: **{RULESET}** (rule-based fallback;
data/raw/oncokb/ is empty). Rules derived from all {mut.SAMPLE_ID.nunique():,} MSK-IMPACT 50K
samples minus {len(hyper_all):,} hypermutated ones: {len(rules['hot_mis']):,} missense hotspot residues
(>= {HOTSPOT_MIN_SAMPLES} samples and >= {HOTSPOT_FOLD_OVER_UNIFORM}x gene background),
{len(rules['hot_inf']):,} in-frame indel hotspots, gene roles by the 20/20 rule
({(roles.role == 'TSG').sum()} TSG, {(roles.role == 'oncogene').sum()} oncogene,
{(roles.role == 'TSG_cna').sum()} TSG by deep deletion, {(roles.role == 'oncogene_cna').sum()} oncogene by amplification).
For the {len(sids):,} cohort samples: {len(nonsyn):,} nonsynonymous mutations in, {int(m.oncogenic.sum()):,}
oncogenic; {int(cna_s.oncogenic.sum()):,} of {len(cna_s):,} amp/homdel calls kept; {int(sv_s.oncogenic.sum()):,} of
{len(sv_s):,} SVs kept; {len(loh):,} TSG LOH calls. Legacy gene symbols mapped via config/gene_aliases.csv.
Matrix: {wide.shape[0]:,} samples x {wide.shape[1]:,} features (NA off-panel). {fga_msg}.
""", {"Gene roles for reference genes": roles.reindex(known)[["n", "trunc", "hot", "n_amp", "n_homdel", "role"]],
      "Oncogenic calls by rule": long[long.oncogenic].groupby(["alt_type", "onco_rule"]).size().to_frame("n"),
      "Controls (T4 stages)": ctrl.set_index(["cohort", "control", "stage"])[["n_pre", "n_post", "pct_pre", "pct_post"]],
      "Signature checks": pd.DataFrame(sig_rows).set_index(["cohort", "signature"])})
    print(f"wrote alteration matrix {wide.shape}")


def controls_after(samples: pd.DataFrame, m: pd.DataFrame, cna_s: pd.DataFrame,
                   wide: pd.DataFrame) -> pd.DataFrame:
    """Section 2.2 controls on the final relaxed cohort after each stage."""
    s = samples
    def onc_mut(g, hg=None):
        x = m[(m.Hugo_Symbol == g) & m.oncogenic]
        return set(x[x.HGVSp_Short == hg].SAMPLE_ID) if hg else set(x.SAMPLE_ID)
    def onc_cna(g, v):
        return set(cna_s[(cna_s.gene == g) & (cna_s.value == v) & cna_s.oncogenic].SAMPLE_ID)
    defs = [("breast", "ESR1 mutation", onc_mut("ESR1"), "ESR1_mut", None),
            ("breast", "ESR1_LBD (AI subclass)", None, "ESR1_LBD", "AI"),
            ("nsclc", "EGFR T790M", onc_mut("EGFR", "p.T790M"), "EGFR_T790M", None),
            ("nsclc", "EGFR C797S", onc_mut("EGFR", "p.C797S"), "EGFR_C797X", None),
            ("nsclc", "MET amplification", onc_cna("MET", 2), "MET_amp", None),
            ("prostate", "AR amplification", onc_cna("AR", 2), "AR_amp", None),
            ("prostate", "AR mutation", onc_mut("AR"), "AR_mut", None),
            ("crc", "KRAS mutation", onc_mut("KRAS"), "KRAS_mut", None)]
    from build_cohort import PLAN_2_2
    rows = []
    for cohort, name, hit, feat, sub in defs:
        d = s[s.cohort == cohort]
        if sub:
            d = d[d.subclass == sub]
        exp = PLAN_2_2.get((cohort, name.split(" (")[0]), (np.nan, np.nan, "up"))
        for stage in ["onco_filter", "panel_mask"]:
            if stage == "onco_filter" and hit is None:
                continue
            if stage == "onco_filter":
                a = d.SAMPLE_ID.isin(hit)
                pre, post = a[d.label_A == 0], a[d.label_A == 1]
            else:
                v = wide.reindex(d.SAMPLE_ID)[feat] if feat in wide.columns else pd.Series(0.0, index=d.SAMPLE_ID)
                v = pd.Series(v.to_numpy(), index=d.index)
                pre, post = v[d.label_A == 0].dropna(), v[d.label_A == 1].dropna()
            rows.append({"cohort": cohort, "control": name, "stage": stage,
                         "n_pre": len(pre), "n_post": len(post),
                         "pct_pre": 100 * pre.mean(), "pct_post": 100 * post.mean(),
                         "plan_pct_pre": exp[0], "plan_pct_post": exp[1], "expected": exp[2]})
    return pd.DataFrame(rows)


def signature_checks(samples: pd.DataFrame, gf: pd.DataFrame) -> list[dict]:
    """COSMIC v2 signature sanity checks (9.7): never features."""
    F = C.pipeline_cfg()["filters"]
    path = L.IMPACT / "data_mutational_signatures_contribution_v2.txt"
    header = pd.read_csv(path, sep="\t", nrows=0).columns
    keep = [c for c in header if c in set(samples.SAMPLE_ID)]
    sig = pd.read_csv(path, sep="\t", usecols=["NAME", *keep]).set_index("NAME").T
    ok = gf.index[gf.n_mutations_total >= F["signature_min_mutations"]]
    rows = []
    for cohort, cols, label in [("crc", ["Signature17 (Unknown)"], "Signature 17 (5-FU)"),
                                ("breast", ["Signature2 (APOBEC)", "Signature13 (APOBEC)"], "APOBEC (2+13)")]:
        d = samples[(samples.cohort == cohort) & samples.SAMPLE_ID.isin(ok) & samples.SAMPLE_ID.isin(sig.index)]
        v = sig.loc[d.SAMPLE_ID, cols].sum(axis=1).to_numpy()
        lab = d.label_A.to_numpy()
        rows.append({"cohort": cohort, "signature": label, "n_pre": int((lab == 0).sum()),
                     "n_post": int((lab == 1).sum()), "mean_pre": float(v[lab == 0].mean()),
                     "mean_post": float(v[lab == 1].mean()),
                     "pct_dominant_pre": float(100 * (v[lab == 0] > 0.2).mean()),
                     "pct_dominant_post": float(100 * (v[lab == 1] > 0.2).mean())})
    pd.DataFrame(rows).to_csv(L.DERIVED / "signature_checks.csv", index=False)
    for r in rows:
        print(f"  signature check {r['cohort']} {r['signature']}: mean {r['mean_pre']:.3f} pre -> "
              f"{r['mean_post']:.3f} post (n={r['n_pre']}/{r['n_post']}, >=10 mutations)")
    return rows


if __name__ == "__main__":
    build()
