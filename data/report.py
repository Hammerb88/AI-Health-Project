"""Section 15 summary tables and figures for the cleaning milestones (M1-M2).

Writes to data/reports/ and builds reports/README.md embedding them in order.
Model-dependent outputs (T6-T8, F3, F7-F12) belong to milestone M3 and are
listed as pending in the README until the modelling stage exists.

Run: ``python report.py``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common as C
import loaders as L
import viz

OUT = C.REPORTS
PRE, POST = viz.CAT[0], viz.CAT[1]          # fixed: pre = slot 1, post = slot 2 in every figure
NAMES = {c: C.config("drug_classes")["cohorts"][c]["label"] for c in C.COHORTS}


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def smd(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    s = np.sqrt((a.var() + b.var()) / 2)
    return np.nan if not s or np.isnan(s) else (b.mean() - a.mean()) / s


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def t1_attrition() -> pd.DataFrame:
    a = pd.read_csv(L.DERIVED / "attrition.csv")
    a = a[~a.description.str.startswith("[")]               # main label variant only
    t = a.pivot_table(index=["step", "description"], columns="cohort", values="samples", aggfunc="first",
                      sort=False)[C.COHORTS]
    post = a.pivot_table(index=["step", "description"], columns="cohort", values="label_1", aggfunc="first",
                         sort=False).reindex(columns=C.COHORTS)
    t = t.astype("Int64").astype(str) + post.reindex(t.index).map(
        lambda v: "" if pd.isna(v) else f" ({int(v):,} post)")
    t.columns = [NAMES[c] for c in t.columns]
    return t.reset_index()


def t2_characteristics(s: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = s[s.main_analysis]
    rows, smds = [], []
    spec = [("n", None), ("age_at_collection", "mean"), ("sex_male", "pct"), ("metastatic_at_collection", "pct"),
            ("adjuvant_start", "pct"), ("archived_tissue", "pct"), ("FACETS_PURITY", "median"),
            ("tmb_log1p", "median"), ("FGA", "median"), ("n_prior_classes", "mean"),
            ("sample_type=Metastasis", "pct"), ("biopsy_organ_coarse=primary_organ", "pct"),
            ("biopsy_organ_coarse=liver", "pct"), ("biopsy_organ_coarse=bone", "pct"),
            ("panel_version=IMPACT341/410", "pct")]
    for cohort in C.COHORTS:
        d = m[m.cohort == cohort]
        for var, how in spec:
            if var == "sex_male" and cohort in ("breast", "prostate"):
                continue
            if "=" in var:
                k, lv = var.split("=")
                x = d[k].isin(lv.split("/")) if k == "panel_version" else (d[k] == lv)
                x = x.astype(float)
            elif var == "n":
                x = None
            else:
                x = d[var].astype(float)
            r = {"cohort": NAMES[cohort], "variable": var}
            for lab, nm in [(0, "pre"), (1, "post")]:
                if x is None:
                    r[nm] = f"{int((d.label_A == lab).sum()):,}"
                    continue
                v = x[d.label_A == lab]
                r[nm] = (f"{100 * v.mean():.1f}%" if how == "pct" else
                         f"{v.median():.2f}" if how == "median" else f"{v.mean():.2f}")
            if x is not None:
                r["SMD"] = round(smd(x[d.label_A == 0], x[d.label_A == 1]), 2)
                smds.append({"cohort": cohort, "variable": var, "smd": r["SMD"]})
            rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(smds)


def t3_exposure(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in C.COHORTS:
        d = s[s.main_analysis & (s.cohort == cohort)]
        p = d[d.label_A == 1]
        between = pd.Series([c for lst in p.classes_between for c in lst]).value_counts()
        rows.append({"cohort": NAMES[cohort], "post n": len(p),
                     "t_days median [IQR]": f"{p.t_days.median():.0f} [{p.t_days.quantile(.25):.0f}-{p.t_days.quantile(.75):.0f}]",
                     "cum_days_on_class median [IQR]": f"{p.cum_days_on_class.median():.0f} "
                                                       f"[{p.cum_days_on_class.quantile(.25):.0f}-{p.cum_days_on_class.quantile(.75):.0f}]",
                     "% on class at collection": f"{100 * p.on_class_at_collection.mean():.0f}%",
                     "% with intervening lines": f"{100 * (p.n_lines_between > 0).mean():.0f}%",
                     "most common intervening classes": ", ".join(f"{k} ({v})" for k, v in between.head(3).items()),
                     "% adjuvant start (all)": f"{100 * d.adjuvant_start.mean():.0f}%"})
    return pd.DataFrame(rows)


def t4_controls() -> pd.DataFrame:
    c = pd.read_csv(L.DERIVED / "controls.csv")
    order = ["raw_plan_denominator", "raw", "onco_filter", "panel_mask", "relaxed", "strict",
             "strict_noint", "S3_survivorship_strict", "A-on_strict"]
    c["stage"] = pd.Categorical(c.stage, order, ordered=True)
    c["pre → post"] = c.apply(lambda r: f"{r.pct_pre:.1f}% → {r.pct_post:.1f}% (n={r.n_pre:,}/{r.n_post:,})", axis=1)
    t = c.pivot_table(index=["cohort", "control"], columns="stage", values="pre → post", aggfunc="first",
                      observed=True)
    return t.reset_index()


def t5_prevalence(fd: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    rows = []
    for cohort in C.COHORTS:
        d = fd[fd.block.isin(["alteration", "variant_target"]) & fd[f"{cohort}_kept_firth"].astype("boolean").fillna(False)]
        d = d.assign(diff=d[f"{cohort}_pct_post"] - d[f"{cohort}_pct_pre"]).sort_values("diff", ascending=False)
        for _, r in pd.concat([d.head(top)]).iterrows():
            rows.append({"cohort": NAMES[cohort], "feature": r.feature, "block": r.block,
                         "% pre": round(r[f"{cohort}_pct_pre"], 1), "% post": round(r[f"{cohort}_pct_post"], 1),
                         "n altered (main)": int(r[f"{cohort}_n_altered_main"])})
    kept = pd.DataFrame({NAMES[c]: [int(fd[f"{c}_kept_firth"].astype("boolean").fillna(False).sum()),
                                    int(fd[f"{c}_main_analysis_feature"].astype("boolean").fillna(False).sum())]
                         for c in C.COHORTS}, index=["kept by prevalence filter (Firth list)", "main-analysis features"])
    return pd.DataFrame(rows), kept


def t10_mappings() -> dict[str, pd.DataFrame]:
    import exposure as E
    tx = E.treatment()
    agents = tx.groupby("cls").agg(agents=("AGENT", "nunique"), rows=("AGENT", "size")).sort_values("rows", ascending=False)
    site = pd.read_csv(C.CONFIG / "site_map.csv").groupby("organ_group").agg(strings=("site", "size"),
                                                                             samples=("n_samples", "sum"))
    pg = {p: len(v) for p, v in __import__("build_alterations").panel_genes().items()}
    meta = pd.DataFrame({"value": [C.pipeline_cfg()["oncogenicity_source"] + " (rules_v1: hotspot residues + "
                                   "truncating-in-TSG + TERT promoter + MET ex14; CNA by amp/homdel dominance; "
                                   "fusions with driver partners)",
                                   ", ".join(f"{k}: {v} genes" for k, v in pg.items()),
                                   "config/gene_aliases.csv (37 legacy symbols)"]},
                        index=["oncogenicity source", "panel gene lists (cBioPortal API)", "gene symbol aliases"])
    return {"Agent classes": agents, "Biopsy site → organ group": site, "Annotation sources": meta}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def f1_attrition() -> None:
    a = pd.read_csv(L.DERIVED / "attrition.csv")
    a = a[~a.description.str.startswith("[") & ~a.step.isin(["0", "4L", "5c"])]
    labels = {"1": "Cancer type matches", "2": "Received the class", "3": "In MSK-IMPACT 50K",
              "4": "Cohort rule (HR+ / EGFR)", "5": "Exclude 0 < t ≤ 90 d", "5b": "Drop pre-tx w/ prior meds",
              "6": "Strict tier", "6b": "Strict, no intervening", "7": "Relaxed + FACETS pass"}
    steps = list(labels)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True)
    for ax, cohort in zip(axes, C.COHORTS):
        d = a[a.cohort == cohort].set_index("step").reindex(steps)
        y = np.arange(len(steps))[::-1]
        bars = ax.barh(y, d.samples, height=0.62, color=viz.SEQ_STEPS[3])
        ax.set_xlim(0, d.samples.max() * 1.32)
        viz.label_bars(ax, bars, d.samples)
        ax.set_yticks(y, [labels[s] for s in steps])
        ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
        viz.finish(ax, title=NAMES[cohort], xlabel="samples")
    fig.suptitle("F1 · Cohort attrition (samples remaining after each step)", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold", color=viz.INK)
    fig.tight_layout()
    fig.savefig(OUT / "F1_attrition.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def f2_timing(s: pd.DataFrame) -> None:
    base = pd.read_parquet(L.DERIVED / "samples.parquet")
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), gridspec_kw={"height_ratios": [1.3, 1]})
    for i, cohort in enumerate(C.COHORTS):
        d = s[s.main_analysis & (s.cohort == cohort)]
        ax = axes[0, i]
        bins = np.arange(-3650, 3651, 120)
        t = d.t_days.clip(-3650, 3650)
        ax.hist(t[d.label_A == 0], bins=bins, color=PRE, label="pre (t ≤ 0)")
        ax.hist(t[d.label_A == 1], bins=bins, color=POST, label="post (t > 90 d)")
        ax.axvspan(0, 90, color=viz.GRID, zorder=0, label="excluded 0-90 d")
        viz.finish(ax, title=NAMES[cohort], xlabel="t_days (collection − class start, clipped ±10 y)",
                   ylabel="samples" if i == 0 else None)
        if i == 0:
            ax.legend(loc="upper left")
        ax2 = axes[1, i]
        p = d[d.label_A == 1]
        ax2.hist(p.cum_days_on_class / 365.25, bins=np.arange(0, 10.01, 0.25), color=POST)
        viz.finish(ax2, xlabel="cumulative years on class at collection (post only)",
                   ylabel="samples" if i == 0 else None)
    fig.suptitle("F2 · Sampling time relative to first dose (main analysis)", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold", color=viz.INK)
    fig.tight_layout()
    fig.savefig(OUT / "F2_sampling_time.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def f4_balance(smds: pd.DataFrame) -> None:
    smds = smds.dropna(subset=["smd"])        # n_prior_classes is 0 by definition in the strict tier
    variables = list(dict.fromkeys(smds.variable))
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    y = {v: i for i, v in enumerate(variables[::-1])}
    for k, cohort in enumerate(C.COHORTS):
        d = smds[smds.cohort == cohort]
        ax.scatter(d.smd, [y[v] + (k - 1.5) * 0.15 for v in d.variable], s=36, color=viz.CAT[k],
                   label=NAMES[cohort], edgecolor=viz.SURFACE, linewidth=1.5, zorder=3)
    ax.axvline(0, color=viz.AXIS, lw=1)
    for x in (-0.1, 0.1):
        ax.axvline(x, color=viz.GRID, lw=1, zorder=0)
    ax.set_yticks(list(y.values()), list(y.keys()))
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.legend(loc="lower right")
    viz.finish(ax, title="F4 · Covariate balance, post vs pre (main analysis)",
               subtitle="standardized mean difference; |SMD| > 0.1 (gray lines) marks imbalance",
               xlabel="SMD (post − pre)")
    fig.tight_layout()
    fig.savefig(OUT / "F4_covariate_balance.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def f5_panels(s: pd.DataFrame) -> None:
    panels = ["IMPACT341", "IMPACT410", "IMPACT468", "IMPACT505"]
    colors = viz.SEQ_STEPS[0::2]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    for ax, cohort in zip(axes, C.COHORTS):
        d = s[s.main_analysis & (s.cohort == cohort)]
        tab = pd.crosstab(d.label_A, d.panel_version, normalize="index").reindex(columns=panels, fill_value=0) * 100
        left = np.zeros(len(tab))
        for p, col in zip(panels, colors):
            ax.barh(["pre", "post"], tab[p].values, left=left, color=col, height=0.6,
                    edgecolor=viz.SURFACE, linewidth=2, label=p)
            left += tab[p].values
        ax.set_xlim(0, 100)
        ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
        viz.finish(ax, title=NAMES[cohort], xlabel="% of samples")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("F5 · Panel version by label (older panels are over-represented post-treatment)",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold", color=viz.INK)
    fig.tight_layout()
    fig.savefig(OUT / "F5_panel_confounding.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def f6_controls() -> None:
    c = pd.read_csv(L.DERIVED / "controls.csv")
    c = c[c.stage == "strict"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    y = np.arange(len(c))[::-1]
    for yi, (_, r) in zip(y, c.iterrows()):
        lo0, hi0 = wilson(r.pct_pre * r.n_pre / 100, r.n_pre)
        lo1, hi1 = wilson(r.pct_post * r.n_post / 100, r.n_post)
        ax.plot([r.pct_pre, r.pct_post], [yi, yi], color=viz.GRID, lw=2, zorder=1)
        ax.errorbar(r.pct_pre, yi + 0.12, xerr=[[r.pct_pre - lo0], [hi0 - r.pct_pre]], fmt="o", color=PRE,
                    ms=7, elinewidth=1.5, capsize=0, zorder=3, label="pre" if yi == y[0] else None)
        ax.errorbar(r.pct_post, yi - 0.12, xerr=[[r.pct_post - lo1], [hi1 - r.pct_post]], fmt="o", color=POST,
                    ms=7, elinewidth=1.5, capsize=0, zorder=3, label="post" if yi == y[0] else None)
    ax.set_yticks(y, [f"{NAMES[r.cohort].split(' /')[0]} · {r.control}" for _, r in c.iterrows()])
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.legend(loc="center right")
    viz.finish(ax, title="F6 · Positive / negative controls, strict tier",
               subtitle="% altered with 95% Wilson CI; KRAS (CRC) is the negative control",
               xlabel="% of samples altered")
    fig.tight_layout()
    fig.savefig(OUT / "F6_controls.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------

def main() -> None:
    viz.apply_theme()
    OUT.mkdir(exist_ok=True)
    s = pd.concat([pd.read_parquet(L.DERIVED / f"model_table_{c}.parquet") for c in C.COHORTS], ignore_index=True)
    samp = pd.read_parquet(L.DERIVED / "samples.parquet")
    s = s.merge(samp[["SAMPLE_ID", "cohort", "subclass", "panel_version", "classes_between",
                      "cum_days_on_class", "on_class_at_collection", "n_lines_between"]],
                on=["SAMPLE_ID", "cohort"], how="left", suffixes=("", "_s"))
    for c in ["on_class_at_collection", "n_lines_between"]:
        if f"{c}_s" in s:
            s[c] = s[f"{c}_s"]
    fd = pd.read_csv(L.DERIVED / "feature_dictionary.csv")

    t1 = t1_attrition()
    t2, smds = t2_characteristics(s)
    t3 = t3_exposure(s)
    t4 = t4_controls()
    t5, t5k = t5_prevalence(fd)
    t10 = t10_mappings()
    for name, t in [("T1_attrition", t1), ("T2_characteristics", t2), ("T3_exposure", t3),
                    ("T4_controls", t4), ("T5_prevalence_top", t5)]:
        t.to_csv(OUT / f"{name}.csv", index=False)
    fd.to_csv(OUT / "T9_feature_dictionary.csv", index=False)

    f1_attrition(); f2_timing(s); f4_balance(smds); f5_panels(s); f6_controls()

    md = C.md_table
    sig = pd.read_csv(L.DERIVED / "signature_checks.csv")
    repro = pd.read_csv(L.DERIVED / "plan_reproduction.csv")
    blocks = fd.block.value_counts().to_frame("features")
    readme = f"""# Data cleaning report (data_cleaning_plan.md v3, milestones M1-M2)

Generated by `data/report.py`. All counts come from the pipeline outputs in `data/derived/`.
Pre = sample collected on or before the first dose of the cohort's drug class (label 0);
post = collected more than 90 days after it (label 1). Main analysis = strict tier minus
hypermutated samples.

## T1 · Cohort attrition

{md(t1, index=False)}

![F1](F1_attrition.png)

### Section 2.1 reproduction (loose cohort, before the MSK-IMPACT filter)

{md(repro, index=False)}

## T2 · Cohort characteristics (main analysis)

{md(t2, index=False)}

![F4](F4_covariate_balance.png)

![F5](F5_panel_confounding.png)

## T3 · Exposure summary (main analysis, post-treatment samples)

{md(t3, index=False)}

![F2](F2_sampling_time.png)

## T4 · Positive / negative controls through the pipeline

`raw_plan_denominator` reproduces the plan's Section 2.2 numbers, which divided by all loose
samples including ~22% with no MSK-IMPACT genomics. Every later stage uses profiled samples only.

{md(t4, index=False)}

![F6](F6_controls.png)

Mutational-signature checks (Section 9.7, samples with ≥ 10 mutations; not model features):

{md(sig, index=False)}

## T5 · Alteration prevalence (top 15 by post − pre difference, Firth-eligible features)

{md(t5k)}

{md(t5, index=False)}

## T9 · Feature dictionary

Full table: [T9_feature_dictionary.csv](T9_feature_dictionary.csv) ({len(fd):,} rows).

{md(blocks)}

## T10 · Mapping logs

{chr(10).join(f"### {k}{chr(10)}{chr(10)}{md(v)}{chr(10)}" for k, v in t10.items())}

## Pending (milestone M3)

T6 covariate-only baseline, T7 per-gene Firth results, T8 attribution stability, F3 per-cohort
event order, F7 volcano, F8 forest, F9 dose-response, F10 clonality shift, F11 SHAP, F12 stability.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote {OUT / 'README.md'}")


if __name__ == "__main__":
    main()
