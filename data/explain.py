"""Section 6: OOF SHAP aggregation, stability, correlated clusters, block shares,
interaction shares, stratified enrichment (site-confounding) and the control gate.

Reads derived/models/phase2_* and sens_* (train_models.py). Writes to derived/models/:
  shap_oof.parquet            long: cohort x model x sample x variable, mean SHAP over 20 repeats
  shap_ranking.csv            MT3: per cohort x model x variable
  shap_stability.csv          top-20 selection frequency per alteration feature
  shap_clusters.csv           correlated alteration clusters (phi > 0.7) with summed SHAP
  block_share.csv             % of total |SHAP| from blocks C / G / X
  interactions.csv            XGBoost interaction share with biopsy_organ / panel_version
  stratified_enrichment.csv   MT7: crude and Mantel-Haenszel OR by sample_type
  control_gate.csv            6.5 gate (main and metastasis-only)

Run: ``python explain.py``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

import common as C
import m3_common as M

OUT = M.MODELS_DIR
TOP = 20
ROBUST = 0.70
PHI = 0.70


def selected() -> dict[str, str]:
    return pd.read_csv(OUT / "selection.csv").set_index("cohort").selected.to_dict()


def per_sample(cohort: str, model: str) -> tuple[pd.DataFrame, pd.Series]:
    sh = pd.read_parquet(OUT / "shap" / f"phase2_{cohort}_A_{model}_F2_main.parquet")
    v = [c for c in sh.columns if c not in ("SAMPLE_ID", "fold", "repeat")]
    kept = sh[v].notna().groupby([sh.repeat, sh.fold]).any().mean()      # share of fold-models keeping it
    ps = sh[v].fillna(0.0).groupby(sh.SAMPLE_ID).mean()                  # filtered in a fold -> 0 (6.2)
    n_rep = sh.groupby("SAMPLE_ID").repeat.nunique()
    assert (n_rep == 20).all(), f"[{cohort}/{model}] every sample has one SHAP vector per repeat"
    return ps, kept


def stability(cohort: str, model: str, stage="phase2", variant="main") -> pd.DataFrame:
    s = pd.read_parquet(OUT / f"{stage}_shap_sum.parquet")
    s = s[(s.cohort == cohort) & (s.model == model) & (s.variant == variant) & (s.fs == "F2") & (s.block == "X")]
    s = s.assign(rank=s.groupby(["repeat", "fold"]).mean_abs.rank(ascending=False, method="first"))
    n_models = s.groupby(["repeat", "fold"]).ngroups
    freq = (s[s["rank"] <= TOP].groupby("feature").size() / n_models).rename("top20_freq")
    return freq.to_frame().assign(n_fold_models=n_models)


def clusters(cohort: str) -> dict[str, int]:
    d, _, _ = M.select_rows(cohort, "main")
    x = [c for c in M.blocks(cohort)["X"] if d[c].std() > 0]
    phi = np.corrcoef(d[x].to_numpy(float), rowvar=False)
    adj = (np.abs(phi) > PHI) & ~np.eye(len(x), dtype=bool)
    n, lab = connected_components(adj, directed=False)
    sizes = np.bincount(lab)
    return {f: int(lab[i]) for i, f in enumerate(x) if sizes[lab[i]] > 1}


def mh_or(a, b, c, d):
    """Mantel-Haenszel OR across strata with 0.5 added to every cell of each stratum."""
    a, b, c, d = (np.asarray(v, float) + 0.5 for v in (a, b, c, d))
    n = a + b + c + d
    return np.sum(a * d / n) / np.sum(b * c / n)


def stratified_enrichment(cohort: str) -> pd.DataFrame:
    d, _, y = M.select_rows(cohort, "main")
    st = d.sample_type.where(d.sample_type.isin(["Primary", "Metastasis"]), "Other")
    rows = []
    for f in M.blocks(cohort)["X"]:
        x = d[f].to_numpy() == 1
        if x.sum() == 0:
            continue
        r = {"cohort": cohort, "feature": f}
        cells = []
        for s_name, m in [("all", np.ones(len(d), bool)), ("met", (st == "Metastasis").to_numpy()),
                          ("prim", (st == "Primary").to_numpy())]:
            yy, xx = y[m], x[m]
            r[f"pct_pre_{s_name}"] = 100 * xx[yy == 0].mean() if (yy == 0).any() else np.nan
            r[f"pct_post_{s_name}"] = 100 * xx[yy == 1].mean() if (yy == 1).any() else np.nan
            a_, b_ = np.sum(xx & (yy == 1)), np.sum(xx & (yy == 0))
            c_, d_ = np.sum(~xx & (yy == 1)), np.sum(~xx & (yy == 0))
            r[f"log_or_{s_name}"] = np.log(mh_or([a_], [b_], [c_], [d_]))
        for s_name in st.unique():
            m = (st == s_name).to_numpy()
            yy, xx = y[m], x[m]
            if len(np.unique(yy)) == 2:
                cells.append([np.sum(xx & (yy == 1)), np.sum(xx & (yy == 0)), np.sum(~xx & (yy == 1)),
                              np.sum(~xx & (yy == 0))])
        r["log_or_mh"] = np.log(mh_or(*np.array(cells).T))
        crude, mh = r["log_or_all"], r["log_or_mh"]
        if abs(crude) < 0.1:
            flag = False
        else:
            flag = bool(np.sign(mh) != np.sign(crude) or abs(mh) < 0.5 * abs(crude)
                        or np.sign(r["log_or_met"]) != np.sign(crude))
        r["site_confounded"] = flag
        r["n_altered"] = int(x.sum())
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    sel = selected()
    t_long, ranking, stab_all, clus_rows, share_rows, gate_rows = [], [], [], [], [], []
    strat = pd.concat([stratified_enrichment(c) for c in M.COHORTS], ignore_index=True)
    inter = pd.read_parquet(OUT / "phase2_inter.parquet")
    inter = (inter.groupby(["cohort", "feature", "with"])[["sum_abs_inter", "sum_abs_shap"]].sum()
             .assign(share=lambda x: x.sum_abs_inter / x.sum_abs_shap.replace(0, np.nan)).reset_index())
    ssum = pd.read_parquet(OUT / "phase2_shap_sum.parquet")
    for cohort in M.COHORTS:
        d, _, _ = M.select_rows(cohort, "main")
        dd = d.set_index("SAMPLE_ID")
        cl = clusters(cohort)
        for model in dict.fromkeys([sel[cohort], "lr"]):
            ps, kept = per_sample(cohort, model)
            blk = ssum[(ssum.cohort == cohort) & (ssum.model == model) & (ssum.fs == "F2")] \
                .groupby("feature").block.first()
            mean_abs = ps.abs().mean()
            alt = [v for v in ps.columns if blk.get(v) == "X"]
            dv = dd.reindex(ps.index)
            mean_alt = pd.Series({v: ps.loc[(dv[v] == 1).to_numpy(), v].mean() for v in alt})
            stab = stability(cohort, model)
            r = pd.DataFrame({"mean_abs_shap": mean_abs, "block": blk.reindex(mean_abs.index),
                              "mean_shap_altered": mean_alt.reindex(mean_abs.index),
                              "kept_share": kept.reindex(mean_abs.index)})
            r = r.join(stab.top20_freq).assign(cohort=cohort, model=model, selected=model == sel[cohort])
            r["top20_freq"] = r.top20_freq.where(r.block != "X", r.top20_freq.fillna(0.0))
            r["robust"] = (r.block == "X") & (r.top20_freq >= ROBUST)
            r["cluster"] = [cl.get(f) for f in r.index]
            r["rank_alt"] = r.mean_abs_shap.where(r.block == "X").rank(ascending=False, method="first")
            r = r.rename_axis("feature").reset_index()
            sc = strat[strat.cohort == cohort].set_index("feature")
            r["site_confounded"] = r.feature.map(sc.site_confounded)
            ii = inter[(inter.cohort == cohort) & (inter["with"] == "biopsy_organ")].set_index("feature").share
            ip = inter[(inter.cohort == cohort) & (inter["with"] == "panel_version")].set_index("feature").share
            r["organ_interaction_share"] = r.feature.map(ii)
            r["panel_interaction_share"] = r.feature.map(ip)
            ranking.append(r)
            long = ps.stack().rename("shap").reset_index().rename(columns={"level_1": "feature"})
            t_long.append(long.assign(cohort=cohort, model=model))
            # cluster SHAP (6.4)
            for k in sorted(set(cl.values())):
                mem = [f for f, v in cl.items() if v == k]
                present = [f for f in mem if f in ps.columns]
                if present:
                    clus_rows.append({"cohort": cohort, "model": model, "cluster": k, "members": ", ".join(mem),
                                      "cluster_mean_abs_shap": ps[present].sum(1).abs().mean(),
                                      "sum_member_mean_abs_shap": mean_abs[present].sum()})
            tot = r.groupby("block").mean_abs_shap.sum()
            share_rows.append({"cohort": cohort, "model": model,
                               **{f"pct_{b}": 100 * tot.get(b, 0) / tot.sum() for b in ("C", "G", "X")}})
            # Control gate (6.5), selected model only
            if model == sel[cohort]:
                gate_rows += gate(cohort, model, r)
    pd.concat(t_long, ignore_index=True).to_parquet(OUT / "shap_oof.parquet", index=False)
    rk = pd.concat(ranking, ignore_index=True)
    rk.to_csv(OUT / "shap_ranking.csv", index=False)
    rk[rk.block == "X"][["cohort", "model", "feature", "top20_freq", "robust"]].to_csv(OUT / "shap_stability.csv", index=False)
    pd.DataFrame(clus_rows).to_csv(OUT / "shap_clusters.csv", index=False)
    pd.DataFrame(share_rows).to_csv(OUT / "block_share.csv", index=False)
    inter.to_csv(OUT / "interactions.csv", index=False)
    strat.to_csv(OUT / "stratified_enrichment.csv", index=False)
    g = pd.DataFrame(gate_rows)
    g.to_csv(OUT / "control_gate.csv", index=False)
    print(g.to_string(index=False))
    gated = g[g.gated]
    for _, x in gated.iterrows():
        print(f"  [{'ok' if x.passed else 'FAIL'}] gate {x.cohort} {x.feature} ({x.variant}): rank {x.rank_alt:.0f}, "
              f"mean SHAP among altered {x.mean_shap_altered:+.3f}")
    assert gated.passed.all(), "6.5 control gate failed: stop and debug before writing figures"
    print("  [ok] 7. control gate passes")


def gate(cohort: str, model: str, r: pd.DataFrame) -> list[dict]:
    rows = []
    x = r[r.block == "X"].set_index("feature")
    top_alt = x.nsmallest(10, "rank_alt").mean_shap_altered.abs().max()
    for f in M.GATE_POS.get(cohort, []):
        ok = x.at[f, "rank_alt"] <= 10 and x.at[f, "mean_shap_altered"] > 0
        rows.append({"cohort": cohort, "model": model, "variant": "main", "feature": f, "expect": "top 10, toward post",
                     "rank_alt": x.at[f, "rank_alt"], "mean_shap_altered": x.at[f, "mean_shap_altered"],
                     "passed": bool(ok), "gated": True})
    for f in M.GATE_NEG.get(cohort, []):
        ok = x.at[f, "rank_alt"] > 10 and abs(x.at[f, "mean_shap_altered"]) < 0.1 * top_alt
        rows.append({"cohort": cohort, "model": model, "variant": "main", "feature": f,
                     "expect": "not top 10, |SHAP among altered| < 10% of top-10 max",
                     "rank_alt": x.at[f, "rank_alt"], "mean_shap_altered": x.at[f, "mean_shap_altered"],
                     "passed": bool(ok), "gated": True})
    # Metastasis-only variant (sens stage): positive controls gated, KRAS reported only.
    s = pd.read_parquet(OUT / "sens_shap_sum.parquet")
    s = s[(s.cohort == cohort) & (s.model == model) & (s.variant == "met_only") & (s.fs == "F2") & (s.block == "X")]
    agg = s.groupby("feature").apply(lambda g: pd.Series({
        "mean_abs": g.mean_abs.mean(),
        "mean_altered": np.average(g.mean_altered.fillna(0), weights=g.n_altered) if g.n_altered.sum() else np.nan}),
        include_groups=False)
    agg["rank_alt"] = agg.mean_abs.rank(ascending=False, method="first")
    for f in M.GATE_POS.get(cohort, []) + M.GATE_NEG.get(cohort, []):
        pos = f in M.GATE_POS.get(cohort, [])
        ok = (agg.at[f, "rank_alt"] <= 10 and agg.at[f, "mean_altered"] > 0) if pos else None
        rows.append({"cohort": cohort, "model": model, "variant": "met_only", "feature": f,
                     "expect": "top 10, toward post" if pos else "reported only",
                     "rank_alt": agg.at[f, "rank_alt"], "mean_shap_altered": agg.at[f, "mean_altered"],
                     "passed": bool(ok) if pos else True, "gated": pos})
    return rows


if __name__ == "__main__":
    main()
