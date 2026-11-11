"""Section 8: M3 figures (M1-M12) and tables (MT1-MT7), plus reports/models/README.md.

Requires train_models.py (all stages), explain.py and confirm_strict.py. Refuses to write
figures unless the control gate (explain.py, 6.5) passed (assert 7).

Colors (viz.py): pre = slot 1, post = slot 2; covariate-only = muted gray, logistic
regression = slot 3, random forest = slot 4, XGBoost = slot 5.

Run: ``python report_models.py``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as tdist
from sklearn.metrics import precision_recall_curve, roc_curve

import common as C
import m3_common as M
import viz

OUT = M.REPORT_DIR
MD = M.MODELS_DIR
PRE, POST = viz.CAT[0], viz.CAT[1]
MCOL = {"covonly": viz.MUTED, "prevalence": viz.AXIS, "lr": viz.CAT[2], "rf": viz.CAT[3], "xgb": viz.CAT[4],
        "enet": viz.CAT[2], "ridge_cov": viz.MUTED, "median": viz.AXIS}
BLOCK_COL = {"C": viz.CAT[5], "G": viz.CAT[6], "X": viz.CAT[7]}
BLOCK_NAME = {"C": "Covariates (sampling context)", "G": "Sample-level genomic", "X": "Gene alterations"}
AGREE_COL = {"Confirmed": viz.CAT[5], "Directional": viz.CAT[3], "Discordant": viz.CAT[7], "Site-confounded": viz.MUTED}
NAMES = {**{c: C.config("drug_classes")["cohorts"][c]["label"] for c in C.COHORTS}, "pooled": "Pooled"}
CONTROLS = {f for v in (*M.GATE_POS.values(), *M.GATE_NEG.values()) for f in v}
FS_NAME = {"F0": "F0 covariates", "F1": "F1 alterations", "F2": "F2 full", "F3": "F3 full + LOH/clonality"}


def load():
    D = {"rep": pd.read_parquet(MD / "metrics_repeats.parquet"), "fold": pd.read_parquet(MD / "metrics_folds.parquet"),
         "sel": pd.read_csv(MD / "selection.csv").set_index("cohort"), "rank": pd.read_csv(MD / "shap_ranking.csv"),
         "firth": pd.read_csv(MD / "firth_strict.csv"), "strat": pd.read_csv(MD / "stratified_enrichment.csv"),
         "share": pd.read_csv(MD / "block_share.csv"), "gate": pd.read_csv(MD / "control_gate.csv"),
         "clus": pd.read_csv(MD / "shap_clusters.csv")}
    # Target B is optional: without its outputs, the Target B figures (M11) and table (MT5) are skipped.
    D["has_B"] = (MD / "selection_targetB.csv").exists() and (D["rep"].stage == "targetB").any()
    if D["has_B"]:
        D["selB"] = pd.read_csv(MD / "selection_targetB.csv")
    return D


def tables_B(T, rep, fold):
    """MT5: Target B regression."""
    rb, fb = rep[rep.stage == "targetB"], fold[fold.stage == "targetB"]
    k = ["cohort", "target", "variant", "model"]
    s = summ(rb, fb, "spearman", k).rename(columns={"mean": "spearman", "lo": "rho_fold_lo", "hi": "rho_fold_hi"})
    for m in ("mae_days", "r2_log"):
        s = s.merge(rb.groupby(k)[m].mean().rename(m).reset_index(), on=k)
    cov = s[s.model == "ridge_cov"].set_index(["cohort", "target", "variant"]).mae_days
    s["delta_mae_vs_cov"] = s.mae_days - s.set_index(["cohort", "target", "variant"]).index.map(cov)
    n = fb.groupby(["cohort", "target", "variant"]).apply(lambda g: g.groupby(["repeat"]).n_test.sum().iloc[0],
                                                          include_groups=False).rename("n_rows")
    s = s.merge(n.reset_index(), on=["cohort", "target", "variant"])
    s["note"] = np.where(s.cohort == "nsclc", "exploratory (fixed hyperparameters)", "")
    T["MT5"] = s


def summ(rep, fold, metric, keys):
    a = rep.groupby(keys)[metric].mean().rename("mean")
    lo = fold.groupby(keys)[metric].quantile(0.025).rename("lo")
    hi = fold.groupby(keys)[metric].quantile(0.975).rename("hi")
    return pd.concat([a, lo, hi], axis=1).reset_index()


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def nb_test(diff: np.ndarray, n_train: float, n_test: float) -> float:
    """Nadeau-Bengio corrected resampled t-test (two-sided p), approximate."""
    J = len(diff)
    v = (1 / J + n_test / n_train) * np.var(diff, ddof=1)
    return float(2 * tdist.sf(abs(diff.mean()) / np.sqrt(v), J - 1)) if v > 0 else np.nan


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def tables(D):
    rep, fold, sel = D["rep"], D["fold"], D["sel"]
    T = {}
    # MT1
    r1 = rep[(rep.stage == "phase1") & (rep.eval_rows == "all")]
    f1 = fold[fold.stage == "phase1"]
    keys = ["cohort", "model", "fs"]
    s = summ(r1, f1, "auc", keys).rename(columns={"mean": "roc_auc", "lo": "auc_fold_lo", "hi": "auc_fold_hi"})
    for m in ("ap", "brier", "logloss", "cal_intercept", "cal_slope", "prevalence"):
        s = s.merge(r1.groupby(keys)[m].mean().rename(m).reset_index(), on=keys)
    s["cohort"] = pd.Categorical(s.cohort, M.COHORTS)
    T["MT1"] = s.sort_values(["cohort", "model", "fs"]).reset_index(drop=True)
    # MT2
    rows = []
    r2, f2 = rep[(rep.stage == "phase2") & (rep.variant == "main")], fold[(fold.stage == "phase2") & (fold.variant == "main")]
    for c in M.COHORTS:
        m = sel.at[c, "selected"]
        a2 = r2[(r2.cohort == c) & (r2.model == m) & (r2.fs == "F2") & (r2.eval_rows == "all")]
        a0 = r2[(r2.cohort == c) & (r2.model == m) & (r2.fs == "F0") & (r2.eval_rows == "all")]
        ac = r2[(r2.cohort == c) & (r2.model == "covonly") & (r2.eval_rows == "all")]
        st = r2[(r2.cohort == c) & (r2.model == m) & (r2.fs == "F2") & (r2.eval_rows == "strict")]
        w = f2[(f2.cohort == c) & (f2.model == m) & f2.fs.isin(["F0", "F2"])].pivot_table(
            index=["repeat", "fold"], columns="fs", values="auc")
        n = f2[(f2.cohort == c) & (f2.model == m) & (f2.fs == "F2")]
        diff = (w.F2 - w.F0).to_numpy()
        rows.append({"cohort": c, "selected": M.MODEL_LABEL[m], "auc_F2": a2.auc.mean(), "auc_F0_same_model": a0.auc.mean(),
                     "auc_covonly": ac.auc.mean(), "delta_auc": diff.mean(),
                     "delta_auc_fold_lo": np.percentile(diff, 2.5), "delta_auc_fold_hi": np.percentile(diff, 97.5),
                     "delta_auc_p_nb": nb_test(diff, n.n_train.mean(), n.n_test.mean()),
                     "pr_auc": a2.ap.mean(), "prevalence": a2.prevalence.mean(), "brier": a2.brier.mean(),
                     "cal_slope": a2.cal_slope.mean(), "auc_strict_rows": st.auc.mean(),
                     "rationale": sel.at[c, "rationale"]})
    T["MT2"] = pd.DataFrame(rows)
    # MT3
    rk = D["rank"]
    x = rk[rk.selected & (rk.block == "X")].sort_values(["cohort", "mean_abs_shap"], ascending=[True, False])
    x = x.groupby("cohort", sort=False).apply(lambda g: g[(g.rank_alt <= 30) | g.robust], include_groups=False)
    T["MT3"] = x.reset_index(level=0).reset_index(drop=True)[
        ["cohort", "feature", "rank_alt", "mean_abs_shap", "mean_shap_altered", "top20_freq", "robust", "cluster",
         "kept_share", "site_confounded", "organ_interaction_share", "panel_interaction_share"]]
    # MT4
    fi = D["firth"]
    keep = fi[(fi.robust == True) | fi.feature.isin(CONTROLS) | (fi.q_adj < 0.10)]
    T["MT4"] = keep.sort_values(["cohort", "q_adj"])[
        ["cohort", "feature", "n_altered_pre", "n_altered_post", "pct_pre", "pct_post", "or_crude", "or_adj",
         "ci_low_adj", "ci_high_adj", "ci_type_adj", "p_adj", "q_adj", "or_full", "full_moved", "robust", "agreement"]]
    T["MT4"][["ci_low_adj", "ci_high_adj"]] = np.exp(T["MT4"][["ci_low_adj", "ci_high_adj"]])
    # MT5
    if D["has_B"]:
        tables_B(T, rep, fold)
    # MT6
    rs, fs_ = rep[(rep.stage == "sens") & (rep.eval_rows == "all")], fold[fold.stage == "sens"]
    s = summ(rs, fs_, "auc", ["cohort", "variant", "model", "fs"]).rename(columns={"mean": "roc_auc"})
    s = s.merge(rs.groupby(["cohort", "variant", "model", "fs"]).n.mean().rename("n").reset_index())
    s = s.merge(rs.groupby(["cohort", "variant", "model", "fs"]).prevalence.mean().rename("pct_post").reset_index())
    main = T["MT1"].set_index(["cohort", "model", "fs"]).roc_auc
    s["auc_main_same_model"] = [main.get((c, m, f), np.nan) for c, m, f in zip(s.cohort, s.model, s.fs)]
    T["MT6"] = s
    # MT7
    rk_sel = rk[rk.selected & (rk.block == "X") & (rk.rank_alt <= 20)][["cohort", "feature", "rank_alt"]]
    T["MT7"] = rk_sel.merge(D["strat"], on=["cohort", "feature"]).sort_values(["cohort", "rank_alt"])
    for kx, v in T.items():
        v.to_csv(OUT / f"{kx}.csv", index=False)
    return T


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def panels(n, w=3.6, h=3.6, rows=1, sharey=False):
    fig, ax = plt.subplots(rows, n, figsize=(w * n, h * rows), sharey=sharey, squeeze=False)
    return fig, ax


def m1(D, T):
    s = T["MT1"]
    fig, ax = panels(5, w=3.4, h=4.6, sharey=True)
    order = [("lr", f) for f in ("F0", "F1", "F2")] + [(m, f) for m in ("rf", "xgb") for f in ("F0", "F1", "F2", "F3")]
    for i, c in enumerate(M.COHORTS):
        a = ax[0, i]
        g = s[s.cohort == c].set_index(["model", "fs"])
        cov = s[(s.cohort == c) & (s.model == "covonly")].iloc[0]
        a.axvspan(cov.auc_fold_lo, cov.auc_fold_hi, color=viz.GRID, zorder=0)
        a.axvline(cov.roc_auc, color=viz.MUTED, lw=1)
        for j, (m, f) in enumerate(order):
            if (m, f) not in g.index:
                continue
            r = g.loc[(m, f)]
            a.plot([r.auc_fold_lo, r.auc_fold_hi], [j, j], color=MCOL[m], lw=2)
            a.plot(r.roc_auc, j, "o", color=MCOL[m], ms=8, mec=viz.SURFACE, mew=2)
        a.set_yticks(range(len(order)), [f"{M.MODEL_LABEL[m]} · {f}" for m, f in order])
        a.invert_yaxis()
        a.grid(axis="x", color=viz.GRID)
        a.grid(axis="y", visible=False)
        star = " *" if c in D["sel"].index else ""
        viz.finish(a, title=NAMES[c], subtitle=f"selected: {M.MODEL_LABEL[D['sel'].at[c, 'selected']]}", xlabel="ROC-AUC")
    fig.suptitle("M1. Model comparison (Phase 1): mean ROC-AUC with 2.5–97.5% fold spread; gray band = covariate-only",
                 x=0.01, ha="left", fontsize=11, color=viz.INK)
    fig.tight_layout()
    save(fig, "M1_model_comparison")


def m2(D, T):
    f = D["fold"]
    f = f[(f.stage == "phase2") & (f.variant == "main")]
    fig, a = plt.subplots(figsize=(7.5, 3.8))
    for i, c in enumerate(M.COHORTS):
        m = D["sel"].at[c, "selected"]
        w = f[(f.cohort == c) & (f.model == m)].pivot_table(index=["repeat", "fold"], columns="fs", values="auc")
        d = (w.F2 - w.F0).to_numpy()
        jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(d))
        a.scatter(np.full(len(d), i) + jit, d, s=10, color=MCOL[m], alpha=0.55, lw=0)
        a.plot([i - 0.28, i + 0.28], [np.median(d)] * 2, color=viz.INK, lw=2)
        a.text(i + 0.32, np.median(d), f"{np.median(d):+.3f}", va="center", fontsize=8.5, color=viz.INK_2)
    a.axhline(0, color=viz.AXIS, lw=1)
    a.set_xticks(range(5), [NAMES[c] for c in M.COHORTS], fontsize=8.5)
    viz.finish(a, title="M2. Predictive value the genome adds beyond sampling context",
               subtitle="Fold-wise ROC-AUC(F2) − AUC(F0), selected model, 100 folds; bar = median",
               ylabel="ΔAUC")
    save(fig, "M2_delta_auc")


def mean_curve(o, kind):
    grid = np.linspace(0, 1, 201)
    ys = []
    for _, g in o.groupby("repeat"):
        if kind == "roc":
            x, y, _ = roc_curve(g.y, g.pred)
            ys.append(np.interp(grid, x, y))
        else:
            p, r, _ = precision_recall_curve(g.y, g.pred)
            ys.append(np.interp(grid, r[::-1], p[::-1]))
    return grid, np.mean(ys, 0)


def m3(D, oof):
    fig, ax = panels(5, w=3.2, h=3.2, rows=2)
    for i, c in enumerate(M.COHORTS):
        m = D["sel"].at[c, "selected"]
        o = oof[(oof.cohort == c) & (oof.stage == "phase2") & (oof.variant == "main")]
        prev = o.y.mean()
        for kind, a in (("roc", ax[0, i]), ("pr", ax[1, i])):
            for mod, fs, lab in ((m, "F2", M.MODEL_LABEL[m] + " (F2)"), ("covonly", "F0", "Covariate-only")):
                x, y = mean_curve(o[(o.model == mod) & (o.fs == fs)], kind)
                a.plot(x, y, color=MCOL[mod], lw=2, label=lab)
            if kind == "roc":
                a.plot([0, 1], [0, 1], color=viz.AXIS, lw=1)
                viz.finish(a, title=NAMES[c], xlabel="False positive rate", ylabel="True positive rate" if i == 0 else None)
            else:
                a.axhline(prev, color=viz.AXIS, lw=1)
                a.text(0.98, prev + 0.02, f"prevalence {prev:.0%}", ha="right", fontsize=8, color=viz.MUTED)
                viz.finish(a, xlabel="Recall", ylabel="Precision" if i == 0 else None)
            a.set_xlim(0, 1)
            a.set_ylim(0, 1.02)
        if i == 0:
            ax[0, 0].legend(loc="lower right")
    fig.suptitle("M3. ROC and precision–recall curves, out-of-fold, averaged over 20 repeats", x=0.01, ha="left",
                 fontsize=11, color=viz.INK)
    fig.tight_layout()
    save(fig, "M3_roc_pr")


def m4(D, oof):
    fig, ax = panels(5, w=3.0, h=3.2, sharey=True)
    for i, c in enumerate(M.COHORTS):
        m = D["sel"].at[c, "selected"]
        o = oof[(oof.cohort == c) & (oof.stage == "phase2") & (oof.variant == "main") & (oof.model == m) & (oof.fs == "F2")]
        b = pd.qcut(o.pred, 10, labels=False, duplicates="drop")
        g = o.groupby(b).agg(p=("pred", "mean"), obs=("y", "mean"), n=("y", "size"))
        a = ax[0, i]
        a.plot([0, 1], [0, 1], color=viz.AXIS, lw=1)
        a.plot(g.p, g.obs, "-o", color=MCOL[m], lw=2, ms=6, mec=viz.SURFACE, mew=1.5)
        sl = D["rep"].query("stage=='phase2' and variant=='main' and cohort==@c and model==@m and fs=='F2' and eval_rows=='all'")
        viz.finish(a, title=NAMES[c], subtitle=f"slope {sl.cal_slope.mean():.2f}, intercept {sl.cal_intercept.mean():+.2f}",
                   xlabel="Predicted P(post)", ylabel="Observed % post" if i == 0 else None)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
    fig.suptitle("M4. Calibration of the selected model (deciles of OOF predictions, 20 repeats pooled)", x=0.01,
                 ha="left", fontsize=11, color=viz.INK)
    fig.tight_layout()
    save(fig, "M4_calibration")


def shap_long(cohort, model):
    s = pd.read_parquet(MD / "shap_oof.parquet", filters=[("cohort", "==", cohort), ("model", "==", model)])
    return s


def m5(D):
    fig, ax = panels(5, w=3.5, h=6.0)
    for i, c in enumerate(M.COHORTS):
        m = D["sel"].at[c, "selected"]
        rk = D["rank"].query("cohort==@c and model==@m and block=='X'").nsmallest(20, "rank_alt")
        s = shap_long(c, m)
        tab = M.load_table(c).set_index("SAMPLE_ID")
        a = ax[0, i]
        for j, f in enumerate(rk.feature):
            v = s[s.feature == f]
            alt = tab.reindex(v.SAMPLE_ID)[f].to_numpy() == 1
            jit = np.random.default_rng(j).uniform(-0.3, 0.3, len(v))
            a.scatter(v.shap[~alt], j + jit[~alt], s=4, color=viz.GRID, lw=0)
            a.scatter(v.shap[alt], j + jit[alt], s=6, color=viz.SEQ_STEPS[5], lw=0)
        a.axvline(0, color=viz.AXIS, lw=1)
        a.set_yticks(range(len(rk)), rk.feature)
        for t in a.get_yticklabels():
            if t.get_text() in CONTROLS:
                t.set_fontweight("bold")
        a.invert_yaxis()
        a.grid(axis="y", visible=False)
        a.grid(axis="x", color=viz.GRID)
        scale = "probability" if m == "rf" else "log-odds"
        viz.finish(a, title=NAMES[c], subtitle=f"{M.MODEL_LABEL[m]}, SHAP ({scale})", xlabel="SHAP → toward post")
    ax[0, 0].scatter([], [], s=20, color=viz.SEQ_STEPS[5], label="altered")
    ax[0, 0].scatter([], [], s=20, color=viz.GRID, label="not altered")
    ax[0, 0].legend(loc="center right")
    fig.suptitle("M5. SHAP for the top 20 alteration features (OOF, mean of 20 repeats); bold = positive / negative control",
                 x=0.01, ha="left", fontsize=11, color=viz.INK)
    fig.tight_layout()
    save(fig, "M5_shap_beeswarm")


def m6(D):
    fig, ax = plt.subplots(1, 10, figsize=(19, 6.0), gridspec_kw={"width_ratios": [2.2, 1] * 5})
    for i, c in enumerate(M.COHORTS):
        m = D["sel"].at[c, "selected"]
        rk = D["rank"].query("cohort==@c and model==@m and block=='X'").nsmallest(20, "rank_alt")
        a, b = ax[2 * i], ax[2 * i + 1]
        y = np.arange(len(rk))
        a.barh(y, rk.mean_abs_shap, color=MCOL[m], height=0.7)
        lab = [f + (f" [c{int(k)}]" if pd.notna(k) else "") for f, k in zip(rk.feature, rk.cluster)]
        a.set_yticks(y, lab, fontsize=8)
        a.invert_yaxis()
        a.grid(axis="y", visible=False)
        viz.finish(a, title=NAMES[c], xlabel="mean |SHAP|")
        b.scatter(rk.top20_freq, y, color=MCOL[m], s=30, zorder=3)
        b.axvline(0.7, color=viz.INK_2, lw=1)
        b.set_xlim(0, 1.05)
        b.set_ylim(a.get_ylim())
        b.set_yticks([])
        b.grid(axis="y", visible=False)
        viz.finish(b, xlabel="top-20 freq.")
    fig.suptitle("M6. Importance (mean |SHAP|) and stability (share of 100 fold-models with the feature in the top 20); "
                 "line = 70% robustness threshold; [cN] = correlated cluster (phi > 0.7)", x=0.01, ha="left",
                 fontsize=11, color=viz.INK)
    fig.tight_layout()
    save(fig, "M6_importance_stability")


def m7(D):
    s = D["share"].merge(D["sel"].reset_index()[["cohort", "selected"]], left_on=["cohort", "model"],
                         right_on=["cohort", "selected"])
    s = s.set_index("cohort").reindex(M.COHORTS)
    fig, a = plt.subplots(figsize=(7.5, 3.4))
    left = np.zeros(len(s))
    for b in ("C", "G", "X"):
        v = s[f"pct_{b}"].to_numpy()
        a.barh(range(len(s)), v, left=left, color=BLOCK_COL[b], height=0.6, label=BLOCK_NAME[b],
               edgecolor=viz.SURFACE, linewidth=2)
        for i, (l_, w) in enumerate(zip(left, v)):
            if w > 7:
                a.text(l_ + w / 2, i, f"{w:.0f}%", ha="center", va="center", fontsize=8.5, color="white")
        left += v
    a.set_yticks(range(len(s)), [NAMES[c] for c in s.index])
    a.invert_yaxis()
    a.set_xlim(0, 100)
    a.grid(axis="y", visible=False)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    viz.finish(a, title="M7. Where the selected model's attribution comes from",
               subtitle="% of total mean |SHAP| per block", xlabel="% of total |SHAP|")
    save(fig, "M7_block_share")


def m8(D):
    rk = D["rank"]
    rows = []
    for c in M.COHORTS:
        m = D["sel"].at[c, "selected"]
        r = rk.query("cohort==@c and model==@m and block=='X' and rank_alt<=20")
        fl = r[r.site_confounded == True]
        pick = fl if len(fl) else r.nlargest(1, "organ_interaction_share")
        rows += [(c, m, f, bool(sc)) for f, sc in zip(pick.feature.head(3), pick.site_confounded.head(3))]
    # Grid: one row per cohort, up to 3 features per row (a single row of 14 panels was unreadable).
    cohorts = list(dict.fromkeys(r[0] for r in rows))
    fig, ax = plt.subplots(len(cohorts), 3, figsize=(4.4 * 3, 2.9 * len(cohorts)), squeeze=False)
    used = set()
    for (c, m, f, flagged) in rows:
        s = shap_long(c, m)
        s = s[s.feature == f]
        tab = M.load_table(c).set_index("SAMPLE_ID").reindex(s.SAMPLE_ID)
        alt = tab[f].to_numpy() == 1
        org = tab.biopsy_organ.to_numpy()[alt]
        v = s.shap.to_numpy()[alt]
        top = pd.Series(org).value_counts().head(6).index
        ri = cohorts.index(c)
        ci = sum(1 for k in used if k[0] == ri)
        used.add((ri, ci))
        a = ax[ri, ci]
        for j, o in enumerate(top):   # organs are labelled on the y-axis; one color, not one per organ
            vv = v[org == o]
            a.scatter(vv, np.full(len(vv), j) + np.random.default_rng(j).uniform(-0.25, 0.25, len(vv)), s=8,
                      color=viz.SEQ_STEPS[5], alpha=0.7, lw=0)
        a.set_yticks(range(len(top)), top, fontsize=8)
        a.invert_yaxis()
        a.axvline(0, color=viz.AXIS, lw=1)
        a.grid(axis="y", visible=False)
        viz.finish(a, title=f"{NAMES[c]}: {f}",
                   subtitle="flagged site-confounded" if flagged else "highest organ-interaction share (not flagged)",
                   xlabel="SHAP among altered samples")
    for ri in range(len(cohorts)):
        for ci in range(3):
            if (ri, ci) not in used:
                ax[ri, ci].set_visible(False)
    fig.suptitle("M8. Site dependence: SHAP of altered samples by biopsy organ", x=0.01, ha="left", fontsize=11,
                 color=viz.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, "M8_site_interaction")


def m9(D):
    fi = D["firth"]
    fi = fi[(fi.robust == True)].copy()
    cohorts = [c for c in M.COHORTS if (fi.cohort == c).any()]
    h = max(3, 0.28 * fi.groupby("cohort").size().max() + 1.2)
    fig, ax = panels(len(cohorts), w=3.6, h=h)
    for i, c in enumerate(cohorts):
        g = fi[fi.cohort == c].sort_values("log_or_adj")
        a = ax[0, i]
        y = np.arange(len(g))
        for j, r in enumerate(g.itertuples()):
            col = AGREE_COL.get(r.agreement, viz.MUTED)
            a.plot([np.exp(r.ci_low_adj), np.exp(r.ci_high_adj)], [j, j], color=col, lw=2)
            a.plot(np.exp(r.log_or_adj), j, "o", color=col, ms=7, mec=viz.SURFACE, mew=1.5)
            a.plot(np.exp(r.log_or_crude), j, "o", mfc="none", mec=viz.INK_2, ms=6, mew=1)
        a.axvline(1, color=viz.AXIS, lw=1)
        a.set_xscale("log")
        a.set_yticks(y, g.feature, fontsize=8)
        a.grid(axis="y", visible=False)
        viz.finish(a, title=NAMES[c], xlabel="Odds ratio, post vs pre (log)")
    for k, col in AGREE_COL.items():
        ax[0, 0].plot([], [], "o", color=col, label=k)
    ax[0, 0].plot([], [], "o", mfc="none", mec=viz.INK_2, label="crude OR")
    ax[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("M9. Strict-tier confirmation: prognostic-score-adjusted Firth OR (95% profile CI) for robust SHAP features",
                 x=0.01, ha="left", fontsize=11, color=viz.INK)
    fig.tight_layout()
    save(fig, "M9_strict_confirmation")


def m10(D):
    rep, fold = D["rep"], D["fold"]
    r = rep[(rep.stage == "lc") & (rep.eval_rows == "all")]
    ss = pd.read_parquet(MD / "lc_shap_sum.parquet")
    ss = ss[ss.block == "X"]
    fig, ax = panels(5, w=3.2, h=3.0, rows=2)
    for i, c in enumerate(M.COHORTS):
        for m in dict.fromkeys([D["sel"].at[c, "selected"], "lr"]):
            g = r[(r.cohort == c) & (r.model == m)].groupby("frac").auc.agg(["mean", "std"])
            ax[0, i].plot(g.index * 100, g["mean"], "-o", color=MCOL[m], ms=6, lw=2, label=M.MODEL_LABEL[m])
            s = ss[(ss.cohort == c) & (ss.model == m)]
            # I7: only features the fold-model uses (nonzero SHAP) can be in its top 20; ties share a rank.
            s = s[s.mean_abs > 1e-10]
            top = s.assign(rk=s.groupby(["repeat", "fold", "frac"]).mean_abs.rank(ascending=False, method="min"))
            top = top[top.rk <= 20]
            full = top[top.frac == 1.0].groupby(["repeat", "fold"]).feature.apply(set)
            ov = top.groupby(["frac", "repeat", "fold"]).feature.apply(set).reset_index()
            ov["overlap"] = [len(f & full[(rp, fo)]) / len(full[(rp, fo)]) if (rp, fo) in full.index else np.nan
                             for f, rp, fo in zip(ov.feature, ov.repeat, ov.fold)]
            o = ov.groupby("frac").overlap.mean()   # fold-models with no feature in use at 100% are skipped
            ax[1, i].plot(o.index * 100, o, "-o", color=MCOL[m], ms=6, lw=2)
        viz.finish(ax[0, i], title=NAMES[c], ylabel="ROC-AUC" if i == 0 else None)
        viz.finish(ax[1, i], xlabel="% of training fold", ylabel="Top-20 overlap with 100%" if i == 0 else None)
        ax[1, i].set_ylim(0, 1.05)
    ax[0, 0].legend(loc="lower right")
    ax[0, 4].legend(loc="lower right")
    fig.suptitle("M10. Learning curve (repeats r00–r04): AUC and top-20 SHAP overlap vs training size", x=0.01,
                 ha="left", fontsize=11, color=viz.INK)
    fig.tight_layout()
    save(fig, "M10_learning_curve")


def m11(D, oof):
    selB = D["selB"].query("target=='B'").set_index("cohort").selected
    fig, ax = panels(4, w=3.4, h=3.4)
    for i, c in enumerate(C.COHORTS):
        m = selB[c]
        o = oof[(oof.stage == "targetB") & (oof.cohort == c) & (oof.target == "B") & (oof.variant == "B_main")
                & (oof.model == m) & (oof.repeat == "fold_r00")]
        a = ax[0, i]
        a.scatter(np.expm1(o.y), np.expm1(o.pred), s=10, color=MCOL[m], alpha=0.6, lw=0)
        lim = [30, max(np.expm1(o.y).max(), np.expm1(o.pred).max()) * 1.1]
        a.plot(lim, lim, color=viz.AXIS, lw=1)
        a.set_xscale("log")
        a.set_yscale("log")
        rho = D["rep"].query("stage=='targetB' and cohort==@c and target=='B' and variant=='B_main' and model==@m").spearman.mean()
        viz.finish(a, title=NAMES[c], subtitle=f"{M.MODEL_LABEL[m]}, Spearman ρ = {rho:.2f}" +
                   (" (exploratory)" if c == "nsclc" else ""), xlabel="Observed days on class",
                   ylabel="Predicted days" if i == 0 else None)
    fig.suptitle("M11a. Target B: cumulative days on the drug class among post samples (≥ 30 days), repeat r00",
                 x=0.01, ha="left", fontsize=11, color=viz.INK)
    fig.tight_layout()
    save(fig, "M11a_targetB_pred_obs")
    # beeswarm for the dose model
    fig, ax = panels(4, w=3.6, h=5.0)
    for i, c in enumerate(C.COHORTS):
        m = selB[c]
        sh = pd.read_parquet(MD / "shap" / f"targetB_{c}_B_{m}_B_full_B_main.parquet")
        v = [k for k in sh.columns if k not in ("SAMPLE_ID", "fold", "repeat")]
        ps = sh[v].fillna(0).groupby(sh.SAMPLE_ID).mean()
        top = ps.abs().mean().nlargest(15).index
        a = ax[0, i]
        for j, f in enumerate(top):
            jit = np.random.default_rng(j).uniform(-0.3, 0.3, len(ps))
            a.scatter(ps[f], j + jit, s=5, color=MCOL[m], alpha=0.5, lw=0)
        a.axvline(0, color=viz.AXIS, lw=1)
        a.set_yticks(range(len(top)), top, fontsize=8)
        a.invert_yaxis()
        a.grid(axis="y", visible=False)
        viz.finish(a, title=NAMES[c], subtitle=M.MODEL_LABEL[m], xlabel="SHAP (log1p days)")
    fig.suptitle("M11b. Target B SHAP: top 15 inputs of the dose model (all blocks)", x=0.01, ha="left", fontsize=11,
                 color=viz.INK)
    fig.tight_layout()
    save(fig, "M11b_targetB_shap")


def m12(D, T):
    s = T["MT6"]
    main = T["MT1"]
    fig, ax = panels(5, w=3.0, h=3.0, sharey=True)
    for i, c in enumerate(M.COHORTS):
        m = D["sel"].at[c, "selected"]
        a = ax[0, i]
        for j, (lab, v) in enumerate((("All", "main"), ("Metastasis", "met_only"), ("Primary", "prim_only"))):
            for fs, col, off in (("F0", viz.MUTED, -0.12), ("F2", MCOL[m], 0.12)):
                if v == "main":
                    r = main[(main.cohort == c) & (main.model == m) & (main.fs == fs)].iloc[0]
                    mean, lo, hi = r.roc_auc, r.auc_fold_lo, r.auc_fold_hi
                else:
                    q = s[(s.cohort == c) & (s.variant == v) & (s.model == m) & (s.fs == fs)]
                    if q.empty:
                        continue
                    mean, lo, hi = q.roc_auc.iloc[0], q.lo.iloc[0], q.hi.iloc[0]
                a.plot([j + off] * 2, [lo, hi], color=col, lw=2)
                a.plot(j + off, mean, "o", color=col, ms=7, mec=viz.SURFACE, mew=1.5,
                       label=("F0 covariates" if fs == "F0" else "F2 full") if (i == 0 and j == 0) else None)
        a.set_xticks(range(3), ["All", "Metastasis", "Primary"], fontsize=8.5)
        viz.finish(a, title=NAMES[c], subtitle=M.MODEL_LABEL[m], ylabel="ROC-AUC" if i == 0 else None)
    ax[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle("M12a. Stratified models: does the genome still separate pre from post within one sample type?",
                 x=0.01, ha="left", fontsize=11, color=viz.INK)
    fig.tight_layout()
    save(fig, "M12a_stratified_auc")
    t7 = T["MT7"]
    fig, ax = panels(5, w=3.4, h=5.6)
    for i, c in enumerate(M.COHORTS):
        g = t7[t7.cohort == c].sort_values("rank_alt")
        a = ax[0, i]
        y = np.arange(len(g))
        for j, r in enumerate(g.itertuples()):
            a.plot([r.pct_pre_met, r.pct_post_met], [j, j], color=viz.GRID, lw=2, zorder=1)
        a.scatter(g.pct_pre_met, y, color=PRE, s=28, zorder=2, label="pre")
        a.scatter(g.pct_post_met, y, color=POST, s=28, zorder=2, label="post")
        lab = [f + (" (site)" if sc else "") for f, sc in zip(g.feature, g.site_confounded)]
        a.set_yticks(y, lab, fontsize=8)
        a.invert_yaxis()
        a.grid(axis="y", visible=False)
        viz.finish(a, title=NAMES[c], xlabel="% altered within metastases")
    ax[0, 0].legend(loc="lower right")
    fig.suptitle("M12b. Top-20 alteration features within metastatic biopsies only; (site) = flagged site-confounded",
                 x=0.01, ha="left", fontsize=11, color=viz.INK)
    fig.tight_layout()
    save(fig, "M12b_stratified_enrichment")


# --------------------------------------------------------------------------
# README
# --------------------------------------------------------------------------

def fmt_table(df, cols=None, n=None, fl="{:.3g}"):
    d = df[cols] if cols else df
    return C.md_table(d.head(n) if n else d, index=False, floatfmt=fl)


def readme(D, T):
    t2 = T["MT2"]
    g = D["gate"]
    md = ["# M3 models: pre- vs post-treatment classification", "",
          "_Generated by `data/report_models.py` from `data/derived/models/`. Plan: `modeling_plan.md` (v2)._", "",
          "All metrics are out-of-fold. Means are over repeats of pooled OOF predictions; intervals are the "
          "2.5–97.5% spread across folds, which is resampling variability, not a confidence interval.", "",
          "## Control gate (Section 6.5)", "", fmt_table(g, ["cohort", "variant", "feature", "expect", "rank_alt",
                                                            "mean_shap_altered", "passed"]), "",
          "## Model selection and headline (MT2)", "",
          fmt_table(t2, ["cohort", "selected", "auc_F2", "auc_F0_same_model", "auc_covonly", "delta_auc",
                         "delta_auc_fold_lo", "delta_auc_fold_hi", "delta_auc_p_nb", "pr_auc", "prevalence",
                         "brier", "cal_slope", "auc_strict_rows"]), "",
          "`delta_auc_p_nb` is the Nadeau–Bengio corrected resampled t-test on the 100 paired fold differences "
          "(approximate). `auc_strict_rows` scores the same OOF predictions on strict-tier rows only (5.3).", "",
          "Selection rationale:", ""] + [f"- **{r.cohort}**: {r.rationale}" for r in t2.itertuples()] + [""]
    figs = [("M1_model_comparison", "Model comparison"), ("M2_delta_auc", "ΔAUC from the genome"),
            ("M3_roc_pr", "ROC and PR curves"), ("M4_calibration", "Calibration"),
            ("M5_shap_beeswarm", "SHAP beeswarm"), ("M6_importance_stability", "Importance and stability"),
            ("M7_block_share", "Block share"), ("M8_site_interaction", "Site interaction"),
            ("M9_strict_confirmation", "Strict-tier confirmation"), ("M10_learning_curve", "Learning curve"),
            *([("M11a_targetB_pred_obs", "Target B predicted vs observed"), ("M11b_targetB_shap", "Target B SHAP")]
              if D["has_B"] else []),
            ("M12a_stratified_auc", "Stratified AUC"), ("M12b_stratified_enrichment", "Stratified enrichment")]
    md += ["## Figures", ""]
    for f, cap in figs:
        md += [f"### {cap}", "", f"![{cap}]({f}.png)", ""]
    md += ["## Tables", "", "### MT1. All models × feature sets (Phase 1, repeats r00–r04)", "",
           fmt_table(T["MT1"], ["cohort", "model", "fs", "roc_auc", "auc_fold_lo", "auc_fold_hi", "ap", "brier",
                                "logloss", "cal_intercept", "cal_slope"]), "",
           "### MT3. SHAP ranking, selected model (top 30 alteration features + all robust)", "",
           fmt_table(T["MT3"], ["cohort", "feature", "rank_alt", "mean_abs_shap", "mean_shap_altered", "top20_freq",
                                "robust", "cluster", "site_confounded", "organ_interaction_share"]), "",
           "### MT4. Strict-tier Firth (robust features, controls, and q < 0.10)", "",
           fmt_table(T["MT4"], ["cohort", "feature", "n_altered_pre", "n_altered_post", "or_crude", "or_adj",
                                "ci_low_adj", "ci_high_adj", "ci_type_adj", "q_adj", "or_full", "full_moved",
                                "robust", "agreement"]), "",
           *(["### MT5. Target B regression (repeats r00–r04)", "",
              fmt_table(T["MT5"], ["cohort", "target", "variant", "model", "n_rows", "spearman", "rho_fold_lo",
                                   "rho_fold_hi", "mae_days", "delta_mae_vs_cov", "r2_log", "note"]), "",
              "`target` B = log1p(cumulative days on class); B_t = log1p(days since first dose). `B_main` excludes "
              "post samples with < 30 recorded days; `B_all` keeps them.", ""] if D["has_B"] else
             ["### MT5. Target B regression", "", "Not run yet: `python train_models.py targetB finalize`, "
              "then rerun this script.", ""]),
           "### MT6. Sensitivity analyses (selected model, repeats r00–r04)", "",
           fmt_table(T["MT6"], ["cohort", "variant", "model", "fs", "n", "pct_post", "roc_auc", "lo", "hi",
                                "auc_main_same_model"]), "",
           "### MT7. Stratified enrichment (top-20 features, relaxed training rows)", "",
           fmt_table(T["MT7"], ["cohort", "feature", "rank_alt", "pct_pre_all", "pct_post_all", "pct_pre_met",
                                "pct_post_met", "pct_pre_prim", "pct_post_prim", "log_or_all", "log_or_mh",
                                "site_confounded"]), ""]
    (OUT / "README.md").write_text("\n".join(md), encoding="utf-8")


def main():
    viz.apply_theme()
    D = load()
    gated = D["gate"][D["gate"].gated]
    assert gated.passed.all(), "7. control gate failed: no figures are written"
    OUT.mkdir(parents=True, exist_ok=True)
    T = tables(D)
    oof = pd.read_parquet(MD / "oof_predictions.parquet",
                          filters=[("stage", "in", ["phase2", "targetB"])])
    for f in (lambda: m1(D, T), lambda: m2(D, T), lambda: m3(D, oof), lambda: m4(D, oof), lambda: m5(D),
              lambda: m6(D), lambda: m7(D), lambda: m8(D), lambda: m9(D), lambda: m10(D), *([lambda: m11(D, oof)] if D['has_B'] else []),
              lambda: m12(D, T)):
        f()
    readme(D, T)
    print(f"wrote {len(list(OUT.glob('*.png')))} figures and {len(T)} tables to {OUT}")


if __name__ == "__main__":
    main()
