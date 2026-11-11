"""Section 7: confirmation on the strict tier (main_analysis rows).

1. prog_score: covariate-only ridge logistic regression (coarse 5-level biopsy organ),
   out-of-fold logit over the strict folds (fold_r00); ridge strength by 5-fold inner CV.
2. Per-feature Firth regressions: label ~ feature (crude) and label ~ feature + prog_score
   (adjusted), profile-likelihood CIs and PLR p-values, Benjamini-Hochberg q per cohort.
3. Top 20 by adjusted q: label ~ feature + all covariates; flag ORs that move by more than
   a factor of 1.5 relative to the adjusted OR.
4. Agreement with the relaxed-tier SHAP (7.3), site-confounding from explain.py.

Writes derived/models/prog_score.parquet and firth_strict.csv (MT4).
Run: ``python confirm_strict.py``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold

import common as C
import loaders as L
import m3_common as M
import preprocess as PP
from firth import firth_logit

OUT = M.MODELS_DIR
Q_CONFIRM = 0.10
MOVE = 1.5


def strict_rows(cohort: str) -> pd.DataFrame:
    t = M.load_table(cohort)
    return t[t.main_analysis == 1].reset_index(drop=True)


def prog_design(cohort: str) -> M.Design:
    types = {c: k for c, k in M.blocks(cohort)["C"].items() if c != "biopsy_organ"}
    types["biopsy_organ_coarse"] = "cat"
    return M.Design([], types, set(types), (), allow={"biopsy_organ_coarse"})


def prog_score(cohort: str) -> pd.DataFrame:
    d = strict_rows(cohort)
    y = d.label_A.to_numpy()
    folds = d.fold_r00.to_numpy()
    out = np.full(len(d), np.nan)
    for k in np.unique(folds):
        te, tr = folds == k, folds != k
        dtr, ytr = d[tr].reset_index(drop=True), y[tr]
        inner = list(StratifiedGroupKFold(5, shuffle=True, random_state=M.SEED + int(k))
                     .split(dtr, ytr, dtr.PATIENT_ID))
        grid = [0.001, 0.01, 0.1, 1, 10]
        sc = np.zeros(len(grid))
        for itr, iva in inner:
            des = prog_design(cohort).fit(dtr.iloc[itr])
            Xa, Xb = des.transform(dtr.iloc[itr]), des.transform(dtr.iloc[iva])
            for i, c in enumerate(grid):
                m = LogisticRegression(C=c, l1_ratio=0, max_iter=5000).fit(Xa, ytr[itr])
                sc[i] += log_loss(ytr[iva], m.predict_proba(Xb)[:, 1], labels=[0, 1])
        des = prog_design(cohort).fit(dtr)
        assert des.fit_ids_.isdisjoint(set(d.SAMPLE_ID[te])), "5. prog_score is out-of-fold"
        m = LogisticRegression(C=grid[int(np.argmin(sc))], l1_ratio=0, max_iter=5000).fit(des.transform(dtr), ytr)
        out[te] = m.decision_function(des.transform(d[te]))
    assert np.isfinite(out).all()
    return pd.DataFrame({"cohort": cohort, "SAMPLE_ID": d.SAMPLE_ID, "prog_score": out})


def feature_list(cohort: str, rank: pd.DataFrame) -> list[str]:
    fd = pd.read_csv(L.DERIVED / "feature_dictionary.csv")
    k, m = f"{cohort}_kept_firth", f"{cohort}_main_analysis_feature"
    fd = fd[fd.block.isin(["alteration", "variant_target"]) & (fd[k] == True) & (fd[m] == True)]
    feats = set(fd.feature)
    rob = rank[(rank.cohort == cohort) & rank.selected & rank.robust].feature
    t = M.load_table(cohort)
    out = [f for f in dict.fromkeys([*rob, *sorted(feats)]) if f in t.columns]
    return out


def bh(p: pd.Series) -> pd.Series:
    p = p.astype(float)
    ok = p.notna()
    q = pd.Series(np.nan, index=p.index)
    v = p[ok].sort_values()
    n = len(v)
    adj = (v * n / np.arange(1, n + 1))[::-1].cummin()[::-1].clip(upper=1)
    q[adj.index] = adj
    return q


def fit_one(cohort, f, x, y, ps):
    r = {"cohort": cohort, "feature": f, "n_altered": int(x.sum()), "n_altered_pre": int(x[y == 0].sum()),
         "n_altered_post": int(x[y == 1].sum()), "pct_pre": 100 * x[y == 0].mean(), "pct_post": 100 * x[y == 1].mean()}
    one = np.ones(len(y))
    for tag, X in (("crude", np.column_stack([one, x])), ("adj", np.column_stack([one, x, ps]))):
        fr = firth_logit(X, y, profile=(1,))
        r[f"log_or_{tag}"] = fr["coef"]["x1"]
        r[f"ci_low_{tag}"], r[f"ci_high_{tag}"] = fr["ci_low"]["x1"], fr["ci_high"]["x1"]
        r[f"p_{tag}"], r[f"ci_type_{tag}"] = fr["p"]["x1"], fr["ci_type"]["x1"]
    return r


def full_cov_matrix(cohort: str, d: pd.DataFrame) -> np.ndarray:
    """All block-C covariates, one-hot with the first level dropped, numerics standardized."""
    types = M.blocks(cohort)["C"]
    pre = PP.make_preprocessor([], types).fit(d)
    X = pre.transform(d)
    drop = []
    for c in pre.cat_:
        lv = [k for k in X.columns if k.startswith(f"cat__{c}_")]
        drop.append(lv[0])
    X = X.drop(columns=drop)
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float)


def main() -> None:
    import importlib.util  # load by path: an installed package named "tests" can shadow the local folder
    spec = importlib.util.spec_from_file_location("test_firth", L.DATA_DIR / "tests" / "test_firth.py")
    TF = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(TF)
    TF.run_all()
    print("  [ok] 9. Firth validation test passes")
    rank = pd.read_csv(OUT / "shap_ranking.csv")
    strat = pd.read_csv(OUT / "stratified_enrichment.csv")
    ps_all = pd.concat([prog_score(c) for c in C.COHORTS], ignore_index=True)
    ps_all.to_parquet(OUT / "prog_score.parquet", index=False)
    rows = []
    for cohort in M.COHORTS:
        d = strict_rows(cohort)
        ps = d[["cohort", "SAMPLE_ID"]].merge(ps_all, on=["cohort", "SAMPLE_ID"], how="left").prog_score.to_numpy()
        assert np.isfinite(ps).all(), f"[{cohort}] every strict row has a prog_score"
        y = d.label_A.to_numpy().astype(float)
        cand = feature_list(cohort, rank)
        # A feature altered in no strict row or in every strict row has no estimable odds ratio
        # (e.g. EGFR_mut in the EGFR-TKI NSCLC cohort).
        const = [f for f in cand if d[f].sum() in (0, len(d))]
        if const:
            print(f"[{cohort}] skipped (no variation in strict rows): {', '.join(const)}")
        feats = [f for f in cand if f not in const]
        res = Parallel(n_jobs=8)(delayed(fit_one)(cohort, f, d[f].to_numpy(float), y, ps) for f in feats)
        r = pd.DataFrame(res)
        r["q_adj"], r["q_crude"] = bh(r.p_adj), bh(r.p_crude)
        # 7.2 step 3: full-covariate check for the top 20 by adjusted q
        Xc = full_cov_matrix(cohort, d)
        top = r.nsmallest(20, "q_adj").feature
        for f in top:
            X = np.column_stack([np.ones(len(y)), d[f].to_numpy(float), Xc])
            fr = firth_logit(X, y, profile=())
            i = r.index[r.feature == f][0]
            r.loc[i, "log_or_full"] = fr["coef"]["x1"]
            r.loc[i, "ci_low_full"], r.loc[i, "ci_high_full"] = fr["ci_low"]["x1"], fr["ci_high"]["x1"]
            r.loc[i, "ci_type_full"] = fr["ci_type"]["x1"]
        r["full_moved"] = (np.abs(r.log_or_full - r.log_or_adj) > np.log(MOVE)).where(r.log_or_full.notna())
        rows.append(r)
        print(f"[{cohort}] {len(r)} features, {int((r.q_adj < Q_CONFIRM).sum())} with adjusted q < {Q_CONFIRM}")
    out = pd.concat(rows, ignore_index=True)
    assert out[["ci_type_crude", "ci_type_adj"]].notna().all().all(), "6. every Firth fit records its CI type"
    # 7.3 agreement with the relaxed-tier SHAP (selected model)
    rk = rank[rank.selected & (rank.block == "X")][["cohort", "feature", "mean_shap_altered", "robust", "top20_freq"]]
    out = out.merge(rk, on=["cohort", "feature"], how="left")
    out = out.merge(strat[["cohort", "feature", "site_confounded"]], on=["cohort", "feature"], how="left")

    def cat(x):
        if pd.isna(x.mean_shap_altered):
            return "no SHAP"
        if x.site_confounded == True:  # noqa: E712 (bool or numpy bool)
            return "Site-confounded"
        same = np.sign(x.mean_shap_altered) == np.sign(x.log_or_adj)
        if not same:
            return "Discordant"
        return "Confirmed" if x.q_adj < Q_CONFIRM else "Directional"
    out["agreement"] = out.apply(cat, axis=1)
    for t in ("crude", "adj", "full"):
        out[f"or_{t}"] = np.exp(out[f"log_or_{t}"])
    out.to_csv(OUT / "firth_strict.csv", index=False)
    rob = out[out.robust == True]
    print(rob.groupby(["cohort", "agreement"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(L.DATA_DIR))
    main()
