"""Shared helpers for the M3 modelling scripts (modeling_plan.md v2).

Feature blocks (Section 3), row selections and folds (Sections 1, 5.5), the
in-fold design matrix (3.7), nested tuning for every model (4.1, 4.3), SHAP
(6.1) and metrics (5.1). Used by train_models.py, explain.py,
confirm_strict.py and report_models.py.
"""

from __future__ import annotations

import functools
import re
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
import xgboost as xgb

import common as C
import loaders as L
import preprocess as PP

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*did not converge.*")
warnings.filterwarnings("ignore", message=".*max_iter was reached.*")

MODELS_DIR = L.DERIVED / "models"
REPORT_DIR = C.REPORTS / "models"
COHORTS = [*C.COHORTS, "pooled"]
CONTROL_COHORTS = C.COHORTS
SEED = 20261106
PHASE1_REPEATS = [f"fold_r{r:02d}" for r in range(5)]
ALL_REPEATS = [f"fold_r{r:02d}" for r in range(20)]
COV_SCALE = 10.0                       # 4.3: covariates approximately unpenalized in elastic-net regression
B_MIN_DAYS = 30                        # P10 / 1.1

# 3.1: never inputs (asserted absent from every design matrix).
NEVER = {"SAMPLE_ID", "PATIENT_ID", "cohort", "label_A", "label_A_on", "label_B", "label_B_log1p",
         "t_days", "tier_strict", "tier_strict_noint", "tier_relaxed", "main_analysis",
         "s3_survivorship_keep", "hypermutated", "facets_pass", "followup_days",
         "biopsy_organ_coarse", "low_purity", "prior_med_to_msk", "PRIOR_MED_TO_MSK"}
C_EXCLUDE = {"biopsy_organ_coarse", "low_purity"}

# Positive / negative controls for the gate (6.5).
GATE_POS = {"breast": ["ESR1_LBD"], "nsclc": ["EGFR_T790M", "MET_amp"], "prostate": ["AR_amp", "AR_LBD"]}
GATE_NEG = {"crc": ["KRAS_any"]}

SIMPLICITY = ["lr", "rf", "xgb"]       # 5.2 order
MODEL_LABEL = {"prevalence": "Prevalence", "covonly": "Covariate-only", "lr": "Elastic-net LR",
               "rf": "Random forest", "xgb": "XGBoost", "median": "Median", "ridge_cov": "Covariate-only ridge",
               "enet": "Elastic net"}


# --------------------------------------------------------------------------
# Data and blocks
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def target_variants() -> dict[str, list[str]]:
    tv = C.config("target_variants")["cohorts"]
    d = {c: list(v) for c, v in tv.items()}
    d["pooled"] = [f for v in tv.values() for f in v]
    return d


@functools.lru_cache(maxsize=None)
def alt_features() -> pd.DataFrame:
    return pd.read_csv(L.DERIVED / "alteration_features.csv").set_index("feature")


@functools.lru_cache(maxsize=None)
def load_table(cohort: str) -> pd.DataFrame:
    return pd.read_parquet(L.DERIVED / f"model_table_{cohort}.parquet")


@functools.lru_cache(maxsize=None)
def folds_wide(tier: str) -> pd.DataFrame:
    """cohort x SAMPLE_ID -> one column per repeat (label variant A)."""
    f = pd.read_parquet(L.DERIVED / "folds.parquet")
    f = f[(f.tier == tier) & (f.label_variant == "A")]
    return f.pivot_table(index=["cohort", "SAMPLE_ID"], columns="repeat", values="fold").astype(int)


def blocks(cohort: str) -> dict[str, object]:
    """Section 3: column lists for blocks C, G, X, B, LOH and clonality."""
    t = load_table(cohort)
    af = alt_features()
    on341 = af.on_IMPACT341.astype(bool)
    tv_all = set(target_variants()["pooled"])
    own = target_variants()[cohort]
    cov_c = {c: k for c, k in PP.COVARIATES_A.items() if c in t.columns and c not in C_EXCLUDE}
    gen = {c: k for c, k in PP.GENOMIC.items() if c in t.columns}
    cov_b = {c: k for c, k in PP.COVARIATES_B.items() if c in t.columns}
    x = [c for c in t.columns
         if (re.search(r"_(mut|amp|del|sv)$", c) or c in tv_all)
         and c in af.index and bool(on341.get(c, False))]
    mut_genes = [c[:-4] for c in x if c.endswith("_mut")]
    clon = [c for g in mut_genes for c in (f"{g}_mut_clonal", f"{g}_mut_subclonal", f"{g}_mut_clonality_unknown")
            if c in t.columns]
    loh = [c for c in t.columns if c.endswith("_loh") and bool(on341.get(c, False))]
    keep = tuple(c for c in own if c in x)
    return {"C": cov_c, "G": gen, "X": x, "B": cov_b, "LOH": loh, "CLON": clon, "always_keep": keep}


def feature_spec(cohort: str, fs: str) -> tuple[list[str], dict[str, str], set[str], tuple[str, ...]]:
    """(alteration columns, covariate types, block-C variable names, always_keep) for a feature set."""
    b = blocks(cohort)
    cset = set(b["C"])
    if fs == "F0":
        return [], dict(b["C"]), cset, ()
    if fs == "F1":
        return list(b["X"]), {}, cset, b["always_keep"]
    if fs == "F2":
        return list(b["X"]), {**b["C"], **b["G"]}, cset, b["always_keep"]
    if fs == "F3":   # clonality triples replace <GENE>_mut; LOH added (trees only)
        x = [c for c in b["X"] if not c.endswith("_mut")] + b["CLON"] + b["LOH"]
        return x, {**b["C"], **b["G"]}, cset, b["always_keep"]
    if fs == "B_cov":
        return [], {**b["C"], **b["B"]}, cset | set(b["B"]), ()
    if fs == "B_full":
        return list(b["X"]), {**b["C"], **b["G"], **b["B"]}, cset | set(b["B"]), b["always_keep"]
    raise ValueError(fs)


@functools.lru_cache(maxsize=None)
def hotspot_mut_matrix() -> pd.DataFrame:
    """Hotspot-only <GENE>_mut calls (sensitivity 'hypermutated included', 5.5)."""
    a = pd.read_parquet(L.DERIVED / "alterations_long.parquet",
                        columns=["SAMPLE_ID", "gene", "alt_type", "onco_rule", "on_panel"])
    a = a[(a.alt_type == "mut") & a.onco_rule.isin(["hotspot_missense", "hotspot_inframe"]) & a.on_panel]
    m = pd.crosstab(a.SAMPLE_ID, a.gene).clip(upper=1).astype(float)
    m.columns = [f"{g}_mut" for g in m.columns]
    return m


def select_rows(cohort: str, variant: str, target: str = "A") -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Rows, fold table (sample x repeat) and outcome for a variant (Sections 1, 1.1, 5.5)."""
    t = load_table(cohort)
    nonhyper = t.hypermutated == 0
    m = {
        "main": nonhyper,
        "met_only": nonhyper & (t.sample_type == "Metastasis"),
        "prim_only": nonhyper & (t.sample_type == "Primary"),
        "s3": nonhyper & (t.s3_survivorship_keep == 1),
        "a_on": nonhyper & ((t.label_A == 0) | (t.label_A_on == 1)),
        "hyper_incl": pd.Series(True, index=t.index),
        "strict_trained": t.main_analysis == 1,
        "facets": nonhyper & (t.facets_pass == 1),
        "B_main": nonhyper & (t.label_A == 1) & (t.label_B >= B_MIN_DAYS),
        "B_all": nonhyper & (t.label_A == 1),
    }[variant]
    d = t[m.to_numpy()].copy().reset_index(drop=True)
    if variant == "hyper_incl":
        hs = hotspot_mut_matrix()
        muts = [c for c in blocks(cohort)["X"] if c.endswith("_mut")]
        rep = hs.reindex(index=d.SAMPLE_ID, columns=muts).fillna(0.0).to_numpy()
        d[muts] = rep
    tier = "strict" if variant == "strict_trained" else "relaxed"
    fw = folds_wide(tier).loc[cohort].reindex(d.SAMPLE_ID)
    assert fw.notna().all().all(), f"[{cohort}/{variant}] every row has a fold"
    fw.index = d.index
    if target == "A":
        y = d.label_A.to_numpy().astype(int)
    elif target == "B":
        y = d.label_B_log1p.to_numpy().astype(float)
    elif target == "B_t":
        y = np.log1p(d.t_days.to_numpy().astype(float))
    else:
        raise ValueError(target)
    # assert 8 / 10
    if variant == "B_main":
        assert (d.label_B >= B_MIN_DAYS).all(), "8. Target B rows have label_B >= 30"
    if variant in ("met_only", "prim_only"):
        want = {"met_only": "Metastasis", "prim_only": "Primary"}[variant]
        assert (d.sample_type == want).all(), "10. stratified variant rows share the stratum's sample_type"
    return d, fw, y


# --------------------------------------------------------------------------
# Design matrix (3.7)
# --------------------------------------------------------------------------

class Design:
    """FoldPreprocessor + bookkeeping: origin variable and block of every column."""

    def __init__(self, alt_cols, cov_types, cov_c, always_keep, scale_cov=False, unpen_cov=False, allow=()):
        self.allow = set(allow)    # 3.1 exception: biopsy_organ_coarse in the prognostic score only
        self.alt_cols, self.cov_types, self.cov_c = list(alt_cols), dict(cov_types), set(cov_c)
        self.always_keep, self.scale_cov, self.unpen_cov = tuple(always_keep), scale_cov, unpen_cov

    def fit(self, df: pd.DataFrame) -> "Design":
        bad = ((set(self.alt_cols) | set(self.cov_types)) & NEVER) - self.allow
        assert not bad, f"1. never-input columns in X: {bad}"
        self.fit_ids_ = frozenset(df.SAMPLE_ID)
        self.pre_ = PP.make_preprocessor(self.alt_cols, self.cov_types, self.always_keep).fit(df)
        cols = list(self.pre_.transform(df.iloc[:1]).columns)
        cats = sorted(self.pre_.cat_, key=len, reverse=True)
        origin, block = [], []
        for c in cols:
            pre, rest = c.split("__", 1)
            if pre == "alt":
                v, b = rest, "X"
            elif pre == "cat":
                v = next(k for k in cats if rest.startswith(k + "_"))
                b = None
            else:
                v, b = rest, None
            if b is None:
                b = "B" if v in PP.COVARIATES_B else ("C" if v in self.cov_c else "G")
            origin.append(v)
            block.append(b)
        self.columns_, self.origin_, self.block_ = cols, np.array(origin), np.array(block)
        # Covariates (block C, and block B in the dose model) are approximately unpenalized.
        # Elastic-net regression (Target B): x10 column scaling. Logistic regression (Target A):
        # exact zero penalty on these columns via PenalizedLR (`unpen_`).
        self.scale_ = np.where(np.isin(self.block_, ["C", "B"]) & self.scale_cov, COV_SCALE, 1.0)
        self.unpen_ = np.isin(self.block_, ["C", "B"]) & self.unpen_cov
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self.pre_.transform(df)[self.columns_].to_numpy(dtype=float)
        return X * self.scale_


def make_design(cohort: str, fs: str, scale_cov: bool = False, unpen_cov: bool = False) -> Design:
    alt, types, cset, keep = feature_spec(cohort, fs)
    return Design(alt, types, cset, keep, scale_cov=scale_cov, unpen_cov=unpen_cov)


class PenalizedLR:
    """Elastic-net logistic regression with an unpenalized covariate block (4.1).

    Minimizes  C * sum(logloss) + l1_ratio * |w_pen|_1 + 0.5 * (1 - l1_ratio) * |w_pen|^2
               + 0.5 * l2_unpen * |w_unpen|^2
    (sklearn's LogisticRegression objective, with per-column penalty weights). The L1 term
    is made smooth by splitting w_pen = u - v with u, v >= 0 and solved with L-BFGS-B, which
    converges far faster and more tightly than saga on this design and gives exact zeros.
    `unpen` marks block-C covariates; l2_unpen is a tiny ridge that only guards against
    divergence on rare covariate levels.
    """

    def __init__(self, C=1.0, l1_ratio=0.0, unpen=None, l2_unpen=1e-2, max_iter=5000):
        self.C, self.l1_ratio, self.unpen, self.l2_unpen, self.max_iter = C, l1_ratio, unpen, l2_unpen, max_iter

    def fit(self, X, y):
        from scipy.optimize import minimize
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n, p = X.shape
        un = np.zeros(p, bool) if self.unpen is None else np.asarray(self.unpen, bool)
        pe = ~un
        Xf, Xp = X[:, un], X[:, pe]
        F, P = Xf.shape[1], Xp.shape[1]
        Cc, a, lu = self.C, self.l1_ratio, self.l2_unpen

        def fg(th):
            b0, wf, u, v = th[0], th[1:1 + F], th[1 + F:1 + F + P], th[1 + F + P:]
            wp = u - v
            for name, arr in [("Xf", Xf), ("Xp", Xp), ("wf", wf), ("wp", wp)]:
                if not np.all(np.isfinite(arr)):
                    raise FloatingPointError(f"{name} contains NaN or Inf")

            if not np.isfinite(b0):
                raise FloatingPointError("b0 contains NaN or Inf")
            if not np.all(np.isfinite(Xf)):
                raise ValueError("Xf contains NaN or Inf")
            if not np.all(np.isfinite(Xp)):
                raise ValueError("Xp contains NaN or Inf")
            if not np.all(np.isfinite(wf)):
                raise ValueError("wf contains NaN or Inf")
            if not np.all(np.isfinite(wp)):
                raise ValueError("wp contains NaN or Inf")
            if not np.isfinite(b0):
                raise ValueError("b0 is not finite")
            with np.errstate(over="raise", invalid="raise"):
                z = b0 + Xf @ wf + Xp @ wp

            if not np.all(np.isfinite(z)):
                raise FloatingPointError("Linear predictor z contains NaN or Inf")
            f = Cc * np.sum(np.logaddexp(0, z) - y * z) + 0.5 * (1 - a) * wp @ wp + a * (u.sum() + v.sum())                 + 0.5 * lu * wf @ wf
            r = Cc * (expit(z) - y)
            gp = Xp.T @ r + (1 - a) * wp
            return f, np.concatenate([[r.sum()], Xf.T @ r + lu * wf, gp + a, -gp + a])

        x0 = np.zeros(1 + F + 2 * P)
        x0[0] = logit(np.clip(y.mean(), 1e-3, 1 - 1e-3))
        bounds = [(None, None)] * (1 + F) + [(0, None)] * (2 * P)
        res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": self.max_iter, "maxfun": 4 * self.max_iter, "gtol": 1e-7, "ftol": 1e-12})
        if not res.success:
            raise RuntimeError(
                f"PenalizedLR optimization failed: "
                f"status={res.status}, message={res.message}"
            )

        if not np.all(np.isfinite(res.x)):
            raise FloatingPointError(
                "PenalizedLR optimization returned non-finite coefficients"
            )
        th = res.x
        coef = np.zeros(p)
        coef[un] = th[1:1 + F]
        coef[pe] = th[1 + F:1 + F + P] - th[1 + F + P:]
        self.coef_, self.intercept_, self.n_iter_, self.converged_ = coef[None, :], np.array([th[0]]), res.nit, res.success
        return self

    def decision_function(self, X):
        return np.asarray(X, float) @ self.coef_[0] + self.intercept_[0]

    def predict_proba(self, X):
        q = expit(self.decision_function(X))
        return np.column_stack([1 - q, q])


# --------------------------------------------------------------------------
# Models and nested tuning (4.1, 4.3)
# --------------------------------------------------------------------------

def grid(kind: str, task: str, fixed: bool) -> list[dict]:
    if task == "clf":
        g = {
            "covonly": [{"C": c} for c in (0.01, 0.1, 1)],
            "lr": [{"C": c, "l1_ratio": l} for c in (0.01, 0.03, 0.1, 0.3, 1) for l in (0, 0.5, 1)],
            "rf": [{"max_features": m, "min_samples_leaf": s} for m in ("sqrt", 0.3) for s in (5, 20)],
            "xgb": [{"max_depth": d, "min_child_weight": w, "reg_lambda": r}
                    for d in (2, 3, 4) for w in (1, 5) for r in (1, 5)],
        }[kind]
    else:
        g = {
            "ridge_cov": [{"alpha": a} for a in (0.1, 1, 10)],
            "enet": [{"alpha": a, "l1_ratio": l} for a in (0.01, 0.1, 1) for l in (0, 0.5, 1)],
            "rf": [{"max_features": m, "min_samples_leaf": s} for m in ("sqrt", 0.3) for s in (5, 20)],
            "xgb": [{"max_depth": d, "min_child_weight": w, "reg_lambda": r}
                    for d in (2, 3, 4) for w in (1, 5) for r in (1, 5)],
        }[kind]
        if fixed:   # 4.3: NSCLC Target B uses fixed settings
            g = [{"ridge_cov": {"alpha": 1}, "enet": {"alpha": 0.1, "l1_ratio": 0.5},
                  "rf": {"max_features": "sqrt", "min_samples_leaf": 20},
                  "xgb": {"max_depth": 2, "min_child_weight": 5, "reg_lambda": 5}}[kind]]
    return g


def build_model(kind: str, task: str, p: dict, n_estimators: int | None = None, early: bool = False,
                unpen=None):
    if task == "clf":
        if kind == "covonly":
            return LogisticRegression(C=p["C"], l1_ratio=0, max_iter=5000)
        if kind == "lr":
            return PenalizedLR(C=p["C"], l1_ratio=p["l1_ratio"], unpen=unpen)
        if kind == "rf":
            return RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=SEED, **p)
        if kind == "xgb":
            return xgb.XGBClassifier(n_estimators=n_estimators or 500, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, n_jobs=1, random_state=SEED, eval_metric="logloss",
                                     early_stopping_rounds=30 if early else None, tree_method="hist", **p)
    else:
        if kind == "ridge_cov":
            return Ridge(alpha=p["alpha"])
        if kind == "enet":
            return ("enet", p)       # built at fit time (needs n for the l1_ratio = 0 case)
        if kind == "rf":
            return RandomForestRegressor(n_estimators=500, n_jobs=1, random_state=SEED, **p)
        if kind == "xgb":
            return xgb.XGBRegressor(n_estimators=n_estimators or 500, learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8, n_jobs=1, random_state=SEED,
                                    early_stopping_rounds=30 if early else None, tree_method="hist", **p)
    raise ValueError(kind)


def _fit(kind, task, p, X, y, Xval=None, yval=None, n_estimators=None, unpen=None):
    m = build_model(kind, task, p, n_estimators=n_estimators, early=Xval is not None and kind == "xgb",
                    unpen=unpen)
    if isinstance(m, tuple):     # elastic net regression; l1_ratio 0 is ridge with matched penalty
        a, l = p["alpha"], p["l1_ratio"]
        m = Ridge(alpha=a * len(y)) if l == 0 else ElasticNet(alpha=a, l1_ratio=l, max_iter=20000)
    if kind == "xgb" and Xval is not None:
        m.fit(X, y, eval_set=[(Xval, yval)], verbose=False)
    else:
        m.fit(X, y)
    return m


def predict(m, task, X):
    return m.predict_proba(X)[:, 1] if task == "clf" else m.predict(X)


def tune_fit(kind: str, task: str, df: pd.DataFrame, y: np.ndarray, design_factory, seed: int,
             fixed: bool = False):
    """Nested tuning: 3-fold patient-grouped inner CV, log-loss (clf) or MSE (reg).
    Transforms are refit on each inner training split. Returns (model, design, params)."""
    g = grid(kind, task, fixed)
    groups = df.PATIENT_ID.to_numpy()
    if task == "clf":
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(df, y, groups)
    else:
        inner = GroupKFold(3, shuffle=True, random_state=seed).split(df, y, groups)
    splits = list(inner)
    scores = np.zeros((len(g), len(splits)))
    iters = np.zeros((len(g), len(splits)))
    if len(g) > 1 or kind == "xgb":
        for j, (tr, va) in enumerate(splits):
            des = design_factory().fit(df.iloc[tr])
            Xtr, Xva = des.transform(df.iloc[tr]), des.transform(df.iloc[va])
            for i, p in enumerate(g):
                m = _fit(kind, task, p, Xtr, y[tr], Xva, y[va]) if kind == "xgb"                     else _fit(kind, task, p, Xtr, y[tr], unpen=des.unpen_)
                pr = predict(m, task, Xva)
                scores[i, j] = log_loss(y[va], np.clip(pr, 1e-6, 1 - 1e-6), labels=[0, 1]) if task == "clf" \
                    else np.mean((y[va] - pr) ** 2)
                if kind == "xgb":
                    iters[i, j] = m.best_iteration + 1
    best = int(np.argmin(scores.mean(1)))
    p = dict(g[best])
    des = design_factory().fit(df)
    X = des.transform(df)
    n_est = int(np.median(iters[best])) if kind == "xgb" else None
    m = _fit(kind, task, p, X, y, n_estimators=n_est, unpen=des.unpen_)
    if kind == "xgb":
        p["n_estimators"] = n_est
    return m, des, p


# --------------------------------------------------------------------------
# SHAP (6.1)
# --------------------------------------------------------------------------

def shap_values(kind: str, m, Xtr: np.ndarray, Xte: np.ndarray, task: str = "clf") -> np.ndarray:
    """Per-column SHAP on the model's output scale.
    LR / linear: interventional linear SHAP with the training fold as background,
      coef_j * (x_j - mean_train_j) (identical to shap.LinearExplainer); log-odds for LR.
    XGBoost: TreeSHAP (path-dependent) via the booster's pred_contribs (identical to
      shap.TreeExplainer, faster); log-odds margin for the classifier.
    Random forest: shap.TreeExplainer (probability scale)."""
    if kind in ("lr", "covonly", "ridge_cov", "enet"):
        coef = np.ravel(m.coef_)
        return (Xte - Xtr.mean(0)) * coef
    if kind == "xgb":
        return m.get_booster().predict(xgb.DMatrix(Xte), pred_contribs=True)[:, :-1]
    if kind == "rf":
        import shap
        sv = shap.TreeExplainer(m).shap_values(Xte, check_additivity=False)
        if task == "clf":
            sv = sv[..., 1] if np.ndim(sv) == 3 else sv[1]
        return sv
    raise ValueError(kind)


def aggregate_shap(sv: np.ndarray, origin: np.ndarray) -> pd.DataFrame:
    """Sum one-hot columns back to their original variable (6.2)."""
    df = pd.DataFrame(sv, columns=origin)
    return df.T.groupby(level=0, sort=False).sum().T


# --------------------------------------------------------------------------
# Metrics (5.1)
# --------------------------------------------------------------------------

def calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Calibration-in-the-large intercept (logit offset) and calibration slope."""
    import statsmodels.api as sm
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))
    try:
        inter = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    except Exception:
        inter = slope = np.nan
    return float(inter), float(slope)


def clf_metrics(y: np.ndarray, p: np.ndarray, calib: bool = False) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return {"auc": np.nan, "ap": np.nan, "brier": np.nan, "logloss": np.nan}
    r = {"auc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
         "brier": brier_score_loss(y, p), "logloss": log_loss(y, p, labels=[0, 1]),
         "prevalence": float(np.mean(y))}
    if calib:
        r["cal_intercept"], r["cal_slope"] = calibration(y, p)
    return r


def reg_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"spearman": spearmanr(y, p).statistic, "mae_days": float(np.mean(np.abs(np.expm1(y) - np.expm1(p)))),
            "r2_log": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)}


def pct_interval(x) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return (np.percentile(x, 2.5), np.percentile(x, 97.5)) if len(x) else (np.nan, np.nan)
