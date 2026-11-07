"""Validation of firth.py (modeling_plan.md Section 7.2, assert 9).

(a) Large non-separated sample: Firth agrees with maximum likelihood (statsmodels Logit).
(b) Complete separation: estimates and both profile-CI bounds are finite.
(c) Exact analytic check, replacing the plan's logistf endometrial example (that dataset
    is not available offline): with a single binary covariate the model is saturated and
    Firth's estimate equals the log odds ratio with 0.5 added to every cell of the 2x2
    table (Haldane correction); the intercept equals the corrected logit of the
    unexposed group.

Run: ``python tests/test_firth.py`` (or pytest).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from firth import firth_logit  # noqa: E402


def test_large_sample_matches_mle():
    import statsmodels.api as sm
    rng = np.random.default_rng(1)
    n = 5000
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.binomial(1, 0.3, n)])
    beta = np.array([-0.5, 0.8, -1.0])
    y = rng.binomial(1, 1 / (1 + np.exp(-X @ beta)))
    f = firth_logit(X, y, profile=(1, 2))
    mle = sm.Logit(y, X).fit(disp=0).params
    est = np.array(list(f["coef"].values()))
    assert np.all(np.abs(est - mle) <= np.maximum(0.01 * np.abs(mle), 0.01)), (est, mle)
    assert all(f["ci_type"][k] == "profile" for k in ("x1", "x2"))


def test_separation_finite():
    x = np.array([0] * 20 + [1] * 10, float)
    y = np.array([0] * 15 + [1] * 5 + [1] * 10, float)   # x = 1 always y = 1: complete separation
    X = np.column_stack([np.ones_like(x), x])
    f = firth_logit(X, y, profile=(1,))
    assert np.isfinite(f["coef"]["x1"]) and f["coef"]["x1"] > 0
    assert f["ci_type"]["x1"] == "profile"
    assert np.isfinite(f["ci_low"]["x1"]) and np.isfinite(f["ci_high"]["x1"])
    assert f["ci_low"]["x1"] < f["coef"]["x1"] < f["ci_high"]["x1"]


def test_two_by_two_haldane():
    for a, b, c, d in [(3, 17, 40, 140), (0, 25, 12, 300), (9, 1, 30, 60)]:
        # a: exposed & y=1, b: exposed & y=0, c: unexposed & y=1, d: unexposed & y=0
        x = np.r_[np.ones(a + b), np.zeros(c + d)]
        y = np.r_[np.ones(a), np.zeros(b), np.ones(c), np.zeros(d)]
        f = firth_logit(np.column_stack([np.ones_like(x), x]), y, profile=(1,))
        want = np.log((a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5)))
        want0 = np.log((c + 0.5) / (d + 0.5))
        assert abs(f["coef"]["x1"] - want) < 1e-6, (f["coef"]["x1"], want)
        assert abs(f["coef"]["x0"] - want0) < 1e-6


def run_all() -> None:
    for t in (test_large_sample_matches_mle, test_separation_finite, test_two_by_two_haldane):
        t()
        print(f"  [ok] firth: {t.__name__}")


if __name__ == "__main__":
    run_all()
