"""Firth-penalized logistic regression (modeling_plan.md Section 7.2).

`firthlogist` has no release for Python >= 3.11 and statsmodels has no Firth
penalty, so this is a self-contained NumPy implementation following Heinze &
Schemper (2002) and R `logistf`:

* Penalized log-likelihood  l*(b) = l(b) + 0.5 log det(X' W X).
* Modified score  U*(b) = X' (y - p + h (0.5 - p)), h = diagonal of the hat matrix.
* Newton-Raphson with step halving on l*.
* Profile penalized-likelihood CIs and likelihood-ratio p-values: a coefficient is
  fixed and the others are re-maximized, with the penalty always computed from the
  full design (as in logistf). If a profile cannot be bracketed, Wald CIs / p-values
  are used and the fit records `ci_type = "wald"`.

Validated by tests/test_firth.py (assert 9).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import chi2, norm

CRIT = chi2.ppf(0.95, 1)


def _pll(X, y, b):
    z = X @ b
    p = expit(z)
    w = p * (1 - p)
    sign, logdet = np.linalg.slogdet((X * w[:, None]).T @ X)
    return np.sum(y * z - np.logaddexp(0, z)) + 0.5 * logdet if sign > 0 else -np.inf


def _fit(X, y, b0=None, fixed=None, max_iter=100, tol=1e-8):
    """Maximize l*; `fixed` = boolean mask of coefficients held at their b0 value."""
    n, k = X.shape
    b = np.zeros(k) if b0 is None else b0.astype(float).copy()
    free = np.ones(k, bool) if fixed is None else ~fixed
    ll = _pll(X, y, b)
    for it in range(max_iter):
        p = expit(X @ b)
        w = p * (1 - p)
        info = (X * w[:, None]).T @ X
        try:
            cov = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(info)
        h = np.einsum("ij,jk,ik->i", X * np.sqrt(w)[:, None], cov, X * np.sqrt(w)[:, None])
        U = X.T @ (y - p + h * (0.5 - p))
        Uf = U[free]
        step = np.zeros(k)
        step[free] = np.linalg.solve(info[np.ix_(free, free)], Uf)
        step = np.clip(step, -5, 5)
        t = 1.0
        while True:
            nb = b + t * step
            nll = _pll(X, y, nb)
            if nll >= ll - 1e-12 or t < 1e-8:
                break
            t /= 2
        b, dll = nb, nll - ll
        ll = nll
        if np.max(np.abs(t * step)) < tol and abs(dll) < 1e-10:
            break
    return b, ll, it + 1


def firth_logit(X, y, names=None, profile=(1,), max_iter=100) -> dict:
    """Fit Firth logistic regression. X must include an intercept column.
    `profile` = indices of coefficients that get profile CIs and PLR p-values; the
    others get Wald CIs only."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    b, ll, nit = _fit(X, y, max_iter=max_iter)
    p = expit(X @ b)
    info = (X * (p * (1 - p))[:, None]).T @ X
    se = np.sqrt(np.diag(np.linalg.pinv(info)))
    k = X.shape[1]
    lo, hi, pv, ctype = b - norm.ppf(0.975) * se, b + norm.ppf(0.975) * se, 2 * norm.sf(np.abs(b / se)), ["wald"] * k
    for j in profile:
        def prof(v, j=j):
            b0 = b.copy()
            b0[j] = v
            mask = np.zeros(k, bool)
            mask[j] = True
            return _fit(X, y, b0=b0, fixed=mask, max_iter=max_iter)[1]

        def g(v):
            return 2 * (ll - prof(v)) - CRIT
        try:
            bounds = []
            for sgn in (-1, 1):
                width = max(2 * se[j], 0.5)
                for _ in range(12):
                    edge = b[j] + sgn * width
                    if g(edge) > 0:
                        break
                    width *= 2
                else:
                    raise RuntimeError("profile not bracketed")
                bounds.append(brentq(g, *sorted([b[j], edge]), xtol=1e-6))
            lo[j], hi[j] = bounds
            stat = max(2 * (ll - prof(0.0)), 0.0)
            pv[j] = chi2.sf(stat, 1)
            ctype[j] = "profile"
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            pass
    names = names or [f"x{i}" for i in range(k)]
    return {"coef": dict(zip(names, b)), "se": dict(zip(names, se)), "ci_low": dict(zip(names, lo)),
            "ci_high": dict(zip(names, hi)), "p": dict(zip(names, pv)), "ci_type": dict(zip(names, ctype)),
            "loglik_pen": ll, "n_iter": nit}
