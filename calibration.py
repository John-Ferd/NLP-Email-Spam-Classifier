"""
Probability calibration (Tier 1).

A model can rank perfectly (AUC = 1) yet output probabilities that are wildly
wrong. Calibration fixes the *mapping from score to probability* on a held-out
calibration set, without touching the ranking. Three methods are provided:

* ``platt``        p = sigmoid(a * s + b).  Two parameters fit by logistic
                   regression on the calibration scores. The standard choice for
                   SVM margins and any 1-D score. Being strictly monotone in s,
                   it CANNOT change the ROC-AUC -- it only relabels scores as
                   probabilities.

* ``temperature``  p = sigmoid(s / T).  Platt with the bias fixed to 0 and the
                   slope constrained to 1/T (one parameter). Appropriate when the
                   score is already a centred logit (e.g. a neural net), less so
                   for a generic margin. Included for comparison.

* ``isotonic``     p = m(s) for the monotone non-decreasing function m that best
                   fits the calibration data (pool-adjacent-violators). Non-
                   parametric and more flexible than Platt, but data-hungry: with
                   few calibration points it overfits. Enron's large calibration
                   split makes it viable here.

Why a single scalar (temperature/Platt) preserves AUC: AUC depends only on the
ordering of scores; a strictly increasing transform preserves order, hence AUC.
Isotonic is only weakly monotone (it can tie scores), so it can perturb AUC
negligibly through ties.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _to_logit(s, eps=1e-6):
    """Map a [0,1] probability to a logit; pass through values already real."""
    s = np.asarray(s, dtype=float)
    if s.min() >= 0.0 and s.max() <= 1.0:
        s = np.clip(s, eps, 1 - eps)
        return np.log(s / (1 - s))
    return s


class Calibrator:
    """Fit a 1-D calibration map and apply it.

    Parameters
    ----------
    method : {"platt", "temperature", "isotonic"}
    """

    def __init__(self, method: str = "platt"):
        self.method = method.lower()

    def fit(self, scores, y):
        s = np.asarray(scores, dtype=float)
        y = np.asarray(y, dtype=int)

        if self.method == "platt":
            z = _to_logit(s).reshape(-1, 1)
            self._lr = LogisticRegression(C=1e6, solver="lbfgs")
            self._lr.fit(z, y)

        elif self.method == "temperature":
            z = _to_logit(s)

            def nll(log_t):
                T = np.exp(log_t)  # keep T > 0
                p = 1.0 / (1.0 + np.exp(-z / T))
                p = np.clip(p, 1e-12, 1 - 1e-12)
                return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

            res = minimize_scalar(nll, bounds=(-4, 4), method="bounded")
            self._T = float(np.exp(res.x))

        elif self.method == "isotonic":
            self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            self._iso.fit(s, y)

        else:
            raise ValueError(f"unknown calibration method {self.method!r}")
        return self

    def predict_proba(self, scores):
        s = np.asarray(scores, dtype=float)
        if self.method == "platt":
            z = _to_logit(s).reshape(-1, 1)
            return self._lr.predict_proba(z)[:, 1]
        if self.method == "temperature":
            z = _to_logit(s)
            return 1.0 / (1.0 + np.exp(-z / self._T))
        return self._iso.predict(s)


def fit_best_calibrator(scores, y, methods=("platt", "isotonic", "temperature"),
                        n_bins=15):
    """Fit several calibrators and return the one with lowest ECE on the cal set.

    NOTE: selecting on the same calibration data used to fit is acceptable for
    choosing *between* low-capacity calibrators but the reported ECE must be
    measured on the held-out TEST set. The README spells this out.
    """
    from .evaluate import expected_calibration_error

    best, best_ece = None, np.inf
    table = {}
    for m in methods:
        cal = Calibrator(m).fit(scores, y)
        ece = expected_calibration_error(y, cal.predict_proba(scores), n_bins)
        table[m] = ece
        if ece < best_ece:
            best, best_ece, best_name = cal, ece, m
    return best, best_name, table
