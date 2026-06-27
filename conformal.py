"""
Conformal prediction (Tier 2).

Calibration makes a probability trustworthy *on average*. Conformal prediction
gives a stronger, finite-sample, distribution-free guarantee on *coverage*:

    P( Y_test in C(X_test) ) >= 1 - alpha

for a prediction *set* C(x), under only the assumption that calibration and test
points are exchangeable (weaker than i.i.d.). No assumption on the model or the
data distribution is required.

Split (inductive) conformal, the variant used here:

1. On a held-out calibration set, compute a nonconformity score for each point.
   We use  s_i = 1 - p_hat(y_i | x_i)  -- one minus the model's probability of
   the *true* label. Large s_i = the model found the true label unlikely.
2. Take the conformal quantile
        q_hat = the  ceil((n+1)(1-alpha)) / n  empirical quantile of {s_i}.
   The (n+1) correction is what delivers the finite-sample guarantee.
3. For a new x, include label k in the set iff  1 - p_hat(k | x) <= q_hat,
   i.e.  p_hat(k | x) >= 1 - q_hat.

For binary spam this yields sets {}, {ham}, {spam}, or {ham, spam}:

    singleton {spam} / {ham}  -> confident prediction
    {ham, spam}               -> the model abstains (genuinely uncertain)
    {}                        -> both labels nonconforming (very atypical input)

Marginal vs class-conditional. Plain split conformal guarantees coverage
*marginally* (averaged over classes), which can hide poor coverage on the
minority class. ``MondrianConformal`` runs step 2 *per class* so coverage holds
*within each class* -- the right notion when false negatives and false positives
are valued differently.

Why this section sets up the drift section: the coverage guarantee rests on
exchangeability. Distribution drift breaks exchangeability, so under drift the
empirical coverage falls below 1 - alpha. Coverage is therefore a *model-
agnostic drift canary*: watching it drop is direct evidence of shift.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The (1-alpha) conformal quantile with the finite-sample (n+1) correction."""
    n = len(scores)
    # rank to use; clip so we never exceed the largest score (then coverage=1).
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


@dataclass
class CoverageResult:
    alpha: float
    target_coverage: float
    empirical_coverage: float
    avg_set_size: float
    singleton_rate: float
    abstain_rate: float          # P(set == {both labels})
    empty_rate: float
    coverage_ham: float
    coverage_spam: float

    def as_dict(self):
        return {
            "alpha": self.alpha,
            "target_coverage": round(self.target_coverage, 3),
            "empirical_coverage": round(self.empirical_coverage, 4),
            "avg_set_size": round(self.avg_set_size, 3),
            "singleton_rate": round(self.singleton_rate, 4),
            "abstain_rate": round(self.abstain_rate, 4),
            "empty_rate": round(self.empty_rate, 4),
            "coverage_ham": round(self.coverage_ham, 4),
            "coverage_spam": round(self.coverage_spam, 4),
        }


class SplitConformal:
    """Marginal split-conformal predictor for binary classification.

    Consumes calibrated probabilities ``p`` = P(spam) (a 1-D array). Internally
    forms the 2-column probability matrix [P(ham), P(spam)].
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    @staticmethod
    def _proba_matrix(p):
        p = np.asarray(p, dtype=float)
        return np.column_stack([1 - p, p])

    def fit(self, p_cal, y_cal):
        P = self._proba_matrix(p_cal)
        y = np.asarray(y_cal, dtype=int)
        true_p = P[np.arange(len(y)), y]
        self.scores_ = 1 - true_p
        self.qhat_ = conformal_quantile(self.scores_, self.alpha)
        return self

    def predict_set(self, p):
        """Return a boolean (n, 2) inclusion matrix over [ham, spam]."""
        P = self._proba_matrix(p)
        nonconf = 1 - P
        return nonconf <= self.qhat_

    def evaluate(self, p, y) -> CoverageResult:
        return _coverage_report(self.predict_set(p), np.asarray(y), self.alpha)


class MondrianConformal:
    """Class-conditional split conformal: a separate q_hat per class.

    Guarantees coverage within each true class, at the cost of a (slightly)
    larger average set size.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def fit(self, p_cal, y_cal):
        P = SplitConformal._proba_matrix(p_cal)
        y = np.asarray(y_cal, dtype=int)
        self.qhat_ = {}
        for c in (0, 1):
            sc = 1 - P[y == c, c]
            self.qhat_[c] = conformal_quantile(sc, self.alpha)
        return self

    def predict_set(self, p):
        P = SplitConformal._proba_matrix(p)
        nonconf = 1 - P
        inc = np.zeros_like(P, dtype=bool)
        for c in (0, 1):
            inc[:, c] = nonconf[:, c] <= self.qhat_[c]
        return inc

    def evaluate(self, p, y) -> CoverageResult:
        return _coverage_report(self.predict_set(p), np.asarray(y), self.alpha)


def _coverage_report(inc: np.ndarray, y: np.ndarray, alpha: float) -> CoverageResult:
    covered = inc[np.arange(len(y)), y]
    set_size = inc.sum(axis=1)
    cov_ham = covered[y == 0].mean() if (y == 0).any() else float("nan")
    cov_spam = covered[y == 1].mean() if (y == 1).any() else float("nan")
    return CoverageResult(
        alpha=alpha,
        target_coverage=1 - alpha,
        empirical_coverage=float(covered.mean()),
        avg_set_size=float(set_size.mean()),
        singleton_rate=float((set_size == 1).mean()),
        abstain_rate=float((set_size == 2).mean()),
        empty_rate=float((set_size == 0).mean()),
        coverage_ham=float(cov_ham),
        coverage_spam=float(cov_spam),
    )
