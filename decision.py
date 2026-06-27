"""
Cost-sensitive decision making (Tier 1).

A classifier outputs a probability; a *decision* requires a loss function. Under
an asymmetric cost matrix the optimal threshold is generally not 0.5.

Setup (positive class = spam). Let p = P(spam | x). Costs:

    c_fp = cost of a false positive  (ham flagged as spam -- a lost real email)
    c_fn = cost of a false negative  (spam delivered -- an annoyance)

Expected cost of each action at a point with spam-probability p:

    predict "spam":   (1 - p) * c_fp     # pay c_fp iff it is really ham
    predict "ham" :       p   * c_fn     # pay c_fn iff it is really spam

We choose "spam" iff its expected cost is lower:

    (1 - p) * c_fp  <  p * c_fn
    c_fp            <  p * (c_fp + c_fn)
    p               >  c_fp / (c_fp + c_fn)   :=  tau*

So the Bayes-optimal threshold is

    tau* = c_fp / (c_fp + c_fn).

With c_fp = 10, c_fn = 1 this is 10/11 ~ 0.909: only flag spam when >90.9%
confident, because deleting a real email is 10x worse than letting spam through.

This formula is ONLY valid for a calibrated p. Feeding raw, over-confident
Naive-Bayes scores into it gives the wrong threshold -- which is exactly why
calibration (the previous module) is a prerequisite for cost-sensitive
decisions, not an optional nicety.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def bayes_threshold(c_fp: float, c_fn: float) -> float:
    """Closed-form Bayes-optimal probability threshold."""
    return c_fp / (c_fp + c_fn)


def expected_cost(y_true, p, threshold, c_fp: float, c_fn: float) -> float:
    """Average realized cost when deciding spam iff p >= threshold."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(p) >= threshold).astype(int)
    fp = np.sum((pred == 1) & (y_true == 0))
    fn = np.sum((pred == 0) & (y_true == 1))
    return float((fp * c_fp + fn * c_fn) / len(y_true))


@dataclass
class CostAnalysis:
    c_fp: float
    c_fn: float
    tau_star: float
    cost_at_tau_star: float
    cost_at_half: float
    best_grid_threshold: float
    best_grid_cost: float

    def as_dict(self):
        return {
            "c_fp": self.c_fp,
            "c_fn": self.c_fn,
            "tau_star": round(self.tau_star, 4),
            "cost_at_tau_star": round(self.cost_at_tau_star, 5),
            "cost_at_0.5": round(self.cost_at_half, 5),
            "best_grid_threshold": round(self.best_grid_threshold, 4),
            "best_grid_cost": round(self.best_grid_cost, 5),
            "savings_vs_0.5_pct": round(
                100 * (self.cost_at_half - self.cost_at_tau_star)
                / max(self.cost_at_half, 1e-12),
                2,
            ),
        }


def analyze_costs(y_true, p, c_fp: float, c_fn: float, grid: int = 501) -> CostAnalysis:
    """Compare the closed-form threshold against 0.5 and against a grid search.

    The grid search is a sanity check: with a *calibrated* p, the empirical
    cost-minimising threshold should sit close to the closed-form tau*. A large
    gap is a red flag that the probabilities are miscalibrated.
    """
    tau = bayes_threshold(c_fp, c_fn)
    thresholds = np.linspace(0.0, 1.0, grid)
    costs = np.array([expected_cost(y_true, p, t, c_fp, c_fn) for t in thresholds])
    j = int(np.argmin(costs))
    return CostAnalysis(
        c_fp=c_fp,
        c_fn=c_fn,
        tau_star=tau,
        cost_at_tau_star=expected_cost(y_true, p, tau, c_fp, c_fn),
        cost_at_half=expected_cost(y_true, p, 0.5, c_fp, c_fn),
        best_grid_threshold=float(thresholds[j]),
        best_grid_cost=float(costs[j]),
    )
