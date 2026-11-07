import numpy as np

from spamguard.decision import analyze_costs, bayes_threshold, expected_cost


def test_bayes_threshold_formula():
    assert bayes_threshold(10, 1) == 10 / 11
    assert bayes_threshold(1, 1) == 0.5


def test_expected_cost_counts_errors():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.9, 0.1, 0.2, 0.8])     # one FP, one FN at threshold 0.5
    assert expected_cost(y, p, 0.5, c_fp=10, c_fn=1) == (10 + 1) / 4


def test_tau_star_is_near_grid_optimum_for_calibrated_probs():
    rng = np.random.default_rng(0)
    p = rng.random(200_000)
    y = (rng.random(p.size) < p).astype(int)   # perfectly calibrated
    res = analyze_costs(y, p, c_fp=10, c_fn=1)
    assert abs(res.best_grid_threshold - res.tau_star) < 0.02
    assert res.cost_at_tau_star < res.cost_at_half
