import numpy as np

from spamguard.conformal import MondrianConformal, SplitConformal, conformal_quantile


def _exchangeable(n, seed):
    """Calibrated synthetic probabilities: y ~ Bernoulli(p)."""
    rng = np.random.default_rng(seed)
    p = rng.beta(0.5, 0.5, size=n)
    y = (rng.random(n) < p).astype(int)
    return p, y


def test_conformal_quantile_uses_n_plus_one_correction():
    s = np.arange(1, 11) / 10.0          # n = 10
    assert conformal_quantile(s, 0.2) == 0.9   # ceil(11 * 0.8) = 9th smallest
    assert conformal_quantile(s, 0.05) == np.inf  # ceil(11 * 0.95) = 11 > n


def test_split_conformal_coverage_holds_on_average():
    covs = []
    for seed in range(20):
        p_cal, y_cal = _exchangeable(500, seed)
        p_te, y_te = _exchangeable(2000, seed + 100)
        covs.append(SplitConformal(0.1).fit(p_cal, y_cal).evaluate(p_te, y_te).empirical_coverage)
    assert np.mean(covs) >= 0.9 - 0.01


def test_mondrian_covers_each_class():
    p_cal, y_cal = _exchangeable(3000, 0)
    p_te, y_te = _exchangeable(5000, 1)
    res = MondrianConformal(0.1).fit(p_cal, y_cal).evaluate(p_te, y_te)
    assert res.coverage_ham >= 0.88 and res.coverage_spam >= 0.88
