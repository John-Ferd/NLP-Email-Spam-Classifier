import numpy as np

from spamguard.drift import DDM


def _first_drift(errors):
    ddm = DDM()
    for i, e in enumerate(errors):
        if ddm.update(int(e)) == "drift":
            return i
    return None


def test_ddm_never_flags_an_error_free_stream():
    # Regression test: with ">=" a perfect classifier (p = s = 0) tripped the
    # drift rule every 100 samples.
    assert _first_drift(np.zeros(5000)) is None


def test_ddm_detects_abrupt_change_quickly():
    rng = np.random.default_rng(0)
    errs = np.concatenate([rng.random(2000) < 0.05, rng.random(1000) < 0.40])
    first = _first_drift(errs)
    assert first is not None and 2000 <= first <= 2200
