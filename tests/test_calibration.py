import numpy as np
from sklearn.metrics import roc_auc_score

from spamguard.calibration import Calibrator
from spamguard.evaluate import expected_calibration_error


def test_platt_fixes_miscalibration_without_changing_auc():
    rng = np.random.default_rng(0)
    p_true = rng.random(20_000)
    y = (rng.random(p_true.size) < p_true).astype(int)
    raw = p_true ** 4                         # monotone distortion -> miscalibrated
    cal = Calibrator("platt").fit(raw[:10_000], y[:10_000])
    p = cal.predict_proba(raw[10_000:])
    yt = y[10_000:]
    assert expected_calibration_error(yt, p) < expected_calibration_error(yt, raw[10_000:]) / 3
    # Platt is strictly monotone; the only possible AUC change comes from
    # clipping scores below 1e-6 into ties, which is negligible.
    assert abs(roc_auc_score(yt, p) - roc_auc_score(yt, raw[10_000:])) < 1e-4
