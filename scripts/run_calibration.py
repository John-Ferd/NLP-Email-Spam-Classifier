"""Compare calibration across nb / logreg / svm on the test split.

Writes artifacts/calibration_comparison.json and artifacts/reliability_svm.png.
"""
from _common import parse_args
from sklearn.metrics import roc_auc_score

from spamguard.evaluate import expected_calibration_error, max_calibration_error, reliability_plot
from spamguard.pipeline import ARTIFACTS, fit_base, fit_calibrator, get_splits, halve_cal, load_config, save_json

args = parse_args(__doc__)
cfg = load_config(args.config)
splits = get_splits(cfg, args.smoke)
cal_a, _ = halve_cal(splits, cfg["data"]["seed"])
y = splits.test["label"].to_numpy()
n_bins = cfg["calibration"]["n_bins"]

rows = {}
for kind in ("nb", "logreg", "svm"):
    fitted = fit_base(cfg, splits.train, kind)
    cal, cal_name, _ = fit_calibrator(cfg, fitted, cal_a)
    raw = fitted.scores(splits.test["text"])
    p = cal.predict_proba(raw)
    rows[kind] = {
        "calibrator": cal_name,
        "raw_ece": round(expected_calibration_error(y, raw, n_bins), 4),
        "cal_ece": round(expected_calibration_error(y, p, n_bins), 4),
        "raw_mce": round(max_calibration_error(y, raw, n_bins), 4),
        "cal_mce": round(max_calibration_error(y, p, n_bins), 4),
        "raw_auc": round(roc_auc_score(y, raw), 4),
        "cal_auc": round(roc_auc_score(y, p), 4),
    }
    if kind == "svm":
        reliability_plot({"svm margin (min-max scaled)": (y, raw), f"svm + {cal_name}": (y, p)},
                         ARTIFACTS / "reliability_svm.png", n_bins, title="SVM: margin vs calibrated")
save_json(rows, "calibration_comparison.json")
