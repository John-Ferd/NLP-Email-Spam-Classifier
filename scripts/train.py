"""Fit the base model + calibrator; report raw vs calibrated metrics and cost analysis.

Writes artifacts/train_metrics.json and artifacts/reliability.png.
"""
from _common import parse_args

from spamguard.decision import analyze_costs
from spamguard.evaluate import reliability_plot, summarize
from spamguard.pipeline import ARTIFACTS, fit_base, fit_calibrator, get_splits, halve_cal, load_config, save_json

args = parse_args(__doc__)
cfg = load_config(args.config)
splits = get_splits(cfg, args.smoke)
cal_a, _ = halve_cal(splits, cfg["data"]["seed"])
print(splits.describe().to_string(index=False))

fitted = fit_base(cfg, splits.train)
cal, cal_name, cal_table = fit_calibrator(cfg, fitted, cal_a)

y = splits.test["label"].to_numpy()
raw = fitted.scores(splits.test["text"])
p = cal.predict_proba(raw)
n_bins = cfg["calibration"]["n_bins"]

out = {
    "model": fitted.kind,
    "calibrator": cal_name,
    "calibrator_ece_on_cal_a": {k: round(v, 4) for k, v in cal_table.items()},
    "test_raw": summarize(y, raw, n_bins),
    "test_calibrated": summarize(y, p, n_bins),
    "costs": [analyze_costs(y, p, c["c_fp"], c["c_fn"]).as_dict() for c in cfg["costs"]],
}
save_json(out, "train_metrics.json")
reliability_plot({f"{fitted.kind} raw": (y, raw), f"{fitted.kind} + {cal_name}": (y, p)},
                 ARTIFACTS / "reliability.png", n_bins, title=f"{fitted.kind}: before/after calibration")
