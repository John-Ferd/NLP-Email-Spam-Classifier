"""Split and Mondrian conformal prediction on the test split.

The calibrator is fit on cal_a; the conformal quantile on cal_b (disjoint).
The coverage guarantee holds on average over the random calibration draw, so
besides the seed-0 split we also repeat the cal_a/cal_b halving 30 times and
report the mean and spread. Writes artifacts/conformal.json.
"""
import numpy as np
from _common import parse_args
from sklearn.model_selection import train_test_split

from spamguard.calibration import fit_best_calibrator
from spamguard.conformal import MondrianConformal, SplitConformal
from spamguard.pipeline import fit_base, fit_calibrator, get_splits, halve_cal, load_config, save_json

args = parse_args(__doc__)
cfg = load_config(args.config)
splits = get_splits(cfg, args.smoke)
cal_a, cal_b = halve_cal(splits, cfg["data"]["seed"])
alpha = cfg["conformal"]["alpha"]

fitted = fit_base(cfg, splits.train)
cal, cal_name, _ = fit_calibrator(cfg, fitted, cal_a)
p_b = cal.predict_proba(fitted.scores(cal_b["text"]))
p_test = cal.predict_proba(fitted.scores(splits.test["text"]))
y_b, y_test = cal_b["label"].to_numpy(), splits.test["label"].to_numpy()

split = SplitConformal(alpha).fit(p_b, y_b)
mond = MondrianConformal(alpha).fit(p_b, y_b)
# Repeat the calibration halving to see the coverage distribution, not one draw.
s_cal, y_cal = fitted.scores(splits.cal["text"]), splits.cal["label"].to_numpy()
s_test = fitted.scores(splits.test["text"])
reps = {"split": [], "mondrian": []}
for rep in range(30):
    ia, ib = train_test_split(np.arange(len(y_cal)), test_size=0.5, stratify=y_cal, random_state=rep)
    c, _, _ = fit_best_calibrator(s_cal[ia], y_cal[ia], tuple(cfg["calibration"]["methods"]),
                                  cfg["calibration"]["n_bins"])
    pb, pt = c.predict_proba(s_cal[ib]), c.predict_proba(s_test)
    for name, cls in (("split", SplitConformal), ("mondrian", MondrianConformal)):
        r = cls(alpha).fit(pb, y_cal[ib]).evaluate(pt, y_test)
        reps[name].append((r.empirical_coverage, r.coverage_ham, r.coverage_spam))
repeated = {
    name: {
        "mean_coverage": round(float(np.mean([v[0] for v in vals])), 4),
        "sd_coverage": round(float(np.std([v[0] for v in vals])), 4),
        "min_coverage": round(float(np.min([v[0] for v in vals])), 4),
        "mean_coverage_ham": round(float(np.mean([v[1] for v in vals])), 4),
        "mean_coverage_spam": round(float(np.mean([v[2] for v in vals])), 4),
    }
    for name, vals in reps.items()
}

save_json({
    "model": fitted.kind,
    "calibrator": cal_name,
    "qhat_split": round(split.qhat_, 5),
    "qhat_mondrian": {int(k): round(v, 5) for k, v in mond.qhat_.items()},
    "frac_test_probs_saturated": round(float(((p_test == 0) | (p_test == 1)).mean()), 4),
    "split": split.evaluate(p_test, y_test).as_dict(),
    "mondrian": mond.evaluate(p_test, y_test).as_dict(),
    "repeated_30_cal_splits": repeated,
}, "conformal.json")
