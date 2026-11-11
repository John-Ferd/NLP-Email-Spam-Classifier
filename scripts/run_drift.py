"""Drift experiment on an injected step concept-drift stream.

The stream is drawn from the TEST split only (never seen in training). Writes
artifacts/drift.json, artifacts/drift_auc.png and artifacts/drift_coverage.png.
"""
import matplotlib
from _common import parse_args

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spamguard.conformal import SplitConformal
from spamguard.data import make_drift_stream
from spamguard.decision import bayes_threshold
from spamguard.drift import run_drift_experiment
from spamguard.models import raw_scores
from spamguard.pipeline import ARTIFACTS, fit_base, fit_calibrator, get_splits, halve_cal, load_config, save_json

args = parse_args(__doc__)
cfg = load_config(args.config)
d = cfg["drift"]
splits = get_splits(cfg, args.smoke)
cal_a, cal_b = halve_cal(splits, cfg["data"]["seed"])

fitted = fit_base(cfg, splits.train)
cal, cal_name, _ = fit_calibrator(cfg, fitted, cal_a)
p_b = cal.predict_proba(fitted.scores(cal_b["text"]))
conf = SplitConformal(cfg["conformal"]["alpha"]).fit(p_b, cal_b["label"].to_numpy())

n = d["n"] if not args.smoke else 2000
stream, cp = make_drift_stream(splits.test, n=n, change_point=d["change_point"],
                               max_obfuscation=d["max_obfuscation"], seed=cfg["data"]["seed"])
tau = bayes_threshold(d["c_fp"], d["c_fn"])
res = run_drift_experiment(
    fitted.model, cal, conf, stream, cp, threshold=tau,
    window=d["window"], step=d["step"], retrain_window=d["retrain_window"],
    alpha=cfg["conformal"]["alpha"],
    get_scores=lambda m, X: raw_scores(m, list(X), fitted.scaler),
)
summary = res.summary()
summary["all_drift_flags"] = res.drift_flags
summary["threshold"] = round(tau, 4)
save_json(summary, "drift.json")

plots = [
    ("drift_auc.png", [("frozen model", res.auc), ("window-retrained", res.retrain_auc)], "ROC-AUC (sliding window)"),
    ("drift_coverage.png", [("split-conformal coverage", res.coverage)], "Empirical coverage"),
]
for name, series, ylabel in plots:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for lab, s in series:
        ax.plot(res.index, s, label=lab)
    ax.axvline(cp, color="red", lw=1, label="injected change point")
    if name == "drift_coverage.png":
        ax.axhline(1 - cfg["conformal"]["alpha"], color="gray", ls="--", lw=1, label="target 1 - alpha")
    ax.set_xlabel("stream index")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / name, dpi=130)
    plt.close(fig)
