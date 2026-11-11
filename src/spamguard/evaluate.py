"""
Evaluation metrics.

Discrimination (ranking) metrics are reported for completeness; the metrics the
project cares about measure whether the *probabilities* are trustworthy:
Brier score, Expected Calibration Error (ECE) and Maximum Calibration Error
(MCE). ECE and MCE are implemented from scratch below.

ECE is mass-weighted, so it can look good while a sparsely populated,
high-confidence bin is badly wrong. MCE reports that worst bin, which is why
both are reported.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def _bins(p, n_bins):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-closed bins, with p == 0 placed in the first bin
    return np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)


def calibration_table(y, p, n_bins: int = 15):
    """Per-bin (mean confidence, empirical frequency, count)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    b = _bins(p, n_bins)
    conf, freq, count = [], [], []
    for k in range(n_bins):
        m = b == k
        count.append(int(m.sum()))
        conf.append(float(p[m].mean()) if m.any() else np.nan)
        freq.append(float(y[m].mean()) if m.any() else np.nan)
    return np.array(conf), np.array(freq), np.array(count)


def expected_calibration_error(y, p, n_bins: int = 15) -> float:
    """ECE = sum_k (n_k / n) * |freq_k - conf_k| over non-empty bins."""
    conf, freq, count = calibration_table(y, p, n_bins)
    m = count > 0
    return float(np.sum(count[m] / count.sum() * np.abs(freq[m] - conf[m])))


def max_calibration_error(y, p, n_bins: int = 15) -> float:
    """MCE = max_k |freq_k - conf_k| over non-empty bins."""
    conf, freq, count = calibration_table(y, p, n_bins)
    m = count > 0
    return float(np.max(np.abs(freq[m] - conf[m])))


def precision_at_recall(y, p, recall: float = 0.99) -> float:
    """Precision at the highest threshold whose recall is >= ``recall``."""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(p), kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    k = int(np.searchsorted(tp, recall * ys.sum(), side="left"))
    return float(tp[k] / (k + 1))


def fpr_at_tpr(y, p, tpr_target: float = 0.99) -> float:
    """Smallest FPR achieving TPR >= ``tpr_target``."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(fpr[np.argmax(tpr >= tpr_target)])


def summarize(y, p, n_bins: int = 15) -> dict:
    """All reported metrics for one set of probabilities."""
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "ece": round(expected_calibration_error(y, p, n_bins), 4),
        "mce": round(max_calibration_error(y, p, n_bins), 4),
        "precision_at_99_recall": round(precision_at_recall(y, p, 0.99), 4),
        "fpr_at_99_tpr": round(fpr_at_tpr(y, p, 0.99), 4),
    }


def reliability_plot(curves: dict, path, n_bins: int = 15, title: str = "Reliability diagram"):
    """Save a reliability diagram. ``curves`` maps label -> (y, p)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for label, (y, p) in curves.items():
        conf, freq, count = calibration_table(y, p, n_bins)
        m = count > 0
        ece = expected_calibration_error(y, p, n_bins)
        ax.plot(conf[m], freq[m], "o-", label=f"{label} (ECE {ece:.3f})")
    ax.set_xlabel("Mean predicted P(spam)")
    ax.set_ylabel("Observed spam frequency")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
