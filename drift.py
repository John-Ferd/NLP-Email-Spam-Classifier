"""
Drift detection (Tier 2).

Because the Enron date axis is confounded with the label (see ``data.py``), we
cannot use natural chronology to study concept drift. Instead we evaluate on a
stream with an *injected* change point of known location and measure:

* streaming AUC of a frozen model over a sliding window -- the analogue of alpha
  decay: a model trained on yesterday's distribution losing edge as the world
  moves;
* conformal coverage over the same stream -- the model-agnostic canary from the
  conformal module: coverage should sit at 1 - alpha before the change and fall
  after it, because drift breaks exchangeability;
* an explicit change detector on the error stream (DDM, implemented from
  scratch) that flags a drift point, giving detection delay and false-alarm rate
  against the known ground truth.

We also include a *sliding-window retrain* baseline to quantify the cost of NOT
adapting. Note: sophisticated online learners with regret guarantees are
deliberately out of scope for this project; the point here is detection and
quantification of damage, plus a simple reference response.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------- #
# DDM -- Drift Detection Method (Gama et al., 2004), from scratch.
# --------------------------------------------------------------------------- #
@dataclass
class DDM:
    """Monitor a stream of 0/1 errors; flag warning / drift on error increase.

    Models the error rate of a binary signal as a Bernoulli process. Tracks the
    online error rate p_t and its std s_t = sqrt(p_t (1-p_t) / t). For a learner
    whose performance is stable or improving, p_t + s_t should not grow. DDM
    records the running minimum (p_min + s_min) and signals:

        warning  when  p_t + s_t >= p_min + warning_level * s_min
        drift    when  p_t + s_t >= p_min + drift_level   * s_min

    Defaults warning_level=2, drift_level=3 (the canonical 2-sigma / 3-sigma
    rule). After a drift signal the statistics reset.
    """

    warning_level: float = 2.0
    drift_level: float = 3.0
    min_samples: int = 100   # warm-up: let the Bernoulli std settle before arming

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.p = 0.0
        self.p_min = np.inf
        self.s_min = np.inf
        self.warning_zone = False

    def update(self, error: int) -> str:
        """Feed one 0/1 error; return '', 'warning', or 'drift'."""
        self.n += 1
        # incremental mean of the error indicator
        self.p += (error - self.p) / self.n
        s = np.sqrt(self.p * (1 - self.p) / self.n)

        if self.n < self.min_samples:
            return ""

        if self.p + s <= self.p_min + self.s_min:
            self.p_min, self.s_min = self.p, s

        if self.p + s >= self.p_min + self.drift_level * self.s_min:
            self.reset()
            return "drift"
        if self.p + s >= self.p_min + self.warning_level * self.s_min:
            self.warning_zone = True
            return "warning"
        self.warning_zone = False
        return ""


# --------------------------------------------------------------------------- #
# Streaming evaluation
# --------------------------------------------------------------------------- #
@dataclass
class StreamResult:
    index: np.ndarray
    auc: np.ndarray
    coverage: np.ndarray
    retrain_auc: np.ndarray
    drift_flags: list = field(default_factory=list)
    first_drift: int | None = None
    change_point: int | None = None

    def summary(self) -> dict:
        delay = (
            self.first_drift - self.change_point
            if (self.first_drift is not None and self.change_point is not None)
            else None
        )
        pre = self.auc[self.index < (self.change_point or 0)]
        post = self.auc[self.index >= (self.change_point or 0)]
        cov_pre = self.coverage[self.index < (self.change_point or 0)]
        cov_post = self.coverage[self.index >= (self.change_point or 0)]
        return {
            "change_point": self.change_point,
            "first_drift_flag": self.first_drift,
            "detection_delay": delay,
            "auc_pre_mean": round(float(np.nanmean(pre)), 4) if len(pre) else None,
            "auc_post_min": round(float(np.nanmin(post)), 4) if len(post) else None,
            "coverage_pre_mean": round(float(np.nanmean(cov_pre)), 4) if len(cov_pre) else None,
            "coverage_post_min": round(float(np.nanmin(cov_post)), 4) if len(cov_post) else None,
            "retrain_auc_post_min": round(float(np.nanmin(
                self.retrain_auc[self.index >= (self.change_point or 0)])), 4)
                if np.isfinite(self.retrain_auc).any() else None,
        }


def run_drift_experiment(
    base_model,
    calibrator,
    conformal,
    stream_df,
    change_point: int,
    threshold: float,
    window: int = 500,
    step: int = 100,
    retrain_window: int = 1500,
    alpha: float = 0.1,
    get_scores=None,
):
    """Slide a window across the stream; track frozen-model AUC, conformal
    coverage, a DDM flag on the error stream, and a retrained-model AUC.

    Parameters
    ----------
    base_model : fitted sklearn pipeline (frozen; trained on pre-stream data)
    calibrator : fitted Calibrator mapping scores -> probabilities
    conformal  : fitted SplitConformal / MondrianConformal
    get_scores : callable(model, texts) -> 1-D scores (defaults to predict_proba)
    """
    if get_scores is None:
        def get_scores(m, X):
            return m.predict_proba(X)[:, 1]

    texts = stream_df["text"].tolist()
    y = stream_df["label"].to_numpy()
    n = len(y)

    # Frozen-model scores / probabilities over the whole stream (one pass).
    raw = get_scores(base_model, texts)
    p = calibrator.predict_proba(raw)
    pred = (p >= threshold).astype(int)
    err = (pred != y).astype(int)

    # DDM on the error stream.
    ddm = DDM()
    first_drift = None
    flags = []
    for i in range(n):
        sig = ddm.update(int(err[i]))
        if sig == "drift":
            flags.append(i)
            if first_drift is None:
                first_drift = i

    # Sliding-window metrics.
    idx, auc_w, cov_w, retr_w = [], [], [], []
    centers = range(window, n + 1, step)
    for end in centers:
        sl = slice(end - window, end)
        ys, ps = y[sl], p[sl]
        center = end - window // 2
        idx.append(center)

        auc_w.append(roc_auc_score(ys, ps) if len(np.unique(ys)) == 2 else np.nan)
        cov_w.append(conformal.evaluate(ps, ys).empirical_coverage)

        # Retrain baseline: train on the immediately preceding window, score this one.
        rstart = max(0, end - window - retrain_window)
        if rstart < end - window and len(np.unique(y[rstart:end - window])) == 2:
            try:
                from sklearn.base import clone

                m2 = clone(base_model)
                m2.fit(texts[rstart:end - window], y[rstart:end - window])
                ps2 = get_scores(m2, texts[sl])
                retr_w.append(roc_auc_score(ys, ps2) if len(np.unique(ys)) == 2 else np.nan)
            except Exception:
                retr_w.append(np.nan)
        else:
            retr_w.append(np.nan)

    return StreamResult(
        index=np.array(idx),
        auc=np.array(auc_w),
        coverage=np.array(cov_w),
        retrain_auc=np.array(retr_w),
        drift_flags=flags,
        first_drift=first_drift,
        change_point=change_point,
    )
