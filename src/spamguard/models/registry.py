"""
Base-model factory: ``nb`` | ``logreg`` | ``svm``.

Every model is a scikit-learn Pipeline that takes raw text, so the drift
experiment can ``clone`` and refit it on a window of the stream.

The three models fail differently, which is the point:

* ``nb``     -- from-scratch multinomial NB on word counts; overconfident.
* ``logreg`` -- logistic regression on the TF-IDF union; logistic loss is a
                proper scoring rule, so it is roughly calibrated already.
* ``svm``    -- LinearSVC on the TF-IDF union; emits an unbounded margin,
                not a probability.

``raw_scores`` returns each model's native score mapped into [0, 1]:
NB and LogReg give P(spam); the SVM margin is squashed with a min-max map
fit on the training margins (clipped), which is deliberately *not* a
probability -- it is the "margin treated as a probability" mistake that
calibration then fixes.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ..features import build_count_vectorizer, build_tfidf_union
from .naive_bayes import MultinomialNaiveBayes


def build_model(kind: str = "nb", alpha: float = 1.0, C: float = 1.0) -> Pipeline:
    kind = kind.lower()
    if kind == "nb":
        return Pipeline([("vec", build_count_vectorizer()), ("clf", MultinomialNaiveBayes(alpha=alpha))])
    if kind == "logreg":
        return Pipeline([("vec", build_tfidf_union()),
                         ("clf", LogisticRegression(C=C, max_iter=2000))])
    if kind == "svm":
        return Pipeline([("vec", build_tfidf_union()), ("clf", LinearSVC(C=C))])
    raise ValueError(f"unknown model kind {kind!r}")


class MarginScaler:
    """Min-max map of SVM margins into [0, 1], fit on training margins."""

    def fit(self, margins):
        m = np.asarray(margins, dtype=float)
        self.lo_, self.hi_ = float(np.min(m)), float(np.max(m))
        return self

    def transform(self, margins):
        m = np.asarray(margins, dtype=float)
        return np.clip((m - self.lo_) / (self.hi_ - self.lo_), 0.0, 1.0)


def raw_scores(model: Pipeline, texts, margin_scaler: MarginScaler | None = None) -> np.ndarray:
    """Native [0, 1] score of a fitted pipeline (see module docstring)."""
    if isinstance(model.named_steps["clf"], LinearSVC):
        margins = model.decision_function(texts)
        if margin_scaler is None:
            raise ValueError("SVM scores need a fitted MarginScaler")
        return margin_scaler.transform(margins)
    return model.predict_proba(texts)[:, 1]
