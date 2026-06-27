"""
From-scratch Multinomial Naive Bayes.

This is the implementation behind the resume line "implemented Naive Bayes from
scratch and validated against the scikit-learn reference." The validation is an
actual test (``tests/test_naive_bayes_parity.py``) asserting that our
``predict_log_proba`` matches ``sklearn.naive_bayes.MultinomialNB`` to ~1e-9.

Math (multinomial event model, Laplace/Lidstone smoothing alpha):

    log p(y=c)            = log( N_c / N )
    log p(t | y=c)        = log( (count_{c,t} + alpha) / (sum_t count_{c,t} + alpha * V) )
    log p(y=c | x)        ∝ log p(y=c) + sum_t x_t * log p(t | y=c)

All computation is in log space for numerical stability; the posterior is
normalised with a log-sum-exp.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin


class MultinomialNaiveBayes(BaseEstimator, ClassifierMixin):
    """Multinomial NB with Lidstone smoothing, log-space inference."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = sp.csr_matrix(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            # Smoothed per-class token counts.
            counts = np.asarray(Xc.sum(axis=0)).ravel() + self.alpha
            total = counts.sum()
            self.feature_log_prob_[i] = np.log(counts) - np.log(total)
            self.class_log_prior_[i] = np.log(Xc.shape[0]) - np.log(X.shape[0])
        return self

    def _joint_log_likelihood(self, X):
        X = sp.csr_matrix(X)
        # (n_samples, n_features) @ (n_features, n_classes) -> (n_samples, n_classes)
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        log_norm = logsumexp(jll, axis=1, keepdims=True)
        return jll - log_norm

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

    def decision_function(self, X):
        """Log-odds of the positive class (binary case), for ROC/threshold use."""
        lp = self.predict_log_proba(X)
        if lp.shape[1] == 2:
            return lp[:, 1] - lp[:, 0]
        return lp
