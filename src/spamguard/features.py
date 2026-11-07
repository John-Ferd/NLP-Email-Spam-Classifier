"""
Feature extraction for SpamGuard.

Two representations, kept separate on purpose:

* ``build_count_vectorizer`` -- raw word counts for the multinomial Naive Bayes.
  NB has a generative story over token counts, so its native input is counts,
  not TF-IDF. Keeping this path clean is what makes the from-scratch NB
  checkable for exact parity against scikit-learn.

* ``build_tfidf_union`` -- a richer union for the linear models:
    1. word 1-2 gram TF-IDF (topical content),
    2. character ``char_wb`` 3-5 gram TF-IDF (sub-word fragments that survive
       segmentation tricks like "v i a g r a"; NOT robust to wholesale
       character substitution like "v1@gr@"),
    3. handcrafted numeric signals (length, digit/punctuation density, URLs,
       '$' and '!' counts), standardised.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

_URL = re.compile(r"https?://|www\.", re.IGNORECASE)


def build_count_vectorizer(min_df: int = 2, max_features: int | None = 50_000) -> CountVectorizer:
    """Word-count features for multinomial Naive Bayes."""
    return CountVectorizer(lowercase=True, min_df=min_df, max_features=max_features)


class HandcraftedFeatures(BaseEstimator, TransformerMixin):
    """Cheap, interpretable numeric signals that stay stable under vocabulary shift."""

    names = ["log_len", "digit_frac", "punct_frac", "upper_frac", "n_urls", "n_dollar", "n_excl"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for t in X:
            n = max(len(t), 1)
            rows.append([
                np.log1p(len(t)),
                sum(c.isdigit() for c in t) / n,
                sum((not c.isalnum()) and (not c.isspace()) for c in t) / n,
                sum(c.isupper() for c in t) / n,
                len(_URL.findall(t)),
                t.count("$"),
                t.count("!"),
            ])
        return np.asarray(rows, dtype=float)


def build_tfidf_union(max_word_features: int = 50_000, max_char_features: int = 50_000) -> FeatureUnion:
    """Word TF-IDF + char n-gram TF-IDF + scaled handcrafted features."""
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_word_features,
                                 sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 max_features=max_char_features, sublinear_tf=True)),
        ("hand", Pipeline([("f", HandcraftedFeatures()), ("scale", StandardScaler())])),
    ])
