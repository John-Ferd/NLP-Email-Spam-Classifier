"""The from-scratch NB must match scikit-learn's MultinomialNB."""
import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from spamguard.data import synthetic_corpus
from spamguard.models.naive_bayes import MultinomialNaiveBayes


@pytest.mark.parametrize("alpha", [0.1, 1.0, 2.5])
def test_log_proba_matches_sklearn(alpha):
    df = synthetic_corpus(n=600, seed=1)
    X = CountVectorizer().fit_transform(df["text"])
    y = df["label"].to_numpy()
    ours = MultinomialNaiveBayes(alpha=alpha).fit(X, y)
    ref = MultinomialNB(alpha=alpha).fit(X, y)
    np.testing.assert_allclose(ours.predict_log_proba(X), ref.predict_log_proba(X), atol=1e-9)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))
