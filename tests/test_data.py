import pandas as pd

from spamguard.data import load_enron, make_drift_stream, make_splits, synthetic_corpus


def test_load_enron_parses_and_drops_empty(tmp_path):
    csv = tmp_path / "mini.csv"
    pd.DataFrame({
        "Message ID": [0, 1, 2],
        "Subject": ["hello", "WIN cash", None],
        "Message": ["see you", "click now", None],
        "Spam/Ham": ["ham", "spam", "spam"],
        "Date": ["2001-01-01", "2005-02-02", "2005-03-03"],
    }).to_csv(csv, index=False)
    df = load_enron(csv)
    assert list(df["label"]) == [0, 1]
    assert df.loc[1, "text"] == "WIN cash click now"


def test_splits_are_disjoint_and_stratified():
    df = synthetic_corpus(n=2000, seed=0)
    s = make_splits(df, seed=0)
    assert len(s.train) + len(s.cal) + len(s.test) == len(df)
    rates = [p["label"].mean() for p in (s.train, s.cal, s.test)]
    assert max(rates) - min(rates) < 0.02


def test_drift_stream_only_obfuscates_spam_after_change():
    df = synthetic_corpus(n=500, seed=0)
    stream, cp = make_drift_stream(df, n=400, change_point=0.5, seed=0)
    assert cp == 200
    pre, post = stream.iloc[:cp], stream.iloc[cp:]

    def leet(t):
        return any(c in t for c in "@3$")

    assert not pre["text"].map(leet).any()
    assert post.loc[post["label"] == 1, "text"].map(leet).all()
    assert not post.loc[post["label"] == 0, "text"].map(leet).any()
