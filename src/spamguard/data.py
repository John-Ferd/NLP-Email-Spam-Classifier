"""
Data loading, splitting, and controlled drift injection for SpamGuard.

Design notes (see README for full reasoning):

* We load the consolidated Enron-Spam corpus (subject + body + label + date).
* The MAIN train/cal/test split is *stratified random*, NOT temporal. The Enron
  corpus has a severe temporal confound: virtually all ham predates the spam
  (ham: 2000-2002, spam: 2004-2005). A temporal split would let any model
  separate classes by date alone, leaking the label. We therefore split at
  random with stratification and treat the date axis as unusable for evaluation.
* Because the natural date axis is confounded, the drift experiment uses
  *controlled injection* with known ground-truth change points. This is the only
  way to obtain meaningful detector metrics (detection delay, false-alarm rate)
  on this corpus.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_enron(csv_path: str | Path) -> pd.DataFrame:
    """Load the consolidated Enron-Spam CSV into a clean frame.

    Returns a DataFrame with columns: ``text`` (subject + body), ``label``
    (1 = spam, 0 = ham), and ``date`` (datetime64, may contain NaT).
    Rows whose combined text is empty after stripping are dropped.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    subject = df["Subject"].fillna("").astype(str)
    body = df["Message"].fillna("").astype(str)
    text = (subject + " " + body).str.strip()
    label = (df["Spam/Ham"].str.strip().str.lower() == "spam").astype(int)
    date = pd.to_datetime(df["Date"], errors="coerce")

    out = pd.DataFrame({"text": text, "label": label.values, "date": date.values})
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    return out


def temporal_confound_report(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise the ham/spam date distributions.

    Used in the README/EDA to *demonstrate* why a temporal split is invalid on
    this corpus. Returns a per-class summary of min/median/max date.
    """
    rows = []
    for name, lab in [("ham", 0), ("spam", 1)]:
        d = df.loc[df["label"] == lab, "date"].dropna()
        rows.append(
            dict(
                label=name,
                n=len(d),
                min=d.min(),
                median=d.median(),
                max=d.max(),
            )
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
@dataclass
class Splits:
    """Train / calibration / test partitions.

    The calibration split is *not optional*: both probability calibration
    (Platt/isotonic) and split-conformal prediction require a held-out
    calibration set that is disjoint from both train and test.
    """

    train: pd.DataFrame
    cal: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> pd.DataFrame:
        rows = []
        for name, part in [("train", self.train), ("cal", self.cal), ("test", self.test)]:
            rows.append(
                dict(
                    split=name,
                    n=len(part),
                    spam_rate=round(float(part["label"].mean()), 4),
                )
            )
        return pd.DataFrame(rows)


def make_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    cal_size: float = 0.2,
    seed: int = 0,
) -> Splits:
    """Stratified random train/cal/test split.

    ``cal_size`` is expressed as a fraction of the *non-test* pool, so the final
    proportions are roughly:  test = test_size,  cal = (1-test_size)*cal_size,
    train = the remainder.
    """
    from sklearn.model_selection import train_test_split

    train_pool, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    train, cal = train_test_split(
        train_pool,
        test_size=cal_size,
        stratify=train_pool["label"],
        random_state=seed,
    )
    return Splits(
        train=train.reset_index(drop=True),
        cal=cal.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


# --------------------------------------------------------------------------- #
# Controlled drift injection
# --------------------------------------------------------------------------- #
_LEET = str.maketrans({"a": "@", "e": "3", "i": "1", "o": "0", "s": "$"})


def obfuscate(text: str, level: float, rng: np.random.Generator) -> str:
    """Apply spammer-style evasion to a message.

    ``level`` in [0, 1] is the per-eligible-character probability of a
    leet-style substitution (a->@, e->3, ...). This is *concept drift*: the
    conditional P(x | y=spam) changes while the label semantics stay fixed.
    Crucially it degrades word-level features (the vocabulary shifts) far more
    than character n-gram features (which still see "v1@gr@" ~ "viagra").
    """
    if level <= 0:
        return text
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch in "aeios" and rng.random() < level:
            chars[i] = ch.translate(_LEET)
    return "".join(chars)


def make_drift_stream(
    df: pd.DataFrame,
    n: int = 6000,
    change_point: float = 0.5,
    max_obfuscation: float = 1.0,
    drift_kind: str = "concept",
    prior_after: float = 0.5,
    ramp: bool = False,
    seed: int = 0,
) -> tuple[pd.DataFrame, int]:
    """Build an ordered evaluation stream with a *known* change point.

    Returns ``(stream_df, change_index)`` where ``stream_df`` has columns
    ``text`` and ``label`` and ``change_index`` is the integer position at which
    drift begins. Two drift kinds are supported:

    * ``"concept"``: after the change point, spam messages are obfuscated
      (P(x|spam) shifts). With ``ramp=False`` (default) this is a clean *step*
      change to ``max_obfuscation`` -- the standard setup for measuring a
      detector's delay against a known change point. With ``ramp=True`` the
      obfuscation grows linearly from 0 to ``max_obfuscation`` (a gradual-drift
      scenario). The label rule is unchanged either way.
    * ``"prior"``: after the change point the class prior P(y=spam) jumps to
      ``prior_after`` (covariate/label-shift).

    The stream is i.i.d. *up to* the injected shift, so before the change point
    exchangeability holds by construction and conformal coverage / model AUC
    should be stable. That is what makes the post-change degradation
    interpretable as drift rather than as a confound.
    """
    rng = np.random.default_rng(seed)
    pool = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    cut = int(n * change_point)
    texts: list[str] = []
    labels: list[int] = []

    # Pre-change segment: i.i.d. sample at the corpus's natural balance.
    spam_pool = pool[pool["label"] == 1]
    ham_pool = pool[pool["label"] == 0]
    base_rate = float(pool["label"].mean())

    def draw(spam_rate: float) -> tuple[str, int]:
        if rng.random() < spam_rate:
            row = spam_pool.iloc[rng.integers(len(spam_pool))]
            return row["text"], 1
        row = ham_pool.iloc[rng.integers(len(ham_pool))]
        return row["text"], 0

    for i in range(n):
        post = i >= cut
        if drift_kind == "prior":
            rate = prior_after if post else base_rate
            t, y = draw(rate)
        else:  # concept
            t, y = draw(base_rate)
            if post and y == 1:
                if ramp:
                    # Gradual drift: obfuscation grows from 0 to max by stream end.
                    frac = (i - cut) / max(1, n - cut)
                    level = max_obfuscation * frac
                else:
                    # Step drift: full obfuscation immediately after the change.
                    level = max_obfuscation
                t = obfuscate(t, level=level, rng=rng)
        texts.append(t)
        labels.append(y)

    stream = pd.DataFrame({"text": texts, "label": labels})
    return stream, cut


# --------------------------------------------------------------------------- #
# Synthetic corpus (offline smoke tests only; never used for reported metrics)
# --------------------------------------------------------------------------- #
def synthetic_corpus(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Tiny separable synthetic corpus for CI / offline development.

    Not representative of Enron; used only so the pipeline runs without the
    real download. Reported metrics in the README come from Enron-Spam.
    """
    rng = np.random.default_rng(seed)
    ham_words = "meeting report schedule project budget quarter team review client".split()
    spam_words = "free winner prize cash viagra loan offer click urgent guaranteed".split()
    filler = "the a to of and in for on with please find attached regards".split()

    texts, labels, dates = [], [], []
    base = np.datetime64("2003-01-01")
    for i in range(n):
        is_spam = rng.random() < 0.5
        topic = spam_words if is_spam else ham_words
        k = rng.integers(8, 25)
        toks = list(rng.choice(topic, size=k // 2)) + list(rng.choice(filler, size=k - k // 2))
        rng.shuffle(toks)
        texts.append(" ".join(toks))
        labels.append(int(is_spam))
        dates.append(base + np.timedelta64(int(rng.integers(0, 600)), "D"))
    return pd.DataFrame({"text": texts, "label": labels, "date": dates})
