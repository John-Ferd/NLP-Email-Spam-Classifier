"""
Shared plumbing for the scripts: config, data, fitting, calibration.

Hygiene: the calibration split is halved. ``cal_a`` fits the probability
calibrator and ``cal_b`` fits the conformal quantile, so the conformal scores
are computed with a calibrator that never saw them.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from .calibration import fit_best_calibrator
from .data import Splits, load_enron, make_splits, synthetic_corpus
from .models import MarginScaler, build_model, raw_scores

ROOT = Path(__file__).resolve().parents[2]
# Real results go to artifacts/. The smoke run sets SPAMGUARD_ARTIFACTS so it
# never overwrites them with synthetic-data numbers.
ARTIFACTS = Path(os.environ.get("SPAMGUARD_ARTIFACTS", ROOT / "artifacts"))


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_data(cfg: dict, smoke: bool = False):
    if smoke:
        return synthetic_corpus(n=4000, seed=cfg["data"]["seed"])
    path = ROOT / cfg["data"]["path"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run `make data` first (or use --smoke)")
    return load_enron(path)


def get_splits(cfg: dict, smoke: bool = False) -> Splits:
    d = cfg["data"]
    return make_splits(load_data(cfg, smoke), d["test_size"], d["cal_size"], d["seed"])


def halve_cal(splits: Splits, seed: int):
    """Stratified halves of the calibration split: (cal_a, cal_b)."""
    a, b = train_test_split(splits.cal, test_size=0.5, stratify=splits.cal["label"], random_state=seed)
    return a.reset_index(drop=True), b.reset_index(drop=True)


@dataclass
class FittedModel:
    kind: str
    model: object
    scaler: MarginScaler | None

    def scores(self, texts) -> np.ndarray:
        return raw_scores(self.model, list(texts), self.scaler)


def fit_base(cfg: dict, train, kind: str | None = None) -> FittedModel:
    kind = kind or cfg["model"]["kind"]
    model = build_model(kind, alpha=cfg["model"]["alpha"], C=cfg["model"]["C"])
    model.fit(train["text"].tolist(), train["label"].to_numpy())
    scaler = None
    if kind == "svm":
        scaler = MarginScaler().fit(model.decision_function(train["text"].tolist()))
    return FittedModel(kind, model, scaler)


def fit_calibrator(cfg: dict, fitted: FittedModel, cal_a):
    s = fitted.scores(cal_a["text"])
    return fit_best_calibrator(s, cal_a["label"].to_numpy(), tuple(cfg["calibration"]["methods"]),
                               cfg["calibration"]["n_bins"])


def save_json(obj, name: str):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / name, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
    print(json.dumps(obj, indent=2, default=float))
