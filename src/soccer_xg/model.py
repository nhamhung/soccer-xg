"""The xG model: LightGBM on shot features, plus a constant for penalties.

Penalties are taken from the same spot with no defenders, so a learned model
has nothing to learn from their features; they get the training set's
conversion rate instead, the convention in public xG models.
"""

from dataclasses import dataclass, field

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from . import config
from .features import FEATURES, shot_features

# Physics the trees must respect, whatever a few odd shots suggest: xG falls
# with distance, width and defenders in the way, and rises with the angle.
# Without these, ~35 freak goals from near the touchline (mishit crosses) gave
# wide positions a higher xG than central ones at the same distance.
MONOTONE = {"distance": -1, "angle": 1, "abs_y_offset": -1, "n_opponents_in_cone": -1}

LGBM_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "cat_smooth": 20,
    "verbose": -1,
    "seed": config.RANDOM_SEED,
    "num_threads": 0,
    "monotone_constraints": [MONOTONE.get(f, 0) for f in FEATURES],
    "monotone_constraints_method": "advanced",
}


def is_penalty(shots: pd.DataFrame) -> np.ndarray:
    return (shots["shot_type"] == "Penalty").to_numpy()


def train_test_matches(shots: pd.DataFrame, test_share: float = 0.2) -> tuple[pd.Index, pd.Index]:
    """Split by match (never by shot: shots in one match share players and
    conditions) and within every competition-season, so the test set covers
    all of them in proportion."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    test = []
    for _, group in shots.groupby(["competition", "season"]):
        matches = np.sort(group["match_id"].unique())
        test.extend(rng.choice(matches, size=max(1, round(len(matches) * test_share)), replace=False))
    is_test = shots["match_id"].isin(test)
    return shots.index[~is_test], shots.index[is_test]


@dataclass
class XGModel:
    booster: lgb.Booster
    penalty_xg: float
    n_rounds: int
    cv_log_loss: float = float("nan")
    meta: dict = field(default_factory=dict)

    def predict(self, shots: pd.DataFrame) -> np.ndarray:
        xg = self.booster.predict(shot_features(shots), num_iteration=self.n_rounds)
        return np.where(is_penalty(shots), self.penalty_xg, xg)

    def save(self, path=config.MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


def load_model(path=config.MODEL_PATH) -> XGModel:
    return joblib.load(path)


def fit_xg_model(shots: pd.DataFrame, params: dict | None = None, n_folds: int = 5,
                 log=print) -> XGModel:
    """Pick the number of boosting rounds by grouped CV, then refit on all of `shots`."""
    params = {**LGBM_PARAMS, **(params or {})}
    pens = is_penalty(shots)
    open_play = shots[~pens]
    X, y = shot_features(open_play), open_play["is_goal"].to_numpy()
    cv = lgb.cv(
        params, lgb.Dataset(X, y), num_boost_round=3000,
        folds=GroupKFold(n_folds).split(X, y, groups=open_play["match_id"]),
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    n_rounds = len(cv["valid binary_logloss-mean"])
    cv_loss = cv["valid binary_logloss-mean"][-1]
    log(f"  grouped {n_folds}-fold CV: {n_rounds} rounds, log loss {cv_loss:.4f}")
    booster = lgb.train(params, lgb.Dataset(X, y), num_boost_round=n_rounds)
    return XGModel(booster=booster, penalty_xg=float(shots.loc[pens, "is_goal"].mean()),
                   n_rounds=n_rounds, cv_log_loss=float(cv_loss),
                   meta={"n_shots": int(len(shots)), "features": FEATURES})


class DistanceAngleBaseline:
    """The textbook xG model: logistic regression on distance and angle only."""

    def fit(self, shots: pd.DataFrame) -> "DistanceAngleBaseline":
        open_play = shots[~is_penalty(shots)]
        self.penalty_xg = float(shots.loc[is_penalty(shots), "is_goal"].mean())
        self.model = LogisticRegression(max_iter=1000).fit(self._X(open_play), open_play["is_goal"])
        return self

    @staticmethod
    def _X(shots: pd.DataFrame) -> np.ndarray:
        f = shot_features(shots)
        return np.column_stack([f["distance"], np.log1p(f["distance"]), f["angle"]])

    def predict(self, shots: pd.DataFrame) -> np.ndarray:
        return np.where(is_penalty(shots), self.penalty_xg, self.model.predict_proba(self._X(shots))[:, 1])


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Shot-weighted mean |observed goal rate - mean xG| over equal-count bins."""
    order = np.argsort(p)
    return float(sum(len(b) * abs(y[b].mean() - p[b].mean()) for b in np.array_split(order, bins)) / len(y))


def score(y, p) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "roc_auc": roc_auc_score(y, p),
        "ece": expected_calibration_error(y, p),
        "goals": int(y.sum()),
        "xg": float(p.sum()),
        "n": int(len(y)),
    }
