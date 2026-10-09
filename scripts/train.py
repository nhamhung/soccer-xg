#!/usr/bin/env python
"""Train the xG model and evaluate it on held-out matches.

Usage:
    python scripts/train.py

1. Splits matches 80/20 within every competition-season (data/processed/split.parquet).
2. Fits the distance + angle baseline and the LightGBM model on the training matches.
3. Scores both, and StatsBomb's own xG, on the test matches (results/test_metrics.csv).
4. Refits LightGBM on all matches for the app (models/xg_model.joblib) and
   writes every shot's xG to data/processed/shots_xg.parquet for the analyses.
"""

import functools
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_xg import config, data, model  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001
RESULTS_DIR = config.PROJECT_ROOT / "results"


def main() -> None:
    shots = data.load_shots()
    train_idx, test_idx = model.train_test_matches(shots)
    train, test = shots.loc[train_idx], shots.loc[test_idx]
    print(f"Train: {len(train):,} shots / {train['match_id'].nunique():,} matches; "
          f"test: {len(test):,} shots / {test['match_id'].nunique():,} matches")

    print("Fitting the distance + angle baseline ...")
    baseline = model.DistanceAngleBaseline().fit(train)
    print("Fitting LightGBM ...")
    xg_model = model.fit_xg_model(train)

    predictions = {
        "Distance + angle (logistic)": baseline.predict(test),
        "This project (LightGBM)": xg_model.predict(test),
        "StatsBomb xG": test["statsbomb_xg"].to_numpy(),
    }
    rows = []
    for subset, mask in {"All shots": slice(None), "Non-penalty": ~model.is_penalty(test)}.items():
        for name, p in predictions.items():
            rows.append({"subset": subset, "model": name, **model.score(test["is_goal"].to_numpy()[mask], p[mask])})
    metrics = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics.to_csv(RESULTS_DIR / "test_metrics.csv", index=False)
    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        print(metrics.to_string(index=False))

    pd.DataFrame({"shot_id": shots["shot_id"], "is_test": shots.index.isin(test_idx)}).to_parquet(
        config.PROCESSED_DIR / "split.parquet", index=False)
    test_xg = pd.DataFrame({"shot_id": test["shot_id"], "xg_holdout": predictions["This project (LightGBM)"],
                            "xg_baseline": predictions["Distance + angle (logistic)"]})

    print("Refitting on all matches for the app ...")
    final = model.fit_xg_model(shots)
    final.save()
    out = shots.assign(xg=final.predict(shots)).merge(test_xg, on="shot_id", how="left")
    out.to_parquet(config.PROCESSED_DIR / "shots_xg.parquet", index=False)
    print(f"Saved {config.MODEL_PATH} ({config.MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
