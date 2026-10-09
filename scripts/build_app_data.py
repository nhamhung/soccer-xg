#!/usr/bin/env python
"""Build the compact data file the Streamlit app ships with (after analyze.py).

Usage:
    python scripts/build_app_data.py

Writes app/data/shots_app.parquet: every shot with its location, outcome and
the three xG values, but freeze frames only for goals and big chances (the
simulator's "load a real shot" examples), so the app needs neither the raw
data nor a download on Streamlit Community Cloud.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_xg import config  # noqa: E402

APP_DATA = config.PROJECT_ROOT / "app" / "data" / "shots_app.parquet"
COLUMNS = ["shot_id", "match_id", "match_date", "competition", "season", "gender", "home_team", "away_team",
           "team", "player", "period", "minute", "second", "x", "y", "end_y", "end_z", "outcome", "is_goal",
           "shot_type", "body_part", "technique", "play_pattern", "first_time", "one_on_one", "open_goal",
           "aerial_won", "follows_dribble", "under_pressure", "assisted", "pass_height", "pass_cross",
           "pass_cut_back", "pass_through_ball", "pass_switch", "pass_length",
           "xg", "xg_oof", "statsbomb_xg", "xg_holdout"]
CATEGORICAL = ["competition", "season", "gender", "home_team", "away_team", "team", "player", "outcome",
               "shot_type", "body_part", "technique", "play_pattern", "pass_height", "match_date"]


def main() -> None:
    shots = pd.read_parquet(config.PROCESSED_DIR / "shots_xg.parquet")
    out = shots[COLUMNS].copy()
    # Keep freeze frames for a pool of examples only: goals and shots with xG >= 0.3.
    example = (shots["is_goal"] == 1) | (shots["xg"] >= 0.3)
    out["freeze_frame"] = shots["freeze_frame"].where(example & shots["has_freeze_frame"])
    for col in CATEGORICAL:
        out[col] = out[col].astype("category")
    for col in out.select_dtypes("float64"):
        out[col] = out[col].astype("float32")
    APP_DATA.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(APP_DATA, index=False, compression="zstd")
    print(f"Wrote {len(out):,} shots to {APP_DATA} ({APP_DATA.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
