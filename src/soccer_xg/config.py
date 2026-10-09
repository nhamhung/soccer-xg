"""Paths, data source and pitch geometry shared by every part of the project.

Data: StatsBomb Open Data (https://github.com/statsbomb/open-data), free for
non-commercial use with attribution. Coordinates are StatsBomb's: a 120 x 80
pitch, the attacking team always shooting towards x = 120, with the goal
mouth between y = 36 and y = 44 (posts 8 yards apart) and the crossbar
2.67 yards high.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_SHOTS_DIR = DATA_DIR / "raw" / "shots"        # one small JSON per match (shots + assisting passes)
PROCESSED_DIR = DATA_DIR / "processed"
SHOTS_PARQUET = PROCESSED_DIR / "shots.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "xg_model.joblib"

OPEN_DATA_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
ATTRIBUTION = "Data: StatsBomb Open Data (github.com/statsbomb/open-data)"

PITCH_LENGTH, PITCH_WIDTH = 120.0, 80.0
GOAL_X = 120.0
GOAL_CENTRE_Y = 40.0
POST_LEFT_Y, POST_RIGHT_Y = 36.0, 44.0

RANDOM_SEED = 42
