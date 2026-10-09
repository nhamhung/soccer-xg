"""Turn a scene built in the app (shooter, defenders, keeper, shot details)
into a one-row shots table the model scores exactly like a real shot."""

import json

import numpy as np
import pandas as pd

from . import config
from .features import FEATURES, shot_features

KEEPER_OFF_LINE = 1.5  # yards: a keeper's usual starting depth, on the ball-goal line

# Pass that set the shot up, as offered in the app -> the shot-table columns it implies.
ASSISTS = {
    "None (own ball / rebound)": {"assisted": False},
    "Ground pass": {"assisted": True, "pass_height": "Ground Pass"},
    "Through ball": {"assisted": True, "pass_height": "Ground Pass", "pass_through_ball": True},
    "Cut-back": {"assisted": True, "pass_height": "Ground Pass", "pass_cut_back": True},
    "Cross": {"assisted": True, "pass_height": "High Pass", "pass_cross": True},
}


def default_keeper(x: float, y: float) -> tuple[float, float]:
    """Where a keeper stands: on the line from the goal centre to the ball,
    KEEPER_OFF_LINE yards out (never past the ball)."""
    dx, dy = x - config.GOAL_X, y - config.GOAL_CENTRE_Y
    dist = max(np.hypot(dx, dy), 1e-9)
    step = min(KEEPER_OFF_LINE, dist / 2)
    return config.GOAL_X + dx / dist * step, config.GOAL_CENTRE_Y + dy / dist * step


def scene_to_shot(shooter: tuple[float, float], defenders=(), keeper=None, teammates=(),
                  **details) -> pd.DataFrame:
    """A one-row shots table for the scene. `keeper=None` puts the keeper in
    the default position; pass keeper=False for an empty net (keeper out of frame)."""
    x, y = shooter
    frame = [[dx, dy, 0, 0] for dx, dy in defenders] + [[tx, ty, 1, 0] for tx, ty in teammates]
    if keeper is not False:
        kx, ky = keeper if keeper is not None else default_keeper(x, y)
        frame.append([kx, ky, 0, 1])
    row = {"x": x, "y": y, "freeze_frame": json.dumps(frame), "shot_type": "Open Play",
           "body_part": "Right Foot", "technique": "Normal", "play_pattern": "Regular Play"}
    assist = details.pop("assist", "None (own ball / rebound)")
    row.update(ASSISTS[assist])
    row.update(details)
    return pd.DataFrame([row])


def xg_grid(model, step: float = 1.0, x_min: float = 84.0, **details) -> pd.DataFrame:
    """xG over the attacking third for an unpressured shot with the keeper
    in the default position - the heatmap behind the simulator."""
    xs = np.arange(x_min + step / 2, config.GOAL_X, step)
    ys = np.arange(step / 2, config.PITCH_WIDTH, step)
    gx, gy = np.meshgrid(xs, ys)
    shots = pd.concat([scene_to_shot((a, b), **details) for a, b in zip(gx.ravel(), gy.ravel())],
                      ignore_index=True)
    return pd.DataFrame({"x": gx.ravel(), "y": gy.ravel(), "xg": model.predict(shots)})


def contributions(model, shot: pd.DataFrame) -> pd.Series:
    """Each feature's push on this shot's xG (LightGBM's exact per-shot
    contributions, in log-odds; the last element is the average shot)."""
    values = model.booster.predict(shot_features(shot), num_iteration=model.n_rounds, pred_contrib=True)[0]
    return pd.Series(values, index=FEATURES + ["baseline"])
