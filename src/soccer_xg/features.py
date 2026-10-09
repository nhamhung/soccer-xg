"""Shot features: geometry, the freeze frame (who is in the way) and the assist.

`shot_features()` is the single place features are built. The training
script, the evaluation, and the app's shot simulator all call it, the
simulator by describing the user's scene as a one-row shots table with a
freeze frame, so a shot in the app is scored exactly like a real one.

Freeze frame encoding (see data.extract_shots): a JSON list of
[x, y, is_teammate, is_goalkeeper] for every visible player except the shooter.
"""

import json

import numpy as np
import pandas as pd

from . import config

# Opponents further than this from the shooter don't affect the shot; capping
# also gives a scene with no outfield opponents at all (possible in the app,
# rare in real freeze frames) a sensible value instead of "missing".
FAR_AWAY = 40.0
# Beyond this, how far the keeper is from the shooter says nothing that the
# shot's own distance doesn't. Uncapped, the model learned "keeper 40 yards
# away = keeper off his line = easy goal", which a keeper standing on his line
# for a long-range or wide shot then wrongly triggered.
KEEPER_NEAR = 12.0

NUMERIC_FEATURES = [
    "distance", "angle", "x", "abs_y_offset",
    "n_opponents_in_cone", "n_teammates_in_cone", "gk_in_cone",
    "gk_distance_to_goal", "gk_distance_to_shooter", "gk_advance", "gk_offset_from_shot_line",
    "nearest_opponent_distance", "n_opponents_within_3", "n_opponents_within_5",
    "pass_length",
]
BOOLEAN_FEATURES = [
    "first_time", "one_on_one", "open_goal", "aerial_won", "follows_dribble", "under_pressure",
    "assisted", "pass_cross", "pass_cut_back", "pass_through_ball", "pass_switch",
]
CATEGORICAL_FEATURES = ["body_part", "shot_type", "technique", "play_pattern", "pass_height"]
FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES

# Category levels fixed up front so a single simulated shot encodes exactly
# like the training data (pandas would otherwise infer levels per frame).
CATEGORY_LEVELS = {
    "body_part": ["Right Foot", "Left Foot", "Head", "Other"],
    "shot_type": ["Open Play", "Free Kick", "Corner", "Kick Off", "Penalty"],
    "technique": ["Normal", "Half Volley", "Volley", "Lob", "Backheel", "Diving Header", "Overhead Kick"],
    "play_pattern": ["Regular Play", "From Throw In", "From Free Kick", "From Corner", "From Counter",
                     "From Goal Kick", "From Keeper", "From Kick Off", "Other"],
    "pass_height": ["Ground Pass", "Low Pass", "High Pass"],
}


def goal_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Angle (radians) the goal mouth subtends from (x, y): the classic xG feature.

    It is the angle between the lines to the two posts, so it is large close
    in and central, and shrinks with distance and towards the byline.
    """
    a = np.arctan2(config.POST_LEFT_Y - y, config.GOAL_X - x)
    b = np.arctan2(config.POST_RIGHT_Y - y, config.GOAL_X - x)
    return np.abs(b - a)


def _in_triangle(points: np.ndarray, a, b, c) -> np.ndarray:
    """Which of `points` (n x 2) lie inside triangle abc (sign-of-cross-product test)."""
    def cross(p, q, r):
        return (q[0] - p[0]) * (r[:, 1] - p[1]) - (q[1] - p[1]) * (r[:, 0] - p[0])
    d1, d2, d3 = cross(a, b, points), cross(b, c, points), cross(c, a, points)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


def freeze_frame_features(x: float, y: float, frame: list) -> dict:
    """Defensive-pressure features for one shot from its freeze frame.

    The "cone" is the triangle from the ball to the two posts: anyone in it
    can block the shot or is in the keeper's way.
    """
    nan = float("nan")
    if not frame:
        return {k: nan for k in ("n_opponents_in_cone", "n_teammates_in_cone", "gk_in_cone",
                                 "gk_distance_to_goal", "gk_distance_to_shooter", "gk_advance", "gk_offset_from_shot_line",
                                 "nearest_opponent_distance", "n_opponents_within_3", "n_opponents_within_5")}
    players = np.asarray(frame, dtype=float).reshape(-1, 4)
    pos, teammate, keeper = players[:, :2], players[:, 2] == 1, players[:, 3] == 1
    opponent_gk = keeper & ~teammate
    outfield_opp = ~teammate & ~opponent_gk
    shooter = np.array([x, y])
    in_cone = _in_triangle(pos, shooter, (config.GOAL_X, config.POST_LEFT_Y), (config.GOAL_X, config.POST_RIGHT_Y))
    opp_dist = np.hypot(*(pos[outfield_opp] - shooter).T) if outfield_opp.any() else np.array([])

    out = {
        "n_opponents_in_cone": int((in_cone & outfield_opp).sum()),
        "n_teammates_in_cone": int((in_cone & teammate).sum()),
        "nearest_opponent_distance": float(min(opp_dist.min(), FAR_AWAY)) if len(opp_dist) else FAR_AWAY,
        "n_opponents_within_3": int((opp_dist <= 3).sum()),
        "n_opponents_within_5": int((opp_dist <= 5).sum()),
    }
    if opponent_gk.any():
        gk = pos[opponent_gk][0]
        goal = np.array([config.GOAL_X, config.GOAL_CENTRE_Y])
        line = goal - shooter
        # Perpendicular distance from the keeper to the shooter -> goal-centre line.
        offset = abs(line[0] * (gk[1] - y) - line[1] * (gk[0] - x)) / max(np.hypot(*line), 1e-9)
        out.update({
            "gk_in_cone": int(in_cone[opponent_gk][0]),
            "gk_distance_to_goal": float(np.hypot(*(gk - goal))),
            "gk_distance_to_shooter": float(min(np.hypot(*(gk - shooter)), KEEPER_NEAR)),
            # How far the keeper has come out towards the shooter (0 = on the line).
            "gk_advance": float(np.hypot(*(goal - shooter)) - np.hypot(*(gk - shooter))),
            "gk_offset_from_shot_line": float(offset),
        })
    else:  # keeper not in frame (e.g. caught upfield): the empty-net case
        out.update({"gk_in_cone": 0, "gk_distance_to_goal": nan, "gk_distance_to_shooter": nan,
                    "gk_advance": nan, "gk_offset_from_shot_line": nan})
    return out


def shot_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Model-ready features for a shots table (see FEATURES)."""
    x, y = shots["x"].to_numpy(float), shots["y"].to_numpy(float)
    out = pd.DataFrame(index=shots.index)
    out["distance"] = np.hypot(config.GOAL_X - x, config.GOAL_CENTRE_Y - y)
    out["angle"] = goal_angle(x, y)
    out["x"] = x
    out["abs_y_offset"] = np.abs(y - config.GOAL_CENTRE_Y)

    frames = shots["freeze_frame"] if "freeze_frame" in shots else pd.Series("[]", index=shots.index)
    ff = pd.DataFrame(
        [freeze_frame_features(xi, yi, json.loads(f) if isinstance(f, str) else (f or []))
         for xi, yi, f in zip(x, y, frames)],
        index=shots.index,
    )
    out = pd.concat([out, ff], axis=1)
    out["pass_length"] = pd.to_numeric(shots.get("pass_length"), errors="coerce") if "pass_length" in shots else np.nan

    for col in BOOLEAN_FEATURES:
        out[col] = shots[col].fillna(False).astype(int) if col in shots else 0
    for col, levels in CATEGORY_LEVELS.items():
        values = shots[col] if col in shots else pd.Series(None, index=shots.index, dtype=object)
        out[col] = pd.Categorical(values.where(values.isin(levels)), categories=levels)  # unknown -> missing
    return out[FEATURES]
