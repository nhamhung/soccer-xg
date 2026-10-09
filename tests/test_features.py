"""Feature tests on hand-built shots with known geometry."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from soccer_xg import features


def make_shot(x=108.0, y=40.0, frame=(), **extra) -> pd.DataFrame:
    row = {"x": x, "y": y, "freeze_frame": json.dumps([list(p) for p in frame]),
           "body_part": "Right Foot", "shot_type": "Open Play", "technique": "Normal",
           "play_pattern": "Regular Play", **extra}
    return pd.DataFrame([row])


class TestGoalAngle:
    def test_penalty_spot_angle(self):
        # 12 yards out, central: 2 * atan(4 / 12)
        assert features.goal_angle(np.array([108.0]), np.array([40.0]))[0] == pytest.approx(2 * math.atan(4 / 12))

    def test_angle_shrinks_with_distance_and_towards_the_byline(self):
        near, far = features.goal_angle(np.array([110.0, 90.0]), np.array([40.0, 40.0]))
        central, wide = features.goal_angle(np.array([110.0, 110.0]), np.array([40.0, 70.0]))
        assert near > far and central > wide


class TestFreezeFrame:
    def test_defender_between_ball_and_goal_is_in_the_cone(self):
        f = features.freeze_frame_features(108, 40, [[114, 40, 0, 0], [100, 40, 0, 0]])
        assert f["n_opponents_in_cone"] == 1          # the one behind the shooter isn't
        assert f["nearest_opponent_distance"] == pytest.approx(6.0)
        assert f["n_opponents_within_5"] == 0

    def test_goalkeeper_features(self):
        f = features.freeze_frame_features(108, 40, [[119, 41, 0, 1], [110, 40, 1, 0]])
        assert f["gk_in_cone"] == 1
        assert f["gk_distance_to_goal"] == pytest.approx(math.hypot(1, 1))
        assert f["gk_offset_from_shot_line"] == pytest.approx(1.0)
        assert f["n_teammates_in_cone"] == 1
        assert f["n_opponents_in_cone"] == 0          # the keeper is counted separately

    def test_keeper_out_of_frame_means_no_keeper_features(self):
        f = features.freeze_frame_features(108, 40, [[114, 40, 0, 0]])
        assert f["gk_in_cone"] == 0 and math.isnan(f["gk_distance_to_goal"])

    def test_missing_frame_is_all_missing(self):
        assert all(math.isnan(v) for v in features.freeze_frame_features(108, 40, []).values())


def test_shot_features_columns_and_category_levels():
    out = features.shot_features(make_shot(frame=[(119, 40, 0, 1)], technique="Something new"))
    assert list(out.columns) == features.FEATURES
    assert list(out["body_part"].cat.categories) == features.CATEGORY_LEVELS["body_part"]
    assert pd.isna(out["technique"].iloc[0])          # unseen level -> missing, not a new category
    assert out["first_time"].iloc[0] == 0             # absent booleans default to False


def test_no_outfield_opponents_means_far_away_not_missing():
    f = features.freeze_frame_features(108, 40, [[119, 40, 0, 1]])
    assert f["nearest_opponent_distance"] == features.FAR_AWAY
    assert features.freeze_frame_features(80, 40, [[30, 40, 0, 0]])["nearest_opponent_distance"] == features.FAR_AWAY


def test_keeper_advance_and_capped_distance():
    on_line = features.freeze_frame_features(90, 40, [[120, 40, 0, 1]])
    rushed = features.freeze_frame_features(108, 40, [[114, 40, 0, 1]])
    assert on_line["gk_advance"] == pytest.approx(0) and on_line["gk_distance_to_shooter"] == features.KEEPER_NEAR
    assert rushed["gk_advance"] == pytest.approx(6) and rushed["gk_distance_to_shooter"] == pytest.approx(6)
