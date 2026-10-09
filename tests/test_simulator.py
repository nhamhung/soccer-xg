"""The simulator's click -> pitch mapping and scene construction."""

import io
import json

import matplotlib.pyplot as plt
import pytest
from PIL import Image

from pages_src import shared, simulator
from soccer_xg import scene


@pytest.mark.parametrize("display_width", [660, 420, 900])
def test_click_maps_back_to_the_pitch_point(display_width):
    pitch, fig, ax = shared.half_pitch(figsize=simulator.FIG_SIZE, x_min=simulator.VIEW_X_MIN)
    png, to_data, height_px = simulator.to_png(fig, ax)
    image_width = Image.open(io.BytesIO(png)).width
    for x, y in [(110.0, 30.0), (95.0, 65.0), (118.0, 40.0)]:
        px, py = ax.transData.transform((y, x))            # axes are (pitch y, pitch x)
        shown = display_width / image_width
        click = {"x": px * shown, "y": (height_px - py) * shown, "width": display_width}
        assert simulator.pixel_to_pitch(click, to_data, height_px, image_width) == pytest.approx((x, y), abs=0.05)
    plt.close(fig)


def test_rendered_image_has_the_size_the_mapping_assumes():
    pitch, fig, ax = shared.half_pitch(figsize=simulator.FIG_SIZE, x_min=simulator.VIEW_X_MIN)
    png, _, height_px = simulator.to_png(fig, ax)
    image = Image.open(io.BytesIO(png))
    assert image.size == (6 * simulator.DPI, round(height_px))
    plt.close(fig)


def test_default_keeper_stands_between_ball_and_goal_centre():
    kx, ky = scene.default_keeper(108.0, 30.0)
    assert 118 < kx < 120 and 30 < ky < 40


def test_scene_to_shot_encodes_players_and_assist():
    shot = scene.scene_to_shot((108.0, 40.0), defenders=[(112.0, 40.0)], teammates=[(115.0, 35.0)],
                               assist="Cut-back", body_part="Left Foot")
    frame = json.loads(shot.loc[0, "freeze_frame"])
    assert [p[2:] for p in frame] == [[0, 0], [1, 0], [0, 1]]   # defender, team-mate, keeper
    assert shot.loc[0, "pass_cut_back"] and shot.loc[0, "assisted"] and shot.loc[0, "body_part"] == "Left Foot"
    empty = scene.scene_to_shot((108.0, 40.0), keeper=False)
    assert json.loads(empty.loc[0, "freeze_frame"]) == []
