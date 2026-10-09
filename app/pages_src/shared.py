"""Cached loaders and pitch-drawing helpers shared by every page."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from mplsoccer import VerticalPitch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from soccer_xg import config, model as xg_model  # noqa: E402

APP_DATA = ROOT / "app" / "data" / "shots_app.parquet"
RESULTS = ROOT / "results"
MESSI = "Lionel Andrés Messi Cuccittini"

GOAL_COLOUR = "#d1495b"
MISS_COLOUR = "#8d99ae"
OURS_COLOUR = "#2b59c3"
SB_COLOUR = "#f4a261"
BASE_COLOUR = "#8d99ae"


@st.cache_resource
def get_model() -> xg_model.XGModel:
    return xg_model.load_model()


@st.cache_data
def get_shots() -> pd.DataFrame:
    return pd.read_parquet(APP_DATA)


@st.cache_data
def get_result(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


@st.cache_data
def get_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def half_pitch(figsize=(6, 5.2), x_min: float = 60.0):
    """A vertical half pitch (attacking upwards), StatsBomb coordinates,
    shown from pitch x = x_min (60 = the halfway line) to the goal.

    mplsoccer plots (x, y) pitch coordinates at axes position (y, x)."""
    pitch = VerticalPitch(pitch_type="statsbomb", half=True, pitch_color="#f7f7f2",
                          line_color="#9a9a9a", linewidth=1.2, pad_bottom=0.5 - (x_min - 60))
    fig, ax = pitch.draw(figsize=figsize)
    fig.set_facecolor("white")
    return pitch, fig, ax


def shot_map(shots: pd.DataFrame, xg_col: str = "xg", title: str | None = None, figsize=(6, 5.2)):
    """Shots sized by xG; goals in red."""
    pitch, fig, ax = half_pitch(figsize)
    for is_goal, colour, z in ((0, MISS_COLOUR, 2), (1, GOAL_COLOUR, 3)):
        s = shots[shots["is_goal"] == is_goal]
        pitch.scatter(s["x"], s["y"], s=40 + 900 * s[xg_col], ax=ax, color=colour,
                      edgecolors="white", linewidth=0.6, alpha=0.75 if is_goal else 0.45, zorder=z)
    if title:
        ax.set_title(title, fontsize=11)
    return fig


def attribution() -> None:
    st.caption(f"{config.ATTRIBUTION}. xG values and analysis: this project.")


def fmt_pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 12) -> pd.DataFrame:
    order = np.argsort(p)
    return pd.DataFrame([{"mean_xg": p[b].mean(), "goal_rate": y[b].mean(), "n": len(b)}
                         for b in np.array_split(order, bins)])


def close(fig) -> None:
    plt.close(fig)
