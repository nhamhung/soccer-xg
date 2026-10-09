"""Shot simulator: click to build a scene, see its xG and why."""

import io
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from pages_src import shared
from soccer_xg import scene

DPI = 110
VIEW_X_MIN = 72.0  # the simulator shows the final 48 yards
FIG_SIZE = (6, 4.4)
DEFAULT_SCENE = {"shooter": (106.0, 34.0), "defenders": [(111.0, 37.0)], "teammates": [], "keeper": None}
FEATURE_LABELS = {
    "distance": "Distance to goal", "angle": "Angle to the goal mouth", "x": "How far up the pitch",
    "abs_y_offset": "How wide", "n_opponents_in_cone": "Defenders in the way",
    "n_teammates_in_cone": "Team-mates in the way", "gk_in_cone": "Keeper in the way",
    "gk_distance_to_goal": "Keeper off his line", "gk_distance_to_shooter": "Keeper's distance to the shooter",
    "gk_offset_from_shot_line": "Keeper off the shooting line",
    "nearest_opponent_distance": "Space to the nearest defender",
    "n_opponents_within_3": "Defenders within 3 yards", "n_opponents_within_5": "Defenders within 5 yards",
    "pass_length": "Length of the assist", "first_time": "First-time shot", "one_on_one": "One-on-one",
    "open_goal": "Open goal", "aerial_won": "Won an aerial duel", "follows_dribble": "After a dribble",
    "under_pressure": "Under pressure", "assisted": "Assisted", "pass_cross": "From a cross",
    "pass_cut_back": "From a cut-back", "pass_through_ball": "From a through ball",
    "pass_switch": "From a switch of play", "body_part": "Body part", "shot_type": "Shot type",
    "technique": "Technique", "play_pattern": "Phase of play", "pass_height": "Height of the assist",
}


def _state() -> dict:
    if "scene" not in st.session_state:
        st.session_state.scene = json.loads(json.dumps(DEFAULT_SCENE))
    return st.session_state.scene


@st.cache_data(show_spinner="Drawing the xG map ...")
def _grid(details_key: str) -> pd.DataFrame:
    return scene.xg_grid(shared.get_model(), step=1.0, x_min=VIEW_X_MIN, **json.loads(details_key))


def _draw(s: dict, details: dict, show_map: bool):
    pitch, fig, ax = shared.half_pitch(figsize=FIG_SIZE, x_min=VIEW_X_MIN)
    if show_map:
        g = _grid(json.dumps(details, sort_keys=True))
        xs, ys = np.sort(g["x"].unique()), np.sort(g["y"].unique())
        grid = g.pivot(index="x", columns="y", values="xg").loc[xs, ys].to_numpy()
        ylim, xlim = ax.get_ylim(), ax.get_xlim()
        ax.imshow(grid, origin="lower", extent=[0, 80, VIEW_X_MIN, 120], cmap="YlOrRd", alpha=0.55,
                  vmin=0, vmax=0.6, interpolation="bilinear", zorder=1, aspect="auto")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    keeper = s["keeper"] or scene.default_keeper(*s["shooter"])
    if s["defenders"]:
        d = np.array(s["defenders"])
        pitch.scatter(d[:, 0], d[:, 1], ax=ax, s=170, color="#264653", edgecolors="white", zorder=4)
    if s["teammates"]:
        t = np.array(s["teammates"])
        pitch.scatter(t[:, 0], t[:, 1], ax=ax, s=170, color="#7fb7be", edgecolors="white", zorder=4)
    if s["keeper"] is not False:
        pitch.scatter([keeper[0]], [keeper[1]], ax=ax, s=190, color="#2a9d8f", marker="s",
                      edgecolors="white", zorder=4)
    x, y = s["shooter"]
    pitch.polygon([[(x, y), (120, 36), (120, 44)]], ax=ax, color=shared.OURS_COLOUR, alpha=0.12, zorder=2)
    pitch.scatter([x], [y], ax=ax, s=260, color=shared.GOAL_COLOUR, edgecolors="white", marker="o", zorder=5)
    pitch.scatter([x], [y], ax=ax, s=40, color="white", zorder=6)

    png, to_data, height_px = to_png(fig, ax)
    plt.close(fig)
    return png, to_data, height_px


def to_png(fig, ax):
    """Render at DPI and return (png bytes, display->data transform, image height in px).

    savefig(dpi=...) restores the figure's own DPI afterwards, so the DPI is set
    on the figure itself and the transform read after the final draw - otherwise
    clicks map to the wrong spot."""
    fig.set_dpi(DPI)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    fig.canvas.draw()
    return buf.getvalue(), ax.transData.inverted(), fig.get_size_inches()[1] * DPI


def pixel_to_pitch(click: dict, to_data, height_px: float, image_width: float) -> tuple[float, float]:
    """A click on the (possibly resized) image -> StatsBomb pitch (x, y).

    The click is in displayed pixels from the top-left; matplotlib's display
    coordinates are in rendered pixels from the bottom-left, and the vertical
    pitch's axes are (pitch y, pitch x)."""
    scale = image_width / click["width"]
    ax_x, ax_y = to_data.transform((click["x"] * scale, height_px - click["y"] * scale))
    return float(np.clip(ax_y, VIEW_X_MIN, 119.5)), float(np.clip(ax_x, 0.5, 79.5))


def _apply_click(click: dict, mode: str, to_data, height_px: float, image_width: float) -> None:
    s = _state()
    x, y = pixel_to_pitch(click, to_data, height_px, image_width)
    if mode == "Shooter":
        s["shooter"] = (x, y)
    elif mode == "Defender":
        s["defenders"].append((x, y))
    elif mode == "Team-mate":
        s["teammates"].append((x, y))
    elif mode == "Goalkeeper":
        s["keeper"] = (x, y)
    else:  # remove the nearest defender / team-mate
        players = [("defenders", i, p) for i, p in enumerate(s["defenders"])] + \
                  [("teammates", i, p) for i, p in enumerate(s["teammates"])]
        if players:
            group, i, _ = min(players, key=lambda t: np.hypot(t[2][0] - x, t[2][1] - y))
            s[group].pop(i)


def _load_real_shot(shot: pd.Series) -> dict:
    frame = json.loads(shot["freeze_frame"])
    keeper = next(((p[0], p[1]) for p in frame if p[3] == 1 and p[2] == 0), False)
    st.session_state.scene = {
        "shooter": (float(shot["x"]), float(shot["y"])),
        "defenders": [(p[0], p[1]) for p in frame if p[2] == 0 and p[3] == 0],
        "teammates": [(p[0], p[1]) for p in frame if p[2] == 1],
        "keeper": keeper,
    }
    assist = ("Cross" if shot["pass_cross"] else "Cut-back" if shot["pass_cut_back"]
              else "Through ball" if shot["pass_through_ball"] else
              "Ground pass" if shot["assisted"] else "None (own ball / rebound)")
    st.session_state.update({
        "sim_body": shot["body_part"] if shot["body_part"] in ("Right Foot", "Left Foot", "Head") else "Right Foot",
        "sim_technique": shot["technique"], "sim_assist": assist,
        "sim_first_time": bool(shot["first_time"]), "sim_pressure": bool(shot["under_pressure"]),
        "sim_one_on_one": bool(shot["one_on_one"]), "sim_dribble": bool(shot["follows_dribble"]),
    })


def _example_picker() -> None:
    shots = shared.get_shots()
    pool = shots[shots["freeze_frame"].notna()]
    with st.expander("⚽ Load a real shot from the data", expanded=False):
        players = pool["player"].value_counts()
        default = list(players.index).index(shared.MESSI) if shared.MESSI in players.index else 0
        player = st.selectbox("Player", players.index, index=default)
        options = pool[pool["player"] == player].sort_values("match_date", ascending=False)
        labels = {i: f"{r.match_date} · {r.home_team} v {r.away_team} · {r.minute}' · "
                     f"{'GOAL' if r.is_goal else r.outcome} · xG {r.xg:.2f}"
                  for i, r in options.iterrows()}
        pick = st.selectbox("Shot", list(labels), format_func=labels.get)
        if st.button("Load this shot", width="stretch"):
            _load_real_shot(options.loc[pick])
            st.session_state.loaded_shot = options.loc[pick].to_dict()
            st.rerun()


def render() -> None:
    st.title("⚽ Shot simulator")
    st.markdown("Click the pitch to place the **shooter**, **defenders**, **team-mates** or the "
                "**keeper**, describe the shot, and watch its expected goals (xG) change: the probability "
                "an average player scores this chance, learned from 100,864 real shots.")
    s = _state()

    left, right = st.columns([1.15, 1])
    with right:
        mode = st.radio("Click places a …", ["Shooter", "Defender", "Team-mate", "Goalkeeper", "Remove nearest"],
                        horizontal=True)
        b1, b2, b3 = st.columns(3)
        if b1.button("Reset scene", width="stretch"):
            st.session_state.scene = json.loads(json.dumps(DEFAULT_SCENE))
            st.session_state.pop("loaded_shot", None)
            st.rerun()
        if b2.button("Clear players", width="stretch"):
            s["defenders"], s["teammates"] = [], []
        if b3.button("Auto keeper" if s["keeper"] is not None else "Empty net", width="stretch"):
            s["keeper"] = None if s["keeper"] is not None else False

        c1, c2 = st.columns(2)
        body = c1.selectbox("Body part", ["Right Foot", "Left Foot", "Head"], key="sim_body")
        technique = c2.selectbox("Technique", ["Normal", "Half Volley", "Volley", "Lob", "Backheel",
                                               "Diving Header", "Overhead Kick"], key="sim_technique")
        assist = st.selectbox("Set up by", list(scene.ASSISTS), key="sim_assist")
        t1, t2, t3, t4 = st.columns(4)
        details = {
            "body_part": body, "technique": technique, "assist": assist,
            "first_time": t1.toggle("First time", key="sim_first_time"),
            "under_pressure": t2.toggle("Pressured", key="sim_pressure"),
            "one_on_one": t3.toggle("1 v 1", key="sim_one_on_one"),
            "follows_dribble": t4.toggle("After dribble", key="sim_dribble"),
        }
        show_map = st.toggle("Show the xG map for this kind of shot", value=True,
                             help="xG from every spot for the same shot with no defenders "
                                  "and the keeper in position.")

    with left:
        png, to_data, height_px = _draw(s, details, show_map)
        click = streamlit_image_coordinates(Image.open(io.BytesIO(png)), key="pitch", width="stretch", cursor="crosshair")
        if click and click.get("unix_time") != st.session_state.get("last_click"):
            st.session_state.last_click = click.get("unix_time")
            _apply_click(click, mode, to_data, height_px, image_width=6 * DPI)
            st.rerun()
        st.caption("🔴 shooter · ⚫ defenders · 🔵 team-mates · 🟩 keeper · shaded: the shooting cone")

    model = shared.get_model()
    shot = scene.scene_to_shot(tuple(s["shooter"]), defenders=s["defenders"], teammates=s["teammates"],
                               keeper=None if s["keeper"] is None else (s["keeper"] or False), **details)
    xg = float(model.predict(shot)[0])
    all_xg = shared.get_shots()["xg"].to_numpy()

    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric("xG", f"{xg:.2f}")
        m2.metric("Scored", f"≈ 1 in {max(1, round(1 / max(xg, 1e-3)))}")
        m3.metric("Better than", f"{(all_xg < xg).mean():.0%} of shots")
        loaded = st.session_state.get("loaded_shot")
        if loaded:
            st.info(f"Loaded: **{loaded['player']}**, {loaded['home_team']} v {loaded['away_team']} "
                    f"({loaded['match_date']}), {loaded['minute']}'. Outcome: **{loaded['outcome']}**. "
                    f"Our xG for the real shot {loaded['xg']:.2f} · StatsBomb's {loaded['statsbomb_xg']:.2f}. "
                    "Now move players and see what changes.")

    st.subheader("Why this xG?")
    contrib = scene.contributions(model, shot).drop("baseline")
    contrib = contrib[contrib.abs() > 0.02].sort_values(key=np.abs, ascending=False).head(8)[::-1]
    if contrib.empty:
        st.write("This is an average chance: nothing pushes it far from the typical shot.")
    else:
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(contrib) + 0.6))
        ax.barh([FEATURE_LABELS.get(f, f) for f in contrib.index], contrib.values,
                color=[shared.OURS_COLOUR if v > 0 else shared.GOAL_COLOUR for v in contrib.values])
        ax.axvline(0, color="#444", lw=0.8)
        ax.set_xlabel("← makes a goal less likely      (log-odds)      more likely →")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, width="stretch")
        shared.close(fig)
    st.caption("Exact per-shot contributions from the LightGBM model (TreeSHAP). They add up, in "
               "log-odds, from the average shot to this one.")
    _example_picker()
    shared.attribution()
