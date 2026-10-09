"""Explore the data: team and player shot maps, and any match's xG race."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from pages_src import shared


def _xg_race(match):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    teams = [match["home_team"].iloc[0], match["away_team"].iloc[0]]
    for team, colour in zip(teams, (shared.OURS_COLOUR, shared.SB_COLOUR)):
        m = match[match["team"] == team].sort_values(["period", "minute", "second"])
        t = np.concatenate([[0], m["minute"] + m["second"] / 60, [max(90, match["minute"].max() + 1)]])
        cum = np.concatenate([[0], m["xg"].cumsum(), [m["xg"].sum()]])
        ax.step(t, cum, where="post", color=colour, lw=2, label=f"{team}  {m['xg'].sum():.2f} xG")
        goals = m[m["is_goal"] == 1]
        ax.scatter(goals["minute"] + goals["second"] / 60, np.interp(goals["minute"] + goals["second"] / 60, t, cum),
                   color=colour, s=90, edgecolors="black", zorder=3)
        for _, g in goals.iterrows():
            ax.annotate(g["player"].split()[-1], (g["minute"] + g["second"] / 60,
                        np.interp(g["minute"] + g["second"] / 60, t, cum)), xytext=(3, 6),
                        textcoords="offset points", fontsize=8)
    ax.set_xlabel("Minute")
    ax.set_ylabel("Cumulative xG")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def render() -> None:
    st.title("🗺️ Shot maps & match xG")
    shots = shared.get_shots()
    c = st.columns(3)
    competition = c[0].selectbox("Competition", sorted(shots["competition"].unique()),
                                 index=sorted(shots["competition"].unique()).index("FIFA World Cup"))
    comp = shots[shots["competition"] == competition]
    seasons = sorted(comp["season"].unique(), reverse=True)
    season = c[1].selectbox("Season", seasons)
    data = comp[comp["season"] == season]
    teams = data["team"].value_counts()
    team = c[2].selectbox("Team", teams.index)
    team_shots = data[data["team"] == team]

    left, right = st.columns([1, 1])
    with left:
        st.subheader(f"{team}: {season}")
        m = st.columns(3)
        m[0].metric("Shots", f"{len(team_shots):,}")
        m[1].metric("Goals", int(team_shots["is_goal"].sum()))
        m[2].metric("xG", f"{team_shots['xg'].sum():.1f}")
        fig = shared.shot_map(team_shots)
        st.pyplot(fig, width="stretch")
        shared.close(fig)
        st.caption("Marker size = xG; red = goal.")
    with right:
        st.subheader("Top shooters")
        top = (team_shots.groupby("player", observed=True)
               .agg(shots=("xg", "size"), goals=("is_goal", "sum"), xg=("xg", "sum"))
               .sort_values("xg", ascending=False).head(12))
        top["goals − xG"] = top["goals"] - top["xg"]
        st.dataframe(top.style.format({"xg": "{:.1f}", "goals − xG": "{:+.1f}"}), width="stretch")

    st.subheader("Match xG race")
    matches = (data[(data["home_team"] == team) | (data["away_team"] == team)]
               .drop_duplicates("match_id").sort_values("match_date"))
    labels = {r.match_id: f"{r.match_date} · {r.home_team} v {r.away_team}" for r in matches.itertuples()}
    match_id = st.selectbox("Match", list(labels), format_func=labels.get, index=len(labels) - 1)
    match = shots[shots["match_id"] == match_id]
    goals = match.groupby("team", observed=True)["is_goal"].sum()
    home, away = match["home_team"].iloc[0], match["away_team"].iloc[0]
    st.markdown(f"**{home} {goals.get(home, 0)} – {goals.get(away, 0)} {away}** "
                "(goals from shots; own goals aren't shots, so a final score can differ)")
    fig = _xg_race(match)
    st.pyplot(fig, width="stretch")
    shared.close(fig)
    shared.attribution()
