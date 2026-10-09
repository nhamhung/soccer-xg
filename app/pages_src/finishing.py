"""Is finishing a skill? Goals minus xG, luck, and Messi."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from pages_src import shared
from soccer_xg import analysis


def render() -> None:
    st.title("🎯 Finishing: skill or luck?")
    st.markdown(
        "Goals minus xG measures how much a player beat the chances they had. But even an average "
        "finisher swings above and below xG by luck. The band below is how far luck alone goes: "
        "95% of simulated average finishers given the *same* chances land inside it."
    )
    st.caption("All values use out-of-fold xG: every shot is scored by a model that never saw its match, "
               "so no player can beat a model fitted to their own shots. Non-penalty shots only.")

    finishing = shared.get_result("player_finishing.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    outside = finishing["beyond_luck"]
    order = finishing.sort_values("xg")
    ax.fill_between(order["xg"], order["luck_low"], order["luck_high"], color="#ddd", step="mid",
                    label="Luck alone (95%)")
    ax.scatter(finishing.loc[~outside, "xg"], finishing.loc[~outside, "goals_minus_xg"], s=25,
               color=shared.MISS_COLOUR, label="Within luck")
    ax.scatter(finishing.loc[outside, "xg"], finishing.loc[outside, "goals_minus_xg"], s=40,
               color=shared.GOAL_COLOUR, label="Beyond luck")
    for _, r in finishing[outside | (finishing["xg"] > 120)].iterrows():
        ax.annotate(r["player"].split()[0] if r["player"] != shared.MESSI else "Messi",
                    (r["xg"], r["goals_minus_xg"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Total xG (log scale)")
    ax.set_ylabel("Goals − xG")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, width="stretch")
    shared.close(fig)
    st.caption(f"{len(finishing)} players with at least 150 non-penalty shots in the data.")

    p = shared.get_json("finishing_persistence.json")["all"]
    st.subheader("Does beating xG repeat?")
    st.markdown(
        f"If finishing were a strong skill, a player who beats xG in half their matches would beat it in "
        f"the other half too. Splitting each of **{p['players']} players'** matches into two halves "
        f"(≥ {p['min_shots_per_half']} shots each), the correlation between halves is "
        f"**r = {p['split_half_r']:.2f}** (95% CI {p['ci_low']:.2f} to {p['ci_high']:.2f}). "
        "For a typical player, a hot or cold finishing spell says almost nothing about the next one."
    )

    st.subheader("…with one exception")
    team = shared.get_result("messi_team_adjusted.csv")
    messi = team.iloc[0]
    st.markdown(
        f"Messi at Barcelona: **{int(messi.goals)} non-penalty goals from {messi.xg:.0f} xG** "
        f"(+{messi.goals - messi.xg:.0f}; luck allows about ±{(messi.luck_high - messi.luck_low) / 2:.0f}). "
        "But his team-mates *also* beat xG, by "
        f"{(team.iloc[1].goals_per_xg - 1):.0%}, more than luck allows, so part of it is a Barcelona effect "
        "the model can't see (the quality of the chances they created). Measured against his own "
        f"team-mates' rate, Messi is still **+{messi.goals_above_team_adjusted:.0f} goals**, far outside luck."
    )
    st.dataframe(
        team[["group", "shots", "goals", "xg", "goals_per_xg"]].rename(columns={
            "group": "", "shots": "Shots", "goals": "Goals", "xg": "xG", "goals_per_xg": "Goals per xG"})
        .style.format({"xG": "{:.0f}", "Goals per xG": "{:.2f}"}),
        width="stretch", hide_index=True)

    seasons = shared.get_result("messi_by_season.csv")
    seasons = seasons[seasons["shots"] >= 30]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    labels = [f"{s}\n{c.replace('La Liga', '').replace('FIFA ', '').strip()}" for s, c in
              zip(seasons["season"], seasons["competition"])]
    colours = [shared.GOAL_COLOUR if g > h else shared.MISS_COLOUR
               for g, h in zip(seasons["goals_minus_xg"], seasons["luck_high"])]
    ax.bar(range(len(seasons)), seasons["goals_minus_xg"], color=colours)
    ax.errorbar(range(len(seasons)), np.zeros(len(seasons)),
                yerr=[-seasons["luck_low"], seasons["luck_high"]], fmt="none", ecolor="#999", capsize=3, lw=1)
    ax.set_xticks(range(len(seasons)), labels, fontsize=7)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_ylabel("Goals − xG")
    ax.set_title("Messi, season by season (grey whiskers: luck range; red: beyond it)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, width="stretch")
    shared.close(fig)

    st.subheader("Check any player")
    shots = shared.get_shots()
    open_play = shots[shots["shot_type"] != "Penalty"]
    counts = open_play["player"].value_counts()
    counts = counts[counts >= 20]
    player = st.selectbox("Player", counts.index, index=0)
    ps = open_play[open_play["player"] == player]
    xg = ps["xg_oof"].to_numpy(float)
    low, high = analysis.goals_minus_xg_interval(xg, sims=4000)
    goals = int(ps["is_goal"].sum())
    c = st.columns(4)
    c[0].metric("Shots", f"{len(ps):,}")
    c[1].metric("Goals", goals)
    c[2].metric("xG", f"{xg.sum():.1f}")
    c[3].metric("Goals − xG", f"{goals - xg.sum():+.1f}", f"luck range {low:+.1f} to {high:+.1f}", delta_color="off")
    verdict = ("beyond what luck explains" if goals - xg.sum() > high or goals - xg.sum() < low
               else "within what luck alone could produce")
    st.write(f"**{player}**'s finishing is {verdict}.")
    fig = shared.shot_map(ps, xg_col="xg_oof", figsize=(6, 5))
    st.pyplot(fig, width="content")
    shared.close(fig)
    shared.attribution()
