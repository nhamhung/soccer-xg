"""About the project: data, method, findings, limitations."""

import streamlit as st

from pages_src import shared


def render() -> None:
    st.title("ℹ️ About this project")
    shots = shared.get_shots()
    metrics = shared.get_result("test_metrics.csv").set_index(["subset", "model"])
    ours = metrics.loc[("Non-penalty", "This project (LightGBM)")]
    sb = metrics.loc[("Non-penalty", "StatsBomb xG")]
    p = shared.get_json("finishing_persistence.json")["all"]
    messi = shared.get_result("messi_team_adjusted.csv").iloc[0]

    c = st.columns(4)
    c[0].metric("Shots", f"{len(shots):,}")
    c[1].metric("Matches", f"{shots['match_id'].nunique():,}")
    c[2].metric("Competition-seasons", f"{shots.groupby(['competition', 'season'], observed=True).ngroups}")
    c[3].metric("Women's shots", f"{(shots['gender'] == 'female').mean():.0%}")

    st.markdown(f"""
**Expected goals (xG)** is the probability that an average player scores a given chance. This project
builds an xG model from scratch on every shot in StatsBomb's free event data (men's and women's
football, 1958–2025) and uses it to ask whether finishing is a real skill.

### Findings
1. **The model matches or beats StatsBomb's own xG on held-out matches**: log loss {ours.log_loss:.4f} vs
   {sb.log_loss:.4f}, ROC AUC {ours.roc_auc:.3f} vs {sb.roc_auc:.3f}. The gain is mostly on women's matches.
2. **The keeper's position matters most.** Freeze frames record where every player stood at the shot;
   the keeper's distance from the shooter carries more of the model's signal than the shot angle.
3. **Men's xG transfers to women's football.** A model trained only on men's shots is well calibrated
   on women's matches.
4. **Finishing is mostly luck.** Beating xG in half of a player's matches barely predicts the other
   half (r = {p['split_half_r']:.2f} across {p['players']} players).
5. **Messi is the exception.** +{messi.goals_above_team_adjusted:.0f} non-penalty goals over xG at Barcelona
   even after adjusting for his team-mates also beating xG.

### Method
- **Data:** StatsBomb Open Data: every shot with its freeze frame (positions of all visible players)
  and the pass that set it up. Penalty shoot-outs excluded.
- **Features:** distance and angle, defenders and team-mates in the shooting cone, the keeper's
  position, the nearest defender, body part, technique, and the type of assist.
- **Model:** LightGBM, rounds chosen by 5-fold CV grouped by match. Penalties get a constant (the
  training conversion rate).
- **Evaluation:** 20% of matches held out within every competition-season; StatsBomb's xG scored on
  the same shots.
- **Finishing analyses:** out-of-fold xG (each shot scored by a model that never saw its match) and
  Poisson-binomial simulation for the range luck allows.

### Limitations
- The open data over-represents some teams (Barcelona's 2004–2021 seasons, international
  tournaments). The analyses are about *these* matches, not football in general.
- xG measures chance quality as the freeze frame sees it. It can't see the ball's spin, the shooter's
  balance, or which foot is stronger. That is part of what "finishing" absorbs.
- StatsBomb's public model was not tuned to these competitions; this one was. The comparison is
  informative, not a claim to beat a professional product.
""")
    shared.attribution()
