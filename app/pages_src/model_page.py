"""How good is the model? Held-out comparison with StatsBomb's xG."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import log_loss, roc_auc_score

from pages_src import shared
from pages_src.simulator import FEATURE_LABELS

MODEL_COLOURS = {"This project (LightGBM)": shared.OURS_COLOUR, "StatsBomb xG": shared.SB_COLOUR,
                 "Distance + angle (logistic)": shared.BASE_COLOUR}


def render() -> None:
    st.title("📏 Model vs StatsBomb")
    st.markdown(
        "The model was trained on 80% of matches and scored on the other 20% (811 matches, "
        "20,794 shots), split within every competition so each is represented. It is compared "
        "with the textbook distance-and-angle model and with **StatsBomb's own xG**, which ships with "
        "the data and is a professional benchmark."
    )
    metrics = shared.get_result("test_metrics.csv")
    np_metrics = metrics[metrics["subset"] == "Non-penalty"].set_index("model")
    ours, sb = np_metrics.loc["This project (LightGBM)"], np_metrics.loc["StatsBomb xG"]
    c = st.columns(4)
    c[0].metric("Log loss (lower is better)", f"{ours.log_loss:.4f}", f"{ours.log_loss - sb.log_loss:+.4f} vs StatsBomb",
                delta_color="inverse")
    c[1].metric("ROC AUC (ranking)", f"{ours.roc_auc:.3f}", f"{ours.roc_auc - sb.roc_auc:+.3f} vs StatsBomb")
    c[2].metric("Calibration error", f"{ours.ece:.4f}", f"{ours.ece - sb.ece:+.4f} vs StatsBomb", delta_color="inverse")
    c[3].metric("Goals vs xG (test)", f"{int(ours.goals):,} vs {ours.xg:,.0f}", f"StatsBomb {sb.xg:,.0f}",
                delta_color="off")

    table = np_metrics[["log_loss", "brier", "roc_auc", "ece", "goals", "xg"]].rename(columns={
        "log_loss": "Log loss", "brier": "Brier", "roc_auc": "ROC AUC", "ece": "Calibration error",
        "goals": "Goals", "xg": "Total xG"})
    st.dataframe(table.style.format({"Goals": "{:,.0f}", "Total xG": "{:,.0f}", "Log loss": "{:.4f}",
                                     "Brier": "{:.4f}", "ROC AUC": "{:.3f}", "Calibration error": "{:.4f}"}),
                 width="stretch")
    st.caption("Non-penalty shots in the 811 held-out matches. Penalties get a constant xG in every model.")

    shots = shared.get_shots()
    test = shots[shots["xg_holdout"].notna() & (shots["shot_type"] != "Penalty")]
    left, right = st.columns(2)
    with left:
        st.subheader("Calibration")
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        for name, col in [("This project (LightGBM)", "xg_holdout"), ("StatsBomb xG", "statsbomb_xg")]:
            b = shared.calibration_bins(test["is_goal"].to_numpy(), test[col].to_numpy(float))
            ax.plot(b["mean_xg"], b["goal_rate"], "o-", color=MODEL_COLOURS[name], label=name)
        ax.plot([0, 0.7], [0, 0.7], "--", color="#999", lw=1)
        ax.set_xlabel("Predicted xG (bin average)")
        ax.set_ylabel("Share actually scored")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, width="stretch")
        shared.close(fig)
        st.caption("On the dashed line, a 0.3 xG chance goes in 30% of the time. StatsBomb's values run "
                   "slightly low on this data: about 6% fewer goals than were scored.")
    with right:
        st.subheader("What the model looks at")
        booster = shared.get_model().booster
        gain = pd.Series(booster.feature_importance("gain"), booster.feature_name())
        gain = (gain / gain.sum()).sort_values().tail(12)
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.barh([FEATURE_LABELS.get(f, f) for f in gain.index], gain.values, color=shared.OURS_COLOUR)
        ax.set_xlabel("Share of the model's total gain")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, width="stretch")
        shared.close(fig)
        st.caption("The keeper's position (from the freeze frame) matters more than the shot angle, "
                   "the classic xG feature.")

    st.subheader("By gender: where the gain comes from")
    rows = []
    for gender, g in test.groupby("gender", observed=True):
        y = g["is_goal"].to_numpy()
        for name, col in [("This project", "xg_holdout"), ("StatsBomb", "statsbomb_xg")]:
            p = np.clip(g[col].to_numpy(float), 1e-6, 1 - 1e-6)
            rows.append({"Matches": gender.capitalize(), "Model": name, "Shots": len(g),
                         "Log loss": log_loss(y, p), "ROC AUC": roc_auc_score(y, p),
                         "Goals": int(y.sum()), "Total xG": p.sum()})
    st.dataframe(pd.DataFrame(rows).style.format({"Log loss": "{:.4f}", "ROC AUC": "{:.3f}", "Total xG": "{:,.0f}",
                                                  "Shots": "{:,}", "Goals": "{:,}"}),
                 width="stretch", hide_index=True)
    st.markdown(
        "On **men's** matches the two models are close to a tie. The gain is on **women's** "
        "matches. And it's not that women's football needs a separate model: one trained *only on men's "
        "shots* still predicts women's goals with good calibration (below). Training on both just gives "
        "more data."
    )
    transfer = shared.get_result("gender_transfer.csv")
    transfer["test_on"] = transfer["test_on"].map({"female": "Women's test matches", "male": "Men's test matches"})
    st.dataframe(transfer.pivot(index="model", columns="test_on", values="log_loss")
                 .style.format("{:.4f}"), width="stretch")
    st.caption("Log loss on held-out matches (lower is better), by training data.")
    cmp = shared.get_json("statsbomb_comparison.json")
    st.info(
        "**Fair-comparison caveats.** StatsBomb's model is general-purpose, built on far more data than is "
        "public, and was not tuned to these competitions; this one was. After giving StatsBomb's values the "
        "same chance (re-calibrating them on the training matches), this model is still better: log loss "
        f"{cmp['difference']:+.4f}, 95% CI [{cmp['ci_low']:+.4f}, {cmp['ci_high']:+.4f}] by bootstrapping "
        f"the {cmp['test_matches']} test matches. The difference is small and mostly from women's matches."
    )
    shared.attribution()
