"""The project's analyses: transfer to women's football, and whether finishing is a skill.

All player-level numbers use out-of-fold xG: every shot is scored by a model
that never saw that shot's match, so a player can't "beat" a model that was
fitted to their own shots.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import config, model


def out_of_fold_xg(shots: pd.DataFrame, n_folds: int = 5, log=print) -> np.ndarray:
    """xG for every shot from a model trained on the other folds' matches."""
    xg = np.zeros(len(shots))
    for fold, (train, valid) in enumerate(GroupKFold(n_folds).split(shots, groups=shots["match_id"])):
        fitted = model.fit_xg_model(shots.iloc[train], log=lambda _: None)
        xg[valid] = fitted.predict(shots.iloc[valid])
        log(f"  out-of-fold xG: fold {fold + 1}/{n_folds}")
    return xg


def compare_with_statsbomb(shots: pd.DataFrame, n_boot: int = 2000) -> dict:
    """Held-out log loss of this model vs StatsBomb's xG, giving StatsBomb a
    fair chance: its values are re-calibrated (Platt scaling, per gender) on
    the training matches first, so a pure calibration edge doesn't count.
    95% CI of the difference by bootstrapping test matches."""
    from sklearn.linear_model import LogisticRegression

    open_play = shots[~model.is_penalty(shots)]
    train, test = open_play[open_play["xg_holdout"].isna()], open_play[open_play["xg_holdout"].notna()]

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def design(d):
        return np.column_stack([logit(d["statsbomb_xg"].to_numpy(float)), d["gender"].eq("female")])

    platt = LogisticRegression().fit(design(train), train["is_goal"])
    sb_recal = platt.predict_proba(design(test))[:, 1]
    y = test["is_goal"].to_numpy()

    def per_shot_loss(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    diff = per_shot_loss(test["xg_holdout"].to_numpy(float)) - per_shot_loss(sb_recal)
    per_match = pd.DataFrame({"m": test["match_id"].to_numpy(), "d": diff}).groupby("m")["d"].agg(["sum", "size"])
    rng = np.random.default_rng(config.RANDOM_SEED)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(per_match), len(per_match))
        boot.append(per_match["sum"].to_numpy()[i].sum() / per_match["size"].to_numpy()[i].sum())
    return {"log_loss_ours": float(per_shot_loss(test["xg_holdout"].to_numpy(float)).mean()),
            "log_loss_statsbomb_recalibrated": float(per_shot_loss(sb_recal).mean()),
            "difference": float(diff.mean()),
            "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
            "test_matches": int(len(per_match)), "test_shots": int(len(test))}


def gender_transfer(shots: pd.DataFrame, test_mask: np.ndarray, log=print) -> pd.DataFrame:
    """Score women's (and men's) test shots with models trained on men only,
    women only, and both, on the training matches."""
    train = shots[~test_mask]
    models = {
        "Trained on men's shots": model.fit_xg_model(train[train["gender"] == "male"], log=log),
        "Trained on women's shots": model.fit_xg_model(train[train["gender"] == "female"], log=log),
        "Trained on both": model.fit_xg_model(train, log=log),
    }
    rows = []
    test = shots[test_mask & ~model.is_penalty(shots)]
    for gender, group in test.groupby("gender"):
        for name, fitted in models.items():
            rows.append({"test_on": gender, "model": name, **model.score(group["is_goal"], fitted.predict(group))})
    return pd.DataFrame(rows)


def goals_minus_xg_interval(xg: np.ndarray, sims: int = 20000, seed: int = config.RANDOM_SEED) -> tuple[float, float]:
    """95% range of (goals - xG) an *average* finisher would produce from these
    exact chances: each shot scores with probability = its xG (Poisson-binomial)."""
    rng = np.random.default_rng(seed)
    goals = (rng.random((sims, len(xg))) < xg).sum(axis=1)
    return tuple(np.percentile(goals - xg.sum(), [2.5, 97.5]))


def player_finishing(shots: pd.DataFrame, xg_col: str = "xg_oof", min_shots: int = 150) -> pd.DataFrame:
    """Non-penalty goals vs xG per player, with the range luck alone allows."""
    open_play = shots[~model.is_penalty(shots)]
    rows = []
    for (player, gender), group in open_play.groupby(["player", "gender"]):
        if len(group) < min_shots:
            continue
        xg = group[xg_col].to_numpy()
        low, high = goals_minus_xg_interval(xg, sims=5000)
        goals = int(group["is_goal"].sum())
        rows.append({"player": player, "gender": gender, "shots": len(group), "goals": goals,
                     "xg": xg.sum(), "goals_minus_xg": goals - xg.sum(),
                     "luck_low": low, "luck_high": high,
                     "beyond_luck": goals - xg.sum() > high or goals - xg.sum() < low})
    return pd.DataFrame(rows).sort_values("goals_minus_xg", ascending=False).reset_index(drop=True)


def finishing_persistence(shots: pd.DataFrame, xg_col: str = "xg_oof", min_shots: int = 40) -> dict:
    """Is over-performing xG repeatable? Split every player's shots into two
    halves by match (odd/even) and correlate per-shot (goals - xG) between
    halves. A real skill shows up as a positive correlation."""
    open_play = shots[~model.is_penalty(shots)].copy()
    open_play["half"] = open_play.groupby("player")["match_id"].rank(method="dense").astype(int) % 2
    per = (open_play.assign(residual=open_play["is_goal"] - open_play[xg_col])
           .groupby(["player", "half"])["residual"].agg(["mean", "size"]).unstack())
    per = per[(per["size"] >= min_shots).all(axis=1)]
    a, b = per["mean"][0].to_numpy(), per["mean"][1].to_numpy()
    r = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(config.RANDOM_SEED)
    boot = [np.corrcoef(a[i], b[i])[0, 1] for i in (rng.integers(0, len(a), len(a)) for _ in range(2000))]
    return {"players": int(len(per)), "min_shots_per_half": min_shots, "split_half_r": r,
            "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
            # Spearman-Brown: reliability of the full sample from the half-sample correlation
            "full_sample_reliability": 2 * r / (1 + r) if r > -1 else float("nan")}


def season_finishing(shots: pd.DataFrame, player: str, xg_col: str = "xg_oof") -> pd.DataFrame:
    """One player's non-penalty goals vs xG per competition-season, in date order."""
    group = shots[(shots["player"] == player) & ~model.is_penalty(shots)]
    rows = []
    for (season, competition, team), s in group.groupby(["season", "competition", "team"]):
        low, high = goals_minus_xg_interval(s[xg_col].to_numpy(), sims=5000)
        rows.append({"season": season, "competition": competition, "team": team,
                     "first_match": s["match_date"].min(), "shots": len(s), "goals": int(s["is_goal"].sum()),
                     "xg": s[xg_col].sum(), "statsbomb_xg": s["statsbomb_xg"].sum(),
                     "luck_low": low, "luck_high": high})
    out = pd.DataFrame(rows).sort_values("first_match").reset_index(drop=True)
    out["goals_minus_xg"] = out["goals"] - out["xg"]
    return out


def team_adjusted_finishing(shots: pd.DataFrame, player: str, team: str, xg_col: str = "xg_oof") -> pd.DataFrame:
    """Separate a player's finishing from a team effect the model can't see.

    If the model under-rates every chance a team creates, all its players
    appear to beat xG. So compare the player with team-mates: scale the
    player's xG by the team-mates' goals/xG ratio, and see what is left.
    """
    open_play = shots[~model.is_penalty(shots)]
    groups = {
        f"{player} ({team})": open_play[(open_play["player"] == player) & (open_play["team"] == team)],
        f"{team} team-mates": open_play[(open_play["team"] == team) & (open_play["player"] != player)],
        f"{player} (other teams)": open_play[(open_play["player"] == player) & (open_play["team"] != team)],
        "All other teams": open_play[open_play["team"] != team],
    }
    rows = []
    for name, g in groups.items():
        xg = g[xg_col].to_numpy()
        low, high = goals_minus_xg_interval(xg, sims=2000)
        rows.append({"group": name, "shots": len(g), "goals": int(g["is_goal"].sum()), "xg": xg.sum(),
                     "goals_per_xg": g["is_goal"].sum() / xg.sum(), "luck_low": low, "luck_high": high})
    out = pd.DataFrame(rows)
    team_rate = out.loc[1, "goals_per_xg"]
    out["team_adjusted_xg"] = np.where(out.index == 0, out["xg"] * team_rate, np.nan)
    out["goals_above_team_adjusted"] = out["goals"] - out["team_adjusted_xg"]
    return out
