#!/usr/bin/env python
"""Run the project's analyses (after scripts/train.py) and save results/.

Usage:
    python scripts/analyze.py      # ~5-10 min (out-of-fold xG + three transfer models)
"""

import functools
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_xg import analysis, config  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001
RESULTS_DIR = config.PROJECT_ROOT / "results"
MESSI = "Lionel Andrés Messi Cuccittini"


def main() -> None:
    shots = pd.read_parquet(config.PROCESSED_DIR / "shots_xg.parquet")
    is_test = shots["xg_holdout"].notna().to_numpy()

    print("Out-of-fold xG for every shot ...")
    shots["xg_oof"] = analysis.out_of_fold_xg(shots)
    shots.to_parquet(config.PROCESSED_DIR / "shots_xg.parquet", index=False)

    comparison = analysis.compare_with_statsbomb(shots)
    (RESULTS_DIR / "statsbomb_comparison.json").write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))

    print("Men's -> women's transfer ...")
    transfer = analysis.gender_transfer(shots, is_test)
    transfer.to_csv(RESULTS_DIR / "gender_transfer.csv", index=False)
    print(transfer[["test_on", "model", "log_loss", "roc_auc", "ece", "goals", "xg"]].to_string(index=False))

    print("Finishing ...")
    finishing = analysis.player_finishing(shots)
    finishing.to_csv(RESULTS_DIR / "player_finishing.csv", index=False)
    persistence = {g: analysis.finishing_persistence(shots[shots["gender"] == g]) for g in ("male", "female")}
    persistence["all"] = analysis.finishing_persistence(shots)
    (RESULTS_DIR / "finishing_persistence.json").write_text(json.dumps(persistence, indent=2))
    print(json.dumps(persistence, indent=2))
    print(finishing.head(10).to_string(index=False))
    print(f"{finishing['beyond_luck'].sum()} of {len(finishing)} players outside the luck range")

    messi = analysis.season_finishing(shots, MESSI)
    messi.to_csv(RESULTS_DIR / "messi_by_season.csv", index=False)
    print(messi.round(1).to_string(index=False))
    team = analysis.team_adjusted_finishing(shots, MESSI, "Barcelona")
    team.to_csv(RESULTS_DIR / "messi_team_adjusted.csv", index=False)
    print(team.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
