# Soccer xG: Is Finishing a Skill?

An expected-goals (xG) model built from scratch on **100,864 shots** from StatsBomb's free event data
(3,961 men's and women's matches), benchmarked against StatsBomb's own xG, and used to separate
finishing skill from luck. It comes with an interactive Streamlit app where you build a shot by
clicking on the pitch.

**[Live app](https://soccer-xg.streamlit.app)** · **[Report](https://nhamhhung.github.io/soccer-xg/)**

## Findings

| | |
|---|---|
| **On par with a professional model** | On 811 held-out matches: log loss **0.2604** vs **0.2640** for StatsBomb's xG, ROC AUC **0.816** vs **0.809**. Still ahead after re-calibrating StatsBomb's values (−0.0033, 95% CI −0.0058 to −0.0010). The gain is mostly on women's matches; on men's it's close to a tie. |
| **The keeper matters most** | Freeze-frame features (where every player stood) carry 56% of the model's signal; the keeper's position alone 45%. Distance and angle together: 25%. |
| **Men's xG transfers to women's football** | Trained only on men's shots, the model predicts 707 goals on women's test matches where 702 were scored. |
| **Finishing is mostly luck** | Beating xG in half of a player's matches barely predicts the other half: r = 0.02 across 146 players. |
| **Except Messi** | +68 non-penalty goals above xG at Barcelona, even after adjusting for his team-mates *also* beating xG by 8%. Luck allows about ±29. |

## The app

| Page | What you can do |
|---|---|
| ⚽ **Shot Simulator** | Click to place the shooter, defenders, team-mates and keeper; pick body part, technique and assist; see the xG, a heatmap for that kind of shot, and exactly which factors push it up or down. Or load any real goal from the data and move the players. |
| 🎯 **Finishing: Skill or Luck?** | Every player's goals − xG against the range luck allows; Messi season by season; check any player. |
| 🗺️ **Shot Maps & Match xG** | Team shot maps for any competition-season, top shooters, and any match's xG race. |
| 📏 **Model vs StatsBomb** | Held-out metrics, calibration, feature importance, the men's/women's transfer test. |

## How it works

- **Data:** `scripts/download_data.py` fetches every match's events from
  [StatsBomb Open Data](https://github.com/statsbomb/open-data) (gzip-compressed, ~1.2 GB transferred)
  and keeps only the shots, their freeze frames and the passes that set them up.
- **Features** (`src/soccer_xg/features.py`): distance and angle; defenders and team-mates in the
  shooting cone; the keeper's distance, advance and offset from the shot line; the nearest defender;
  body part, technique, phase of play and assist type. The app's simulator builds its scene as a
  one-row shots table and goes through the same function.
- **Model** (`src/soccer_xg/model.py`): LightGBM with monotonic constraints (xG must fall with
  distance, width and defenders in the way). Rounds chosen by 5-fold CV grouped by match. Penalties get
  a constant.
- **Evaluation:** 20% of matches held out within every competition-season. StatsBomb's xG scored on the
  same shots; a fair-comparison check re-calibrates it first.
- **Finishing analyses** (`src/soccer_xg/analysis.py`): out-of-fold xG (each shot scored by a model that
  never saw its match), Poisson-binomial simulation for the luck range, split-half reliability, and a
  team-mate adjustment.

Building the simulator exposed two flaws the test metrics couldn't see (wide shots scoring higher than
central ones, and "keeper far away" being read as "easy goal"). Both fixes are explained in the report.

## Project layout

```
src/soccer_xg/      config, data download, features, model, scene (simulator), analysis
scripts/            download_data, train, analyze, build_app_data, smoke_app
app/                Streamlit app (pages_src/) + data/shots_app.parquet (6 MB, committed)
models/             xg_model.joblib (1 MB, committed)
results/            analysis outputs read by the app and report (committed)
notebooks/          01_eda_and_modeling.ipynb
report/             report.qmd → report.html (committed)
tests/              pytest suite
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r app/requirements.txt
```

The model, app data and results are committed, so the app runs straight away:

```bash
streamlit run app/streamlit_app.py
```

To rebuild everything from the raw data (~25 minutes):

```bash
python scripts/download_data.py    # ~5 min, ~1.2 GB transferred, ~120 MB kept
python scripts/train.py            # model + held-out metrics
python scripts/analyze.py          # out-of-fold xG, transfer, finishing analyses
python scripts/build_app_data.py   # compact data file for the app
```

## Report

```bash
python -m ipykernel install --user --name soccer-xg
cd report && QUARTO_PYTHON=../.venv/bin/python quarto render report.qmd
```

## Tests

```bash
pytest tests/                 # features, model behaviour, simulator click mapping
python scripts/smoke_app.py   # every app page renders; streamlit starts healthy
```

## Deploy

Streamlit Community Cloud: new app → this repository, branch `main`, file `app/streamlit_app.py`. No
secrets needed. See [docs/SETUP_AND_DEPLOYMENT.md](docs/SETUP_AND_DEPLOYMENT.md).

## Data and licence

Data: [StatsBomb Open Data](https://github.com/statsbomb/open-data), free for non-commercial use with
attribution. The open data favours some teams and tournaments (notably Barcelona 2004–2021), so findings
describe these matches, not football in general.
