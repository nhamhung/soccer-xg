# Setup and Deployment

## Local setup

```bash
git clone https://github.com/nhamhhung/soccer-xg.git
cd soccer-xg
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r app/requirements.txt
pytest tests/
python scripts/smoke_app.py
streamlit run app/streamlit_app.py
```

The repository ships everything the app needs (the trained model, `app/data/shots_app.parquet`
and `results/`), so the app runs straight after cloning. To rebuild them from the raw data:

```bash
python scripts/download_data.py    # ~1.2 GB transferred, ~5 min; keeps ~120 MB of shots
python scripts/train.py            # ~5 min
python scripts/analyze.py          # ~10 min
python scripts/build_app_data.py
```

## Deploy your fork

1. Fork or clone this repository and replace the values in `docs/deployment-config.yml`.
2. Push to your own public GitHub repository with `main` as the default branch.
3. In Settings → Pages, select **GitHub Actions** as the source.
4. In Streamlit Community Cloud, create a new app from your repository, branch `main`, file `app/streamlit_app.py`. No secrets are needed: StatsBomb Open Data is public and the app ships its own data.
5. Wait for CI and Pages to pass, then record the URLs and revision in `docs/DEPLOYMENT_ACCEPTANCE.md`.

Do not copy another owner's application URL or Pages URL. GitHub Pages hosts static documentation; Streamlit Community Cloud runs the Python app.

## Data licence

StatsBomb Open Data is free for non-commercial use with attribution; every app page and the report credit it. Read StatsBomb's terms in the [open-data repository](https://github.com/statsbomb/open-data) before any other use.

## Required checks

Protect `main` after the initial bootstrap. Require `ruff`, `pytest (3.11)`, `pytest (3.12)`, `smoke (3.11)`, `smoke (3.12)`, and `pages-build`. Disable force pushes and direct pushes; use up-to-date pull requests and squash merges.

## Rollback

Use the most recent SHA recorded as known-good in `DEPLOYMENT_ACCEPTANCE.md`. Revert later commits on a `rollback/<date>` branch, pass all required checks, and merge normally. Verify both Pages and Streamlit before closing the rollback.
