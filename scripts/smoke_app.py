"""Deployment smoke check for the soccer-xG app.

Usage:
    python scripts/smoke_app.py

1. The packaged model loads and scores a shot.
2. Every bundled data/results file the app reads exists.
3. Every page renders headlessly without an exception (Streamlit AppTest).
4. `streamlit run` starts and answers its health check.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from soccer_xg import config, model, scene  # noqa: E402

PAGES = ["simulator", "finishing", "shot_maps", "model_page", "about"]
RESULT_FILES = ["test_metrics.csv", "gender_transfer.csv", "player_finishing.csv",
                "finishing_persistence.json", "messi_by_season.csv", "messi_team_adjusted.csv",
                "statsbomb_comparison.json"]


def check_model() -> None:
    xg = model.load_model().predict(scene.scene_to_shot((108.0, 40.0)))
    if len(xg) != 1 or not 0 < float(xg[0]) < 1:
        raise RuntimeError("The packaged model did not return one probability.")


def check_bundled_assets() -> None:
    missing = [p for p in [PROJECT_ROOT / "app" / "data" / "shots_app.parquet", config.MODEL_PATH]
               + [PROJECT_ROOT / "results" / f for f in RESULT_FILES] if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing bundled files: {missing}")


def _render_page(app_dir: str, page: str) -> None:
    import importlib
    import sys

    sys.path.insert(0, app_dir)
    importlib.import_module(f"pages_src.{page}").render()


def check_pages() -> None:
    for page in PAGES:
        at = AppTest.from_function(_render_page, args=(str(PROJECT_ROOT / "app"), page), default_timeout=180)
        at.run()
        if at.exception:
            raise RuntimeError(f"Page {page} raised: {at.exception[0].value}")
        print(f"  page {page}: ok")


def check_streamlit() -> None:
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.headless=true",
         "--server.port=8502"], cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Streamlit exited before becoming healthy.")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8502/_stcore/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(1)
        raise RuntimeError("Streamlit health endpoint did not become ready within 90 seconds.")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    check_model()
    check_bundled_assets()
    check_pages()
    check_streamlit()
    print("soccer-xg deployment smoke check passed")
