"""Download StatsBomb Open Data and build the shots table.

StatsBomb publishes one JSON file of events per match (~3 MB, ~0.3 MB
gzip-compressed in transit). We only need the shots, so each match is
fetched compressed, its shots (with their freeze frames) and the passes that
set them up are kept in a small per-match file under data/raw/shots/, and the
rest is discarded. Re-running skips matches already fetched.
"""

import gzip
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import pandas as pd

from . import config

PENALTY_SHOOTOUT_PERIOD = 5


def fetch_json(path: str, retries: int = 4):
    """GET a file from the open-data repo, asking for gzip to cut transfer ~10x."""
    request = urllib.request.Request(f"{config.OPEN_DATA_URL}/{path}", headers={"Accept-Encoding": "gzip"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                return json.loads(body)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def list_matches() -> pd.DataFrame:
    """Every match in the open data, with its competition, season and teams."""
    competitions = fetch_json("competitions.json")

    def season_matches(comp: dict) -> list[dict]:
        rows = []
        for m in fetch_json(f"matches/{comp['competition_id']}/{comp['season_id']}.json"):
            rows.append({
                "match_id": m["match_id"],
                "match_date": m["match_date"],
                "competition": comp["competition_name"],
                "season": comp["season_name"],
                "gender": comp["competition_gender"],
                "home_team": m["home_team"]["home_team_name"],
                "away_team": m["away_team"]["away_team_name"],
                "home_score": m["home_score"],
                "away_score": m["away_score"],
            })
        return rows

    with ThreadPoolExecutor(16) as pool:
        rows = [r for season in pool.map(season_matches, competitions) for r in season]
    return pd.DataFrame(rows).drop_duplicates("match_id").reset_index(drop=True)


def _name(obj: dict | None) -> str | None:
    return obj.get("name") if obj else None


def extract_shots(events: list[dict]) -> list[dict]:
    """Flatten a match's shots into rows, joining each to the pass that set it up.

    Penalty-shootout kicks are dropped: they aren't part of open play and
    their outcome doesn't count towards the match.
    """
    by_id = {e["id"]: e for e in events}
    rows = []
    for event in events:
        if _name(event.get("type")) != "Shot" or event.get("period") == PENALTY_SHOOTOUT_PERIOD:
            continue
        shot = event["shot"]
        end = shot.get("end_location") or [None, None, None]
        frame = shot.get("freeze_frame") or []
        row = {
            "shot_id": event["id"],
            "period": event["period"],
            "minute": event["minute"],
            "second": event["second"],
            "team": _name(event.get("team")),
            "player": _name(event.get("player")),
            "player_id": (event.get("player") or {}).get("id"),
            "position": _name(event.get("position")),
            "play_pattern": _name(event.get("play_pattern")),
            "x": event["location"][0],
            "y": event["location"][1],
            "end_x": end[0],
            "end_y": end[1] if len(end) > 1 else None,
            "end_z": end[2] if len(end) > 2 else None,
            "outcome": _name(shot.get("outcome")),
            "statsbomb_xg": shot.get("statsbomb_xg"),
            "shot_type": _name(shot.get("type")),
            "body_part": _name(shot.get("body_part")),
            "technique": _name(shot.get("technique")),
            "first_time": bool(shot.get("first_time", False)),
            "one_on_one": bool(shot.get("one_on_one", False)),
            "open_goal": bool(shot.get("open_goal", False)),
            "aerial_won": bool(shot.get("aerial_won", False)),
            "follows_dribble": bool(shot.get("follows_dribble", False)),
            "deflected": bool(shot.get("deflected", False)),
            "under_pressure": bool(event.get("under_pressure", False)),
            "has_freeze_frame": bool(frame),
            "freeze_frame": json.dumps([
                [round(p["location"][0], 1), round(p["location"][1], 1),
                 int(p["teammate"]), int(_name(p.get("position")) == "Goalkeeper")]
                for p in frame
            ]),
        }
        key_pass = by_id.get(shot.get("key_pass_id"))
        pass_ = (key_pass or {}).get("pass", {})
        row.update({
            "assisted": key_pass is not None,
            "pass_height": _name(pass_.get("height")),
            "pass_type": _name(pass_.get("type")),          # Corner, Free Kick, Throw-in, ...
            "pass_technique": _name(pass_.get("technique")),  # Through Ball, Inswinging, ...
            "pass_cross": bool(pass_.get("cross", False)),
            "pass_cut_back": bool(pass_.get("cut_back", False)),
            "pass_through_ball": bool(pass_.get("through_ball", False)) or _name(pass_.get("technique")) == "Through Ball",
            "pass_switch": bool(pass_.get("switch", False)),
            "pass_length": pass_.get("length"),
            "pass_start_x": key_pass["location"][0] if key_pass else None,
            "pass_start_y": key_pass["location"][1] if key_pass else None,
        })
        rows.append(row)
    return rows


def download_shots(matches: pd.DataFrame, workers: int = 16, log: Callable[[str], None] = print) -> None:
    """Fetch every match not yet cached and keep its shots under RAW_SHOTS_DIR."""
    config.RAW_SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    todo = [m for m in matches["match_id"] if not (config.RAW_SHOTS_DIR / f"{m}.json").exists()]
    log(f"{len(matches) - len(todo):,} matches cached, {len(todo):,} to fetch")

    def fetch(match_id: int) -> None:
        shots = extract_shots(fetch_json(f"events/{match_id}.json"))
        tmp = config.RAW_SHOTS_DIR / f"{match_id}.json.tmp"
        tmp.write_text(json.dumps(shots))
        tmp.rename(config.RAW_SHOTS_DIR / f"{match_id}.json")  # atomic: a killed run never leaves half a file

    failed = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(fetch, m): m for m in todo}
        for done, future in enumerate(as_completed(futures), 1):
            if future.exception():
                failed.append(futures[future])
            if done % 200 == 0 or done == len(todo):
                log(f"  {done:,}/{len(todo):,} fetched ({len(failed)} failed)")
    if failed:
        raise RuntimeError(f"{len(failed)} matches failed to download; re-run to retry: {failed[:10]}")


def build_shots_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Combine the per-match shot files with match metadata into one table."""
    frames = []
    for match_id in matches["match_id"]:
        shots = json.loads((config.RAW_SHOTS_DIR / f"{match_id}.json").read_text())
        if shots:
            frames.append(pd.DataFrame(shots).assign(match_id=match_id))
    shots = pd.concat(frames, ignore_index=True).merge(matches, on="match_id", how="left")
    shots["is_goal"] = (shots["outcome"] == "Goal").astype(int)
    return shots


def load_shots() -> pd.DataFrame:
    if not config.SHOTS_PARQUET.exists():
        raise FileNotFoundError(f"{config.SHOTS_PARQUET} not found - run scripts/download_data.py first.")
    return pd.read_parquet(config.SHOTS_PARQUET)
