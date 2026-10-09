#!/usr/bin/env python
"""Download every StatsBomb Open Data match and build data/processed/shots.parquet.

Usage:
    python scripts/download_data.py

~4,000 matches, ~1.2 GB transferred (gzip), a few minutes. Only shots and
their assisting passes are kept on disk (~50-100 MB). Re-running resumes.
"""

import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_xg import config, data  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001


def main() -> None:
    matches = data.list_matches()
    print(f"{len(matches):,} matches in {matches.groupby(['competition', 'season']).ngroups} competition-seasons")
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(config.PROCESSED_DIR / "matches.parquet", index=False)
    data.download_shots(matches, log=print)
    shots = data.build_shots_table(matches)
    shots.to_parquet(config.SHOTS_PARQUET, index=False)
    print(f"Wrote {len(shots):,} shots ({shots['is_goal'].sum():,} goals) to {config.SHOTS_PARQUET}")


if __name__ == "__main__":
    main()
