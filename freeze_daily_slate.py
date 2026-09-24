"""Save the latest available pregame row for each player and game."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def freeze_rows(current: pd.DataFrame, previous: pd.DataFrame, now: datetime) -> pd.DataFrame:
    required = {"Date", "Game", "Player", "Team", "StartTimeUTC", "Fired_Moves", "Move_Kit_Version"}
    if not required.issubset(current.columns):
        raise ValueError(f"Tracker missing freeze fields: {sorted(required - set(current.columns))}")
    starts = pd.to_datetime(current["StartTimeUTC"], utc=True, errors="coerce")
    if starts.isna().any():
        raise ValueError("Cannot freeze slate: one or more game start times are missing")
    if current["Fired_Moves"].isna().any() or current["Move_Kit_Version"].isna().any():
        raise ValueError("Cannot freeze slate: move tags or version are missing")
    eligible = current.loc[starts > now.astimezone(timezone.utc)].copy()
    eligible["Frozen_At_UTC"] = now.astimezone(timezone.utc).isoformat()
    if previous.empty:
        result = eligible
    else:
        if not required.issubset(previous.columns) or "Frozen_At_UTC" not in previous.columns:
            raise ValueError("Existing frozen slate is missing provenance fields")
        combined = pd.concat([previous, eligible], ignore_index=True)
        key = ["Game_ID", "Player_ID"] if combined[["Game_ID", "Player_ID"]].notna().all().all() else ["Date", "Game", "Team", "Player"]
        result = combined.drop_duplicates(subset=key, keep="last")
    return result.sort_values(["Game", "Team", "Player"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    current = pd.read_csv(args.tracker, low_memory=False)
    if current.empty or set(current["Date"].astype(str)) != {args.date}:
        raise ValueError("Latest tracker has no rows or the wrong slate date")
    previous = pd.read_csv(args.output, low_memory=False) if args.output.exists() else pd.DataFrame()
    result = freeze_rows(current, previous, datetime.now(timezone.utc))
    if result.empty:
        print("No pregame rows to freeze")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".csv.tmp")
    result.to_csv(temporary, index=False)
    temporary.replace(args.output)
    print(f"Frozen {len(result)} pregame rows -> {args.output}")


if __name__ == "__main__":
    main()
