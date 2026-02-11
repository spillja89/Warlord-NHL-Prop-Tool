#!/usr/bin/env python
"""
grade_tracker_baseline_b.py — Baseline B Assists grader (ENGINE-ONLY)

What it does:
- Grades ASSISTS outcomes for rows where:
    - Assists_Line == 0.5
    - Matrix_Assists == "Green"
  EVEN IF Plays_EV_Assists == False / missing.

What it does NOT do:
- Does not change your production grade_tracker.py
- Does not touch app.py / nhl_edge.py / gating / EV logic

Usage (PowerShell, from repo root):
  python grade_tracker_baseline_b.py --tracker "Tracker_2-01-2026.csv"

Output:
  Tracker_2-01-2026_BASELINE_B_GRADED.csv
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# We reuse YOUR existing grader helpers that already work.
import grade_tracker as gt


def _is_green(x: Any) -> bool:
    return str(x or "").strip().lower() == "green"


def _is_half(x: Any) -> bool:
    try:
        return float(x) == 0.5
    except Exception:
        return False


def grade_row_baseline_b(
    r: pd.Series,
    matchup: Optional[Tuple[str, str]],
    idx_cache: Dict[Tuple[str, str], Dict[str, Dict[str, float]]],
) -> Dict[str, Any]:
    """
    Copy of grade_tracker.grade_row logic, but ONLY for Assists and
    with Baseline-B eligibility:
      - Assists_Line == 0.5
      - Matrix_Assists == Green
      - EV flag is ignored
    """
    out: Dict[str, Any] = {}

    # Baseline B eligibility gate (per-row)
    if not (_is_half(r.get("Assists_Line")) and _is_green(r.get("Matrix_Assists"))):
        return out

    # Need matchup stats to grade
    if matchup is None or matchup not in idx_cache:
        out["Match_Status_Assists"] = "NO_GAME"
        return out

    stat_idx = idx_cache.get(matchup)
    if not stat_idx:
        out["Match_Status_Assists"] = "NO_STATS"
        return out

    player_norm = gt._norm_name(str(r.get("Player", "") or ""))
    matched_key = gt._match_player_name(player_norm, stat_idx)

    if not matched_key:
        out["Match_Status_Assists"] = "NO_MATCH"
        return out

    st = stat_idx[matched_key]

    line = gt._num(r.get("Assists_Line"))
    if line is None:
        out["Match_Status_Assists"] = "NO_LINE"
        return out

    actual = float(st.get("assists", 0.0))
    out["Actual_Assists"] = actual
    out["Outcome_Assists"] = gt._settle_over(actual, float(line))
    out["Match_Status_Assists"] = "OK"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", required=True, help="Path to Tracker_*.csv")
    ap.add_argument("--out", default="", help="Output path (default: adds _BASELINE_B_GRADED)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.tracker, low_memory=False)
    slate_date = gt._parse_slate_date(df, args.tracker)

    game_ids = gt.fetch_game_ids_for_local_slate(slate_date)
    if not game_ids:
        raise SystemExit(f"No games found for local slate date {slate_date} via schedule API.")

    idx_cache: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    for matchup, gid in game_ids.items():
        box = gt.fetch_boxscore(gid)
        idx_cache[matchup] = gt.build_player_stat_index(box)

    if args.debug:
        sample = list(idx_cache.keys())[:5]
        nonempty = sum(1 for v in idx_cache.values() if isinstance(v, dict) and len(v) > 0)
        print(f"[debug] slate_date={slate_date} games={len(idx_cache)} nonempty_stat_indexes={nonempty} sample_matchups={sample}")

    graded_rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        matchup = gt.resolve_matchup_key(r, idx_cache)
        graded_rows.append(grade_row_baseline_b(r, matchup, idx_cache))

    out_df = pd.concat([df, pd.DataFrame(graded_rows)], axis=1)

    out_path = args.out.strip() or args.tracker.replace(".csv", "_BASELINE_B_GRADED.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved Baseline B graded tracker: {out_path}")


if __name__ == "__main__":
    main()
