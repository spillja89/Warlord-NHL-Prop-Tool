"""Grade frozen 2026-27 goal signals against the price saved before each game.

This is a shadow audit. It never edits a tracker, move rule, confidence weight,
or board order. Run after grade_daily_tracker.py; output is cumulative.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GRADED = ROOT / "output" / "graded"
DESTINATION = GRADED / "goal_move_audit.json"


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def break_even(american):
    odds = number(american)
    if not odds:
        return None
    return 100 / (100 + odds) if odds > 0 else -odds / (100 - odds)


def summary(rows):
    if not rows:
        return {"n": 0}
    wins = sum(row["Outcome_Goal"] == "W" for row in rows)
    expected = sum(break_even(row["Goal_Odds_Over"]) for row in rows)
    return {"n": len(rows), "wins": wins,
            "hit_pct": round(100 * wins / len(rows), 1),
            "book_break_even_pct": round(100 * expected / len(rows), 1),
            "hit_minus_book_pp": round(100 * (wins - expected) / len(rows), 1),
            "dates": len({row["Date"] for row in rows}),
            "players": len({row["Player"] for row in rows})}


def season(date_text):
    day = date.fromisoformat(date_text)
    start = day.year if day.month >= 8 else day.year - 1
    return f"{start}-{start + 1}"


def rows_from_archive(directory=GRADED):
    rows = []
    for path in sorted(directory.glob("tracker_2026-*_GRADED.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if (row.get("Grade_Status") == "FINAL"
                        and row.get("Outcome_Goal") in {"W", "L"}
                        and number(row.get("Goal_Line")) == 0.5
                        and break_even(row.get("Goal_Odds_Over")) is not None):
                    rows.append(row)
    return rows


def build_report(rows):
    green = [r for r in rows if str(r.get("Matrix_Goal", "")).casefold() == "green"]
    current_green = [r for r in green if r.get("Model_Stats_Season") == season(r["Date"])]
    by_score = {
        "below_80": [r for r in current_green if (number(r.get("Conf_Goal")) or 0) < 80],
        "80_to_83": [r for r in current_green if 80 <= (number(r.get("Conf_Goal")) or 0) < 84],
        "84_to_89": [r for r in current_green if 84 <= (number(r.get("Conf_Goal")) or 0) < 90],
        "90_plus": [r for r in current_green if (number(r.get("Conf_Goal")) or 0) >= 90],
    }
    by_move = defaultdict(list)
    for row in rows:
        try:
            tags = json.loads(row.get("Fired_Moves") or "[]")
        except (TypeError, json.JSONDecodeError):
            continue
        for name in {tag.get("name") for tag in tags if isinstance(tag, dict)
                     and tag.get("market") == "Goal" and tag.get("name")}:
            by_move[name].append(row)
    goal_only = [r for r in current_green if (number(r.get("Goal_mu")) or 0) >= .4048
                 and (number(r.get("Conf_Goal")) or 0) >= 84]
    return {
        "version": "2026-10-22-goal-monitor-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latest_slate": max((r["Date"] for r in rows), default=None),
        "scope": "graded over 0.5 goal prices frozen before games; regular season only",
        "all_priced": summary(rows),
        "green": summary(green),
        "current_source_green": summary(current_green),
        "current_source_conf_goal": {key: summary(group) for key, group in by_score.items()},
        "current_source_goal_only_screen": summary(goal_only),
        "named_frozen_goal_moves": {key: summary(group) for key, group in sorted(by_move.items())},
        "by_date_green": {day: summary([r for r in green if r["Date"] == day])
                          for day in sorted({r["Date"] for r in green})},
        "note": "Observed hit rates are not tonight's probabilities; named moves can overlap.",
    }


if __name__ == "__main__":
    report = build_report(rows_from_archive())
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved goal audit through {report['latest_slate']}: "
          f"{report['green']['n']} Green goal rows, "
          f"{len(report['named_frozen_goal_moves'])} named moves")

