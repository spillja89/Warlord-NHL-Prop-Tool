"""Forward price audit from frozen NHL slates and official graded outcomes.

Observed returns are per one unit at the captured book price. Historical
move percentages are intentionally not treated as predictive probabilities.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd


QUOTES = {
    "Points": ("Points_Line", "Points_Odds_Over", "Points_Book"),
    "Assists": ("Assists_Line", "Assists_Odds_Over", "Assists_Book"),
    "SOG": ("SOG_Line", "SOG_Odds_Over", "SOG_Book"),
    "Goal": ("Goal_Line", "Goal_Odds_Over", "Goal_Book"),
}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _price_bucket(implied: float) -> str:
    if implied < 0.4:
        return "Under 40%"
    if implied < 0.5:
        return "40–49%"
    if implied < 0.6:
        return "50–59%"
    return "60%+"


def price_records(graded: pd.DataFrame) -> pd.DataFrame:
    """One settled row per quoted market and fired move; no synthetic odds."""
    records = []
    for row in graded.to_dict("records"):
        if row.get("Grade_Status") != "FINAL":
            continue
        try:
            tags = json.loads(str(row.get("Fired_Moves") or "[]"))
        except (TypeError, ValueError):
            tags = []
        if not isinstance(tags, list):
            tags = []
        for market, (line_col, odds_col, book_col) in QUOTES.items():
            outcome = str(row.get(f"Outcome_{market}") or "")
            if outcome not in {"W", "L", "P"}:
                continue
            odds = _number(row.get(odds_col))
            line = _number(row.get(line_col))
            book = row.get(book_col)
            if market == "Goal" and odds is None and line == _number(row.get("ATG_Line")):
                odds = _number(row.get("ATG_Odds_Over"))
                book = row.get("ATG_Book")
            if odds is None or abs(odds) < 100 or line is None:
                continue
            implied = 100.0 / (odds + 100.0) if odds > 0 else -odds / (-odds + 100.0)
            profit = (odds / 100.0 if odds > 0 else 100.0 / -odds) if outcome == "W" else (-1.0 if outcome == "L" else 0.0)
            matching = [tag for tag in tags if isinstance(tag, dict) and tag.get("market") == market]
            for tag in matching:
                records.append({
                    "Date": str(row.get("Date")), "Game": row.get("Game"),
                    "Player": row.get("Player"), "Market": market,
                    "Move": str(tag.get("name")), "Kit": str(tag.get("kit_version") or row.get("Move_Kit_Version") or ""),
                    "Book": str(book or ""), "Line": line, "Odds": odds,
                    "Captured_UTC": row.get("Odds_Checked_UTC"),
                    "Break_even_pct": round(implied * 100.0, 2),
                    "Price_band": _price_bucket(implied), "Outcome": outcome,
                    "Profit_units": round(profit, 4),
                })
    return pd.DataFrame.from_records(records)


def summarize_prices(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(columns=["Market", "Move", "Line", "Price band", "Picks", "Dates",
                                     "Hit %", "Avg book break-even %", "Profit (1u/pick)",
                                     "ROI %", "Evidence"])
    rows = []
    keys = ["Market", "Move", "Kit", "Line", "Price_band"]
    for (market, move, kit, line, band), group in records.groupby(keys, dropna=False):
        settled = group[group["Outcome"].isin(["W", "L"])]
        picks = len(settled)
        if not picks:
            continue
        dates = int(settled["Date"].nunique())
        wins = int(settled["Outcome"].eq("W").sum())
        profit = float(settled["Profit_units"].sum())
        rows.append({
            "Market": market, "Move": move, "Kit": kit, "Line": line, "Price band": band,
            "Picks": picks, "Dates": dates, "Hit %": round(100.0 * wins / picks, 1),
            "Avg book break-even %": round(float(settled["Break_even_pct"].mean()), 1),
            "Profit (1u/pick)": round(profit, 2), "ROI %": round(100.0 * profit / picks, 1),
            "Evidence": "early sample" if dates < 20 or picks < 100 else "larger sample; still exploratory",
        })
    return pd.DataFrame(rows).sort_values(["Picks", "Dates"], ascending=False) if rows else pd.DataFrame()
