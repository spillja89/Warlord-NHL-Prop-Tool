"""Pregame research tags for Green over 1.5 Points plays.

These tags describe player and matchup context. They never admit a card, fire
a named move, change a projection, or claim a tested hit probability.
"""

from __future__ import annotations

import math

import pandas as pd


VERSION = "2026-10-28-points15-research-v1"
FORWARD_START = "2026-10-28"


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def context(row) -> dict:
    """Classify only current-season Green 1.5-point rows; keep missing distinct."""
    line = _number(row.get("Points_Line"))
    green = str(row.get("Matrix_Points", "")).strip().casefold() == "green"
    season = str(row.get("Model_Stats_Season", "")).strip() == "2026-2027"
    eligible = line == 1.5 and green and season
    pp_xgf60 = _number(row.get("PP_OnIce_xGF60"))
    attempts60 = _number(row.get("i5v5_iCF60"))
    opp_hdca60 = _number(row.get("opp_5v5_HDCA60"))
    pp = eligible and pp_xgf60 is not None and pp_xgf60 >= 9.0
    shots = eligible and attempts60 is not None and attempts60 >= 15.0
    return {
        "eligible": eligible,
        "metrics_complete": eligible and pp_xgf60 is not None and attempts60 is not None,
        "pp_xgf60": pp_xgf60,
        "attempts60": attempts60,
        "opp_hdca60": opp_hdca60,
        "pp_pressure": pp,
        "shot_generator": shots,
        "dual": pp and shots,
        "stingy_defense": eligible and opp_hdca60 is not None and opp_hdca60 < 1.2,
    }


def attach_flags(tracker: pd.DataFrame) -> pd.DataFrame:
    """Freeze research tags with the pregame tracker for future grading."""
    result = tracker.copy()
    flags = [context(row) for _, row in result.iterrows()]
    result["P15_Research_Version"] = VERSION
    for source, column in (
        ("eligible", "P15_Research_Eligible"),
        ("metrics_complete", "P15_Research_Metrics_Complete"),
        ("pp_pressure", "P15_Research_PP_Pressure"),
        ("shot_generator", "P15_Research_Shot_Generator"),
        ("dual", "P15_Research_Dual"),
        ("stingy_defense", "P15_Research_Stingy_Defense"),
    ):
        result[column] = [item[source] for item in flags]
    return result


def _break_even(odds: float) -> float:
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def _saved_true(value) -> bool:
    return str(value).strip().casefold() in {"true", "1"}


def summarize(graded: pd.DataFrame, *, forward: bool) -> pd.DataFrame:
    """Compare overlapping tags with the matched Green sample at saved prices."""
    columns = ["Group", "Wins", "Picks", "Hit %", "Avg book BE %", "Players", "Slates", "Net units"]
    required = {"Date", "Game", "Player", "Points_Line", "Points_Odds_Over",
                "Matrix_Points", "Model_Stats_Season", "Outcome_Points", "Match_Status_Points"}
    saved_flags = ("P15_Research_Version", "P15_Research_Metrics_Complete",
                   "P15_Research_PP_Pressure", "P15_Research_Shot_Generator",
                   "P15_Research_Dual", "P15_Research_Stingy_Defense")
    if forward:
        required.update(saved_flags)
    if graded.empty or not required.issubset(graded.columns):
        return pd.DataFrame(columns=columns)
    rows = graded.copy()
    dates = rows["Date"].astype(str)
    rows = rows[dates.ge(FORWARD_START) if forward else dates.lt(FORWARD_START)]
    rows = rows[
        pd.to_numeric(rows["Points_Line"], errors="coerce").eq(1.5)
        & rows["Matrix_Points"].astype(str).str.strip().str.casefold().eq("green")
        & rows["Outcome_Points"].isin(("W", "L"))
        & rows["Match_Status_Points"].eq("OK")
    ].drop_duplicates(["Date", "Game", "Player"])
    rows = rows[rows["Model_Stats_Season"].astype(str).eq("2026-2027")]
    if forward:
        rows = rows[rows["P15_Research_Version"].astype(str).eq(VERSION)]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    odds = pd.to_numeric(rows["Points_Odds_Over"], errors="coerce")
    rows = rows[odds.notna() & odds.ne(0)].copy()
    rows["_be"] = odds.loc[rows.index].map(_break_even) * 100
    rows["_win"] = rows["Outcome_Points"].eq("W")
    rows["_profit"] = [
        ((price / 100) if price > 0 else (100 / -price)) if hit else -1
        for price, hit in zip(odds.loc[rows.index], rows["_win"])
    ]
    if forward:
        for output, source in (
            ("_complete", "P15_Research_Metrics_Complete"),
            ("_pp", "P15_Research_PP_Pressure"),
            ("_shots", "P15_Research_Shot_Generator"),
            ("_dual", "P15_Research_Dual"),
            ("_stingy", "P15_Research_Stingy_Defense"),
        ):
            rows[output] = rows[source].map(_saved_true)
    else:
        flags = [context(row) for _, row in rows.iterrows()]
        rows["_complete"] = [item["metrics_complete"] for item in flags]
        rows["_pp"] = [item["pp_pressure"] for item in flags]
        rows["_shots"] = [item["shot_generator"] for item in flags]
        rows["_dual"] = [item["dual"] for item in flags]
        rows["_stingy"] = [item["stingy_defense"] for item in flags]
    complete = rows[rows["_complete"]]
    groups = (
        ("Green baseline · all", rows),
        ("Stingy 5v5 defense · caution", rows[rows["_stingy"]]),
        ("Green baseline · PP + shots recorded", complete),
        ("PP pressure ≥9 xGF/60", complete[complete["_pp"]]),
        ("5v5 shot attempts ≥15/60", complete[complete["_shots"]]),
        ("Both research signals", complete[complete["_dual"]]),
    )
    result = []
    for name, subset in groups:
        if subset.empty:
            continue
        result.append({
            "Group": name, "Wins": int(subset["_win"].sum()), "Picks": len(subset),
            "Hit %": round(subset["_win"].mean() * 100, 1),
            "Avg book BE %": round(subset["_be"].mean(), 1),
            "Players": subset["Player"].nunique(), "Slates": subset["Date"].nunique(),
            "Net units": round(subset["_profit"].sum(), 1),
        })
    return pd.DataFrame(result, columns=columns)
