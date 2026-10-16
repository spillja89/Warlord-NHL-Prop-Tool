"""Exploratory PPP audit from frozen, graded pregame trackers.

No live ranking is changed. Signals are evaluated on the over 0.5 PPP line
using only fields saved before each game; repeated players and team contexts
make the observations correlated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from audit_pp_weights import auc


def named_main_move(raw: object) -> tuple[bool, bool]:
    try:
        moves = json.loads(raw) if isinstance(raw, str) else []
    except ValueError:
        moves = []
    if not isinstance(moves, list):
        return False, False
    markets = {str(move.get("market")) for move in moves
               if isinstance(move, dict) and str(move.get("kind")) != "STANCE"}
    return "Assists" in markets, "Points" in markets


def summarize(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {"n": 0}
    hit = rows["hit"]
    return {
        "n": len(rows),
        "dates": int(rows["Date"].nunique()),
        "hits": int(hit.sum()),
        "hit_pct": round(float(hit.mean() * 100), 1),
        "mean_book_break_even_pct": round(float(rows["break_even"].mean() * 100), 1),
        "flat_unit_roi_pct": round(float(rows["profit"].mean() * 100), 1),
    }


def build_report(paths: list[Path]) -> dict:
    frames = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        required = {"Date", "Game", "Team", "Opp", "Player_ID", "Grade_Status",
                    "Match_Status_PPP", "BDL_PPP_Line", "BDL_PPP_Odds",
                    "Outcome_PPP", "Odds_Checked_UTC", "StartTimeUTC"}
        if required.issubset(frame.columns):
            frames.append(frame)
    if not frames:
        return {"status": "no_compatible_graded_slates"}
    rows = pd.concat(frames, ignore_index=True)
    for column in ("BDL_PPP_Line", "BDL_PPP_Odds", "PP_Role",
                   "PP_TOI_per_game", "PPP10_total", "PP_Matchup",
                   "Team_PP_xGF60", "Opp_PK_xGA60", "PP_iXG60", "PP_iXA60",
                   "Opp_GAA", "Opp_SV", "Goalie_Weak"):
        rows[column] = pd.to_numeric(rows.get(column), errors="coerce")
    rows = rows.loc[
        rows["Grade_Status"].eq("FINAL")
        & rows["Match_Status_PPP"].eq("OK")
        & rows["BDL_PPP_Line"].eq(0.5)
        & rows["BDL_PPP_Odds"].notna()
        & rows["BDL_PPP_Odds"].ne(0)
        & rows["Outcome_PPP"].isin(("W", "L"))
    ].copy()
    checked = pd.to_datetime(rows["Odds_Checked_UTC"], errors="coerce", utc=True)
    started = pd.to_datetime(rows["StartTimeUTC"], errors="coerce", utc=True)
    rows = rows.loc[checked.notna() & started.notna() & checked.lt(started)].copy()
    rows = rows.drop_duplicates(["Date", "Game", "Team", "Player_ID"])
    if rows.empty:
        return {"status": "no_eligible_pregame_pp_props"}

    odds = rows["BDL_PPP_Odds"]
    rows["break_even"] = (100 / (odds + 100)).where(odds > 0, -odds / (-odds + 100))
    rows["hit"] = rows["Outcome_PPP"].eq("W").astype(int)
    rows["profit"] = rows["hit"] * odds.apply(
        lambda value: value / 100 if value > 0 else 100 / -value) - (1 - rows["hit"])
    moves = rows.get("Fired_Moves", pd.Series(index=rows.index, dtype=object)).map(named_main_move)
    rows["assist_move"] = moves.map(lambda pair: pair[0])
    rows["points_move"] = moves.map(lambda pair: pair[1])
    rows["pp1"] = rows["PP_Role"].ge(2)
    rows["pp_3min"] = rows["PP_TOI_per_game"].ge(3)
    rows["recent_2ppp"] = rows["PPP10_total"].ge(2)
    rows["boost"] = rows.get("PP_BOOST", pd.Series(index=rows.index, dtype=object)).eq("ON")
    rows["context_60"] = rows["PP_Matchup"].ge(60)
    for column, team_column, signal in (("Team_PP_xGF60", "Team", "strong_pp"),
                                        ("Opp_PK_xGA60", "Opp", "weak_pk")):
        team_rates = rows.groupby(["Date", team_column])[column].median()
        team_percentile = team_rates.groupby(level=0).rank(pct=True)
        keys = pd.MultiIndex.from_frame(rows[["Date", team_column]])
        rows[signal] = team_percentile.reindex(keys).to_numpy() >= 0.75

    dates = sorted(rows["Date"].unique().tolist())
    split = len(dates) // 2
    signals = {}
    for name in ("pp1", "pp_3min", "recent_2ppp", "boost", "context_60",
                 "strong_pp", "weak_pk", "assist_move", "points_move"):
        selected = rows.loc[rows[name]]
        signals[name] = {
            "all": summarize(selected),
            "early": summarize(selected.loc[selected["Date"].isin(dates[:split])]),
            "later": summarize(selected.loc[selected["Date"].isin(dates[split:])]),
            "pp1": summarize(selected.loc[selected["pp1"]]),
        }
    continuous = {}
    for column in ("PP_TOI_per_game", "PPP10_total", "PP_Matchup",
                   "Team_PP_xGF60", "Opp_PK_xGA60", "PP_iXG60", "PP_iXA60"):
        continuous[column] = {
            "coverage": int(rows[column].notna().sum()),
            "auc": auc(rows["hit"], rows[column]),
            "pp1_auc": auc(rows.loc[rows["pp1"], "hit"], rows.loc[rows["pp1"], column]),
            "early_auc": auc(rows.loc[rows["Date"].isin(dates[:split]), "hit"],
                             rows.loc[rows["Date"].isin(dates[:split]), column]),
            "later_auc": auc(rows.loc[rows["Date"].isin(dates[split:]), "hit"],
                             rows.loc[rows["Date"].isin(dates[split:]), column]),
        }
    seasons = rows.get("Model_Stats_Season", pd.Series(index=rows.index, dtype=object))
    latest_stats_season = max(seasons.dropna().astype(str), default=None)
    current = rows.loc[seasons.eq(latest_stats_season)].copy()
    confirmed = rows.loc[rows.get("Opp_Goalie_Status", pd.Series(index=rows.index, dtype=object)).eq("Confirmed")]
    goalie_check = {}
    for column, direction in (("Opp_GAA", 1), ("Opp_SV", -1), ("Goalie_Weak", 1)):
        goalie_check[column] = {
            "coverage": int(rows[column].notna().sum()),
            "all_auc": auc(rows["hit"], direction * rows[column]),
            "confirmed_auc": auc(confirmed["hit"], direction * confirmed[column]),
            "latest_stats_season_auc": auc(current["hit"], direction * current[column]),
        }
    return {
        "status": "exploratory",
        "first_date": dates[0], "last_date": dates[-1],
        "all": summarize(rows), "pp1": summarize(rows.loc[rows["pp1"]]),
        "early_dates": dates[:split], "later_dates": dates[split:],
        "signals": signals, "continuous": continuous,
        "model_stats_seasons": seasons.value_counts(dropna=False).to_dict(),
        "latest_stats_season": latest_stats_season,
        "goalie_check": goalie_check,
        "current_season_source": {
            "all": summarize(current),
            "pp1": summarize(current.loc[current["pp1"]]),
            "signals": {name: summarize(current.loc[current[name]])
                        for name in ("pp_3min", "recent_2ppp", "context_60",
                                     "strong_pp", "weak_pk", "assist_move", "points_move")},
            "continuous_auc": {column: auc(current["hit"], current[column])
                               for column in continuous},
        },
        "method_note": "One record per priced player-game. Pregame saved tracker only; no postgame features. Rates are descriptive, heavily correlated within teams and dates, and not validated betting probabilities. Older graded trackers lack opposing goalie games played, so tiny goalie samples cannot be screened. The best captured book price may not have been available to every bettor. No live PP ranking or move threshold changes.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    report = build_report(sorted((args.repo / "output" / "graded").glob("tracker_*_GRADED.csv")))
    dest = args.repo / "output" / "graded" / "pp_early_audit.json"
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PP early audit: {report['status']} -> {dest}")


if __name__ == "__main__":
    main()
