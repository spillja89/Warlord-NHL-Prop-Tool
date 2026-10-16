"""Offline PP unit audit using NHL game logs and saved pregame prop prices.

PP1/PP2 are estimated from each team's *prior* PP ice time among skaters who
appeared in a game. The NHL game log does not identify formal line assignments.
This script does not alter the live board.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


API = "https://api.nhle.com/stats/rest/en/skater/powerplay"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json", "Referer": "https://www.nhl.com/"}


def fetch_month(season: int, year: int, month: int, cache: Path) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"nhl_pp_{season}_{year}{month:02d}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(payload)
    start = f"{year}-{month:02d}-01"
    next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
    end = f"{next_year}-{next_month:02d}-01"
    expression = f'seasonId={season}{season + 1} and gameTypeId=2 and gameDate>="{start}" and gameDate<"{end}"'
    records = []
    offset = 0
    while True:
        params = urllib.parse.urlencode({"isAggregate": "false", "isGame": "true", "cayenneExp": expression,
                                         "limit": 1000, "start": offset})
        request = urllib.request.Request(f"{API}?{params}", headers=HEADERS)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    result = json.load(response)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1 + attempt * 2)
        batch = result.get("data", [])
        records.extend(batch)
        offset += len(batch)
        if not batch or offset >= result.get("total", 0):
            break
    path.write_text(json.dumps(records, separators=(",", ":")), encoding="utf-8")
    return pd.DataFrame(records)


def season_logs(season: int, cache: Path, last_month: tuple[int, int] | None = None) -> pd.DataFrame:
    months = [(season, month) for month in (10, 11, 12)] + [(season + 1, month) for month in (1, 2, 3, 4)]
    if last_month:
        months = [pair for pair in months if pair <= last_month]
    parts = [fetch_month(season, year, month, cache) for year, month in months]
    result = pd.concat(parts, ignore_index=True)
    return result.drop_duplicates(["gameId", "playerId"])


def add_prior_roles(logs: pd.DataFrame, min_prior_games: int = 3) -> pd.DataFrame:
    logs = logs.copy()
    logs["gameDate"] = pd.to_datetime(logs["gameDate"])
    logs["ppTimeOnIce"] = pd.to_numeric(logs["ppTimeOnIce"], errors="coerce").fillna(0)
    logs["ppPoints"] = pd.to_numeric(logs["ppPoints"], errors="coerce").fillna(0)
    logs = logs.sort_values(["gameDate", "gameId", "playerId"])
    group = logs.groupby(["teamAbbrev", "playerId"], sort=False)
    logs["prior_games"] = group.cumcount()
    logs["prior_pp_sec_per_game"] = group["ppTimeOnIce"].transform(
        lambda series: series.shift().rolling(5, min_periods=1).mean())
    eligible = logs["prior_games"].ge(min_prior_games)
    ranks = logs.loc[eligible].groupby(["gameId", "teamAbbrev"])["prior_pp_sec_per_game"].rank(
        method="first", ascending=False)
    logs["estimated_unit"] = pd.NA
    logs.loc[ranks.index[ranks.le(5)], "estimated_unit"] = "PP1"
    logs.loc[ranks.index[ranks.gt(5) & ranks.le(10)], "estimated_unit"] = "PP2"
    logs["ppp_hit"] = logs["ppPoints"].ge(1)
    return logs


def summarize(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {"player_games": 0}
    return {
        "player_games": int(len(rows)),
        "team_games": int(rows[["gameId", "teamAbbrev"]].drop_duplicates().shape[0]),
        "pp_points": int(rows["ppPoints"].sum()),
        "players_with_ppp": int(rows["ppp_hit"].sum()),
        "ppp_hit_pct": round(float(rows["ppp_hit"].mean() * 100), 2),
        "pp_points_per_player_game": round(float(rows["ppPoints"].mean()), 3),
        "actual_pp_min_per_player_game": round(float(rows["ppTimeOnIce"].mean() / 60), 2),
        "pp_points_per_60_actual_pp_minutes": round(float(rows["ppPoints"].sum() * 3600 / rows["ppTimeOnIce"].sum()), 2)
        if rows["ppTimeOnIce"].sum() else None,
        "prior_pp_min_per_game": round(float(rows["prior_pp_sec_per_game"].mean() / 60), 2),
    }


def paired_team_games(rows: pd.DataFrame) -> dict:
    counts = rows.groupby(["gameId", "teamAbbrev", "estimated_unit"]).agg(
        pp_points=("ppPoints", "sum"), skaters=("playerId", "size")).reset_index()
    points = counts.pivot(index=["gameId", "teamAbbrev"], columns="estimated_unit", values="pp_points")
    skaters = counts.pivot(index=["gameId", "teamAbbrev"], columns="estimated_unit", values="skaters")
    if not {"PP1", "PP2"}.issubset(points.columns):
        return {"paired_team_games": 0}
    paired = points.loc[skaters["PP1"].ge(3) & skaters["PP2"].ge(3)].dropna()
    return {"paired_team_games": int(len(paired)), "pp1_more": int((paired["PP1"] > paired["PP2"]).sum()),
            "tied": int((paired["PP1"] == paired["PP2"]).sum()),
            "pp2_more": int((paired["PP2"] > paired["PP1"]).sum())}


def priced_current(logs: pd.DataFrame, paths: list[Path]) -> dict:
    if not paths:
        return {"status": "no_saved_prices"}
    frames = [pd.read_csv(path, low_memory=False) for path in paths]
    rows = pd.concat(frames, ignore_index=True)
    needed = ["Date", "Game", "Team", "Player_ID", "BDL_PPP_Line", "BDL_PPP_Odds",
              "Odds_Checked_UTC", "StartTimeUTC", "Grade_Status", "Outcome_PPP", "Match_Status_PPP"]
    if any(column not in rows.columns for column in needed):
        return {"status": "missing_saved_price_fields"}
    line = pd.to_numeric(rows["BDL_PPP_Line"], errors="coerce")
    odds = pd.to_numeric(rows["BDL_PPP_Odds"], errors="coerce")
    checked = pd.to_datetime(rows["Odds_Checked_UTC"], errors="coerce", utc=True)
    started = pd.to_datetime(rows["StartTimeUTC"], errors="coerce", utc=True)
    rows = rows.loc[line.eq(0.5) & odds.notna() & odds.ne(0) & checked.lt(started)
                    & rows["Grade_Status"].eq("FINAL") & rows["Match_Status_PPP"].eq("OK")
                    & rows["Outcome_PPP"].isin(("W", "L"))].copy()
    rows = rows.drop_duplicates(["Date", "Game", "Team", "Player_ID"])
    rows["Player_ID"] = pd.to_numeric(rows["Player_ID"], errors="coerce")
    rows["Date"] = pd.to_datetime(rows["Date"]).dt.strftime("%Y-%m-%d")
    logs = logs.copy()
    logs["Date"] = logs["gameDate"].dt.strftime("%Y-%m-%d")
    merged = rows.merge(logs[["Date", "teamAbbrev", "playerId", "estimated_unit", "prior_games", "ppPoints"]],
                        left_on=["Date", "Team", "Player_ID"], right_on=["Date", "teamAbbrev", "playerId"],
                        how="left", validate="one_to_one")
    matched = merged.loc[merged["ppPoints"].notna()].copy()
    grade_disagreements = int(matched["Outcome_PPP"].eq("W").ne(matched["ppPoints"].ge(1)).sum())
    merged = merged.loc[merged["estimated_unit"].isin(("PP1", "PP2"))].copy()
    odds = pd.to_numeric(merged["BDL_PPP_Odds"])
    merged["book_break_even"] = np.where(odds > 0, 100 / (odds + 100), -odds / (-odds + 100))
    merged["hit"] = merged["Outcome_PPP"].eq("W")
    merged["profit"] = np.where(merged["hit"], np.where(odds > 0, odds / 100, 100 / -odds), -1)

    def unit_result(group: pd.DataFrame) -> dict:
        date_groups = [item for _, item in group.groupby("Date")]
        rng = np.random.default_rng(20261016)
        draws = []
        for _ in range(5000):
            sampled = [date_groups[index] for index in rng.integers(len(date_groups), size=len(date_groups))]
            profit_sum = sum(float(item["profit"].sum()) for item in sampled)
            count = sum(len(item) for item in sampled)
            draws.append(profit_sum / count * 100)
        return {"props": int(len(group)), "hits": int(group["hit"].sum()),
                "hit_pct": round(float(group["hit"].mean() * 100), 2),
                "mean_book_break_even_pct": round(float(group["book_break_even"].mean() * 100), 2),
                "flat_unit_roi_pct": round(float(group["profit"].mean() * 100), 2),
                "date_bootstrap_roi_95pct": [round(float(x), 1) for x in np.percentile(draws, (2.5, 97.5))],
                "dates": int(group["Date"].nunique())}

    return {
        "matched_priced_props": int(len(merged)),
        "official_result_mismatches": grade_disagreements,
        "official_result_comparisons": int(len(matched)),
        "dates": int(merged["Date"].nunique()),
        "by_estimated_unit": {unit: unit_result(group) for unit, group in merged.groupby("estimated_unit")},
        "by_date": {
            date: {unit: {"props": int(len(group)), "hits": int(group["hit"].sum()),
                          "flat_unit_roi_pct": round(float(group["profit"].mean() * 100), 1)}
                   for unit, group in day.groupby("estimated_unit")}
            for date, day in merged.groupby("Date")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("cache/pp_unit_audit"))
    parser.add_argument("--out", type=Path, default=Path("reports/pp_unit_history_2026-10-16.json"))
    parser.add_argument("--graded", type=Path, default=Path("output/graded"))
    args = parser.parse_args()
    prior = add_prior_roles(season_logs(2025, args.cache))
    current = add_prior_roles(season_logs(2026, args.cache, (2026, 10)))
    report = {
        "method": "Top five and next five dressed skaters by their own previous five games' PP ice time; three prior games required. Formal PP units are not recorded by this NHL endpoint.",
        "source": API,
        "historical_2025_26": {unit: summarize(group) for unit, group in prior.groupby("estimated_unit")},
        "historical_2025_26_paired": paired_team_games(prior.dropna(subset=["estimated_unit"])),
        "historical_2025_26_by_month": {
            str(month): {unit: summarize(group) for unit, group in frame.groupby("estimated_unit")}
            for month, frame in prior.groupby(prior["gameDate"].dt.to_period("M"))
        },
        "current_2026_27": {unit: summarize(group) for unit, group in current.groupby("estimated_unit")},
        "current_2026_27_paired": paired_team_games(current.dropna(subset=["estimated_unit"])),
        "current_saved_prices": priced_current(current, sorted(args.graded.glob("tracker_*_GRADED.csv"))),
        "limitations": ["Historical pregame PP prop prices are not archived locally.",
                        "Player PP points count scorer and assists; they are not unit goals.",
                        "Active players are identified from final game records, while unit estimates use prior PP minutes."],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
