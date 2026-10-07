"""Read real power play point prices from a generated tracker."""

from __future__ import annotations

import math

import pandas as pd
import requests


COLUMNS = ("Game", "Player", "Team", "Opponent", "PPP line", "Over odds",
           "Book", "Book break-even %", "PP unit", "PP TOI/game",
           "Opp PK xGA/60", "PP matchup /100", "PPP last 10",
           "Stats season", "Context")


def summarize_pp_usage(rows: list[dict], before_date: str) -> dict[int, tuple[float, int]]:
    """Average official PP minutes per completed regular-season game before a slate."""
    totals: dict[int, list[float]] = {}
    for row in rows:
        game_date = str(row.get("gameDate") or "")[:10]
        if not game_date or game_date >= before_date:
            continue
        try:
            player_id = int(row["playerId"])
            seconds = float(row["ppTimeOnIce"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(seconds) or seconds < 0:
            continue
        bucket = totals.setdefault(player_id, [0.0, 0.0])
        bucket[0] += seconds
        bucket[1] += 1
    return {pid: (round(total / games / 60, 1), int(games))
            for pid, (total, games) in totals.items()}


def fetch_current_pp_usage(season_id: int, before_date: str) -> dict[int, tuple[float, int]]:
    """Get current-season PP usage from official NHL game-level TOI records."""
    response = requests.get(
        "https://api.nhle.com/stats/rest/en/skater/timeonice",
        params={"cayenneExp": f"seasonId={season_id} and gameTypeId=2",
                "isAggregate": "false", "isGame": "true", "limit": "-1", "start": "0"},
        timeout=12,
    )
    response.raise_for_status()
    payload = response.json()
    return summarize_pp_usage(payload.get("data", []), before_date)


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _break_even_pct(odds: float) -> float:
    return round(100 * (100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)), 1)


def _is_true(value) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _unit_label(row: dict) -> str:
    if _is_true(row.get("Team_Changed")):
        return "New team · verify unit"
    role = _finite(row.get("PP_Role"))
    if role is None:
        return "Unit unknown"
    return "PP1 history" if role >= 2 else ("PP2 history" if role >= 1 else "No PP unit in history")


def _context_label(row: dict) -> str:
    if _is_true(row.get("Roster_Watch")) or str(row.get("Model_Stats_Season") or "").casefold() == "unavailable":
        return "Odds only · history missing"
    usage = _finite(row.get("PP_TOI_per_game")) is not None
    pk = _finite(row.get("Opp_PK_xGA60")) is not None
    if usage and pk:
        return "Usage + PK context"
    if usage:
        return "PK context missing"
    if pk:
        return "PP usage missing"
    return "Usage + PK missing"


def priced_ppp_quotes(frame: pd.DataFrame, *, line_filter: float | None = None) -> pd.DataFrame:
    """One best observed quote per player and PPP line, including alternates."""
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)
    quotes = {}
    for row in frame.to_dict("records"):
        player = str(row.get("Player") or "").strip()
        team = str(row.get("Team") or "").strip()
        if not player:
            continue
        for suffix in ("", "_1", "_2", "_3", "_4"):
            line = _finite(row.get(f"BDL_PPP_Line{suffix}"))
            odds = _finite(row.get(f"BDL_PPP_Odds{suffix}"))
            if line is None or odds is None or odds == 0:
                continue
            if line_filter is not None and not math.isclose(line, line_filter):
                continue
            book = str(row.get(f"BDL_PPP_Book{suffix}") or "").strip()
            key = (team.casefold(), player.casefold(), line)
            candidate = {
                "Game": str(row.get("Game") or ""), "Player": player,
                "Team": team, "Opponent": str(row.get("Opp") or ""),
                "PPP line": line, "Over odds": int(odds), "Book": book,
                "Book break-even %": _break_even_pct(odds),
                "PP unit": _unit_label(row),
                "PP TOI/game": _finite(row.get("PP_TOI_per_game")),
                "PPP last 10": _finite(row.get("PPP10_total")),
                "Opp PK xGA/60": _finite(row.get("Opp_PK_xGA60")),
                "PP matchup /100": _finite(row.get("PP_Matchup"))
                if _finite(row.get("Opp_PK_xGA60")) is not None
                and _finite(row.get("Team_PP_xGF60")) is not None else None,
                "Stats season": str(row.get("Model_Stats_Season") or ""),
                "Context": _context_label(row),
            }
            if key not in quotes or odds > quotes[key]["Over odds"]:
                quotes[key] = candidate
    result = pd.DataFrame(quotes.values(), columns=COLUMNS)
    if not result.empty:
        result = result.sort_values(["Game", "Player", "PPP line"],
                                    kind="stable").reset_index(drop=True)
    return result
