"""Read real power play point prices from a generated tracker."""

from __future__ import annotations

import json
import math

import pandas as pd
import requests


COLUMNS = ("Game", "Player", "Team", "Opponent", "PPP line", "Over odds",
           "Book", "Book break-even %", "PP unit", "PP TOI/game",
           "Team PP time share %",
           "5v5 OZ starts %", "5v5 OZ starts", "5v5 DZ starts",
           "Team EV OZ time %", "Team PP OZ time %",
           "Opp PK xGA/60", "PP matchup /100", "PPP last 10",
           "PP on-ice xG/game", "PP shooter xG/game", "Equal-weight PP context",
           "Tonight PPP mean", "Tonight PPP 1+ %", "Tonight PPP status",
           "Stats season", "Context", "Book prices")


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


def team_pp_time_share_pct(player_pp_minutes, team_pp_minutes):
    """Player PP minutes/game divided by team PP minutes/game, as a percentage."""
    player = _finite(player_pp_minutes)
    team = _finite(team_pp_minutes)
    if player is None or team is None or player < 0 or team <= 0:
        return None
    share = 100 * player / team
    # Different game denominators can make a season-average ratio exceed 100%.
    # Show unavailable instead of presenting an impossible share.
    return round(share, 1) if share <= 100.5 else None


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
    return "PP1" if role >= 2 else ("PP2" if role >= 1 else "No PP unit recorded")


def format_ppp_book_prices(raw: str) -> str:
    """Show only captured prices for the same over 0.5 PPP line."""
    try:
        quotes = json.loads(raw)
    except (TypeError, ValueError):
        return ""
    if not isinstance(quotes, list):
        return ""
    parts = []
    for quote in quotes:
        if not isinstance(quote, dict):
            continue
        book = str(quote.get("book") or "").strip()
        odds = _finite(quote.get("odds"))
        if book and odds is not None:
            parts.append(f"{book} {int(odds):+d}")
    return " · ".join(parts)


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
                "Team PP time share %": team_pp_time_share_pct(
                    row.get("PP_TOI_per_game"), row.get("Team_PP_Time_PG")),
                "5v5 OZ starts %": _finite(row.get("EV5_OZS_Pct")),
                "5v5 OZ starts": _finite(row.get("EV5_OZ_Starts")),
                "5v5 DZ starts": _finite(row.get("EV5_DZ_Starts")),
                "Team EV OZ time %": _finite(row.get("Team_EV_OZ_Time_Pct")),
                "Team PP OZ time %": _finite(row.get("Team_PP_OZ_Time_Pct")),
                "PPP last 10": _finite(row.get("PPP10_total")),
                "Opp PK xGA/60": _finite(row.get("Opp_PK_xGA60")),
                "PP matchup /100": _finite(row.get("PP_Matchup"))
                if _finite(row.get("Opp_PK_xGA60")) is not None
                and _finite(row.get("Team_PP_xGF60")) is not None else None,
                "PP on-ice xG/game": _finite(row.get("PP_OnIce_xG_per_game")),
                "PP shooter xG/game": _finite(row.get("PP_iXG_per_game")),
                "Equal-weight PP context": _finite(row.get("PP_Matchup_Equal")),
                "Tonight PPP mean": _finite(row.get("Tonight_PPP_mu")) if
                row.get("Tonight_PPP_Version") == "2026-10-19-research-v3" else None,
                "Tonight PPP 1+ %": _finite(row.get("Tonight_PPP_1plus_pct")) if
                row.get("Tonight_PPP_Version") == "2026-10-19-research-v3" else None,
                "Tonight PPP status": str(row.get("Tonight_PPP_Status") or ""),
                "Stats season": str(row.get("Model_Stats_Season") or ""),
                "Context": _context_label(row),
                "Book prices": str(row.get("PPP_Book_Prices_05") or "") if line == 0.5 else "",
            }
            if key not in quotes or odds > quotes[key]["Over odds"]:
                quotes[key] = candidate
    result = pd.DataFrame(quotes.values(), columns=COLUMNS)
    if not result.empty:
        result = result.sort_values(["Game", "Player", "PPP line"],
                                    kind="stable").reset_index(drop=True)
    return result
