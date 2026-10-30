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
           "PPP drought (games)",
           "PP on-ice xG/game", "PP shooter xG/game", "Equal-weight PP context",
           "PP research signals", "PP research count", "PP research eligible",
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


def pp_player_research_signals(row):
    """Historical PPP research profile; never a probability or board gate."""
    on_ice_xg = _finite(row.get("PP_OnIce_xG_per_game"))
    if on_ice_xg is None:
        return "Data missing", -1, False
    if on_ice_xg < 0.55:
        return "Below 0.55 base", -1, False
    inputs = (
        (_finite(row.get("PP_Observed_Assists_per_game")), 0.15, ">="),
        (_finite(row.get("PP_ActualPoints60")), 4.5, ">="),
        (_finite(row.get("PPP10_total")), 4, ">="),
        (_finite(row.get("Drought_PPP")), 1, "<="),
    )
    if any(value is None for value, _, _ in inputs):
        return "Data missing", -1, True
    count = sum(value >= cutoff if direction == ">=" else value <= cutoff
                for value, cutoff, direction in inputs)
    return f"{count}/4", count, True


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


def team_pp_goal_research_label(row) -> str:
    """Descriptive team PP-goal research tags; never a player PPP forecast."""
    pk_minutes = _finite(row.get("Opp_PK_Time_PG"))
    if pk_minutes is None:
        return "PK time unavailable"
    if pk_minutes > 4.5:
        return "—"
    parts = ["⚡ PK ≤4:30"]
    team_ppo = _finite(row.get("Team_PPO_PG"))
    opp_pko = _finite(row.get("Opp_PKO_PG"))
    opp_pk_ga = _finite(row.get("Opp_PK_GA_PG"))
    if team_ppo is not None and team_ppo >= 3.25:
        parts.append("Team PPO ≥3.25")
    if opp_pko is not None and opp_pko >= 2.75:
        parts.append("Opp PKO ≥2.75")
        if opp_pk_ga is not None and opp_pk_ga >= 0.9:
            parts.append("PK GA ≥0.9")
    return " · ".join(parts)


def opponent_pk_seconds_per_opportunity(row):
    """Observed opponent PK seconds divided by times shorthanded, pregame."""
    pk_minutes = _finite(row.get("Opp_PK_Time_PG"))
    opportunities = _finite(row.get("Opp_PKO_PG"))
    if pk_minutes is None or opportunities is None or pk_minutes < 0 or opportunities <= 0:
        return None
    return round(60 * pk_minutes / opportunities, 1)


def format_pk_minutes_per_game(value):
    """Show a pregame average in clock notation, such as 4.5 minutes as 4:30."""
    minutes = _finite(value)
    if minutes is None or minutes < 0:
        return "—"
    seconds = round(minutes * 60)
    return f"{seconds // 60}:{seconds % 60:02d}"


def team_pp_goal_research_tier(row) -> str:
    """Visual team PP-goal research tier; does not affect player rankings."""
    pk_minutes = _finite(row.get("Opp_PK_Time_PG"))
    if pk_minutes is None or pk_minutes > 4.5:
        return ""
    opportunities = _finite(row.get("Opp_PKO_PG"))
    seconds = (60 * pk_minutes / opportunities
               if opportunities is not None and opportunities > 0 else None)
    if seconds is not None and seconds <= 85:
        return "watch"
    if seconds is not None and seconds <= 90:
        return "buff"
    return "base"


def team_pp_goal_research_flag(row) -> str:
    """Compact badge for the team-scoring matchup on player research rows."""
    return {"watch": "💥 PK BUFF · ≤85s WATCH",
            "buff": "✨ PK BUFF · ≤90s",
            "base": "⚡ PK MATCHUP · ≤4:30"}.get(team_pp_goal_research_tier(row), "—")


def team_pp_goal_research_signals(row) -> list[str]:
    """All matching team PP-goal research signals, frozen through Oct 29, 2026.

    Records count team games with at least one PP goal. They are not player
    power-play-point rates or probabilities for the next game.
    """
    pk_minutes = _finite(row.get("Opp_PK_Time_PG"))
    if pk_minutes is None or pk_minutes > 4.5:
        return []
    signals = ["⚡ PK ≤4:30 · 47/66 (71.2%)"]
    pk_seconds = opponent_pk_seconds_per_opportunity(row)
    if pk_seconds is not None and pk_seconds <= 90:
        signals.append("✨ PK ≤90s/opportunity · 17/21 (81.0%)")
    if pk_seconds is not None and pk_seconds <= 85:
        signals.append("💥 PK ≤85s/opportunity · 13/15 (86.7%)")
    team_ppo = _finite(row.get("Team_PPO_PG"))
    opp_pko = _finite(row.get("Opp_PKO_PG"))
    opp_pk_ga = _finite(row.get("Opp_PK_GA_PG"))
    if team_ppo is not None and team_ppo >= 3.25:
        signals.append("Team PPO ≥3.25/game · 28/36 (77.8%)")
    if opp_pko is not None and opp_pko >= 2.75:
        signals.append("Opp PKO ≥2.75/game · 20/25 (80.0%)")
        if opp_pk_ga is not None and opp_pk_ga >= 0.9:
            signals.append("Opp PK GA ≥0.9/game · 12/14 (85.7%)")
    return signals



def team_pp_goal_research_compact(row) -> str:
    """Short, horizontal summary for the priced-player table."""
    labels = team_pp_goal_research_signals(row)
    short = []
    for label in labels:
        if label.startswith("⚡"):
            short.append("⚡71%")
        elif label.startswith("✨"):
            short.append("✨81%")
        elif label.startswith("💥"):
            short.append("💥87%")
        elif label.startswith("Team PPO"):
            short.append("PPO78%")
        elif label.startswith("Opp PKO"):
            short.append("PKO80%")
        elif label.startswith("Opp PK GA"):
            short.append("GA86%")
    return " + ".join(short) if short else "—"



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
            research_label, research_count, research_eligible = pp_player_research_signals(row)
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
                "PPP drought (games)": _finite(row.get("Drought_PPP")),
                "Opp PK xGA/60": _finite(row.get("Opp_PK_xGA60")),
                "PP matchup /100": _finite(row.get("PP_Matchup"))
                if _finite(row.get("Opp_PK_xGA60")) is not None
                and _finite(row.get("Team_PP_xGF60")) is not None else None,
                "PP on-ice xG/game": _finite(row.get("PP_OnIce_xG_per_game")),
                "PP shooter xG/game": _finite(row.get("PP_iXG_per_game")),
                "Equal-weight PP context": _finite(row.get("PP_Matchup_Equal")),
                "PP research signals": research_label,
                "PP research count": research_count,
                "PP research eligible": research_eligible,
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
