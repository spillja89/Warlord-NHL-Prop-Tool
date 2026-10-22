"""Pregame current-season goal form for the 2026-27 Goal move tree."""

from __future__ import annotations

from collections import defaultdict
from io import BytesIO
import json
from zipfile import ZipFile

import pandas as pd


def recent_goals_from_log(raw, slate_date: str):
    try:
        games = json.loads(raw).get("games", [])
        prior = sorted((game for game in games if game.get("d", "") < slate_date),
                       key=lambda game: game["d"])
        return sum(float(game.get("g") or 0) for game in prior[-5:]) / min(5, len(prior)) if prior else None
    except (TypeError, ValueError, AttributeError):
        return None


def recent_goal_form_by_player(session, season_start: int, slate_date: str) -> dict[int, dict[str, float]]:
    """Use only completed regular-season MoneyPuck games before the slate date.

    Missing or invalid source data stays unavailable; no season-average proxy is
    substituted for the five-game xG measure used in the graded move research.
    """
    url = ("https://peter-tanner.com/moneypuck/downloads/"
           f"seasonPlayersSummary/skaters/{season_start}.zip")
    response = session.get(url, timeout=45)
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        with archive.open(f"{season_start}.csv") as source:
            frame = pd.read_csv(source,
                                usecols=["playerId", "gameDate", "situation", "I_F_goals", "I_F_xGoals"])
    frame = frame.loc[frame["situation"].eq("all")].copy()
    frame["gameDate"] = frame["gameDate"].astype(str)
    frame = frame.loc[frame["gameDate"].lt(slate_date.replace("-", ""))]
    for field in ("I_F_goals", "I_F_xGoals"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame = frame.dropna(subset=["playerId", "I_F_goals", "I_F_xGoals"])
    grouped = defaultdict(list)
    for player_id, game_date, goals, xg in frame[["playerId", "gameDate", "I_F_goals", "I_F_xGoals"]].itertuples(index=False, name=None):
        grouped[int(player_id)].append((game_date, float(goals), float(xg)))
    return {player_id: {
                "G": round(sum(game[1] for game in sorted(games)[-5:]) / min(5, len(games)), 4),
                "xG": round(sum(game[2] for game in sorted(games)[-5:]) / min(5, len(games)), 4),
            } for player_id, games in grouped.items()}
