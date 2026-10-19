"""Conservative game-day lineup checks from the official NHL gamecenter.

The active roster is club membership, not the dressed lineup.  A gamecenter
boxscore is only treated as a lineup when *both* clubs have a full skater list.
Missing or incomplete data stays unconfirmed; it never becomes a scratch.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _ids(players: Any) -> set[int]:
    result: set[int] = set()
    for player in players or []:
        if isinstance(player, dict):
            try:
                result.add(int(player["playerId"]))
            except (KeyError, TypeError, ValueError):
                pass
    return result


def inspect_boxscore_lineup(box: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return team lineup evidence only if both sides have >=18 skaters."""
    by_game = box.get("playerByGameStats")
    if not isinstance(by_game, dict):
        return {}
    sides: dict[str, dict[str, Any]] = {}
    for side in ("awayTeam", "homeTeam"):
        team = str((box.get(side) or {}).get("abbrev") or "").upper().strip()
        groups = by_game.get(side) or {}
        if not team or not isinstance(groups, dict):
            return {}
        skaters = _ids(groups.get("forwards")) | _ids(groups.get("defense")) | _ids(groups.get("defencemen"))
        if len(skaters) < 18:
            return {}
        starters = [p for p in groups.get("goalies") or []
                    if isinstance(p, dict) and p.get("starter") is True]
        starter = starters[0] if len(starters) == 1 else None
        sides[team] = {"skater_ids": skaters,
                       "starter_id": int(starter["playerId"]) if starter and starter.get("playerId") else None,
                       "starter_name": str(starter.get("name") or "") if starter else ""}
    return sides


def fetch_official_lineups(session: Any, games: list[dict[str, Any]], *,
                           now: datetime | None = None, window_minutes: int = 120,
                           timeout: int = 8) -> tuple[dict[str, dict[str, Any]], str]:
    """Check games near puck drop; never infer a scratch from an incomplete feed."""
    now = now or datetime.now(timezone.utc)
    checked = now.isoformat(timespec="seconds")
    result: dict[str, dict[str, Any]] = {}
    for game in games:
        teams = [str(game.get(k) or "").upper().strip() for k in ("away", "home")]
        for team in teams:
            if team:
                result[team] = {"status": "Lineup pending", "skater_ids": set(),
                                "starter_id": None, "starter_name": ""}
        try:
            start = datetime.fromisoformat(str(game.get("startTimeUTC") or "").replace("Z", "+00:00"))
            game_id = int(game.get("id") or game.get("gameId"))
        except (TypeError, ValueError):
            continue
        if (start - now).total_seconds() > window_minutes * 60:
            continue
        try:
            response = session.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore", timeout=timeout)
            response.raise_for_status()
            lineups = inspect_boxscore_lineup(response.json())
        except Exception:
            continue
        for team in teams:
            if team in lineups:
                result[team] = {"status": "Official lineup", **lineups[team]}
    return result, checked


def player_lineup_status(player_id: Any, team: str,
                         lineups: dict[str, dict[str, Any]]) -> str:
    evidence = lineups.get(str(team or "").upper().strip()) or {}
    if evidence.get("status") != "Official lineup":
        return "Lineup pending"
    try:
        return "Dressed" if int(player_id) in evidence["skater_ids"] else "Not dressed"
    except (TypeError, ValueError, KeyError):
        return "Lineup pending"

