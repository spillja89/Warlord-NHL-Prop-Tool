"""Pregame, line-specific player form from one NHL regular season."""

import json
from statistics import mean, median


def compact_regular_log(payload, before_date=None):
    """Keep only pre-slate regular-season skater games, newest first."""
    if not isinstance(payload, dict):
        return None
    season = str(payload.get("seasonId") or "")
    if str(payload.get("gameTypeId")) != "2" or len(season) != 8:
        return None
    games = []
    for row in payload.get("gameLog", []):
        if not isinstance(row, dict):
            continue
        game_id = str(row.get("gameId") or "")
        if len(game_id) < 6 or game_id[4:6] != "02":
            continue
        try:
            goals = int(row["goals"])
            assists = int(row["assists"])
            shots = int(row["shots"])
            pp_points = int(row.get("powerPlayPoints") or 0)
        except (KeyError, ValueError, TypeError):
            continue
        date = str(row.get("gameDate") or "")[:10]
        if len(date) != 10 or (before_date and date >= str(before_date)[:10]):
            continue
        games.append({
            "d": date, "g": goals, "a": assists, "s": shots,
            "pp": pp_points, "h": str(row.get("homeRoadFlag") or ""),
            "toi": str(row.get("toi") or ""),
        })
    games.sort(key=lambda game: game["d"], reverse=True)
    return {"season": f"{season[:4]}-{season[6:]}", "games": games}


def summarize_form(raw, market, line):
    """Compare observed game values with today's line, without mixing seasons."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        games = data["games"]
        line = float(line)
        key = {"Goal": "g", "Assists": "a", "Points": "p", "SOG": "s"}[market]
    except (TypeError, ValueError, KeyError):
        return None
    if not isinstance(games, list):
        return None
    if not games:
        return {"season": str(data.get("season") or "Unknown"), "empty": True}
    values = []
    for game in games:
        try:
            value = int(game["g"]) + int(game["a"]) if key == "p" else int(game[key])
            values.append((value, game))
        except (TypeError, ValueError, KeyError):
            continue
    if not values:
        return {"season": str(data.get("season") or "Unknown"), "empty": True}
    recent = values[:10]
    first_five = values[:5]
    wins = lambda sample: sum(value > line for value, _ in sample)
    shots = [int(game["s"]) for _, game in recent if str(game.get("s", "")).isdigit()]
    toi = []
    for _, game in recent:
        try:
            minutes, seconds = str(game["toi"]).split(":", 1)
            toi.append(int(minutes) + int(seconds) / 60)
        except (KeyError, TypeError, ValueError):
            pass
    splits = {}
    for flag, label in (("H", "Home"), ("R", "Away")):
        subset = [(value, game) for value, game in values if game.get("h") == flag]
        if subset:
            splits[label] = (wins(subset), len(subset))
    return {
        "season": str(data.get("season") or "Unknown"),
        "recent": [(value, value > line) for value, _ in recent],
        "l10": (wins(recent), len(recent)), "l5": (wins(first_five), len(first_five)),
        "season_rate": (wins(values), len(values)),
        "average": mean(value for value, _ in recent),
        "median": median(value for value, _ in recent),
        "shots": mean(shots) if shots else None,
        "toi": mean(toi) if toi else None,
        "splits": splits,
    }
