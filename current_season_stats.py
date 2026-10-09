"""Observed current-season skater stats from the tracker's NHL game logs."""

import json
from datetime import date


def season_for_day(day):
    """Return the NHL season label for a slate date (for example 2026-27)."""
    parsed = date.fromisoformat(str(day)[:10])
    start = parsed.year if parsed.month >= 7 else parsed.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def _minutes(raw):
    try:
        minutes, seconds = str(raw).split(":", 1)
        return int(minutes) + int(seconds) / 60
    except (TypeError, ValueError):
        return None


def current_season_rows(records, season, before_date):
    """One player row and dated log per skater; no previous-season fill."""
    summary = []
    logs = {}
    seen = set()
    cutoff = str(before_date)[:10]
    for record in records:
        player = str(record.get("Player") or "").strip()
        team = str(record.get("Team") or "").strip()
        player_id = str(record.get("Player_ID") or "").strip()
        key = player_id if player_id and player_id.lower() != "nan" else f"{team}|{player}"
        if not player or key in seen:
            continue
        raw = record.get("Form_Log")
        try:
            form = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            continue
        if not isinstance(form, dict) or form.get("season") != season:
            continue
        games = []
        source_games = form.get("games")
        if not isinstance(source_games, list):
            continue
        for game in source_games:
            if not isinstance(game, dict):
                continue
            played = str(game.get("d") or "")[:10]
            if len(played) != 10 or played >= cutoff:
                continue
            try:
                goals, assists, shots = (int(game[field]) for field in ("g", "a", "s"))
            except (KeyError, TypeError, ValueError):
                continue
            pp_raw = game.get("pp")
            try:
                pp_points = int(pp_raw) if pp_raw not in (None, "") else None
            except (TypeError, ValueError):
                pp_points = None
            games.append({"Date": played, "G": goals, "A": assists,
                          "P": goals + assists, "SOG": shots, "PPP": pp_points,
                          "TOI": _minutes(game.get("toi")),
                          "Home/Away": game.get("h") or ""})
        games.sort(key=lambda game: game["Date"], reverse=True)
        if not games:
            continue
        seen.add(key)
        gp = len(games)
        last_five = games[:5]
        ice_times = [game["TOI"] for game in games if game["TOI"] is not None]
        pp_complete = all(game["PPP"] is not None for game in games)
        goals = sum(game["G"] for game in games)
        assists = sum(game["A"] for game in games)
        shots = sum(game["SOG"] for game in games)
        summary.append({
            "Player": player, "Team": team, "GP": gp,
            "G": goals, "A": assists, "P": goals + assists, "SOG": shots,
            "P/GP": round((goals + assists) / gp, 2),
            "SOG/GP": round(shots / gp, 2),
            "PPP": sum(game["PPP"] for game in games) if pp_complete else None,
            "PP logs": f"{sum(game['PPP'] is not None for game in games)}/{gp}",
            "TOI/GP": round(sum(ice_times) / len(ice_times), 1) if ice_times else None,
            "L5 P": sum(game["P"] for game in last_five),
            "L5 SOG": sum(game["SOG"] for game in last_five),
            "Last game": games[0]["Date"],
            "_key": key,
        })
        logs[key] = games
    return summary, logs
