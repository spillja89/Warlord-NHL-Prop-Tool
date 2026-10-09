"""Live, observed NHL and MoneyPuck regular-season skater metrics."""

from datetime import datetime, timezone
from io import BytesIO

import pandas as pd
import requests


def fetch_nhl_skaters(season_id, session=None):
    """Fetch every skater, not only players priced on today's slate."""
    http = session or requests.Session()
    url = "https://api.nhle.com/stats/rest/en/skater/summary"
    rows = []
    total = None
    while total is None or len(rows) < total:
        response = http.get(url, params={
            "isAggregate": "false", "isGame": "false",
            "cayenneExp": f"gameTypeId=2 and seasonId={int(season_id)}",
            "limit": 100, "start": len(rows),
        }, timeout=25)
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("data") or []
        if not batch:
            break
        rows.extend(batch)
        total = int(payload.get("total") or len(rows))
    return rows, datetime.now(timezone.utc).isoformat(timespec="minutes")


def fetch_moneypuck_skaters(start_year, session=None):
    """Fetch current-season MoneyPuck; never fall back to the model's prior year."""
    http = session or requests.Session()
    url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{int(start_year)}/regular/skaters.csv"
    response = http.get(url, timeout=35)
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content)), response.headers.get("Last-Modified", "Unknown")


def fetch_moneypuck_teams(start_year, session=None):
    """Fetch the current regular-season team file used for observed matchup form."""
    http = session or requests.Session()
    url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{int(start_year)}/regular/teams.csv"
    response = http.get(url, timeout=35)
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content)), response.headers.get("Last-Modified", "Unknown")


def team_season_rows(money_puck):
    """Team scoring/defense rates by situation; iceTime is seconds."""
    needed = {"team", "situation", "games_played", "iceTime", "goalsFor",
              "goalsAgainst", "xGoalsFor", "xGoalsAgainst", "shotsOnGoalAgainst"}
    if not needed.issubset(money_puck.columns):
        raise ValueError(f"MoneyPuck team file missing {sorted(needed - set(money_puck.columns))}")
    by_team = {}
    for _, item in money_puck.iterrows():
        team, situation = str(item["team"]), str(item["situation"])
        if situation in ("all", "5on5", "5on4", "4on5"):
            by_team.setdefault(team, {})[situation] = item

    def per_game(item, field, games):
        value = _number(item.get(field)) if item is not None else None
        return round(value / games, 2) if value is not None and games else None

    def per_60(item, field):
        if item is None:
            return None
        value, seconds = _number(item.get(field)), _number(item.get("iceTime"))
        return round(value * 3600 / seconds, 2) if value is not None and seconds and seconds > 0 else None

    output = []
    for team, situations in by_team.items():
        all_sits = situations.get("all")
        if all_sits is None:
            continue
        games = _number(all_sits.get("games_played"))
        if not games or games < 1:
            continue
        five, pp, pk = (situations.get(key) for key in ("5on5", "5on4", "4on5"))
        output.append({
            "Team": team, "GP": int(games),
            "GF/GP": per_game(all_sits, "goalsFor", games),
            "GA/GP": per_game(all_sits, "goalsAgainst", games),
            "SOG against/GP": per_game(all_sits, "shotsOnGoalAgainst", games),
            "5v5 xG%": round(_number(five.get("xGoalsPercentage")) * 100, 1)
            if five is not None and _number(five.get("xGoalsPercentage")) is not None else None,
            "5v5 xGA/60": per_60(five, "xGoalsAgainst"),
            "PP xGF/60": per_60(pp, "xGoalsFor"),
            "PP GF/60": per_60(pp, "goalsFor"),
            "PK xGA/60": per_60(pk, "xGoalsAgainst"),
            "PK GA/60": per_60(pk, "goalsAgainst"),
            "PK min": round(_number(pk.get("iceTime")) / 60, 1)
            if pk is not None and _number(pk.get("iceTime")) is not None else None,
        })
    return sorted(output, key=lambda row: row["Team"])


def fetch_nhl_player_log(player_id, season_id, session=None):
    http = session or requests.Session()
    url = f"https://api-web.nhle.com/v1/player/{int(player_id)}/game-log/{int(season_id)}/2"
    response = http.get(url, timeout=20)
    response.raise_for_status()
    return response.json()


def league_season_rows(nhl_rows, money_puck=None):
    """Join independent sources by stable NHL player ID; retain NHL-only rows."""
    metrics = {}
    if money_puck is not None and not money_puck.empty:
        needed = {"playerId", "situation", "icetime", "I_F_xGoals", "I_F_goals", "I_F_shotAttempts"}
        if not needed.issubset(money_puck.columns):
            raise ValueError(f"MoneyPuck missing {sorted(needed - set(money_puck.columns))}")
        for _, item in money_puck.iterrows():
            try:
                pid = int(item["playerId"])
            except (TypeError, ValueError):
                continue
            situation = str(item["situation"])
            if situation in ("all", "5on5", "5on4"):
                metrics.setdefault(pid, {})[situation] = item

    output = []
    for row in nhl_rows:
        try:
            pid = int(row["playerId"])
            gp = int(row["gamesPlayed"])
        except (KeyError, TypeError, ValueError):
            continue
        if gp < 1:
            continue
        g, a = int(row.get("goals") or 0), int(row.get("assists") or 0)
        sog = int(row.get("shots") or 0)
        record = {
            "Player": str(row.get("skaterFullName") or ""),
            "Team": str(row.get("teamAbbrevs") or ""),
            "GP": gp, "G": g, "A": a, "P": g + a, "SOG": sog,
            "PPP": int(row.get("ppPoints") or 0),
            "P/GP": round((g + a) / gp, 2),
            "SOG/GP": round(sog / gp, 2),
            "MP GP": None, "xG": None, "xG/GP": None, "xG/60": None,
            "Goals − xG": None, "Attempts": None, "Attempts/GP": None,
            "Attempts/60": None, "5v5 xG%": None, "PP TOI/GP": None,
            "_id": pid,
        }
        mp = metrics.get(pid, {})
        all_sits = mp.get("all")
        if all_sits is not None:
            mp_gp = _number(all_sits.get("games_played"))
            xg = _number(all_sits.get("I_F_xGoals"))
            mp_goals = _number(all_sits.get("I_F_goals"))
            seconds = _number(all_sits.get("icetime"))
            attempts = _number(all_sits.get("I_F_shotAttempts"))
            record["MP GP"] = int(mp_gp) if mp_gp is not None else None
            record["xG"] = round(xg, 2) if xg is not None else None
            record["Goals − xG"] = round(mp_goals - xg, 2) if mp_goals is not None and xg is not None else None
            record["Attempts"] = int(attempts) if attempts is not None else None
            if mp_gp and mp_gp > 0:
                record["xG/GP"] = round(xg / mp_gp, 2) if xg is not None else None
                record["Attempts/GP"] = round(attempts / mp_gp, 1) if attempts is not None else None
            if seconds and seconds > 0:
                record["xG/60"] = round(xg * 3600 / seconds, 2) if xg is not None else None
                record["Attempts/60"] = round(attempts * 3600 / seconds, 1) if attempts is not None else None
        five = mp.get("5on5")
        if five is not None:
            xg_pct = _number(five.get("onIce_xGoalsPercentage"))
            record["5v5 xG%"] = round(xg_pct * 100, 1) if xg_pct is not None else None
        pp = mp.get("5on4")
        if pp is not None:
            pp_seconds = _number(pp.get("icetime"))
            mp_gp = record["MP GP"]
            record["PP TOI/GP"] = round(pp_seconds / mp_gp / 60, 1) if pp_seconds is not None and mp_gp else None
        output.append(record)
    return output


def _number(value):
    try:
        result = float(value)
        return result if pd.notna(result) else None
    except (TypeError, ValueError):
        return None
