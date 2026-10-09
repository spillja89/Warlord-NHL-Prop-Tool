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
            "xG": None, "xG/60": None, "Goals − xG": None,
            "Attempts/60": None, "5v5 xG%": None, "PP TOI/GP": None,
            "_id": pid,
        }
        mp = metrics.get(pid, {})
        all_sits = mp.get("all")
        if all_sits is not None:
            xg = _number(all_sits.get("I_F_xGoals"))
            seconds = _number(all_sits.get("icetime"))
            attempts = _number(all_sits.get("I_F_shotAttempts"))
            record["xG"] = round(xg, 2) if xg is not None else None
            record["Goals − xG"] = round(g - xg, 2) if xg is not None else None
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
            record["PP TOI/GP"] = round(pp_seconds / gp / 60, 1) if pp_seconds is not None else None
        output.append(record)
    return output


def _number(value):
    try:
        result = float(value)
        return result if pd.notna(result) else None
    except (TypeError, ValueError):
        return None
