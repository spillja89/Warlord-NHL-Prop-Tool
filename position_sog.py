"""Observed shots allowed by position from Shot Propz's public NHL table.

This is a scouting display only. It never changes the tracker, move rules, or
rankings. Shot Propz assigns some forwards by their game-day line position, so
these numbers should not be mixed with NHL roster-position aggregates.
"""

from datetime import date
import re

from bs4 import BeautifulSoup


SOURCE_URL = "https://shotpropz.com/nhl/sog-against-by-position/"
POSITIONS = {"C": "C", "L": "LW", "LW": "LW", "R": "RW", "RW": "RW", "D": "D", "LD": "D", "RD": "D"}


def parse_position_sog(html: str, season: str) -> dict:
    """Read only the four all-games, year-to-date SOG tables."""
    soup = BeautifulSoup(html, "html.parser")
    season_value = f"{season[2:4]}_{season[-2:]}"  # 2026-27 -> 26_27
    selected = soup.select_one('select[data-spz-f="season"] option[selected]')
    if selected is None or selected.get("value") != season_value:
        raise ValueError("Shot Propz returned a different NHL season")
    span = soup.select_one('select[data-spz-f="span"] option[selected]')
    if span is None or span.get("value") != "YTD":
        raise ValueError("Shot Propz returned a different date span")

    by_position = {}
    updated = None
    for code, position in (("C", "C"), ("L", "LW"), ("R", "RW"), ("D", "D")):
        table = soup.select_one(f"table.sob-{code}")
        if table is None:
            raise ValueError(f"Shot Propz {position} table is missing")
        rows = {}
        for tr in table.select("tbody tr[data-team]"):
            team = str(tr.get("data-team") or "").strip().upper()
            try:
                rank = int(tr.select_one("td.col-rk").get("data-sort-value"))
                gp = int(tr.select_one("td.col-gp").get("data-sort-value"))
                shots = float(tr.select_one("td.value").get("data-sort-value"))
            except (AttributeError, TypeError, ValueError):
                continue
            if re.fullmatch(r"[A-Z]{3}", team) and gp > 0 and 0 <= shots <= 30:
                rows[team] = {"rank": rank, "gp": gp, "shots_per_game": shots}
        if len(rows) < 16:
            raise ValueError(f"Shot Propz {position} table is incomplete")
        by_position[position] = rows
        if updated is None:
            text = table.parent.get_text(" ", strip=True)
            match = re.search(r"Last updated:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})", text)
            if match:
                updated = match.group(1)
    if updated is None:
        raise ValueError("Shot Propz update date is missing")
    return {"season": season, "updated": updated, "positions": by_position}


def fetch_position_sog(season: str, session=None) -> dict:
    """One small public-page request; caller caches and handles outages."""
    import requests

    http = session or requests.Session()
    response = http.get(SOURCE_URL, params={"season": f"{season[2:4]}_{season[-2:]}", "span": "YTD"}, timeout=20)
    response.raise_for_status()
    return parse_position_sog(response.text, season)


def position_matchup(team: str, game: str, position: str, data: dict | None) -> dict | None:
    """Match a skater to the other team in his listed game, then his position."""
    if not data:
        return None
    pos = POSITIONS.get(str(position or "").strip().upper())
    teams = str(game or "").strip().upper().split("@")
    team = str(team or "").strip().upper()
    if pos is None or len(teams) != 2 or team not in teams:
        return None
    opponent = teams[1] if team == teams[0] else teams[0]
    row = data.get("positions", {}).get(pos, {}).get(opponent)
    return {"opponent": opponent, "position": pos, **row} if row else None


def current_slate_only(slate_day: str, today: date) -> bool:
    """Never attach today's observations to a past saved slate."""
    return str(slate_day)[:10] == today.isoformat()
