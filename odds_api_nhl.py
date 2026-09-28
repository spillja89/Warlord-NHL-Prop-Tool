"""Bring NHL player prop prices from The Odds API into the tracker odds schema.

The existing app reads BDL_* columns.  Keep those columns as the compatibility
surface, while recording the actual sportsbook and provider for every quote.
Only observed Over/Yes prices are used; no line or price is fabricated.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from zoneinfo import ZoneInfo
import json
from typing import Any

import pandas as pd
import requests

from odds_ev_bdl import _norm_name, _safe_float


BASE = "https://api.the-odds-api.com/v4/sports/icehockey_nhl"
MARKETS = (
    "player_points", "player_assists", "player_shots_on_goal", "player_goals",
    "player_goal_scorer_anytime", "player_points_alternate",
    "player_assists_alternate", "player_shots_on_goal_alternate",
)
MARKET_NAMES = {
    "player_points": "Points", "player_points_alternate": "Points",
    "player_assists": "Assists", "player_assists_alternate": "Assists",
    "player_shots_on_goal": "SOG", "player_shots_on_goal_alternate": "SOG",
    "player_goals": "Goal", "player_goal_scorer_anytime": "ATG",
}
TEAM_NAMES = {
    "anaheim ducks": "ANA", "boston bruins": "BOS", "buffalo sabres": "BUF",
    "calgary flames": "CGY", "carolina hurricanes": "CAR", "chicago blackhawks": "CHI",
    "colorado avalanche": "COL", "columbus blue jackets": "CBJ", "dallas stars": "DAL",
    "detroit red wings": "DET", "edmonton oilers": "EDM", "florida panthers": "FLA",
    "los angeles kings": "LAK", "minnesota wild": "MIN", "montreal canadiens": "MTL",
    "nashville predators": "NSH", "new jersey devils": "NJD", "new york islanders": "NYI",
    "new york rangers": "NYR", "ottawa senators": "OTT", "philadelphia flyers": "PHI",
    "pittsburgh penguins": "PIT", "san jose sharks": "SJS", "seattle kraken": "SEA",
    "st louis blues": "STL", "st. louis blues": "STL", "tampa bay lightning": "TBL",
    "toronto maple leafs": "TOR", "utah mammoth": "UTA", "utah hockey club": "UTA",
    "vancouver canucks": "VAN", "vegas golden knights": "VGK",
    "washington capitals": "WSH", "winnipeg jets": "WPG",
}


def _get(path: str, key: str, **params: Any) -> Any:
    try:
        response = requests.get(
            f"{BASE}/{path}",
            params={"apiKey": key, **params},
            headers={"Accept": "application/json"},
            timeout=35,
        )
    except requests.RequestException as exc:
        raise RuntimeError("The Odds API could not be reached; retry the slate run.") from None
    # Avoid exposing a secret embedded in the request URL in exceptions or logs.
    if response.status_code != 200:
        try:
            message = str(response.json().get("message") or "")[:180]
        except (ValueError, AttributeError):
            message = ""
        raise RuntimeError(f"The Odds API HTTP {response.status_code}: {message}")
    try:
        return response.json()
    except ValueError:
        raise RuntimeError("The Odds API returned invalid data.") from None


def _event_on_date(event: dict, day: date) -> bool:
    try:
        instant = datetime.fromisoformat(str(event["commence_time"]).replace("Z", "+00:00"))
        return instant.astimezone(ZoneInfo("America/Chicago")).date() == day
    except (ValueError, KeyError, TypeError):
        return False


def _market_quote(market_key: str, outcome: dict) -> tuple[str, str, float, float] | None:
    market = MARKET_NAMES.get(market_key)
    if not market:
        return None
    name = str(outcome.get("name") or "").strip()
    if market == "ATG":
        if name.casefold() == "no":
            return None
        player = str(outcome.get("description") or name).strip()
        line = 0.5
    else:
        if name.casefold() != "over":
            return None
        player = str(outcome.get("description") or "").strip()
        line = _safe_float(outcome.get("point"))
    price = _safe_float(outcome.get("price"))
    if not player or line is None or price is None or price == 0:
        return None
    if abs(line * 2 - round(line * 2)) > 0.001 or round(line * 2) % 2 != 1:
        return None
    if market in {"Goal", "ATG"} and line != 0.5:
        return None
    if market == "SOG" and line < 1.5:
        return None
    return market, _norm_name(player), float(line), float(price)


def merge_odds_api_props(
    tracker: pd.DataFrame, game_date: date, api_key: str, debug: bool = False,
) -> pd.DataFrame:
    """Select the best observed Over price per player/market/line across books.

    Provider requests are limited to games on the Chicago slate date.  Prices
    merge with any BallDontLie prices already present; the better price wins.
    """
    if tracker.empty or not api_key.strip():
        return tracker
    events = _get("events", api_key)
    if not isinstance(events, list):
        raise RuntimeError("The Odds API returned an invalid events list")
    slate_events = [e for e in events if _event_on_date(e, game_date)]
    if debug:
        print(f"[odds-api] {len(slate_events)} events on {game_date}")
    if not slate_events:
        return tracker

    quotes: dict[tuple[str, str, float], list[tuple[float, str, frozenset[str]]]] = defaultdict(list)
    for event in slate_events:
        teams = frozenset(filter(None, (
            TEAM_NAMES.get(str(event.get("home_team") or "").strip().casefold()),
            TEAM_NAMES.get(str(event.get("away_team") or "").strip().casefold()),
        )))
        if len(teams) != 2:
            if debug:
                print("[odds-api] skipped event with unknown team mapping")
            continue
        payload = _get(
            f"events/{event['id']}/odds", api_key, regions="us,us2",
            markets=",".join(MARKETS), oddsFormat="american",
        )
        for book in payload.get("bookmakers", []):
            book_name = str(book.get("title") or book.get("key") or "").strip()
            if not book_name:
                continue
            for market in book.get("markets", []):
                market_key = str(market.get("key") or "")
                for outcome in market.get("outcomes", []):
                    parsed = _market_quote(market_key, outcome)
                    if parsed:
                        mkt, player, line, price = parsed
                        quotes[(player, mkt, line)].append((price, book_name, teams))
                        if mkt == "ATG":
                            # Most books call the ordinary 0.5 Goals line "anytime scorer".
                            quotes[(player, "Goal", line)].append((price, book_name, teams))
    if not quotes:
        return tracker

    df = tracker.copy()
    for idx, row in df.iterrows():
        player = _norm_name(row.get("Player"))
        team = str(row.get("Team") or "").strip().upper()
        opp = str(row.get("Opp") or "").strip().upper()
        if not player or not team or not opp:
            continue
        for market in {"Points", "Assists", "SOG", "Goal", "ATG"}:
            by_line: dict[float, tuple[float, str]] = {}
            source_by_line: dict[float, str] = {}
            for (p, m, line), available in quotes.items():
                if p != player or m != market:
                    continue
                matching = [(price, book) for price, book, teams in available if {team, opp} == teams]
                if matching:
                    by_line[line] = max(matching)
                    source_by_line[line] = "The Odds API"
            # Retain existing BallDontLie prices when they are better.
            for n in range(1, 5):
                line = _safe_float(row.get(f"BDL_{market}_Line_{n}"))
                price = _safe_float(row.get(f"BDL_{market}_Odds_{n}"))
                if line is None or price is None:
                    continue
                book = str(row.get(f"BDL_{market}_Book_{n}") or "").strip()
                if line not in by_line or price > by_line[line][0]:
                    by_line[line] = (price, book)
                    source_by_line[line] = "BallDontLie"
            if not by_line:
                continue
            lines = sorted(by_line)
            for n, line in enumerate(lines[:4], 1):
                price, book = by_line[line]
                df.at[idx, f"BDL_{market}_Line_{n}"] = line
                df.at[idx, f"BDL_{market}_Odds_{n}"] = price
                df.at[idx, f"BDL_{market}_Book_{n}"] = book
            # Clear stale extra rungs when an alternate list has changed.
            for n in range(len(lines[:4]) + 1, 5):
                df.at[idx, f"BDL_{market}_Line_{n}"] = None
                df.at[idx, f"BDL_{market}_Odds_{n}"] = None
                df.at[idx, f"BDL_{market}_Book_{n}"] = ""
            old_main = _safe_float(row.get(f"BDL_{market}_Line"))
            chosen = old_main if old_main in by_line else min(lines)
            price, book = by_line[chosen]
            df.at[idx, f"BDL_{market}_Line"] = chosen
            df.at[idx, f"BDL_{market}_Odds"] = price
            df.at[idx, f"BDL_{market}_Book"] = book
            df.at[idx, f"{market}_Odds_Source"] = source_by_line[chosen]
            known_books = {
                book for line in lines for _, book, teams in quotes.get((player, market, line), [])
                if {team, opp} == teams
            }
            known_books.update(str(row.get(f"BDL_{market}_Book_{n}") or "").strip() for n in range(1, 5))
            df.at[idx, f"{market}_Available_Books"] = json.dumps(sorted(known_books - {"", "nan", "None"}))
    return df
