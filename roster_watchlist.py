"""Pregame roster verification and odds-only coverage for missing skaters."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import unicodedata

import pandas as pd


def _name_key(value: str) -> str:
    plain = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(ch for ch in plain.casefold() if ch.isalnum() and not unicodedata.combining(ch))


def _roster_players(session, team: str) -> dict[str, dict]:
    response = session.get(f"https://api-web.nhle.com/v1/roster/{team}/current", timeout=15)
    response.raise_for_status()
    payload = response.json()
    found = {}
    for group in ("forwards", "defensemen"):
        for player in payload.get(group, []):
            first = player.get("firstName", {}).get("default", "")
            last = player.get("lastName", {}).get("default", "")
            name = f"{first} {last}".strip()
            if name:
                found[_name_key(name)] = {"name": name, "id": player.get("id"),
                                          "position": player.get("positionCode", "F")}
    if not found:
        raise ValueError(f"Official NHL roster is empty for {team}")
    return found


def load_active_rosters(session, teams: set[str], *, debug: bool = False) -> dict[str, dict[str, dict]]:
    """Fetch today's official skater rosters once for modeling and verification."""
    rosters = {}
    for team in sorted(teams):
        try:
            rosters[team] = _roster_players(session, team)
        except Exception as exc:
            if debug:
                print(f"[roster] {team}: {exc}")
    return rosters


def current_roster_teams_by_id(rosters: dict[str, dict[str, dict]]) -> dict[int, str]:
    """Match a skater's prior stats to his current club by stable NHL player ID."""
    clubs: dict[int, set[str]] = {}
    for team, players in rosters.items():
        for player in players.values():
            try:
                clubs.setdefault(int(player["id"]), set()).add(team)
            except (TypeError, ValueError, KeyError):
                continue
    return {player_id: next(iter(teams)) for player_id, teams in clubs.items() if len(teams) == 1}


def _best_price(props: list[dict], kind: str, wanted: tuple[float, ...]) -> tuple[float, int, str] | None:
    candidates = []
    for prop in props:
        if prop.get("prop_type") != kind:
            continue
        try:
            line = float(prop.get("line_value"))
        except (TypeError, ValueError):
            continue
        market = prop.get("market") or {}
        over = market.get("over_odds") if market.get("type") == "over_under" else market.get("odds")
        try:
            odds = int(over)
        except (TypeError, ValueError):
            continue
        if market.get("type") == "milestone":
            line -= 0.5  # one-or-more is the same outcome as over 0.5
        if not any(math.isclose(line, target) for target in wanted):
            continue
        candidates.append((wanted.index(next(x for x in wanted if math.isclose(line, x))), -odds,
                           line, odds, str(prop.get("vendor") or "")))
    if not candidates:
        return None
    _, _, line, odds, book = min(candidates)
    return line, odds, book


def add_roster_watchlist(tracker: pd.DataFrame, session, teams: set[str],
                         game_map: dict[str, str], today, api_key: str | None,
                         *, injury_reports: pd.DataFrame | None = None,
                         debug: bool = False,
                         rosters: dict[str, dict[str, dict]] | None = None) -> pd.DataFrame:
    """Verify known rows and append active, priced players missing from model data.

    A missing skater has no model probability, confidence, Green label, or move.
    Official roster membership confirms club status, not tonight's dressed lineup.
    """
    tracker = tracker.copy()
    tracker["Roster_Status"] = "Unverified"
    tracker["Roster_Check_UTC"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tracker["Roster_Watch"] = False
    if rosters is None:
        rosters = load_active_rosters(session, teams, debug=debug)

    if rosters:
        known = tracker["Team"].astype(str).isin(rosters)
        active = pd.Series(True, index=tracker.index)
        for index in tracker.index[known]:
            row = tracker.loc[index]
            team = str(row.get("Team") or "")
            player_id = pd.to_numeric(row.get("Player_ID"), errors="coerce")
            listed = any(info["id"] == int(player_id) for info in rosters[team].values()) if pd.notna(player_id) else False
            if not listed:
                listed = _name_key(row.get("Player")) in rosters[team]
            active.at[index] = listed
            tracker.at[index, "Roster_Status"] = "Active roster" if listed else "Not on active roster"
        removed = int((~active).sum())
        tracker = tracker.loc[active].copy()
        if debug:
            print(f"[roster] verified {len(rosters)}/{len(teams)} teams; removed {removed} non-roster rows")

    if not rosters or not api_key:
        return tracker

    injury_statuses = {}
    if injury_reports is not None and not injury_reports.empty:
        for report in injury_reports.to_dict("records"):
            status = str(report.get("Status") or "").strip().casefold()
            injury_statuses[(str(report.get("Team") or ""), _name_key(report.get("Player")))] = status

    from odds_ev_bdl import fetch_bdl_games_for_date, fetch_bdl_props_for_game, fetch_bdl_players_map

    existing = {(str(row.Team), _name_key(row.Player)) for row in tracker[["Team", "Player"]].itertuples(index=False)}
    rows = []
    try:
        games = fetch_bdl_games_for_date(today.isoformat(), api_key=api_key)
        games += fetch_bdl_games_for_date((today + timedelta(days=1)).isoformat(), api_key=api_key)
    except Exception as exc:
        if debug:
            print(f"[roster] odds-only watchlist unavailable: {exc}")
        return tracker
    seen_games = set()
    for game in games:
        if game.get("id") in seen_games:
            continue
        seen_games.add(game.get("id"))
        game_teams = {str(game.get(side, {}).get("tricode") or "") for side in ("home_team", "away_team")}
        if not game_teams.issubset(teams) or not game_teams.intersection(rosters):
            continue
        try:
            props = fetch_bdl_props_for_game(int(game["id"]), api_key=api_key)
            player_ids = sorted({int(p["player_id"]) for p in props if p.get("player_id") is not None})
            players = fetch_bdl_players_map(player_ids, api_key=api_key)
        except Exception as exc:
            if debug:
                print(f"[roster] game {game.get('id')}: {exc}")
            continue
        grouped: dict[int, list[dict]] = {}
        for prop in props:
            if prop.get("player_id") is not None:
                grouped.setdefault(int(prop["player_id"]), []).append(prop)
        for bdl_id, player_props in grouped.items():
            name = str(players.get(bdl_id, {}).get("full_name") or "")
            key = _name_key(name)
            for team in game_teams.intersection(rosters):
                official = rosters[team].get(key)
                if not official or (team, key) in existing:
                    continue
                injury_status = injury_statuses.get((team, key), "")
                if any(token in injury_status for token in ("out", "scratch", "injured reserve", "ltir", " ir")) or injury_status == "ir":
                    continue
                prices = {
                    "Goal": _best_price(player_props, "anytime_goal", (0.5,)),
                    "Assists": _best_price(player_props, "assists", (0.5, 1.5)),
                    "Points": _best_price(player_props, "points", (0.5, 1.5)),
                    "SOG": _best_price(player_props, "shots_on_goal", (2.5, 3.5, 1.5)),
                    "PPP": _best_price(player_props, "power_play_points", (0.5, 1.5)),
                }
                if not any(prices.values()):
                    continue
                row = {"Date": today.isoformat(), "Game": game_map.get(team, ""),
                       "Player": official["name"], "Player_ID": official["id"],
                       "Team": team, "Pos": official["position"],
                       "Roster_Status": "Active roster", "Roster_Watch": True,
                       "Roster_Check_UTC": tracker["Roster_Check_UTC"].iloc[0] if len(tracker) else "",
                       "Fired_Moves": "[]", "Model_Stats_Season": "Unavailable",
                       "Injury_Status": "GTD" if any(token in injury_status for token in ("gtd", "day", "dtd")) else "Unknown",
                       "Injury_Badge": "🟡 GTD" if any(token in injury_status for token in ("gtd", "day", "dtd")) else "",
                       "Available": True}
                for market, price in prices.items():
                    if price:
                        if market == "PPP":
                            row["BDL_PPP_Line"], row["BDL_PPP_Odds"], row["BDL_PPP_Book"] = price
                        else:
                            row[f"{market}_Line"], row[f"{market}_Odds_Over"], row[f"{market}_Book"] = price
                rows.append(row)
                existing.add((team, key))
                break
    if rows:
        tracker = pd.concat([tracker, pd.DataFrame(rows)], ignore_index=True)
        if debug:
            print(f"[roster] added {len(rows)} active, priced players without model stats")
    return tracker
