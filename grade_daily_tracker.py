"""Grade a frozen NHL slate from final NHL box scores.

Every tracker row receives actual stats when its player played in a final game.
Each available market line receives W/L/P; missing lines remain ungraded.
No model estimate, odds, or historical player stat is ever used as an actual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


VERSION = "2026-09-23-v1"
API = "https://api-web.nhle.com/v1"
MARKETS = {
    "Points": ("Points_Line", "points"),
    "Assists": ("Assists_Line", "assists"),
    "SOG": ("SOG_Line", "sog"),
    "Goal": ("Goal_Line", "goals"),
    "ATG": ("ATG_Line", "goals"),
}
MOVE_MARKETS = {"Points": "Points", "Assists": "Assists", "SOG": "SOG", "Goal": "Goal"}
FINAL_STATES = {"OFF", "FINAL"}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _name(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("default", "")
    plain = unicodedata.normalize("NFKD", str(value or ""))
    plain = "".join(char for char in plain if not unicodedata.combining(char))
    plain = re.sub(r"[^a-z0-9 ]", " ", plain.casefold())
    tokens = [part for part in plain.split() if part not in {"jr", "sr", "ii", "iii", "iv"}]
    return " ".join(tokens)


def _team(value: Any) -> str:
    return re.sub(r"[^A-Z]", "", str(value or "").upper())


def _local_day(start_utc: str) -> str | None:
    try:
        return datetime.fromisoformat(start_utc.replace("Z", "+00:00")).astimezone(
            ZoneInfo("America/Chicago")
        ).date().isoformat()
    except (TypeError, ValueError):
        return None


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "Warlord-NHL-Prop-Tool/2026-grader"})
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def _json(session: requests.Session, path: str) -> dict[str, Any]:
    response = session.get(f"{API}{path}", timeout=25)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected NHL response at {path}")
    return payload


def schedule_for_day(session: requests.Session, slate_day: str) -> dict[tuple[str, str], int]:
    day = date.fromisoformat(slate_day)
    games: dict[tuple[str, str], int] = {}
    for query_day in (day, day + timedelta(days=1)):
        payload = _json(session, f"/schedule/{query_day.isoformat()}")
        for week in payload.get("gameWeek") or []:
            for game in week.get("games") or []:
                if _local_day(str(game.get("startTimeUTC") or "")) != slate_day:
                    continue
                away = _team((game.get("awayTeam") or {}).get("abbrev"))
                home = _team((game.get("homeTeam") or {}).get("abbrev"))
                gid = _number(game.get("id"))
                if away and home and gid is not None:
                    games[(away, home)] = int(gid)
    return games


def _game_id(row: dict[str, Any], schedule: dict[tuple[str, str], int]) -> int | None:
    explicit = _number(row.get("Game_ID"))
    if explicit is not None and explicit > 0:
        return int(explicit)
    game = str(row.get("Game") or "").upper().replace(" ", "")
    if "@" in game:
        away, home = (_team(part) for part in game.split("@", 1))
        if (away, home) in schedule:
            return schedule[(away, home)]
    team, opp = _team(row.get("Team")), _team(row.get("Opp"))
    return schedule.get((team, opp)) or schedule.get((opp, team))


def _players(box: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_game = box.get("playerByGameStats")
    if not isinstance(by_game, dict):
        return {}
    result: dict[str, list[dict[str, Any]]] = {}
    for side in ("awayTeam", "homeTeam"):
        team = _team((box.get(side) or {}).get("abbrev"))
        groups = by_game.get(side) or {}
        if not team or not isinstance(groups, dict):
            continue
        players = []
        for group in ("forwards", "defense", "defencemen"):
            for player in groups.get(group) or []:
                if isinstance(player, dict):
                    players.append(player)
        result[team] = players
    return result


def _match(row: dict[str, Any], box: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    roster = _players(box).get(_team(row.get("Team")), [])
    if not roster:
        return None, "NO_TEAM_STATS"
    player_id = _number(row.get("Player_ID"))
    if player_id is not None:
        matches = [player for player in roster if _number(player.get("playerId")) == player_id]
        return (matches[0], "PLAYER_ID") if len(matches) == 1 else (None, "NO_PLAYER" if not matches else "AMBIGUOUS_PLAYER")
    target = _name(row.get("Player"))
    if not target:
        return None, "NO_PLAYER"
    exact = [player for player in roster if _name(player.get("name")) == target]
    if len(exact) == 1:
        return exact[0], "EXACT_NAME"
    if len(exact) > 1:
        return None, "AMBIGUOUS_PLAYER"
    bits = target.split()
    if len(bits) < 2:
        return None, "NO_PLAYER"
    initial_last = (bits[0][0], bits[-1])
    matches = []
    for player in roster:
        name_bits = _name(player.get("name")).split()
        if len(name_bits) >= 2 and (name_bits[0][0], name_bits[-1]) == initial_last:
            matches.append(player)
    if len(matches) == 1:
        return matches[0], "INITIAL_LAST"
    return None, "AMBIGUOUS_PLAYER" if matches else "NO_PLAYER"


def _actuals(player: dict[str, Any]) -> dict[str, float | None]:
    goals = _number(player.get("goals"))
    assists = _number(player.get("assists"))
    points = _number(player.get("points"))
    if points is None and goals is not None and assists is not None:
        points = goals + assists
    sog = next((_number(player.get(key)) for key in ("sog", "shots", "shotsOnGoal") if _number(player.get(key)) is not None), None)
    return {"goals": goals, "assists": assists, "points": points, "sog": sog}


def grade_tracker(frame: pd.DataFrame, slate_day: str, session: requests.Session) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty or not {"Date", "Player", "Team"}.issubset(frame.columns):
        raise ValueError("Tracker must have player rows and Date, Player, Team columns")
    dates = pd.to_datetime(frame["Date"], errors="coerce").dt.date
    if dates.isna().any() or any(day.isoformat() != slate_day for day in dates):
        raise ValueError("Every tracker row must match the selected slate date")
    schedule = schedule_for_day(session, slate_day)
    rows = frame.to_dict("records")
    game_ids = {_game_id(row, schedule) for row in rows}
    boxes: dict[int, dict[str, Any] | None] = {}
    for gid in sorted(gid for gid in game_ids if gid is not None):
        try:
            boxes[gid] = _json(session, f"/gamecenter/{gid}/boxscore")
        except (requests.RequestException, ValueError):
            boxes[gid] = None

    additions: list[dict[str, Any]] = []
    for row in rows:
        gid = _game_id(row, schedule)
        added: dict[str, Any] = {"Resolved_Game_ID": gid, "Resolved_Player_ID": None, "Player_Match_Method": ""}
        box = boxes.get(gid)
        if gid is None:
            status = "NO_GAME"
        elif box is None:
            status = "API_ERROR"
        elif _local_day(str(box.get("startTimeUTC") or "")) != slate_day:
            status = "GAME_DATE_MISMATCH"
        elif _team(row.get("Team")) not in {
            _team((box.get("awayTeam") or {}).get("abbrev")),
            _team((box.get("homeTeam") or {}).get("abbrev")),
        }:
            status = "TEAM_MISMATCH"
        elif str(box.get("gameState") or "").upper() not in FINAL_STATES:
            status = "PENDING_GAME"
        else:
            player, method = _match(row, box)
            if player is None:
                status = method
            else:
                status = "FINAL"
                added["Resolved_Player_ID"] = player.get("playerId")
                added["Player_Match_Method"] = method
                stats = _actuals(player)
        added["Grade_Status"] = status
        for suffix, (line_col, stat_key) in MARKETS.items():
            actual = stats.get(stat_key) if status == "FINAL" else None
            line = _number(row.get(line_col))
            added[f"Actual_{suffix}"] = actual
            if status != "FINAL":
                outcome, market_status = "", status
            elif line is None:
                outcome, market_status = "", "NO_LINE"
            elif actual is None:
                outcome, market_status = "", "NO_STAT"
            else:
                outcome = "W" if actual > line else "L" if actual < line else "P"
                market_status = "OK"
            added[f"Outcome_{suffix}"] = outcome
            added[f"Match_Status_{suffix}"] = market_status
        additions.append(added)

    output = frame.copy().reset_index(drop=True)
    results = pd.DataFrame(additions)
    for column in results:
        output[column] = results[column].values
    summary = {
        "grader_version": VERSION,
        "slate_date": slate_day,
        "tracker_rows": len(output),
        "games_found": len(schedule),
        "grade_status": dict(Counter(output["Grade_Status"])),
        "markets": {suffix: dict(Counter(output[f"Outcome_{suffix}"].replace("", "UNSET"))) for suffix in MARKETS},
        "frozen_moves_available": "Fired_Moves" in output.columns,
    }
    return output, summary


def move_results(graded: pd.DataFrame) -> pd.DataFrame:
    """Expand frozen pregame move tags; never recompute old rules."""
    columns = ["Date", "Game", "Game_ID", "Player", "Player_ID", "Team", "Market",
               "Move", "Kind", "Rule", "Icon", "Kit_Version", "Line", "Actual",
               "Outcome", "Grade_Status", "Match_Status"]
    if "Fired_Moves" not in graded.columns:
        return pd.DataFrame(columns=columns)
    records = []
    for _, row in graded.iterrows():
        raw = row.get("Fired_Moves")
        if pd.isna(raw) or str(raw).strip() in ("", "[]"):
            continue
        try:
            tags = json.loads(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid frozen move tags for {row.get('Player')}: {error}") from error
        if not isinstance(tags, list):
            raise ValueError("Frozen move tags must be a JSON list")
        seen = set()
        for tag in tags:
            if not isinstance(tag, dict) or tag.get("market") not in MOVE_MARKETS or not tag.get("name"):
                raise ValueError("Frozen move tag has an invalid market or name")
            market = tag["market"]
            identity = (market, tag["name"])
            if identity in seen:
                raise ValueError(f"Duplicate frozen move: {identity}")
            seen.add(identity)
            suffix = MOVE_MARKETS[market]
            line = _number(row.get(f"{suffix}_Line"))
            outcome = row.get(f"Outcome_{suffix}", "")
            match_status = row.get(f"Match_Status_{suffix}", "")
            if market == "Goal" and line is None:
                line = _number(row.get("ATG_Line"))
                outcome = row.get("Outcome_ATG", "")
                match_status = row.get("Match_Status_ATG", "")
            records.append({
                "Date": row.get("Date"), "Game": row.get("Game"),
                "Game_ID": row.get("Resolved_Game_ID", row.get("Game_ID")),
                "Player": row.get("Player"), "Player_ID": row.get("Resolved_Player_ID", row.get("Player_ID")),
                "Team": row.get("Team"), "Market": market, "Move": tag["name"],
                "Kind": tag.get("kind", ""), "Rule": tag.get("rule", ""),
                "Icon": tag.get("icon", ""), "Kit_Version": tag.get("kit_version", ""),
                "Line": line, "Actual": row.get(f"Actual_{suffix}"),
                "Outcome": outcome if isinstance(outcome, str) else "",
                "Grade_Status": row.get("Grade_Status"), "Match_Status": match_status,
            })
    return pd.DataFrame(records, columns=columns)


def summarize_moves(moves: pd.DataFrame) -> list[dict[str, Any]]:
    if moves.empty:
        return []
    result = []
    keys = ["Market", "Move", "Kind", "Rule", "Kit_Version"]
    for group_key, rows in moves.groupby(keys, dropna=False, sort=True):
        counts = Counter(rows["Outcome"])
        wins, losses, pushes = (int(counts.get(key, 0)) for key in ("W", "L", "P"))
        settled = wins + losses
        result.append(dict(zip(keys, group_key)) | {
            "Wins": wins, "Losses": losses, "Pushes": pushes,
            "Unresolved": int(len(rows) - wins - losses - pushes),
            "Picks": settled, "Hit_Pct": round(100 * wins / settled, 1) if settled else None,
            "Players": int(rows["Player"].nunique()), "Games": int(rows["Game_ID"].nunique()),
        })
    return sorted(result, key=lambda row: (-row["Picks"], row["Market"], row["Move"]))


def _source_path(root: Path, slate_day: str) -> Path | None:
    frozen = root / "output" / "slates" / f"tracker_{slate_day}.csv"
    return frozen if frozen.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=(datetime.now(ZoneInfo("America/Chicago")).date() - timedelta(days=1)).isoformat())
    parser.add_argument("--tracker", type=Path, help="Explicit dated tracker CSV")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--skip-missing", action="store_true", help="Exit successfully when no dated tracker exists")
    args = parser.parse_args()
    date.fromisoformat(args.date)
    source = args.tracker or _source_path(args.repo, args.date)
    if source is None or not source.is_file():
        if args.skip_missing:
            print(f"No frozen tracker for {args.date}; skipping")
            return
        parser.error(f"No dated tracker for {args.date}")
    original = source.read_bytes()
    frame = pd.read_csv(source, low_memory=False)
    graded, summary = grade_tracker(frame, args.date, _session())
    moves = move_results(graded)
    summary["moves"] = summarize_moves(moves)
    summary["source_file"] = source.name
    summary["source_sha256"] = hashlib.sha256(original).hexdigest()
    destination = args.repo / "output" / "graded"
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / f"tracker_{args.date}_GRADED.csv"
    json_path = destination / f"tracker_{args.date}_summary.json"
    moves_path = destination / f"moves_{args.date}.csv"
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    json_tmp = json_path.with_suffix(".json.tmp")
    moves_tmp = moves_path.with_suffix(".csv.tmp")
    graded.to_csv(csv_tmp, index=False)
    moves.to_csv(moves_tmp, index=False)
    json_tmp.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    csv_tmp.replace(csv_path)
    moves_tmp.replace(moves_path)
    json_tmp.replace(json_path)
    print(f"Graded {args.date}: {summary['grade_status']} -> {csv_path}")


if __name__ == "__main__":
    main()
