"""Read-only, pregame quality checks for a saved NHL slate.

Findings are review prompts, never betting signals or changes to board ranking.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re

import pandas as pd

from warlords_night_board import rank_priced_slate


ROLE_MARKET = {"Carry": "Goals", "Support": "Assists", "Tank": "Points", "Jungle": "Shots"}


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _text(value):
    return "" if value is None or pd.isna(value) else str(value).strip()


def _true(value):
    return _text(value).casefold() in {"true", "1", "yes"}


def _stamp(value):
    return pd.to_datetime(value, utc=True, errors="coerce")


def _counts(frame):
    boards = rank_priced_slate(frame)
    counts = {ROLE_MARKET[role]: len(cards) for role, cards in boards.items()}
    counts["Power play"] = len(_pp_half_rows(frame))
    return boards, counts


def _pp_half_rows(frame):
    """One player per posted PPP 0.5 quote, using the tracker's best-price fields."""
    quotes = {}
    for row in frame.to_dict("records"):
        key = (_text(row.get("Team")).casefold(), _text(row.get("Player")).casefold())
        if not key[0] or not key[1]:
            continue
        for suffix in ("", "_1", "_2", "_3", "_4"):
            line = _number(row.get(f"BDL_PPP_Line{suffix}"))
            odds = _number(row.get(f"BDL_PPP_Odds{suffix}"))
            if line is not None and math.isclose(line, 0.5) and odds is not None and odds != 0:
                if key not in quotes or odds > quotes[key][0]:
                    quotes[key] = (odds, row)
    return [item[1] for item in quotes.values()]


def check_slate(frame: pd.DataFrame, *, previous: pd.DataFrame | None = None,
                selected_day=None, now_utc: datetime | None = None) -> dict:
    """Audit the entire loaded tracker; compare a previous same-day snapshot if supplied."""
    if frame.empty:
        return {"counts": {}, "findings": [], "review_candidates": []}
    now = pd.Timestamp(now_utc or datetime.now(timezone.utc)).tz_convert("UTC")
    findings = []
    review_candidates = []
    rows = frame.to_dict("records")
    by_player = {
        (_text(row.get("Team")).casefold(), _text(row.get("Player")).casefold()): row
        for row in rows
    }

    def add(level, scope, issue, evidence, *, market="", player="", team=""):
        findings.append({"Level": level, "Scope": scope, "Market": market,
                         "Player": player, "Team": team, "Check": issue,
                         "Evidence": evidence})

    boards, counts = _counts(frame)
    today = now.tz_convert("America/Chicago").date()
    slate_days = pd.to_datetime(frame.get("Date", pd.Series(dtype=str)), errors="coerce").dropna()
    slate_day = slate_days.max().date() if not slate_days.empty else None
    if slate_day is None:
        add("Action", "Slate", "Slate date missing", "No valid Date in tracker")
    elif selected_day is not None and slate_day != selected_day:
        add("Action", "Slate", "Loaded slate differs from selected date",
            f"Loaded {slate_day}; selected {selected_day}")
    elif slate_day < today:
        add("Action", "Slate", "Saved slate is from an earlier day", str(slate_day))

    checked = _stamp(frame.iloc[0].get("Odds_Checked_UTC"))
    if pd.isna(checked):
        add("Action", "Slate", "Odds check time missing", "Refresh to obtain timestamped prices")
    else:
        age = (now - checked).total_seconds() / 60
        if age >= 30:
            add("Action" if age >= 120 else "Watch", "Slate", "Saved prices need rechecking",
                f"Last checked {age:.0f} minutes ago; verify at the book")

    if "Injury_Reports_Count" in frame:
        n = _number(frame.iloc[0].get("Injury_Reports_Count"))
        if n is None or n <= 0:
            add("Action", "Slate", "Injury feed returned no reports", "Verify availability before using cards")

    if slate_day:
        expected_start = slate_day.year if slate_day.month >= 7 else slate_day.year - 1
        seasons = {_text(value) for value in frame.get("Model_Stats_Season", pd.Series(dtype=str))}
        old = sorted(s for s in seasons if s[:4].isdigit() and int(s[:4]) < expected_start)
        if old:
            add("Action", "Slate", "Prior-season model stats found", ", ".join(old))

    for market, count in counts.items():
        if count == 0:
            add("Watch", "Market", "No priced cards", "The books may not have posted this market yet",
                market=market)

    teams = {_text(x) for x in frame.get("Team", pd.Series(dtype=str)) if _text(x)}
    scheduled_teams = set()
    for game in frame.get("Game", pd.Series(dtype=str)):
        matchup = re.fullmatch(r"([A-Z]{2,3})@([A-Z]{2,3})", _text(game))
        if matchup:
            scheduled_teams.update(matchup.groups())
    missing_sides = sorted(scheduled_teams - teams)
    if missing_sides:
        add("Action", "Slate", "A matchup has no skaters from one team",
            ", ".join(missing_sides))
    if previous is not None and not previous.empty:
        prev_days = pd.to_datetime(previous.get("Date", pd.Series(dtype=str)), errors="coerce").dropna()
        if slate_day and not prev_days.empty and prev_days.max().date() == slate_day:
            _, old_counts = _counts(previous)
            old_teams = {_text(x) for x in previous.get("Team", pd.Series(dtype=str)) if _text(x)}
            lost = sorted(old_teams - teams)
            if lost:
                add("Action", "Refresh", "Teams disappeared since the last run", ", ".join(lost))
            for market, old_count in old_counts.items():
                new_count = counts[market]
                if old_count >= 8 and new_count < old_count * 0.6:
                    add("Action", "Refresh", "Priced cards dropped sharply",
                        f"{old_count} → {new_count} cards", market=market)

    starts = pd.to_datetime(frame.get("StartTimeUTC", pd.Series(dtype=str)), utc=True, errors="coerce")
    near_start = {}
    for index, row in frame.iterrows():
        start = starts.get(index, pd.NaT)
        near_start[(_text(row.get("Team")).casefold(), _text(row.get("Player")).casefold())] = (
            pd.notna(start) and -10 <= (start - now).total_seconds() / 60 <= 90
        )

    for role, cards in boards.items():
        market = ROLE_MARKET[role]
        for place, card in enumerate(cards):
            key = (card["team"].casefold(), card["player"].casefold())
            row = by_player.get(key)
            if row is None:
                continue
            notable = place < 10 or bool(card.get("baseline_rule")) or bool(card.get("move"))
            if not notable:
                continue
            issues = []
            if _true(row.get("Roster_Watch")):
                issues.append("Odds-only roster watch; model history missing")
            if _true(row.get("Team_Changed")):
                issues.append("Changed teams; inspect current-team stats and role")
            if near_start.get(key) and _text(row.get("Lineup_Status")) != "Dressed":
                issues.append("Game within 90 minutes; lineup still unconfirmed")
            if market == "Goals":
                ixg = _number(row.get("iXG%"))
                intent = _number(row.get("ShotIntent_Pct"))
                if place < 10 and ixg is not None and intent is not None and ixg < 50 and intent < 50:
                    issues.append(f"Top Goal card but goal intent is modest (iXG {ixg:g}%, shot intent {intent:g}%)")
                gp = _number(row.get("Opp_GP"))
                if gp is not None and gp < 3:
                    issues.append(f"Opponent goalie has only {gp:g} recorded game(s); goalie rates are fragile")
                if near_start.get(key) and _text(row.get("Opp_Goalie_Status")).casefold() not in {"confirmed", "official"}:
                    issues.append("Game within 90 minutes; opposing goalie unconfirmed")
            if issues:
                for issue in issues:
                    add("Watch", "Card", issue,
                        f"Rank #{place + 1} · {card['line']:g} at {card['odds']:+g} · {card['book'] or 'book unknown'}",
                        market=market, player=card["player"], team=card["team"])
                review_candidates.append({"market": market, "row": row, "player": card["player"],
                                          "team": card["team"], "rank": place + 1, "issues": issues})

    for row in _pp_half_rows(frame):
        minutes = _number(row.get("PP_TOI_per_game"))
        role = _number(row.get("PP_Role"))
        if role is not None and role >= 2 and minutes is not None and minutes < 2:
            add("Watch", "Card", "PP1 season tag with limited recorded PP time",
                f"{minutes:.1f} PP minutes/game; PP1 is a season rank, not tonight's confirmed unit",
                market="Power play", player=_text(row["Player"]), team=_text(row["Team"]))
            review_candidates.append({"market": "Power play", "row": row,
                                      "player": _text(row["Player"]), "team": _text(row["Team"]),
                                      "rank": None, "issues": ["PP1 tag and observed time differ"]})

    findings.sort(key=lambda item: (item["Level"] != "Action", item["Scope"] != "Refresh",
                                    item["Market"], item["Player"], item["Check"]))
    return {"counts": counts, "findings": findings, "review_candidates": review_candidates,
            "slate_day": slate_day, "teams": len(teams), "players": len(frame)}
