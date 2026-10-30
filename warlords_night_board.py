"""Read-only class board built from the current tracker and frozen move rules."""

from __future__ import annotations

import math
import base64
from functools import lru_cache
from html import escape
from pathlib import Path

import pandas as pd

from warlord_moves_2026 import fired_moves
from player_availability import is_unavailable
from player_form import summarize_form
from points15_research import context as points15_research_context


CLASSES = (
    ("Carry", "Goal", "⚔️", "#ef6a65"),
    ("Support", "Assists", "🪄", "#b692ff"),
    ("Tank", "Points", "🛡️", "#79afff"),
    ("Jungle", "SOG", "🌿", "#77d6a6"),
)


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _value(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None and not pd.isna(value) and str(value).strip():
            return value
    return None


def _line_price(row, market):
    prefix = {"Goal": ("Goal", "ATG"), "Points": ("Points",),
              "Assists": ("Assists",), "SOG": ("SOG",)}[market]
    for item in prefix:
        line = _number(row.get(f"{item}_Line"))
        if line is not None:
            odds = _number(row.get(f"{item}_Odds_Over"))
            book = _value(row, f"{item}_Book")
            if odds is None and market == "Goal":
                atg_line = _number(row.get("ATG_Line"))
                if atg_line is not None and math.isclose(atg_line, line):
                    odds = _number(row.get("ATG_Odds_Over"))
                    book = book or _value(row, "ATG_Book")
            if odds is None:
                continue
            return line, odds, book
    return None, None, None


def _price_comparison(row, market, line, odds):
    """Model Over probability and break-even rate for this exact posted line."""
    if line is None or odds is None or odds == 0:
        return None, None, None
    prefixes = ("Goal", "ATG") if market == "Goal" else (market,)
    model_prob = None
    for prefix in prefixes:
        quoted_line = _number(row.get(f"{prefix}_Line"))
        if quoted_line is None or not math.isclose(quoted_line, line):
            continue
        raw = _number(row.get(f"{prefix}_p_model_over"))
        if raw is None:
            pct = _number(row.get(f"{prefix}_Model%"))
            raw = pct / 100 if pct is not None else None
        if raw is not None and 0 < raw < 1:
            model_prob = raw
            break
    book_prob = 100 / (100 + odds) if odds > 0 else abs(odds) / (100 + abs(odds))
    return model_prob, book_prob, (model_prob - book_prob) if model_prob is not None else None


def _move_rank(move):
    wins, picks = int(move["wins"]), int(move["picks"])
    later_wins, later_picks = int(move["later_wins"]), int(move["later_picks"])
    if move.get("market") == "Assists":
        tier = {"CONTEXT": 0, "STANCE": 1, "VOLUME": 2,
                "SPECIAL": 3, "CRIT": 4}.get(move["kind"], 0)
        return (tier, wins / picks if picks else -1, picks)
    return (wins / picks if picks else -1, picks,
            later_wins / later_picks if later_picks else -1)


def _baseline_rule(row, market, line):
    matrix = str(row.get(f"Matrix_{market}", "")).strip().casefold()
    if matrix not in {"green", "🟢"}:
        return None
    if market == "Goal" and line == 0.5 and _number(row.get("Conf_Goal")) is not None:
        return "Green + Goal 0.5 + Conf_Goal ≥ 85" if _number(row.get("Conf_Goal")) >= 85 else None
    if market == "Assists" and line == 0.5:
        return "Green + Assists 0.5"
    if market == "Points" and line in (0.5, 1.5):
        threshold = 78 if line == 0.5 else 75
        if _number(row.get("Conf_Points")) is not None and _number(row.get("Conf_Points")) >= threshold:
            return f"Green + Points {line:g} + Conf_Points ≥ {threshold}"
    if market == "SOG" and line in (2.5, 3.5):
        if _number(row.get("Conf_SOG")) is not None and _number(row.get("Conf_SOG")) >= 75:
            return f"Green + SOG {line:g} + Conf_SOG ≥ 75"
    return None


def _display_moves(moves, market, line):
    """Keep Points 1.5 cards focused without changing saved move records."""
    if market == "Points" and line == 1.5:
        return [move for move in moves if move["name"] in {"DPS Tank stance", "Backbone"}]
    return moves


def points_current_tier(row, line, baseline):
    """Current-season 0.5 Points screen; old fired moves remain separate context."""
    if not baseline or line != 0.5:
        return None
    assists = _number(row.get("L10_A"))
    pp_minutes = _number(row.get("PP_TOI_per_game"))
    if assists is not None and assists >= 8:
        name, kind, wins, picks, rule = (
            "Points Special", "SPECIAL", 27, 33, "Last 10 assists ≥ 8")
    elif pp_minutes is not None and pp_minutes >= 3.5:
        name, kind, wins, picks, rule = (
            "Points Heavy", "HEAVY", 39, 52, "PP minutes/game ≥ 3.5")
    else:
        name, kind, wins, picks, rule = (
            "Points Base", "BASE", 51, 76, "Green · Points 0.5 · current confidence ≥ 78")
    return {"name": name, "kind": kind, "rule": rule,
            "wins": wins, "picks": picks, "later_wins": 0, "later_picks": 0,
            "sample_season": "2026-27", "sample_through": "2026-10-26",
            "pp_minutes": pp_minutes, "recent_assists": assists}


def rank_warlords(frame: pd.DataFrame) -> dict[str, list[dict]]:
    """Best fired historical move per player and class on this filtered slate."""
    boards = {role: {} for role, *_ in CLASSES}
    if frame.empty:
        return {role: [] for role in boards}
    for row in frame.to_dict("records"):
        if is_unavailable(row):
            continue
        if str(row.get("Roster_Watch", "")).strip().casefold() in {"true", "1"}:
            continue
        if not _value(row, "Player"):
            continue
        active = fired_moves(row)
        for role, market, _, _ in CLASSES:
            moves = active[market]
            line, odds, book = _line_price(row, market)
            # A model signal without a posted price is not a ready board pick.
            if line is None or odds is None or odds == 0:
                continue
            model_prob, book_prob, price_gap = _price_comparison(row, market, line, odds)
            baseline_rule = _baseline_rule(row, market, line)
            if not moves and not baseline_rule:
                continue
            baseline_only = not moves
            if baseline_only:
                move = {"name": "Green Baseline", "kind": "BASELINE", "rule": baseline_rule,
                        "wins": 0, "picks": 0, "later_wins": 0, "later_picks": 0}
            else:
                attacks = [move for move in moves if move["kind"] not in {"STANCE", "CONTEXT"}]
                move = max(attacks or moves, key=_move_rank)
            player = str(row["Player"]).strip()
            team = str(_value(row, "Team") or "").strip()
            key = (team.casefold(), player.casefold())
            candidate = {
                "player": player, "team": team,
                "talent_tier": str(_value(row, "Talent_Tier") or "NONE").strip().upper(),
                "position": str(_value(row, "Pos") or "").strip(),
                "opponent": str(_value(row, "Opp") or "").strip(),
                "game": str(_value(row, "Game") or "").strip(),
                "time": str(_value(row, "Time") or _value(row, "StartTimeLocal") or "").strip(),
                "market": market, "line": line, "odds": odds, "book": book,
                "form_log": _value(row, "Form_Log"),
                "form_season": _value(row, "Form_Season"),
                "model_prob": model_prob, "book_prob": book_prob, "price_gap": price_gap,
                "tonight_goal_xg": _number(row.get("Tonight_Goal_xG")) if market == "Goal" else None,
                "tonight_goal_pct": _number(row.get("Tonight_Goal_1plus_pct")) if market == "Goal" else None,
                "tonight_goal_status": str(_value(row, "Tonight_Goal_Status") or "") if market == "Goal" else "",
                "goal_profile": {field: _number(row.get(field)) for field in
                                 ("iXG%", "ShotIntent", "L10_G", "L10_S", "TOI_per_game", "Goal_mu")}
                                if market == "Goal" else None,
                "tonight_mu": _number(row.get(f"Tonight_{market}_mu")) if
                row.get(f"Tonight_{market}_Version") == "2026-10-19-research-v3" else None,
                "tonight_over_pct": _number(row.get(f"Tonight_{market}_over_pct")) if
                row.get(f"Tonight_{market}_Version") == "2026-10-19-research-v3" else None,
                "tonight_status": str(_value(row, f"Tonight_{market}_Status") or ""),
                "team_pp_xgf60": _number(row.get("Team_PP_xGF60")) if market == "Assists" else None,
                "goalie": str(_value(row, "Opp_Goalie") or ""),
                "goalie_gp": _number(_value(row, "Opp_GP")),
                "goalie_status": str(_value(row, "Opp_Goalie_Status") or "Unknown"),
                "lineup_status": str(_value(row, "Lineup_Status") or "Lineup pending"),
                "baseline_only": baseline_only,
                "move": move, "moves": sorted(moves, key=lambda item: (
                    item["kind"] != "STANCE", _move_rank(item)), reverse=True),
                "move_count": len(moves),
            }
            previous = boards[role].get(key)
            if previous is None or _move_rank(move) > _move_rank(previous["move"]):
                boards[role][key] = candidate
    return {role: sorted(players.values(),
                         key=lambda card: (_move_rank(card["move"]), card["player"]),
                         reverse=True)
            for role, players in boards.items()}


def rank_priced_slate(frame: pd.DataFrame) -> dict[str, list[dict]]:
    """Each player's displayed priced prop, ordered by current model confidence.

    Historical moves and baseline rules annotate cards; neither admits nor
    excludes a player from this board.
    """
    boards = {role: {} for role, *_ in CLASSES}
    if frame.empty:
        return {role: [] for role in boards}
    for row in frame.to_dict("records"):
        if is_unavailable(row):
            continue
        player = str(_value(row, "Player") or "").strip()
        if not player:
            continue
        team = str(_value(row, "Team") or "").strip()
        watch = str(row.get("Roster_Watch", "")).strip().casefold() in {"true", "1"}
        active = {market: [] for _, market, *_ in CLASSES} if watch else fired_moves(row)
        for role, market, _, _ in CLASSES:
            line, odds, book = _line_price(row, market)
            if line is None or odds is None or odds == 0:
                continue
            model_prob, book_prob, price_gap = _price_comparison(row, market, line, odds)
            tracked_moves = active[market]
            moves = _display_moves(tracked_moves, market, line)
            attacks = [move for move in moves if move["kind"] not in {"STANCE", "CONTEXT"}]
            best = max(attacks or moves, key=_move_rank) if moves else None
            baseline = None if watch else _baseline_rule(row, market, line)
            current_points_tier = points_current_tier(row, line, baseline) if market == "Points" else None
            confidence = _number(row.get(f"Conf_{market}"))
            matrix = str(_value(row, f"Matrix_{market}") or "Unknown").strip()
            move_block_reason = ("No current-season Goal attack fired"
                                 if market == "Goal" and baseline and not moves else None)
            move_confidence = (_number(row.get("Conf_Points_Move"))
                               if market == "Points" else None)
            if market == "Points" and baseline and not moves and move_confidence is not None:
                move_threshold = 80 if line == 0.5 else 75
                if move_confidence < move_threshold:
                    move_block_reason = f"Frozen move score below {move_threshold}"
            candidate = {
                "player": player, "team": team,
                "talent_tier": str(_value(row, "Talent_Tier") or "NONE").strip().upper(),
                "position": str(_value(row, "Pos") or "").strip(),
                "opponent": str(_value(row, "Opp") or "").strip(),
                "game": str(_value(row, "Game") or "").strip(),
                "time": str(_value(row, "Time") or _value(row, "StartTimeLocal") or "").strip(),
                "market": market, "line": line, "odds": odds, "book": book,
                "form_log": _value(row, "Form_Log"),
                "form_season": _value(row, "Form_Season"),
                "model_prob": model_prob, "book_prob": book_prob, "price_gap": price_gap,
                "tonight_goal_xg": _number(row.get("Tonight_Goal_xG")) if market == "Goal" else None,
                "tonight_goal_pct": _number(row.get("Tonight_Goal_1plus_pct")) if market == "Goal" else None,
                "tonight_goal_status": str(_value(row, "Tonight_Goal_Status") or "") if market == "Goal" else "",
                "goal_profile": {field: _number(row.get(field)) for field in
                                 ("iXG%", "ShotIntent", "L10_G", "L10_S", "TOI_per_game", "Goal_mu")}
                                if market == "Goal" else None,
                "tonight_mu": _number(row.get(f"Tonight_{market}_mu")) if
                row.get(f"Tonight_{market}_Version") == "2026-10-19-research-v3" else None,
                "tonight_over_pct": _number(row.get(f"Tonight_{market}_over_pct")) if
                row.get(f"Tonight_{market}_Version") == "2026-10-19-research-v3" else None,
                "tonight_status": str(_value(row, f"Tonight_{market}_Status") or ""),
                "team_pp_xgf60": _number(row.get("Team_PP_xGF60")) if market == "Assists" else None,
                "confidence": confidence, "move_confidence": move_confidence,
                "matrix": matrix,
                "baseline_rule": baseline, "baseline_only": bool(baseline and not best),
                "points_current_tier": current_points_tier,
                "points15_research": points15_research_context(row) if market == "Points" and line == 1.5 else None,
                "move_block_reason": move_block_reason,
                "goalie": str(_value(row, "Opp_Goalie") or ""),
                "goalie_status": str(_value(row, "Opp_Goalie_Status") or "Unknown"),
                "lineup_status": str(_value(row, "Lineup_Status") or "Lineup pending"),
                "move": best,
                "moves": sorted(moves, key=lambda item: (
                    item["kind"] != "STANCE", _move_rank(item)), reverse=True),
                "move_count": len(moves),
                "points_power_tier": market == "Points" and line == 1.5 and any(
                    item["name"] == "Blade Impale (Power Tier)" for item in tracked_moves),
                "roster_watch": watch,
            }
            key = (team.casefold(), player.casefold())
            previous = boards[role].get(key)
            if previous is None or (
                confidence is not None, confidence or -1,
                matrix.casefold() == "green", odds is not None
            ) > (
                previous["confidence"] is not None,
                previous["confidence"] or -1,
                previous["matrix"].casefold() == "green",
                previous["odds"] is not None,
            ):
                boards[role][key] = candidate
    return {role: sorted(players.values(), key=lambda card: (
        card["confidence"] is not None, card["confidence"] or -1,
        card["matrix"].casefold() == "green", card["player"].casefold()
    ), reverse=True) for role, players in boards.items()}


def featured_warlords(boards: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Character cards ranked by their strongest qualifying fired move.

    The complete priced slate stays in ``boards`` for the searchable tables.
    Historical rates select a display tier, not a claim of future probability.
    """
    featured = {}
    for role, cards in boards.items():
        selected = []
        for card in cards:
            if not card.get("baseline_rule"):
                continue
            if role == "Tank" and card.get("points_current_tier"):
                selected.append({**card, "move": card["points_current_tier"]})
                continue
            qualifying = [move for move in card["moves"] if move["kind"] != "CONTEXT"
                          and int(move["picks"]) > 0
                          and int(move["wins"]) / int(move["picks"]) >= 0.5]
            if not qualifying:
                continue
            selected.append({**card, "move": max(qualifying, key=_move_rank)})
        featured[role] = sorted(selected, key=lambda card: (
            {"SPECIAL": 3, "HEAVY": 2, "BASE": 1}.get(
                (card.get("points_current_tier") or {}).get("kind"), 0) if role == "Tank" else 0,
            _move_rank(card["move"]), card.get("confidence") or -1,
            card["player"].casefold()), reverse=True)
    return featured


def unfeatured_baselines(cards: list[dict]) -> list[dict]:
    """Priced Green candidates without a qualifying historical move."""
    return [card for card in cards if card.get("baseline_rule") and not any(
        move["kind"] != "CONTEXT" and int(move["picks"]) > 0
        and int(move["wins"]) / int(move["picks"]) >= 0.5
        for move in card["moves"])]


def goal_display_cards(cards: list[dict]) -> list[dict]:
    """Only named current-season Goal attacks reach the Warlord cards."""
    eligible = []
    for card in cards:
        if not card.get("baseline_rule"):
            continue
        attacks = [move for move in card.get("moves", [])
                   if int(move["picks"]) > 0 and int(move["wins"]) / int(move["picks"]) >= .5]
        if attacks:
            eligible.append({**card, "move": max(attacks, key=_move_rank)})
    return sorted(eligible, key=lambda card: (_move_rank(card["move"]),
                  card.get("confidence") or -1, card["player"].casefold()), reverse=True)


def baseline_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Show each priced baseline and where the other posted lines drop out.

    These are the current class entry rules. A baseline can qualify without a
    stronger named move, so this audit does not change the frozen move kit or grading.
    """
    specs = (
        ("Goals", "Goal", (0.5,), "Matrix_Goal", "Conf_Goal"),
        ("Assists", "Assists", (0.5,), "Matrix_Assists", None),
        ("Points", "Points", (0.5, 1.5), "Matrix_Points", "Conf_Points"),
        ("Shots", "SOG", (2.5, 3.5), "Matrix_SOG", "Conf_SOG"),
    )
    stages = ("Priced", "Supported line", "Green", "Baseline", "Named move")
    totals: dict[str, dict[str, set]] = {}
    teams: dict[tuple[str, str], dict[str, set]] = {}
    baseline_rows: list[dict] = []
    for label, market, lines, matrix_col, conf_col in specs:
        totals[label] = {stage: set() for stage in stages}
        for row in frame.to_dict("records"):
            if is_unavailable(row):
                continue
            player = str(_value(row, "Player") or "").strip()
            if not player:
                continue
            team = str(_value(row, "Team") or "").strip()
            identity = (team.casefold(), player.casefold())
            bucket = teams.setdefault((label, team), {stage: set() for stage in stages})
            line, odds, book = _line_price(row, market)
            if line is None or odds is None or odds == 0:
                continue
            for counts in (totals[label], bucket):
                counts["Priced"].add(identity)
            if str(row.get("Roster_Watch", "")).strip().casefold() in {"true", "1"}:
                continue
            if line not in lines:
                continue
            for counts in (totals[label], bucket):
                counts["Supported line"].add(identity)
            if str(row.get(matrix_col, "")).strip().casefold() not in {"green", "🟢"}:
                continue
            for counts in (totals[label], bucket):
                counts["Green"].add(identity)
            conf = _number(row.get(conf_col)) if conf_col else None
            if not _baseline_rule(row, market, line):
                continue
            for counts in (totals[label], bucket):
                counts["Baseline"].add(identity)
            named = any(move["kind"] not in {"STANCE", "CONTEXT"}
                        for move in fired_moves(row)[market])
            if named:
                for counts in (totals[label], bucket):
                    counts["Named move"].add(identity)
            stats_source = str(_value(row, "Opp_Goalie_Source") or "none")
            stats_label = {
                "dailyfaceoff_name_and_stats": "Matched starter stats",
                "dailyfaceoff_name_only": "Starter named; stats unavailable",
                "moneypuck_team_proxy_unconfirmed": "Team goalie proxy",
                "moneypuck_team_proxy": "Team goalie proxy",
                "none": "Unavailable",
            }.get(stats_source, stats_source)
            baseline_rows.append({
                "Prop": label, "Player": player, "Team": team,
                "Game": str(_value(row, "Game") or ""),
                "Line": line, "Odds": int(odds), "Book": str(book or ""),
                "Confidence": int(conf) if conf is not None else "—",
                "Status": "Named move" if named else "Baseline only",
                "Opp goalie": str(_value(row, "Opp_Goalie") or ""),
                "Goalie status": str(_value(row, "Opp_Goalie_Status") or "Unknown"),
                "Goalie stats": stats_label,
            })
    summary = pd.DataFrame([{"Prop": label, **{stage: len(totals[label][stage]) for stage in stages}}
                            for label, *_ in specs])
    by_team = pd.DataFrame([{"Prop": label, "Team": team,
                             **{stage: len(bucket[stage]) for stage in stages}}
                            for (label, team), bucket in sorted(teams.items())])
    roster = pd.DataFrame(baseline_rows, columns=(
        "Prop", "Player", "Team", "Game", "Line", "Odds", "Book", "Confidence", "Status",
        "Opp goalie", "Goalie status", "Goalie stats"
    )).drop_duplicates(["Prop", "Player", "Team"])
    return summary, by_team, roster


def _h(value):
    return escape(str(value), quote=True)


def _odds(value):
    if value is None:
        return "Odds pending"
    return f"{value:+.0f}" if value > 0 else f"{value:.0f}"


def _form_html(card, *, expanded=False):
    form = summarize_form(card.get("form_log"), card["market"], card["line"])
    if not form:
        season = _h(card.get("form_season") or "Current")
        return f'<div class="wn-form-empty">{season} regular-season form unavailable right now.</div>'
    if form.get("empty"):
        return f'<div class="wn-form-empty">{_h(form["season"])} regular season · awaiting completed games.</div>'
    l10_w, l10_n = form["l10"]
    l5_w, l5_n = form["l5"]
    season_w, season_n = form["season_rate"]
    avg, med = form["average"], form["median"]
    line = card["line"]
    tiles = (
        (f"LAST {l10_n}", f"{l10_w}/{l10_n} · {100*l10_w/l10_n:.0f}%"),
        ("AVG / MEDIAN", f"{avg:.1f} / {med:.1f}"),
        ("VS TODAY'S LINE", f"Avg {avg-line:+.1f} · Med {med-line:+.1f}"),
        (f"LAST {l5_n}", f"{l5_w}/{l5_n} · {100*l5_w/l5_n:.0f}%"),
        ("SEASON", f"{season_w}/{season_n} · {100*season_w/season_n:.0f}%"),
    )
    tiles_html = "".join(f'<div><small>{_h(label)}</small><strong>{_h(value)}</strong></div>'
                         for label, value in tiles)
    splits = " · ".join(f"{label} {won}/{count} ({100*won/count:.0f}%)"
                        for label, (won, count) in form["splits"].items())
    opportunity = []
    if form["shots"] is not None:
        opportunity.append(f"L{l10_n} shots {form['shots']:.1f}/game")
    if form["toi"] is not None:
        opportunity.append(f"ice time {form['toi']:.1f} min/game")
    strip = "".join(f'<span class="{"hit" if hit else "miss"}" title="{_h(date)} · {value} {_h(card["market"])}">'
                    f'<b>{"✓" if hit else "×"} {value}</b><small>{_h(date[5:].replace("-", "/") if date else "—")}</small></span>'
                    for value, hit, date in reversed(form["recent"]))
    label = "Power-play points" if card["market"] == "PPP" else "Recent form"
    return (f'<details class="wn-form"{" open" if expanded else ""}><summary>{label} · {_h(form["season"])} regular season</summary>'
            f'<p>Against today\'s over {line:g} line · results read left to right, oldest to newest. '
            'Historical results are context, not the move record.</p>'
            f'<div class="wn-form-tiles">{tiles_html}</div>'
            f'<div class="wn-form-meta">{_h(splits)}</div>'
            f'<div class="wn-form-meta">{_h(" · ".join(opportunity))}</div>'
            f'<div class="wn-form-strip">{strip}</div></details>')


def render_power_play_quotes_table(rows, signals_by_team):
    """Readable PP scouting rows with a vertical team-signal stack per player."""
    def cell(value):
        if value is None or pd.isna(value):
            return "—"
        return _h(value)

    body = []
    for row in rows:
        team = str(row.get("Team") or "").upper()
        signals = signals_by_team.get(team, [])
        stack = "".join(
            f'<div class="pp-signal">{ "+ " if index else ""}{_h(signal)}</div>'
            for index, signal in enumerate(signals)
        ) or '<div class="pp-signal">—</div>'
        onice = _number(row.get("PP on-ice xG/game"))
        onice_text = f"{onice:.3f}" if onice is not None else "—"
        pp_time = _number(row.get("Current PP min/game"))
        pp_time_text = f"{pp_time:.1f}" if pp_time is not None else "—"
        body.append(
            '<tr>'
            f'<td>{cell(row.get("PP rank"))}</td>'
            f'<td class="pp-player">{cell(row.get("Player"))}</td>'
            f'<td>{cell(row.get("Best feed price"))}</td>'
            f'<td>{cell(row.get("Game"))}</td>'
            f'<td>{onice_text}</td>'
            f'<td class="pp-signals">{stack}</td>'
            f'<td>{cell(row.get("PP research signals"))}</td>'
            f'<td>{cell(row.get("PP unit"))}</td>'
            f'<td>{pp_time_text}</td>'
            '</tr>'
        )
    return '''<style>
      .pp-scout-wrap{max-height:650px;overflow:auto;border:1px solid #d7dce4;border-radius:8px;margin:8px 0 12px}
      .pp-scout{border-collapse:separate;border-spacing:0;width:100%;min-width:1050px;font:12px Inter,system-ui,sans-serif;color:#1c2a3b}
      .pp-scout th,.pp-scout td{padding:7px 9px;border-bottom:1px solid #dce1e8;text-align:left;vertical-align:top;background:#fff}
      .pp-scout thead th{position:sticky;top:0;z-index:2;background:#eef2f7;white-space:nowrap;font-weight:700}
      .pp-scout .pp-player{position:sticky;left:0;z-index:1;background:#fff;font-weight:700;min-width:145px}
      .pp-scout thead .pp-player{z-index:3;background:#eef2f7}
      .pp-scout .pp-signals{min-width:260px;max-width:340px;white-space:normal;line-height:1.35}
      .pp-scout .pp-signal+ .pp-signal{margin-top:3px}
      .pp-scout tbody tr:nth-child(even) td{background:#f8fafc}
      .pp-scout tbody tr:nth-child(even) .pp-player{background:#f8fafc}
    </style><div class="pp-scout-wrap"><table class="pp-scout"><thead><tr>
      <th>Rank</th><th class="pp-player">Player</th><th>Best price</th><th>Game</th>
      <th>PP on-ice xG/g</th><th>Team PP matchup signals</th><th>Player signals</th>
      <th>PP unit</th><th>PP min/g</th></tr></thead><tbody>''' + "".join(body) + '</tbody></table></div>'



def render_power_play_form(card):
    """Show the same dated player form card on the separate PPP scouting page."""
    portrait = _character_uri("Support")
    image = f'<img src="{portrait}" alt="" />' if portrait else "🪄"
    price = _odds(card.get("odds"))
    form = _form_html({**card, "market": "PPP", "line": 0.5}, expanded=True)
    def metric(value, suffix=""):
        try:
            number = float(value)
            if math.isfinite(number):
                return f"{number:.1f}{suffix}"
        except (TypeError, ValueError):
            pass
        return "—"

    def precise_metric(value):
        try:
            number = float(value)
            if math.isfinite(number):
                return f"{number:.3f}"
        except (TypeError, ValueError):
            pass
        return "—"

    def rank(value):
        try:
            number = float(value)
            return f"#{int(number)}" if math.isfinite(number) else "—"
        except (TypeError, ValueError):
            return "—"

    tiles = (
        ("Player PP role", str(card.get("pp_unit") or "Unconfirmed")),
        ("PP on-ice xG/game", precise_metric(card.get("pp_onice_xg_game"))),
        ("Player signals", str(card.get("pp_research") or "Data missing")),
        ("Player PP iXG/game", precise_metric(card.get("pp_ixg_game"))),
        ("PPP L10 observed", card.get("ppp_l10") or "—"),
        ("PP-point drought", f'{int(card["ppp_drought"])} game' +
         ('' if int(card["ppp_drought"]) == 1 else 's')
         if _number(card.get("ppp_drought")) is not None else "—"),
        ("Opp PK time/game", str(card.get("opp_pk_time_pg") or "—")),
        ("Team PP xGF/60", f'{metric(card.get("team_pp_xgf60"))} · {rank(card.get("team_pp_rank"))} attack'),
        ("Opp PK xGA/60", f'{metric(card.get("opp_pk_xga60"))} · {rank(card.get("opp_pk_rank"))} allowed'),
        ("Opp PK GA/60", metric(card.get("opp_pk_ga60"))),
    )
    matchup = "".join(f'<div class="ppp-matchup-tile"><small>{_h(label)}</small><strong>{_h(value)}</strong></div>'
                      for label, value in tiles)
    research_signals = card.get("team_research_signals") or []
    research_banner = (
        '<div class="ppp-research"><small>ALL MATCHING TEAM PP-GOAL SIGNALS · 2026–27 THROUGH OCT 29</small>'
        + "".join(f'<strong class="ppp-research-item">{ "+ " if index else ""}{_h(signal)}</strong>'
                  for index, signal in enumerate(research_signals))
        + '<small>Overlapping team-game records; not this player’s PPP chance.</small></div>'
        if research_signals else
        '<div class="ppp-research"><small>No saved team PP-goal matchup signal for this opponent.</small></div>'
    )
    return f'''<style>
      .ppp-card,.ppp-card *{{box-sizing:border-box}}
      .ppp-card{{font-family:Inter,system-ui,sans-serif;color:#edf1f8;background:#1b2739;border:1px solid #4d416e;border-left:3px solid #b692ff;border-radius:10px;padding:14px;margin:12px 0}}
      .ppp-head{{display:flex;align-items:center;gap:12px;margin-bottom:8px}}
      .ppp-portrait{{width:50px;height:50px;flex:none;overflow:hidden;border-radius:9px;border:1px solid #8d72c4;background:#172039}}
      .ppp-portrait img{{width:100%;height:100%;object-fit:cover;object-position:center top}}
      .ppp-title{{min-width:0;flex:1}}.ppp-title strong{{display:block;font-size:17px;color:#fff}}.ppp-title span{{font-size:11px;color:#b9c9dc}}
      .ppp-price{{text-align:right;white-space:nowrap;color:#e9d5ff;font-size:20px;font-weight:900}}.ppp-price small{{display:block;color:#aebbd0;font-size:10px;font-weight:600}}
      .ppp-research{{display:flex;flex-wrap:wrap;align-items:center;gap:6px 12px;margin:8px 0;padding:8px 10px;border-radius:7px;border:1px solid #5884a1;background:#123044;color:#d1f1ff}}
      .ppp-research strong{{font-size:12px;letter-spacing:.04em}}.ppp-research small{{font-size:9px;opacity:.85;font-weight:700}}
      .ppp-research-item{{display:block;width:100%;padding:3px 0;border-top:1px solid #ffffff2b}}
      .ppp-research--buff{{border-color:#a87de4;background:#33244c;color:#f4e4ff}}
      .ppp-research--watch{{border-color:#e1ae4f;background:#4d3617;color:#fff0c6}}
      .ppp-matchup{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:6px;margin:10px 0}}
      .ppp-matchup-tile{{padding:8px;background:#111b2e;border:1px solid #ffffff26;border-radius:6px}}
      .ppp-matchup-tile small{{display:block;color:#afbed6;font-size:9px;text-transform:uppercase;letter-spacing:.06em}}
      .ppp-matchup-tile strong{{display:block;color:#fff;font-size:13px;margin-top:4px}}
      .ppp-context-note{{color:#aebbd0;font-size:10px;margin:2px 0 9px}}
      .wn-form,.wn-form-empty{{color:#b9c9dc;border-top:1px solid #ffffff24;padding-top:8px;font-size:12px}}.wn-form summary{{cursor:pointer;color:#e4c892;font-weight:800}}.wn-form p{{margin:7px 0;color:#aebbd0}}
      .wn-form-tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:5px;margin:8px 0}}.wn-form-tiles>div{{border:1px solid #ffffff1d;border-radius:5px;padding:7px;background:#0b1629a8}}.wn-form-tiles small{{display:block;color:#98adc6;font-size:9px}}.wn-form-tiles strong{{display:block;color:#fff;font-size:12px;margin-top:3px}}
      .wn-form-meta{{margin:4px 0}}.wn-form-strip{{display:flex;flex-wrap:wrap;gap:5px;margin:8px 0}}.wn-form-strip span{{border-radius:4px;padding:5px 7px;text-align:center;min-width:42px}}.wn-form-strip span b,.wn-form-strip span small{{display:block}}.wn-form-strip span small{{font-size:9px;opacity:.85}}.wn-form-strip .hit{{background:#1d6145;color:#c4ffde}}.wn-form-strip .miss{{background:#603036;color:#ffd7d7}}
    </style><section class="ppp-card"><div class="ppp-head"><div class="ppp-portrait">{image}</div>
      <div class="ppp-title"><strong>{_h(card.get("player") or "Player")}</strong><span>{_h(card.get("game") or "")} · Over 0.5 power-play points · {_h(card.get("book") or "Book pending")}</span></div>
      <div class="ppp-price">{_h(price)}<small>BEST FEED PRICE</small></div></div>
      {research_banner}
      <div class="ppp-matchup">{matchup}</div>
      <div class="ppp-context-note">Research order {rank(card.get("pp_rank"))} · PP1 is inferred from recorded usage · the team matchup flag applies to every teammate · player signals are past research, not tonight's probability</div>
      {form}</section>'''


@lru_cache(maxsize=4)
def _character_uri(role: str) -> str:
    """Inline small artwork so class cards render behind Cloud's app proxy."""
    image = Path(__file__).parent / "static" / "characters" / f"{role.lower()}-gorilla-v2.webp"
    if not image.is_file():
        return ""
    return "data:image/webp;base64," + base64.b64encode(image.read_bytes()).decode("ascii")


def render_warlords(boards: dict[str, list[dict]], limit: int = 5, icon_loader=None,
                    roles: tuple[str, ...] | None = None, show_hero: bool = True) -> str:
    """Compact, escaped HTML for the raid board or selected class lanes."""
    descriptions = {"Carry": "Finish the fight", "Support": "Set the play",
                    "Tank": "Hold the line", "Jungle": "Control the lanes"}
    selected = tuple(item for item in CLASSES if roles is None or item[0] in roles)
    lanes = []
    for role, market, symbol, color in selected:
        cards = boards.get(role, [])
        character_uri = _character_uri(role)
        class_icon = symbol
        units = []
        for rank, card in enumerate(cards[:limit], 1):
            move = card.get("move")
            points_tier = card.get("points_current_tier") if role == "Tank" else None
            tier_badge = ""
            tier_class = ""
            if role == "Support" and move and move["kind"] == "CRIT":
                tier_badge = '<span class="wn-tier wn-tier--s">CRIT</span>'
                tier_class = " wn-unit--crit"
            if points_tier:
                tier_style = {"SPECIAL": "s", "HEAVY": "a", "BASE": "b"}[points_tier["kind"]]
                tier_badge = f'<span class="wn-tier wn-tier--{tier_style}">{points_tier["kind"]}</span>'
                tier_class = f" wn-unit--{tier_style}"
            baseline_only = bool(card.get("baseline_only") and not points_tier)
            wins, picks = (int(move["wins"]), int(move["picks"])) if move else (0, 0)
            late_wins, late_picks = ((int(move["later_wins"]), int(move["later_picks"]))
                                     if move else (0, 0))
            pct = 100 * wins / picks if picks else 0
            late_pct = 100 * late_wins / late_picks if late_picks else 0
            status = (("ELITE GOAL WATCH" if card.get("goal_candidate") and card.get("talent_tier") == "ELITE" else
                       "STAR GOAL WATCH" if card.get("goal_candidate") and card.get("talent_tier") == "STAR" else
                       "BASELINE" if "confidence" in card else "BASELINE ONLY") if baseline_only
                      else "OPEN POOL" if not move else
                      "GOAL SIGNAL" if card.get("goal_candidate") and picks and wins / picks < .5 else
                      "TRACK" if move.get("track") else "LAB" if move.get("experimental")
                      or move["kind"] == "LAB CRIT" else move["kind"])
            if points_tier:
                status = f'{points_tier["kind"]} · EARLY 2026–27 SCREEN'
            sample = " · SMALL SAMPLE" if 0 < picks < 30 else ""
            line = f"OVER {card['line']:g} {market.upper()}" if card["line"] is not None else market.upper()
            matchup = card["game"] or card["team"]
            if card.get("time"):
                matchup = f'{matchup} · {card["time"]} CT'
            portrait = (f'<img src="{character_uri}" alt="" />' if character_uri else symbol)
            name_backdrop = (f'<img class="wn-unit-ghost" src="{character_uri}" alt="" aria-hidden="true" />'
                             if character_uri else "")
            price = _odds(card["odds"])
            book = str(card.get("book") or "").strip()
            book_note = f' <small class="wn-book">BEST PRICE · {_h(book)}</small>' if book else ""
            book_prob = card.get("book_prob")
            book_rate_note = (f'<div class="wn-book-rate">BOOK BREAK-EVEN {book_prob*100:.1f}%</div>'
                              if book_prob is not None else "")
            tonight_xg = _number(card.get("tonight_goal_xg")) if role == "Carry" else None
            tonight_pct = _number(card.get("tonight_goal_pct")) if role == "Carry" else None
            tonight_note = (f'<div class="wn-goalie">TONIGHT RESEARCH PROJECTION · '
                            f'{tonight_xg:.2f} expected goals · {tonight_pct:.1f}% for 1+ '
                            f'· being graded, not a tested move</div>'
                            if tonight_xg is not None and tonight_pct is not None else "")
            profile = card.get("goal_profile") or {}
            profile_parts = []
            for field, label, fmt in (("iXG%", "iXG rank", ".0f"),
                                      ("ShotIntent", "shot intent", ".1f"),
                                      ("L10_S", "L10 shots", ".0f"),
                                      ("L10_G", "L10 goals", ".0f"),
                                      ("TOI_per_game", "TOI/game", ".1f"),
                                      ("Goal_mu", "model goal μ", ".2f")):
                value = _number(profile.get(field))
                if value is not None:
                    profile_parts.append(f"{label} {value:{fmt}}")
            goal_profile_note = (f'<div class="wn-goal-profile">GOAL PROFILE · {_h(" · ".join(profile_parts))}</div>'
                                 if role == "Carry" and profile_parts else "")
            if role != "Carry":
                projected_mu = _number(card.get("tonight_mu"))
                projected_pct = _number(card.get("tonight_over_pct"))
                if projected_mu is not None and projected_pct is not None:
                    unit = "shots" if role == "Jungle" else "assists" if role == "Support" else "points"
                    tonight_note = (f'<div class="wn-goalie">TONIGHT RESEARCH PROJECTION · '
                                    f'{projected_mu:.2f} expected {unit} · {projected_pct:.1f}% '
                                    f'over {card["line"]:g} · being graded, not a tested move</div>')
            goalie_name = str(card.get("goalie") or "").strip()
            goalie_status = str(card.get("goalie_status") or "Unknown").strip()
            if goalie_status.casefold() == "confirmed":
                goalie_status = "DailyFaceoff confirmed"
            elif goalie_status.casefold() == "official":
                goalie_status = "NHL official starter"
            goalie_gp = _number(card.get("goalie_gp"))
            sample_note = (f' · {int(goalie_gp)} GP, goalie matchup neutral'
                           if goalie_gp is not None and goalie_gp < 3 else
                           f' · {int(goalie_gp)} GP' if goalie_gp is not None else '')
            goalie_note = (f'<div class="wn-goalie">Opp goalie: {_h(goalie_name)} · {_h(goalie_status)}{sample_note}</div>'
                           if goalie_name else '<div class="wn-goalie">Opp goalie: unknown</div>')
            lineup_note = f'<div class="wn-goalie">Player lineup: {_h(card.get("lineup_status") or "Lineup pending")}</div>'
            positional = card.get("position_sog") if role == "Jungle" else None
            position_note = (f'<div class="wn-position-sog">{_h(positional["opponent"])} allows '
                             f'<b>{positional["shots_per_game"]:.1f} SOG/game</b> to {_h(positional["position"])} '
                             f'· #{positional["rank"]} most · {positional["gp"]} GP</div>'
                             if positional else "")
            fired = card.get("moves") or ([] if not move or baseline_only else [move])
            support_badges = ""
            pp_crit_note = ""
            points_power_note = ('<div class="wn-support-badges">Enhancement: Blade Impale (Power Tier)</div>'
                                 if role == "Tank" and card.get("points_power_tier") else "")
            points_tier_note = (f'<div class="wn-crit-env">{_h(points_tier["rule"])} · '
                                f'sample through Oct 26 · overlaps with other tiers</div>'
                                if points_tier else "")
            points_research_note = ""
            research = card.get("points15_research") or {}
            if role == "Tank" and card.get("line") == 1.5 and research.get("eligible"):
                metrics = []
                if research.get("pp_xgf60") is not None:
                    metrics.append(f'PP on-ice xGF/60 {research["pp_xgf60"]:.1f}')
                if research.get("attempts60") is not None:
                    metrics.append(f'5v5 attempts/60 {research["attempts60"]:.1f}')
                if research.get("opp_hdca60") is not None:
                    metrics.append(f'Opp 5v5 HDCA/60 {research["opp_hdca60"]:.2f}')
                tags = []
                if research.get("pp_pressure"):
                    tags.append("PP PRESSURE")
                if research.get("shot_generator"):
                    tags.append("5v5 SHOT GENERATOR")
                if research.get("dual"):
                    tags.append("DUAL PROFILE")
                if research.get("stingy_defense"):
                    tags.append("STINGY DEFENSE · CAUTION")
                detail = " · ".join(metrics) if metrics else "Research metrics pending"
                tag_text = " · ".join(tags) if tags else "NO RESEARCH TAG"
                points_research_note = (f'<div class="wn-research">RESEARCH · {_h(tag_text)}<br>'
                                        f'{_h(detail)} · tracking only; no card or move change</div>')
            if role == "Support" and move:
                secondary = list(dict.fromkeys(
                    item["name"] for item in fired
                    if item["name"] != move["name"] and item["kind"] not in {"STANCE", "CRIT"}
                ))
                if secondary:
                    support_badges = (f'<div class="wn-support-badges">Also: '
                                      f'{_h(" · ".join(secondary[:4]))}</div>')
                pp_rate = _number(card.get("team_pp_xgf60"))
                if move["kind"] == "CRIT" and pp_rate is not None:
                    pp_crit_note = (f'<div class="wn-crit-env">TEAM PP xGF/60 '
                                    f'{pp_rate:.2f} · CRIT GATE 8.3</div>')
            move_rows = []
            for fired_move in fired:
                fired_wins, fired_picks = int(fired_move["wins"]), int(fired_move["picks"])
                fired_late_wins = int(fired_move["later_wins"])
                fired_late_picks = int(fired_move["later_picks"])
                fired_pct = 100 * fired_wins / fired_picks if fired_picks else 0
                fired_late_pct = 100 * fired_late_wins / fired_late_picks if fired_late_picks else 0
                fired_rule = str(fired_move.get("rule", fired_move.get("condition", "")))
                # Historical move keys stay frozen for comparable records.
                # Show their actual source meaning rather than implying xA.
                fired_rule = (fired_rule.replace("PP_iXA60", "weighted PP assists/60 (actual)")
                                         .replace("PP_Points60", "PP involvement proxy/60")
                                         .replace("iXA%", "observed assist rank %"))
                fired_label = ("TRACK" if fired_move.get("track") else
                               "LAB" if fired_move.get("experimental") else fired_move["kind"])
                later_text = (f"Mar 15+ {fired_late_wins}/{fired_late_picks} · {fired_late_pct:.1f}%"
                              if fired_late_picks else "2026–27 regular season"
                              if fired_move.get("sample_season") == "2026-27" else "No separate later sample")
                move_rows.append(f'''<div class="wn-move-entry">
                  <div class="wn-move-title"><strong>{_h(fired_move["name"])}</strong><em>{_h(fired_label)}</em></div>
                  <div class="wn-move-record">{fired_wins}/{fired_picks} · {fired_pct:.1f}% <span>{later_text}</span></div>
                  <div class="wn-move-rule">{_h(fired_rule)}</div>
                </div>''')
            later_note = (f"MOVE {wins}/{picks}" +
                          (f" · MAR 15+ {late_wins}/{late_picks} ({late_pct:.1f}%)" if late_picks else
                           " · GOAL RECORD WITHOUT POINTS GATE" if card.get("goal_candidate") else "")
                          if picks else "No tested move · posted line" if "confidence" in card
                          else "Baseline screen · no upgraded move")
            if points_tier:
                later_note = f'CURRENT SCREEN {wins}/{picks}'
            confidence = card.get("confidence")
            confidence_note = (f" · CONF {confidence:.0f} {_h(card.get('matrix') or 'Unknown')}"
                               if confidence is not None else "")
            record_label = ("2026–27 REG MOVE" if role == "Carry" or
                            (move and move.get("sample_season") == "2026-27") else "HISTORICAL MOVE")
            if points_tier:
                record_label = "2026–27 REG SCREEN"
            record_html = (f'<div class="wn-record"><strong>{pct:.1f}%</strong><span>{wins}/{picks}</span><em>{record_label}</em></div>'
                           if picks else '<div class="wn-record"><strong>BASE</strong><em>SCREEN ONLY</em></div>')
            details_html = (f'<details class="wn-details"><summary>Full fired move list ({len(fired)})</summary>'
                            f'<div class="wn-move-list">{"".join(move_rows)}</div></details>' if fired else
                            f'<div class="wn-details">{_h((points_tier or {}).get("rule") or card.get("move_block_reason") or card.get("baseline_rule") or (move or {}).get("rule") or "No tested move fired")}</div>')
            move_name = move["name"] if move else ("Green Baseline" if baseline_only else "Scouting pool")
            units.append(f"""<article class="wn-unit{tier_class}">
              {name_backdrop}
              <div class="wn-portrait" aria-hidden="true">{portrait}</div>
              <div class="wn-unit-body">
                <div class="wn-unit-head"><span class="wn-rank">{rank:02d}</span>{tier_badge}<strong>{_h(card['player'])}</strong><span class="wn-match">{_h(matchup)}</span></div>
                <div class="wn-attack"><span class="wn-attack-name">{_h(move_name)}</span><span class="wn-badge">{_h(status + sample)}</span></div>
                <div class="wn-unit-foot"><span>{_h(line)} <b>{_h(price)}</b>{book_note}</span><span>{_h(later_note)}{confidence_note}</span></div>
                {book_rate_note}
                {pp_crit_note}
                {points_power_note}
                {points_tier_note}
                {points_research_note}
                {support_badges}
                {goal_profile_note}
                {tonight_note}
                {position_note}
                {goalie_note}
                {lineup_note}
              </div>
              {record_html}
              {_form_html(card)}
              {details_html}
            </article>""")
        if not units:
            units = ['<div class="wn-empty">No featured move for this class on this slate. '
                     'Check the complete priced slate below for all posted lines.</div>']
        backdrop = f'<img class="wn-gorilla" src="{character_uri}" alt="" />' if character_uri else ""
        count_label = "FEATURED" if cards and "confidence" in cards[0] else "PRICED"
        lane_subtitle = ("GOAL ✦ 2026–27 MOVE % ✦ BOOK BREAK-EVEN % FROM POSTED ODDS"
                         if role == "Carry" else
                         "POINTS ✦ CURRENT-SEASON SCREEN % ✦ BOOK BREAK-EVEN %" if role == "Tank" else
                         f"{market.upper()} ✦ HISTORICAL MOVE % ✦ BOOK BREAK-EVEN % FROM POSTED ODDS")
        lanes.append(f"""<section class="wn-lane wn-lane--{role.lower()}" style="--accent:{color}">
          <header class="wn-lane-head">{backdrop}<div class="wn-class-icon" aria-hidden="true">{class_icon}</div>
            <div class="wn-class-text"><span class="wn-kicker">{_h(descriptions[role])}</span><h2>{_h(role)}</h2></div>
            <div class="wn-count"><strong>{len(cards)}</strong><span>{count_label}</span></div></header>
          <div class="wn-lane-sub">{_h(lane_subtitle)}</div>
          <div class="wn-units">{''.join(units)}</div></section>""")
    total = sum(len(cards) for cards in boards.values())
    styles = """<style>
      .wn-board,.wn-board *{box-sizing:border-box}
      .wn-board{font-family:Inter,system-ui,sans-serif;color:#edf1f8}
      .wn-hero{position:relative;overflow:hidden;background:radial-gradient(circle at 89% 4%,#663c274d,transparent 36%),linear-gradient(115deg,#111a2a,#1b1a2b 64%,#261b23);border:1px solid #53516b;border-radius:14px;padding:22px 25px;margin:10px 0 15px;box-shadow:inset 0 0 50px #0005}
      .wn-hero:after{content:'⚔';position:absolute;right:28px;top:-36px;font-size:145px;line-height:1;color:#ffffff0e;transform:rotate(-18deg)}
      .wn-eyebrow,.wn-hero-foot{font-size:10px;letter-spacing:.23em;color:#e9b875;font-weight:800}
      .wn-hero h1{font-size:clamp(27px,3vw,43px);letter-spacing:.06em;line-height:1.05;margin:7px 0 8px;font-weight:950;color:#fff;text-shadow:0 3px 18px #0009}
      .wn-hero p{margin:0 0 14px;color:#cbd2df;font-size:13px}.wn-hero-foot{color:#aebbd1;letter-spacing:.11em}
      .wn-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;align-items:start}
      .wn-lane{min-width:0;background:#111a2a;border:1px solid #38435a;border-radius:14px;overflow:hidden;box-shadow:0 8px 28px #1018282e}
      .wn-lane-head{position:relative;isolation:isolate;display:flex;align-items:center;gap:12px;min-height:96px;padding:16px;background:linear-gradient(100deg,color-mix(in srgb,var(--accent) 23%,#111a2a),#111a2a 82%);border-bottom:1px solid #ffffff16;overflow:hidden}
      .wn-gorilla{position:absolute;z-index:-1;right:8px;top:0;height:230px;width:auto;opacity:.34;pointer-events:none;mask-image:linear-gradient(90deg,transparent,#000 35%)}
      .wn-class-icon{width:42px;height:42px;flex:none;display:grid;place-items:center;border:1px solid color-mix(in srgb,var(--accent) 50%,transparent);border-radius:9px;background:#0a1425e8;font-size:25px}
      .wn-class-icon svg{width:29px;height:29px;max-width:29px;max-height:29px;fill:var(--accent)}
      .wn-class-text{flex:1}.wn-kicker{font-size:10px;text-transform:uppercase;letter-spacing:.15em;color:#c5c6d3}.wn-class-text h2{font-size:23px;line-height:1;margin:3px 0 0;color:var(--accent);font-weight:950}
      .wn-count{display:flex;flex-direction:column;align-items:center;color:var(--accent);line-height:1;background:#0a1425b8;border:1px solid #ffffff16;border-radius:8px;padding:6px 8px}.wn-count strong{font-size:25px}.wn-count span{font-size:9px;letter-spacing:.12em;margin-top:3px}
      .wn-lane-sub{font-size:9px;font-weight:800;letter-spacing:.12em;color:#8898b2;padding:8px 16px;background:#0d1625}.wn-lane-sub span{color:var(--accent);padding:0 4px}
      .wn-units{padding:8px}.wn-unit{position:relative;isolation:isolate;overflow:hidden;display:flex;flex-wrap:wrap;gap:4px 10px;min-height:104px;align-items:center;background:#1b2739;border:1px solid #3b4b64;border-left:3px solid var(--accent);border-radius:9px;padding:10px;margin-bottom:7px}
      .wn-unit--s{border-left:4px solid #f4bf55;background:linear-gradient(100deg,#3d3026,#1b2739 39%);box-shadow:inset 0 0 0 1px #f4bf5538}
      .wn-unit--a{border-left:4px solid #9bbcff;background:linear-gradient(100deg,#263653,#1b2739 39%);box-shadow:inset 0 0 0 1px #9bbcff30}
      .wn-unit--b{border-left:4px solid #82d5b0;background:linear-gradient(100deg,#203b36,#1b2739 39%);box-shadow:inset 0 0 0 1px #82d5b030}
      .wn-unit--crit{border-left:4px solid #f4bf55;background:linear-gradient(100deg,#413015,#1b2739 43%);box-shadow:inset 0 0 0 1px #f4bf5540}
      .wn-tier{flex:none;border-radius:4px;padding:2px 5px;font-size:9px;font-weight:950;letter-spacing:.07em}
      .wn-tier--s{color:#211803;background:#f4bf55}.wn-tier--a{color:#0a1832;background:#9bbcff}.wn-tier--b{color:#09251c;background:#82d5b0}
      .wn-unit-ghost{position:absolute;z-index:0;left:65px;top:-28px;height:190px;width:auto;opacity:.16;pointer-events:none;mask-image:linear-gradient(90deg,#000 25%,transparent 95%)}
      .wn-unit:last-child{margin-bottom:0}.wn-portrait{position:relative;z-index:1;width:46px;height:46px;flex:none;display:grid;place-items:center;border:1px solid #ffffff2e;border-radius:9px;background:radial-gradient(circle at top left,color-mix(in srgb,var(--accent) 34%,#162035),#162035 75%);font-size:24px}
      .wn-portrait svg{width:29px;height:29px;max-width:29px;max-height:29px;fill:var(--accent)}
      .wn-portrait img{width:100%;height:100%;object-fit:cover;object-position:center top}
      .wn-unit-body{position:relative;z-index:1;flex:1;min-width:0}.wn-unit-head{display:flex;align-items:baseline;gap:6px;white-space:nowrap;min-width:0}.wn-rank{font-size:10px;color:var(--accent);font-weight:900}.wn-unit-head strong{overflow:hidden;text-overflow:ellipsis;font-size:14px;text-shadow:0 1px 9px #091321}.wn-match{font-size:10px;color:#a4b3c7;flex:none}
      .wn-attack{display:flex;gap:5px;align-items:center;margin-top:5px;min-width:0}.wn-attack-name{font-size:12px;font-weight:800;color:#eac483;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.wn-badge{font-size:8px;letter-spacing:.04em;color:var(--accent);border:1px solid color-mix(in srgb,var(--accent) 40%,transparent);border-radius:4px;padding:2px 4px;white-space:nowrap}
      .wn-unit-foot{display:flex;flex-wrap:wrap;gap:2px 10px;margin-top:5px;font-size:9px;color:#afbed0;letter-spacing:.01em}.wn-unit-foot b{color:#fff;margin-left:3px}
      .wn-book{font-size:9px;color:var(--accent);font-weight:800;white-space:nowrap;margin-left:5px}
      .wn-book-rate{font-size:9px;color:#b9c9dc;margin-top:4px;font-weight:800;letter-spacing:.06em}
      .wn-crit-env{font-size:9px;color:#f4d47d;margin-top:4px;font-weight:800;letter-spacing:.05em}
      .wn-research{font-size:10px;line-height:1.45;color:#b9d8ff;margin-top:5px;padding:5px 7px;border:1px solid #79afff43;border-radius:5px;background:#10213a}
      .wn-support-badges{font-size:9px;color:#b8c8e7;margin-top:4px;line-height:1.4}
      .wn-goalie{font-size:9px;color:#9fb6cb;margin-top:3px}
      .wn-goal-profile{font-size:10px;color:#e7cc8b;margin-top:4px;font-weight:700}
      .wn-position-sog{font-size:10px;color:#9edac1;margin-top:5px}.wn-position-sog b{color:#e6fff3}
      .wn-record{position:relative;z-index:1;text-align:right;flex:none;min-width:66px;display:flex;flex-direction:column;line-height:1.1}.wn-record strong{font-size:21px;color:#fff}.wn-record span{color:var(--accent);font-size:12px;font-weight:900;margin-top:3px}.wn-record em{font-style:normal;color:#8092a9;font-size:8px;letter-spacing:.08em;margin-top:3px}
      .wn-details{position:relative;z-index:1;flex:0 0 100%;font-size:10px;color:#aebbd0;border-top:1px solid #ffffff14;padding-top:5px}.wn-details summary{cursor:pointer;color:var(--accent);font-weight:700}.wn-move-list{max-height:320px;overflow:auto;display:grid;gap:6px;margin-top:8px;padding-right:3px}
      .wn-form,.wn-form-empty{position:relative;z-index:1;flex:0 0 100%;font-size:10px;color:#b9c9dc;border-top:1px solid #ffffff14;padding-top:5px}.wn-form summary{cursor:pointer;color:#e4c892;font-weight:800}.wn-form p{margin:7px 0;color:#aebbd0}.wn-form-tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(102px,1fr));gap:5px;margin:8px 0}.wn-form-tiles>div{border:1px solid #ffffff1d;border-radius:5px;padding:6px;background:#0b1629a8}.wn-form-tiles small{display:block;color:#98adc6;font-size:8px;letter-spacing:.07em}.wn-form-tiles strong{display:block;color:#fff;font-size:11px;margin-top:3px}.wn-form-meta{margin:4px 0}.wn-form-strip{display:flex;flex-wrap:wrap;gap:4px;margin:8px 0}.wn-form-strip span{border-radius:4px;padding:3px 5px;font-weight:800;text-align:center;min-width:36px}.wn-form-strip span b{display:block}.wn-form-strip span small{display:block;font-size:8px;font-weight:600;margin-top:2px;opacity:.85}.wn-form-strip .hit{background:#1d6145;color:#c4ffde}.wn-form-strip .miss{background:#603036;color:#ffd7d7}
      .wn-move-entry{border:1px solid #ffffff20;border-radius:6px;background:#0b1629e8;padding:7px}.wn-move-title{display:flex;align-items:center;justify-content:space-between;gap:8px}.wn-move-title strong{font-size:11px;color:#f1e4ca}.wn-move-title em{font-size:8px;font-style:normal;color:var(--accent);text-align:right}.wn-move-record{font-size:10px;font-weight:800;color:#fff;margin-top:3px}.wn-move-record span{color:#b7c7df;margin-left:5px}.wn-move-rule{font-size:9px;color:#b6c5da;overflow-wrap:anywhere;margin-top:4px}
      .wn-empty{padding:26px 12px;text-align:center;color:#aebbd0;font-size:12px}
      .wn-board--compact .wn-grid{grid-template-columns:1fr}
      .wn-board--compact .wn-lane-head{display:none}
      @media(max-width:1050px){.wn-grid{grid-template-columns:1fr}}
      @media(max-width:540px){.wn-unit{gap:7px;padding:8px}.wn-unit-ghost{left:45px;opacity:.12}.wn-portrait{width:34px;height:34px}.wn-portrait svg{width:23px;height:23px}.wn-record{min-width:56px}.wn-record strong{font-size:17px}.wn-match{display:none}}
    </style>"""
    goal_count = len(boards.get("Carry", []))
    other_count = total - goal_count
    hero_text = ("Goal cards show named 2026–27 attacks. Points 0.5 cards show current-season Base, Heavy, or Special screens; "
                 "older Points moves remain in card details. Other featured props require a Green baseline and a 50%+ historical move. "
                 "Compare records with posted book break-even; past rates are not tonight's probabilities."
                 if goal_count else
                 "Points 0.5 cards show current-season Base, Heavy, or Special screens. Other featured cards require a Green baseline and a 50%+ historical move. "
                 "Compare past results with the posted book break-even percentage.")
    hero_foot = (f"⚔ {goal_count} NAMED GOAL ATTACKS · {other_count} FEATURED OTHER PROP ENTRIES"
                 if goal_count else f"⚔ {total} FEATURED PLAYER PROP ENTRIES ACROSS FOUR CLASSES")
    hero = f"""<div class="wn-hero"><span class="wn-eyebrow">WARLORDS OF THE NIGHT · 2026</span>
      <h1>THE NIGHT RAID</h1><p>{_h(hero_text)}</p>
      <div class="wn-hero-foot">{_h(hero_foot)} · HISTORICAL MOVE RATE IS NOT A FORECAST</div></div>"""
    board_class = "wn-board" if show_hero else "wn-board wn-board--compact"
    return styles + f'<div class="{board_class}">' + (hero if show_hero else "") + '<div class="wn-grid">' + ''.join(lanes) + '</div></div>'

