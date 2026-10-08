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
    return (wins / picks if picks else -1, picks,
            later_wins / later_picks if later_picks else -1)


def _baseline_rule(row, market, line):
    matrix = str(row.get(f"Matrix_{market}", "")).strip().casefold()
    if matrix not in {"green", "🟢"}:
        return None
    if market == "Goal" and line == 0.5 and _number(row.get("Conf_Points")) is not None:
        return "Green + Goal 0.5 + Conf_Points ≥ 84" if _number(row.get("Conf_Points")) >= 84 else None
    if market == "Assists" and line == 0.5:
        return "Green + Assists 0.5"
    if market == "Points" and line in (0.5, 1.5):
        threshold = 80 if line == 0.5 else 75
        if _number(row.get("Conf_Points")) is not None and _number(row.get("Conf_Points")) >= threshold:
            return f"Green + Points {line:g} + Conf_Points ≥ {threshold}"
    if market == "SOG" and line in (2.5, 3.5):
        if _number(row.get("Conf_SOG")) is not None and _number(row.get("Conf_SOG")) >= 75:
            return f"Green + SOG {line:g} + Conf_SOG ≥ 75"
    return None


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
                attacks = [move for move in moves if move["kind"] != "STANCE"]
                move = max(attacks or moves, key=_move_rank)
            player = str(row["Player"]).strip()
            team = str(_value(row, "Team") or "").strip()
            key = (team.casefold(), player.casefold())
            candidate = {
                "player": player, "team": team,
                "opponent": str(_value(row, "Opp") or "").strip(),
                "game": str(_value(row, "Game") or "").strip(),
                "time": str(_value(row, "Time") or _value(row, "StartTimeLocal") or "").strip(),
                "market": market, "line": line, "odds": odds, "book": book,
                "form_log": _value(row, "Form_Log"),
                "form_season": _value(row, "Form_Season"),
                "model_prob": model_prob, "book_prob": book_prob, "price_gap": price_gap,
                "goalie": str(_value(row, "Opp_Goalie") or ""),
                "goalie_status": str(_value(row, "Opp_Goalie_Status") or "Unknown"),
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
            moves = active[market]
            attacks = [move for move in moves if move["kind"] != "STANCE"]
            best = max(attacks or moves, key=_move_rank) if moves else None
            baseline = None if watch else _baseline_rule(row, market, line)
            confidence = _number(row.get(f"Conf_{market}"))
            matrix = str(_value(row, f"Matrix_{market}") or "Unknown").strip()
            candidate = {
                "player": player, "team": team,
                "opponent": str(_value(row, "Opp") or "").strip(),
                "game": str(_value(row, "Game") or "").strip(),
                "time": str(_value(row, "Time") or _value(row, "StartTimeLocal") or "").strip(),
                "market": market, "line": line, "odds": odds, "book": book,
                "form_log": _value(row, "Form_Log"),
                "form_season": _value(row, "Form_Season"),
                "model_prob": model_prob, "book_prob": book_prob, "price_gap": price_gap,
                "confidence": confidence, "matrix": matrix,
                "baseline_rule": baseline, "baseline_only": bool(baseline and not best),
                "goalie": str(_value(row, "Opp_Goalie") or ""),
                "goalie_status": str(_value(row, "Opp_Goalie_Status") or "Unknown"),
                "move": best,
                "moves": sorted(moves, key=lambda item: (
                    item["kind"] != "STANCE", _move_rank(item)), reverse=True),
                "move_count": len(moves),
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
            qualifying = [move for move in card["moves"] if int(move["picks"]) > 0
                          and int(move["wins"]) / int(move["picks"]) >= 0.5]
            if not qualifying:
                continue
            selected.append({**card, "move": max(qualifying, key=_move_rank)})
        featured[role] = sorted(selected, key=lambda card: (
            _move_rank(card["move"]), card.get("confidence") or -1,
            card["player"].casefold()), reverse=True)
    return featured


def baseline_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Show each priced baseline and where the other posted lines drop out.

    These are the frozen class entry rules. A baseline can qualify without a
    stronger named move, so this audit does not change the move kit or grading.
    """
    specs = (
        ("Goals", "Goal", (0.5,), "Matrix_Goal", "Conf_Points"),
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
            named = any(move["kind"] != "STANCE" for move in fired_moves(row)[market])
            if named:
                for counts in (totals[label], bucket):
                    counts["Named move"].add(identity)
            stats_source = str(_value(row, "Opp_Goalie_Source") or "none")
            stats_label = {
                "dailyfaceoff_name_and_stats": "Last season: matched goalie",
                "dailyfaceoff_name_only": "Starter named; stats unavailable",
                "moneypuck_team_proxy_unconfirmed": "Last season: team proxy",
                "moneypuck_team_proxy": "Last season: team proxy",
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


def render_power_play_form(card):
    """Show the same dated player form card on the separate PPP scouting page."""
    portrait = _character_uri("Support")
    image = f'<img src="{portrait}" alt="" />' if portrait else "🪄"
    price = _odds(card.get("odds"))
    form = _form_html({**card, "market": "PPP", "line": 0.5}, expanded=True)
    return f'''<style>
      .ppp-card,.ppp-card *{{box-sizing:border-box}}
      .ppp-card{{font-family:Inter,system-ui,sans-serif;color:#edf1f8;background:#1b2739;border:1px solid #4d416e;border-left:3px solid #b692ff;border-radius:10px;padding:14px;margin:12px 0}}
      .ppp-head{{display:flex;align-items:center;gap:12px;margin-bottom:8px}}
      .ppp-portrait{{width:50px;height:50px;flex:none;overflow:hidden;border-radius:9px;border:1px solid #8d72c4;background:#172039}}
      .ppp-portrait img{{width:100%;height:100%;object-fit:cover;object-position:center top}}
      .ppp-title{{min-width:0;flex:1}}.ppp-title strong{{display:block;font-size:17px;color:#fff}}.ppp-title span{{font-size:11px;color:#b9c9dc}}
      .ppp-price{{text-align:right;white-space:nowrap;color:#e9d5ff;font-size:20px;font-weight:900}}.ppp-price small{{display:block;color:#aebbd0;font-size:10px;font-weight:600}}
      .wn-form,.wn-form-empty{{color:#b9c9dc;border-top:1px solid #ffffff24;padding-top:8px;font-size:12px}}.wn-form summary{{cursor:pointer;color:#e4c892;font-weight:800}}.wn-form p{{margin:7px 0;color:#aebbd0}}
      .wn-form-tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:5px;margin:8px 0}}.wn-form-tiles>div{{border:1px solid #ffffff1d;border-radius:5px;padding:7px;background:#0b1629a8}}.wn-form-tiles small{{display:block;color:#98adc6;font-size:9px}}.wn-form-tiles strong{{display:block;color:#fff;font-size:12px;margin-top:3px}}
      .wn-form-meta{{margin:4px 0}}.wn-form-strip{{display:flex;flex-wrap:wrap;gap:5px;margin:8px 0}}.wn-form-strip span{{border-radius:4px;padding:5px 7px;text-align:center;min-width:42px}}.wn-form-strip span b,.wn-form-strip span small{{display:block}}.wn-form-strip span small{{font-size:9px;opacity:.85}}.wn-form-strip .hit{{background:#1d6145;color:#c4ffde}}.wn-form-strip .miss{{background:#603036;color:#ffd7d7}}
    </style><section class="ppp-card"><div class="ppp-head"><div class="ppp-portrait">{image}</div>
      <div class="ppp-title"><strong>{_h(card.get("player") or "Player")}</strong><span>{_h(card.get("game") or "")} · Over 0.5 power-play points · {_h(card.get("book") or "Book pending")}</span></div>
      <div class="ppp-price">{_h(price)}<small>BEST FEED PRICE</small></div></div>{form}</section>'''


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
            baseline_only = bool(card.get("baseline_only"))
            wins, picks = (int(move["wins"]), int(move["picks"])) if move else (0, 0)
            late_wins, late_picks = ((int(move["later_wins"]), int(move["later_picks"]))
                                     if move else (0, 0))
            pct = 100 * wins / picks if picks else 0
            late_pct = 100 * late_wins / late_picks if late_picks else 0
            status = (("BASELINE" if "confidence" in card else "BASELINE ONLY") if baseline_only
                      else "OPEN POOL" if not move else
                      "TRACK" if move.get("track") else "LAB" if move.get("experimental")
                      or move["kind"] == "LAB CRIT" else move["kind"])
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
            goalie_name = str(card.get("goalie") or "").strip()
            goalie_status = str(card.get("goalie_status") or "Unknown").strip()
            goalie_note = (f'<div class="wn-goalie">Opp goalie: {_h(goalie_name)} · {_h(goalie_status)}</div>'
                           if goalie_name else '<div class="wn-goalie">Opp goalie: unknown</div>')
            fired = card.get("moves") or ([] if not move or baseline_only else [move])
            move_rows = []
            for fired_move in fired:
                fired_wins, fired_picks = int(fired_move["wins"]), int(fired_move["picks"])
                fired_late_wins = int(fired_move["later_wins"])
                fired_late_picks = int(fired_move["later_picks"])
                fired_pct = 100 * fired_wins / fired_picks if fired_picks else 0
                fired_late_pct = 100 * fired_late_wins / fired_late_picks if fired_late_picks else 0
                fired_rule = fired_move.get("rule", fired_move.get("condition", ""))
                fired_label = ("TRACK" if fired_move.get("track") else
                               "LAB" if fired_move.get("experimental") else fired_move["kind"])
                move_rows.append(f'''<div class="wn-move-entry">
                  <div class="wn-move-title"><strong>{_h(fired_move["name"])}</strong><em>{_h(fired_label)}</em></div>
                  <div class="wn-move-record">{fired_wins}/{fired_picks} · {fired_pct:.1f}% <span>Later {fired_late_wins}/{fired_late_picks} · {fired_late_pct:.1f}%</span></div>
                  <div class="wn-move-rule">{_h(fired_rule)}</div>
                </div>''')
            later_note = (f"MOVE {wins}/{picks} · LATER {late_wins}/{late_picks} ({late_pct:.1f}%)"
                          if picks else "No tested move · posted line" if "confidence" in card
                          else "Baseline screen · no upgraded move")
            confidence = card.get("confidence")
            confidence_note = (f" · CONF {confidence:.0f} {_h(card.get('matrix') or 'Unknown')}"
                               if confidence is not None else "")
            record_html = (f'<div class="wn-record"><strong>{pct:.1f}%</strong><span>{wins}/{picks}</span><em>HISTORICAL MOVE</em></div>'
                           if picks else '<div class="wn-record"><strong>BASE</strong><em>SCREEN ONLY</em></div>')
            details_html = (f'<details class="wn-details"><summary>Full fired move list ({len(fired)})</summary>'
                            f'<div class="wn-move-list">{"".join(move_rows)}</div></details>' if fired else
                            f'<div class="wn-details">{_h(card.get("baseline_rule") or (move or {}).get("rule") or "No tested move fired")}</div>')
            move_name = move["name"] if move else ("Green Baseline" if baseline_only else "Scouting pool")
            units.append(f"""<article class="wn-unit">
              {name_backdrop}
              <div class="wn-portrait" aria-hidden="true">{portrait}</div>
              <div class="wn-unit-body">
                <div class="wn-unit-head"><span class="wn-rank">{rank:02d}</span><strong>{_h(card['player'])}</strong><span class="wn-match">{_h(matchup)}</span></div>
                <div class="wn-attack"><span class="wn-attack-name">{_h(move_name)}</span><span class="wn-badge">{_h(status + sample)}</span></div>
                <div class="wn-unit-foot"><span>{_h(line)} <b>{_h(price)}</b>{book_note}</span><span>{_h(later_note)}{confidence_note}</span></div>
                {book_rate_note}
                {goalie_note}
              </div>
              {record_html}
              {_form_html(card)}
              {details_html}
            </article>""")
        if not units:
            units = ['<div class="wn-empty">No posted player line and price for this class yet.</div>']
        backdrop = f'<img class="wn-gorilla" src="{character_uri}" alt="" />' if character_uri else ""
        count_label = "FEATURED" if cards and "confidence" in cards[0] else "PRICED"
        lanes.append(f"""<section class="wn-lane wn-lane--{role.lower()}" style="--accent:{color}">
          <header class="wn-lane-head">{backdrop}<div class="wn-class-icon" aria-hidden="true">{class_icon}</div>
            <div class="wn-class-text"><span class="wn-kicker">{_h(descriptions[role])}</span><h2>{_h(role)}</h2></div>
            <div class="wn-count"><strong>{len(cards)}</strong><span>{count_label}</span></div></header>
          <div class="wn-lane-sub">{_h(market.upper())} <span>✦</span> HISTORICAL MOVE % <span>✦</span> BOOK BREAK-EVEN % FROM POSTED ODDS</div>
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
      .wn-unit-ghost{position:absolute;z-index:0;left:65px;top:-28px;height:190px;width:auto;opacity:.16;pointer-events:none;mask-image:linear-gradient(90deg,#000 25%,transparent 95%)}
      .wn-unit:last-child{margin-bottom:0}.wn-portrait{position:relative;z-index:1;width:46px;height:46px;flex:none;display:grid;place-items:center;border:1px solid #ffffff2e;border-radius:9px;background:radial-gradient(circle at top left,color-mix(in srgb,var(--accent) 34%,#162035),#162035 75%);font-size:24px}
      .wn-portrait svg{width:29px;height:29px;max-width:29px;max-height:29px;fill:var(--accent)}
      .wn-portrait img{width:100%;height:100%;object-fit:cover;object-position:center top}
      .wn-unit-body{position:relative;z-index:1;flex:1;min-width:0}.wn-unit-head{display:flex;align-items:baseline;gap:6px;white-space:nowrap;min-width:0}.wn-rank{font-size:10px;color:var(--accent);font-weight:900}.wn-unit-head strong{overflow:hidden;text-overflow:ellipsis;font-size:14px;text-shadow:0 1px 9px #091321}.wn-match{font-size:10px;color:#a4b3c7;flex:none}
      .wn-attack{display:flex;gap:5px;align-items:center;margin-top:5px;min-width:0}.wn-attack-name{font-size:12px;font-weight:800;color:#eac483;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.wn-badge{font-size:8px;letter-spacing:.04em;color:var(--accent);border:1px solid color-mix(in srgb,var(--accent) 40%,transparent);border-radius:4px;padding:2px 4px;white-space:nowrap}
      .wn-unit-foot{display:flex;flex-wrap:wrap;gap:2px 10px;margin-top:5px;font-size:9px;color:#afbed0;letter-spacing:.01em}.wn-unit-foot b{color:#fff;margin-left:3px}
      .wn-book{font-size:9px;color:var(--accent);font-weight:800;white-space:nowrap;margin-left:5px}
      .wn-book-rate{font-size:9px;color:#b9c9dc;margin-top:4px;font-weight:800;letter-spacing:.06em}
      .wn-goalie{font-size:9px;color:#9fb6cb;margin-top:3px}
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
    hero = f"""<div class="wn-hero"><span class="wn-eyebrow">WARLORDS OF THE NIGHT · 2026</span>
      <h1>THE NIGHT RAID</h1><p>Featured cards require a Green baseline and a fired move with a historical hit rate of at least 50%. Compare each move's past results with the break-even percentage of the posted book price.</p>
      <div class="wn-hero-foot">⚔ {total} FEATURED PLAYER PROP ENTRIES ACROSS FOUR CLASSES · HISTORICAL MOVE RATE IS NOT A FORECAST</div></div>"""
    board_class = "wn-board" if show_hero else "wn-board wn-board--compact"
    return styles + f'<div class="{board_class}">' + (hero if show_hero else "") + '<div class="wn-grid">' + ''.join(lanes) + '</div></div>'
