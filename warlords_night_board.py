"""Read-only class board built from the current tracker and frozen move rules."""

from __future__ import annotations

import math
from html import escape

import pandas as pd

from warlord_moves_2026 import fired_moves


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
            return line, odds, book
    return None, None, None


def _move_rank(move):
    wins, picks = int(move["wins"]), int(move["picks"])
    later_wins, later_picks = int(move["later_wins"]), int(move["later_picks"])
    return (wins / picks if picks else -1, picks,
            later_wins / later_picks if later_picks else -1)


def rank_warlords(frame: pd.DataFrame) -> dict[str, list[dict]]:
    """Best fired historical move per player and class on this filtered slate."""
    boards = {role: {} for role, *_ in CLASSES}
    if frame.empty:
        return {role: [] for role in boards}
    for row in frame.to_dict("records"):
        if not _value(row, "Player"):
            continue
        active = fired_moves(row)
        for role, market, _, _ in CLASSES:
            moves = active[market]
            if not moves:
                continue
            attacks = [move for move in moves if move["kind"] != "STANCE"]
            move = max(attacks or moves, key=_move_rank)
            line, odds, book = _line_price(row, market)
            player = str(row["Player"]).strip()
            team = str(_value(row, "Team") or "").strip()
            key = (team.casefold(), player.casefold())
            candidate = {
                "player": player, "team": team,
                "opponent": str(_value(row, "Opp") or "").strip(),
                "game": str(_value(row, "Game") or "").strip(),
                "time": str(_value(row, "Time") or "").strip(),
                "market": market, "line": line, "odds": odds, "book": book,
                "move": move, "move_count": len(moves),
            }
            previous = boards[role].get(key)
            if previous is None or _move_rank(move) > _move_rank(previous["move"]):
                boards[role][key] = candidate
    return {role: sorted(players.values(),
                         key=lambda card: (_move_rank(card["move"]), card["player"]),
                         reverse=True)
            for role, players in boards.items()}


def _h(value):
    return escape(str(value), quote=True)


def _odds(value):
    if value is None:
        return "Odds pending"
    return f"{value:+.0f}" if value > 0 else f"{value:.0f}"


def render_warlords(boards: dict[str, list[dict]], limit: int = 5, icon_loader=None) -> str:
    """Safe HTML for four responsive MMO class lanes."""
    lanes = []
    for role, market, symbol, color in CLASSES:
        cards = boards.get(role, [])
        class_icon_file = {"Carry": "role_carry.svg", "Support": "role_support.svg",
                           "Tank": "role_tank.svg", "Jungle": "role_jungle.svg"}[role]
        class_icon = icon_loader(class_icon_file) if icon_loader else ""
        content = []
        for rank, card in enumerate(cards[:limit], 1):
            move = card["move"]
            wins, picks = int(move["wins"]), int(move["picks"])
            late_wins, late_picks = int(move["later_wins"]), int(move["later_picks"])
            pct = 100 * wins / picks
            late_pct = 100 * late_wins / late_picks if late_picks else 0
            status = "TRACK" if move.get("track") else "EXPLORATORY" if move.get("experimental") or move["kind"] == "LAB CRIT" else "MOVE"
            if picks < 30:
                status += " · SMALL SAMPLE"
            line = f"Over {card['line']:g} {market}" if card["line"] is not None else market
            match = card["game"] or (f"{card['team']} vs {card['opponent']}" if card["opponent"] else card["team"])
            rule = move.get("rule", move.get("condition", ""))
            move_icon = icon_loader(move.get("icon", "")) if icon_loader else ""
            content.append(f"""
              <article class="wn-card">
                <div class="wn-card-top"><span class="wn-rank">#{rank:02d}</span><span class="wn-status">{_h(status)}</span></div>
                <div class="wn-battle"><span class="wn-avatar">{move_icon or symbol}</span><span class="wn-arrow">━━➤</span><span class="wn-tower">🏰</span></div>
                <div class="wn-player">{_h(card['player'])}</div>
                <div class="wn-match">{_h(match)}{(' · ' + _h(card['time'])) if card['time'] else ''}</div>
                <div class="wn-move-type">{_h(move['kind'])}</div>
                <div class="wn-move">{_h(move['name'])}</div>
                <div class="wn-rule">{_h(rule)}</div>
                <div class="wn-meter"><span style="width:{pct:.1f}%"></span></div>
                <div class="wn-score"><strong>{pct:.1f}%</strong><span>{wins}/{picks} historical</span></div>
                <div class="wn-later">Later: {late_wins}/{late_picks} · {late_pct:.1f}%</div>
                <div class="wn-line">{_h(line)} <span>{_h(_odds(card['odds']))}{(' · ' + _h(card['book'])) if card['book'] else ''}</span></div>
              </article>""")
        if not content:
            content = ['<div class="wn-empty">No priced player has a fired move in this class yet.</div>']
        lanes.append(f"""<section class="wn-lane" style="--accent:{color}">
          <div class="wn-lane-head"><span class="wn-class-icon">{class_icon or symbol}</span>
            <div><div class="wn-role">{role}</div><div class="wn-market">{market} · {len(cards)} ready</div></div>
          </div>{''.join(content)}</section>""")
    return """<style>
    .wn-board,.wn-board *{box-sizing:border-box}.wn-board{font-family:Inter,system-ui,sans-serif;color:#eaf2ff}
    .wn-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;align-items:start}
    .wn-lane{min-width:0;background:#111c30;border:1px solid #35445d;border-top:3px solid var(--accent);border-radius:16px;padding:11px}
    .wn-lane-head{display:flex;gap:10px;align-items:center;padding:5px 3px 14px}
    .wn-class-icon{display:grid;place-items:center;background:#233149;border-radius:12px;width:39px;height:39px;font-size:22px}
    .wn-class-icon svg{width:28px;height:28px;max-width:28px;max-height:28px;fill:var(--accent)}
    .wn-role{font-size:18px;font-weight:900;color:var(--accent)}.wn-market{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#aab9ce}
    .wn-card{background:linear-gradient(160deg,#233149,#182538);border:1px solid #40506a;border-radius:13px;padding:13px;margin-bottom:11px;box-shadow:0 8px 20px #070e1c55}
    .wn-card-top,.wn-score,.wn-line{display:flex;justify-content:space-between;gap:7px;align-items:center}
    .wn-rank{font-weight:900;color:var(--accent);font-size:13px}.wn-status{font-size:9px;letter-spacing:.04em;color:#f3dc99;text-align:right}
    .wn-battle{display:flex;align-items:center;justify-content:space-between;margin:10px 0;color:var(--accent)}
    .wn-avatar,.wn-tower{display:grid;place-items:center;width:45px;height:45px;border-radius:11px;background:#354461;font-size:25px}
    .wn-tower{background:#462e3a}.wn-arrow{font-size:18px;letter-spacing:4px}
    .wn-avatar svg{width:31px;height:31px;max-width:31px;max-height:31px;fill:var(--accent)}
    .wn-player{font-size:17px;font-weight:900;line-height:1.2}.wn-match{font-size:11px;color:#aebed3;margin:5px 0 11px}
    .wn-move-type{font-size:9px;letter-spacing:.12em;color:var(--accent);font-weight:800}.wn-move{font-weight:800;margin:3px 0;font-size:14px}
    .wn-rule{font-size:10px;color:#afc0d3;min-height:30px;overflow-wrap:anywhere}
    .wn-meter{height:7px;border-radius:10px;background:#44506a;margin:10px 0 5px;overflow:hidden}.wn-meter span{display:block;height:100%;border-radius:10px;background:var(--accent)}
    .wn-score strong{font-size:24px;color:#fff}.wn-score span,.wn-later{font-size:11px;color:#c3d0e0}.wn-later{margin-top:3px}
    .wn-line{border-top:1px solid #43516a;margin-top:10px;padding-top:10px;font-size:11px;font-weight:800}.wn-line span{font-weight:500;color:#b8c9dc;text-align:right}
    .wn-empty{padding:30px 12px;color:#aebed3;font-size:13px;text-align:center}
    @media(max-width:1200px){.wn-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
    @media(max-width:650px){.wn-grid{grid-template-columns:1fr}}
    </style><div class="wn-board"><div class="wn-grid">""" + "".join(lanes) + "</div></div>"
