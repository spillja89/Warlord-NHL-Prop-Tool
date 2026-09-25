"""Read-only class board built from the current tracker and frozen move rules."""

from __future__ import annotations

import math
import base64
from functools import lru_cache
from html import escape
from pathlib import Path

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


@lru_cache(maxsize=4)
def _character_uri(role: str) -> str:
    """Inline small artwork so class cards render behind Cloud's app proxy."""
    image = Path(__file__).parent / "static" / "characters" / f"{role.lower()}-gorilla.webp"
    if not image.is_file():
        return ""
    return "data:image/webp;base64," + base64.b64encode(image.read_bytes()).decode("ascii")


def render_warlords(boards: dict[str, list[dict]], limit: int = 5, icon_loader=None) -> str:
    """Compact, escaped HTML for four responsive class lanes."""
    descriptions = {"Carry": "Finish the fight", "Support": "Set the play",
                    "Tank": "Hold the line", "Jungle": "Control the lanes"}
    lanes = []
    for role, market, symbol, color in CLASSES:
        cards = boards.get(role, [])
        character_uri = _character_uri(role)
        class_icon = symbol
        units = []
        for rank, card in enumerate(cards[:limit], 1):
            move = card["move"]
            wins, picks = int(move["wins"]), int(move["picks"])
            late_wins, late_picks = int(move["later_wins"]), int(move["later_picks"])
            pct = 100 * wins / picks
            late_pct = 100 * late_wins / late_picks if late_picks else 0
            status = ("TRACK" if move.get("track") else "LAB" if move.get("experimental")
                      or move["kind"] == "LAB CRIT" else move["kind"])
            sample = " · SMALL SAMPLE" if picks < 30 else ""
            line = f"OVER {card['line']:g} {market.upper()}" if card["line"] is not None else market.upper()
            matchup = card["game"] or card["team"]
            rule = move.get("rule", move.get("condition", ""))
            portrait = (f'<img src="{character_uri}" alt="" />' if character_uri else symbol)
            price = _odds(card["odds"])
            units.append(f"""<article class="wn-unit">
              <div class="wn-portrait" aria-hidden="true">{portrait}</div>
              <div class="wn-unit-body">
                <div class="wn-unit-head"><span class="wn-rank">{rank:02d}</span><strong>{_h(card['player'])}</strong><span class="wn-match">{_h(matchup)}</span></div>
                <div class="wn-attack"><span class="wn-attack-name">{_h(move['name'])}</span><span class="wn-badge">{_h(status + sample)}</span></div>
                <div class="wn-unit-foot"><span>{_h(line)} <b>{_h(price)}</b></span><span>LATER {late_wins}/{late_picks} · {late_pct:.1f}%</span></div>
                <details class="wn-details"><summary>Move conditions</summary><span>{_h(rule)}</span></details>
              </div>
              <div class="wn-record"><strong>{pct:.1f}%</strong><span>{wins}/{picks}</span><em>HISTORICAL</em></div>
            </article>""")
        if not units:
            units = ['<div class="wn-empty">No move has fired on a posted line yet.</div>']
        backdrop = f'<img class="wn-gorilla" src="{character_uri}" alt="" />' if character_uri else ""
        lanes.append(f"""<section class="wn-lane wn-lane--{role.lower()}" style="--accent:{color}">
          <header class="wn-lane-head">{backdrop}<div class="wn-class-icon" aria-hidden="true">{class_icon}</div>
            <div class="wn-class-text"><span class="wn-kicker">{_h(descriptions[role])}</span><h2>{_h(role)}</h2></div>
            <div class="wn-count"><strong>{len(cards)}</strong><span>READY</span></div></header>
          <div class="wn-lane-sub">{_h(market.upper())} <span>✦</span> TOP MOVE PER PLAYER <span>✦</span> ⚔ AGAINST THE BOOKS</div>
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
      .wn-gorilla{position:absolute;z-index:-1;right:8px;top:-40px;height:230px;width:auto;opacity:.28;pointer-events:none;mask-image:linear-gradient(90deg,transparent,#000 35%)}
      .wn-class-icon{width:42px;height:42px;flex:none;display:grid;place-items:center;border:1px solid color-mix(in srgb,var(--accent) 50%,transparent);border-radius:9px;background:#0a1425e8;font-size:25px}
      .wn-class-icon svg{width:29px;height:29px;max-width:29px;max-height:29px;fill:var(--accent)}
      .wn-class-text{flex:1}.wn-kicker{font-size:10px;text-transform:uppercase;letter-spacing:.15em;color:#c5c6d3}.wn-class-text h2{font-size:23px;line-height:1;margin:3px 0 0;color:var(--accent);font-weight:950}
      .wn-count{display:flex;flex-direction:column;align-items:center;color:var(--accent);line-height:1;background:#0a1425b8;border:1px solid #ffffff16;border-radius:8px;padding:6px 8px}.wn-count strong{font-size:25px}.wn-count span{font-size:9px;letter-spacing:.12em;margin-top:3px}
      .wn-lane-sub{font-size:9px;font-weight:800;letter-spacing:.12em;color:#8898b2;padding:8px 16px;background:#0d1625}.wn-lane-sub span{color:var(--accent);padding:0 4px}
      .wn-units{padding:8px}.wn-unit{display:flex;gap:10px;min-height:104px;align-items:center;background:#1b2739;border:1px solid #3b4b64;border-left:3px solid var(--accent);border-radius:9px;padding:10px;margin-bottom:7px}
      .wn-unit:last-child{margin-bottom:0}.wn-portrait{width:46px;height:46px;flex:none;display:grid;place-items:center;border:1px solid #ffffff2e;border-radius:9px;background:radial-gradient(circle at top left,color-mix(in srgb,var(--accent) 34%,#162035),#162035 75%);font-size:24px}
      .wn-portrait svg{width:29px;height:29px;max-width:29px;max-height:29px;fill:var(--accent)}
      .wn-portrait img{width:100%;height:100%;object-fit:cover;object-position:center top}
      .wn-unit-body{flex:1;min-width:0}.wn-unit-head{display:flex;align-items:baseline;gap:6px;white-space:nowrap;min-width:0}.wn-rank{font-size:10px;color:var(--accent);font-weight:900}.wn-unit-head strong{overflow:hidden;text-overflow:ellipsis;font-size:14px}.wn-match{font-size:10px;color:#a4b3c7;flex:none}
      .wn-attack{display:flex;gap:5px;align-items:center;margin-top:5px;min-width:0}.wn-attack-name{font-size:12px;font-weight:800;color:#eac483;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.wn-badge{font-size:8px;letter-spacing:.04em;color:var(--accent);border:1px solid color-mix(in srgb,var(--accent) 40%,transparent);border-radius:4px;padding:2px 4px;white-space:nowrap}
      .wn-unit-foot{display:flex;flex-wrap:wrap;gap:2px 10px;margin-top:5px;font-size:9px;color:#afbed0;letter-spacing:.01em}.wn-unit-foot b{color:#fff;margin-left:3px}
      .wn-record{text-align:right;flex:none;min-width:66px;display:flex;flex-direction:column;line-height:1.1}.wn-record strong{font-size:21px;color:#fff}.wn-record span{color:var(--accent);font-size:12px;font-weight:900;margin-top:3px}.wn-record em{font-style:normal;color:#8092a9;font-size:8px;letter-spacing:.08em;margin-top:3px}
      .wn-details{font-size:9px;color:#96a8bf;margin-top:4px}.wn-details summary{cursor:pointer;color:#9eadbf}.wn-details span{display:block;overflow-wrap:anywhere;margin-top:3px}
      .wn-empty{padding:26px 12px;text-align:center;color:#aebbd0;font-size:12px}
      @media(max-width:1050px){.wn-grid{grid-template-columns:1fr}}
      @media(max-width:540px){.wn-unit{gap:7px;padding:8px}.wn-portrait{width:34px;height:34px}.wn-portrait svg{width:23px;height:23px}.wn-record{min-width:56px}.wn-record strong{font-size:17px}.wn-match{display:none}}
    </style>"""
    hero = f"""<div class="wn-hero"><span class="wn-eyebrow">WARLORDS OF THE NIGHT · 2026</span>
      <h1>THE NIGHT RAID</h1><p>Choose your class. Every card shows the strongest move this player can fire.</p>
      <div class="wn-hero-foot">⚔ {total} READY PLAYERS ACROSS FOUR CLASSES · RECORDS ARE HISTORICAL</div></div>"""
    return styles + '<div class="wn-board">' + hero + '<div class="wn-grid">' + ''.join(lanes) + '</div></div>'
