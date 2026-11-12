"""Read-only PPP beta overlays for frozen Assist profiles; not a new move gate."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from html import escape
from pathlib import Path

from assist_moves_current import assists_moves

VERSION = "pp-assist-overlap-beta-topmove-2026-11-07-v2"
RECORD_FILE = Path(__file__).resolve().parent / "pp_assist_beta_records.json"


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def _records():
    return json.loads(RECORD_FILE.read_text(encoding="utf-8"))


def assist_ppp_beta(row):
    """Match priced PPP0.5 + on-ice xG>=.66 + each unchanged Assist route.

    Layer1, PP1 and the four old player checks are not added to this parent.
    Missing inputs never count as a failure/zero or create a beta qualifier.
    """
    if row.get("Model_Stats_Season") != "2026-2027":
        return None
    xg = _number(row.get("PP_OnIce_xG_per_game"))
    if xg is None:
        rate = _number(row.get("PP_OnIce_xGF60"))
        minutes = _number(row.get("PP_TOI_per_game"))
        if rate is not None and minutes is not None and rate >= 0 and minutes >= 0:
            xg = round(rate * minutes / 60, 3)
    if xg is None or xg < .66:
        return None
    quotes = []
    for suffix in ("", "_1", "_2", "_3", "_4"):
        line = _number(row.get(f"BDL_PPP_Line{suffix}"))
        odds = _number(row.get(f"BDL_PPP_Odds{suffix}"))
        if line == .5 and odds is not None and odds != 0:
            book = row.get(f"BDL_PPP_Book{suffix}")
            quotes.append((odds, book if isinstance(book, str) else ""))
    if not quotes:
        return None
    odds, book = max(quotes, key=lambda item: item[0])
    archive = _records()
    names = {move["name"] for move in assists_moves(row)}
    matched = []
    for record in archive["routes"]:
        if record["name"] not in names:
            continue
        # Keep unknown inputs separate even when a production route has aliases.
        if any(_number(row.get(field)) is None for field, _, _ in record["conditions"]):
            continue
        if any(field.startswith(("Team_", "opp_")) for field, _, _ in record["conditions"]):
            if row.get("Team_Stats_Season") != "2026-2027":
                continue
        matched.append(dict(record))
    if not matched:
        return None
    matched.sort(key=_route_order)
    return {"version": VERSION, "xg": xg, "ppp_line": .5,
            "ppp_odds": odds, "ppp_book": book,
            "ppp_be": 100 / (1 + odds / 100) if odds > 0 else 100 * -odds / (100 - odds),
            "routes": matched, "sample_from": archive["sample_from"],
            "sample_through": archive["sample_through"]}


def _route_order(route):
    """PPP historical rate first; ties favor volume then the simpler route."""
    return (-route["hits"] / route["plays"], -route["plays"],
            len(route.get("conditions", [])), route["display_name"].casefold())


def top_beta_route(data):
    routes = (data or {}).get("routes") or []
    return min(routes, key=_route_order) if routes else None


def beta_summary(data):
    if not data or not data.get("routes"):
        return ""
    top = top_beta_route(data)
    remaining=len(data['routes'])-1
    names=top['display_name']+(f' · {remaining} other matches inside Inspect' if remaining else '')
    status = " · MONITOR" if top["plays"] < 30 else ""
    return (f'<span class="pp-assist-beta-pill" title="{escape(names, quote=True)}">'
            f'<span class="pp-assist-beta-label">⚡ PPP BETA · {escape(top["display_name"])}</span>'
            f'<span class="pp-assist-beta-record">PPP {top["hits"]}/{top["plays"]}'
            f' · {100 * top["hits"] / top["plays"]:.1f}% · sample BE {top["BE"]:.1f}%{status}'
            + (f' · +{remaining} other matches' if remaining else '') + '</span></span>')


def beta_details(data):
    if not data or not data.get("routes"):
        return ""
    entries = []
    for route in sorted(data["routes"], key=_route_order):
        status = "MONITOR · under 30 plays" if route["plays"] < 30 else "BETA SAMPLE · 30+ plays"
        returns = (f'<small>Saved-price 1u PPP replay · {route["units"]:+.2f}u'
                   f' · {route["roi"]:+.1f}% ROI</small>'
                   if route.get("units") is not None and route.get("roi") is not None else '')
        entries.append(
            '<div class="pp-assist-beta-route">'
            f'<b>{escape(route["display_name"])}</b><br>'
            f'PPP {route["hits"]}/{route["plays"]} · {100 * route["hits"] / route["plays"]:.1f}%'
            f' · sample PPP BE {route["BE"]:.1f}% · {route["players"]} players'
            f'{returns}<small>{status}</small></div>')
    top=top_beta_route(data)
    count=len(data['routes'])
    return ('<details class="pp-assist-beta-details"><summary>'
            '<b>PPP BETA · ASSIST ROUTE OVERLAP</b>'
            f'<span>{escape(top["display_name"])} · {top["hits"]}/{top["plays"]}'
            f' · {100*top["hits"]/top["plays"]:.1f}% · sample PPP BE {top["BE"]:.1f}%</span>'
            f'<small>View {count} matching route'+('s' if count!=1 else '')+'</small></summary>'
            '<div class="pp-assist-beta-body">'
            f'<p>PP on-ice xG/game {data["xg"]:.3f} (≥0.66) · '
            f'Over 0.5 PPP {data["ppp_odds"]:+.0f} · {escape(data["ppp_book"])}'
            f' · current PPP BE {data["ppp_be"]:.1f}%</p>'
            + ''.join(entries)
            + f'<p>2026–27 PPP results · {escape(data["sample_from"])} through '
            f'{escape(data["sample_through"])}. Overlapping frozen Assist routes; '
            'these records grade PPP, not assists. Research samples, not tonight’s probabilities.</p></div></details>')


BETA_CSS = """<style>
.pp-assist-beta-pill{display:inline-block;margin-top:5px;padding:4px 7px;border:1px solid #57cbd3;border-radius:5px;background:#103d48;color:#e6fdff;font:800 11px/1.5 system-ui,sans-serif;white-space:normal}
.pp-assist-beta-label,.pp-assist-beta-record{display:block;overflow-wrap:anywhere}.pp-assist-beta-label{color:#baf5ff}.pp-assist-beta-record{color:#f1fbff;font-size:12px;font-weight:650}
.pp-assist-beta-details{padding:9px 10px;border:1px solid #4b9db1;border-radius:7px;background:#102a3c;color:#e4f5ff;font:12px/1.6 system-ui,sans-serif;overflow-wrap:anywhere}
.pp-assist-beta-details p{margin:5px 0;color:#d4eaf5}.pp-assist-beta-details b{color:#c6f8ff}
.pp-assist-beta-details>summary{display:flex;flex-direction:column;gap:2px;cursor:pointer;list-style:none;line-height:1.4}
.pp-assist-beta-details>summary:before{content:'▸';color:#8de4f2;float:left}.pp-assist-beta-details[open]>summary:before{content:'▾'}
.pp-assist-beta-details>summary>span{color:#f1fbff;font-size:12px}.pp-assist-beta-details>summary>small{color:#9ed6e2;font-size:10px}
.pp-assist-beta-details[open]>.pp-assist-beta-body{padding-top:7px}
.pp-assist-beta-route{padding:7px 0;border-top:1px solid #36566b}.pp-assist-beta-route small{display:block;color:#d4def1;font-size:10px}
.pp-assist-beta-cell{min-width:230px;max-width:330px;white-space:normal}.pp-assist-beta-cell summary{cursor:pointer;list-style:none}.pp-assist-beta-cell .pp-assist-beta-details{margin-top:5px}
</style>"""
