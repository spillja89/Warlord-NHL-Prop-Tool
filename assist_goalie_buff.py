"""Optional Assist matchup display. Never changes moves, scores or card admission."""
from functools import lru_cache
from html import escape
from pathlib import Path
import json
import math

CUTOFF = 2.95


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def text(value, fallback=""):
    value = str(value or "").strip()
    return fallback if value.casefold() in {"", "nan", "none", "<na>"} else value


def identity(move):
    return str(move.get("kit_version")) + "|" + str(move.get("move_id") or move.get("name"))


@lru_cache(maxsize=1)
def records():
    try:
        data = json.loads(Path(__file__).with_name("assist_goalie_buff_records.json").read_text(encoding="utf-8"))
        data["by_key"] = {row["key"]: row for row in data["moves"]}
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def record_for(move):
    if not isinstance(move, dict):
        return None
    record = records().get("by_key", {}).get(identity(move))
    if (not record or record["name"] != move.get("name") or record["rule"] != move.get("rule")
            or (record["original_hits"], record["original_plays"]) != (move.get("wins"), move.get("picks"))):
        return None
    return record


def context(row, moves):
    """Only the same matched current-season goalie can supply displayed stats."""
    name = text(row.get("Opp_Goalie"), "Unknown")
    status = text(row.get("Opp_Goalie_Status"), "Unknown")
    matched = (row.get("Goalie_Stats_Season") == "2026-2027"
               and text(row.get("Opp_Goalie_Source")).endswith("name_and_stats")
               and name != "Unknown")
    gaa = number(row.get("Opp_GAA")) if matched else None
    sv = number(row.get("Opp_SV")) if matched else None
    gp = number(row.get("Opp_GP")) if matched else None
    gaa = gaa if gaa is not None and gaa >= 0 else None
    sv = sv if sv is not None and 0 <= sv <= 1 else None
    gp = gp if gp is not None and gp >= 0 else None
    matched_records = [identity(move) for move in moves if record_for(move)]
    qualifies = (number(row.get("Assists_Line")) == .5
                 and text(row.get("Matrix_Assists")).casefold() in {"green", "🟢"}
                 and row.get("Model_Stats_Season") == "2026-2027"
                 and status.casefold() in {"confirmed", "official"}
                 and gaa is not None and gaa >= CUTOFF and bool(matched_records))
    return dict(name=name, status=status, gaa=gaa, sv=sv, gp=gp,
                qualifies=bool(qualifies), matched_records=matched_records)


def badge_html(value):
    if not value or not value.get("qualifies"):
        return ""
    return ('<span class="wn-goalie-buff" title="Optional matchup buff: confirmed opposing goalie GAA ≥2.95. '
            'Full and recent records are in Inspect; this does not change the move.">'
            '<span aria-hidden="true">✦</span> GOALIE EXPOSURE <b>GAA ≥2.95</b></span>')


def goalie_html(value):
    if not value:
        return ""
    gaa = f'{value["gaa"]:.2f}' if value.get("gaa") is not None else "—"
    sv = f'{value["sv"]:.3f}' if value.get("sv") is not None else "—"
    gp = f'{value["gp"]:g}' if value.get("gp") is not None else "—"
    return ('<div class="wn-goalie-strip"><span>OPP GOALIE</span> '
            f'<strong>{escape(value["name"])}</strong> <em>{escape(value["status"])}</em>'
            f'<span class="wn-goalie-stat">GAA <b>{gaa}</b></span>'
            f'<span class="wn-goalie-stat">SV <b>{sv}</b></span>'
            f'<span class="wn-goalie-stat">GP <b>{gp}</b></span></div>')


def inspect_html(value, move):
    if not value or not value.get("qualifies"):
        return ""
    record = record_for(move)
    if not record or identity(move) not in value.get("matched_records", []):
        data = records()
        record = data.get("combined")
        label = "Any current Assist move · one appearance per player-game"
    else:
        label = record["name"]
    if not record:
        return ""
    rows = []
    for period, label_period in [("full", "FULL"), ("recent", "RECENT")]:
        result = record["with_buff"][period]
        if not result["plays"]:
            rows.append(f'<tr><th>{label_period}</th><td colspan="7">No graded appearances</td></tr>')
            continue
        rows.append(f'<tr><th>{label_period}</th><td><b>{result["hits"]}/{result["plays"]}</b></td>'
                    f'<td><b>{result["rate"]:.1f}%</b></td><td>{result["BE"]:.1f}%</td>'
                    f'<td>{result["gap"]:+.1f} pp</td><td>{result["players"]}</td>'
                    f'<td>{result["units"]:+.2f}u</td><td>{result["ROI"]:+.1f}%</td></tr>')
    return ('<section class="wn-goalie-buff-detail"><strong>✦ GOALIE EXPOSURE · GAA ≥2.95</strong>'
            f'<div class="wn-goalie-buff-route">{escape(label)} + goalie buff</div>'
            '<div class="wn-goalie-buff-table"><table><thead><tr><th>Window</th><th>Hits / plays</th>'
            '<th>Hit rate</th><th>Book BE</th><th>Gap</th><th>Players</th><th>Units</th><th>ROI</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
            '<p>Full: Oct 13–Nov 12, 2026 · Recent: Nov 1–12, included in full. '
            'Saved pregame prices, flat 1u. These are the move’s buffed historical appearances; '
            'the original move and its record stay unchanged.</p></section>')


CSS = """
.wn-goalie-buff{display:inline-flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:8px;padding:7px 11px;border:1px solid #f9d875;border-radius:7px;background:linear-gradient(105deg,#6137a7,#35235e 64%,#6b4e24);color:#fff3bf;font-size:12px;font-weight:950;letter-spacing:.045em;box-shadow:0 0 12px #a477ff55,inset 0 0 12px #fbd87522;animation:wnGoalieBuffPulse 2.8s ease-in-out infinite}
.wn-goalie-buff>span{font-size:19px;color:#ffe995;text-shadow:0 0 9px #f8d678}.wn-goalie-buff>b{padding-left:7px;border-left:1px solid #e9cd8f88;color:#fff;letter-spacing:0}
@keyframes wnGoalieBuffPulse{0%,100%{box-shadow:0 0 9px #a477ff44,inset 0 0 10px #fbd87515}50%{box-shadow:0 0 18px #c28eff88,inset 0 0 15px #fbd87533}}
@media(prefers-reduced-motion:reduce){.wn-goalie-buff{animation:none}}
.wn-goalie-strip{display:flex;align-items:center;flex-wrap:wrap;gap:5px 10px;margin:5px 0 1px;padding:7px 9px;background:#111b2d;border:1px solid #65728c;border-radius:6px;color:#e6eeff;font-size:12px;line-height:1.5}.wn-goalie-strip>span:first-child{font-size:10px;font-weight:850;letter-spacing:.07em;color:#d9c1ff}.wn-goalie-strip>strong,.wn-goalie-stat>b{color:#fff}.wn-goalie-strip>em{font-size:11px;font-style:normal;color:#c5d0e5}.wn-goalie-stat{white-space:nowrap}
.wn-goalie-buff-detail{margin:8px 0;padding:11px;border:1px solid #af86ed;border-radius:8px;background:linear-gradient(120deg,#24213d,#142436);color:#edf3ff}.wn-goalie-buff-detail>strong{color:#ffe092;font-size:13px}.wn-goalie-buff-route{margin:5px 0 8px;color:#e5ceff;font-weight:750}.wn-goalie-buff-table{overflow-x:auto}.wn-goalie-buff-table table{width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap}.wn-goalie-buff-table th,.wn-goalie-buff-table td{padding:7px 8px;text-align:left;border-bottom:1px solid #63708455}.wn-goalie-buff-table thead th{color:#d6b9ff;font-size:10px}.wn-goalie-buff-table tbody th{color:#ffe1a0}.wn-goalie-buff-table td>b{color:#fff}.wn-goalie-buff-detail p{font-size:11px;line-height:1.5;color:#bccbe1;margin:8px 0 0}
"""
