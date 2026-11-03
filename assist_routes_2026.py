"""Additive, frozen November Assist profiles; no replacement of existing moves."""
from __future__ import annotations

import math

VERSION = "assist-2026-27-additive-2026-11-03-v1"
ENTRY = "Assist 0.5 + Matrix_Assists Green + Conf_Assists >=80 + iXA% >=94"
VARIANTS = {
    "Arcanist": ("#b692ff", "staff.svg"),
    "Whisperer": ("#a5a9e8", "stars.svg"),
    "Oracle": ("#c9b6ff", "odins_arcane_orb.svg"),
    "Runesmith": ("#79afff", "runic_infusion.svg"),
    "Stormcaller": ("#63e1cf", "arcane_channel_ii.svg"),
    "Oracle + Runesmith": ("#f4bf55", "odins_arcane_orb.svg"),
}


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def matches(row, field, op, cutoff):
    n = number(row.get(field))
    return n is not None and (n >= cutoff if op == ">=" else n <= cutoff)


def new_assists_moves(row):
    """Evaluate approved profiles without inheriting the old 81/95 gate."""
    line = number(row.get("Assists_Line"))
    if line is None:
        line = number(row.get("Line_Assists"))
    matrix = str(row.get("Matrix_Assists", row.get("Matrix_A", "")) or "").strip().casefold()
    if (line != .5 or matrix not in {"green", "🟢"}
            or row.get("Model_Stats_Season") != "2026-2027"
            or not matches(row, "Conf_Assists", ">=", 80)
            or not matches(row, "iXA%", ">=", 94)):
        return []
    fired = []
    for spec in FROZEN_MOVES:
        conditions = spec["conditions"]
        environment = any(field in {"Team_PP_GF60", "opp_5v5_HDCA60"} for field, _, _ in conditions)
        if environment and row.get("Team_Stats_Season") != "2026-2027":
            continue
        if not all(matches(row, *condition) for condition in conditions):
            continue
        variant = spec["variant"]
        accent, icon = VARIANTS[variant]
        rule = ENTRY + "".join(f" + {f} {op} {cut:g}" for f, op, cut in conditions)
        fired.append({
            "name": spec["name"], "move_id": spec["id"], "kind": spec["kind"],
            "level_label": spec["level"], "display_priority": spec["priority"],
            "icon": icon, "rule": rule, "wins": spec["wins"], "picks": spec["picks"],
            "later_wins": 0, "later_picks": 0, "unique_players": spec["players"],
            "market": "Assists", "sample_season": "2026-27", "sample_from": "2026-10-13",
            "sample_through": "2026-11-02", "sample_book_be_pct": spec["be"],
            "sample_players": spec["players"], "sample_dates": spec["dates"],
            "kit_version": VERSION, "track": False, "route_variant": variant,
            "route_accent": accent, "environment_enhanced": environment,
        })
    return fired


# FROZEN APPROVED DATA
FROZEN_MOVES = [{'name': 'Staff of Insight', 'id': 'assist-nov03-staff-of-insight', 'kind': 'STANCE', 'level': 'BASE', 'priority': 1, 'variant': 'Arcanist', 'conditions': [], 'wins': 133, 'picks': 233, 'players': 46, 'be': 52.099433988158154, 'dates': 21}, {'name': 'Silent Invocation', 'id': 'assist-nov03-silent-invocation', 'kind': 'SPECIAL', 'level': 'SPECIAL', 'priority': 3, 'variant': 'Whisperer', 'conditions': [('L10_S', '<=', 19)], 'wins': 34, 'picks': 54, 'players': 21, 'be': 51.36846568918696, 'dates': 14}, {'name': 'Oracle’s Channel', 'id': 'assist-nov03-oracle-s-channel', 'kind': 'SPECIAL', 'level': 'SPECIAL', 'priority': 3, 'variant': 'Arcanist', 'conditions': [('Goal_mu', '<=', 0.33)], 'wins': 67, 'picks': 101, 'players': 26, 'be': 52.338418711357846, 'dates': 19}, {'name': 'Oracle’s Guidance', 'id': 'assist-nov03-oracle-s-guidance', 'kind': 'SPECIAL', 'level': 'WOMBO', 'priority': 4, 'variant': 'Oracle', 'conditions': [('Goal_mu', '<=', 0.33), ('i5v5_primaryAssists60', '<=', 1.05)], 'wins': 50, 'picks': 67, 'players': 19, 'be': 51.61464171164869, 'dates': 17}, {'name': 'Runic Resonance I', 'id': 'assist-nov03-runic-resonance-i', 'kind': 'SPECIAL', 'level': 'WOMBO', 'priority': 4, 'variant': 'Runesmith', 'conditions': [('Goal_mu', '<=', 0.33), ('PPP10_total', '>=', 4)], 'wins': 42, 'picks': 55, 'players': 14, 'be': 53.798608151326576, 'dates': 18}, {'name': 'Storm Conduit I', 'id': 'assist-nov03-storm-conduit-i', 'kind': 'SPECIAL', 'level': 'WOMBO', 'priority': 4, 'variant': 'Stormcaller', 'conditions': [('Goal_mu', '<=', 0.33), ('Player_SOG_Share_Proxy', '>=', 11)], 'wins': 40, 'picks': 53, 'players': 14, 'be': 52.88397399236147, 'dates': 19}, {'name': 'Storm Conduit II', 'id': 'assist-nov03-storm-conduit-ii', 'kind': 'SPECIAL', 'level': 'STRONGER', 'priority': 5, 'variant': 'Stormcaller', 'conditions': [('Goal_mu', '<=', 0.33), ('Player_SOG_Share_Proxy', '>=', 11), ('Player_5v5_SOG_Share', '>=', 8)], 'wins': 32, 'picks': 40, 'players': 9, 'be': 53.82568170286036, 'dates': 17}, {'name': 'Oracle’s Guidance II', 'id': 'assist-nov03-oracle-s-guidance-ii', 'kind': 'SPECIAL', 'level': 'ENHANCED', 'priority': 6, 'variant': 'Oracle', 'conditions': [('Goal_mu', '<=', 0.33), ('i5v5_primaryAssists60', '<=', 1.05), ('Team_PP_GF60', '>=', 5.7), ('opp_5v5_HDCA60', '<=', 2.2)], 'wins': 36, 'picks': 44, 'players': 14, 'be': 51.93628787557438, 'dates': 15}, {'name': 'Runic Resonance II', 'id': 'assist-nov03-runic-resonance-ii', 'kind': 'SPECIAL', 'level': 'ENHANCED', 'priority': 6, 'variant': 'Runesmith', 'conditions': [('Goal_mu', '<=', 0.33), ('PPP10_total', '>=', 4), ('Team_PP_GF60', '>=', 5.7), ('opp_5v5_HDCA60', '<=', 2.2)], 'wins': 33, 'picks': 40, 'players': 13, 'be': 53.57474173750709, 'dates': 15}, {'name': 'Empowered Storm Conduit', 'id': 'assist-nov03-empowered-storm-conduit', 'kind': 'SPECIAL', 'level': 'ENHANCED', 'priority': 6, 'variant': 'Stormcaller', 'conditions': [('Goal_mu', '<=', 0.33), ('Player_SOG_Share_Proxy', '>=', 11), ('Team_PP_GF60', '>=', 5.7), ('opp_5v5_HDCA60', '<=', 2.2)], 'wins': 31, 'picks': 37, 'players': 11, 'be': 52.669530996262964, 'dates': 15}, {'name': 'Runic Orchestration Crit', 'id': 'assist-nov03-runic-orchestration-crit', 'kind': 'CRIT', 'level': 'CRIT', 'priority': 7, 'variant': 'Oracle + Runesmith', 'conditions': [('Goal_mu', '<=', 0.33), ('i5v5_primaryAssists60', '<=', 1.05), ('PPP10_total', '>=', 4), ('Team_PP_GF60', '>=', 5.7), ('opp_5v5_HDCA60', '<=', 2.2)], 'wins': 27, 'picks': 31, 'players': 10, 'be': 53.40010282435993, 'dates': 14}]
