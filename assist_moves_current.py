"""2026-27 Assist move tree from saved, graded over-0.5 pregame trackers.

The PP Crits are upgrades of player Specials, not independent picks. Every
record below counts a player-game once inside that move's own historical set.
"""

from __future__ import annotations

import math


ASSIST_KIT_VERSION = "assist-2026-27-pp-crits-v1"
ENTRY = "Assist 0.5 + Matrix_Assists Green + Conf_Assists >= 81 + iXA% >= 95"


def _number(row, key):
    try:
        value = float(row.get(key))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _at_least(row, key, cutoff):
    value = _number(row, key)
    return value is not None and value >= cutoff


def _move(name, kind, icon, rule, wins, picks, unique_players):
    return {
        "name": name, "kind": kind, "icon": icon, "rule": rule,
        "wins": wins, "picks": picks, "later_wins": 0, "later_picks": 0,
        "unique_players": unique_players, "market": "Assists",
        "sample_season": "2026-27", "kit_version": ASSIST_KIT_VERSION,
        "track": False,
    }


def assists_moves(row):
    """Return the live Assist tier and its supporting fired routes."""
    line = _number(row, "Assists_Line")
    if line is None:
        line = _number(row, "Line_Assists")
    matrix = str(row.get("Matrix_Assists", row.get("Matrix_A", "")) or "").strip().casefold()
    if line != 0.5 or matrix not in {"green", "🟢"} or not _at_least(row, "Conf_Assists", 81):
        return []
    if not _at_least(row, "iXA%", 95):
        return []

    moves = [_move("Staff (Green Stance)", "STANCE", "staff.svg", ENTRY,
                   137, 238, 54)]

    if _at_least(row, "PP_Points60", 3.75) and _at_least(row, "PP_iXA60", 2.5):
        moves.append(_move("Arcane Channel I", "VOLUME", "arcane_channel_i.svg",
                           ENTRY + " + PP_Points60 >= 3.75 + PP_iXA60 >= 2.5",
                           106, 172, 41))

    pp_environment = _at_least(row, "Team_PP_xGF60", 8.3)
    specials = (
        ("Runic Infusion", "runic_infusion.svg", "PP_iXA60", 3.9,
         (70, 105, 29), (29, 41, 13)),
        ("Rune Conduit", "arcane_channel_ii.svg", "PP_TOI_per_game", 3.75,
         (50, 75, 24), (20, 25, 11)),
        ("Odin's Vigil", "odins_arcane_orb.svg", "TOI_per_game", 22,
         (47, 69, 20), (20, 24, 7)),
    )
    for name, icon, field, cutoff, normal_record, crit_record in specials:
        if not _at_least(row, field, cutoff):
            continue
        rule = ENTRY + f" + {field} >= {cutoff:g}"
        moves.append(_move(name, "SPECIAL", icon, rule, *normal_record))
        if pp_environment:
            moves.append(_move(name + " Crit", "CRIT", icon,
                               rule + " + Team_PP_xGF60 >= 8.3", *crit_record))

    # These are supporting traits. They never become a second primary pick.
    if _at_least(row, "Assists_mu", 1.1):
        moves.append(_move("High Assist μ", "CONTEXT", "arcane_channel_iii.svg",
                           ENTRY + " + Assists_mu >= 1.1", 100, 163, 37))
    if _at_least(row, "Drought_G", 2):
        moves.append(_move("Arcane Rebound", "CONTEXT", "stars.svg",
                           ENTRY + " + Drought_G >= 2", 66, 100, 44))
    return moves
