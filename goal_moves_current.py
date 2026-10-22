"""2026-27 regular-season Goal attacks selected from graded pregame trackers.

Records are retrospectively selected hit rates, never tonight probabilities.
Each rule uses the same saved inputs and exact cutoff as the research replay.
"""

from __future__ import annotations

import math


GOAL_KIT_VERSION = "goal-2026-27-current-v1"
ENTRY = "Goal 0.5 + Matrix_Goal Green + Conf_Goal >= 85"

# (name, tier, icon, added pregame cutoffs, graded wins, graded plays)
RULES = (
    ("Base Attack", "BASE ATTACK", "base.svg", (("TOI_per_game", 19), ("Conf_SOG", 80)), 60, 116),
    ("Shooter's Command", "SPECIAL ATTACK", "fenrir_claw.svg", (("TOI_per_game", 19), ("Conf_SOG", 80), ("TOI_Pct", 88), ("Player_5v5_SOG_Share", 15), ("Goal_mu", .45)), 17, 21),
    ("Power Finisher", "SPECIAL ATTACK", "fury.svg", (("TOI_per_game", 19), ("Conf_SOG", 80), ("TOI_Pct", 88), ("PP_iXG60", 1.6), ("Goal_mu", .45)), 23, 31),
    ("Star Pressure", "HEAVY ATTACK", "smash.svg", (("TOI_Pct", 84), ("StarScore", 88)), 46, 82),
    ("Power Shooter", "HEAVY ATTACK", "fury.svg", (("TOI_Pct", 84), ("PP_iXG60", 1.2)), 44, 77),
    ("Scorer's Form", "HEAVY ATTACK", "fenrir_claw.svg", (("TalentMult", 1.2), ("Pregame_G_L5_per_game", .4), ("TOI_per_game", 19)), 41, 73),
    ("Star Scorer", "HEAVY ATTACK", "smash.svg", (("TOI_Pct", 84), ("StarScore", 88), ("Pregame_G_L5_per_game", .6)), 33, 51),
    ("Power Shot", "HEAVY ATTACK", "fury.svg", (("TOI_Pct", 84), ("PP_iXG60", 1.2), ("Goal_mu", .5)), 22, 32),
    ("Scorer's Threat", "HEAVY ATTACK", "fenrir_claw.svg", (("TalentMult", 1.2), ("Pregame_G_L5_per_game", .4), ("TOI_per_game", 19), ("Pregame_xG_L5_per_game", .55)), 19, 29),
    ("Team Scoring Surge", "SPECIAL ATTACK", "smash.svg", (("TOI_Pct", 84), ("StarScore", 88), ("Pregame_G_L5_per_game", .6), ("Team_GF_Avg_L5", 3.2)), 26, 37),
    ("Shotmaker", "SPECIAL ATTACK", "armor_shred.svg", (("Points_mu", 1.25), ("Avg5_SOG", 3.4)), 36, 60),
    ("Expected Finisher", "SPECIAL ATTACK", "fenrir_claw.svg", (("Points_mu", 1.25), ("Pregame_xG_L5_per_game", .55)), 25, 38),
    ("Ice Time Finisher", "SPECIAL ATTACK", "smash.svg", (("TOI_per_game", 21), ("Goal_mu", .45)), 26, 38),
    ("Shotmaker Strike", "SPECIAL ATTACK", "armor_shred.svg", (("Points_mu", 1.25), ("Avg5_SOG", 3.4), ("Goal_mu", .45)), 30, 45),
    ("Expected Shot Barrage", "SPECIAL ATTACK", "fenrir_claw.svg", (("Points_mu", 1.25), ("Pregame_xG_L5_per_game", .55), ("L10_S", 19)), 21, 29),
    ("Power Play Finisher", "SPECIAL ATTACK", "fury.svg", (("TOI_per_game", 21), ("Goal_mu", .45), ("PP_iXG60", 1.6)), 24, 32),
    ("FOR VALHALLA!", "ULTIMATE ATTACK", "valhalla.svg", (("Points_mu", 1.25), ("Avg5_SOG", 3.4), ("Goal_mu", .45), ("Opp_PK_xGA60", 6)), 26, 34),
)

# Old card name accounted for by each new rule. NEW means a genuinely new
# branch. Historical records are never carried across this name map.
REPLACES = {
    "Base Attack": "Base Attack",
    "Shooter's Command": "NEW",
    "Power Finisher": "NEW",
    "Star Pressure": "Press the Attack",
    "Power Shooter": "Fury Shredder",
    "Scorer's Form": "Finisher Strike",
    "Star Scorer": "Enraged Finisher",
    "Power Shot": "Bloodlust",
    "Scorer's Threat": "Enraged Fury Shredder",
    "Team Scoring Surge": "Bloodlusted Surge",
    "Shotmaker": "Open Lanes Assault",
    "Expected Finisher": "Warlord Smash Attack",
    "Ice Time Finisher": "Berserker Aggression",
    "Shotmaker Strike": "Warlord Smash Attack (Crit)",
    "Expected Shot Barrage": "NEW",
    "Power Play Finisher": "Tyr’s Wrath Unleashed",
    "FOR VALHALLA!": "FOR VALHALLA!",
}


def _number(row, key):
    try:
        value = float(row.get(key))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def goals_moves(row):
    line = _number(row, "Goal_Line")
    if line is None:
        line = _number(row, "ATG_Line")
    if (line != .5 or str(row.get("Matrix_Goal", "")).strip().casefold() != "green"
            or (_number(row, "Conf_Goal") or 0) < 85):
        return []
    moves = []
    for name, kind, icon, checks, wins, picks in RULES:
        if all((_number(row, field) is not None and _number(row, field) >= cutoff)
               for field, cutoff in checks):
            condition = " + ".join(f"{field} >= {cutoff:g}" for field, cutoff in checks)
            display_name = REPLACES[name] if REPLACES[name] != "NEW" else name
            moves.append({"name": display_name, "research_name": name,
                          "kind": kind, "condition": condition,
                          "rule": ENTRY + " + " + condition,
                          "wins": wins, "picks": picks, "later_wins": 0,
                          "later_picks": 0, "icon": icon, "crit": False,
                          "experimental": False, "sample_season": "2026-27",
                          "replaces": REPLACES[name],
                          "kit_version": GOAL_KIT_VERSION})
    return moves
