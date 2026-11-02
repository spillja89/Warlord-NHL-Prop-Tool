"""Current-season Goal profile attacks from saved pregame tracker decisions.

The records below are searched, in-sample hit counts, not calibrated chances.
The defensive matchup is a separate buff and requires labeled 2026-27 team
stats. Player attacks never depend on that buff.
"""

from __future__ import annotations

import math


GOAL_KIT_VERSION = "goal-2026-27-profile-v2"
ENTRY = "Goal 0.5 + Matrix_Goal Green + Goal profile >= 93 + last-five xG/game >= 0.55"

# Name, tier, icon, added checks, wins, plays. Every check is applied after ENTRY.
# Plays can fire in multiple routes; their records must not be added together.
RULES = (
    ("Base Attack", "BASE ATTACK", "base.svg", (), 43, 85),
    ("Creator Fade", "HEAVY ATTACK", "fenrir_claw.svg", (("L20_A", "<=", 8),), 42, 77),
    ("Star Pressure", "HEAVY ATTACK", "smash.svg", (("StarScore", ">=", 84.5),), 40, 74),
    ("Shot Control", "HEAVY ATTACK", "armor_shred.svg", (("L10_S", "<=", 38),), 40, 73),
    ("Drought Strike", "SPECIAL ATTACK", "fury.svg", (("Drought_G", ">=", 1),), 22, 36),
    ("Measured Intent", "SPECIAL ATTACK", "armor_shred.svg", (("ShotIntent_Pct", "<=", 91.1),), 23, 35),
    ("Creator's Cut", "SPECIAL ATTACK", "fenrir_claw.svg",
     (("L20_A", "<=", 8), ("ShotIntent_Pct", "<=", 91.1)), 22, 32),
    ("Star Shot Strike", "SPECIAL ATTACK", "smash.svg",
     (("StarScore", ">=", 84.5), ("L20_S", "<=", 31)), 27, 39),
    ("Volume Rebound", "SPECIAL ATTACK", "fury.svg",
     (("L10_S", "<=", 38), ("Drought_G", ">=", 1)), 19, 26),
)

# This is an additional context signal, not an entry gate or a player goal chance.
DEFENSE_BUFF = ("Defense Breach", "MATCHUP BUFF", "armor_shred.svg",
                (("v2_defense_vulnerability", ">=", 51.5),), 21, 35)

# Descriptive migration map only; old hit records are never transferred.
REPLACES = {
    "Base Attack": "Base Attack",
    "Creator Fade": "Scorer's Form",
    "Star Pressure": "Star Pressure",
    "Shot Control": "Shotmaker",
    "Drought Strike": "Scorer's Threat",
    "Measured Intent": "NEW",
    "Creator's Cut": "Expected Finisher",
    "Star Shot Strike": "Star Scorer",
    "Volume Rebound": "Expected Shot Barrage",
    "Defense Breach": "NEW",
}


def _number(row, key):
    try:
        value = float(row.get(key))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _passes(row, checks):
    for field, comparison, cutoff in checks:
        value = _number(row, field)
        if value is None or not (value >= cutoff if comparison == ">=" else value <= cutoff):
            return False
    return True


def _move(spec, *, buff=False):
    name, kind, icon, checks, wins, picks = spec
    condition = " + ".join(f"{field} {comparison} {cutoff:g}"
                           for field, comparison, cutoff in checks)
    if buff:
        condition = "verified 2026-27 team stats + " + condition
    rule = ENTRY + (" + " + condition if condition else "")
    return {"name": name, "research_name": name, "kind": kind, "icon": icon,
            "condition": condition or ENTRY, "rule": rule, "wins": wins,
            "picks": picks, "later_wins": 0, "later_picks": 0,
            "crit": False, "experimental": False, "sample_season": "2026-27",
            "sample_through": "2026-10-31", "replaces": REPLACES[name],
            "kit_version": GOAL_KIT_VERSION, "matchup_buff": buff}


def goals_moves(row):
    """Apply the same player and source gates used in the research cohort."""
    line = _number(row, "Goal_Line")
    if line is None:
        line = _number(row, "ATG_Line")
    ixg = _number(row, "iXG%")
    if ixg is None:
        ixg = _number(row, "iXG_pct")
    toi = _number(row, "TOI_Pct")
    profile = round(.82 * ixg + .18 * toi) if ixg is not None and toi is not None else None
    recent_xg = _number(row, "Pregame_xG_L5_per_game")
    if (line != .5 or str(row.get("Matrix_Goal", "")).strip().casefold() != "green"
            or profile is None or profile < 93 or recent_xg is None or recent_xg < .55
            or str(row.get("Model_Stats_Season", "")) != "2026-2027"
            or str(row.get("Form_Season", "")) != "2026-27"):
        return []
    moves = [_move(spec) for spec in RULES if _passes(row, spec[3])]
    if (str(row.get("Team_Stats_Season", "")) == "2026-2027"
            and _passes(row, DEFENSE_BUFF[3])):
        moves.append(_move(DEFENSE_BUFF, buff=True))
    return moves
