"""Current-season Goal profile attacks from saved pregame tracker decisions.

The records below are searched, in-sample hit counts, not calibrated chances.
The defensive matchup is a separate buff and requires labeled 2026-27 team
stats. Player attacks never depend on that buff.
"""

from __future__ import annotations

import math


GOAL_KIT_VERSION = "goal-2026-27-profile-v3"
ENTRY = "Goal 0.5 + Matrix_Goal Green + last-five xG/game >= 0.55"

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

# Each player route fires at level I. Confidence upgrades only the routes where
# the complete saved move retained a useful separation. These are overlapping
# retrospective records, not independent samples or tonight's probabilities.
# (minimum rounded Goal profile, level, wins, plays); None means no score cutoff.
LEVELS = {
    "Base Attack": ((None, "I", 47, 100), (89, "II", 47, 96), (93, "III", 43, 85)),
    "Creator Fade": ((None, "I", 46, 92), (89, "II", 46, 88), (93, "III", 42, 77)),
    "Star Pressure": ((None, "I", 41, 78),),
    "Shot Control": ((None, "I", 44, 88), (89, "II", 44, 84), (93, "III", 40, 73)),
    "Drought Strike": ((None, "I", 23, 43), (89, "II", 23, 41), (93, "III", 22, 36)),
    "Measured Intent": ((None, "I", 26, 43), (89, "II", 26, 40)),
    "Creator's Cut": ((None, "I", 25, 40), (89, "II", 25, 37)),
    "Star Shot Strike": ((None, "I", 28, 43), (93, "II", 27, 39)),
    "Volume Rebound": ((None, "I", 20, 33), (89, "II", 20, 31), (93, "III", 19, 26)),
}

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


def _move(spec, *, buff=False, level=None):
    name, kind, icon, checks, wins, picks = spec
    profile_cutoff = None
    if level is not None:
        profile_cutoff, label, wins, picks = level
    conditions = ([f"Goal profile >= {profile_cutoff:g}"]
                  if profile_cutoff is not None else [])
    conditions += [f"{field} {comparison} {cutoff:g}"
                   for field, comparison, cutoff in checks]
    condition = " + ".join(conditions)
    if buff:
        condition = "verified 2026-27 team stats + Goal profile >= 93 + " + condition
    rule = ENTRY + (" + " + condition if condition else "")
    return {"name": f"{name} {label}" if level is not None else name,
            "research_name": name, "level": label if level is not None else None,
            "profile_cutoff": profile_cutoff, "kind": kind, "icon": icon,
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
            or profile is None or recent_xg is None or recent_xg < .55
            or str(row.get("Model_Stats_Season", "")) != "2026-2027"
            or str(row.get("Form_Season", "")) != "2026-27"):
        return []
    moves = []
    for spec in RULES:
        if _passes(row, spec[3]):
            level = next(candidate for candidate in reversed(LEVELS[spec[0]])
                         if candidate[0] is None or profile >= candidate[0])
            moves.append(_move(spec, level=level))
    if (profile >= 93 and str(row.get("Team_Stats_Season", "")) == "2026-2027"
            and _passes(row, DEFENSE_BUFF[3])):
        moves.append(_move(DEFENSE_BUFF, buff=True))
    return moves
