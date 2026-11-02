"""Current-season Goal profile attacks from saved pregame tracker decisions.

The records below are searched, in-sample hit counts, not calibrated chances.
The defensive matchup is a separate buff and requires labeled 2026-27 team
stats. Player attacks never depend on that buff.
"""

from __future__ import annotations

import math


GOAL_KIT_VERSION = "goal-2026-27-three-paths-v6"
ENTRY = "Goal 0.5 + Matrix_Goal Green + last-five xG/game >= 0.50"

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
    ("Reaper's Return", "SPECIAL ATTACK", "fury.svg",
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
    "Reaper's Return": ((None, "I", 20, 33), (89, "II", 20, 31), (93, "III", 19, 26)),
}

# The same pregame cohort with a broader 0.50 xG entry. All 0.50–0.54
# marginal groups returned positive flat-stake units at their saved prices,
# but these remain in-sample records rather than expected profit tonight.
BASE_LEVELS = {
    "Base Attack": {"I": (67, 150), "II": (65, 141), "III": (57, 120)},
    "Creator Fade": {"I": (62, 133), "II": (60, 124), "III": (52, 103)},
    "Star Pressure": {"I": (56, 113)},
    "Shot Control": {"I": (64, 134), "II": (62, 125), "III": (54, 104)},
    "Drought Strike": {"I": (34, 66), "II": (32, 60), "III": (29, 51)},
    "Measured Intent": {"I": (40, 73), "II": (38, 65)},
    "Creator's Cut": {"I": (35, 64), "II": (33, 56)},
    "Star Shot Strike": {"I": (37, 63), "II": (34, 55)},
    "Reaper's Return": {"I": (31, 53), "II": (29, 47), "III": (26, 38)},
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
    "Reaper's Return": "Expected Shot Barrage",
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


def goal_profile_score(row):
    """Rounded profile used to award Goal attack levels."""
    ixg = _number(row, "iXG%")
    if ixg is None:
        ixg = _number(row, "iXG_pct")
    toi = _number(row, "TOI_Pct")
    return round(.82 * ixg + .18 * toi) if ixg is not None and toi is not None else None


def _move(spec, *, buff=False, level=None, xg_boost=False):
    name, kind, icon, checks, wins, picks = spec
    profile_cutoff = None
    if level is not None:
        profile_cutoff, label, wins, picks = level
        if not xg_boost:
            wins, picks = BASE_LEVELS[name][label]
    conditions = ([f"Goal profile >= {profile_cutoff:g}"]
                  if profile_cutoff is not None else [])
    conditions += [f"{field} {comparison} {cutoff:g}"
                   for field, comparison, cutoff in checks]
    condition = " + ".join(conditions)
    if buff:
        condition = "last-five xG/game >= 0.55 + verified 2026-27 team stats + Goal profile >= 93 + " + condition
    elif xg_boost:
        condition = "last-five xG/game >= 0.55" + (" + " + condition if condition else "")
    rule = ENTRY + (" + " + condition if condition else "")
    display_name = f"{name} {label}{' · xG+' if xg_boost else ''}" if level is not None else name
    if name in {"Drought Strike", "Reaper's Return"}:
        display_name += " · xG route"
    return {"name": display_name, "route_family": "xg", "market": "Goal",
            "research_name": name, "level": label if level is not None else None,
            "profile_cutoff": profile_cutoff, "xg_boost": xg_boost,
            "kind": kind, "icon": icon,
            "condition": condition or ENTRY, "rule": rule, "wins": wins,
            "picks": picks, "later_wins": 0, "later_picks": 0,
            "crit": False, "experimental": False, "sample_season": "2026-27",
            "sample_through": "2026-10-31", "replaces": REPLACES[name],
            "kit_version": GOAL_KIT_VERSION, "matchup_buff": buff}


def xg_goals_moves(row):
    """Apply the same player and source gates used in the research cohort."""
    line = _number(row, "Goal_Line")
    if line is None:
        line = _number(row, "ATG_Line")
    profile = goal_profile_score(row)
    recent_xg = _number(row, "Pregame_xG_L5_per_game")
    if (line != .5 or str(row.get("Matrix_Goal", "")).strip().casefold() != "green"
            or profile is None or recent_xg is None or recent_xg < .50
            or str(row.get("Model_Stats_Season", "")) != "2026-2027"
            or str(row.get("Form_Season", "")) != "2026-27"):
        return []
    moves = []
    for spec in RULES:
        if _passes(row, spec[3]):
            level = next(candidate for candidate in reversed(LEVELS[spec[0]])
                         if candidate[0] is None or profile >= candidate[0])
            moves.append(_move(spec, level=level, xg_boost=recent_xg >= .55))
    if (recent_xg >= .55 and profile >= 93
            and str(row.get("Team_Stats_Season", "")) == "2026-2027"
            and _passes(row, DEFENSE_BUFF[3])):
        moves.append(_move(DEFENSE_BUFF, buff=True))
    return moves


def goals_moves(row):
    """Independent xG, StarScore/drought and StarScore/TOI20 Goal paths."""
    from goal_star_moves import star_goals_moves
    from goal_toi_moves import toi_goals_moves

    return xg_goals_moves(row) + star_goals_moves(row) + toi_goals_moves(row)


def goal_move_summary(moves):
    """Shared tracker summary; environment records never replace player attacks."""
    attacks = sorted((move for move in moves if not move.get("matchup_buff")),
                     key=lambda move: (move["wins"] / move["picks"], move["picks"]), reverse=True)
    best = attacks[0] if attacks else None
    def record(move):
        return f"{move['wins']}/{move['picks']} · {100 * move['wins'] / move['picks']:.1f}%"
    def entries(items):
        return "\n".join(f"{move['name']} · {record(move)}" for move in items) or "—"
    return {"Goal_Attack": best["name"] if best else "—",
            "Goal_Attack_Record": record(best) if best else "—",
            "Goal_Attack_Sample_BE": best.get("sample_book_be_pct") if best else None,
            "Goal_All_Attacks": entries(attacks),
            "Goal_Crits": entries(move for move in attacks if move.get("crit")),
            "Goal_Buffs": entries(move for move in moves if move.get("matchup_buff"))}
