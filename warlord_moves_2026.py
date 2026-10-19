"""Graded 2026 Points and SOG display moves.

These are historical subsets, not model probabilities. The rules only label
tracker rows; they do not change model predictions or eligibility.
Regular-season archive: Jan 18–Apr 13, 2026. Apr 18/20 playoff rows are
excluded. "Later" is Mar 15 onward within that regular-season archive.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


VERSION = "2026-10-16-opportunity-v4"


def _number(row: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(str(row.get(key, "")).strip().replace("%", ""))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _at_most(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def _green(row: Mapping[str, Any], market: str) -> bool:
    return str(row.get(f"Matrix_{market}", "")).strip().casefold() == "green"


def _move(
    name: str, kind: str, icon: str, rule: str,
    wins: int, picks: int, later_wins: int, later_picks: int,
    *, track: bool = False,
) -> dict[str, Any]:
    return {
        "name": name, "kind": kind, "icon": icon, "rule": rule,
        "wins": wins, "picks": picks,
        "later_wins": later_wins, "later_picks": later_picks,
        "track": track,
    }


def points_moves(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Only exact 0.5 or 1.5 Points lines with a Green matrix."""
    line = _number(row, "Points_Line")
    if line not in (0.5, 1.5) or not _green(row, "Points"):
        return []
    conf = _number(row, "Conf_Points_Move" if "Conf_Points_Move" in row else "Conf_Points")
    assists_mu = _number(row, "Assists_mu")
    toi = _number(row, "TOI_per_game")
    l10_s = _number(row, "L10_S")
    l10_p = _number(row, "L10_P")
    pk = _number(row, "Opp_PK_xGA60")
    moves: list[dict[str, Any]] = []

    if line == 0.5:
        if not _at_least(conf, 80):
            return []
        moves.append(_move("Hammer Fists", "FLOOR", "PTS05_HAMMER_FISTS.svg",
                           "Conf_Points >= 80", 220, 358, 147, 245))
        if _at_least(l10_s, 17):
            moves.append(_move("Echo Stomp I", "TIER", "PTS05_HAMMER_STOMP.svg",
                               "Conf_Points >= 80 + L10_S >= 17", 184, 280, 128, 196))
        if _at_least(toi, 19.18):
            moves.append(_move("Echo Stomp II", "TIER", "PTS05_HAMMER_STOMP.svg",
                               "Conf_Points >= 80 + TOI_per_game >= 19.18", 157, 228, 107, 161))
        if _at_least(assists_mu, 0.746):
            moves.append(_move("Gaia’s Blessing", "TIER", "PTS05_GAIAS_BLESSING.svg",
                               "Conf_Points >= 80 + Assists_mu >= 0.746", 152, 234, 105, 163))
            if _at_least(conf, 81):
                moves.append(_move("Gaia’s Blessing+ (Press)", "HEAVY", "PTS05_GAIAS_BLESSING.svg",
                                   "Conf_Points >= 81 + Assists_mu >= 0.746", 133, 200, 92, 139))
            if _at_least(conf, 83):
                moves.append(_move("Gaia’s Blessing++ (Smash)", "HEAVY", "PTS05_GAIAS_BLESSING.svg",
                                   "Conf_Points >= 83 + Assists_mu >= 0.746", 98, 140, 68, 98))
        if _at_least(toi, 19.18) and _at_least(l10_s, 17):
            moves.append(_move("Hammer Fists II", "HEAVY", "PTS05_HAMMER_FISTS.svg",
                               "Conf_Points >= 80 + TOI_per_game >= 19.18 + L10_S >= 17",
                               143, 203, 99, 144))
        if _at_least(conf, 83) and _at_least(l10_p, 8) and _at_least(l10_s, 17):
            moves.append(_move("Enraged Engine", "CRIT", "PTS05_ENRAGED_FURY.svg",
                               "Conf_Points >= 83 + L10_P >= 8 + L10_S >= 17",
                               46, 58, 29, 37, track=True))
        if _at_least(conf, 83) and _at_least(toi, 19.18) and _at_least(pk, 7.784):
            moves.append(_move("Blood Stomp", "CRIT", "PTS05_BLEED_ENV.svg",
                               "Conf_Points >= 83 + TOI_per_game >= 19.18 + Opp_PK_xGA60 >= 7.784",
                               38, 46, 22, 29, track=True))
        return moves

    if not _at_least(conf, 75):
        return []
    moves.append(_move("DPS Tank stance", "STANCE", "PTS15_TWO_HANDED_HAMMER.svg",
                       "Conf_Points >= 75", 135, 329, 65, 171))
    conf_goal = _number(row, "Conf_Goal")
    conf_sog = _number(row, "Conf_SOG")
    volume = _at_most(conf_goal, 88) and _at_least(conf_sog, 80)
    if volume:
        moves.append(_move("Backbone", "FLOOR", "PTS15_TWO_HANDED_HAMMER.svg",
                           "Conf_Points >= 75 + Conf_Goal <= 88 + Conf_SOG >= 80",
                           72, 141, 30, 54))
        if _at_most(assists_mu, 1.55):
            moves.append(_move("Blade Impale (Power Tier)", "HEAVY", "PTS15_BLADE_IMPALE.svg",
                               "Backbone + Assists_mu <= 1.55", 53, 88, 25, 38))
    proof = _at_least(conf, 83) and _at_least(_number(row, "Goal_ProofCount"), 3)
    # The observed boundary is 7.382492...; 7.3825 rounds away one graded loss.
    model_pk = _at_least(_number(row, "Goal_Model%"), 33.4) and _at_least(pk, 7.38249)
    if proof:
        moves.append(_move("Blade Impale", "SPECIAL", "PTS15_BLADE_IMPALE.svg",
                           "Conf_Points >= 83 + Goal_ProofCount >= 3", 33, 53, 12, 21))
    if model_pk:
        moves.append(_move("Blade Slash", "SPECIAL", "PTS15_BLADE_SLASH.svg",
                           "Conf_Points >= 75 + Goal_Model% >= 33.4 + Opp_PK_xGA60 >= 7.38249",
                           32, 52, 17, 30))
    if proof and model_pk:
        moves.append(_move("Ragnarok", "ULTIMATE", "PTS15_POLARIZING_SMASH.svg",
                           "Blade Impale + Blade Slash", 16, 18, 5, 6, track=True))
    return moves


def sog_moves(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Only exact 2.5 or 3.5 SOG lines with Green and Conf_SOG >= 75."""
    line = _number(row, "SOG_Line")
    if line not in (2.5, 3.5) or not _green(row, "SOG") or not _at_least(_number(row, "Conf_SOG"), 75):
        return []
    moves: list[dict[str, Any]] = []
    # The archived Volley record used a slate-level SOG proxy, not the full
    # team's actual 5v5 SOG share. Preserve that input for comparability.
    share = _number(row, "Player_SOG_Share_Proxy")

    if line == 2.5:
        moves.append(_move("Jungle Stance", "STANCE", "sog_basic_swipe.svg",
                           "Green + SOG 2.5 + Conf_SOG >= 75", 100, 196, 57, 110))
        ixg = _number(row, "iXG%")
        if _at_least(ixg, 93.2):
            moves.append(_move("Berserker Swipe", "FLOOR", "sog_basic_swipe.svg",
                               "iXG% >= 93.2", 77, 135, 45, 76))
        if _at_least(_number(row, "Goalie_Weak"), 62.5):
            moves.append(_move("Paralysis", "ENV", "sog_env_paralysis.svg",
                               "Goalie_Weak >= 62.5", 34, 52, 23, 33))
        rage = _at_least(_number(row, "L20_A"), 9) and _at_most(_number(row, "L40_A"), 29)
        volley = _at_least(_number(row, "L10_Rate_SOG"), 2.9) and _at_least(share, 18.8)
        if rage:
            moves.append(_move("Berserker’s Rage", "HEAVY", "sog_berserkers_rage.svg",
                               "L20_A >= 9 + L40_A <= 29", 48, 73, 34, 51))
        if volley:
            moves.append(_move("Berserker Volley", "SUPPORT", "sog_berserker_volley.svg",
                               "L10 SOG/game >= 2.9 + slate SOG share proxy >= 18.8", 34, 50, 21, 33,
                               track=True))
        if rage and volley:
            moves.append(_move("Assassin’s Overdrive", "CRIT", "sog_assassins_overdrive.svg",
                               "Rage + Volley", 26, 33, 18, 22, track=True))
        if _at_least(_number(row, "PP_TOI_Pct"), 15.9) and _at_least(ixg, 93.2):
            moves.append(_move("Bloodthirst", "HEAVY", "sog_bloodthirst.svg",
                               "PP_TOI_Pct >= 15.9 + iXG% >= 93.2", 43, 66, 21, 36))
        if rage and _at_least(_number(row, "i5v5_points60"), 2.2):
            moves.append(_move("Siege", "ULTIMATE", "sog_berserker_siege.svg",
                               "Rage + i5v5_points60 >= 2.2", 30, 39, 19, 24, track=True))
        return moves

    moves.append(_move("Sniper Stance", "STANCE", "sog_class_shooter.svg",
                       "Green + SOG 3.5 + Conf_SOG >= 75", 135, 334, 78, 189))
    hdca = _at_least(_number(row, "opp_5v5_HDCA60"), 2.36)
    strong = hdca and _at_least(_number(row, "PPP10_total"), 3)
    permission = hdca and _at_least(_number(row, "Best_Conf"), 89)
    assist_model = _at_least(_number(row, "Assists_Model%"), 56.3)
    if strong:
        moves.append(_move("Strong", "FLOOR", "sog_berserkers_rage.svg",
                           "PPP10_total >= 3 + opp_5v5_HDCA60 >= 2.36",
                           66, 123, 35, 68))
    if permission:
        moves.append(_move("Permission Special", "SUPPORT", "sog_env_paralysis.svg",
                           "Best_Conf >= 89 + opp_5v5_HDCA60 >= 2.36",
                           60, 112, 36, 68, track=True))
    if strong and assist_model:
        moves.append(_move("Enhanced Enraged", "SPECIAL", "sog_enhanced_enraged_1.svg",
                           "Strong + Assists_Model% >= 56.3", 42, 62, 25, 39,
                           track=True))
        if _at_least(share, 19):
            moves.append(_move("Sniper Crit", "CRIT", "sog_elite_enraged_strike.svg",
                               "Strong + Assists_Model% >= 56.3 + SOG share >= 19",
                               25, 33, 16, 21, track=True))
    return moves


def best_move(moves: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose a sorting label from non-experimental moves using shrinkage."""
    eligible = [m for m in moves if m["kind"] != "STANCE" and not m["track"]]
    if not eligible:
        eligible = [m for m in moves if m["kind"] in ("FLOOR", "STANCE")]
    if not eligible:
        return None

    def ranking(move: dict[str, Any]) -> tuple[float, int]:
        rate = 100.0 * move["wins"] / move["picks"]
        adjusted = (rate * move["picks"] + 50.0 * 20) / (move["picks"] + 20)
        return adjusted, move["picks"]

    return max(eligible, key=ranking)


# Shared pregame rules for the app and frozen slate. Keep move names and icons stable.
def _safe_float(value, default=None):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


# Incremental prior-season graded Green 0.5-goal rows with Conf_Points 82–83.
# Add these to the original 84+ records to replay each move at an 82+ cutoff.
# Each tuple is wins, graded rows, later wins, later rows (Mar 15 onward).
GOAL_A_TIER_RECORDS = {
    "Base Attack": (8, 22, 4, 12),
    "Weakened Armor": (10, 21, 6, 14),
    "Open Lanes": (6, 14, 4, 10),
    "Fallen Guardian": (8, 15, 6, 11),
    "Open Lanes Assault": (4, 9, 2, 6),
    "Armor Sunder": (5, 9, 2, 5),
    "Bloodlust": (4, 11, 3, 8),
    "Berserker Aggression": (3, 7, 1, 3),
    "Finisher Strike": (5, 13, 3, 9),
    "Enraged Finisher": (4, 11, 2, 8),
    "Tyr’s Wrath Unleashed": (2, 9, 1, 6),
    "Armor Annihilation": (2, 4, 2, 3),
    "Warlord Smash Attack": (2, 6, 1, 4),
    "Warlord Smash Attack (Crit)": (1, 4, 1, 3),
    "Press the Attack": (2, 4, 1, 2),
    "Press the Attack (Crit)": (2, 4, 1, 2),
    "Fury Shredder": (2, 5, 1, 3),
    "Enraged Fury Shredder": (2, 5, 1, 3),
    "Bloodlusted": (4, 8, 3, 6),
    "Bloodlusted Surge": (2, 6, 1, 4),
    "FOR VALHALLA!": (3, 7, 1, 3),
}

# Incremental 2025–26 graded Green 0.5-goal rows at Conf_Points 81.
# Add these to the 82+ records for the cumulative B-tier 81+ sample.
GOAL_B_TIER_RECORDS = {
    "Base Attack": (4, 16, 2, 9),
    "Weakened Armor": (2, 15, 1, 7),
    "Open Lanes": (0, 8, 0, 4),
    "Fallen Guardian": (3, 7, 2, 3),
    "Open Lanes Assault": (0, 3, 0, 2),
    "Armor Sunder": (2, 10, 1, 5),
    "Bloodlust": (0, 7, 0, 3),
    "Berserker Aggression": (2, 5, 1, 2),
    "Finisher Strike": (0, 7, 0, 4),
    "Enraged Finisher": (0, 2, 0, 1),
    "Tyr’s Wrath Unleashed": (0, 5, 0, 3),
    "Armor Annihilation": (0, 5, 0, 1),
    "Warlord Smash Attack": (0, 3, 0, 2),
    "Warlord Smash Attack (Crit)": (0, 1, 0, 0),
    "Press the Attack": (1, 5, 0, 1),
    "Press the Attack (Crit)": (1, 3, 0, 1),
    "Fury Shredder": (0, 2, 0, 1),
    "Enraged Fury Shredder": (0, 2, 0, 1),
    "Bloodlusted": (0, 4, 0, 3),
    "Bloodlusted Surge": (0, 1, 0, 1),
    "FOR VALHALLA!": (2, 5, 1, 2),
}


def _goals_carry_moves(r) -> list[dict]:
    """Current 0.5 Goal Carry attacks and opponent weakness effects."""
    line = _safe_float(r.get("Goal_Line"), None)
    if line is None:
        line = _safe_float(r.get("ATG_Line"), None)
    matrix = str(r.get("Matrix_Goal", "") or "").strip().casefold()
    conf_points = _safe_float(r.get("Conf_Points_Move", r.get("Conf_Points")), None)
    if line != 0.5 or matrix != "green" or conf_points is None or conf_points < 81:
        return []
    goal_tier = "S" if conf_points >= 84 else "A" if conf_points >= 82 else "B"

    goal_mu = _safe_float(r.get("Goal_mu"), None)
    opp_pk = _safe_float(r.get("Opp_PK_xGA60"), None)
    opp_sog = _safe_float(r.get("Opp_SOG_Against_L50"), None)
    opp_gaa = _safe_float(r.get("Opp_GAA"), None)
    opp_gp = _safe_float(r.get("Opp_GP"), None)
    # Current slates must not call an unconfirmed or tiny-sample starter a
    # weak goalie. Frozen old rows without this audit field keep old rules.
    if "Goalie_Context_Status" in r:
        shots_faced = _safe_float(r.get("Opp_Goalie_Shots_Faced"), None)
        if str(r.get("Goalie_Context_Status")) != "confirmed_sample_shrunk" or shots_faced is None or shots_faced < 150:
            opp_gaa = None
    # Current-season tracker rows carry GP. Do not let a one- or two-game
    # goalie line activate a historical GAA-based move. Older frozen rows
    # without GP retain their recorded rule for replay.
    if opp_gp is not None and opp_gp < 3:
        opp_gaa = None
    ixg = _safe_float(r.get("iXG%"), None)
    opp_hdca = _safe_float(r.get("opp_5v5_HDCA60"), None)
    shot_intent = _safe_float(r.get("ShotIntent"), None)
    opp_xga = _safe_float(r.get("opp_5v5_xGA60"), None)
    toi = _safe_float(r.get("TOI_per_game"), None)
    pp_toi_pct = _safe_float(r.get("PP_TOI_Pct"), None)
    star_score = _safe_float(r.get("StarScore"), None)
    attack = goal_mu is not None and goal_mu >= 0.4048
    pk_weak = opp_pk is not None and opp_pk >= 7.3
    sog_weak = opp_sog is not None and opp_sog >= 29.2
    goalie_weak = opp_gaa is not None and opp_gaa >= 3.01

    moves = []
    def add(name, kind, condition, wins, picks, later_wins, later_picks, icon,
            crit=False, experimental=False):
        a_wins = a_picks = b_wins = b_picks = 0
        if goal_tier in {"A", "B"}:
            a_wins, a_picks, a_later_wins, a_later_picks = GOAL_A_TIER_RECORDS[name]
            wins += a_wins
            picks += a_picks
            later_wins += a_later_wins
            later_picks += a_later_picks
        if goal_tier == "B":
            b_wins, b_picks, b_later_wins, b_later_picks = GOAL_B_TIER_RECORDS[name]
            wins += b_wins
            picks += b_picks
            later_wins += b_later_wins
            later_picks += b_later_picks
        moves.append({"name": name, "kind": kind, "condition": condition,
                      "wins": wins, "picks": picks, "later_wins": later_wins,
                      "later_picks": later_picks, "icon": icon, "crit": crit,
                      "goal_tier": goal_tier,
                      "record_scope": f"Points confidence {81 if goal_tier == 'B' else 82 if goal_tier == 'A' else 84}+",
                      "a_band_wins": a_wins, "a_band_picks": a_picks,
                      "b_band_wins": b_wins, "b_band_picks": b_picks,
                      "experimental": experimental})
    if attack:
        add("Base Attack", "BASE ATTACK", "Goal_mu >= 0.4048",
            59, 105, 33, 55, "base.svg")
    if pk_weak:
        add("Weakened Armor", "WEAK DEFENSE", "Opp_PK_xGA60 >= 7.3",
            57, 100, 32, 54, "shattered_armor.svg")
    if sog_weak:
        add("Open Lanes", "WEAK DEFENSE", "Opp_SOG_Against_L50 >= 29.2",
            36, 58, 20, 32, "armor_shred.svg")
    if goalie_weak:
        add("Fallen Guardian", "WEAK DEFENSE", "Opp_GAA >= 3.01",
            36, 59, 19, 32, "fenrir_claw.svg")
    if attack and sog_weak:
        add("Open Lanes Assault", "HEAVY ATTACK",
            "Goal_mu >= 0.4048 + Opp_SOG_Against_L50 >= 29.2",
            27, 39, 16, 21, "fury.svg")
    if attack and pk_weak:
        add("Armor Sunder", "HEAVY ATTACK",
            "Goal_mu >= 0.4048 + Opp_PK_xGA60 >= 7.3",
            47, 74, 27, 38, "smash.svg")
    if sog_weak and pk_weak:
        add("Bloodlust", "HEAVY ATTACK",
            "Opp_SOG_Against_L50 >= 29.2 + Opp_PK_xGA60 >= 7.3",
            29, 42, 17, 23, "armor_shred.svg")
    if attack and goalie_weak:
        add("Berserker Aggression", "HEAVY ATTACK",
            "Goal_mu >= 0.4048 + Opp_GAA >= 3.01",
            26, 36, 14, 18, "fury.svg")
    if sog_weak and ixg is not None and ixg >= 93.2:
        add("Finisher Strike", "HEAVY ATTACK",
            "Opp_SOG_Against_L50 >= 29.2 + iXG% >= 93.2",
            33, 49, 19, 27, "fenrir_claw.svg")
    if sog_weak and ixg is not None and ixg >= 96.2:
        add("Enraged Finisher", "HEAVY ATTACK",
            "Opp_SOG_Against_L50 >= 29.2 + iXG% >= 96.2",
            27, 39, 18, 24, "fenrir_claw.svg", crit=True)
    if sog_weak and pk_weak and ixg is not None and ixg >= 95:
        add("Tyr’s Wrath Unleashed", "SPECIAL ATTACK",
            "Opp_SOG_Against_L50 >= 29.2 + Opp_PK_xGA60 >= 7.3 + iXG% >= 95",
            25, 32, 16, 18, "fury.svg")
    if sog_weak and pk_weak and opp_hdca is not None and opp_hdca >= 2.5:
        add("Armor Annihilation", "SPECIAL ATTACK",
            "Opp_SOG_Against_L50 >= 29.2 + Opp_PK_xGA60 >= 7.3 + opp_5v5_HDCA60 >= 2.5",
            23, 29, 14, 17, "stack_armor_annihilation.svg")
    if sog_weak and attack and pk_weak:
        add("Warlord Smash Attack", "SPECIAL ATTACK",
            "Opp_SOG_Against_L50 >= 29.2 + Goal_mu >= 0.4048 + Opp_PK_xGA60 >= 7.3",
            23, 30, 14, 16, "smash.svg")
    if sog_weak and attack and opp_pk is not None and opp_pk >= 7.5:
        add("Warlord Smash Attack (Crit)", "SPECIAL ATTACK",
            "Opp_SOG_Against_L50 >= 29.2 + Goal_mu >= 0.4048 + Opp_PK_xGA60 >= 7.5",
            19, 24, 13, 15, "smash.svg", crit=True)
    if attack and pk_weak and opp_gaa is not None and opp_gaa >= 2.85:
        add("Press the Attack", "SPECIAL ATTACK",
            "Goal_mu >= 0.4048 + Opp_PK_xGA60 >= 7.3 + Opp_GAA >= 2.85",
            29, 39, 14, 16, "smash.svg")
    if attack and pk_weak and goalie_weak:
        add("Press the Attack (Crit)", "SPECIAL ATTACK",
            "Goal_mu >= 0.4048 + Opp_PK_xGA60 >= 7.3 + Opp_GAA >= 3.01",
            20, 25, 10, 11, "smash.svg", crit=True)
    shot_barrage = sog_weak and shot_intent is not None and shot_intent >= 3.4
    if shot_barrage:
        add("Fury Shredder", "SPECIAL ATTACK",
            "Opp_SOG_Against_L50 >= 29.2 + ShotIntent >= 3.4",
            13, 15, 6, 8, "stack_fury_shredder.svg")
    if shot_barrage and opp_xga is not None and opp_xga <= 2.8:
        add("Enraged Fury Shredder", "LAB CRIT",
            "Fury Shredder + opp_5v5_xGA60 <= 2.8",
            13, 14, 6, 7, "stack_fury_shredder.svg", crit=True, experimental=True)
    if sog_weak and pk_weak and toi is not None and toi <= 21.2:
        add("Bloodlusted", "LAB CRIT",
            "Bloodlust + TOI_per_game <= 21.2",
            19, 22, 12, 13, "armor_shred.svg", crit=True, experimental=True)
    if sog_weak and pk_weak and pp_toi_pct is not None and pp_toi_pct >= 17.2:
        add("Bloodlusted Surge", "LAB CRIT",
            "Bloodlust + PP_TOI_Pct >= 17.2",
            17, 20, 12, 12, "armor_shred.svg", crit=True, experimental=True)
    if attack and goalie_weak and star_score is not None and star_score <= 96.9:
        add("FOR VALHALLA!", "ULTIMATE ATTACK",
            "Goal_mu >= 0.4048 + Opp_GAA >= 3.01 + StarScore <= 96.9",
            18, 19, 9, 9, "valhalla.svg", experimental=True)
    return moves


def _assists_mapped_moves(r) -> list[dict]:
    """Tested Assists HUD rules. Display only; no pick or EV gate changes."""
    line = _safe_float(r.get("Assists_Line", r.get("Line_Assists")))
    matrix = str(r.get("Matrix_Assists", r.get("Matrix_A", "")) or "").strip().casefold()
    if line != 0.5 or matrix not in {"green", "🟢"}:
        return []
    conf = _safe_float(r.get("Conf_Assists", r.get("Conf_A")), default=float("nan"))
    specs = [('STANCE', 'Staff (Green Stance)', 'staff.svg', [], 711, 1591, 368, 857, False), ('STANCE', 'Staff Awakened', 'staff.svg', [], 307, 627, 162, 345, False), ('VOLUME', "Odin's Arcane Orb", 'odins_arcane_orb.svg', [('L40_A', '≥', 32.0)], 130, 229, 73, 130, True), ('VOLUME', 'Arcane Channel I', 'arcane_channel_i.svg', [('PP_Points60', '≥', 5.3)], 91, 155, 48, 82, True), ('VOLUME', 'Runic Infusion', 'runic_infusion.svg', [('PP_iXA60', '≥', 4.4)], 91, 154, 52, 87, True), ('VOLUME', 'Assist Dagger', 'magic-dagger.svg', [('Assist_Dagger', '≥', 86.0)], 125, 226, 68, 129, True), ('HEAVY', 'Arcane Channel II', 'arcane_channel_ii.svg', [('PP_Points60', '≥', 5.3), ('L40_A', '≥', 32.0)], 67, 103, 36, 51, True), ('HEAVY', 'Rune Orchestration', 'runic_infusion.svg', [('PP_Points60', '≥', 5.3), ('Assists_mu', '≥', 1.4)], 66, 101, 34, 47, False), ('HEAVY', 'Enchanted Dagger', 'magic-dagger.svg', [('Assist_Dagger', '≥', 86.0), ('L40_A', '≥', 32.0)], 77, 119, 45, 67, True), ('HEAVY', 'Stars Aligned', 'stars.svg', [('PP_Points60', '≥', 5.3), ('L40_A', '≥', 32.0), ('StarScore', '≤', 96.2)], 45, 63, 25, 30, False), ('HEAVY', 'Starlit Dagger', 'stars.svg', [('Assist_Dagger', '≥', 86.0), ('L40_A', '≥', 32.0), ('StarScore', '≤', 96.2)], 52, 70, 35, 43, False), ('HEAVY', 'Runic Dagger', 'runic_infusion.svg', [('Assist_Dagger', '≥', 86.0), ('L40_A', '≥', 32.0), ('PP_iXA60', '≥', 4.0)], 69, 100, 42, 57, False), ('SPECIAL', 'Arcane Alignment', 'odins_arcane_orb.svg', [('PP_Points60', '≥', 5.3), ('L40_A', '≥', 32.0), ('Exp_P_10', '≤', 22.0)], 45, 60, 24, 30, False), ('SPECIAL', 'Rune Orchestration II', 'runic_infusion.svg', [('PP_Points60', '≥', 5.3), ('Assists_mu', '≥', 1.4), ('Exp_P_10', '≤', 22.0)], 42, 56, 22, 26, False), ('SPECIAL', 'Dagger Ascension', 'magic-dagger.svg', [('Assist_Dagger', '≥', 86.0), ('L40_A', '≥', 32.0), ('Exp_P_10', '≤', 22.0)], 52, 70, 34, 45, True), ('SPECIAL', "Odin's Blessing", 'odins-eye.svg', [('PP_Points60', '≥', 5.3), ('L40_A', '≥', 32.0), ('Team_GF_L5', '≤', 18.0), ('Opp_PK_xGA60', '≥', 6.8)], 45, 56, 25, 30, False), ('SPECIAL', 'Runic Overdrive', 'runic_infusion.svg', [('Assist_Dagger', '≥', 86.0), ('L40_A', '≥', 32.0), ('Exp_P_10', '≤', 22.0), ('PPP10_total', '≥', 3.0)], 46, 55, 31, 38, False), ('ULTIMATE', 'Valhalla', 'valhalla.svg', [('PP_Points60', '≥', 5.3), ('L40_A', '≥', 32.0), ('Team_GF_L5', '≤', 18.0), ('Conf_Goal', '≤', 84.0)], 34, 37, 16, 17, False), ('LAB CRIT', 'Arcane Transcendence', 'magic_mans_transcendence.svg', [('PP_Points60', '≥', 5.3), ('L40_A', '≥', 32.0), ('Team_GF_L5', '≤', 18.0), ('Opp_PK_xGA60', '≥', 6.8), ('Goal_Model%', '≤', 37.0)], 38, 42, 20, 21, False), ('LAB CRIT', "Odin's Blessing CRIT", 'odins-eye.svg', [('PP_Points60', '≥', 5.3), ('L40_A', '≥', 32.0), ('Team_GF_L5', '≤', 18.0), ('Opp_PK_xGA60', '≥', 6.8), ('L20_S', '≤', 61.0)], 24, 24, 14, 14, False), ('LAB CRIT', 'Supernova Overdrive', 'supernova.svg', [('Assist_Dagger', '≥', 86.0), ('L40_A', '≥', 32.0), ('Exp_P_10', '≤', 22.0), ('PPP10_total', '≥', 3.0), ('L5_A', '≤', 4.0)], 25, 25, 15, 15, False), ('LAB CRIT', 'Arcane Supernova', 'supernova.svg', [('PP_Points60', '≥', 5.3), ('L40_A', '≥', 32.0), ('Team_GF_L5', '≤', 18.0), ('Conf_Goal', '≤', 84.0), ('SOG_Model%', '≤', 46.4)], 31, 31, 16, 16, False)]
    active = []
    for kind, name, icon, checks, wins, picks, later_wins, later_picks, track in specs:
        if kind != "STANCE" and conf < 83:
            continue
        if name == "Staff Awakened" and conf < 83:
            continue
        passed = True
        for col, op, threshold in checks:
            value = _safe_float(r.get(col), default=float("nan"))
            if not ((value >= threshold) if op == "≥" else (value <= threshold)):
                passed = False
                break
        if passed:
            extras = " + ".join(f"{col} {op} {value:g}" for col, op, value in checks)
            base = "Green + Assists 0.5" + (" + Conf_Assists ≥ 83" if name != "Staff (Green Stance)" else "")
            active.append({"kind":kind,"name":name,"icon":icon,"rule":base+(" + "+extras if extras else ""),
                           "wins":wins,"picks":picks,"later_wins":later_wins,"later_picks":later_picks,
                           "track":track})
    return active

goals_moves = _goals_carry_moves
assists_moves = _assists_mapped_moves


def fired_moves(row: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {"Points": points_moves(row), "Assists": assists_moves(row),
            "SOG": sog_moves(row), "Goal": goals_moves(row)}


def frozen_move_tags(row: Mapping[str, Any]) -> list[dict[str, str]]:
    """Capture only rules that fired before puck drop, with their identity and rule."""
    tags = []
    for market, moves in fired_moves(row).items():
        for move in moves:
            tags.append({key: str(value) for key, value in {
                "market": market,
                "name": move["name"],
                "kind": move["kind"],
                "rule": move.get("rule", move.get("condition", "")),
                "icon": move.get("icon", ""),
                "kit_version": VERSION,
            }.items()})
    return tags

