"""Graded 2026 Points and SOG display moves.

These are historical subsets, not model probabilities. The rules only label
tracker rows; they do not change model predictions or eligibility.
Discovery: Jan 18–Apr 13, 2026. "Later" is Mar 15 onward, an analysis split.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


VERSION = "2026-09-24-frozen-v2"


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
    conf = _number(row, "Conf_Points")
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
                           "Conf_Points >= 80", 226, 367, 153, 254))
        if _at_least(l10_s, 17):
            moves.append(_move("Echo Stomp I", "TIER", "PTS05_HAMMER_STOMP.svg",
                               "Conf_Points >= 80 + L10_S >= 17", 189, 287, 133, 203))
        if _at_least(toi, 19.18):
            moves.append(_move("Echo Stomp II", "TIER", "PTS05_HAMMER_STOMP.svg",
                               "Conf_Points >= 80 + TOI_per_game >= 19.18", 161, 234, 111, 167))
        if _at_least(assists_mu, 0.746):
            moves.append(_move("Gaia’s Blessing", "TIER", "PTS05_GAIAS_BLESSING.svg",
                               "Conf_Points >= 80 + Assists_mu >= 0.746", 158, 242, 111, 171))
            if _at_least(conf, 81):
                moves.append(_move("Gaia’s Blessing+ (Press)", "HEAVY", "PTS05_GAIAS_BLESSING.svg",
                                   "Conf_Points >= 81 + Assists_mu >= 0.746", 138, 206, 97, 145))
            if _at_least(conf, 83):
                moves.append(_move("Gaia’s Blessing++ (Smash)", "HEAVY", "PTS05_GAIAS_BLESSING.svg",
                                   "Conf_Points >= 83 + Assists_mu >= 0.746", 98, 141, 68, 99))
        if _at_least(toi, 19.18) and _at_least(l10_s, 17):
            moves.append(_move("Hammer Fists II", "HEAVY", "PTS05_HAMMER_FISTS.svg",
                               "Conf_Points >= 80 + TOI_per_game >= 19.18 + L10_S >= 17",
                               146, 207, 102, 148))
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
                       "Conf_Points >= 75", 136, 331, 66, 173))
    conf_goal = _number(row, "Conf_Goal")
    conf_sog = _number(row, "Conf_SOG")
    volume = _at_most(conf_goal, 88) and _at_least(conf_sog, 80)
    if volume:
        moves.append(_move("Backbone", "FLOOR", "PTS15_TWO_HANDED_HAMMER.svg",
                           "Conf_Points >= 75 + Conf_Goal <= 88 + Conf_SOG >= 80",
                           73, 142, 31, 55))
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
    share = _number(row, "Player_5v5_SOG_Share")

    if line == 2.5:
        moves.append(_move("Jungle Stance", "STANCE", "sog_basic_swipe.svg",
                           "Green + SOG 2.5 + Conf_SOG >= 75", 103, 200, 60, 114))
        ixg = _number(row, "iXG%")
        if _at_least(ixg, 93.2):
            moves.append(_move("Berserker Swipe", "FLOOR", "sog_basic_swipe.svg",
                               "iXG% >= 93.2", 78, 136, 46, 77))
        if _at_least(_number(row, "Goalie_Weak"), 62.5):
            moves.append(_move("Paralysis", "ENV", "sog_env_paralysis.svg",
                               "Goalie_Weak >= 62.5", 34, 53, 23, 34))
        rage = _at_least(_number(row, "L20_A"), 9) and _at_most(_number(row, "L40_A"), 29)
        volley = _at_least(_number(row, "L10_Rate_SOG"), 2.9) and _at_least(share, 18.8)
        if rage:
            moves.append(_move("Berserker’s Rage", "HEAVY", "sog_berserkers_rage.svg",
                               "L20_A >= 9 + L40_A <= 29", 50, 75, 36, 53))
        if volley:
            moves.append(_move("Berserker Volley", "SUPPORT", "sog_berserker_volley.svg",
                               "L10_Rate_SOG >= 2.9 + SOG share >= 18.8", 34, 50, 21, 33,
                               track=True))
        if rage and volley:
            moves.append(_move("Assassin’s Overdrive", "CRIT", "sog_assassins_overdrive.svg",
                               "Rage + Volley", 26, 33, 18, 22, track=True))
        if _at_least(_number(row, "PP_TOI_Pct"), 15.9) and _at_least(ixg, 93.2):
            moves.append(_move("Bloodthirst", "HEAVY", "sog_bloodthirst.svg",
                               "PP_TOI_Pct >= 15.9 + iXG% >= 93.2", 44, 67, 22, 37))
        if rage and _at_least(_number(row, "i5v5_points60"), 2.2):
            moves.append(_move("Siege", "ULTIMATE", "sog_berserker_siege.svg",
                               "Rage + i5v5_points60 >= 2.2", 32, 41, 21, 26, track=True))
        return moves

    moves.append(_move("Sniper Stance", "STANCE", "sog_class_shooter.svg",
                       "Green + SOG 3.5 + Conf_SOG >= 75", 136, 338, 79, 193))
    hdca = _at_least(_number(row, "opp_5v5_HDCA60"), 2.36)
    strong = hdca and _at_least(_number(row, "PPP10_total"), 3)
    permission = hdca and _at_least(_number(row, "Best_Conf"), 89)
    assist_model = _at_least(_number(row, "Assists_Model%"), 56.3)
    if strong:
        moves.append(_move("Strong", "FLOOR", "sog_berserkers_rage.svg",
                           "PPP10_total >= 3 + opp_5v5_HDCA60 >= 2.36",
                           66, 124, 35, 69))
    if permission:
        moves.append(_move("Permission Special", "SUPPORT", "sog_env_paralysis.svg",
                           "Best_Conf >= 89 + opp_5v5_HDCA60 >= 2.36",
                           60, 113, 36, 69, track=True))
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


def _goals_carry_moves(r) -> list[dict]:
    """Current 0.5 Goal Carry attacks and opponent weakness effects."""
    line = _safe_float(r.get("Goal_Line"), None)
    if line is None:
        line = _safe_float(r.get("ATG_Line"), None)
    matrix = str(r.get("Matrix_Goal", "") or "").strip().casefold()
    conf_points = _safe_float(r.get("Conf_Points"), None)
    if line != 0.5 or matrix != "green" or conf_points is None or conf_points < 84:
        return []

    goal_mu = _safe_float(r.get("Goal_mu"), None)
    opp_pk = _safe_float(r.get("Opp_PK_xGA60"), None)
    opp_sog = _safe_float(r.get("Opp_SOG_Against_L50"), None)
    opp_gaa = _safe_float(r.get("Opp_GAA"), None)
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
        moves.append({"name": name, "kind": kind, "condition": condition,
                      "wins": wins, "picks": picks, "later_wins": later_wins,
                      "later_picks": later_picks, "icon": icon, "crit": crit,
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
    specs = [('STANCE', 'Staff (Green Stance)', 'staff.svg', [], 724, 1616, 381, 882, False), ('STANCE', 'Staff Awakened', 'staff.svg', [], 310, 633, 165, 351, False), ('VOLUME', "Odin's Arcane Orb", 'odins_arcane_orb.svg', [('L40_A', '≥', 32.0)], 131, 232, 74, 133, True), ('VOLUME', 'Arcane Channel I', 'arcane_channel_i.svg', [('PP_Points60', '≥', 5.3)], 92, 157, 49, 84, True), ('VOLUME', 'Runic Infusion', 'runic_infusion.svg', [('PP_iXA60', '≥', 4.4)], 92, 157, 53, 90, True), ('VOLUME', 'Assist Dagger', 'magic-dagger.svg', [('Assist_Dagger', '≥', 86.0)], 126, 229, 69, 132, True), ('HEAVY', 'Arcane Channel II', 'arcane_channel_ii.svg', [('PP_Points60', '≥', 5.3), ('L40_A', '≥', 32.0)], 68, 105, 37, 53, True), ('HEAVY', 'Rune Orchestration', 'runic_infusion.svg', [('PP_Points60', '≥', 5.3), ('Assists_mu', '≥', 1.4)], 67, 102, 35, 48, False), ('HEAVY', 'Enchanted Dagger', 'magic-dagger.svg', [('Assist_Dagger', '≥', 86.0), ('L40_A', '≥', 32.0)], 78, 122, 46, 70, True), ('HEAVY', 'Stars Aligned', 'stars.svg', [('PP_Points60', '≥', 5.3), ('L40_A', '≥', 32.0), ('StarScore', '≤', 96.2)], 46, 65, 26, 32, False), ('HEAVY', 'Starlit Dagger', 'stars.svg', [('Assist_Dagger', '≥', 86.0), ('L40_A', '≥', 32.0), ('StarScore', '≤', 96.2)], 53, 73, 36, 46, False), ('HEAVY', 'Runic Dagger', 'runic_infusion.svg', [('Assist_Dagger', '≥', 86.0), ('L40_A', '≥', 32.0), ('PP_iXA60', '≥', 4.0)], 70, 103, 43, 60, False), ('SPECIAL', 'Arcane Alignment', 'odins_arcane_orb.svg', [('PP_Points60', '≥', 5.3), ('L40_A', '≥', 32.0), ('Exp_P_10', '≤', 22.0)], 46, 62, 25, 32, False), ('SPECIAL', 'Rune Orchestration II', 'runic_infusion.svg', [('PP_Points60', '≥', 5.3), ('Assists_mu', '≥', 1.4), ('Exp_P_10', '≤', 22.0)], 43, 57, 23, 27, False), ('SPECIAL', 'Dagger Ascension', 'magic-dagger.svg', [('Assist_Dagger', '≥', 86.0), ('L40_A', '≥', 32.0), ('Exp_P_10', '≤', 22.0)], 53, 73, 35, 48, True), ('SPECIAL', "Odin's Blessing", 'odins-eye.svg', [('PP_Points60', '≥', 5.3), ('L40_A', '≥', 32.0), ('Team_GF_L5', '≤', 18.0), ('Opp_PK_xGA60', '≥', 6.8)], 46, 57, 26, 31, False), ('SPECIAL', 'Runic Overdrive', 'runic_infusion.svg', [('Assist_Dagger', '≥', 86.0), ('L40_A', '≥', 32.0), ('Exp_P_10', '≤', 22.0), ('PPP10_total', '≥', 3.0)], 47, 58, 32, 41, False), ('ULTIMATE', 'Valhalla', 'valhalla.svg', [('PP_Points60', '≥', 5.3), ('L40_A', '≥', 32.0), ('Team_GF_L5', '≤', 18.0), ('Conf_Goal', '≤', 84.0)], 35, 38, 17, 18, False), ('LAB CRIT', 'Arcane Transcendence', 'magic_mans_transcendence.svg', [('PP_Points60', '≥', 5.3), ('L40_A', '≥', 32.0), ('Team_GF_L5', '≤', 18.0), ('Opp_PK_xGA60', '≥', 6.8), ('Goal_Model%', '≤', 37.0)], 38, 42, 20, 21, False), ('LAB CRIT', "Odin's Blessing CRIT", 'odins-eye.svg', [('PP_Points60', '≥', 5.3), ('L40_A', '≥', 32.0), ('Team_GF_L5', '≤', 18.0), ('Opp_PK_xGA60', '≥', 6.8), ('L20_S', '≤', 61.0)], 25, 25, 15, 15, False), ('LAB CRIT', 'Supernova Overdrive', 'supernova.svg', [('Assist_Dagger', '≥', 86.0), ('L40_A', '≥', 32.0), ('Exp_P_10', '≤', 22.0), ('PPP10_total', '≥', 3.0), ('L5_A', '≤', 4.0)], 26, 27, 16, 17, False), ('LAB CRIT', 'Arcane Supernova', 'supernova.svg', [('PP_Points60', '≥', 5.3), ('L40_A', '≥', 32.0), ('Team_GF_L5', '≤', 18.0), ('Conf_Goal', '≤', 84.0), ('SOG_Model%', '≤', 46.4)], 32, 32, 17, 17, False)]
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
