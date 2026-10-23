"""Graded display moves for the NHL prop markets.

These are historical subsets, not model probabilities. The rules only label
tracker rows; they do not change model predictions or eligibility.
Regular-season archive: Jan 18–Apr 13, 2026. Apr 18/20 playoff rows are
excluded. "Later" is Mar 15 onward within that regular-season archive.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


VERSION = "2026-10-23-assist-current-v1"


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

from goal_moves_current import goals_moves
from assist_moves_current import assists_moves

# Preserve the previous rules for saved-slate comparisons. Live cards use the
# current Assist tree above; historical replay scripts can use this explicitly.
legacy_assists_moves = _assists_mapped_moves


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
                "kit_version": move.get("kit_version", VERSION),
            }.items()})
    return tags

