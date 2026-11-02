"""Approved StarScore Goal route and its separately graded enhancements."""

from goal_moves_current import _number, goal_profile_score, GOAL_KIT_VERSION


ENTRY = "Green + Goal 0.5 + StarScore >= 84.5 + goal drought >= 1 + TOI rank >= 81"
SHARED = (("StarScore", ">=", 84.5), ("Drought_G", ">=", 1), ("TOI_Pct", ">=", 81))
# id, name, kind, icon, extra checks, goals, plays, mean book BE, players, dates
ATTACKS = (
    ("drought_i", "Drought Strike I", "HEAVY ATTACK", "fury.svg", (), 47, 89, 38.4, 27, 17),
    ("drought_ii", "Drought Strike II", "HEAVY ATTACK", "fury.svg",
     (("Goal_Profile", ">=", 93),), 44, 81, 38.4, 24, 16),
    ("reaper", "Reaper's Return", "SPECIAL ATTACK", "fury.svg",
     (("L10_S", "<=", 35),), 38, 60, 37.5, 24, 16),
    ("measured", "Measured Strike", "SPECIAL ATTACK", "armor_shred.svg",
     (("ShotIntent", "<=", 3.8),), 39, 64, 37.8, 26, 16),
    ("arcane", "Arcane Strike", "SPECIAL ATTACK", "smash.svg",
     (("Points_mu", ">=", 1.4),), 37, 60, 39.7, 19, 17),
    ("precision", "Precision Crit", "CRIT", "armor_shred.svg",
     (("ShotIntent_Pct", "<=", 90),), 24, 33, 36.9, 19, 13),
    ("controlled", "Controlled Crit", "CRIT", "smash.svg",
     (("Avg5_SOG", "<=", 3.2),), 25, 36, 38.1, 19, 14),
    ("patient", "Patient Crit", "CRIT", "fury.svg",
     (("L10_G", "<=", 4),), 26, 38, 35.8, 21, 13),
)

# Each record belongs to one parent attack plus ONE environment condition.
BUFFS = (
    ("barrage", "Barrage", "team_5v5_SF60", 25.8,
     {"reaper": (20, 24, 36.8, 14, 12), "measured": (23, 29, 37.0, 17, 12),
      "arcane": (23, 33, 38.9, 14, 12)}),
    ("pk_breach", "PK Breach", "Opp_PK_GA60", 8.2,
     {"reaper": (13, 17, 37.4, 10, 9), "measured": (16, 23, 37.9, 15, 10),
      "arcane": (16, 24, 39.4, 12, 11)}),
    ("power_surge", "Power Surge", "Team_PP_xGF60", 8.0,
     {"reaper": (23, 32, 38.1, 13, 12), "measured": (26, 37, 38.4, 16, 12),
      "arcane": (20, 32, 39.6, 12, 12)}),
)

LABELS = {"Goal_Profile": "Goal confidence", "L10_S": "L10 shots",
          "ShotIntent": "shot intent", "Points_mu": "points model mean",
          "ShotIntent_Pct": "shot-intent rank", "Avg5_SOG": "L5 SOG/game",
          "L10_G": "L10 goals", "team_5v5_SF60": "team 5v5 shots/60",
          "Opp_PK_GA60": "opponent PK goals allowed/60", "Team_PP_xGF60": "team PP xGF/60"}


def _passes(row, checks):
    for field, op, cutoff in checks:
        value = goal_profile_score(row) if field == "Goal_Profile" else _number(row, field)
        if value is None or not (value >= cutoff if op == ">=" else value <= cutoff):
            return False
    return True


def _extra_text(checks):
    return " + ".join(f"{LABELS.get(field, field)} {op} {cutoff:g}" for field, op, cutoff in checks)


def _record(key, name, kind, icon, rule, record, *, buff=False, parent=None):
    wins, picks, be, players, dates = record
    return {"id": f"goal_star:{key}", "name": name, "research_name": name,
            "route_family": "star_drought", "market": "Goal", "kind": kind,
            "icon": icon, "rule": rule, "condition": rule, "wins": wins, "picks": picks,
            "later_wins": 0, "later_picks": 0, "crit": kind == "CRIT", "experimental": False,
            "sample_season": "2026-27", "sample_from": "2026-10-13", "sample_through": "2026-10-31",
            "sample_book_be_pct": be, "sample_players": players, "sample_dates": dates,
            "kit_version": GOAL_KIT_VERSION, "matchup_buff": buff, "parent_attack": parent,
            "xg_boost": False, "level": "II" if key == "drought_ii" else "I" if key == "drought_i" else None,
            "profile_cutoff": 93 if key == "drought_ii" else None}


def star_goals_moves(row):
    line = _number(row, "Goal_Line")
    if line is None:
        line = _number(row, "ATG_Line")
    if (line != .5 or str(row.get("Matrix_Goal", "")).strip().casefold() != "green"
            or str(row.get("Model_Stats_Season", "")) != "2026-2027"
            or str(row.get("Form_Season", "")) != "2026-27" or not _passes(row, SHARED)):
        return []
    moves = []
    active = {}
    for key, name, kind, icon, checks, wins, picks, be, players, dates in ATTACKS:
        if not _passes(row, checks):
            continue
        extra = _extra_text(checks)
        move = _record(key, name, kind, icon, ENTRY + (" + " + extra if extra else ""),
                       (wins, picks, be, players, dates))
        active[key] = move
        moves.append(move)
    if str(row.get("Team_Stats_Season", "")) == "2026-2027":
        for key, label, field, cutoff, records in BUFFS:
            if not _passes(row, ((field, ">=", cutoff),)):
                continue
            for parent_key, record in records.items():
                attack = active.get(parent_key)
                if attack is None:
                    continue
                name = f"{label} · {attack['name']}"
                rule = attack["rule"] + " + verified 2026-27 team stats + " + _extra_text(((field, ">=", cutoff),))
                moves.append(_record(f"{key}:{parent_key}", name, "MATCHUP BUFF", "armor_shred.svg",
                                     rule, record, buff=True, parent=attack["name"]))
    return moves
