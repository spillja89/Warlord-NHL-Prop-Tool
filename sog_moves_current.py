"""Approved 2026-27 SOG2.5 Jungle kit. Fixed records, not probabilities."""
from __future__ import annotations
import math

from sog_confidence import VERSION as CONFIDENCE_VERSION

from sog_matrix import VERSION as MATRIX_VERSION

LEGACY_VERSION = "2026-11-04-sog25-current-v1"
VERSION = "2026-11-04-sog25-green60-v2"
SOURCE = "NHL Stats · current season"


def _num(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def is_current_sog25(row):
    return (_num(row.get("SOG_Line")) == 2.5
            and row.get("Conf_SOG_Version") == CONFIDENCE_VERSION)


def sog_base_eligible(row):
    conf = _num(row.get("Conf_SOG"))
    return (is_current_sog25(row) and row.get("Model_Stats_Season") == "2026-2027"
            and conf is not None and conf >= 67
            and str(row.get("Matrix_SOG", "")).strip().casefold() in {"green", "🟢"})


def pk_duration_seconds(row):
    """Exact pregame PK seconds/opportunity; never round before the104 gate."""
    gp = _num(row.get("Opp_PKO_GP"))
    if row.get("PPO_Source") != SOURCE or gp is None or gp < 3:
        return None
    minutes = _num(row.get("Opp_PK_Time_PG"))
    opportunities = _num(row.get("Opp_PKO_PG"))
    if minutes is not None and minutes >= 0 and opportunities is not None and opportunities > 0:
        return 60.0 * minutes / opportunities
    exact = _num(row.get("Opp_PK_Seconds_Per_Opportunity"))
    return exact if exact is not None and exact >= 0 else None


def _move(name, kind, icon, rule, wins, picks, be, players, dates, **extra):
    return {"name":name,"kind":kind,"icon":icon,"rule":rule,
      "wins":wins,"picks":picks,"later_wins":0,"later_picks":0,"track":False,
      "market":"SOG","kit_version":VERSION,"sog_current":True,
      "sample_season":"2026-27","sample_from":"2026-10-13","sample_through":"2026-11-03",
      "sample_book_be_pct":be,"sample_players":players,"sample_dates":dates,**extra}


LEGACY_RECORDS = {'Berserker Swipe': [31, 52, 51.21717476347496, 15, 19], 'Berserker’s Rage': [27, 37, 51.50987491351554, 13, 18], 'Paralysis': [28, 40, 51.3778968767278, 14, 17], 'Assassin’s Overdrive': [24, 31, 52.044069416904534, 13, 16]}
CURRENT_RECORDS = {'Berserker Swipe': [50, 84, 50.11407533326924, 24, 20], 'Berserker’s Rage': [46, 69, 50.03122626223179, 22, 20], 'Paralysis': [42, 64, 50.19089625954928, 21, 18], 'Assassin’s Overdrive': [38, 55, 50.372138863019664, 20, 18]}


def version_for(row):
    return VERSION if row.get("Matrix_SOG_Version") == MATRIX_VERSION else LEGACY_VERSION


def sog_moves(row):
    if not sog_base_eligible(row):
        return []
    version = version_for(row)
    records = CURRENT_RECORDS if version == VERSION else LEGACY_RECORDS
    base = "SOG2.5 + Matrix_SOG Green + Conf_SOG >=67 (Shot Profile v2)"
    if version == VERSION:
        base += " · Green: original profile/Elite OR remaining shot checks + iXG% >=60"
    def move(name, kind, icon, rule, **extra):
        return _move(name,kind,icon,rule,*records[name],kit_version=version,**extra)
    moves = [move("Berserker Swipe","BASE","sog_basic_swipe.svg",base)]
    stronger = _num(row.get("Conf_SOG")) >= 71
    seconds = pk_duration_seconds(row)
    buff = seconds is not None and seconds <= 104
    if stronger:
        moves.append(move("Berserker’s Rage","HEAVY","sog_berserkers_rage.svg",
            base+" + Conf_SOG >=71"))
    if buff:
        moves.append(move("Paralysis","ENV","sog_env_paralysis.svg",
            base+" + opponent PK seconds/opportunity <=104",matchup_buff=True))
    if stronger and buff:
        moves.append(move("Assassin’s Overdrive","SPECIAL","sog_assassins_overdrive.svg",
            base+" + Conf_SOG >=71 + opponent PK seconds/opportunity <=104",empowered=True))
    from sog_profiles_2026 import profile_moves
    return moves + profile_moves(row)


def attach_sog_route_context(frame):
    """Save exact duration, strongest attack and all buffs with the tracker."""
    import pandas as pd
    out = frame.copy()
    rows = out.to_dict("records")
    durations = [pk_duration_seconds(row) for row in rows]
    out["Opp_PK_Seconds_Per_Opportunity"] = pd.Series(durations,index=out.index,dtype="Float64")
    out["Opp_PK_Duration_Source"] = [SOURCE if v is not None else "unavailable" for v in durations]
    moves = [sog_moves(row) for row in rows]
    from move_return_grades import grade_rank
    from sog_profiles_2026 import active_moves
    def attack(items):
        active=[m for m in active_moves(items) if not m.get('matchup_buff')]
        return max(active,key=lambda m:grade_rank(m) or (0,False,m['wins']/m['picks'],0,m['picks'],m['name'])) if active else None
    attacks = [attack(items) for items in moves]
    out["SOG_Attack"] = [m["name"] if m else "" for m in attacks]
    out["SOG_Buff"] = [' · '.join(m['name'] for m in items if m.get('matchup_buff')) for items in moves]
    out["SOG_Move_Record"] = [f'{m["wins"]}/{m["picks"]}' if m else "" for m in attacks]
    out["SOG_Move_Pct"] = [100*m["wins"]/m["picks"] if m else None for m in attacks]
    out["SOG_Sample_BE_Pct"] = [m["sample_book_be_pct"] if m else None for m in attacks]
    out["SOG_Move_Kit_Version"] = [m['kit_version'] if m else version_for(row) for m,row in zip(attacks,rows)]
    from sog_profiles_2026 import VERSION as PROFILE_VERSION, profile_values, profile_moves
    values=[profile_values(row) for row in rows]
    for field in ('L20_S','Player_Log_L20_S_hit_pct','Player_Log_L20_G_hit_pct','Player_Log_L5_TOI_per_game','Player_Log_L10_S_per_game','Drought_G','SOG_Profile_Log_GP'):
        out['SOG_Profile_'+field]=[v.get(field) for v in values]
    out['SOG_Profile_Kit_Version']=PROFILE_VERSION
    out['SOG_Research_Monitors']=[' · '.join(m['name'] for m in profile_moves(row,monitors=True)) for row in rows]
    out['SOG_Fired_Profile_Routes']=[' · '.join(m['name'] for m in items if m.get('sog_profile')) for items in moves]
    return out

