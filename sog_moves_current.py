"""Approved 2026-27 SOG2.5 Jungle kit. Fixed records, not probabilities."""
from __future__ import annotations
import math

from sog_confidence import VERSION as CONFIDENCE_VERSION

VERSION = "2026-11-04-sog25-current-v1"
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


def sog_moves(row):
    if not sog_base_eligible(row):
        return []
    base = "SOG2.5 + Matrix_SOG Green + Conf_SOG >=67 (Shot Profile v2)"
    moves = [_move("Berserker Swipe","BASE","sog_basic_swipe.svg",base,
                   31,52,51.21717476347496,15,19)]
    stronger = _num(row.get("Conf_SOG")) >= 71
    seconds = pk_duration_seconds(row)
    buff = seconds is not None and seconds <= 104
    if stronger:
        moves.append(_move("Berserker’s Rage","HEAVY","sog_berserkers_rage.svg",
          base+" + Conf_SOG >=71",27,37,51.50987491351554,13,18))
    if buff:
        moves.append(_move("Paralysis","ENV","sog_env_paralysis.svg",
          base+" + opponent PK seconds/opportunity <=104",28,40,51.3778968767278,14,17,
          matchup_buff=True))
    if stronger and buff:
        moves.append(_move("Assassin’s Overdrive","SPECIAL","sog_assassins_overdrive.svg",
          base+" + Conf_SOG >=71 + opponent PK seconds/opportunity <=104",
          24,31,52.044069416904534,13,16,empowered=True))
    return moves


def attach_sog_route_context(frame):
    """Save exact duration, strongest attack and all buffs with the tracker."""
    import pandas as pd
    out = frame.copy()
    rows = out.to_dict("records")
    durations = [pk_duration_seconds(row) for row in rows]
    out["Opp_PK_Seconds_Per_Opportunity"] = pd.Series(durations,index=out.index,dtype="Float64")
    out["Opp_PK_Duration_Source"] = [SOURCE if v is not None else "unavailable" for v in durations]
    moves = [sog_moves(row) for row in rows]
    attacks = [next((m for m in reversed(items) if not m.get("matchup_buff")),None) for items in moves]
    out["SOG_Attack"] = [m["name"] if m else "" for m in attacks]
    out["SOG_Buff"] = ["Paralysis" if any(m.get("matchup_buff") for m in items) else "" for items in moves]
    out["SOG_Move_Record"] = [f'{m["wins"]}/{m["picks"]}' if m else "" for m in attacks]
    out["SOG_Move_Pct"] = [100*m["wins"]/m["picks"] if m else None for m in attacks]
    out["SOG_Sample_BE_Pct"] = [m["sample_book_be_pct"] if m else None for m in attacks]
    out["SOG_Move_Kit_Version"] = VERSION
    return out
