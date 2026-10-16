"""Descriptive pregame opportunity views; never calibrated hit probabilities.

The existing Warlord moves and Green rules are intentionally left unchanged.
Every value is saved with the frozen slate so later grades can test whether an
opportunity view adds value beyond the historical rules and posted price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_opportunity_views(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    def num(name: str) -> pd.Series:
        return pd.to_numeric(out.get(name, pd.Series(np.nan, index=out.index)), errors="coerce")

    ev_minutes = num("EV5_TOI_per_game")
    pp_minutes = num("PP_TOI_per_game")
    ev_sog60 = num("i5v5_SOG60")
    pp_sog60 = num("PP_SOG60")
    ev_sog = (ev_minutes * ev_sog60 / 60.0).where(ev_minutes.ge(0) & ev_sog60.ge(0))
    pp_sog = (pp_minutes * pp_sog60 / 60.0).where(pp_minutes.ge(0) & pp_sog60.ge(0))
    pp_sog = pp_sog.where(~pp_minutes.eq(0), 0.0)
    out["SOG_Opportunity_EV5"] = ev_sog.round(3)
    out["SOG_Opportunity_PP"] = pp_sog.round(3)
    out["SOG_Opportunity_Base"] = (ev_sog + pp_sog.fillna(0)).where(ev_sog.notna()).round(3)
    out["SOG_Opportunity_Coverage"] = np.select(
        [ev_sog.notna() & pp_sog.notna(), ev_sog.notna()],
        ["5v5+PP", "5v5 only"], default="unavailable",
    )

    # Opponent shots allowed is a team-level context, not the player's personal
    # shot probability. Ten observed games are treated as a modest sample.
    opp_sog = num("Opp_SOG_Against_L10")
    opp_games = num("Opp_SOG_Used_L10").fillna(0).clip(lower=0)
    raw_factor = (opp_sog / 30.0).clip(lower=0.85, upper=1.15)
    strength = opp_games / (opp_games + 10.0)
    factor = (1.0 + (raw_factor - 1.0) * strength).where(opp_sog.gt(0), 1.0)
    out["SOG_Opp_Volume_Factor"] = factor.round(3)
    out["SOG_Opportunity_Matchup"] = (out["SOG_Opportunity_Base"] * factor).round(3)

    onice = num("PP_OnIce_xGF60")
    individual = num("PP_iXG60")
    out["PP_OnIce_xG_per_game"] = (onice * pp_minutes / 60.0).where(
        onice.ge(0) & pp_minutes.ge(0)
    ).round(3)
    out["PP_iXG_per_game"] = (individual * pp_minutes / 60.0).where(
        individual.ge(0) & pp_minutes.ge(0)
    ).round(3)
    ev_ixg = (num("i5v5_xG60") * ev_minutes / 60.0).where(
        num("i5v5_xG60").ge(0) & ev_minutes.ge(0)
    )
    out["Goal_Opportunity_xG"] = (ev_ixg + out["PP_iXG_per_game"].fillna(0)).where(ev_ixg.notna()).round(3)
    out["Assist_5v5_Setups_per_game"] = (num("i5v5_shotAssists60") * ev_minutes / 60.0).where(
        num("i5v5_shotAssists60").ge(0) & ev_minutes.ge(0)
    ).round(3)
    out["PP_Observed_Assists_per_game"] = (num("PP_WeightedAssists60") * pp_minutes / 60.0).where(
        num("PP_WeightedAssists60").ge(0) & pp_minutes.ge(0)
    ).round(3)
    # An alternative to the legacy 55/45 context blend, retained only for
    # forward comparison. Rank percentages are not probabilities.
    out["PP_Matchup_Equal"] = (0.5 * num("PP_Score") + 0.5 * num("Opp_PK_Weak")).round(1)
    out["Opportunity_View_Version"] = "2026-10-16-v1"
    return out
