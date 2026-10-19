"""Pregame goal projection kept separate from the frozen Warlord move kit.

This is a research forecast. Its assumptions are saved with every slate so
future grades can measure calibration before it affects card selection.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


VERSION = "2026-10-19-research-v1"
MULTI_MARKET_VERSION = "2026-10-19-research-v2"


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame.get(name, pd.Series(np.nan, index=frame.index)), errors="coerce")


def add_tonight_goal_projection(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a matchup-aware goal count and 1+ rate without changing Goal_mu.

    Ten games/expected goals are neutral priors, and factors are capped to
    prevent a few early games from dominating. These choices are provisional;
    the saved forecasts must be evaluated forward before being used as edges.
    """
    out = frame.copy()
    if out.empty:
        for name in ("Tonight_Goal_xG", "Tonight_Goal_1plus_pct",
                     "Tonight_Goal_EV_xG", "Tonight_Goal_PP_xG",
                     "Tonight_Goal_EV_factor", "Tonight_Goal_PK_factor",
                     "Tonight_Goal_Goalie_factor", "Tonight_Goal_vs_Book_pp"):
            out[name] = pd.Series(dtype=float)
        out["Tonight_Goal_Status"] = pd.Series(dtype=str)
        out["Tonight_Goal_Version"] = pd.Series(dtype=str)
        return out

    player_gp = _numeric(out, "Model_Player_GP").clip(lower=0)
    ev_minutes = _numeric(out, "EV5_TOI_per_game")
    ev_rate = _numeric(out, "i5v5_xG60")
    pp_minutes = _numeric(out, "PP_TOI_per_game")
    pp_rate = _numeric(out, "PP_iXG60")

    ev_xg = (ev_minutes * ev_rate / 60.0).where(ev_minutes.ge(0) & ev_rate.ge(0))
    pp_xg = (pp_minutes * pp_rate / 60.0).where(pp_minutes.ge(0) & pp_rate.ge(0))
    pp_xg = pp_xg.where(~pp_minutes.eq(0), 0.0)

    opp_gp = _numeric(out, "Opp_Team_GP").clip(lower=0)
    team_reliability = opp_gp / (opp_gp + 10.0)
    ev_opp = _numeric(out, "opp_5v5_xGA60")
    ev_league = _numeric(out, "League_5v5_xGA60")
    pk_opp = _numeric(out, "Opp_PK_xGA60")
    pk_league = _numeric(out, "League_PK_xGA60")

    # The legacy tracker uses 50 as a missing-data sentinel for 5v5 rates.
    ev_valid = ev_opp.between(0.5, 8.0) & ev_league.between(0.5, 8.0)
    pk_valid = pk_opp.between(1.0, 25.0) & pk_league.between(1.0, 25.0)
    ev_factor = (1.0 + team_reliability * (ev_opp / ev_league - 1.0)).clip(0.80, 1.20)
    pk_factor = (1.0 + team_reliability * (pk_opp / pk_league - 1.0)).clip(0.80, 1.20)
    ev_factor = ev_factor.where(ev_valid, 1.0).fillna(1.0)
    pk_factor = pk_factor.where(pk_valid, 1.0).fillna(1.0)

    goalie_status = out.get("Opp_Goalie_Status", pd.Series("", index=out.index)).astype(str).str.casefold()
    goalie_confirmed = goalie_status.isin({"confirmed", "official"})
    goalie_xga = _numeric(out, "Opp_Goalie_xGA")
    goalie_ga = _numeric(out, "Opp_Goalie_GA")
    goalie_factor = ((goalie_ga + 10.0) / (goalie_xga + 10.0)).clip(0.85, 1.15)
    goalie_factor = goalie_factor.where(goalie_confirmed & goalie_xga.gt(0) & goalie_ga.ge(0), 1.0).fillna(1.0)

    opportunity = ((ev_xg * ev_factor + pp_xg.fillna(0) * pk_factor)
                   * goalie_factor).where(ev_xg.notna())
    base_mu = _numeric(out, "Goal_mu")
    player_reliability = (player_gp / (player_gp + 10.0)).fillna(0.0)
    projected = (base_mu * (1.0 - player_reliability) + opportunity * player_reliability)
    projected = projected.where(opportunity.notna() & base_mu.gt(0))
    projected = projected.clip(lower=0.0, upper=2.0)

    # For 1+ goals, NB(alpha=.4) P(X=0) = (1 + alpha*mu)^(-1/alpha).
    # This is the same count family and alpha as the existing Goal model.
    probability = projected.map(
        lambda value: 100.0 * (1.0 - math.pow(1.0 + 0.4 * float(value), -2.5))
        if pd.notna(value) else np.nan
    )
    book_line = _numeric(out, "Goal_Line")
    book_odds = _numeric(out, "Goal_Odds_Over")
    atg_line = _numeric(out, "ATG_Line")
    atg_odds = _numeric(out, "ATG_Odds_Over")
    use_atg = book_line.isna() | book_odds.isna()
    book_line = book_line.where(~use_atg, atg_line)
    book_odds = book_odds.where(~use_atg, atg_odds)
    break_even = pd.Series(np.nan, index=out.index, dtype=float)
    pos = book_odds.gt(0)
    neg = book_odds.lt(0)
    break_even.loc[pos] = 10000.0 / (100.0 + book_odds.loc[pos])
    break_even.loc[neg] = 100.0 * -book_odds.loc[neg] / (100.0 - book_odds.loc[neg])

    out["Tonight_Goal_EV_xG"] = ev_xg.round(3)
    out["Tonight_Goal_PP_xG"] = pp_xg.round(3)
    out["Tonight_Goal_EV_factor"] = ev_factor.round(3)
    out["Tonight_Goal_PK_factor"] = pk_factor.round(3)
    out["Tonight_Goal_Goalie_factor"] = goalie_factor.round(3)
    out["Tonight_Goal_xG"] = projected.round(3)
    out["Tonight_Goal_1plus_pct"] = probability.round(1)
    out["Tonight_Goal_vs_Book_pp"] = (probability - break_even).where(book_line.eq(0.5)).round(1)
    out["Tonight_Goal_Status"] = np.where(
        projected.notna(), "Research · uncalibrated", "Unavailable · missing player opportunity"
    )
    out["Tonight_Goal_Version"] = VERSION
    return out


def _nb_over_pct(mu: pd.Series, line: pd.Series, alpha: float) -> pd.Series:
    """Market-line NB tail, matching the count family used by the book model."""
    result = pd.Series(np.nan, index=mu.index, dtype=float)
    for index in mu.index:
        mean, threshold = mu.loc[index], line.loc[index]
        if pd.isna(mean) or pd.isna(threshold) or mean < 0 or threshold < 0:
            continue
        k = math.floor(float(threshold)) + 1
        r = 1.0 / alpha
        zero = (r / (r + float(mean))) ** r
        mass = zero
        below = zero
        for count in range(1, k):
            mass *= (count - 1 + r) / count * float(mean) / (r + float(mean))
            below += mass
        result.loc[index] = max(0.0, min(100.0, 100.0 * (1.0 - below)))
    return result


def add_tonight_market_projections(frame: pd.DataFrame) -> pd.DataFrame:
    """Research estimates for priced assists, points, shots and PP points.

    Existing market means remain untouched. Opponent adjustments are deliberately
    shrunk; none of these rates changes the frozen moves or board selection.
    """
    out = frame.copy()
    if out.empty:
        return out

    gp = _numeric(out, "Model_Player_GP").clip(lower=0)
    opp_gp = _numeric(out, "Opp_Team_GP").clip(lower=0)
    team_gp = _numeric(out, "Team_GP").clip(lower=0)
    ev_minutes = _numeric(out, "EV5_TOI_per_game").clip(lower=0)
    pp_minutes = _numeric(out, "PP_TOI_per_game").clip(lower=0)
    pp_weight = (2 * pp_minutes / (ev_minutes + 2 * pp_minutes)).clip(0, 0.7).fillna(0.0)

    ev_opp = _numeric(out, "opp_5v5_xGA60")
    ev_league = _numeric(out, "League_5v5_xGA60")
    pk_opp = _numeric(out, "Opp_PK_xGA60")
    pk_league = _numeric(out, "League_PK_xGA60")
    ev_valid = ev_opp.between(0.5, 8.0) & ev_league.between(0.5, 8.0)
    pk_valid = pk_opp.between(1.0, 25.0) & pk_league.between(1.0, 25.0)
    reliability = opp_gp / (opp_gp + 10.0)
    ev_factor = (1.0 + reliability * (ev_opp / ev_league - 1.0)).clip(.8, 1.2)
    pk_factor = (1.0 + reliability * (pk_opp / pk_league - 1.0)).clip(.8, 1.2)
    ev_factor = ev_factor.where(ev_valid, 1.0).fillna(1.0)
    pk_factor = pk_factor.where(pk_valid, 1.0).fillna(1.0)

    goalie_status = out.get("Opp_Goalie_Status", pd.Series("", index=out.index)).astype(str).str.casefold()
    goalie_xga = _numeric(out, "Opp_Goalie_xGA")
    goalie_ga = _numeric(out, "Opp_Goalie_GA")
    goalie_factor = ((goalie_ga + 10.0) / (goalie_xga + 10.0)).clip(.85, 1.15)
    goalie_factor = goalie_factor.where(
        goalie_status.isin({"confirmed", "official"}) & goalie_xga.gt(0) & goalie_ga.ge(0), 1.0
    ).fillna(1.0)
    scoring_factor = ((1.0 - pp_weight) * ev_factor + pp_weight * pk_factor) * (1.0 + .5 * (goalie_factor - 1.0))
    scoring_factor = scoring_factor.clip(.8, 1.2)

    # The shot factor is already computed from opponent SOG allowed with a
    # ten-game prior. No goal-scoring or goalie factor belongs on shot volume.
    sog_factor = _numeric(out, "SOG_Opp_Volume_Factor").where(
        _numeric(out, "SOG_Opp_Volume_Factor").between(.8, 1.2), 1.0
    ).fillna(1.0)

    for market, factor, alpha in (
        ("Assists", scoring_factor, .50),
        ("Points", scoring_factor, .35),
        ("SOG", sog_factor, .25),
    ):
        base = _numeric(out, f"{market}_mu")
        valid = base.gt(0) & gp.gt(0)
        projected = (base * factor).clip(lower=0, upper=8 if market == "SOG" else 4).where(valid)
        out[f"Tonight_{market}_mu"] = projected.round(3)
        out[f"Tonight_{market}_over_pct"] = _nb_over_pct(
            projected, _numeric(out, f"{market}_Line"), alpha
        ).round(1)
        out[f"Tonight_{market}_Factor"] = factor.round(3)
        out[f"Tonight_{market}_Status"] = np.where(valid, "Research · uncalibrated", "Unavailable · missing player rate")
        out[f"Tonight_{market}_Version"] = MULTI_MARKET_VERSION

    # PP_iP60 is an opportunity proxy: individual PP xG plus weighted *actual*
    # assists. It is not an expected-assist measure. Require five player games
    # and regular PP time, and keep its forecast explicitly experimental.
    pp_proxy = _numeric(out, "PP_Points60")
    pp_actual = _numeric(out, "PP_ActualPoints60")
    pp_base = pp_proxy * pp_minutes / 60.0
    pp_observed = pp_actual * pp_minutes / 60.0
    blend = (gp / (gp + 10.0)).clip(0, .5).fillna(0.0)
    pp_base = pp_base * (1.0 - blend) + pp_observed.fillna(pp_base) * blend
    team_pp = _numeric(out, "Team_PP_xGF60")
    league_pp = _numeric(out, "League_PP_xGF60")
    pp_attack = (1.0 + team_gp / (team_gp + 10.0) * (team_pp / league_pp - 1.0)).clip(.85, 1.15)
    pp_attack = pp_attack.where(team_pp.gt(0) & league_pp.gt(0), 1.0).fillna(1.0)
    pp_matchup = (pp_attack * pk_factor).clip(.75, 1.3)
    pp_valid = gp.ge(5) & pp_minutes.ge(1) & pp_proxy.ge(0)
    pp_mu = (pp_base * pp_matchup).clip(lower=0, upper=2.0).where(pp_valid)
    out["Tonight_PPP_mu"] = pp_mu.round(3)
    out["Tonight_PPP_1plus_pct"] = _nb_over_pct(
        pp_mu, pd.Series(.5, index=out.index), .50
    ).round(1)
    out["Tonight_PPP_Matchup_Factor"] = pp_matchup.round(3)
    out["Tonight_PPP_Status"] = np.where(
        pp_valid, "Research · observed-assist proxy · uncalibrated", "Unavailable · limited PP sample"
    )
    out["Tonight_PPP_Version"] = MULTI_MARKET_VERSION
    return out

