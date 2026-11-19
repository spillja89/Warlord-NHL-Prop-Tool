"""Recover missing PP table context without modifying the frozen tracker."""
from __future__ import annotations
import pandas as pd

FIELDS = ('Team_PPO_PG', 'Team_PP_Time_PG', 'Team_PPO_GP',
          'Opp_PKO_PG', 'Opp_PK_Time_PG', 'Opp_PK_GA_PG', 'Opp_PKO_GP',
          'NHL_Opp_PK_Success_pct')


def recover_missing(frame, before_day, reports, codes):
    out = frame.copy()
    if set(out.Date.astype(str).str[:10]) != {str(before_day)}:
        raise ValueError('PP recovery requires the exact saved slate date')
    prepared = {}
    for name in ('powerplay', 'penaltykill'):
        data = reports[name].copy()
        data = data.loc[data.gameDate.astype(str).str[:10].lt(str(before_day))]
        if data.duplicated(['teamId', 'gameId']).any() or not data.gameId.astype(str).str.startswith('202602').all():
            raise ValueError('Invalid PP recovery game identity')
        data['Team'] = data.teamId.map(codes)
        prepared[name] = data
    pp, pk = prepared['powerplay'], prepared['penaltykill']
    if set(zip(pp.teamId, pp.gameId)) != set(zip(pk.teamId, pk.gameId)):
        raise ValueError('PP recovery reports cover different games')
    team = {}
    for code, data in pp.groupby('Team'):
        gp = len(data)
        team.setdefault(code, {}).update(Team_PPO_PG=data.ppOpportunities.sum()/gp,
            Team_PP_Time_PG=data.ppTimeOnIcePerGame.sum()/gp/60, Team_PPO_GP=gp)
    for code, data in pk.groupby('Team'):
        gp = len(data); ops = data.timesShorthanded.sum(); goals = data.ppGoalsAgainst.sum()
        team.setdefault(code, {}).update(Opp_PKO_PG=ops/gp,
            Opp_PK_Time_PG=data.pkTimeOnIcePerGame.sum()/gp/60,
            Opp_PK_GA_PG=goals/gp, Opp_PKO_GP=gp,
            NHL_Opp_PK_Success_pct=100*(1-goals/ops) if ops > 0 else None)
    for field in FIELDS:
        key = 'Team' if field.startswith('Team_') else 'Opp'
        fresh = out[key].map(lambda code: team.get(str(code).upper(), {}).get(field))
        old = pd.to_numeric(out[field], errors='coerce') if field in out else pd.Series(float('nan'), index=out.index)
        out[field] = old.where(old.notna(), fresh)
    return out
