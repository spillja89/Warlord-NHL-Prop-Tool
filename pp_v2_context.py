"""V2 official PK windows. Reuse the run's reports; never fetch from a card."""
from __future__ import annotations
import math
import pandas as pd

VERSION='pp-v2-context-2026-11-19'
FIELDS=('Official_Opp_PK_L5_PPP_per_O','Official_Opp_PK_L10_Goals_PG','Official_Opp_PK_L10_Seconds_per_O')

def number(v):
    try:
        v=float(v)
        return v if math.isfinite(v) else None
    except (TypeError,ValueError):return None

def attach_context(frame,before_day,reports,codes):
    out=frame.copy()
    if out.empty:return out
    before=str(before_day)[:10]
    if set(out.Date.astype(str).str[:10])!={before} or not ('2026-07-01'<=before<'2027-07-01'):
        raise ValueError('PP V2 requires one current-season pregame date')
    for field in FIELDS:out[field]=None
    out['PP_V2_Context_Version']=VERSION
    out['PP_V2_Context_AsOf']=before
    pk=reports.get('penaltykill',pd.DataFrame()).copy()
    sk=reports.get('skater_powerplay',pd.DataFrame()).copy()
    required={'gameId','gameDate','teamId','opponentTeamAbbrev','ppGoalsAgainst','timesShorthanded','pkTimeOnIcePerGame'}
    if pk.empty or not required.issubset(pk.columns):return out
    pk=pk.loc[pk.gameDate.astype(str).str[:10].lt(before)].copy()
    if pk.duplicated(['gameId','teamId']).any() or not pk.gameId.astype(str).str.startswith('202602').all():
        raise ValueError('Invalid PP V2 official team identities')
    pk['Team']=pk.teamId.map(codes)
    for f in ['ppGoalsAgainst','timesShorthanded','pkTimeOnIcePerGame']:
        pk[f]=pd.to_numeric(pk[f],errors='coerce')
        pk.loc[~pk[f].map(lambda v: pd.notna(v) and math.isfinite(v) and v>=0),f]=float('nan')
    credits={}
    needed={'gameId','gameDate','playerId','teamAbbrev','ppGoals','ppAssists','ppPoints'}
    if not sk.empty and needed.issubset(sk.columns):
        sk=sk.loc[sk.gameDate.astype(str).str[:10].lt(before)].copy()
        if sk.duplicated(['gameId','playerId']).any() or not sk.gameId.astype(str).str.startswith('202602').all():
            raise ValueError('Invalid PP V2 official skater identities')
        for f in ['ppGoals','ppAssists','ppPoints']:sk[f]=pd.to_numeric(sk[f],errors='coerce')
        for (game,team),g in sk.groupby(['gameId','teamAbbrev']):
            vals=g[['ppGoals','ppAssists','ppPoints']]
            if not vals.notna().all().all() or not vals.ge(0).all().all() or not g.ppPoints.eq(g.ppGoals+g.ppAssists).all():continue
            goals,assists,points=(float(vals[c].sum()) for c in vals)
            if assists<=2*goals:credits[(int(game),str(team))]=(goals,points)
    cache={}
    for team,history in pk.groupby('Team'):
        history=history.sort_values(['gameDate','gameId']);z={f:None for f in FIELDS}
        for size in (5,10):
            h=history.tail(size);z[f'PP_V2_Opp_L{size}_GP']=len(h)
            z[f'PP_V2_Opp_L{size}_Last_Date']=str(h.gameDate.max())[:10] if len(h) else None
            if len(h)!=size:continue
            valid=h[['ppGoalsAgainst','timesShorthanded','pkTimeOnIcePerGame']].notna().all().all()
            if not valid or not h.ppGoalsAgainst.le(h.timesShorthanded).all():continue
            goals=float(h.ppGoalsAgainst.sum());opps=float(h.timesShorthanded.sum())
            if size==10:
                z['Official_Opp_PK_L10_Goals_PG']=goals/10
                if opps>0:z['Official_Opp_PK_L10_Seconds_per_O']=float(h.pkTimeOnIcePerGame.sum())/opps
            else:
                matched=[credits.get((int(r.gameId),str(r.opponentTeamAbbrev))) for r in h.itertuples()]
                if opps>0 and all(c is not None and c[0]==r.ppGoalsAgainst for c,r in zip(matched,h.itertuples())):
                    z['Official_Opp_PK_L5_PPP_per_O']=sum(c[1] for c in matched)/opps
        cache[team]=z
    values=[cache.get(r.get('Opp'),{}) for r in out.to_dict('records')]
    for f in set(FIELDS).union(*(v.keys() for v in values)):out[f]=[v.get(f) for v in values]
    return out
