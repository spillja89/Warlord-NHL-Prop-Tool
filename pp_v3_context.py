"""Approved PP V3 environment values from the run's existing official reports."""
import math
import pandas as pd

VERSION='pp-v3-context-2026-11-21'
FIELDS=('PPV3_Opp_PK_Season_Minutes_PG','PPV3_Opp_PK_L5_Seconds_per_O',
 'PPV3_Opp_Season_GA_PG','PPV3_Opp_PK_L5_PPA_per_O','PPV3_Own_PP_Venue_O_PG',
 'PPV3_Own_PK_Season_Seconds_per_O','PPV3_Own_PP_L5_O_PG','PPV3_Own_PP_Venue_Minutes_PG')

def attach_context(frame,before_day,reports,codes):
    out=frame.copy()
    if out.empty:return out
    before=str(before_day)[:10]
    if set(out.Date.astype(str).str[:10])!={before} or not '2026-07-01'<=before<'2027-07-01':raise ValueError('PP V3 needs one current-season pregame date')
    for f in FIELDS:out[f]=None
    out['PP_V3_Context_Version']=VERSION;out['PP_V3_Context_AsOf']=before
    prepared={}
    for name,cols in [('summary',['goalsAgainst']),('powerplay',['ppOpportunities','ppTimeOnIcePerGame']),('penaltykill',['ppGoalsAgainst','timesShorthanded','pkTimeOnIcePerGame'])]:
        h=reports.get(name,pd.DataFrame()).copy();need={'gameId','gameDate','teamId','homeRoad',*cols}
        if h.empty or not need.issubset(h.columns):prepared[name]=pd.DataFrame();continue
        h=h.loc[h.gameDate.astype(str).str[:10].lt(before)].copy()
        if h.duplicated(['teamId','gameId']).any() or not h.gameId.astype(str).str.startswith('202602').all():raise ValueError('Invalid current-season PP V3 report identities')
        if 'gamesPlayed' in h and not h.gamesPlayed.eq(1).all():raise ValueError('PP V3 requires game-level reports')
        h['Team']=h.teamId.map(codes)
        for c in cols:
            h[c]=pd.to_numeric(h[c],errors='coerce');h.loc[~h[c].map(lambda v:pd.notna(v) and math.isfinite(v) and v>=0),c]=float('nan')
        prepared[name]=h.sort_values(['gameDate','gameId'])
    credits={};sk=reports.get('skater_powerplay',pd.DataFrame()).copy()
    if not sk.empty and {'gameId','gameDate','playerId','teamAbbrev','ppGoals','ppAssists','ppPoints'}.issubset(sk.columns):
        sk=sk.loc[sk.gameDate.astype(str).str[:10].lt(before)].copy()
        if sk.duplicated(['gameId','playerId']).any() or not sk.gameId.astype(str).str.startswith('202602').all():raise ValueError('Invalid PP V3 skater identities')
        for c in ['ppGoals','ppAssists','ppPoints']:sk[c]=pd.to_numeric(sk[c],errors='coerce')
        for key,g in sk.groupby(['gameId','teamAbbrev']):
            v=g[['ppGoals','ppAssists','ppPoints']]
            if v.notna().all().all() and v.ge(0).all().all() and g.ppPoints.eq(g.ppGoals+g.ppAssists).all() and g.ppAssists.sum()<=2*g.ppGoals.sum():credits[(int(key[0]),str(key[1]))]=(float(g.ppGoals.sum()),float(g.ppAssists.sum()))
    cache={}
    def context(team,venue):
        if (team,venue) in cache:return cache[team,venue]
        z={};hist={k:(v.loc[v.Team.eq(team)] if not v.empty else v) for k,v in prepared.items()}
        pp,pk,sm=(hist[k] for k in ['powerplay','penaltykill','summary'])
        def good(h,cols):return not h.empty and h[cols].notna().all().all()
        if good(sm,['goalsAgainst']):z['PPV3_Opp_Season_GA_PG']=float(sm.goalsAgainst.sum())/len(sm)
        if good(pk,['timesShorthanded','pkTimeOnIcePerGame','ppGoalsAgainst']) and pk.ppGoalsAgainst.le(pk.timesShorthanded).all():
            seconds=float(pk.pkTimeOnIcePerGame.sum());opps=float(pk.timesShorthanded.sum())
            z['PPV3_Opp_PK_Season_Minutes_PG']=seconds/60/len(pk)
            if opps>0:z['PPV3_Own_PK_Season_Seconds_per_O']=seconds/opps
        h=pk.tail(5)
        if len(h)==5 and good(h,['timesShorthanded','pkTimeOnIcePerGame','ppGoalsAgainst']) and h.ppGoalsAgainst.le(h.timesShorthanded).all():
            opps=float(h.timesShorthanded.sum())
            if opps>0:
                z['PPV3_Opp_PK_L5_Seconds_per_O']=float(h.pkTimeOnIcePerGame.sum())/opps
                if 'opponentTeamAbbrev' in h:
                    matches=[credits.get((int(g.gameId),str(g.opponentTeamAbbrev))) for g in h.itertuples()]
                    if all(c is not None and c[0]==g.ppGoalsAgainst for c,g in zip(matches,h.itertuples())):z['PPV3_Opp_PK_L5_PPA_per_O']=sum(c[1] for c in matches)/opps
        if not pp.empty:
            h=pp.loc[pp.homeRoad.eq(venue)]
            if good(h,['ppOpportunities','ppTimeOnIcePerGame']):
                z['PPV3_Own_PP_Venue_O_PG']=float(h.ppOpportunities.sum())/len(h)
                z['PPV3_Own_PP_Venue_Minutes_PG']=float(h.ppTimeOnIcePerGame.sum())/60/len(h)
            h=pp.tail(5)
            if len(h)==5 and good(h,['ppOpportunities']):z['PPV3_Own_PP_L5_O_PG']=float(h.ppOpportunities.sum())/5
        z['PP_V3_Prior_Last_Date']=max((str(v.gameDate.max())[:10] for v in hist.values() if not v.empty),default=None)
        cache[team,venue]=z;return z
    values=[]
    for row in out.to_dict('records'):
        game=str(row.get('Game','')).replace(' ','').split('@');home=game[1] if len(game)==2 else None
        own,opp=row.get('Team'),row.get('Opp');z={}
        if home in {own,opp}:
            a=context(own,'H' if home==own else 'R');b=context(opp,'H' if home==opp else 'R')
            z={f:(a if f.startswith('PPV3_Own_') else b).get(f) for f in FIELDS}
            z['PP_V3_Own_Last_Date']=a.get('PP_V3_Prior_Last_Date');z['PP_V3_Opp_Last_Date']=b.get('PP_V3_Prior_Last_Date')
        values.append(z)
    for f in FIELDS+('PP_V3_Own_Last_Date','PP_V3_Opp_Last_Date'):out[f]=[v.get(f) for v in values]
    return out
