"""Strict prior Assist context, reusing the slate's existing official downloads."""
import pandas as pd
from assist_routes_v3 import CONTEXT_VERSION, OFFICIAL, number

def attach_context(frame,before_day,reports,codes):
    out=frame.copy();before=str(before_day)[:10]
    if out.empty:return out
    if set(out.Date.astype(str).str[:10])!={before} or not '2026-07-01'<=before<'2027-07-01':raise ValueError('Assist V3 requires one current-season slate')
    prepared={}
    requirements={'summary':['goalsFor','goalsAgainst','shotsAgainstPerGame'],
      'powerplay':['ppOpportunities','ppTimeOnIcePerGame','powerPlayGoalsFor'],
      'penaltykill':['ppGoalsAgainst'], 'skater_powerplay':['ppTimeOnIce']}
    for name,cols in requirements.items():
        h=reports.get(name,pd.DataFrame()).copy();identity='playerId' if name=='skater_powerplay' else 'teamId'
        if h.empty or not {'gameDate','gameId',identity,*cols}.issubset(h):continue
        h=h.loc[h.gameDate.astype(str).str[:10].lt(before)].copy()
        if h.duplicated([identity,'gameId']).any() or not h.gameId.astype(str).str.startswith('202602').all():raise ValueError('Invalid Assist V3 official identities')
        if 'gamesPlayed' in h and not h.gamesPlayed.eq(1).all():raise ValueError('Assist V3 needs game reports')
        for col in cols:
            h[col]=pd.to_numeric(h[col],errors='coerce');h.loc[h[col].map(lambda v:number(v) is None or v<0),col]=float('nan')
        if identity=='teamId':h['Team']=h.teamId.map(codes)
        prepared[name]=h.sort_values(['gameDate','gameId'])
    cache={}
    def team_values(team,venue):
        if (team,venue) in cache:return cache[team,venue]
        z={};hist={k:(h.loc[h.Team.eq(team)] if 'Team' in h else h.iloc[:0]) for k,h in prepared.items()};sm=hist.get('summary',pd.DataFrame())
        def good(h,cols):return not h.empty and h[cols].notna().all().all()
        if sm.empty:return z
        z['Last_Date']=str(sm.gameDate.max())[:10]
        for n,col,field in [(10,'goalsAgainst','L10_GA_PG'),(20,'goalsFor','L20_GF_PG'),(20,'shotsAgainstPerGame','L20_SA_PG')]:
            h=sm.tail(n)
            if len(h)==n and good(h,[col]):z[field]=float(h[col].mean())
        pk=hist.get('penaltykill',pd.DataFrame());pp=hist.get('powerplay',pd.DataFrame())
        if set(pk.get('gameId',[]))==set(sm.gameId) and good(pk,['ppGoalsAgainst']):z['PK_Season_Goals_PG']=float(pk.ppGoalsAgainst.mean())
        if set(pp.get('gameId',[]))==set(sm.gameId):
            if good(pp,['ppOpportunities']):z['PP_Season_O_PG']=float(pp.ppOpportunities.mean())
            h=pp.loc[pp.homeRoad.eq(venue)] if 'homeRoad' in pp else pp.iloc[:0]
            if good(h,['powerPlayGoalsFor','ppTimeOnIcePerGame']) and h.ppTimeOnIcePerGame.sum()>0:z['PP_Venue_G60']=3600*float(h.powerPlayGoalsFor.sum())/float(h.ppTimeOnIcePerGame.sum())
        cache[team,venue]=z;return z
    sk=prepared.get('skater_powerplay',pd.DataFrame());players={}
    if not sk.empty:
        for pid,h in sk.groupby('playerId'):
            if h.ppTimeOnIce.notna().all():players[int(pid)]={'minutes':float(h.ppTimeOnIce.sum())/60/len(h),'last':str(h.gameDate.max())[:10]}
    values=[]
    for row in out.to_dict('records'):
        game=str(row.get('Game','')).replace(' ','').split('@');z={}
        if len(game)==2 and row.get('Team') in game and row.get('Opp') in game:
            for side,team in [('Own',row['Team']),('Opp',row['Opp'])]:
                c=team_values(team,'H' if team==game[1] else 'R');z['AssistV3_'+side+'_Last_Date']=c.get('Last_Date')
                for f in OFFICIAL:
                    if f.startswith('Official_'+side+'_'):z['AssistV3_'+f]=c.get(f.removeprefix('Official_'+side+'_'))
        # The player ladder used the already frozen official player-usage field.
        # Preserve its source/as-of and coverage instead of replacing an older
        # saved value with a later, more complete/corrected report.
        last=str(row.get('PP_Lower_Player_Last_Date',''))[:10]
        minutes=number(row.get('Official_Player_MinutesPG_Season'))
        if (row.get('PP_Lower_Context_Version')=='pp-lower-context-2026-11-07-v1'
            and row.get('PP_Lower_Context_AsOf')==before and len(last)==10 and last<before
            and minutes is not None and minutes>=0):
            z.update(AssistV3_Official_Player_MinutesPG_Season=minutes,AssistV3_Player_Last_Date=last)
        values.append(z)
    for f in ['AssistV3_'+f for f in OFFICIAL]+['AssistV3_Own_Last_Date','AssistV3_Opp_Last_Date','AssistV3_Player_Last_Date']:out[f]=[v.get(f) for v in values]
    out['AssistV3_Context_Version']=CONTEXT_VERSION;out['AssistV3_Context_AsOf']=before
    return out
