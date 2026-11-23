"""Approved Goal V3 additions. Independent profiles; exact-parent optional buffs.

No V3 confidence or xG condition is inherited by an existing V2 rule.
"""
from functools import lru_cache
from pathlib import Path
from statistics import mean, median, pstdev
from datetime import datetime
import json, re, unicodedata
from goal_routes_v2 import number, profile_values as v2_values, _passes

VERSION = 'goal-2026-27-moves-v3-2026-11-23'
CONTEXT_VERSION = 'goal-v3-context-2026-11-23'
LABELS = {
 'Goal_Profile':'Goal profile confidence',
 'Player_Log_L20_TOI_median':'Up-to-L20 median ice time (min)',
 'Trim10_SOG':'Trimmed L10 average SOG',
 'Player_Log_L5_PPP_per60_total_TOI':'L5 PPP per 60 total ice minutes',
 'Player_Log_L5_S_sd':'L5 shot variability',
 'Player_Log_L10_TOI_sd':'L10 ice-time variability (min)',
 'Player_Log_L40_S_hit_pct':'Up-to-L40 3+ SOG game frequency %',
 'PP_TOI_Pct_Game':'PP time as % of player total ice time',
 'Official_Own_PP_Venue_O_PG':'Own PP opportunities/game at venue',
 'Official_Opp_PP_Season_G60':'Opponent actual PP goals/60',
 'Official_Opp_L10_SA_PG':'Opponent L10 SOG allowed/game',
 'Official_Opp_PK_L10_G60':'Opponent L10 PK goals allowed/60',
 'Official_Opp_PK_Season_O_PG':'Opponent season PK opportunities/game',
 'Official_Opp_Season_GA_PG':'Opponent team goals allowed/game',
 'Goalie_Expected_SV':'Opponent goalie expected save %',
}

@lru_cache(maxsize=1)
def catalog():
    return json.loads(Path(__file__).with_name('goal_v3_rules.json').read_text(encoding='utf-8'))

@lru_cache(maxsize=4096)
def _form(raw, before):
    try:
        doc=json.loads(raw)
        if not isinstance(doc,dict) or doc.get('season')!='2026-27':return {}
        games=doc.get('games')
        if not isinstance(games,list) or not games or any(not isinstance(g,dict) for g in games):return {}
        games=sorted(games,key=lambda g:str(g['d']),reverse=True)
        if any(len(str(g['d']))!=10 or str(g['d'])>=before for g in games) or len({g['d'] for g in games})!=len(games):return {}
        if any(number(g.get(k)) is None or number(g[k])<0 for g in games for k in ('g','a','s')):return {}
        out={}
        for window in (5,10,20,40):
            sample=games[:window];shots=[float(g['s']) for g in sample];times=[]
            if window==5:out['Player_Log_L5_S_sd']=pstdev(shots)
            if window==40:out['Player_Log_L40_S_hit_pct']=100*sum(s>=3 for s in shots)/len(shots)
            for g in sample:
                try:
                    m,s=map(int,str(g.get('toi','')).split(':'))
                    if m<0 or not 0<=s<60:break
                    times.append(m+s/60)
                except (TypeError,ValueError):break
            if len(times)==len(sample):
                if window==20:out['Player_Log_L20_TOI_median']=median(times)
                if window==10:out['Player_Log_L10_TOI_sd']=pstdev(times)
                if window==5 and sum(times)>0 and all(number(g.get('pp')) is not None and number(g['pp'])>=0 for g in sample):
                    out['Player_Log_L5_PPP_per60_total_TOI']=60*sum(float(g['pp']) for g in sample)/sum(times)
        return out
    except (TypeError,ValueError,KeyError):return {}

def _normal(value):
    return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',str(value)).encode('ascii','ignore').decode().lower())

def goalie_expected(row):
    sv,gp=number(row.get('Opp_SV')),number(row.get('Opp_GP'))
    if not (row.get('Goalie_Stats_Season')=='2026-2027' and str(row.get('Opp_Goalie_Source','')).endswith('name_and_stats') and str(row.get('Opp_Goalie_Status','')).lower() in ('confirmed','official') and sv is not None and 0<=sv<=1 and gp is not None and gp>0):return None
    name=_normal(row.get('Opp_Goalie'))
    if name in ('','none','nan','unknown'):return None
    matched=row.get('Opp_Goalie_Stats_Name')
    if isinstance(matched,str) and matched.strip() and _normal(matched)!=name:return None
    checked=str(row.get('Goalie_Check_UTC') or '')
    if checked and checked.lower() not in ('nan','none'):
        try:
            check=datetime.fromisoformat(checked.replace('Z','+00:00'));start=datetime.fromisoformat(str(row.get('StartTimeUTC')).replace('Z','+00:00'))
            if not check.tzinfo or not start.tzinfo or check>=start:return None
        except (TypeError,ValueError):return None
    xga,shots=number(row.get('Opp_Goalie_xGA')),number(row.get('Opp_Goalie_Shots_Faced'))
    if xga is None or xga<0 or shots is None or shots<=0:return None
    return 1-xga/shots

def profile_values(row):
    # Old V2 values retain their existing source guards and independent gates.
    out=v2_values(row)
    if row.get('Model_Stats_Season')!='2026-2027' or row.get('Form_Season')!='2026-27':return out
    before=str(row.get('Date',''))[:10];raw=row.get('Form_Log')
    if len(before)!=10:return out
    raw=json.dumps(raw) if isinstance(raw,dict) else raw
    derived=_form(raw,before) if isinstance(raw,str) else {}
    if not derived:return out
    out.update(derived)
    from goal_moves_current import goal_profile_score
    score=goal_profile_score(row)
    if score is not None:out['Goal_Profile']=score
    for f in ('StarScore','Trim10_SOG','PP_TOI_Pct_Game','i5v5_primaryAssists60','Drought_G','Drought_PPP'):
        if f=='i5v5_primaryAssists60' and row.get('Opportunity_View_Version')!='2026-10-16-v1':continue
        value=number(row.get(f))
        if value is not None and value>=0:out[f]=value
    if row.get('GoalV3_Context_Version')==CONTEXT_VERSION and row.get('GoalV3_Context_AsOf')==before:
        for f in catalog()['environment_fields']:
            side='Own' if f.startswith('Official_Own_') else 'Opp'
            last=str(row.get('GoalV3_'+side+'_Last_Date',''))[:10]
            value=number(row.get('GoalV3_'+f))
            if len(last)==10 and last<before and value is not None and value>=0:out[f]=value
    value=goalie_expected(row)
    if value is not None:out['Goalie_Expected_SV']=value
    return out

def _move(spec):
    r=spec['record'];buff=spec['kind']=='environment_buff'
    return dict(id='goal_v3:'+spec['id'],name=spec['name'],kind='CONTEXT' if buff else spec['display_kind'],icon=spec['icon'],
        rule=spec['rule'],wins=r['hits'],picks=r['plays'],later_wins=0,later_picks=0,market='Goal',kit_version=VERSION,
        goal_v2=True,goal_v3=True,goal_record_key='v3:'+spec['id'],route_family='v3',goal_family=spec['family'],
        goal_primary=not buff,goal_complete_entry=buff,goal_visible_buff=buff,matchup_buff=buff,goal_retired=False,
        goal_level=spec.get('level',1),goal_parent=spec.get('parent'),goal_parent_name=spec.get('parent_name',''),
        goal_character=spec['character'],checks=spec['checks'],all_checks=spec['all_checks'],crit=spec['display_kind']=='CRIT',
        sample_season='2026-27',sample_from='2026-10-13',sample_through='2026-11-21',
        sample_book_be_pct=r['BE'],sample_players=r['players'],sample_dates=r['dates'],record_version='V3')

def records_for(row):
    line=number(row.get('Goal_Line'))
    if line is None:line=number(row.get('ATG_Line'))
    if line!=.5 or str(row.get('Matrix_Goal','')).strip().casefold()!='green':return []
    if str(row.get('Roster_Watch','')).lower() in ('true','1') or str(row.get('Lineup_Status','')).lower()=='not dressed':return []
    values=profile_values(row)
    return [_move(s) for s in catalog()['moves'] if _passes(values,s['all_checks'])]

def attach_context(frame,before_day,reports,codes):
    """Only the six approved official fields, reusing existing report downloads."""
    import pandas as pd
    out=frame.copy();before=str(before_day)[:10]
    if out.empty:return out
    if set(out.Date.astype(str).str[:10])!={before}:raise ValueError('Goal V3 requires one exact slate date')
    prepared={}
    requirements={'summary':['goalsAgainst','shotsAgainstPerGame'], 'powerplay':['ppOpportunities','ppTimeOnIcePerGame','powerPlayGoalsFor'],'penaltykill':['timesShorthanded','pkTimeOnIcePerGame','ppGoalsAgainst']}
    for name,cols in requirements.items():
        f=reports.get(name,pd.DataFrame()).copy()
        if f.empty or not {'gameDate','gameId','teamId',*cols}.issubset(f):continue
        f=f.loc[f.gameDate.astype(str).str[:10].lt(before)].copy()
        if f.duplicated(['teamId','gameId']).any() or not f.gameId.astype(str).str.startswith('202602').all():raise ValueError('Invalid Goal V3 official identity')
        for col in cols:
            f[col]=pd.to_numeric(f[col],errors='coerce')
            f.loc[~f[col].map(lambda x:pd.notna(x) and number(x) is not None and x>=0),col]=float('nan')
        f['Team']=f.teamId.map(codes);prepared[name]=f.sort_values(['gameDate','gameId'])
    cache={}
    def context(team,venue):
        if (team,venue) in cache:return cache[team,venue]
        z={};a=prepared.get('summary',pd.DataFrame())
        if a.empty:return z
        a=a.loc[a.Team.eq(team)]
        if a.empty:return z
        z['Last_Date']=str(a.gameDate.max())[:10]
        if a.goalsAgainst.notna().all():z['Season_GA_PG']=float(a.goalsAgainst.mean())
        if len(a)>=10 and a.tail(10).shotsAgainstPerGame.notna().all():z['L10_SA_PG']=float(a.tail(10).shotsAgainstPerGame.mean())
        for name in ('powerplay','penaltykill'):
            f=prepared.get(name,pd.DataFrame())
            if f.empty:continue
            h=f.loc[f.Team.eq(team)]
            if set(h.gameId)!=set(a.gameId):continue
            if name=='powerplay':
                if h[['powerPlayGoalsFor','ppTimeOnIcePerGame']].notna().all().all() and h.ppTimeOnIcePerGame.sum()>0:z['PP_Season_G60']=3600*float(h.powerPlayGoalsFor.sum())/float(h.ppTimeOnIcePerGame.sum())
                v=h.loc[h.homeRoad.eq(venue)] if 'homeRoad' in h else h.iloc[:0]
                if len(v) and v.ppOpportunities.notna().all():z['PP_Venue_O_PG']=float(v.ppOpportunities.mean())
            else:
                if h.timesShorthanded.notna().all():z['PK_Season_O_PG']=float(h.timesShorthanded.mean())
                t=h.tail(10)
                if len(t)==10 and t[['ppGoalsAgainst','pkTimeOnIcePerGame']].notna().all().all() and t.pkTimeOnIcePerGame.sum()>0:z['PK_L10_G60']=3600*float(t.ppGoalsAgainst.sum())/float(t.pkTimeOnIcePerGame.sum())
        cache[team,venue]=z;return z
    values=[]
    for row in out.to_dict('records'):
        game=str(row.get('Game','')).replace(' ','').split('@');z={}
        if len(game)==2 and row.get('Team') in game and row.get('Opp') in game:
            for side,team in [('Own',row['Team']),('Opp',row['Opp'])]:
                c=context(team,'H' if team==game[1] else 'R');z['GoalV3_'+side+'_Last_Date']=c.get('Last_Date')
                for f in catalog()['environment_fields']:
                    if f.startswith('Official_'+side+'_'):z['GoalV3_'+f]=c.get(f.removeprefix('Official_'+side+'_'))
        values.append(z)
    for field in ['GoalV3_'+f for f in catalog()['environment_fields']]+['GoalV3_Own_Last_Date','GoalV3_Opp_Last_Date']:
        out[field]=[v.get(field) for v in values]
    out['GoalV3_Context_Version']=CONTEXT_VERSION;out['GoalV3_Context_AsOf']=before
    return out

def attach_routes(frame):
    from goal_routes_v2 import attach_routes as attach_old
    out=attach_old(frame);rows=out.to_dict('records');values=[profile_values(r) for r in rows]
    for f in LABELS:out['GoalV3_'+f]=[v.get(f) for v in values]
    out['Goal_Move_Version']=VERSION;out['Goal_Move_Label']='Moves V3'
    out['GoalV3_Fired_Ids']=[json.dumps([m['goal_record_key'] for m in records_for(r)]) for r in rows]
    return out
