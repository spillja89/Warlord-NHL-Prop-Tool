"""Approved Goal V2 player profiles and optional exact-parent environment buffs."""
from functools import lru_cache
from pathlib import Path
import json, math

VERSION = 'goal-2026-27-moves-v2-2026-11-11'
CONTEXT_VERSION = 'goal-v2-context-2026-11-11'
COMMON = 'Goal 0.5 + Matrix_Goal Green + Pregame_xG_L5_per_game >= 0.52 + Player_Log_L5_TOI_per_game >= 18.8'

def number(value):
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (TypeError,ValueError):return None

@lru_cache(maxsize=1)
def catalog():return json.loads(Path(__file__).with_name('goal_v2_rules.json').read_text(encoding='utf-8'))

@lru_cache(maxsize=4096)
def _log(raw,before):
    try:
        data=json.loads(raw)
        if data.get('season')!='2026-27':return {}
        games=sorted(data['games'],key=lambda g:str(g['d']),reverse=True)
        if not games or any(str(g['d'])>=before for g in games) or len({g['d'] for g in games})!=len(games):return {}
        if any(number(g.get(k)) is None or number(g[k])<0 for g in games for k in ('g','a','s')):return {}
        five,ten,twenty=games[:5],games[:10],games[:20]
        result={'GoalV2_Log_GP':len(games),'L20_A':sum(g['a'] for g in twenty)}
        times=[]
        for g in five:
            a,b=str(g.get('toi','')).split(':');a,b=int(a),int(b)
            if a<0 or not 0<=b<60:return result
            times.append(a+b/60)
        result['Player_Log_L5_TOI_per_game']=sum(times)/len(times)
        shots=sum(g['s'] for g in ten)
        if shots>0:result['Player_Log_L10_Shooting_pct']=100*sum(g['g'] for g in ten)/shots
        goals=sum(g['g'] for g in twenty);points=goals+result['L20_A']
        if points>0:result['Player_Log_L20_Goal_fraction_pct']=100*goals/points
        if len(ten)>=5:result['GoalV2_G10_Pace']=10*sum(g['g'] for g in ten)/len(ten)
        return result
    except (TypeError,ValueError,KeyError):return {}

def profile_values(row):
    if row.get('Model_Stats_Season')!='2026-2027' or row.get('Form_Season')!='2026-27':return {}
    before=str(row.get('Date',''))[:10];raw=row.get('Form_Log')
    if len(before)!=10:return {}
    raw=json.dumps(raw) if isinstance(raw,dict) else raw
    values=dict(_log(raw,before)) if isinstance(raw,str) else {}
    if not values:return {}
    xg=number(row.get('Pregame_xG_L5_per_game'))
    if xg is not None and xg>=0 and (row.get('Opportunity_View_Version')=='2026-10-16-v1' or row.get('Player_PregameForm_Source')=='RECONSTRUCTED_STRICT_PRIOR'):
        values['Pregame_xG_L5_per_game']=xg
    if row.get('Opportunity_View_Version')=='2026-10-16-v1':
        for field in ('PP_SOG60','i5v5_xG60'):
            value=number(row.get(field))
            if value is not None and value>=0:values[field]=value
    expectation=number(row.get('Exp_G_10'))
    if expectation is not None and 'GoalV2_G10_Pace' in values:values['Player_Rebuilt_Gap_G10']=expectation-values['GoalV2_G10_Pace']
    if row.get('Team_Stats_Season')=='2026-2027':
        value=number(row.get('Team_PP_xGF60'))
        if value is not None and value>=0:values['Team_PP_xGF60']=value
    if row.get('GoalV2_Context_Version')==CONTEXT_VERSION and row.get('GoalV2_Context_AsOf')==before:
        for field in catalog()['environment_fields']:
            side='Opp' if field.startswith('NHL_Opp_') else 'Team'
            last=str(row.get('GoalV2_'+side+'_Last_Date',''))[:10];value=number(row.get('GoalV2_'+field))
            if len(last)==10 and last<before and value is not None and value>=0:values[field]=value
    return values

def _passes(values,checks):
    for field,op,cut in checks:
        value=number(values.get(field))
        if value is None or not (value>=cut-1e-10 if op=='>=' else value<=cut+1e-10):return False
    return True

def _move(spec):
    r=spec['record'];buff=spec['kind']=='environment_buff'
    return dict(id='goal_v2:'+spec['id'],name=spec['name'],kind='CONTEXT' if buff else spec['display_kind'],icon=spec['icon'],
        rule=spec['rule'],wins=r['hits'],picks=r['plays'],later_wins=0,later_picks=0,market='Goal',kit_version=VERSION,
        goal_v2=True,goal_record_key='v2:'+spec['id'],route_family='v2',goal_family=spec['family'],goal_primary=not buff,
        goal_visible_buff=buff,matchup_buff=buff,goal_retired=False,goal_level=spec.get('level',1),goal_parent=spec['parent'],
        checks=spec['checks'],all_checks=spec['all_checks'],crit=False,sample_season='2026-27',sample_from='2026-10-13',sample_through='2026-11-10',
        sample_book_be_pct=r['BE'],sample_players=r['players'],sample_dates=r['dates'])

def records_for(row):
    line=number(row.get('Goal_Line'))
    if line is None:line=number(row.get('ATG_Line'))
    if line!=.5 or str(row.get('Matrix_Goal','')).strip().casefold()!='green':return []
    if str(row.get('Roster_Watch','')).lower() in ('true','1') or str(row.get('Lineup_Status','')).lower()=='not dressed':return []
    values=profile_values(row)
    if not _passes(values,[('Pregame_xG_L5_per_game','>=',.52),('Player_Log_L5_TOI_per_game','>=',18.8)]):return []
    specs=catalog()['moves'];found={s['id'] for s in specs if s['kind']=='player_attack' and _passes(values,s['checks'])}
    if not found:return []
    return [_move(s) for s in specs if s['id'] in found or s['kind']=='environment_buff' and s['parent'] in found and _passes(values,s['checks'])]

def attach_context(frame,before_day,reports,codes):
    """Reuse the existing official downloads; never overwrite another prop's fields."""
    import pandas as pd
    out=frame.copy();before=str(before_day)[:10]
    if out.empty:return out
    if set(out.Date.astype(str).str[:10])!={before}:raise ValueError('Goal V2 context requires one exact slate date')
    prepared={}
    for name,cols in [('summary',['goalsAgainst','shotsAgainstPerGame']),('powerplay',['ppOpportunities','ppTimeOnIcePerGame','powerPlayGoalsFor']),('penaltykill',['timesShorthanded','pkTimeOnIcePerGame'])]:
        f=reports.get(name,pd.DataFrame()).copy()
        if f.empty or not {'gameDate','gameId','teamId',*cols}.issubset(f):continue
        f=f.loc[f.gameDate.astype(str).str[:10].lt(before)].copy()
        if f.duplicated(['teamId','gameId']).any() or not f.gameId.astype(str).str.startswith('202602').all():raise ValueError('Invalid Goal V2 official game identity')
        for col in cols:
            f[col]=pd.to_numeric(f[col],errors='coerce');f.loc[~f[col].map(lambda x:pd.notna(x) and math.isfinite(x) and x>=0),col]=float('nan')
        f['Team']=f.teamId.map(codes);prepared[name]=f.sort_values(['gameDate','gameId'])
    cache={}
    def context(team,site):
        if (team,site) in cache:return cache[team,site]
        z={};summary=prepared.get('summary',pd.DataFrame())
        if summary.empty:return z
        a=summary.loc[summary.Team.eq(team)]
        if a.empty:return z
        z['Last_Date']=str(a.gameDate.max())[:10]
        if a.shotsAgainstPerGame.notna().all():z['NHL_SA_Season_PG']=float(a.shotsAgainstPerGame.mean())
        if a.tail(5).shotsAgainstPerGame.notna().all():z['NHL_SA_L5_PG']=float(a.tail(5).shotsAgainstPerGame.mean())
        venue=a.loc[a.homeRoad.eq(site)]
        if len(venue)>=3 and venue.goalsAgainst.notna().all():z['NHL_GA_Venue_PG']=float(venue.goalsAgainst.mean())
        for special,name,oc,tc in [('PP','powerplay','ppOpportunities','ppTimeOnIcePerGame'),('PK','penaltykill','timesShorthanded','pkTimeOnIcePerGame')]:
            f=prepared.get(name,pd.DataFrame())
            if f.empty:continue
            h=f.loc[f.Team.eq(team)]
            if set(h.gameId)!=set(a.gameId):continue
            for window,size in [('Season',None),('L5',5),('L10',10)]:
                t=h if size is None else h.tail(size)
                if not t[[oc,tc]].notna().all().all():continue
                ops,secs=float(t[oc].sum()),float(t[tc].sum())
                z[f'NHL_{special}_{window}_O_PG']=ops/len(t);z[f'NHL_{special}_{window}_Time_PG']=secs/len(t)/60
                if ops>0:z[f'NHL_{special}_{window}_SecondsPerOpportunity']=secs/ops
                if special=='PP' and secs>0 and t.powerPlayGoalsFor.notna().all():z[f'NHL_PP_{window}_G60']=3600*float(t.powerPlayGoalsFor.sum())/secs
        cache[team,site]=z;return z
    contexts=[]
    for row in out.to_dict('records'):
        game=str(row.get('Game','')).replace(' ','').split('@');z={}
        if len(game)==2 and row.get('Team') in game and row.get('Opp') in game:
            home=row['Team']==game[1]
            for side,team,site in [('Team',row['Team'],'H' if home else 'R'),('Opp',row['Opp'],'R' if home else 'H')]:
                c=context(team,site);z['GoalV2_'+side+'_Last_Date']=c.get('Last_Date')
                for field in catalog()['environment_fields']:
                    if field.startswith('NHL_'+side+'_'):z['GoalV2_'+field]=c.get(field.replace('NHL_'+side+'_','NHL_',1))
        contexts.append(z)
    fields=['GoalV2_'+f for f in catalog()['environment_fields']]+['GoalV2_Team_Last_Date','GoalV2_Opp_Last_Date']
    for field in fields:out[field]=[z.get(field) for z in contexts]
    out['GoalV2_Context_Version']=CONTEXT_VERSION;out['GoalV2_Context_AsOf']=before
    return out

def attach_routes(frame):
    import json
    from goal_card_routes import card_moves,summary
    out=frame.copy();rows=out.to_dict('records');vals=[profile_values(r) for r in rows]
    for field in catalog()['player_fields']:
        out['GoalV2_'+field]=[v.get(field) for v in vals]
    summaries=[summary(card_moves(r)) for r in rows]
    for field in ('Goal_Attack','Goal_Attack_Record','Goal_Attack_Sample_BE','Goal_All_Attacks','Goal_Crits','Goal_Buffs'):
        out[field]=[s[field] for s in summaries]
    out['Goal_Move_Version']=VERSION;out['Goal_Move_Label']='Moves V2'
    out['GoalV2_Fired_Ids']=[json.dumps([m['goal_record_key'] for m in records_for(r)]) for r in rows]
    return out
