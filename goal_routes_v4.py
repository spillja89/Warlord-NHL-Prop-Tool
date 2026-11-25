"""Goal V4: approved exact-parent optional buffs. Player routes stay unchanged."""
from functools import lru_cache
from pathlib import Path
import json
from goal_routes_v2 import number, _passes
from goal_routes_v3 import profile_values as previous_values, LABELS as PREVIOUS_LABELS

VERSION='goal-2026-27-buffs-v4-2026-11-25'
CONTEXT_VERSION='goal-v4-context-2026-11-25'
LABELS={**PREVIOUS_LABELS,
 'opp_5v5_HDCA60':'Opponent 5v5 high-danger chances allowed/60',
 'Official_Own_PK_Venue_Minutes_PG':'Own PK minutes/game at matching home/away venue',
 'Official_Opp_PK_L10_Success_pct':'Opponent L10 PK success %'}
OFFICIAL=('Official_Own_PP_Venue_O_PG','Official_Own_PK_Venue_Minutes_PG','Official_Opp_PK_L10_Success_pct')

@lru_cache(maxsize=1)
def catalog():return json.loads(Path(__file__).with_name('goal_v4_buffs.json').read_text(encoding='utf-8'))

def profile_values(row):
    out=previous_values(row)
    if row.get('Model_Stats_Season')!='2026-2027':return out
    value=number(row.get('opp_5v5_HDCA60'))
    if row.get('Team_Stats_Season')=='2026-2027' and value is not None and value>=0 and value!=50:out['opp_5v5_HDCA60']=value
    else:out.pop('opp_5v5_HDCA60',None)
    before=str(row.get('Date',''))[:10]
    # New official fields require their own dated source marker. They do not
    # silently reuse similarly named values or fill missing values with zero.
    for field in OFFICIAL:out.pop(field,None)
    if row.get('GoalV4_Context_Version')==CONTEXT_VERSION and row.get('GoalV4_Context_AsOf')==before:
        for field in OFFICIAL:
            side='Own' if field.startswith('Official_Own_') else 'Opp'
            last=str(row.get('GoalV4_'+side+'_Last_Date',''))[:10]
            value=number(row.get('GoalV4_'+field))
            if len(last)==10 and last<before and value is not None and value>=0:out[field]=value
    return out

def _player_moves(row):
    from goal_routes_v2 import records_for as old
    from goal_routes_v3 import records_for as recent
    return {m['id']:m for m in old(row)+recent(row) if not m.get('matchup_buff')}

def _move(spec,parent):
    record=spec['record'];family=spec['family']
    character=parent.get('goal_character') or {'A':'Gorilla · Ravager','F':'Chimp · Reaper','P':'Panda · Executioner','X':'Chimp · Reaper','G':'Gorilla · Ravager','S':'Panda · Executioner'}.get(parent.get('goal_family'),'Gorilla · Ravager')
    return dict(id='goal_v4:'+spec['id'],name=spec['name'],kind='CONTEXT',icon=spec['icon'],rule=spec['rule'],
        wins=record['hits'],picks=record['plays'],later_wins=0,later_picks=0,market='Goal',kit_version=VERSION,
        goal_v2=True,goal_v3=True,goal_v4=True,goal_record_key='v4:'+spec['id'],route_family='v4',
        goal_family='v4:'+family,goal_buff_family=family,goal_parent=spec['parent_id'].split(':',1)[1],
        goal_parent_id=spec['parent_id'],goal_parent_name=spec['parent_name'],goal_character=character,
        goal_primary=False,goal_complete_entry=False,goal_visible_buff=True,matchup_buff=True,goal_retired=False,
        goal_level=1,checks=spec['checks'],all_checks=spec['all_checks'],crit=False,record_version='V4',
        sample_season='2026-27',sample_from='2026-10-13',sample_through='2026-11-24',
        sample_book_be_pct=record['BE'],sample_players=record['players'],sample_dates=record['dates'])

def records_for(row):
    parents=_player_moves(row)
    if not parents:return []
    values=profile_values(row)
    return [_move(s,parents[s['parent_id']]) for s in catalog()['moves']
            if s['parent_id'] in parents and _passes(values,s['checks'])]

def decorate_existing(move):
    """Reuse existing rule identities and history; only annotate display grouping."""
    if move.get('id')=='goal_v3:GUARD':
        return {**move,'goal_v4':True,'goal_buff_family':'fractured_guard','goal_family':'v4:fractured_guard',
                'goal_parent_id':'goal_v3:X_creatorII','record_version':'V4'}
    if move.get('id')=='goal_v3:WAR':
        return {**move,'goal_buff_family':'war_chant','goal_family':'v4:war_chant'}
    return move

def attach_context(frame,before_day,reports,codes):
    """Reuse already downloaded official reports; no extra network requests."""
    import pandas as pd
    out=frame.copy();before=str(before_day)[:10]
    if out.empty:return out
    if set(out.Date.astype(str).str[:10])!={before}:raise ValueError('Goal V4 requires one exact slate date')
    frames={}
    for name,cols in [('powerplay',['ppOpportunities']),('penaltykill',['timesShorthanded','ppGoalsAgainst','pkTimeOnIcePerGame'])]:
        h=reports.get(name,pd.DataFrame()).copy()
        if h.empty or not {'gameDate','gameId','teamId','homeRoad',*cols}.issubset(h):continue
        h=h[h.gameDate.astype(str).str[:10].lt(before)].copy()
        if h.duplicated(['teamId','gameId']).any() or not h.gameId.astype(str).str.startswith('202602').all():raise ValueError('Invalid Goal V4 official history identity')
        for col in cols:
            h[col]=pd.to_numeric(h[col],errors='coerce');h.loc[~h[col].map(lambda x:number(x) is not None and x>=0),col]=float('nan')
        h['Team']=h.teamId.map(codes);frames[name]=h.sort_values(['gameDate','gameId'])
    cache={}
    def context(team,venue):
        if (team,venue) in cache:return cache[team,venue]
        z={}
        for name,h in frames.items():
            h=h[h.Team.eq(team)]
            if h.empty:continue
            z['Last_Date']=max(str(h.gameDate.max())[:10],z.get('Last_Date',''))
            site=h[h.homeRoad.eq(venue)]
            if name=='powerplay' and len(site) and site.ppOpportunities.notna().all():z['Official_Own_PP_Venue_O_PG']=float(site.ppOpportunities.mean())
            if name=='penaltykill':
                if len(site) and site.pkTimeOnIcePerGame.notna().all():z['Official_Own_PK_Venue_Minutes_PG']=float(site.pkTimeOnIcePerGame.mean())/60
                t=h.tail(10)
                if len(t)==10 and t[['timesShorthanded','ppGoalsAgainst']].notna().all().all() and t.timesShorthanded.sum()>0:
                    z['Official_Opp_PK_L10_Success_pct']=100*(1-float(t.ppGoalsAgainst.sum())/float(t.timesShorthanded.sum()))
        cache[team,venue]=z;return z
    values=[]
    for row in out.to_dict('records'):
        teams=str(row.get('Game','')).replace(' ','').split('@');z={}
        if len(teams)==2 and row.get('Team') in teams and row.get('Opp') in teams:
            for side,team in [('Own',row['Team']),('Opp',row['Opp'])]:
                c=context(team,'H' if team==teams[1] else 'R');z['GoalV4_'+side+'_Last_Date']=c.get('Last_Date')
                for field in OFFICIAL:
                    if field.startswith('Official_'+side+'_'):z['GoalV4_'+field]=c.get(field)
        values.append(z)
    for f in ['GoalV4_'+f for f in OFFICIAL]+['GoalV4_Own_Last_Date','GoalV4_Opp_Last_Date']:out[f]=[z.get(f) for z in values]
    out['GoalV4_Context_Version']=CONTEXT_VERSION;out['GoalV4_Context_AsOf']=before
    return out

def attach_routes(frame):
    from goal_routes_v3 import attach_routes as old_attach
    out=old_attach(frame);rows=out.to_dict('records');values=[profile_values(row) for row in rows]
    for f in LABELS:out['GoalV4_'+f]=[v.get(f) for v in values]
    out['Goal_Move_Version']=VERSION;out['Goal_Move_Label']='Moves V4'
    out['GoalV4_Fired_Ids']=[json.dumps([m['goal_record_key'] for m in records_for(row)]) for row in rows]
    return out
