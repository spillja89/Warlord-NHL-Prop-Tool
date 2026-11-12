"""Approved SOG player routes, strict pregame sources, and separate monitors."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import json,math

VERSION='2026-11-10-sog25-profiles-v3'
CONTEXT_VERSION='sog-profile-context-2026-11-10-v1'

def number(value):
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (TypeError,ValueError):return None

@lru_cache(maxsize=1)
def catalog():return json.loads(Path(__file__).with_name('sog_profile_records.json').read_text(encoding='utf-8'))

@lru_cache(maxsize=4096)
def _log_values(raw,before):
    try:
        data=json.loads(raw)
        if data.get('season')!='2026-27':return {}
        games=sorted(data['games'],key=lambda g:g['d'],reverse=True)
        if not games or any(str(g.get('d',''))>=before for g in games) or len({g['d'] for g in games})!=len(games):return {}
        if any(number(g.get(k)) is None or number(g.get(k))<0 for g in games for k in ('g','s')):return {}
        last=games[:20];five=games[:5];ten=games[:10]
        z={'L20_S':sum(g['s'] for g in last),
           'Player_Log_L20_S_hit_pct':100*sum(g['s']>2.5 for g in last)/len(last),
           'Player_Log_L20_G_hit_pct':100*sum(g['g']>0 for g in last)/len(last),
           'Player_Log_L10_S_per_game':sum(g['s'] for g in ten)/len(ten),
           'Drought_G':next((i for i,g in enumerate(games) if g['g']>0),len(games)),
           'SOG_Profile_Log_GP':len(games)}
        times=[]
        for g in five:
            try:
                a,b=str(g['toi']).split(':');a,b=int(a),int(b)
                if a<0 or not 0<=b<60:raise ValueError('Invalid clock')
                times.append(a+b/60)
            except (TypeError,ValueError,KeyError):break
        if len(times)==len(five):z['Player_Log_L5_TOI_per_game']=sum(times)/len(times)
        return z
    except (TypeError,ValueError,KeyError):return {}

def profile_values(row):
    """Rebuild player summaries from archived pregame logs, never current outcomes."""
    if row.get('Model_Stats_Season')!='2026-2027':return {}
    before=str(row.get('Date',''))[:10]
    raw=row.get('Form_Log');raw=json.dumps(raw) if isinstance(raw,dict) else raw
    z=dict(_log_values(raw,before)) if isinstance(raw,str) and len(before)==10 else {}
    # The saved opportunity is EV5 rate*minutes + PP rate*minutes. Require both
    # components; the old 5v5-only fallback does not become an observed zero PP.
    if row.get('Opportunity_View_Version')=='2026-10-16-v1' and row.get('Team_Stats_Season')=='2026-2027':
        base,ev,pp=(number(row.get(f)) for f in ('SOG_Opportunity_Base','SOG_Opportunity_EV5','SOG_Opportunity_PP'))
        if None not in (base,ev,pp) and min(base,ev,pp)>=0 and abs(base-ev-pp)<=.00151:z['SOG_Opportunity_Base']=base
    # Official context is attached once from the same reports already fetched
    # for Points/PP. Do not trust an undated numeric alias on an old snapshot.
    if row.get('SOG_Profile_Context_Version')==CONTEXT_VERSION and row.get('SOG_Profile_Context_AsOf')==before:
        for field,side in [('NHL_Opp_PK_Goal_drought','Opp'),('NHL_Opp_PK_L5_Success_pct','Opp'),('NHL_Team_PP_L5_O_PG','Team')]:
            latest=str(row.get('SOG_Profile_'+side+'_Last_Date',''))[:10];v=number(row.get(field))
            if latest and latest<before and v is not None and v>=0:z[field]=v
    return z

def attach_context(frame,before_day,reports,codes):
    import pandas as pd
    out=frame.copy();before=str(before_day)[:10]
    if out.empty:return out
    if set(out.Date.astype(str).str[:10])!={before}:raise ValueError('SOG context requires one exact pregame date')
    prepared={}
    for name,required in [('powerplay',['ppOpportunities']),('penaltykill',['timesShorthanded','ppGoalsAgainst'])]:
        source=reports.get(name,pd.DataFrame()).copy()
        if source.empty or not {'gameDate','gameId','teamId',*required}.issubset(source):continue
        source=source.loc[source.gameDate.astype(str).str[:10].lt(before)].copy()
        if source.duplicated(['teamId','gameId']).any() or not source.gameId.astype(str).str.startswith('202602').all():raise ValueError('Invalid SOG official game identity')
        for field in required:
            source[field]=pd.to_numeric(source[field],errors='coerce')
            source.loc[~source[field].map(lambda x:pd.notna(x) and math.isfinite(x) and x>=0),field]=float('nan')
        source['Team']=source.teamId.map(codes);prepared[name]=source.sort_values(['gameDate','gameId'])
    team={};opp={}
    if 'powerplay' in prepared:
        for name,h in prepared['powerplay'].groupby('Team'):
            recent=h.tail(5)
            if recent.ppOpportunities.notna().all():team[name]={'NHL_Team_PP_L5_O_PG':float(recent.ppOpportunities.mean()),'SOG_Profile_Team_Last_Date':str(h.gameDate.max())[:10]}
    if 'penaltykill' in prepared:
        for name,h in prepared['penaltykill'].groupby('Team'):
            z={'SOG_Profile_Opp_Last_Date':str(h.gameDate.max())[:10]};recent=h.tail(5)
            if h.ppGoalsAgainst.notna().all():z['NHL_Opp_PK_Goal_drought']=next((i for i,g in enumerate(h.ppGoalsAgainst.iloc[::-1]) if g>0),len(h))
            if recent[['timesShorthanded','ppGoalsAgainst']].notna().all().all() and recent.timesShorthanded.sum()>0:
                z['NHL_Opp_PK_L5_Success_pct']=100*(1-recent.ppGoalsAgainst.sum()/recent.timesShorthanded.sum())
            opp[name]=z
    rows=[]
    for row in out.to_dict('records'):
        z={**team.get(row.get('Team'),{}),**opp.get(row.get('Opp'),{})}
        rows.append(z)
    for field in ['NHL_Team_PP_L5_O_PG','NHL_Opp_PK_Goal_drought','NHL_Opp_PK_L5_Success_pct','SOG_Profile_Team_Last_Date','SOG_Profile_Opp_Last_Date']:
        out[field]=[z.get(field) for z in rows]
    out['SOG_Profile_Context_Version']=CONTEXT_VERSION;out['SOG_Profile_Context_AsOf']=before
    return out

def _eligible(row):
    from sog_moves_current import sog_base_eligible
    return sog_base_eligible(row) and number(row.get('Conf_SOG'))>=71

def _passed(values,spec):
    for field,op,cut in spec['conditions']:
        v=number(values.get(field))
        if v is None or not (v<=cut+1e-10 if op=='<=' else v>=cut-1e-10):return False
    return True

def _move(spec,monitor=False):
    r=spec['record']
    return {'name':spec['name'],'kind':spec['kind'],'icon':spec['icon'],'rule':spec['rule'],
            'wins':r['hits'],'picks':r['plays'],'later_wins':0,'later_picks':0,'track':monitor,
            'market':'SOG','kit_version':VERSION,'sog_current':True,'sog_profile':True,
            'sog_family':spec['family'],'sog_variant':spec['variant'],'route_id':spec['id'],
            'conditions':spec['conditions'],'sample_season':'2026-27','sample_from':'2026-10-13','sample_through':'2026-11-09',
            'sample_book_be_pct':r['BE'],'sample_players':r['players'],'sample_dates':r['dates'],
            'matchup_buff':spec.get('matchup_buff',False),'research_monitor':monitor}

def profile_moves(row,*,monitors=False):
    if not _eligible(row):return []
    values=profile_values(row)
    return [_move(s,monitors) for s in catalog()['monitors' if monitors else 'moves'] if _passed(values,s)]

def retired_from_cards(move):
    """User-approved Nov11 display retirement; raw evaluation/tracking stays intact."""
    return bool(move.get('sog_current') and not move.get('sog_profile')
                and move.get('name')=='Assassin’s Overdrive')

def active_moves(moves):
    return [m for m in moves if not retired_from_cards(m)]

def inspect_moves(moves,rank):
    archived=[{**m,'sog_retired':True} for m in moves if retired_from_cards(m)]
    return sorted(display_moves(moves,rank),key=rank,reverse=True)+archived

def display_moves(moves,rank):
    """One lead per route family. Frozen captures still retain every rule."""
    chosen={}
    for move in sorted(moves,key=rank,reverse=True):
        if move.get('research_monitor') or retired_from_cards(move):continue
        family=move.get('sog_family') or ('legacy_environment' if move.get('matchup_buff') else 'legacy_sog')
        chosen.setdefault(family,move)
    return list(chosen.values())
