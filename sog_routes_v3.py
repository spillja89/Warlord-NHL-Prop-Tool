"""Approved Nov19 SOG routes. Optional contexts are complete alternative routes.

Pure pregame evaluation: no downloads, odds changes, score fitting or pick writes.
"""
from functools import lru_cache
from pathlib import Path
from statistics import mean,pstdev
import json,math

VERSION='2026-11-19-sog25-keepers-v3'
CONTEXT_VERSION='sog-context-2026-11-19-v3'
ENV_FIELDS=('NHL_Opp_SA_L5_PG','NHL_Team_Goals_per_shot_allowed_pct',
            'NHL_Opp_PK_L10_O_PG','NHL_Opp_PK_L5_Time_PG')
PLAYER_FIELDS=('Player_Log_L5_TOI_per_game','Player_Log_L20_TOI_sd',
               'i5v5_iCF60','Trim10_SOG')

def number(v):
    try:
        n=float(v);return n if math.isfinite(n) else None
    except (TypeError,ValueError):return None

@lru_cache(maxsize=1)
def catalog():return json.loads(Path(__file__).with_name('sog_v3_records.json').read_text(encoding='utf8'))

@lru_cache(maxsize=4096)
def _minutes(raw,before):
    try:
        log=json.loads(raw)
        if not isinstance(log,dict) or log.get('season')!='2026-27':return {}
        games=sorted(log['games'],key=lambda g:g['d'],reverse=True)
        if not games or any(str(g.get('d',''))>=before for g in games) or len({g['d'] for g in games})!=len(games):return {}
        times=[]
        for g in games[:20]:
            a,b=str(g['toi']).split(':');a,b=int(a),int(b)
            if a<0 or not 0<=b<60:return {}
            times.append(a+b/60)
        result={'Player_Log_L5_TOI_per_game':mean(times[:5]),'SOGV3_TOI_Window_GP':len(times)}
        if len(times)>=2:result['Player_Log_L20_TOI_sd']=pstdev(times)
        return result
    except (TypeError,ValueError,KeyError):return {}

def values(row):
    if row.get('Model_Stats_Season')!='2026-2027':return {}
    before=str(row.get('Date',''))[:10]
    raw=row.get('Form_Log');raw=json.dumps(raw) if isinstance(raw,dict) else raw
    z=dict(_minutes(raw,before)) if isinstance(raw,str) and len(before)==10 else {}
    trim=number(row.get('Trim10_SOG'))
    if trim is not None and trim>=0:z['Trim10_SOG']=trim
    if row.get('Opportunity_View_Version')=='2026-10-16-v1':
        attempts=number(row.get('i5v5_iCF60'))
        if attempts is not None and attempts>=0:z['i5v5_iCF60']=attempts
    if row.get('SOGV3_Context_Version')==CONTEXT_VERSION and row.get('SOGV3_Context_AsOf')==before:
        for field in ENV_FIELDS:
            side='Team' if field.startswith('NHL_Team_') else 'Opp'
            latest=str(row.get('SOGV3_'+side+'_Last_Date',''))[:10]
            v=number(row.get('SOGV3_'+field))
            if len(latest)==10 and latest<before and v is not None and v>=0:z[field]=v
    return z

def _passed(z,conditions):
    for field,op,cut in conditions:
        v=number(z.get(field))
        if v is None or not (v<=cut+1e-10 if op=='<=' else v>=cut-1e-10):return False
    return True

def moves(row):
    from sog_moves_current import sog_base_eligible
    if not sog_base_eligible(row) or number(row.get('Conf_SOG'))<71:return []
    z=values(row);result=[]
    for s in catalog()['moves']:
        if not _passed(z,s['conditions']):continue
        r=s['record']
        result.append(dict(name=s['name'],kind=s['kind'],icon=s['icon'],rule=s['rule'],
            wins=r['hits'],picks=r['plays'],later_wins=0,later_picks=0,track=False,
            market='SOG',kit_version=VERSION,sog_current=True,sog_profile=True,sog_v3=True,
            sog_family=s['family'],sog_variant=s['variant'],route_id=s['id'],conditions=s['conditions'],
            entry=[['Conf_SOG','>=',71]],sample_season='2026-27',sample_from='2026-10-13',
            sample_through='2026-11-18',sample_book_be_pct=r['BE'],sample_players=r['players'],
            sample_dates=r['dates'],matchup_buff=s.get('matchup_buff',False),optional_buff=s.get('matchup_buff',False),
            parent_route=s['parent_route']))
    return result

def attach_context(frame,before_day,reports,codes):
    """Reuse existing official reports, isolate new columns from other props."""
    import pandas as pd
    out=frame.copy();before=str(before_day)[:10]
    if out.empty:return out
    if set(out.Date.astype(str).str[:10])!={before}:raise ValueError('SOG V3 context requires one pregame date')
    prepared={}
    for name,cols in [('summary',['goalsAgainst','shotsAgainstPerGame']),
                      ('penaltykill',['timesShorthanded','pkTimeOnIcePerGame'])]:
        f=reports.get(name,pd.DataFrame()).copy()
        if f.empty or not {'gameDate','gameId','teamId',*cols}.issubset(f):continue
        f=f.loc[f.gameDate.astype(str).str[:10].lt(before)].copy()
        if f.duplicated(['teamId','gameId']).any() or not f.gameId.astype(str).str.startswith('202602').all():raise ValueError('Invalid SOG V3 official identity')
        for col in cols:
            f[col]=pd.to_numeric(f[col],errors='coerce')
            f.loc[~f[col].map(lambda x:pd.notna(x) and math.isfinite(x) and x>=0),col]=float('nan')
        f['Team']=f.teamId.map(codes);prepared[name]=f.sort_values(['gameDate','gameId'])
    cache={}
    def team_context(team):
        if team in cache:return cache[team]
        z={};a=prepared.get('summary',pd.DataFrame());b=prepared.get('penaltykill',pd.DataFrame())
        if not a.empty:
            a=a.loc[a.Team.eq(team)]
            if not a.empty:
                z['Last_Date']=str(a.gameDate.max())[:10]
                if a.tail(5).shotsAgainstPerGame.notna().all():z['NHL_SA_L5_PG']=float(a.tail(5).shotsAgainstPerGame.mean())
                if a[['goalsAgainst','shotsAgainstPerGame']].notna().all().all() and a.shotsAgainstPerGame.sum()>0:
                    z['NHL_Goals_per_shot_allowed_pct']=100*float(a.goalsAgainst.sum())/float(a.shotsAgainstPerGame.sum())
        if not a.empty and not b.empty:
            b=b.loc[b.Team.eq(team)]
            if set(a.gameId)==set(b.gameId):
                if b.tail(10).timesShorthanded.notna().all():z['NHL_PK_L10_O_PG']=float(b.tail(10).timesShorthanded.mean())
                if b.tail(5).pkTimeOnIcePerGame.notna().all():z['NHL_PK_L5_Time_PG']=float(b.tail(5).pkTimeOnIcePerGame.mean())/60
        cache[team]=z;return z
    contexts=[]
    for row in out.to_dict('records'):
        z={}
        for side in ('Team','Opp'):
            c=team_context(row.get(side));z['SOGV3_'+side+'_Last_Date']=c.get('Last_Date')
            for field in ENV_FIELDS:
                if field.startswith('NHL_'+side+'_'):z['SOGV3_'+field]=c.get(field.replace('NHL_'+side+'_','NHL_',1))
        contexts.append(z)
    for field in ['SOGV3_'+x for x in ENV_FIELDS]+['SOGV3_Team_Last_Date','SOGV3_Opp_Last_Date']:
        out[field]=[z.get(field) for z in contexts]
    out['SOGV3_Context_Version']=CONTEXT_VERSION;out['SOGV3_Context_AsOf']=before
    return out
