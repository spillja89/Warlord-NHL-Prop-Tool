"""Approved Assist V3 profiles and exact-parent optional context records."""
from pathlib import Path
from functools import lru_cache
from statistics import mean, median, pstdev
from datetime import date
import json, math
from assist_additions import number

VERSION='assist-2026-27-moves-v3-2026-11-24'
CONTEXT_VERSION='assist-v3-context-2026-11-24'
LABELS={
 'Log_L10_Goals_per60':'L10 goals/60 total ice minutes',
 'Assists_mu':'Saved assist mean', 'Log_L5_A_2plus_pct':'L5 2+ assist game frequency %',
 'Log_L40_A_1plus_pct':'Up-to40 assist hit frequency %',
 'Log_L20_TOI_median':'Up-to20 median ice time min/game',
 'Log_L40_S_per_game':'Up-to40 average SOG/game',
 'Log_L10_SOG_per60':'L10 SOG/60 total ice minutes',
 'Log_L20_A_per_game':'Up-to20 assists/game',
 'EV5_TOI_per_game':'Season 5v5 minutes/game',
 'Log_L5_SOG_SD':'L5 shot variability (population SD)',
 'Pace_Gap_S10':'Saved expected minus observed 10-game shot pace',
 'Pace_Gap_A10':'Saved expected minus observed 10-game assist pace',
 'Official_Player_MinutesPG_Season':'Official season PP minutes/dressed game',
 'Log_L20_TOI_SD':'Up-to20 ice-time variability min (population SD)',
 'Log_Season_A_1plus_pct':'Season assist hit frequency %',
 'ShotIntent_Pct':'Shot-intent percentile',
 'Log_L40_S_4plus_pct':'Up-to40 4+ SOG game frequency %',
 'Log_Season_Assists_per60':'Season assists/60 total ice minutes',
 'Player_5v5_SOG_Share':'Measured player 5v5 shot share %',
 'PP_Points60':'PP xG + weighted-assist proxy/60',
 'Official_Own_PK_Season_Goals_PG':'Own season PK goals allowed/game',
 'Official_Opp_L10_GA_PG':'Opponent complete L10 goals allowed/game',
 'Opp_5v5_xGA_relative_league':'Opponent 5v5 xGA/60 divided by league rate',
 'Opp_EV_DZ_Time_Pct':'Opponent even-strength defensive-zone time %',
 'Official_Opp_PP_Venue_G60':'Opponent PP goals/60 at matching home/away venue',
 'Official_Own_PP_Season_O_PG':'Own season PP opportunities/game',
 'Official_Opp_L20_GF_PG':'Opponent complete L20 goals scored/game',
 'Official_Opp_L20_SA_PG':'Opponent complete L20 SOG allowed/game',
 'Goalie_GAA':'Matched opponent goalie GAA', 'Conf_Assists':'Assist confidence'}
OFFICIAL=tuple(f for f in LABELS if f.startswith('Official_'))
LOG_FIELDS=tuple(f for f in LABELS if f.startswith('Log_'))
ACCENTS={'Runesmith':'#79afff','Oracle':'#c9b6ff','Stormcaller':'#63e1cf'}
ICONS={'Runesmith':'runic_infusion.svg','Oracle':'odins_arcane_orb.svg','Stormcaller':'arcane_channel_ii.svg'}

@lru_cache(maxsize=1)
def catalog():return json.loads(Path(__file__).with_name('assist_v3_rules.json').read_text(encoding='utf-8'))

@lru_cache(maxsize=4096)
def form_values(raw,before):
    try:
        doc=json.loads(raw)
        if not isinstance(doc,dict) or doc.get('season')!='2026-27':return {}
        games=doc.get('games')
        if not isinstance(games,list) or not games or any(not isinstance(g,dict) for g in games):return {}
        games=sorted(games,key=lambda g:str(g['d']),reverse=True)
        if any(date.fromisoformat(g['d'])>=date.fromisoformat(before) for g in games) or len({g['d'] for g in games})!=len(games):return {}
        if any(number(g.get(k)) is None or number(g[k])<0 for g in games for k in ('g','a','s')):return {}
        z={}
        for w in (5,10,20,40,'Season'):
            gs=games if w=='Season' else games[:w];n=len(gs);a=[float(g['a']) for g in gs];s=[float(g['s']) for g in gs]
            pre='Log_Season' if w=='Season' else f'Log_L{w}'
            if w in (40,'Season'):z[pre+'_A_1plus_pct']=100*sum(v>=1 for v in a)/n
            if w==5:z.update(Log_L5_A_2plus_pct=100*sum(v>=2 for v in a)/n,Log_L5_SOG_SD=pstdev(s))
            if w==20:z['Log_L20_A_per_game']=mean(a)
            if w==40:z.update(Log_L40_S_per_game=mean(s),Log_L40_S_4plus_pct=100*sum(v>=4 for v in s)/n)
            times=[]
            for g in gs:
                try:
                    m,sec=map(int,str(g.get('toi','')).split(':'))
                    if m<0 or not 0<=sec<60:break
                    times.append(m+sec/60)
                except (ValueError,TypeError):break
            if len(times)==n:
                if w==20:z.update(Log_L20_TOI_median=median(times),Log_L20_TOI_SD=pstdev(times))
                if sum(times)>0:
                    if w==10:z.update(Log_L10_Goals_per60=60*sum(float(g['g']) for g in gs)/sum(times),Log_L10_SOG_per60=60*sum(s)/sum(times))
                    if w=='Season':z['Log_Season_Assists_per60']=60*sum(a)/sum(times)
        if len(games)>=5:
            gs=games[:10];z['_A10']=mean(float(g['a']) for g in gs)*10;z['_S10']=mean(float(g['s']) for g in gs)*10
        return z
    except (TypeError,ValueError,KeyError):return {}

def goalie_gaa(row):
    from goal_routes_v3 import _normal
    from datetime import datetime
    gp=number(row.get('Opp_GP'));gaa=number(row.get('Opp_GAA'));sv=number(row.get('Opp_SV'))
    name=_normal(row.get('Opp_Goalie'));matched=row.get('Opp_Goalie_Stats_Name')
    if (row.get('Goalie_Stats_Season')!='2026-2027' or not str(row.get('Opp_Goalie_Source','')).endswith('name_and_stats')
        or str(row.get('Opp_Goalie_Status','')).lower() not in ('confirmed','official')
        or name in ('','none','nan','unknown') or gp is None or gp<=0 or gaa is None or gaa<0 or sv is None or not 0<=sv<=1):return None
    if isinstance(matched,str) and matched.strip() and _normal(matched)!=name:return None
    check=str(row.get('Goalie_Check_UTC') or '')
    if check and check.lower() not in ('nan','none'):
        try:
            c=datetime.fromisoformat(check.replace('Z','+00:00'));s=datetime.fromisoformat(str(row.get('StartTimeUTC')).replace('Z','+00:00'))
            if not c.tzinfo or not s.tzinfo or c>=s:return None
        except (TypeError,ValueError):return None
    return gaa

def prepare(row):
    r=dict(row)
    for f in LOG_FIELDS+OFFICIAL+('Pace_Gap_A10','Pace_Gap_S10','Goalie_GAA','Opp_5v5_xGA_relative_league','Opp_EV_DZ_Time_Pct'):r[f]=None
    if row.get('Model_Stats_Season')!='2026-2027':return r
    before=str(row.get('Date',''))[:10]
    try:date.fromisoformat(before)
    except (TypeError,ValueError):return r
    raw=row.get('Form_Log');raw=json.dumps(raw,sort_keys=True) if isinstance(raw,dict) else raw
    z=form_values(raw,before) if isinstance(raw,str) else {};r.update({k:v for k,v in z.items() if not k.startswith('_')})
    for f,basis,actual in [('Pace_Gap_A10','Exp_A_10_Gap_Basis','_A10'),('Pace_Gap_S10','Exp_S_10','_S10')]:
        b=number(row.get(basis))
        if b is not None and actual in z:r[f]=round(b-z[actual],2)
    if row.get('AssistV3_Context_Version')==CONTEXT_VERSION and row.get('AssistV3_Context_AsOf')==before:
        for f in OFFICIAL:
            side='Player' if f.startswith('Official_Player_') else 'Own' if f.startswith('Official_Own_') else 'Opp'
            last=str(row.get('AssistV3_'+side+'_Last_Date',''))[:10]
            if len(last)==10 and last<before:r[f]=number(row.get('AssistV3_'+f))
    if row.get('Team_Stats_Season')=='2026-2027':
        a,b=number(row.get('opp_5v5_xGA60')),number(row.get('League_5v5_xGA60'))
        if a is not None and a!=50 and b is not None and b>0:r['Opp_5v5_xGA_relative_league']=a/b
    dz=number(row.get('Opp_EV_DZ_Time_Pct'));r['Opp_EV_DZ_Time_Pct']=dz if dz is not None and 0<=dz<=100 else None
    r['Goalie_GAA']=goalie_gaa(row)
    # Missing 5v5/PP sources cannot qualify via saved neutral placeholders.
    if not isinstance(row.get('PP_Points60_Source'),str) or not row['PP_Points60_Source'].strip():r['PP_Points60']=None
    return r

def matches(row,checks):
    for f,op,c in checks:
        n=number(row.get(f))
        if n is None or not (n>=c-1e-10 if op=='>=' else n<=c+1e-10):return False
    return True

def entry(row):
    return (row.get('Model_Stats_Season')=='2026-2027' and number(row.get('Assists_Line',row.get('Line_Assists')))==.5
      and str(row.get('Matrix_Assists',row.get('Matrix_A',''))).strip().casefold() in ('green','🟢')
      and number(row.get('Conf_Assists')) is not None and number(row['Conf_Assists'])>=81)

def rule_text(spec):return 'Assist0.5 + Matrix_Assists Green + Conf_Assists >=81'+''.join(f' + {f} {op} {c:g}' for f,op,c in spec['conditions'])

def all_fired(row):
    if not entry(row):return []
    r=prepare(row);out=[]
    for spec in catalog()['moves']:
        if not matches(r,spec['conditions']):continue
        v=spec['route_variant'];f=spec['full'];buff=spec['optional_buff']
        out.append({**spec,'move_id':spec['id'],'market':'Assists','rule':rule_text(spec),
          'wins':f['hits'],'picks':f['plays'],'later_wins':0,'later_picks':0,
          'sample_players':f['players'],'unique_players':f['players'],'sample_dates':f['dates'],
          'sample_book_be_pct':f['be'],'sample_from':'2026-10-13','sample_through':'2026-11-23',
          'sample_season':'2026-27','kit_version':VERSION,'record_version':VERSION,
          'assist_v3':True,'track':False,'icon':ICONS[v],'route_accent':ACCENTS[v],
          'kind':'BUFFED ATTACK' if buff else spec['display_level'],'level_label':spec['display_level'],
          'display_priority':4,'environment_enhanced':buff})
    return out

def attach_routes(frame):
    out=frame.copy();rows=out.to_dict('records');vals=[prepare(r) for r in rows]
    for f in LABELS:out['AssistV3_Value_'+f]=[v.get(f) for v in vals]
    out['Assist_Move_Version']=VERSION;out['Assist_Move_Label']='Moves V3'
    out['AssistV3_Fired_Ids']=[json.dumps([m['id'] for m in all_fired(r)]) for r in rows]
    return out
