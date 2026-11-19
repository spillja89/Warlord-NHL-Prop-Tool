"""Approved PP V2 player profiles and optional buffs, frozen research baseline."""
from functools import lru_cache
from pathlib import Path
import json,math,re,unicodedata

VERSION='pp-moves-v2-2026-11-19'
PLAYER_FIELDS=('Player_L10_P_per60','Player_L5_A_per60','Player_L20_P_total','Player_L20_P_1plus_pct')
GOALIE_FIELDS=('Goalie_SV','Goalie_GAA')
OWNED=PLAYER_FIELDS+GOALIE_FIELDS

def number(v):
    try:
        n=float(v)
        return n if math.isfinite(n) else None
    except (ValueError,TypeError):return None

@lru_cache(maxsize=1)
def catalog():return json.loads(Path(__file__).with_name('pp_v2_rules.json').read_text(encoding='utf-8'))

@lru_cache(maxsize=4096)
def log_values(raw,before):
    out={f:None for f in PLAYER_FIELDS}
    try:
        obj=json.loads(raw)
        if not isinstance(obj,dict) or obj.get('season')!='2026-27' or not isinstance(obj.get('games'),list):return out
        games=obj['games']
        if any(not isinstance(g,dict) or not isinstance(g.get('d'),str) for g in games):return out
        games=sorted([g for g in games if g['d']<before],key=lambda g:g['d'],reverse=True)
        if not games or len({g['d'] for g in games})!=len(games):return out
        def totals(part):
            vals=[(number(g.get('g')),number(g.get('a'))) for g in part]
            return vals if all(a is not None and b is not None and a>=0 and b>=0 for a,b in vals) else None
        last20=games[:20];values=totals(last20)
        if values is not None:
            out['Player_L20_P_total']=sum(g+a for g,a in values)
            out['Player_L20_P_1plus_pct']=100*sum(g+a>=1 for g,a in values)/len(values)
        for size,field,metric in [(10,'Player_L10_P_per60','P'),(5,'Player_L5_A_per60','A')]:
            part=games[:size];values=totals(part);minutes=[]
            for g in part:
                m=re.fullmatch(r'(\d+):(\d{2})',str(g.get('toi','')))
                minutes.append(int(m[1])+int(m[2])/60 if m and int(m[2])<60 else None)
            if values is not None and all(t is not None for t in minutes) and sum(minutes)>0:
                out[field]=60*sum(g+a if metric=='P' else a for g,a in values)/sum(minutes)
    except (ValueError,TypeError,KeyError,ZeroDivisionError):pass
    return out

def normal_name(v):return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',str(v)).encode('ascii','ignore').decode().lower())

def goalie_valid(row):
    name=normal_name(row.get('Opp_Goalie',''));stats=normal_name(row.get('Opp_Goalie_Stats_Name',''))
    gp,sv=number(row.get('Opp_GP')),number(row.get('Opp_SV'))
    return bool(row.get('Goalie_Stats_Season')=='2026-2027'
                and str(row.get('Opp_Goalie_Source','')).endswith('name_and_stats')
                and str(row.get('Opp_Goalie_Status','')).casefold() in {'confirmed','official'}
                and name not in {'','none','nan','unknown'} and (stats in {'','none','nan'} or stats==name)
                and gp is not None and gp>0 and sv is not None and 0<=sv<=1)

def prepare(row):
    out=dict(row)
    for f in OWNED:out[f]=None
    if row.get('Model_Stats_Season')!='2026-2027':return out
    raw=row.get('Form_Log')
    if isinstance(raw,dict):raw=json.dumps(raw,sort_keys=True)
    if isinstance(raw,str):out.update(log_values(raw,str(row.get('Date',''))[:10]))
    if goalie_valid(row):
        out['Goalie_SV']=number(row.get('Opp_SV'))
        g=number(row.get('Opp_GAA'));out['Goalie_GAA']=g if g is not None and 0<=g<=15 else None
    return out

def value(row,field,line=.5):
    if field=='PPP_Line':return number(line)
    if field.startswith('Official_Opp_PK_'):
        from pp_v2_context import VERSION as CONTEXT
        if row.get('PP_V2_Context_Version')!=CONTEXT or str(row.get('PP_V2_Context_AsOf'))!=str(row.get('Date'))[:10]:return None
    if field=='opp_5v5_xGA60':
        v=number(row.get(field))
        return v if row.get('Team_Stats_Season')=='2026-2027' and v is not None and 0<=v<50 else None
    if field=='i5v5_primaryAssists60' and (number(row.get('EV5_TOI_per_game')) or 0)<=0:return None
    if field=='PP_OnIce_xG_per_game':
        from pp_cards import onice_xg
        return onice_xg(row)
    return number(row.get(field))

def fired(row,line=.5):
    if number(line)!=.5 or row.get('Model_Stats_Season')!='2026-2027':return []
    row=prepare(row);result=[]
    for move in catalog()['moves']:
        matched=True
        for f,op,c in move['entry']+move['conditions']:
            v=value(row,f,line)
            if v is None or not (v>=c if op=='>=' else v<=c if op=='<=' else v==c):matched=False;break
        if matched:result.append(move)
    return result

def attach_fields(frame):
    out=frame.copy()
    values=[prepare(r) for r in out.to_dict('records')]
    for f in OWNED:out[f]=[r.get(f) for r in values]
    out['PP_V2_Player_Version']=VERSION
    return out
