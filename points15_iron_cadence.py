"""Approved Iron Cadence player route and independent optional buffs."""
from pathlib import Path
import json,math
CATALOG=json.loads(Path(__file__).with_name('points15_iron_cadence_rules.json').read_text(encoding='utf-8'))
VERSION=CATALOG['version']

def number(v):
    try:
        v=float(v);return v if math.isfinite(v) else None
    except (ValueError,TypeError):return None
def prepare(row):
    out=dict(row);out['Log_L5_TOI_per_game']=None;out['Log_L10_PPP_1plus_pct']=None
    try:
        obj=json.loads(row['Form_Log']) if isinstance(row.get('Form_Log'),str) else row['Form_Log']
        if obj.get('season')!='2026-27':return out
        before=str(row.get('Date',''))[:10]
        games=sorted([g for g in obj['games'] if str(g.get('d',''))<before],key=lambda g:g['d'],reverse=True)
        if not games or len({g['d'] for g in games})!=len(games):return out
        times=[]
        for g in games[:5]:
            m,s=str(g['toi']).split(':')
            if int(m)<0 or not 0<=int(s)<60:return out
            times.append(int(m)+int(s)/60)
        if len(times)==min(5,len(games)):out['Log_L5_TOI_per_game']=sum(times)/len(times)
        pp=[number(g.get('pp')) for g in games[:10]]
        if pp and all(v is not None for v in pp):out['Log_L10_PPP_1plus_pct']=100*sum(v>=1 for v in pp)/len(pp)
    except (KeyError,ValueError,TypeError):pass
    return out
def qualifies(row,conditions):
    for f,op,c in conditions:
        v=number(row.get(f))
        if v is None or not (v>=c-1e-10 if op=='>=' else v<=c+1e-10):return False
    return True
def match_prepared(row):
    if number(row.get('Points_Line'))!=1.5 or str(row.get('Matrix_Points','')).strip().lower() not in ('green','🟢'):return []
    if row.get('Model_Stats_Season')!='2026-2027':return []
    conf=number(row.get('Conf_Points'))
    if conf is None or conf<82:return []
    out=[]
    for move in CATALOG['moves']:
        env=move.get('environment_condition')
        if env:
            field=env[0]
            if field in ['opp_5v5_xGA60','opp_5v5_HDCA60','Team_PP_xGF60'] and row.get('Team_Stats_Season')!='2026-2027':continue
            value=number(row.get(field))
            if field in ['opp_5v5_xGA60','opp_5v5_HDCA60'] and (value is None or value<0 or abs(value-50)<1e-10):continue
            if field=='Team_PP_xGF60' and (value is None or value<0):continue
            if field in ['Opp_SV','Opp_GAA']:
                if row.get('Goalie_Stats_Season')!='2026-2027' or str(row.get('Opp_Goalie_Status','')).lower()!='confirmed' or (number(row.get('Opp_GP')) or 0)<=0:continue
                if value is None or (not 0<=value<=1 if field=='Opp_SV' else not 0<=value<=15):continue
            if field.startswith('NHL_'):
                side='Team' if field.startswith('NHL_Team_') else 'Opp'
                last=str(row.get('NHL_'+side+'_Latest_source_date',''))[:10]
                if not last or last>=str(row.get('Date',''))[:10]:continue
        if qualifies(row,move['conditions']):out.append(move)
    return out
def rule_text(move):
    return 'Points1.5 + Green + Conf_Points >= 82 + '+' + '.join(f'{f} {op} {cut:g}' for f,op,cut in move['conditions'])
def fired(row):
    return [{**m,'rule':rule_text(m),'market':'Points','points_current':True,'points15_current':True,
             'points15_v3':True,'track':False,'later_wins':0,'later_picks':0,
             'kit_version':VERSION,'record_version':VERSION} for m in match_prepared(prepare(row))]


def render_family(entries,rank):
    """One visible lane head, nested levels; preserve all eligible records."""
    from html import escape
    ordered=sorted(entries,key=lambda item:rank(item[0]),reverse=True)
    lanes={}
    for m,html in ordered:
        if not m.get('points15_v3') or m.get('kit_version')!=VERSION:return None
        lanes.setdefault(m['display_lane'],[]).append((m,html))
    if not ordered:return ''
    content=[]
    for lane,items in lanes.items():
        lead,html=items[0];rest=''.join(h for _,h in items[1:])
        more=('<details class="wn-details"><summary>Other matched levels &amp; aspects ('+str(len(items)-1)+')</summary>'+rest+'</details>') if rest else ''
        content.append('<details class="wn-details wn-iron-lane"><summary>'+escape(lane)+' · '+escape(lead['name'])+' ('+str(len(items))+')</summary>'+html+more+'</details>')
    return '<details class="wn-goal-record-group wn-iron-family"><summary>V3 · Iron Cadence · '+str(len(lanes))+' lanes · '+str(len(ordered))+' records</summary>'+''.join(content)+'</details>'
