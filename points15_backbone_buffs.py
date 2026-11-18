"""Approved independent Backbone environment buffs; existing attacks stay external."""
from datetime import date
from html import escape
from pathlib import Path
import json,math

CATALOG=json.loads(Path(__file__).with_name('points15_backbone_buffs_rules.json').read_text(encoding='utf-8'))
VERSION=CATALOG['version']

def number(value):
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (ValueError,TypeError,OverflowError):return None

def prepare(row):
    from points15_moves_current import prepare_row
    result=prepare_row(row)
    # Derive the same research definitions; never trust an old derived column.
    result['Goalie_GSAA_per_game']=None
    gp,shots,ga,league=(number(row.get(f)) for f in ('Opp_GP','Opp_Goalie_Shots_Faced','Opp_Goalie_GA','Goalie_League_SV'))
    if (row.get('Goalie_Stats_Season')=='2026-2027' and str(row.get('Opp_Goalie_Status','')).strip().casefold()=='confirmed'
            and all(v is not None for v in (gp,shots,ga,league)) and gp>0 and shots>=ga>=0 and 0<=league<=1):
        result['Goalie_GSAA_per_game']=(shots*(1-league)-ga)/gp
    result['NHL_Team_Rest_days']=None
    try:
        before=date.fromisoformat(str(row.get('Date',''))[:10])
        last=date.fromisoformat(str(row.get('NHL_Team_Latest_source_date',''))[:10])
        if last<before:result['NHL_Team_Rest_days']=(before-last).days-1
    except (ValueError,TypeError):pass
    return result

def environment_available(row,field):
    value=number(row.get(field))
    if value is None:return False
    if field in {'opp_5v5_HDCA60','Opp_PK_GA60'}:
        return row.get('Team_Stats_Season')=='2026-2027' and value>=0 and not (field=='opp_5v5_HDCA60' and abs(value-50)<1e-10)
    if field in {'Opp_SV','Goalie_GSAA_per_game'}:
        gp=number(row.get('Opp_GP'))
        return (row.get('Goalie_Stats_Season')=='2026-2027' and str(row.get('Opp_Goalie_Status','')).strip().casefold()=='confirmed'
                and gp is not None and gp>0 and (0<=value<=1 if field=='Opp_SV' else True))
    if field.startswith('NHL_'):
        from points_context import VERSION as CONTEXT_VERSION
        if row.get('Points_Context_Version')!=CONTEXT_VERSION:return False
        side='Team' if field.startswith('NHL_Team_') else 'Opp'
        gp=number(row.get('NHL_'+side+'_Prior_GP'))
        try:
            last=date.fromisoformat(str(row.get('NHL_'+side+'_Latest_source_date',''))[:10])
            before=date.fromisoformat(str(row.get('Date',''))[:10])
            return gp is not None and gp>=3 and last<before and value>=0
        except (ValueError,TypeError):return False
    return False

def rule_text(move):
    return 'Points1.5 + Green + Conf_Points >= 82 + '+' + '.join(f'{f} {op} {c:g}' for f,op,c in move['conditions'])

def match_prepared(row):
    if (number(row.get('Points_Line'))!=1.5 or row.get('Model_Stats_Season')!='2026-2027'
            or str(row.get('Matrix_Points','')).strip().casefold() not in {'green','🟢'}):return []
    conf,gap=number(row.get('Conf_Points')),number(row.get('Pace_Gap_P10'))
    if conf is None or conf<82 or gap is None or gap<.6-1e-10:return []
    matches=[]
    for move in CATALOG['moves']:
        field,op,cut=move['environment_condition']
        if not environment_available(row,field):continue
        value=number(row.get(field))
        if value>=cut-1e-10 if op=='>=' else value<=cut+1e-10:matches.append(move)
    return matches

def fired(row):
    return [{**m,'rule':rule_text(m),'market':'Points','points_current':True,'points15_current':True,
             'points15_v4':True,'track':False,'later_wins':0,'later_picks':0,
             'kit_version':VERSION,'record_version':VERSION} for m in match_prepared(prepare(row))]

def render_family(entries,rank):
    ordered=sorted(entries,key=lambda item:rank(item[0]),reverse=True)
    lanes={}
    for move,html in ordered:
        if not move.get('points15_v4') or move.get('kit_version')!=VERSION:return None
        lanes.setdefault(move['display_lane'],[]).append((move,html))
    if not ordered:return ''
    content=[]
    for lane,items in lanes.items():
        lead,html=items[0];rest=''.join(h for _,h in items[1:])
        more=('<details class="wn-details"><summary>Other matched levels &amp; aspects ('+str(len(items)-1)+')</summary>'+rest+'</details>') if rest else ''
        content.append('<details class="wn-details wn-backbone-lane"><summary>'+escape(lane)+' · '+escape(lead['name'])+' ('+str(len(items))+')</summary>'+html+more+'</details>')
    return '<details class="wn-goal-record-group wn-backbone-family"><summary>V4 · Backbone · '+str(len(lanes))+' families · '+str(len(ordered))+' records</summary>'+''.join(content)+'</details>'
