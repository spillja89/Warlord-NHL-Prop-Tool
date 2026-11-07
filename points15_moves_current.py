"""Approved current-season 1.5 Points profiles; dated records, not forecasts."""
from __future__ import annotations
import json,math
from pathlib import Path
from points_context import VERSION as CONTEXT_VERSION

KIT=json.loads(Path(__file__).with_name('points15_current_rules.json').read_text(encoding='utf-8'))
VERSION=KIT['version']
def number(value):
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (TypeError,ValueError):return None

def quote_rows(frame):
    """One resolved primary Points line per real source row, with same-line prices."""
    result=frame.copy()
    if '_points15_alternate' in result.columns:
        result=result.loc[~result['_points15_alternate'].eq(True)].copy()
    from points_primary import resolve_frame
    return resolve_frame(result)

def prepare_row(row):
    result=dict(row)
    fields=('Log_L10_S_per_game','Log_L20_P_per_game','Log_L5_P_per_game','Pace_Gap_P10')
    result.update({field:None for field in fields})
    try:
        log=json.loads(row['Form_Log']) if isinstance(row.get('Form_Log'),str) else row.get('Form_Log')
        if not isinstance(log,dict) or log.get('season')!='2026-27':return result
        date=str(row.get('Date',''))[:10]
        games=sorted([g for g in log['games'] if str(g.get('d',''))<date],key=lambda g:g['d'],reverse=True)
        if not games or len({g['d'] for g in games})!=len(games):return result
        for window,metric,target in ((10,'s','Log_L10_S_per_game'),(20,'p','Log_L20_P_per_game'),(5,'p','Log_L5_P_per_game')):
            recent=games[:window]
            values=[number(g.get('s')) if metric=='s' else
                    number(g.get('g'))+number(g.get('a')) if number(g.get('g')) is not None and number(g.get('a')) is not None else None for g in recent]
            if values and all(v is not None for v in values):result[target]=sum(values)/len(values)
        recent=games[:10];expected=number(row.get('Exp_P_10_Gap_Basis'))
        values=[number(g.get('g'))+number(g.get('a')) if number(g.get('g')) is not None and number(g.get('a')) is not None else None for g in recent]
        if len(recent)>=5 and expected is not None and all(v is not None for v in values):
            result['Pace_Gap_P10']=round(expected-10*sum(values)/len(values),2)
    except (TypeError,KeyError,ValueError):pass
    return result

def points15_moves(row):
    if number(row.get('Points_Line'))!=1.5 or row.get('Model_Stats_Season')!='2026-2027':return []
    if str(row.get('Matrix_Points','')).strip().casefold() not in ('green','🟢'):return []
    row=prepare_row(row)
    def matches(condition):
        field,op,cut=condition
        if field=='PP_Points60':
            source=row.get('PP_Points60_Source')
            if not isinstance(source,str) or not source.strip() or source.strip().casefold() in ('nan','none'):return False
        if field.startswith('NHL_') and row.get('Points_Context_Version')!=CONTEXT_VERSION:return False
        if field.startswith(('team_','Team_')) and row.get('Team_Stats_Season')!='2026-2027':return False
        value=number(row.get(field))
        return value is not None and (value>=cut-1e-10 if op=='>=' else value<=cut+1e-10)
    moves=[]
    for spec in KIT['moves']:
        if not all(matches(c) for c in spec['conditions']):continue
        if spec.get('any_conditions') and not any(all(matches(c) for c in group) for group in spec['any_conditions']):continue
        rule='Points1.5 + Green + '+ ' + '.join(f'{f} {op} {c:g}' for f,op,c in spec['conditions'])
        if spec.get('any_conditions'):
            rule+=' + ('+' OR '.join(' AND '.join(f'{f} {op} {c:g}' for f,op,c in group) for group in spec['any_conditions'])+')'
        moves.append({**spec,'rule':rule,'market':'Points','points_current':True,'points15_current':True,
          'track':False,'later_wins':0,'later_picks':0,'kit_version':VERSION,'record_version':VERSION,
          'sample_season':'2026-27','sample_from':KIT['sample_from'],'sample_through':KIT['sample_through']})
    return moves
