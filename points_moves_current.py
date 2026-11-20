"""Approved current-season Points0.5 profiles and consolidated context moves."""
from __future__ import annotations
import json
import math
from pathlib import Path
from points_context import VERSION as CONTEXT_VERSION, log_context
from card_return_store import record_snapshot

KIT = json.loads(Path(__file__).with_name('points_current_rules.json').read_text(encoding='utf-8'))
VERSION = KIT['version']

def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None

def prepare_row(row):
    result = dict(row)
    logs = log_context(result)
    for field, value in logs.items():
        result[field] = value
    for target, numerator, denominator in [('Team_PP_xGF_relative_league', 'Team_PP_xGF60', 'League_PP_xGF60'), ('Opp_PK_xGA_relative_league', 'Opp_PK_xGA60', 'League_PK_xGA60'), ('Opp_5v5_xGA_relative_league', 'opp_5v5_xGA60', 'League_5v5_xGA60')]:
        if number(result.get(target)) is None:
            a, b = number(result.get(numerator)), number(result.get(denominator))
            if result.get('Team_Stats_Season') == '2026-2027' and a is not None and b is not None and b > 0:
                result[target] = a / b
    if number(result.get('Points_Line')) == 1.5:
        from points15_moves_current import prepare_row as prepare15
        result = prepare15(result)
    if number(row.get('Points_Line')) == .5:
        from points05_v6 import prepare as prepare6
        result = prepare6(result)
    return result

def base_eligible(row):
    confidence = number(row.get('Conf_Points'))
    return (number(row.get('Points_Line')) == .5 and row.get('Model_Stats_Season') == '2026-2027'
            and str(row.get('Matrix_Points', '')).strip().casefold() in {'green', '🟢'}
            and confidence is not None and confidence >= 77)

def field_value(row, field):
    if field.startswith('NHL_') and row.get('Points_Context_Version') != CONTEXT_VERSION:
        return None
    if field in {'iXA%', 'PP_TOI_per_game', 'Assists_mu', 'Drought_G', 'Log_L20_Point2plus_pct', 'Log_L20_PPP_hit_pct'}:
        return number(row.get(field))
    if field in {'PP_PPO_Blend_PG', 'PP_Time_Blend_PG'}:
        return number(row.get(field)) if row.get('PPO_Source') == 'NHL Stats · current season' else None
    if row.get('Team_Stats_Season') != '2026-2027' and not field.startswith('NHL_'):
        return None
    return number(row.get(field))

def points_moves(row):
    if number(row.get('Points_Line')) == 1.5:
        from points15_moves_current import points15_moves
        return points15_moves(row)
    if not base_eligible(row):
        return []
    row = prepare_row(row)
    moves = []
    for spec in KIT['moves']:
        matched = True
        for field, direction, cut in spec['conditions']:
            value = field_value(row, field)
            if value is None or not (value >= cut - 1e-10 if direction == '>=' else value <= cut + 1e-10):
                matched = False
                break
        if not matched:
            continue
        rule = KIT['entry'] + ''.join(f' + {f} {op} {cut:g}' for f, op, cut in spec['conditions'])
        moves.append({**spec, 'rule': rule, 'later_wins': 0, 'later_picks': 0, 'track': False,
                      'market': 'Points', 'points_current': True, 'kit_version': VERSION,
                      'record_version': VERSION, 'sample_season': '2026-27',
                      'sample_from': KIT['sample_from'], 'sample_through': KIT['sample_through']})
    from points05_v6 import moves as moves6
    return moves + moves6(row)

def display_moves(moves):
    """One visible badge/record per nested family; every rule remains frozen."""
    if any(move.get('kit_version')==VERSION or move.get('points05_v6') or move.get('points15_v2') or move.get('points15_v3') or move.get('points15_v4') for move in moves):
        from move_return_grades import grade_rank, stable_identity
        selected={}
        def rank(move):
            return grade_rank(move) or (0,False,move['wins']/move['picks'],move['picks'],0,stable_identity(move))
        for move in sorted(moves,key=rank,reverse=True):
            if move.get('points_current') and not move.get('suppress_badge'):
                selected.setdefault(move['family'],move)
        return list(selected.values())
    selected = {}
    for move in moves:
        if not move.get('points_current') or move.get('suppress_badge'):
            continue
        family = move['family']
        stronger = family not in selected or move.get('level', 0) > selected[family].get('level', 0)
        if not stronger and move.get('points15_current'):
            stronger = (move.get('level', 0), move['wins']/move['picks']) > (selected[family].get('level', 0), selected[family]['wins']/selected[family]['picks'])
        if stronger:
            selected[family] = move
    return list(selected.values())

def best_move(row):
    moves = [m for m in display_moves(points_moves(row)) if m['kind'] != 'CONTEXT']
    if number(row.get('Points_Line'))==.5 or any(m.get('points05_v6') or m.get('points15_v3') or m.get('points15_v4') for m in moves):
        from move_return_grades import grade_rank, stable_identity
        return max(moves,key=lambda m:grade_rank(m) or (0,False,m['wins']/m['picks'],m['picks'],0,stable_identity(m)),default=None)
    return max(moves, key=lambda m: (m['wins'] / m['picks'], m['picks'])) if moves else None

@record_snapshot()
def attach_points_moves(frame):
    import pandas as pd
    out = frame.copy()
    records = out.to_dict('records')
    all_moves = [display_moves(points_moves(row)) for row in records]
    best = [max((m for m in moves if m['kind'] != 'CONTEXT'), key=lambda m: (m['wins'] / m['picks'], m['picks']), default=None) for moves in all_moves]
    # V2 additions use the same return rank as cards; V1-only rows stay identical.
    from move_return_grades import grade_rank, stable_identity
    for i,moves in enumerate(all_moves):
        if moves:
            best[i]=max((m for m in moves if m['kind']!='CONTEXT'),
                        key=lambda m:grade_rank(m) or (0,False,m['wins']/m['picks'],m['picks'],0,stable_identity(m)),default=None)
    from points_return_monitor import display_stats
    shown=[display_stats(m) if m else None for m in best]
    out['Points_Attack'] = [m['name'] if m else '' for m in best]
    out['Points_Move_Record'] = [f"{s['hits']}/{s['plays']}" if s else '' for s in shown]
    out['Points_Move_Pct'] = [100 * s['hits'] / s['plays'] if s and s['plays'] else None for s in shown]
    out['Points_Sample_BE_Pct'] = [s.get('be') if s else None for s in shown]
    out['Points_Move_Record_Version']=[('V6' if number(row.get('Points_Line'))==.5 else 'V5') if m and not m.get('legacy') else '' for row,m in zip(records,best)]
    out['Points_Profile_Routes'] = [' · '.join(m['name'] for m in moves if m['kind'] != 'CONTEXT' and m['id'] != 'base77') for moves in all_moves]
    from points15_moves_current import VERSION as VERSION15
    out['Points_Move_Kit_Version'] = [m['kit_version'] if m and (m.get('points15_v2') or m.get('points15_v3') or m.get('points15_v4')) else VERSION15 if number(row.get('Points_Line')) == 1.5 else VERSION for row,m in zip(records,best)]
    prepared_rows = [prepare_row(row) for row in records]
    for field in ['Log_L10_S_per_game','Log_L20_P_per_game','Log_L5_P_per_game','Pace_Gap_P10']:
        out[field] = pd.Series([row.get(field) for row in prepared_rows], index=out.index, dtype='Float64')
    for field in ['Log_L20_Point2plus_pct', 'Log_L20_PPP_hit_pct', 'Team_PP_xGF_relative_league', 'Opp_PK_xGA_relative_league', 'Opp_5v5_xGA_relative_league']:
        out[field] = pd.Series([row.get(field) for row in prepared_rows], index=out.index, dtype='Float64')
    from points15_iron_cadence import prepare as prepare_iron
    prepared_iron=[prepare_iron(row) if number(row.get('Points_Line'))==1.5 else {} for row in records]
    for field in ['Log_L5_TOI_per_game','Log_L10_PPP_1plus_pct']:
        out[field]=pd.Series([row.get(field) for row in prepared_iron],index=out.index,dtype='Float64')
    from points15_backbone_buffs import prepare as prepare_backbone
    prepared_backbone=[prepare_backbone(row) if number(row.get('Points_Line'))==1.5 else {} for row in records]
    for field in ['Goalie_GSAA_per_game','NHL_Team_Rest_days']:
        out[field]=pd.Series([row.get(field) for row in prepared_backbone],index=out.index,dtype='Float64')
    from points05_v6 import OWNED
    prepared6=[prepared if number(row.get('Points_Line'))==.5 else {} for row,prepared in zip(records,prepared_rows)]
    for field in OWNED:
        out[field]=pd.Series([r.get(field) for r in prepared6],index=out.index,dtype='Float64')
    return out

