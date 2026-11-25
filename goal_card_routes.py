"""Approved Goal V2 board; immutable V1 records remain separately in Inspect."""
from collections import OrderedDict
from goal_routes_v2 import records_for as v2_records, catalog
from goal_routes_v3 import records_for as v3_records
from goal_routes_v4 import records_for as v4_records, decorate_existing, VERSION

def records_for(row):
    """Additive eligibility. Never pass new route gates into retained rules."""
    return [decorate_existing(m) for m in v2_records(row)+v3_records(row)]+v4_records(row)

GROUP_NAMES={'A':'Creator Fade · base attack','F':'Reaper’s Return · finishing','P':'Runic Barrage · PP shooting',
 'X':'Measured Strike · levels','G':'Controlled Strike · goal pace','S':'Predator’s Mark · scoring share'}

def parent_attack_name(move):
    """Name the exact attack whose population produced a buff's record."""
    if move.get('goal_v4'):return move.get('goal_parent_name','')
    if move.get('goal_v3'):return move.get('goal_parent_name','')
    return next((s['name'] for s in catalog()['moves']
                 if s['id']==move.get('goal_parent') and s['kind']=='player_attack'), '')

def buffs_for_attack(moves,attack):
    return [m for m in moves if m.get('goal_visible_buff')
            and m.get('goal_parent')==str((attack or {}).get('id','')).removeprefix('goal_v2:').removeprefix('goal_v3:')
            and bool(m.get('goal_v3'))==bool((attack or {}).get('goal_v3'))]

def player_route_family(move):
    if move.get('goal_v3'):return move.get('goal_family')
    if not move.get('matchup_buff'):return move.get('goal_family')
    return next((s['family'] for s in catalog()['moves']
                 if s['id']==move.get('goal_parent') and s['kind']=='player_attack'),None)

def move_rank(move):
    from move_return_grades import grade_rank,stable_identity
    n=int(move['picks'])
    if move.get('matchup_buff') and not move.get('goal_v2'):
        rank=grade_rank(move)
        return (-1,False,rank[0] if rank else 0,rank[2] if rank else move['wins']/n,rank[3] if rank else n,stable_identity(move))
    return grade_rank(move) or (0,False,move['wins']/n if n else -1,n,0,stable_identity(move))

def display_moves(moves,rank=move_rank):
    selected=OrderedDict()
    for m in sorted(moves,key=rank,reverse=True):
        if m.get('goal_v2'):selected.setdefault((bool(m.get('matchup_buff')),m['goal_family']),m)
    return list(selected.values()) if any(m.get('goal_primary') or m.get('goal_complete_entry') for m in selected.values()) else []

def best_move(moves):
    """A player route admits the card; the best qualified complete rule leads it."""
    return max(display_moves(moves),key=move_rank,default=None)

def inspect_records_for(row):
    from goal_card_routes_v1 import records_for as old_records
    current=records_for(row)
    if not current:return []
    archive=[{**m,'goal_archived':True,'goal_retired':True,'goal_family':'archive_v1','goal_primary':False,'goal_visible_buff':False} for m in old_records(row)]
    return current+archive

def inspect_groups(moves):
    groups=OrderedDict()
    for m in moves:
        family=m.get('goal_family')
        name=('Archived V1 · prior rules and returns' if m.get('goal_archived') else
              'V4 buff family · '+m['name'] if m.get('goal_v4') else
              f'{"V3" if m.get("goal_v3") else "V2"} buffs · {parent_attack_name(m)}' if m.get('goal_v2') and m.get('matchup_buff') else
              'V1 buffs · exact player-parent records' if m.get('matchup_buff') else
              'V3 · '+str(family) if m.get('goal_v3') else GROUP_NAMES.get(family,'Other Goal records'))
        groups.setdefault(name,[]).append(m)
    return groups

def summary(moves):
    visible=display_moves(moves);attacks=sorted((m for m in visible if m.get('goal_primary')),key=move_rank,reverse=True)
    best=best_move(moves)
    def record(m):return f"{m['wins']}/{m['picks']} · {100*m['wins']/m['picks']:.1f}%"
    def entries(ms):return '\n'.join(f"{m['name']} · {record(m)}" for m in ms) or '—'
    return {'Goal_Attack':best['name'] if best else '—','Goal_Attack_Record':record(best) if best else '—',
      'Goal_Attack_Sample_BE':best.get('sample_book_be_pct') if best else None,'Goal_All_Attacks':entries(attacks),
      'Goal_Crits':entries(m for m in visible if m.get('crit')),'Goal_Buffs':entries(m for m in visible if m.get('goal_visible_buff'))}

def card_moves(row):return display_moves(records_for(row))
