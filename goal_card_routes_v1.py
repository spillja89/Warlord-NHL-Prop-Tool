"""Approved Goal family presentation; immutable research identities stay separate.

Broad Star Barrage is retired as a board entry, not erased from tracking.
Every matching historical rule remains available in Inspect and return capture.
"""
from collections import OrderedDict

import goal_moves_current as xg
import goal_star_moves as star
import goal_toi_moves as toi

VERSION = 'goal-families-2026-11-11-v1'
PRIMARY = {
    'xg:Base Attack:I:False': 'xg_base',
    "xg:Reaper's Return:I:False": 'xg_reaper',
    'star:drought_i': 'star_drought',
    'star:measured': 'measured',
    'star:controlled': 'controlled',
    'toi:I': 'runic',
}
BUFFS = {'power_surge': 'Power Surge', 'barrage': 'Barrage'}
GROUP_NAMES = {
    'xg_base': 'Base Attack · xG profiles & levels',
    'xg_reaper': "Reaper’s Return · xG route & levels",
    'star_drought': 'Drought Strike · star profiles & levels',
    'measured': 'Measured Strike', 'controlled': 'Controlled Crit',
    'runic': 'Runic Barrage · levels & retired base',
    'power_surge': 'Power Surge · exact parent records',
    'barrage': 'Barrage · exact parent records',
    'context_monitor': 'Other matchup records · monitoring',
}

def _annotate(move, identity):
    m = dict(move)
    primary = identity in PRIMARY
    buff = identity.split(':')[1] if identity.startswith('star:') and identity.count(':') == 2 else None
    if primary:
        family = PRIMARY[identity]
    elif buff in BUFFS:
        family = buff
    elif m.get('matchup_buff'):
        family = 'context_monitor'
    elif identity.startswith("xg:Reaper's Return:"):
        family = 'xg_reaper'
    elif identity.startswith('xg:'):
        family = 'xg_base'
    elif identity.startswith('toi:'):
        family = 'runic'
    else:
        family = 'star_drought'
    m.update(goal_record_key=identity, goal_family=family,
             goal_primary=primary, goal_visible_buff=buff in BUFFS,
             goal_retired=identity == 'toi:base', goal_display_version=VERSION)
    return m

def records_for(row):
    """All literally qualified existing records; never combine their samples."""
    found = []
    # The production scorer supplies all existing source/line/matrix guards.
    active_xg = xg.xg_goals_moves(row)
    if active_xg:
        score = xg.goal_profile_score(row)
        recent_xg = xg._number(row, 'Pregame_xG_L5_per_game')
        for spec in xg.RULES:
            if not xg._passes(row, spec[3]):
                continue
            for level in xg.LEVELS[spec[0]]:
                if level[0] is not None and score < level[0]:
                    continue
                for boosted in (False, True):
                    if boosted and recent_xg < .55:
                        continue
                    identity = f'xg:{spec[0]}:{level[1]}:{boosted}'
                    found.append(_annotate(xg._move(spec, level=level, xg_boost=boosted), identity))
        found.extend(_annotate(m, 'xg:defense') for m in active_xg if m.get('matchup_buff'))
    for m in star.star_goals_moves(row):
        found.append(_annotate(m, 'star:' + m['id'].split(':', 1)[1]))
    active_toi = toi.toi_goals_moves(row)
    if active_toi:
        found.append(_annotate(active_toi[0], 'toi:base'))
        drought = xg._number(row, 'Drought_PPP')
        for cutoff, label, record in toi.LEVELS:
            if drought is not None and drought >= cutoff:
                m = toi._make(f'ppp_{label.lower()}', f'Runic Barrage {label}',
                              'SPECIAL ATTACK', toi.ENTRY + f' + PPP drought >= {cutoff}', record, level=label)
                found.append(_annotate(m, 'toi:' + label))
    return found

def display_moves(moves, rank):
    """Six player families plus at most one exact record per approved buff type."""
    selected = OrderedDict()
    for m in sorted(moves, key=rank, reverse=True):
        if m.get('goal_primary') or m.get('goal_visible_buff'):
            selected.setdefault(m['goal_family'], m)
    # A context buff can never admit a card without a primary player attack.
    return list(selected.values()) if any(m.get('goal_primary') for m in selected.values()) else []

def inspect_groups(moves):
    groups = OrderedDict()
    for m in moves:
        key = m.get('goal_family', 'context_monitor')
        groups.setdefault(GROUP_NAMES[key], []).append(m)
    return groups

def summary(moves):
    """Tracker uses the same approved family selection and exact return ranking."""
    visible = display_moves(moves, move_rank)
    attacks = sorted((m for m in visible if m.get('goal_primary')), key=move_rank, reverse=True)
    best = attacks[0] if attacks else None
    def record(m): return f"{m['wins']}/{m['picks']} · {100*m['wins']/m['picks']:.1f}%"
    def entries(items): return '\n'.join(f"{m['name']} · {record(m)}" for m in items) or '—'
    return {'Goal_Attack': best['name'] if best else '—',
            'Goal_Attack_Record': record(best) if best else '—',
            'Goal_Attack_Sample_BE': best.get('sample_book_be_pct') if best else None,
            'Goal_All_Attacks': entries(attacks),
            'Goal_Crits': entries(m for m in attacks if m.get('crit')),
            'Goal_Buffs': entries(m for m in visible if m.get('goal_visible_buff'))}

def move_rank(move):
    """Identical to the board's established Goal return ranking."""
    from move_return_grades import grade_rank, stable_identity
    n = int(move['picks'])
    if move.get('matchup_buff'):
        return (-1, False, n, move['wins']/n if n else -1, 0, stable_identity(move))
    return grade_rank(move) or (0, False, move['wins']/n if n else -1, n,
                               move['later_wins']/move['later_picks'] if move['later_picks'] else -1,
                               stable_identity(move))

def card_moves(row):
    return display_moves(records_for(row), move_rank)
