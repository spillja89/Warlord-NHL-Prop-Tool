"""Presentation-only V1 Points1.5 families; never filters or ranks card eligibility."""
from html import escape

V1_KIT = 'points15-2026-27-iron-march-2026-11-06-v2'

# Route membership follows the approved rules, not empirical player overlap.
# Branches may be alternatives: a family label never creates a new AND gate.
FAMILIES = {}

def _add(family, branch, *ids):
    for ident in ids:
        assert ident not in FAMILIES
        FAMILIES[ident] = (family, branch)

_add('DPS Tank', 'Spellbreaker · PP scoring', 'env_0_0', 'env_0_1', 'env_0_2')
_add('DPS Tank', 'Blood Fury · Opp shooting', 'env_1_0', 'env_1_1')
_add('DPS Tank', 'Siege Charge · Opp scoring', 'env_2_0', 'env_2_1', 'env_2_2')
_add('DPS Tank', 'Last Bastion · Venue defense', 'env_3_0')
_add('DPS Tank', 'Iron Vengeance · Own PK', 'env_4_0', 'env_4_1')
_add('Backbone', 'Reawakening · Pace profile', 'points15_pace_gap_base_attack')
_add('Backbone', 'Siege Charge · Opp scoring', 'env_5_0', 'env_5_1')
_add('Backbone', 'Stonewall · Own recent defense', 'env_6_0', 'env_6_1')
_add('Backbone', 'Last Bastion · Venue defense', 'env_7_0')
_add('Backbone', 'Arcane / Power Siege · PP scoring', 'env_8_0', 'env_8_1',
     'env_9_0', 'env_9_1', 'gap_pp_display_0', 'gap_pp_display_1', 'gap_pp_display_2')
_add('Backbone', 'Blood Fury · Opp shooting', 'env_10_0', 'env_10_1', 'env_10_2')
_add('Titan’s Resolve', 'Talent / star profiles', 'points15_talent_pace_gap', 'points15_star_pace_gap')
_add('Titan’s Resolve', 'Siege Charge · Opp scoring', 'env_11_0', 'env_11_1')
_add('Titan’s Resolve', 'Stonewall · Own recent defense', 'env_12_0')
_add('Titan’s Resolve', 'Spellbreaker / Power Siege · PP scoring',
     'env_13_0', 'env_13_1', 'env_14_0', 'env_14_1')
_add('Titan’s Resolve', 'Blood Fury · Opp shooting', 'env_15_0', 'env_15_1')
_add('Titan’s Resolve', 'Siege Hammer · Scoring combo', 'tg_combo_K1')
_add('Titan’s Resolve', 'Iron / Bloodforged Fortress', 'tg_combo_K4', 'tg_combo_K3', 'tg_combo_J')
_add('Ragnarok', 'Warpath / Iron March', 'STAR_pk_time_scoring_broad',
     'STAR_pk_time_scoring_stronger', 'STAR_warpath_iron_march_1', 'STAR_warpath_iron_march_2')
_add('War Drum', 'Battle Rhythm · Recorded pace', 'points15_recorded_points_pace155')
_add('Echo Stomp', 'Rally · Recent points', 'points15_l5_points_ceiling100')
_add('Runebreaker / Ironhide', 'Guard · PP involvement profile', 'points15_pp_proxy_ceiling460')
_add('Runebreaker / Ironhide', 'Advance · 5v5 ice-time profile', 'points15_ev5_toi154')
_add('Runebreaker / Ironhide', 'Smash · Guard AND Advance', 'points15_pp_proxy_ev5_special')
_add('Gaia’s Blessing', 'Measured Pressure', 'gaia_measured_pressure_15')

PP_NOTES = {
    'env_8_0': 'PP condition: opponent PP conversion ≥19%.',
    'env_8_1': 'PP condition: opponent PP conversion ≥25%.',
    'env_9_0': 'PP condition: opponent PP goals/game ≥0.55.',
    'env_9_1': 'PP condition: opponent PP goals/game ≥0.75.',
    'gap_pp_display_0': 'PP condition: opponent PP goals/game ≥0.55 OR PP conversion ≥19%.',
    'gap_pp_display_1': 'PP condition: opponent PP goals/game ≥0.55 AND PP conversion ≥19%.',
    'gap_pp_display_2': 'PP condition: opponent PP goals/game ≥0.75 OR PP conversion ≥25%.',
}

def family_branch(move):
    """An exact kit guard leaves V2 additions, legacy and other props alone."""
    if (move.get('points15_current') and not move.get('points15_v2')
            and move.get('kit_version') == V1_KIT):
        return FAMILIES.get(move.get('id'))
    return None

def inspect_group(move):
    route = family_branch(move)
    if route:
        return 'V1 · ' + route[0]
    if move.get('points15_current') and not move.get('points15_v2'):
        return 'V1 · Other preserved routes'
    return None

def condition_note(move):
    text = PP_NOTES.get(move.get('id')) if family_branch(move) else None
    return ('<div class="wn-move-rule"><b>'+escape(text)+'</b></div>') if text else ''

def render_family(group, entries, rank):
    """Every input record appears once; only the grouping is new."""
    ordered = sorted(entries, key=lambda item: rank(item[0]), reverse=True)
    branches = {}
    for move, html in ordered:
        route = family_branch(move)
        if route is None or group != 'V1 · ' + route[0]:
            return None  # Caller retains its existing generic renderer.
        branches.setdefault(route[1], []).append((move, html))
    if not ordered:
        return None
    lead = ordered[0][0]
    family = family_branch(lead)[0]
    def short_name(move):
        prefix = family + ' · '
        name = move['name']
        return name[len(prefix):] if name.startswith(prefix) else name
    parts = []
    if len(branches) == 1:
        branch_entries = next(iter(branches.values()))
        parts.append(branch_entries[0][1])
        if len(branch_entries) > 1:
            rest = ''.join(html for _, html in branch_entries[1:])
            parts.append('<details class="wn-details"><summary>Other matched levels &amp; conditions ('
                         +str(len(branch_entries)-1)+')</summary>'+rest+'</details>')
    else:
        for branch, items in branches.items():
            branch_lead, lead_html = items[0]
            rest = ''.join(html for _, html in items[1:])
            more = ('<details class="wn-details"><summary>Other matched levels &amp; conditions ('
                    +str(len(items)-1)+')</summary>'+rest+'</details>') if rest else ''
            parts.append('<details class="wn-details wn-points15-branch"><summary>'
                         +escape(branch)+' · '+escape(short_name(branch_lead))
                         +' ('+str(len(items))+(' record' if len(items)==1 else ' records')
                         +')</summary>'+lead_html+more+'</details>')
    return ('<details class="wn-goal-record-group wn-points15-family"><summary>'
            +escape(group)+' · '+escape(short_name(lead))+' · '+str(len(branches))
            +(' branch' if len(branches)==1 else ' branches')+' · '+str(len(ordered))
            +(' record' if len(ordered)==1 else ' records')+'</summary>'+''.join(parts)+'</details>')
