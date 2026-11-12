"""Read-only class board built from the current tracker and frozen move rules."""

from __future__ import annotations

import math
import re
import base64
from functools import lru_cache
from html import escape
from pathlib import Path

import pandas as pd

from warlord_moves_2026 import fired_moves
from sog_moves_current import is_current_sog25, sog_base_eligible, pk_duration_seconds
from sog_profiles_2026 import profile_values as sog_profile_values, display_moves as sog_profile_display
from sog_profiles_2026 import inspect_moves as sog_profile_inspect
from player_availability import is_unavailable
from player_form import summarize_form
from goal_routes_v2 import profile_values as goal_v2_values
from points15_research import context as points15_research_context
from assist_routes_2026 import assist_display_tier, assist_display_name
from points_moves_current import best_move as current_points_best, display_moves as current_points_display, prepare_row as prepare_points_row, points_moves as current_points_moves
from pp_assist_beta import assist_ppp_beta, beta_summary, beta_details, BETA_CSS
from goal_return_monitor import return_panel as goal_return_panel, RETURN_CSS as GOAL_RETURN_CSS
from goal_card_routes import records_for as goal_card_records, display_moves as goal_family_display, inspect_groups as goal_inspect_groups, inspect_records_for as goal_inspect_records
from goal_card_routes import parent_attack_name as goal_buff_parent_name, buffs_for_attack as goal_buffs_for_attack
from goal_card_routes import best_move as goal_best_move, move_rank as goal_move_rank, player_route_family as goal_player_route_family
from assist_return_monitor import return_panel as assist_return_panel
from points_return_monitor import return_panel as points_return_panel, original_caption as points_original_caption
from special_return_monitor import sog_return_panel
from move_return_grades import grade_rank, move_grade, grade_badges, stable_identity, GRADE_CSS


CLASSES = (
    ("Carry", "Goal", "⚔️", "#ef6a65"),
    ("Support", "Assists", "🪄", "#b692ff"),
    ("Tank", "Points", "🛡️", "#79afff"),
    ("Jungle", "SOG", "🌿", "#77d6a6"),
)


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _value(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None and not pd.isna(value) and str(value).strip():
            return value
    return None


_GOAL_V2_LABELS = {'Pregame_xG_L5_per_game': 'L5 individual xG/game', 'Player_Log_L5_TOI_per_game': 'L5 actual ice time min/game', 'Player_Log_L10_Shooting_pct': 'L10 shooting %', 'Player_Log_L20_Goal_fraction_pct': 'L20 goal share of points %', 'Player_Rebuilt_Gap_G10': 'Saved10-game goal-pace gap', 'L20_A': 'Assists L20', 'PP_SOG60': 'Individual PP SOG/60', 'i5v5_xG60': 'Individual5v5 xG/60', 'NHL_Opp_PP_L5_O_PG': 'Opp L5 earned PPO/game', 'NHL_Opp_PP_Season_SecondsPerOpportunity': 'Opp season PP seconds/opportunity', 'NHL_Team_PP_Season_G60': 'Team actual season PP goals/60', 'NHL_Opp_PP_L10_Time_PG': 'Opp L10 PP time/game', 'NHL_Team_PK_L10_SecondsPerOpportunity': 'Team L10 PK seconds/opportunity', 'NHL_Team_SA_Season_PG': 'Team season SOG allowed/game', 'NHL_Opp_SA_L5_PG': 'Opp L5 SOG allowed/game'}

_CARD_LABELS = {
    'Conf_Points': 'Points profile confidence', 'Conf_Goal': 'Goal profile confidence',
    'Log_L10_S_per_game': 'L10 average SOG', 'Log_L20_P_per_game': 'Recorded points/game',
    'Log_L5_P_per_game': 'L5 points/game', 'Pace_Gap_P10': 'Point-pace gap',
    'StarScore': 'Star score', 'TalentMult': 'Talent multiplier', 'TOI_per_game': 'Ice time/game',
    'TOI_Pct': 'Ice-time rank', 'PP_TOI_per_game': 'Player PP min/game',
    'EV5_TOI_per_game': '5v5 ice time/game', 'Assists_mu': 'Saved assist μ',
    'Points_mu': 'Saved points μ', 'Goal_mu': 'Saved goal μ',
    'iXA%': 'Observed assist rank', 'iXG%': 'Individual xG rank',
    'ShotIntent': 'Shot intent', 'ShotIntent_Pct': 'Shot-intent rank',
    'Avg5_SOG': 'L5 average SOG', 'L10_S': 'Shots L10', 'L10_G': 'Goals L10',
    'Drought_G': 'Goal drought', 'Drought_PPP': 'PPP drought',
    'Log_L20_Point2plus_pct': 'Recorded 2+ point frequency %',
    'Log_L20_PPP_hit_pct': 'Recorded PPP frequency %',
    'PP_Points60': 'PP involvement proxy/60', 'PPP10_total': 'PPP L10',
    'team_5v5_SF60': 'Team 5v5 shots/60',
    'NHL_Opp_PK_Time_PG': 'Opp PK min/game', 'NHL_Opp_GF_Season_PG': 'Opp goals scored/game',
    'NHL_Opp_PP_Success_pct': 'Opp PP conversion %', 'NHL_Team_PP_Time_PG': 'Team PP min/game',
    'NHL_Opp_PP_Goals_PG': 'Opp PP goals/game', 'NHL_Opp_PP_G60': 'Opp PP goals/60',
    'NHL_Opp_PKO_PG': 'Opp PK opportunities/game', 'NHL_Team_PP_G60': 'Team PP goals/60',
    'NHL_Team_PP_Goals_PG': 'Team PP goals/game', 'NHL_Opp_GA_10_PG': 'Opp L10 goals allowed/game',
    'NHL_Opp_GA_Venue_PG': 'Opp venue goals allowed/game',
    'NHL_Opp_Goals_per_shot_allowed_pct': 'Opp shots converted against %',
    'NHL_Opp_Goal_conversion_pct': 'Opp shooting %',
    'Opp_PK_xGA60': 'Opp PK xGA/60', 'Opp_PK_GA60': 'Opp PK goals allowed/60',
    'opp_5v5_HDCA60': 'Opp 5v5 HDCA/60', 'opp_5v5_xGA60': 'Opp 5v5 xGA/60',
    'Opp_5v5_xGA_relative_league': 'Opp 5v5 xGA / league',
    'Opp_PK_xGA_relative_league': 'Opp PK xGA / league',
    'Team_PP_xGF_relative_league': 'Team PP xGF / league',
    'Team_PP_GF60': 'Team PP actual GF/60', 'Team_PP_xGF60': 'Team PP xGF/60',
    'PP_PPO_Blend_PG': 'PP opportunity blend/game', 'PP_Time_Blend_PG': 'PP time blend/game',
    'Team_PP_OZ_Time_Pct': 'Team PP offensive-zone time %', 'Team_EV_OZ_Rank': 'Team EV zone-time rank',
    'NHL_Team_PK_goal_drought': 'Team PK goal-allowed drought',
    'NHL_Opp_GF_10_PG': 'Opp L10 goals scored/game', 'NHL_Opp_PK_G60': 'Opp PK goals allowed/60',
    'NHL_Opp_SA_Venue_PG': 'Opp venue shots allowed/game', 'NHL_Team_GA_5_PG': 'Team L5 goals allowed/game',
    'NHL_Team_GA_Venue_PG': 'Team venue goals allowed/game', 'NHL_Team_PK_G60': 'Team PK goals allowed/60',
    'Opp_DefWeak': 'Opponent defensive weakness', 'PP_Matchup_Equal': 'Equal-weight PP context',
    'Player_5v5_SOG_Share': '5v5 team shot share %', 'Opp_SOG_Against_L10': 'Opp L10 SOG allowed/game',
}


def _role_card_context(row):
    from power_play_quotes import _unit_label, team_pp_time_share_pct
    values = {field: _number(row.get(field)) for field in _CARD_LABELS}
    values['unit'] = _unit_label(row)
    values['team_pp_share'] = team_pp_time_share_pct(row.get('PP_TOI_per_game'), row.get('Team_PP_Time_PG'))
    return values


def _card_usage_panel(values):
    def fmt(key, suffix=''):
        value = _number(values.get(key))
        return f'{value:.1f}{suffix}' if value is not None else 'Unavailable'
    return ('<div class="wn-assist-fact"><small>ICE TIME &amp; ROLE</small>'
            f'Ice time <b>{fmt("TOI_per_game", " min/g")}</b> · PP <b>{fmt("PP_TOI_per_game", " min/g")}</b><br>'
            f'Recorded unit <b>{_h(values.get("unit") or "Unit unknown")}</b> · '
            f'Team PP time <b>{fmt("team_pp_share", "%")}</b></div>')


def _fact_line(field, value, op=None, cutoff=None, *, goal_v2=False):
    label = (_GOAL_V2_LABELS if goal_v2 else {}).get(field, _CARD_LABELS.get(field, field.replace('_', ' ')))
    value = _number(value)
    if value is None:
        actual = 'Unavailable'
    elif field.endswith('Time_PG'):
        seconds = round(value * 60)
        actual = f'{seconds // 60}:{seconds % 60:02d}'
    else:
        digits = 0 if field in {'Drought_G', 'Drought_PPP', 'L10_S', 'L10_G', 'PPP10_total'} else 2
        actual = f'{value:.{digits}f}'
    target = ''
    if op is not None and cutoff is not None:
        sign = '≥' if op == '>=' else '≤'
        if field.endswith('Time_PG'):
            seconds = round(cutoff * 60)
            limit = f'{seconds // 60}:{seconds % 60:02d}'
        else:
            limit = f'{cutoff:g}'
        target = f' <span>({sign}{limit})</span>'
    return f'<span title="{_h(label + chr(32) + actual)}">{_h(label)} <b>{_h(actual)}</b>{target}</span>'


def _render_goal_facts(card, move):
    values = {**(card.get('role_context') or {}), **(card.get('goal_profile') or {})}
    rule = str((move or {}).get('rule', ''))
    checks = {}
    for field in _CARD_LABELS:
        found = re.search(r'(?<![\w])' + re.escape(field) + r'\s*(>=|<=)\s*([0-9.]+)', rule)
        if found:
            checks[field] = (found[1], float(found[2]))
    aliases = {'TOI_per_game': 'TOI/game', 'L10_S': 'L10 shots', 'Drought_PPP': 'PPP drought',
               'Drought_G': 'Goal drought', 'Avg5_SOG': 'L5 average SOG',
               'ShotIntent_Pct': 'Shot-intent percentile', 'ShotIntent': 'Shot intent'}
    for field, alias in aliases.items():
        found = re.search(re.escape(alias) + r'\s*(>=|<=)\s*([0-9.]+)', rule, re.IGNORECASE)
        if found:
            checks[field] = (found[1], float(found[2]))
    profile_fields = ['iXG%', 'StarScore', 'TOI_per_game', 'ShotIntent', 'ShotIntent_Pct', 'Avg5_SOG', 'L10_S', 'L10_G', 'Drought_G', 'Drought_PPP', 'Goal_mu', 'Points_mu']
    if (move or {}).get('goal_v2'):
        for field,op,cut in move.get('all_checks',[]):checks[field]=(op,cut)
        checks.update(Pregame_xG_L5_per_game=('>=',.52),Player_Log_L5_TOI_per_game=('>=',18.8))
        for buff in goal_buffs_for_attack(card.get('moves',[]),move):
            if buff.get('goal_visible_buff'):
                for field,op,cut in buff.get('checks',[]):
                    if field.startswith(('NHL_','Team_')):checks[field]=(op,cut)
        profile_fields=['Pregame_xG_L5_per_game','Player_Log_L5_TOI_per_game']+[f for f in checks if not f.startswith(('NHL_','Team_')) and f not in {'Pregame_xG_L5_per_game','Player_Log_L5_TOI_per_game'}]
    profile = '<br>'.join(_fact_line(field, values.get(field), *checks.get(field, (None, None)), goal_v2=bool((move or {}).get('goal_v2')))
                          for field in profile_fields if values.get(field) is not None)
    environment = '<br>'.join(_fact_line(field, values.get(field), op, cut, goal_v2=bool((move or {}).get('goal_v2')))
                              for field, (op, cut) in checks.items()
                              if field.startswith(('NHL_', 'Opp_', 'opp_', 'Team_', 'team_')))
    environment = environment or 'Move-specific matchup requirements are listed in the full fired move list.'
    sample = _number((move or {}).get('sample_book_be_pct'))
    sample_note = f'<div class="wn-assist-sample">MOVE SAMPLE AVG BOOK BE <b>{sample:.1f}%</b></div>' if sample is not None else ''
    return (sample_note + '<div class="wn-assist-facts">' + _card_usage_panel(values) +
            f'<div class="wn-assist-fact"><small>GOAL PROFILE · ACTUAL / CUTOFF</small>{profile}</div>' +
            f'<div class="wn-assist-fact"><small>MATCHUP · ACTUAL / CUTOFF</small>{environment}</div></div>')


def _class_identity(role, card, move):
    """Stable cosmetic identities, separate from eligibility and move records."""
    if role == 'Support':
        variant = _support_variant(move)
        animal = 'Orangutan' if variant in {'Runesmith', 'Oracle + Runesmith'} else 'Chimp' if variant == 'Stormcaller' else 'Gorilla'
        return animal, variant
    if role == 'Tank':
        if card.get('line') == 1.5:
            return 'Panda', 'War Drummer'
        return ('Orangutan', 'Siegebreaker') if (move or {}).get('kind') in {'HEAVY', 'SPECIAL', 'CRIT'} else ('Gorilla', 'Bulwark')
    if role == 'Jungle':
        variant=(move or {}).get('sog_variant')
        if variant in {'Berserker','Ranger','Warden'}:
            return {'Berserker':'Gorilla','Ranger':'Orangutan','Warden':'Panda'}[variant],variant
        if (move or {}).get('empowered'):
            return 'Orangutan', 'Ranger'
        if (move or {}).get('matchup_buff'):
            return 'Panda', 'Warden'
        return 'Gorilla', 'Berserker'
    if (move or {}).get("goal_v2"):
        return {"A":("Gorilla","Ravager"),"F":("Chimp","Reaper"),"P":("Panda","Executioner"),"X":("Chimp","Reaper"),"G":("Gorilla","Ravager"),"S":("Panda","Executioner")}.get(goal_player_route_family(move),("Gorilla","Ravager"))
    text = str((move or {}).get('name', ''))
    if (move or {}).get('crit') or any(word in text for word in ('Reaper', 'Measured', 'Creator', 'Controlled', 'Patient')):
        return 'Chimp', 'Reaper'
    if any(word in text for word in ('Star ', 'Star’s', 'Star Shot', 'Runic', 'Barrage')):
        return 'Panda', 'Executioner'
    return 'Gorilla', 'Ravager'


def _class_roster_art(role):
    if role == 'Support':
        return _support_roster_art()
    variants = {'Tank': [('Bulwark', 'Gorilla Bulwark'), ('Siegebreaker', 'Orangutan Siegebreaker'), ('War Drummer', 'Panda War Drummer')],
                'Jungle': [('Berserker', 'Gorilla Berserker'), ('Ranger', 'Orangutan Ranger'), ('Warden', 'Panda Warden')],
                'Carry': [('Ravager', 'Gorilla Ravager'), ('Reaper', 'Chimp Reaper'), ('Executioner', 'Panda Executioner')]}
    portraits = [f'<img src="{uri}" alt="{_h(label)}" />' for variant, label in variants.get(role, [])
                 if (uri := _character_uri(role, variant))]
    return '<div class="wn-support-roster wn-class-roster">' + ''.join(portraits) + '</div>'


def _assist_card_context(row):
    from power_play_quotes import _unit_label, team_pp_time_share_pct
    fields = ('TOI_per_game', 'PP_TOI_per_game', 'Assists_mu', 'iXA%',
              'Goal_mu', 'i5v5_primaryAssists60', 'PPP10_total', 'L10_S',
              'Player_SOG_Share_Proxy', 'Player_5v5_SOG_Share',
              'Team_PP_GF60', 'Team_PP_xGF60', 'opp_5v5_HDCA60', 'Opp_PK_xGA60')
    result = {field: _number(row.get(field)) for field in fields}
    result['unit'] = _unit_label(row)
    result['team_pp_share'] = team_pp_time_share_pct(row.get('PP_TOI_per_game'), row.get('Team_PP_Time_PG'))
    return result


def _render_assist_facts(card, move):
    values = card.get('assist_context') or {}
    def metric(field, digits=2, suffix=''):
        value = _number(values.get(field))
        return f'{value:.{digits}f}{suffix}' if value is not None else '—'
    usage = (f'TOI <b>{metric("TOI_per_game", 1)} min/g</b> · '
             f'PP <b>{metric("PP_TOI_per_game", 1)} min/g</b><br>'
             f'Recorded unit <b>{_h(values.get("unit") or "Unit unknown")}</b> · '
             f'Team PP time <b>{metric("team_pp_share", 1, "%")}</b>')
    profile = (f'Assist μ <b>{metric("Assists_mu")}</b> · '
               f'Observed assist rank <b>{metric("iXA%", 0)}</b>')
    rule = (move or {}).get('rule', '')
    checks = []
    for field, label in [('Goal_mu','Goal μ'), ('i5v5_primaryAssists60','5v5 primary A/60'),
                         ('PPP10_total','PPP L10'), ('L10_S','Shots L10'),
                         ('Player_SOG_Share_Proxy','Shot-share proxy'),
                         ('Player_5v5_SOG_Share','5v5 shot share')]:
        if field in rule:
            checks.append(f'{label} <b>{metric(field)}</b>')
    if checks:
        profile += '<br>' + ' · '.join(checks)
    environment=[]
    for field,label in [('Team_PP_GF60','Team PP actual GF/60'),
                        ('Team_PP_xGF60','Team PP xGF/60'),
                        ('opp_5v5_HDCA60','Opp 5v5 HDCA/60')]:
        match=re.search(re.escape(field)+r'\s*(>=|<=)\s*([0-9.]+)',rule)
        if match:
            op='≥' if match[1]=='>=' else '≤'
            environment.append(f'{label} <b>{metric(field)}</b> <span>({op}{match[2]})</span>')
    sample_be=_number((move or {}).get('sample_book_be_pct'))
    sample=(f'<div class="wn-assist-sample">MOVE SAMPLE AVG BOOK BE <b>{sample_be:.1f}%</b></div>'
            if sample_be is not None else '')
    env=('<div class="wn-assist-fact"><small>MATCHUP · VALUE / CUTOFF</small>'
         + '<br>'.join(environment) + '</div>') if environment else ''
    return (f'{sample}<div class="wn-assist-facts">'
            f'<div class="wn-assist-fact"><small>ICE TIME &amp; ROLE</small>{usage}</div>'
            f'<div class="wn-assist-fact"><small>PLAYER PROFILE · SAVED MODEL</small>{profile}</div>{env}</div>')


def _render_points_facts(card, move):
    move = move or {}
    values = {**(card.get('role_context') or {}), **(card.get('points_context') or {})}
    profile = [f'Saved points μ <b>{_number(values.get("Points_mu")):.2f}</b>'
               if _number(values.get('Points_mu')) is not None else 'Saved points μ <b>Unavailable</b>']
    environment = []
    for field, op, cut in move.get('conditions', []):
        line = _fact_line(field, values.get(field), op, cut)
        target = environment if field.startswith(('NHL_', 'Opp_', 'opp_', 'Team_', 'team_')) else profile
        target.append(line)
    alternatives = []
    for group in move.get('any_conditions', []):
        alternatives.append('<br>'.join(_fact_line(field, values.get(field), op, cut) for field, op, cut in group))
    if alternatives:
        environment.append('<b>ANY MATCHING PATH</b><br>' + '<br><b>OR</b><br>'.join(alternatives))
    sample = _number(move.get('sample_book_be_pct'))
    sample_note = (f'<div class="wn-assist-sample">MOVE SAMPLE AVG BOOK BE <b>{sample:.1f}%</b> · '
                   f'{_h(move.get("sample_from", ""))} through {_h(move.get("sample_through", ""))}</div>') if sample is not None else ''
    matchup = '<br>'.join(environment) or 'No additional environment gate for this move.'
    return (sample_note + '<div class="wn-assist-facts">' + _card_usage_panel(values) +
            '<div class="wn-assist-fact"><small>POINTS PROFILE · ACTUAL / CUTOFF</small>' + '<br>'.join(profile) + '</div>' +
            '<div class="wn-assist-fact"><small>MATCHUP · ACTUAL / CUTOFF</small>' + matchup + '</div></div>')


def _render_sog_facts(card, move):
    if move and move.get('sog_profile'):
        values=card.get('sog_context') or {}
        labels={'L20_S':'Shots · up to20 games','Drought_G':'Goal drought (games)',
                'Player_Log_L5_TOI_per_game':'L5 average TOI (minutes)',
                'Player_Log_L10_S_per_game':'L10 average SOG',
                'Player_Log_L20_S_hit_pct':'3+ SOG game frequency % · up to20',
                'Player_Log_L20_G_hit_pct':'Goal-game frequency % · up to20',
                'SOG_Opportunity_Base':'Saved shot opportunity',
                'NHL_Opp_PK_Goal_drought':'Opponent PP-goal allowed drought',
                'NHL_Team_PP_L5_O_PG':'Own L5 PPO/game',
                'NHL_Opp_PK_L5_Success_pct':'Opponent L5 PK success %'}
        player=[];environment=[]
        for field,op,cut in move.get('conditions',[]):
            v=_number(values.get(field));actual='Unavailable' if v is None else f'{v:.2f}'
            fact=f'{_h(labels.get(field,field))} <b>{actual}</b> ({_h(op)} {cut:g})'
            (environment if field.startswith('NHL_') else player).append(fact)
        return ('<div class="wn-assist-facts">'+_card_usage_panel(card.get('role_context') or {})+
                '<div class="wn-assist-fact"><small>SHOT ROUTE · ACTUAL / CUTOFF</small>'+'<br>'.join(player or ['No extra player gate.'])+'</div>'+
                '<div class="wn-assist-fact"><small>MATCHUP · ACTUAL / CUTOFF</small>'+'<br>'.join(environment or ['No additional environment gate.'])+'</div></div>')
    if not move or not move.get("sog_current"):
        values = card.get("sog_context") or {}
        profile = '<br>'.join(_fact_line(field, values.get(field)) for field in ('TOI_Pct', 'Player_5v5_SOG_Share', 'iXG%'))
        matchup = _fact_line('Opp_SOG_Against_L10', values.get('Opp_SOG_Against_L10'))
        return ('<div class="wn-assist-facts">' + _card_usage_panel(card.get('role_context') or {}) +
                f'<div class="wn-assist-fact"><small>SHOT PROFILE</small>{profile}</div>' +
                f'<div class="wn-assist-fact"><small>MATCHUP</small>{matchup}</div></div>')
    values = card.get("sog_context") or {}
    def metric(field, digits=1):
        value = _number(values.get(field))
        return f"{value:.{digits}f}" if value is not None else "—"
    conf = _number(card.get("confidence"))
    conf_label = f"{conf:.0f}" if conf is not None else "—"
    seconds = _number(values.get("pk_seconds"))
    duration = f"{seconds:.2f}s" if seconds is not None else "Unavailable"
    buff = any(item.get("name")=='Paralysis' for item in card.get("moves", []))
    state = "PARALYSIS ACTIVE" if buff else "PK duration unavailable" if seconds is None else "Outside PK buff"
    sample = _number(move.get("sample_book_be_pct"))
    sample_note = f'<div class="wn-assist-sample">MOVE SAMPLE AVG BOOK BE <b>{sample:.1f}%</b></div>' if sample is not None else ""
    buff_move = next((item for item in card.get("moves", []) if item.get("name")=='Paralysis'), None)
    buff_note = ""
    if buff_move and buff_move.get("picks"):
        bw, bn = int(buff_move["wins"]), int(buff_move["picks"])
        buff_note = f'<div class="wn-crit-env">+ PARALYSIS · opponent PK ≤104s/opportunity · {bw}/{bn} · {100*bw/bn:.1f}%</div>'
    route = str(values.get("green_route") or "")
    route_note = f'<br>Green: {escape(route)}' if route else ""
    if route == "Shot profile · iXG60":
        route_note += f' · iXG <b>{metric("iXG%")}</b> ≥60'
    return (sample_note + '<div class="wn-assist-facts">'
      f'<div class="wn-assist-fact"><small>SHOT PROFILE V2</small>Confidence <b>{conf_label}</b> · base ≥67 · stronger ≥71<br>TOI <b>{metric("TOI_per_game")} min/g</b> · rank {metric("TOI_Pct")}{route_note}</div>'
      f'<div class="wn-assist-fact"><small>SHOT SHARE &amp; MATCHUP</small>5v5 team shot share <b>{metric("Player_5v5_SOG_Share")}%</b><br>Opponent L10 SOG allowed <b>{metric("Opp_SOG_Against_L10")}/g</b></div>'
      f'<div class="wn-assist-fact"><small>PK DURATION · VALUE / CUTOFF</small>Opponent <b>{duration}</b> · cutoff ≤104s/opportunity<br>{state}</div>'
      '</div>' + buff_note)


def _line_price(row, market):
    prefix = {"Goal": ("Goal", "ATG"), "Points": ("Points",),
              "Assists": ("Assists",), "SOG": ("SOG",)}[market]
    for item in prefix:
        line = _number(row.get(f"{item}_Line"))
        if line is not None:
            odds = _number(row.get(f"{item}_Odds_Over"))
            book = _value(row, f"{item}_Book")
            if odds is None and market == "Goal":
                atg_line = _number(row.get("ATG_Line"))
                if atg_line is not None and math.isclose(atg_line, line):
                    odds = _number(row.get("ATG_Odds_Over"))
                    book = book or _value(row, "ATG_Book")
            if odds is None:
                continue
            return line, odds, book
    return None, None, None


def _price_comparison(row, market, line, odds):
    """Model Over probability and break-even rate for this exact posted line."""
    if line is None or odds is None or odds == 0:
        return None, None, None
    prefixes = ("Goal", "ATG") if market == "Goal" else (market,)
    model_prob = None
    for prefix in prefixes:
        quoted_line = _number(row.get(f"{prefix}_Line"))
        if quoted_line is None or not math.isclose(quoted_line, line):
            continue
        raw = _number(row.get(f"{prefix}_p_model_over"))
        if raw is None:
            pct = _number(row.get(f"{prefix}_Model%"))
            raw = pct / 100 if pct is not None else None
        if raw is not None and 0 < raw < 1:
            model_prob = raw
            break
    book_prob = 100 / (100 + odds) if odds > 0 else abs(odds) / (100 + abs(odds))
    return model_prob, book_prob, (model_prob - book_prob) if model_prob is not None else None


def _move_rank(move):
    if move.get('goal_v2'):
        return goal_move_rank(move)
    wins, picks = int(move["wins"]), int(move["picks"])
    later_wins, later_picks = int(move["later_wins"]), int(move["later_picks"])
    if move.get("matchup_buff"):
        return (-1, False, picks, wins / picks if picks else -1, 0, stable_identity(move))
    performance = grade_rank(move)
    if performance is not None:
        return performance
    if move.get("market") == "Assists":
        tier = assist_display_tier(move)["priority"]
        return (0, False, tier, wins / picks if picks else -1, picks, stable_identity(move))
    return (0, False, wins / picks if picks else -1, picks,
            later_wins / later_picks if later_picks else -1, stable_identity(move))


def _baseline_rule(row, market, line):
    matrix = str(row.get(f"Matrix_{market}", "")).strip().casefold()
    if matrix not in {"green", "🟢"}:
        return None
    if market == "Goal" and line == 0.5:
        return "Green + Goal 0.5"
    if market == "Assists" and line == 0.5:
        return "Green + Assists 0.5"
    if market == "Points" and line in (0.5, 1.5):
        threshold = 77
        if _number(row.get("Conf_Points")) is not None and _number(row.get("Conf_Points")) >= threshold:
            return f"Green + Points {line:g} + Conf_Points ≥ {threshold}"
    if market == "SOG" and line == 2.5 and is_current_sog25(row):
        return "Green + SOG2.5 + Shot Profile v2 confidence ≥67" if sog_base_eligible(row) else None
    if market == "SOG" and line in (2.5, 3.5):
        threshold = 75 if line == 2.5 else 86
        move_conf = _number(row.get("Conf_SOG_Move", row.get("Conf_SOG")))
        if move_conf is not None and move_conf >= threshold:
            label = "Original SOG confidence" if "Conf_SOG_Move" in row else "Conf_SOG"
            return f"Green + SOG {line:g} + {label} ≥ {threshold}"
    return None


def _display_moves(moves, market, line):
    """Keep Points 1.5 cards focused without changing saved move records."""
    if market == 'Goal' and line == .5:
        return goal_family_display(moves, _move_rank)
    if market=='SOG' and line==2.5:
        return sog_profile_display(moves,_move_rank)
    if market == "Points" and line in (0.5, 1.5):
        current = current_points_display(moves)
        families = {move['family'] for move in current}
        selected = {}
        for move in sorted(moves, key=_move_rank, reverse=True):
            if move.get('points_current') and not move.get('suppress_badge') and move.get('family') in families:
                selected.setdefault(move['family'], move)
        return list(selected.values()) or ([move for move in moves if move["name"] in {"DPS Tank stance", "Backbone"}] if line == 1.5 else [])
    return moves


def _inspect_moves(moves, market, line):
    """Preserve every matched current definition behind Inspect, including nested steps."""
    if market=='SOG' and line==2.5:
        return sog_profile_inspect(moves,_move_rank)
    if market == 'Points':
        current = [move for move in moves if move.get('points_current')]
        moves = current or _display_moves(moves, market, line)
    return sorted(moves, key=_move_rank, reverse=True)


def points_current_tier(row, line, baseline):
    """Strongest current Points move; no additional hand-built tier policy."""
    original = current_points_best(row) if baseline and line in (0.5, 1.5) else None
    if original is None:
        return None
    moves = [move for move in _display_moves(current_points_moves(row), 'Points', line)
             if move['kind'] != 'CONTEXT']
    return max(moves, key=_move_rank, default=original)


def rank_warlords(frame: pd.DataFrame) -> dict[str, list[dict]]:
    """Best fired historical move per player and class on this filtered slate."""
    boards = {role: {} for role, *_ in CLASSES}
    if frame.empty:
        return {role: [] for role in boards}
    for row in frame.to_dict("records"):
        if is_unavailable(row):
            continue
        if str(row.get("Roster_Watch", "")).strip().casefold() in {"true", "1"}:
            continue
        if not _value(row, "Player"):
            continue
        active = fired_moves(row)
        active['Goal'] = goal_card_records(row)
        for role, market, _, _ in CLASSES:
            moves = active[market]
            line, odds, book = _line_price(row, market)
            moves = _display_moves(moves, market, line)
            # A model signal without a posted price is not a ready board pick.
            if line is None or odds is None or odds == 0:
                continue
            model_prob, book_prob, price_gap = _price_comparison(row, market, line, odds)
            baseline_rule = _baseline_rule(row, market, line)
            if not moves and not baseline_rule:
                continue
            baseline_only = not moves
            if baseline_only:
                move = {"name": "Green Baseline", "kind": "BASELINE", "rule": baseline_rule,
                        "wins": 0, "picks": 0, "later_wins": 0, "later_picks": 0}
            else:
                attacks = [move for move in moves if move["kind"] != "CONTEXT" and not move.get("matchup_buff")]
                move = goal_best_move(moves) if market == 'Goal' else max(attacks or moves, key=_move_rank)
            player = str(row["Player"]).strip()
            team = str(_value(row, "Team") or "").strip()
            key = (team.casefold(), player.casefold())
            candidate = {
                "player": player, "team": team,
                "talent_tier": str(_value(row, "Talent_Tier") or "NONE").strip().upper(),
                "position": str(_value(row, "Pos") or "").strip(),
                "opponent": str(_value(row, "Opp") or "").strip(),
                "game": str(_value(row, "Game") or "").strip(),
                "time": str(_value(row, "Time") or _value(row, "StartTimeLocal") or "").strip(),
                "market": market, "line": line, "odds": odds, "book": book,
                "form_log": _value(row, "Form_Log"),
                "form_season": _value(row, "Form_Season"),
                "model_prob": model_prob, "book_prob": book_prob, "price_gap": price_gap,
                "tonight_goal_xg": _number(row.get("Tonight_Goal_xG")) if market == "Goal" else None,
                "tonight_goal_pct": _number(row.get("Tonight_Goal_1plus_pct")) if market == "Goal" else None,
                "tonight_goal_status": str(_value(row, "Tonight_Goal_Status") or "") if market == "Goal" else "",
                "goal_profile": {**goal_v2_values(row), **{field: _number(row.get(field)) for field in
                                 ("iXG%", "StarScore", "Drought_G", "Drought_PPP", "TOI_Pct", "ShotIntent", "ShotIntent_Pct", "Avg5_SOG", "L10_G", "L10_S", "TOI_per_game", "Goal_mu", "Points_mu")}}
                                if market == "Goal" else None,
                "tonight_mu": _number(row.get(f"Tonight_{market}_mu")) if
                row.get(f"Tonight_{market}_Version") == "2026-10-19-research-v3" else None,
                "tonight_over_pct": _number(row.get(f"Tonight_{market}_over_pct")) if
                row.get(f"Tonight_{market}_Version") == "2026-10-19-research-v3" else None,
                "tonight_status": str(_value(row, f"Tonight_{market}_Status") or ""),
                "team_pp_xgf60": _number(row.get("Team_PP_xGF60")) if market == "Assists" else None,
                "assist_context": _assist_card_context(row) if market == "Assists" else None,
                "role_context": _role_card_context(row),
                "sog_context": {**sog_profile_values(row), **{key:_number(row.get(key)) for key in
                    ("TOI_Pct", "TOI_per_game", "Player_5v5_SOG_Share", "Opp_SOG_Against_L10", "iXG%")},
                    "green_route":str(row.get("Matrix_SOG_Route", "") or ""),
                    "pk_seconds":pk_duration_seconds(row)} if market == "SOG" else None,
                "goalie": str(_value(row, "Opp_Goalie") or ""),
                "goalie_gp": _number(_value(row, "Opp_GP")),
                "goalie_status": str(_value(row, "Opp_Goalie_Status") or "Unknown"),
                "lineup_status": str(_value(row, "Lineup_Status") or "Lineup pending"),
                "baseline_only": baseline_only,
                "inspect_moves": _inspect_moves(goal_inspect_records(row) if market == "Goal" else active[market], market, line),
                "move": move, "moves": sorted(moves, key=lambda item: (
                    item["kind"] != "STANCE", _move_rank(item)), reverse=True),
                "move_count": len(moves),
            }
            candidate["pp_assist_beta"] = assist_ppp_beta(row) if market == "Assists" else None
            previous = boards[role].get(key)
            if previous is None or _move_rank(move) > _move_rank(previous["move"]):
                boards[role][key] = candidate
    return {role: sorted(players.values(),
                         key=lambda card: (_move_rank(card["move"]), card["player"]),
                         reverse=True)
            for role, players in boards.items()}


def rank_priced_slate(frame: pd.DataFrame) -> dict[str, list[dict]]:
    """Each player's displayed priced prop, ordered by current model confidence.

    Historical moves and baseline rules annotate cards; neither admits nor
    excludes a player from this board.
    """
    from points15_moves_current import quote_rows
    frame = quote_rows(frame)
    boards = {role: {} for role, *_ in CLASSES}
    if frame.empty:
        return {role: [] for role in boards}
    for row in frame.to_dict("records"):
        if is_unavailable(row):
            continue
        player = str(_value(row, "Player") or "").strip()
        if not player:
            continue
        team = str(_value(row, "Team") or "").strip()
        watch = str(row.get("Roster_Watch", "")).strip().casefold() in {"true", "1"}
        active = {market: [] for _, market, *_ in CLASSES} if watch else fired_moves(row)
        if not watch:
            active['Goal'] = goal_card_records(row)
        for role, market, _, _ in CLASSES:
            if row.get('_points15_alternate') is True and market != 'Points':
                continue
            line, odds, book = _line_price(row, market)
            if line is None or odds is None or odds == 0:
                continue
            model_prob, book_prob, price_gap = _price_comparison(row, market, line, odds)
            tracked_moves = active[market]
            moves = _display_moves(tracked_moves, market, line)
            attacks = [move for move in moves if move["kind"] != "CONTEXT" and not move.get("matchup_buff")]
            best = (goal_best_move(moves) if market == 'Goal' else max(attacks or moves, key=_move_rank)) if moves else None
            baseline = None if watch else _baseline_rule(row, market, line)
            current_points_tier = points_current_tier(row, line, baseline) if market == "Points" else None
            confidence = _number(row.get(f"Conf_{market}"))
            matrix = str(_value(row, f"Matrix_{market}") or "Unknown").strip()
            move_block_reason = ("No current-season Goal attack fired"
                                 if market == "Goal" and baseline and not moves else None)
            move_confidence = (_number(row.get("Conf_Points_Move"))
                               if market == "Points" else None)
            if market == "Points" and baseline and not moves and move_confidence is not None:
                move_threshold = 80 if line == 0.5 else 75
                if move_confidence < move_threshold:
                    move_block_reason = f"Frozen move score below {move_threshold}"
            candidate = {
                "player": player, "team": team,
                "talent_tier": str(_value(row, "Talent_Tier") or "NONE").strip().upper(),
                "position": str(_value(row, "Pos") or "").strip(),
                "opponent": str(_value(row, "Opp") or "").strip(),
                "game": str(_value(row, "Game") or "").strip(),
                "time": str(_value(row, "Time") or _value(row, "StartTimeLocal") or "").strip(),
                "market": market, "line": line, "odds": odds, "book": book,
                "form_log": _value(row, "Form_Log"),
                "form_season": _value(row, "Form_Season"),
                "model_prob": model_prob, "book_prob": book_prob, "price_gap": price_gap,
                "tonight_goal_xg": _number(row.get("Tonight_Goal_xG")) if market == "Goal" else None,
                "tonight_goal_pct": _number(row.get("Tonight_Goal_1plus_pct")) if market == "Goal" else None,
                "tonight_goal_status": str(_value(row, "Tonight_Goal_Status") or "") if market == "Goal" else "",
                "goal_profile": {**goal_v2_values(row), **{field: _number(row.get(field)) for field in
                                 ("iXG%", "StarScore", "Drought_G", "Drought_PPP", "TOI_Pct", "ShotIntent", "ShotIntent_Pct", "Avg5_SOG", "L10_G", "L10_S", "TOI_per_game", "Goal_mu", "Points_mu")}}
                                if market == "Goal" else None,
                "tonight_mu": _number(row.get(f"Tonight_{market}_mu")) if
                row.get(f"Tonight_{market}_Version") == "2026-10-19-research-v3" else None,
                "tonight_over_pct": _number(row.get(f"Tonight_{market}_over_pct")) if
                row.get(f"Tonight_{market}_Version") == "2026-10-19-research-v3" else None,
                "tonight_status": str(_value(row, f"Tonight_{market}_Status") or ""),
                "team_pp_xgf60": _number(row.get("Team_PP_xGF60")) if market == "Assists" else None,
                "assist_context": _assist_card_context(row) if market == "Assists" else None,
                "role_context": _role_card_context(row),
                "sog_context": {**sog_profile_values(row), **{key:_number(row.get(key)) for key in
                    ("TOI_Pct", "TOI_per_game", "Player_5v5_SOG_Share", "Opp_SOG_Against_L10", "iXG%")},
                    "green_route":str(row.get("Matrix_SOG_Route", "") or ""),
                    "pk_seconds":pk_duration_seconds(row)} if market == "SOG" else None,
                "confidence": confidence, "move_confidence": move_confidence,
                "matrix": matrix,
                "baseline_rule": baseline, "baseline_only": bool(baseline and not best),
                "points_current_tier": current_points_tier,
                "points_context": prepare_points_row(row) if market == "Points" else None,
                "points15_research": points15_research_context(row) if market == "Points" and line == 1.5 else None,
                "move_block_reason": move_block_reason,
                "goalie": str(_value(row, "Opp_Goalie") or ""),
                "goalie_status": str(_value(row, "Opp_Goalie_Status") or "Unknown"),
                "lineup_status": str(_value(row, "Lineup_Status") or "Lineup pending"),
                "move": best,
                "inspect_moves": _inspect_moves(goal_inspect_records(row) if market == "Goal" else tracked_moves, market, line),
                "moves": sorted(moves, key=lambda item: (
                    item["kind"] != "STANCE", _move_rank(item)), reverse=True),
                "move_count": len(moves),
                "points_power_tier": market == "Points" and line == 1.5 and any(
                    item["name"] == "Blade Impale (Power Tier)" for item in tracked_moves),
                "roster_watch": watch,
            }
            candidate["pp_assist_beta"] = assist_ppp_beta(row) if market == "Assists" and not watch else None
            key = (team.casefold(), player.casefold(), line) if market == 'Points' else (team.casefold(), player.casefold())
            previous = boards[role].get(key)
            if previous is None or (
                confidence is not None, confidence or -1,
                matrix.casefold() == "green", odds is not None
            ) > (
                previous["confidence"] is not None,
                previous["confidence"] or -1,
                previous["matrix"].casefold() == "green",
                previous["odds"] is not None,
            ):
                boards[role][key] = candidate
    return {role: sorted(players.values(), key=lambda card: (
        card["confidence"] is not None, card["confidence"] or -1,
        card["matrix"].casefold() == "green", card["player"].casefold()
    ), reverse=True) for role, players in boards.items()}


def featured_warlords(boards: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Character cards ranked by their strongest qualifying fired move.

    The complete priced slate stays in ``boards`` for the searchable tables.
    Historical rates select a display tier, not a claim of future probability.
    """
    featured = {}
    for role, cards in boards.items():
        selected = []
        for card in cards:
            if not card.get("baseline_rule"):
                continue
            if role == "Tank" and card.get("points_current_tier"):
                selected.append({**card, "move": card["points_current_tier"]})
                continue
            qualifying = [move for move in card["moves"] if move["kind"] != "CONTEXT"
                          and not move.get("matchup_buff")
                          and int(move["picks"]) > 0
                          and int(move["wins"]) / int(move["picks"]) >= 0.5]
            if not qualifying:
                continue
            selected.append({**card, "move": max(qualifying, key=_move_rank)})
        featured[role] = sorted(selected, key=lambda card: (
            _move_rank(card["move"]), card.get("confidence") or -1,
            card["player"].casefold()), reverse=True)
    return featured


def unfeatured_baselines(cards: list[dict]) -> list[dict]:
    """Priced Green candidates without a qualifying historical move."""
    return [card for card in cards if card.get("baseline_rule") and not any(
        move["kind"] != "CONTEXT" and int(move["picks"]) > 0
        and int(move["wins"]) / int(move["picks"]) >= 0.5
        for move in card["moves"])]


def goal_display_cards(cards: list[dict]) -> list[dict]:
    """Only fired Goal attacks get cards; scouting rows stay in the research table."""
    eligible = []
    for card in cards:
        if not card.get("baseline_rule"):
            continue
        attacks = [move for move in card.get("moves", [])
                   if not move.get("matchup_buff")
                   if int(move["picks"]) > 0]
        if not attacks:
            continue
        eligible.append({**card, "move": goal_best_move(card.get('moves',[])),
                         "baseline_only": False})
    return sorted(eligible, key=lambda card: (
        card["move"] is not None,
        _move_rank(card["move"]) if card["move"] else (-1, -1, -1),
        card.get("confidence") or -1, card["player"].casefold()), reverse=True)


def baseline_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Show each priced baseline and where the other posted lines drop out.

    These are the current class entry rules. A baseline can qualify without a
    stronger named move, so this audit does not change the frozen move kit or grading.
    """
    specs = (
        ("Goals", "Goal", (0.5,), "Matrix_Goal", "Conf_Goal"),
        ("Assists", "Assists", (0.5,), "Matrix_Assists", None),
        ("Points", "Points", (0.5, 1.5), "Matrix_Points", "Conf_Points"),
        ("Shots", "SOG", (2.5, 3.5), "Matrix_SOG", "Conf_SOG"),
    )
    stages = ("Priced", "Supported line", "Green", "Baseline", "Named move")
    totals: dict[str, dict[str, set]] = {}
    teams: dict[tuple[str, str], dict[str, set]] = {}
    baseline_rows: list[dict] = []
    for label, market, lines, matrix_col, conf_col in specs:
        totals[label] = {stage: set() for stage in stages}
        for row in frame.to_dict("records"):
            if is_unavailable(row):
                continue
            player = str(_value(row, "Player") or "").strip()
            if not player:
                continue
            team = str(_value(row, "Team") or "").strip()
            identity = (team.casefold(), player.casefold())
            bucket = teams.setdefault((label, team), {stage: set() for stage in stages})
            line, odds, book = _line_price(row, market)
            if line is None or odds is None or odds == 0:
                continue
            for counts in (totals[label], bucket):
                counts["Priced"].add(identity)
            if str(row.get("Roster_Watch", "")).strip().casefold() in {"true", "1"}:
                continue
            if line not in lines:
                continue
            for counts in (totals[label], bucket):
                counts["Supported line"].add(identity)
            if str(row.get(matrix_col, "")).strip().casefold() not in {"green", "🟢"}:
                continue
            for counts in (totals[label], bucket):
                counts["Green"].add(identity)
            conf = _number(row.get(conf_col)) if conf_col else None
            if not _baseline_rule(row, market, line):
                continue
            for counts in (totals[label], bucket):
                counts["Baseline"].add(identity)
            named_moves = (goal_family_display(goal_card_records(row), _move_rank)
                           if market == 'Goal' else fired_moves(row)[market])
            named = any(move["kind"] not in {"STANCE", "CONTEXT"} and not move.get('matchup_buff')
                        for move in named_moves) if market == 'Goal' else any(
                            move['kind'] not in {'STANCE', 'CONTEXT'} for move in named_moves)
            if named:
                for counts in (totals[label], bucket):
                    counts["Named move"].add(identity)
            stats_source = str(_value(row, "Opp_Goalie_Source") or "none")
            stats_label = {
                "dailyfaceoff_name_and_stats": "Matched starter stats",
                "dailyfaceoff_name_only": "Starter named; stats unavailable",
                "moneypuck_team_proxy_unconfirmed": "Team goalie proxy",
                "moneypuck_team_proxy": "Team goalie proxy",
                "none": "Unavailable",
            }.get(stats_source, stats_source)
            baseline_rows.append({
                "Prop": label, "Player": player, "Team": team,
                "Game": str(_value(row, "Game") or ""),
                "Line": line, "Odds": int(odds), "Book": str(book or ""),
                "Confidence": int(conf) if conf is not None else "—",
                "Status": "Named move" if named else "Baseline only",
                "Opp goalie": str(_value(row, "Opp_Goalie") or ""),
                "Goalie status": str(_value(row, "Opp_Goalie_Status") or "Unknown"),
                "Goalie stats": stats_label,
            })
    summary = pd.DataFrame([{"Prop": label, **{stage: len(totals[label][stage]) for stage in stages}}
                            for label, *_ in specs])
    by_team = pd.DataFrame([{"Prop": label, "Team": team,
                             **{stage: len(bucket[stage]) for stage in stages}}
                            for (label, team), bucket in sorted(teams.items())])
    roster = pd.DataFrame(baseline_rows, columns=(
        "Prop", "Player", "Team", "Game", "Line", "Odds", "Book", "Confidence", "Status",
        "Opp goalie", "Goalie status", "Goalie stats"
    )).drop_duplicates(["Prop", "Player", "Team"])
    return summary, by_team, roster


def _h(value):
    return escape(str(value), quote=True)


def _odds(value):
    if value is None:
        return "Odds pending"
    return f"{value:+.0f}" if value > 0 else f"{value:.0f}"


def _form_html(card, *, expanded=False):
    form = summarize_form(card.get("form_log"), card["market"], card["line"])
    if not form:
        season = _h(card.get("form_season") or "Current")
        return f'<div class="wn-form-empty">{season} regular-season form unavailable right now.</div>'
    if form.get("empty"):
        return f'<div class="wn-form-empty">{_h(form["season"])} regular season · awaiting completed games.</div>'
    l10_w, l10_n = form["l10"]
    l5_w, l5_n = form["l5"]
    season_w, season_n = form["season_rate"]
    avg, med = form["average"], form["median"]
    line = card["line"]
    tiles = (
        (f"LAST {l10_n}", f"{l10_w}/{l10_n} · {100*l10_w/l10_n:.0f}%"),
        ("AVG / MEDIAN", f"{avg:.1f} / {med:.1f}"),
        ("VS TODAY'S LINE", f"Avg {avg-line:+.1f} · Med {med-line:+.1f}"),
        (f"LAST {l5_n}", f"{l5_w}/{l5_n} · {100*l5_w/l5_n:.0f}%"),
        ("SEASON", f"{season_w}/{season_n} · {100*season_w/season_n:.0f}%"),
    )
    tiles_html = "".join(f'<div><small>{_h(label)}</small><strong>{_h(value)}</strong></div>'
                         for label, value in tiles)
    splits = " · ".join(f"{label} {won}/{count} ({100*won/count:.0f}%)"
                        for label, (won, count) in form["splits"].items())
    opportunity = []
    if form["shots"] is not None:
        opportunity.append(f"L{l10_n} shots {form['shots']:.1f}/game")
    if form["toi"] is not None:
        opportunity.append(f"ice time {form['toi']:.1f} min/game")
    strip = "".join(f'<span class="{"hit" if hit else "miss"}" title="{_h(date)} · {value} {_h(card["market"])}">'
                    f'<b>{"✓" if hit else "×"} {value}</b><small>{_h(date[5:].replace("-", "/") if date else "—")}</small></span>'
                    for value, hit, date in reversed(form["recent"]))
    label = "Power-play points" if card["market"] == "PPP" else "Recent form"
    return (f'<details class="wn-form"{" open" if expanded else ""}><summary>{label} · {_h(form["season"])} regular season</summary>'
            f'<p>Against today\'s over {line:g} line · results read left to right, oldest to newest. '
            'Historical results are context, not the move record.</p>'
            f'<div class="wn-form-tiles">{tiles_html}</div>'
            f'<div class="wn-form-meta">{_h(splits)}</div>'
            f'<div class="wn-form-meta">{_h(" · ".join(opportunity))}</div>'
            f'<div class="wn-form-strip">{strip}</div></details>')


def _render_pp_player_signal_details(details):
    """Show actual values beside the four frozen player-signal cutoffs."""
    result = []
    for signal in details or []:
        passed = signal.get("passed")
        mark, state = ("✓", "hit") if passed is True else ("✕", "miss") if passed is False else ("—", "missing")
        value = _number(signal.get("value"))
        digits = int(signal.get("digits", 2))
        shown = f"{value:.{digits}f}" if value is not None else "Unavailable"
        direction = "≥" if signal.get("direction") == ">=" else "≤"
        cutoff = f'{float(signal["cutoff"]):g}'
        result.append(
            f'<div class="pp-player-check pp-player-check--{state}">'
            f'<b>{mark}</b> {_h(signal.get("label"))}: <strong>{_h(shown)}</strong>'
            f' <span>({direction}{cutoff})</span></div>')
    return '<div class="pp-player-checks">' + "".join(result) + '</div>' if result else ""


def render_power_play_quotes_table(rows, signals_by_team):
    """Full PP scouting table with clickable, typed column sorting."""
    def cell(value):
        if value is None or pd.isna(value):
            return "—"
        return _h(value)

    def decimal(value, digits=1):
        number = _number(value)
        return f"{number:.{digits}f}" if number is not None else "—"

    def sort_value(label, row, signals):
        key = next((field for title, field, _ in columns if title == label), None)
        if label == "PPP Assist beta":
            routes = (row.get("PP Assist beta") or {}).get("routes", [])
            return str(max(100 * route["hits"] / route["plays"] for route in routes)) if routes else ""
        if label == "Team PP matchup signals":
            return str(len(signals))
        if label == "Player signals":
            value = row.get("PP research count")
        elif label == "Best price":
            value = str(row.get("Best feed price") or "").split(" · ", 1)[0]
        elif label in ("PPP L5", "PPP L10", "PPP season"):
            value = str(row.get(key) or "")
            try:
                numerator, denominator = value.split(" · ", 1)[0].split("/", 1)
                return str(int(numerator) / int(denominator))
            except (ValueError, ZeroDivisionError):
                return ""
        elif label == "Opp PK time/g":
            value = str(row.get(key) or "")
            try:
                minutes, seconds = value.split(":", 1)
                return str(int(minutes) * 60 + int(seconds))
            except ValueError:
                return ""
        elif label == "Boost":
            return "1" if row.get(key) else "0"
        else:
            value = row.get(key)
        number = _number(value)
        if label not in ("Player", "Team", "Game", "PP unit", "Assists cross-check") and number is not None:
            return str(number)
        if value is None or pd.isna(value):
            return ""
        return str(value).casefold()

    columns = (
        ("Rank", "PP rank", cell),
        ("Player", "Player", cell),
        ("Team", "Team", cell),
        ("Game", "Game", cell),
        ("Best price", "Best feed price", cell),
        ("PP on-ice xG/g", "PP on-ice xG/game", lambda value: decimal(value, 3)),
        ("PPP Assist beta", "PP Assist beta", lambda value: ('<details class="pp-assist-beta-cell"><summary>' + beta_summary(value) + '</summary>' + beta_details(value) + '</details>') if value else "—"),
        ("Team PP matchup signals", None, None),
        ("Player signals", "PP research signals", cell),
        ("PP unit", "PP unit", cell),
        ("PP min/g", "Current PP min/game", decimal),
        ("Team PP time %", "Team PP time share %", decimal),
        ("Player PP iXG/g", "PP shooter xG/game", lambda value: decimal(value, 3)),
        ("PPP L5", "Current PPP L5", cell),
        ("PPP L10", "Current PPP L10", cell),
        ("PPP season", "Current PPP season", cell),
        ("PPP drought", "PPP drought (games)", cell),
        ("Tonight PPP mean", "Tonight PPP mean", lambda value: decimal(value, 3)),
        ("Tonight PPP 1+ %", "Tonight PPP 1+ %", decimal),
        ("Team PP xGF/60", "Team PP xGF/60 now", lambda value: decimal(value, 2)),
        ("Opp PK xGA/60", "Opp PK xGA/60 now", lambda value: decimal(value, 2)),
        ("Opp PK time/g", "Opp PK time/game", cell),
        ("Opp PK sec/opp", "Opp PK sec/opportunity", decimal),
        ("Boost", "PP boost", lambda value: "ON" if value else "—"),
        ("Assists cross-check", "Assists cross-check", cell),
    )

    body = []
    for row in rows:
        team = str(row.get("Team") or "").upper()
        signals = signals_by_team.get(team, [])
        stack = "".join(
            f'<div class="pp-signal pp-signal--move">{ "+ " if index else ""}{_h(signal)}</div>'
            for index, signal in enumerate(signals)
        ) or '<div class="pp-signal">—</div>'
        cells = []
        for label, key, render in columns:
            css = ' class="pp-player"' if label == "Player" else ' class="pp-signals"' if key is None else ""
            content = stack if key is None else render(row.get(key))
            if label == "Player signals" and row.get("PP research count"):
                content = f'<span class="pp-player-move">{content}</span>'
            if label == "Player signals":
                content += _render_pp_player_signal_details(row.get("PP player signal details"))
            cells.append(f'<td{css} data-sort="{_h(sort_value(label, row, signals))}">{content}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    headings = []
    for label, _, _ in columns:
        css = ' class="pp-player"' if label == "Player" else ""
        kind = "text" if label in ("Player", "Team", "Game", "PP unit", "Assists cross-check") else "number"
        headings.append(
            f'<th scope="col" aria-sort="none"{css}>'
            f'<button type="button" class="pp-sort" data-kind="{kind}">'
            f'{_h(label)} <span class="pp-sort-arrow" aria-hidden="true">↕</span></button></th>'
        )
    return BETA_CSS + '''<style>
      .pp-scout-wrap{max-height:650px;overflow:auto;border:1px solid #d7dce4;border-radius:8px;margin:8px 0 12px}
      .pp-scout{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%;font:12px Inter,system-ui,sans-serif;color:#1c2a3b}
      .pp-scout th,.pp-scout td{padding:7px 9px;border-bottom:1px solid #dce1e8;text-align:left;vertical-align:top;background:#fff}
      .pp-scout thead th{position:sticky;top:0;z-index:2;background:#eef2f7;white-space:nowrap;font-weight:700}
      .pp-scout .pp-player{position:sticky;left:0;z-index:1;background:#fff;font-weight:700;min-width:145px}
      .pp-scout thead .pp-player{z-index:3;background:#eef2f7}
      .pp-scout .pp-signals{min-width:260px;max-width:340px;white-space:normal;line-height:1.35}
      .pp-player-checks{display:grid;gap:4px;margin-top:7px;min-width:245px;white-space:normal;font-size:12px;line-height:1.4}
      .pp-player-check{padding:4px 7px;border-radius:4px;background:#152237;color:#f1f5fb;border-left:3px solid #7d8ba0}
      .pp-player-check--hit{border-color:#37c789}.pp-player-check--hit>b{color:#72efb2}
      .pp-player-check--miss{border-color:#d08b6a}.pp-player-check--miss>b{color:#ffc3a6}
      .pp-player-check--missing{color:#c4cddd}.pp-player-check span{color:#c5d1e3}
      .pp-scout .pp-signal+ .pp-signal{margin-top:3px}
      .pp-scout .pp-sort{border:0;background:none;padding:0;color:inherit;font:inherit;font-weight:700;cursor:pointer;text-align:left;white-space:nowrap}
      .pp-scout .pp-sort:hover,.pp-scout .pp-sort:focus-visible{color:#186598;text-decoration:underline}
      .pp-scout .pp-sort-arrow{color:#668399;margin-left:2px}
      .pp-scout .pp-signal--move{padding:3px 5px;border-left:3px solid #22a9d7;border-radius:3px;background:#e9f7fd;font-weight:800;color:#075f87;animation:ppMovePulse 2.4s ease-in-out infinite}
      .pp-scout .pp-player-move{display:inline-block;padding:3px 5px;border-radius:4px;background:#f3e9ff;font-weight:800;color:#7030a1;animation:ppMovePulse 2.4s ease-in-out infinite}
      @keyframes ppMovePulse{0%,100%{filter:brightness(1);box-shadow:0 0 0 transparent}50%{filter:brightness(1.25);box-shadow:0 0 9px currentColor}}
      @media(prefers-reduced-motion:reduce){.pp-scout .pp-signal--move,.pp-scout .pp-player-move{animation:none}}
      .pp-scout tbody tr:nth-child(even) td{background:#f8fafc}
      .pp-scout tbody tr:nth-child(even) .pp-player{background:#f8fafc}
    </style><div class="pp-scout-root"><div class="pp-scout-wrap"><table class="pp-scout"><thead><tr>''' + "".join(headings) + '</tr></thead><tbody>' + "".join(body) + '''</tbody></table></div>
    <script>
    (() => {
      const root = document.currentScript.parentElement;
      const table = root.querySelector('.pp-scout');
      const tbody = table.tBodies[0];
      const headers = [...table.tHead.rows[0].cells];
      headers.forEach((header, column) => {
        header.querySelector('button').addEventListener('click', () => {
          const ascending = header.getAttribute('aria-sort') !== 'ascending';
          const kind = header.querySelector('button').dataset.kind;
          const rows = [...tbody.rows];
          rows.sort((a, b) => {
            const left = a.cells[column].dataset.sort;
            const right = b.cells[column].dataset.sort;
            if (!left) return right ? 1 : 0;
            if (!right) return -1;
            const comparison = kind === 'number'
              ? Number(left) - Number(right)
              : left.localeCompare(right, undefined, {numeric: true});
            return (ascending ? comparison : -comparison) ||
              Number(a.cells[0].dataset.sort) - Number(b.cells[0].dataset.sort);
          });
          tbody.append(...rows);
          headers.forEach(item => {
            item.setAttribute('aria-sort', 'none');
            item.querySelector('.pp-sort-arrow').textContent = '↕';
          });
          header.setAttribute('aria-sort', ascending ? 'ascending' : 'descending');
          header.querySelector('.pp-sort-arrow').textContent = ascending ? '↑' : '↓';
        });
      });
    })();
    </script></div>'''


def render_power_play_form(card):
    """Show the same dated player form card on the separate PPP scouting page."""
    portrait = _character_uri("Support")
    image = f'<img src="{portrait}" alt="" />' if portrait else "🪄"
    price = _odds(card.get("odds"))
    form = _form_html({**card, "market": "PPP", "line": 0.5}, expanded=True)
    def metric(value, suffix=""):
        try:
            number = float(value)
            if math.isfinite(number):
                return f"{number:.1f}{suffix}"
        except (TypeError, ValueError):
            pass
        return "—"

    def precise_metric(value):
        try:
            number = float(value)
            if math.isfinite(number):
                return f"{number:.3f}"
        except (TypeError, ValueError):
            pass
        return "—"

    def rank(value):
        try:
            number = float(value)
            return f"#{int(number)}" if math.isfinite(number) else "—"
        except (TypeError, ValueError):
            return "—"

    tiles = (
        ("Player PP role", str(card.get("pp_unit") or "Unconfirmed")),
        ("PP on-ice xG/game", precise_metric(card.get("pp_onice_xg_game"))),
        ("Player signals", str(card.get("pp_research") or "Data missing")),
        ("Player PP iXG/game", precise_metric(card.get("pp_ixg_game"))),
        ("PPP L10 observed", card.get("ppp_l10") or "—"),
        ("PP-point drought", f'{int(card["ppp_drought"])} game' +
         ('' if int(card["ppp_drought"]) == 1 else 's')
         if _number(card.get("ppp_drought")) is not None else "—"),
        ("Opp PK time/game", str(card.get("opp_pk_time_pg") or "—")),
        ("Team PP xGF/60", f'{metric(card.get("team_pp_xgf60"))} · {rank(card.get("team_pp_rank"))} attack'),
        ("Opp PK xGA/60", f'{metric(card.get("opp_pk_xga60"))} · {rank(card.get("opp_pk_rank"))} allowed'),
        ("Opp PK GA/60", metric(card.get("opp_pk_ga60"))),
    )
    matchup = "".join(f'<div class="ppp-matchup-tile"><small>{_h(label)}</small><strong>{_h(value)}</strong></div>'
                      for label, value in tiles)
    research_signals = card.get("team_research_signals") or []
    research_banner = (
        '<div class="ppp-research"><small>ALL MATCHING TEAM PP-GOAL SIGNALS · 2026–27 THROUGH OCT 29</small>'
        + "".join(f'<strong class="ppp-research-item">{ "+ " if index else ""}{_h(signal)}</strong>'
                  for index, signal in enumerate(research_signals))
        + '<small>Overlapping team-game records; not this player’s PPP chance.</small></div>'
        if research_signals else
        '<div class="ppp-research"><small>No saved team PP-goal matchup signal for this opponent.</small></div>'
    )
    return f'''<style>
      .ppp-card,.ppp-card *{{box-sizing:border-box}}
      .ppp-card{{font-family:Inter,system-ui,sans-serif;color:#edf1f8;background:#1b2739;border:1px solid #4d416e;border-left:3px solid #b692ff;border-radius:10px;padding:14px;margin:12px 0}}
      .ppp-head{{display:flex;align-items:center;gap:12px;margin-bottom:8px}}
      .ppp-portrait{{width:50px;height:50px;flex:none;overflow:hidden;border-radius:9px;border:1px solid #8d72c4;background:#172039}}
      .ppp-portrait img{{width:100%;height:100%;object-fit:cover;object-position:center top}}
      .ppp-title{{min-width:0;flex:1}}.ppp-title strong{{display:block;font-size:17px;color:#fff}}.ppp-title span{{font-size:11px;color:#b9c9dc}}
      .ppp-price{{text-align:right;white-space:nowrap;color:#e9d5ff;font-size:20px;font-weight:900}}.ppp-price small{{display:block;color:#aebbd0;font-size:10px;font-weight:600}}
      .ppp-research{{display:flex;flex-wrap:wrap;align-items:center;gap:6px 12px;margin:8px 0;padding:8px 10px;border-radius:7px;border:1px solid #5884a1;background:#123044;color:#d1f1ff}}
      .ppp-research strong{{font-size:12px;letter-spacing:.04em}}.ppp-research small{{font-size:9px;opacity:.85;font-weight:700}}
      .ppp-research-item{{display:block;width:100%;padding:3px 0;border-top:1px solid #ffffff2b}}
      .ppp-research--buff{{border-color:#a87de4;background:#33244c;color:#f4e4ff}}
      .ppp-research--watch{{border-color:#e1ae4f;background:#4d3617;color:#fff0c6}}
      .ppp-matchup{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:6px;margin:10px 0}}
      .ppp-matchup-tile{{padding:8px;background:#111b2e;border:1px solid #ffffff26;border-radius:6px}}
      .ppp-matchup-tile small{{display:block;color:#afbed6;font-size:9px;text-transform:uppercase;letter-spacing:.06em}}
      .ppp-matchup-tile strong{{display:block;color:#fff;font-size:13px;margin-top:4px}}
      .ppp-player-details{{margin:10px 0;padding:10px;background:#111b2e;border:1px solid #ffffff26;border-radius:6px}}
      .ppp-player-details>small{{color:#c6d4e9;font-size:10px;font-weight:800}}
      .ppp-player-details .pp-player-checks{{display:grid;gap:5px;margin-top:7px;font-size:13px;line-height:1.4}}
      .ppp-player-details .pp-player-check{{padding:5px 8px;color:#f1f5fb;background:#19283d;border-left:3px solid #7d8ba0;border-radius:4px}}
      .ppp-player-details .pp-player-check--hit{{border-color:#37c789}}.ppp-player-details .pp-player-check--hit>b{{color:#72efb2}}
      .ppp-player-details .pp-player-check--miss{{border-color:#d08b6a}}.ppp-player-details .pp-player-check--miss>b{{color:#ffc3a6}}
      .ppp-player-details .pp-player-check span{{color:#c5d1e3}}
      .ppp-context-note{{color:#aebbd0;font-size:10px;margin:2px 0 9px}}
      .wn-form,.wn-form-empty{{color:#b9c9dc;border-top:1px solid #ffffff24;padding-top:8px;font-size:12px}}.wn-form summary{{cursor:pointer;color:#e4c892;font-weight:800}}.wn-form p{{margin:7px 0;color:#aebbd0}}
      .wn-form-tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:5px;margin:8px 0}}.wn-form-tiles>div{{border:1px solid #ffffff1d;border-radius:5px;padding:7px;background:#0b1629a8}}.wn-form-tiles small{{display:block;color:#98adc6;font-size:9px}}.wn-form-tiles strong{{display:block;color:#fff;font-size:12px;margin-top:3px}}
      .wn-form-meta{{margin:4px 0}}.wn-form-strip{{display:flex;flex-wrap:wrap;gap:5px;margin:8px 0}}.wn-form-strip span{{border-radius:4px;padding:5px 7px;text-align:center;min-width:42px}}.wn-form-strip span b,.wn-form-strip span small{{display:block}}.wn-form-strip span small{{font-size:9px;opacity:.85}}.wn-form-strip .hit{{background:#1d6145;color:#c4ffde}}.wn-form-strip .miss{{background:#603036;color:#ffd7d7}}
    </style><section class="ppp-card"><div class="ppp-head"><div class="ppp-portrait">{image}</div>
      <div class="ppp-title"><strong>{_h(card.get("player") or "Player")}</strong><span>{_h(card.get("game") or "")} · Over 0.5 power-play points · {_h(card.get("book") or "Book pending")}</span></div>
      <div class="ppp-price">{_h(price)}<small>BEST FEED PRICE</small></div></div>
      {research_banner}
      <div class="ppp-matchup">{matchup}</div>
      <div class="ppp-player-details"><small>PLAYER SIGNAL CHECKS · BASE: PP ON-ICE xG/GAME ≥0.55</small>{_render_pp_player_signal_details(card.get("pp_player_signal_details"))}</div>
      <div class="ppp-context-note">Research order {rank(card.get("pp_rank"))} · PP1 is inferred from recorded usage · the team matchup flag applies to every teammate · player signals are past research, not tonight's probability</div>
      {form}</section>'''


@lru_cache(maxsize=16)
def _character_uri(role: str, variant: str = "") -> str:
    """Inline small artwork so class cards render behind Cloud's app proxy."""
    image = Path(__file__).parent / "static" / "characters" / f"{role.lower()}-gorilla-v2.webp"
    if role == "Support":
        candidate = image.parent / "support-gorilla-v3.webp"
        if candidate.is_file():
            image = candidate
    if role == "Support" and variant in {"Runesmith", "Oracle + Runesmith", "Stormcaller"}:
        filename = ("support-chimp-stormcaller-v1.webp" if variant == "Stormcaller"
                    else "support-orangutan-runesmith-v1.webp")
        candidate = image.parent / filename
        if candidate.is_file():
            image = candidate
    variants = {
        ('Tank', 'Siegebreaker'): 'tank-orangutan-hammer-v1.webp',
        ('Tank', 'War Drummer'): 'tank-panda-drum-v1.webp',
        ('Jungle', 'Ranger'): 'jungle-orangutan-ranger-v1.webp',
        ('Jungle', 'Warden'): 'jungle-panda-warden-v1.webp',
        ('Carry', 'Reaper'): 'carry-chimp-reaper-v1.webp',
        ('Carry', 'Executioner'): 'carry-panda-executioner-v1.webp',
    }
    if (role, variant) in variants:
        candidate = image.parent / variants[(role, variant)]
        if candidate.is_file():
            image = candidate
    if not image.is_file():
        return ""
    return "data:image/webp;base64," + base64.b64encode(image.read_bytes()).decode("ascii")


def _support_variant(move):
    """Cosmetic route identity only; no effect on move eligibility."""
    if not move:
        return "Arcanist"
    if move.get("route_variant"):
        return move["route_variant"]
    name = move.get("name", "")
    if name.startswith("Runic Infusion"):
        return "Runesmith"
    if name.startswith("Rune Conduit"):
        return "Stormcaller"
    return "Arcanist"


def _support_roster_art():
    """The three approved Support characters, for Support class banners only."""
    portraits = []
    for variant, label in [('Arcanist', 'Gorilla Arcanist'),
                           ('Runesmith', 'Pearl-eyed orangutan Runesmith'),
                           ('Stormcaller', 'Lightning-crowned chimp Stormcaller')]:
        uri = _character_uri('Support', variant)
        if uri:
            portraits.append(f'<img src="{uri}" alt="{_h(label)}" />')
    return '<div class="wn-support-roster">' + ''.join(portraits) + '</div>'


def _card_quick_summary(role, card):
    values = card.get('role_context') or {}
    if role == 'Carry':
        values = {**values, **(card.get('goal_profile') or {})}
        fields = [('Pregame_xG_L5_per_game', 'L5 xG/g', 2), ('Player_Log_L5_TOI_per_game', 'L5 TOI', 1), ('Player_Log_L10_Shooting_pct', 'L10 shooting %', 1)] if (card.get('move') or {}).get('goal_v2') else [('iXG%', 'iXG rank', 0), ('ShotIntent', 'Shot intent', 2), ('Goal_mu', 'Goal μ', 2)]
    elif role == 'Support':
        values = {**values, **(card.get('assist_context') or {})}
        fields = [('iXA%', 'Assist rank', 0), ('Assists_mu', 'Assist μ', 2), ('PP_TOI_per_game', 'PP min/g', 1)]
    elif role == 'Tank':
        values = {**values, **(card.get('points_context') or {})}
        fields = [('Points_mu', 'Points μ', 2), ('Assists_mu', 'Assist μ', 2), ('StarScore', 'Star score', 1)]
    else:
        values = {**values, **(card.get('sog_context') or {})}
        fields = [('TOI_per_game', 'TOI min/g', 1), ('Player_5v5_SOG_Share', '5v5 shot share %', 1), ('Opp_SOG_Against_L10', 'Opp SOG/g', 1)]
    pieces = [f'{_h(label)} <b>{value:.{digits}f}</b>' for field, label, digits in fields
              if (value := _number(values.get(field))) is not None]
    return '<div class="wn-quick-summary">' + ' · '.join(pieces) + '</div>' if pieces else ''


def render_warlords(boards: dict[str, list[dict]], limit: int = 5, icon_loader=None,
                    roles: tuple[str, ...] | None = None, show_hero: bool = True) -> str:
    """Compact, escaped HTML for the raid board or selected class lanes."""
    descriptions = {"Carry": "Finish the fight", "Support": "Set the play",
                    "Tank": "Hold the line", "Jungle": "Control the lanes"}
    selected = tuple(item for item in CLASSES if roles is None or item[0] in roles)
    lanes = []
    for role, market, symbol, color in selected:
        cards = boards.get(role, [])
        character_uri = _character_uri(role)
        class_icon = symbol
        units = []
        for rank, card in enumerate(cards[:limit], 1):
            move = card.get("move")
            points_tier = card.get("points_current_tier") if role == "Tank" else None
            tier_badge = ""
            tier_class = ""
            support_tier = assist_display_tier(move) if role == "Support" and move else None
            animal, character_variant = _class_identity(role, card, move)
            character_uri = _character_uri(role, character_variant)
            if move and ((role == "Support" and support_tier["crit"]) or
                         (role == "Carry" and move.get("crit"))):
                tier_label = support_tier["label"] if support_tier else "CRIT"
                tier_badge = f'<span class="wn-tier wn-tier--s">{tier_label}</span>'
                tier_class = " wn-unit--crit"
                if support_tier:
                    tier_class += " wn-support-crit-ii" if support_tier["label"] == "CRIT II" else " wn-support-crit-i"
            if role == "Jungle" and move and move.get("sog_current"):
                tier_label = "EMPOWERED" if move.get("empowered") else move["kind"]
                tier_badge = f'<span class="wn-tier wn-tier--s">{tier_label}</span>'
                if move.get("empowered"):
                    tier_class = " wn-unit--crit"
            if points_tier:
                tier_style = {"SPECIAL": "s", "HEAVY": "a", "BASE": "b"}.get(points_tier["kind"], "a")
                tier_badge = f'<span class="wn-tier wn-tier--{tier_style}">{points_tier["kind"]}</span>'
                tier_class = f" wn-unit--{tier_style}"
            performance = move_grade(move)
            if performance:
                tier_badge = grade_badges(move)
                tier_class = ' wn-unit--return-' + performance['tier'].lower()
                if performance['watch']:
                    tier_class += ' wn-unit--return-watch'
            elif move and move.get('picks'):
                tier_badge = grade_badges(move) + tier_badge
            baseline_only = bool(card.get("baseline_only") and not points_tier)
            wins, picks = (int(move["wins"]), int(move["picks"])) if move else (0, 0)
            late_wins, late_picks = ((int(move["later_wins"]), int(move["later_picks"]))
                                     if move else (0, 0))
            pct = 100 * wins / picks if picks else 0
            late_pct = 100 * late_wins / late_picks if late_picks else 0
            status = (("NO GOAL ATTACK" if role == "Carry" else
                       "ELITE GOAL WATCH" if card.get("goal_candidate") and card.get("talent_tier") == "ELITE" else
                       "STAR GOAL WATCH" if card.get("goal_candidate") and card.get("talent_tier") == "STAR" else
                       "BASELINE" if "confidence" in card else "BASELINE ONLY") if baseline_only
                      else "OPEN POOL" if not move else
                      "GOAL SIGNAL" if card.get("goal_candidate") and picks and wins / picks < .5 else
                      "TRACK" if move.get("track") else "LAB" if move.get("experimental")
                      or move["kind"] == "LAB CRIT" else move["kind"])
            if points_tier:
                status = f'{points_tier["kind"]} · 2026–27 MOVE'
            if support_tier:
                status = support_tier["label"]
            if role == "Jungle" and move and move.get("empowered"):
                status = "EMPOWERED"
            if role == 'Carry' and move and move.get('goal_v2') and move.get('matchup_buff'):
                status = 'BUFFED ATTACK'
            sample = " · SMALL SAMPLE" if 0 < picks < 30 else ""
            line = f"OVER {card['line']:g} {market.upper()}" if card["line"] is not None else market.upper()
            matchup = card["game"] or card["team"]
            if card.get("time"):
                matchup = f'{matchup} · {card["time"]} CT'
            portrait = (f'<img src="{character_uri}" alt="" />' if character_uri else symbol)
            route_style = ""
            support_route_note = ""
            if role == "Support" and move:
                variant = _support_variant(move)
                accent = move.get("route_accent") or {"Runesmith":"#79afff", "Stormcaller":"#63e1cf"}.get(variant, "#b692ff")
                route_style = f' style="--accent:{_h(accent)}"'
                route_labels = list(dict.fromkeys(
                    item["route_variant"] for item in card.get("moves", []) if item.get("route_variant")))
                animal = "Orangutan" if variant in {"Runesmith", "Oracle + Runesmith"} else "Chimp" if variant == "Stormcaller" else "Gorilla"
                support_route_note = (f'<div class="wn-support-route"><b>{_h(animal + " · " + variant)}</b>'
                                      f' · {_h(support_tier["label"])}'
                                      f'<span>Routes: {_h(" · ".join(route_labels))}</span></div>')
            if role != 'Support':
                support_route_note = (f'<div class="wn-support-route"><b>{_h(animal + " · " + character_variant)}</b></div>')
                if role == 'Carry' and move and move.get('goal_v2') and move.get('matchup_buff'):
                    support_route_note += f'<div class="wn-support-route"><span>Route: {_h(goal_buff_parent_name(move))} + {_h(move["name"])}</span></div>'
            name_backdrop = (f'<img class="wn-unit-ghost" src="{character_uri}" alt="" aria-hidden="true" />'
                             if character_uri else "")
            price = _odds(card["odds"])
            book = str(card.get("book") or "").strip()
            book_note = f' <small class="wn-book">BEST PRICE · {_h(book)}</small>' if book else ""
            book_prob = card.get("book_prob")
            book_rate_note = (f'<div class="wn-book-rate">BOOK BREAK-EVEN {book_prob*100:.1f}%</div>'
                              if book_prob is not None else "")
            tonight_xg = _number(card.get("tonight_goal_xg")) if role == "Carry" else None
            tonight_pct = _number(card.get("tonight_goal_pct")) if role == "Carry" else None
            tonight_note = (f'<div class="wn-goalie wn-projection">TONIGHT RESEARCH PROJECTION · '
                            f'{tonight_xg:.2f} expected goals · {tonight_pct:.1f}% for 1+ '
                            f'· being graded, not a tested move</div>'
                            if tonight_xg is not None and tonight_pct is not None else "")
            profile = card.get("goal_profile") or {}
            profile_parts = []
            for field, label, fmt in (("StarScore", "star score", ".1f"),
                                      ("Drought_G", "goal drought", ".0f"),
                                      ("Drought_PPP", "PPP drought", ".0f"),
                                      ("TOI_Pct", "TOI rank", ".1f"),
                                      ("iXG%", "iXG rank", ".0f"),
                                      ("ShotIntent", "shot intent", ".1f"),
                                      ("L10_S", "L10 shots", ".0f"),
                                      ("L10_G", "L10 goals", ".0f"),
                                      ("TOI_per_game", "TOI/game", ".1f"),
                                      ("Goal_mu", "model goal μ", ".2f"),
                                      ("Points_mu", "points μ", ".2f")):
                value = _number(profile.get(field))
                if value is not None:
                    profile_parts.append(f"{label} {value:{fmt}}")
            goal_profile_note = _render_goal_facts(card, move) if role == 'Carry' else '' 
            if role != "Carry":
                projected_mu = _number(card.get("tonight_mu"))
                projected_pct = _number(card.get("tonight_over_pct"))
                if projected_mu is not None and projected_pct is not None:
                    unit = "shots" if role == "Jungle" else "assists" if role == "Support" else "points"
                    tonight_note = (f'<div class="wn-goalie wn-projection">TONIGHT RESEARCH PROJECTION · '
                                    f'{projected_mu:.2f} expected {unit} · {projected_pct:.1f}% '
                                    f'over {card["line"]:g} · being graded, not a tested move</div>')
            goalie_name = str(card.get("goalie") or "").strip()
            goalie_status = str(card.get("goalie_status") or "Unknown").strip()
            if goalie_status.casefold() == "confirmed":
                goalie_status = "DailyFaceoff confirmed"
            elif goalie_status.casefold() == "official":
                goalie_status = "NHL official starter"
            goalie_gp = _number(card.get("goalie_gp"))
            sample_note = (f' · {int(goalie_gp)} GP, goalie matchup neutral'
                           if goalie_gp is not None and goalie_gp < 3 else
                           f' · {int(goalie_gp)} GP' if goalie_gp is not None else '')
            goalie_note = (f'<div class="wn-goalie">Opp goalie: {_h(goalie_name)} · {_h(goalie_status)}{sample_note}</div>'
                           if goalie_name else '<div class="wn-goalie">Opp goalie: unknown</div>')
            lineup_note = f'<div class="wn-goalie">Player lineup: {_h(card.get("lineup_status") or "Lineup pending")}</div>'
            positional = card.get("position_sog") if role == "Jungle" else None
            position_note = (f'<div class="wn-position-sog">{_h(positional["opponent"])} allows '
                             f'<b>{positional["shots_per_game"]:.1f} SOG/game</b> to {_h(positional["position"])} '
                             f'· #{positional["rank"]} most · {positional["gp"]} GP</div>'
                             if positional else "")
            fired = card.get("inspect_moves") or card.get("moves") or ([] if not move or baseline_only else [move])
            # Keep secondary Goal attacks in the full move list below.
            goal_signals = ""
            if role == 'Carry':
                buffs = [m for m in card.get('moves', []) if m.get('goal_visible_buff') and m.get('id') != (move or {}).get('id')]
                if buffs:
                    goal_signals = ('<div class="wn-assist-sample">BUFF RECORDS · NAMED ATTACK + MATCHUP</div>'
                                    if move.get('goal_v2') else '') + '<div class="wn-goal-signals">' + ''.join(
                        f'<div class="wn-goal-signal"><strong>{_h(goal_buff_parent_name(m)) + " + " if m.get("goal_v2") else "+ "}{_h(m["name"])}</strong>'
                        f'<span>{"V2" if m.get("goal_v2") else "V1"} {m["wins"]}/{m["picks"]} · {100*m["wins"]/m["picks"]:.1f}%</span></div>'
                        for m in buffs) + '</div>'
            support_badges = ""
            assist_facts = _render_assist_facts(card, move) if role == "Support" else ""
            sog_facts = _render_sog_facts(card, move) if role == "Jungle" else ""
            points_facts = _render_points_facts(card, move) if role == "Tank" else ""
            pp_crit_note = ""
            points_power_note = ('<div class="wn-support-badges">Enhancement: Blade Impale (Power Tier)</div>'
                                 if role == "Tank" and card.get("points_power_tier") else "")
            points_tier_note = (f'<div class="wn-crit-env">{_h(points_tier["rule"])} · '
                                f'sample through {"Nov 5" if points_tier.get("points15_current") else "Nov 4"} · overlapping move records</div>'
                                if points_tier else "")
            points_research_note = ""
            research = card.get("points15_research") or {}
            if role == "Tank" and card.get("line") == 1.5 and research.get("eligible"):
                metrics = []
                if research.get("pp_xgf60") is not None:
                    metrics.append(f'PP on-ice xGF/60 {research["pp_xgf60"]:.1f}')
                if research.get("attempts60") is not None:
                    metrics.append(f'5v5 attempts/60 {research["attempts60"]:.1f}')
                if research.get("opp_hdca60") is not None:
                    metrics.append(f'Opp 5v5 HDCA/60 {research["opp_hdca60"]:.2f}')
                tags = []
                if research.get("pp_pressure"):
                    tags.append("PP PRESSURE")
                if research.get("shot_generator"):
                    tags.append("5v5 SHOT GENERATOR")
                if research.get("dual"):
                    tags.append("DUAL PROFILE")
                if research.get("stingy_defense"):
                    tags.append("STINGY DEFENSE · CAUTION")
                detail = " · ".join(metrics) if metrics else "Research metrics pending"
                tag_text = " · ".join(tags) if tags else "NO RESEARCH TAG"
                points_research_note = (f'<div class="wn-research">RESEARCH · {_h(tag_text)}<br>'
                                        f'{_h(detail)} · tracking only; no card or move change</div>')
            if role == "Support" and move:
                secondary = list(dict.fromkeys(
                    item["name"] for item in fired
                    if item["name"] != move["name"] and item["kind"] not in {"STANCE", "CRIT"}
                ))
                if secondary:
                    support_badges = (f'<div class="wn-support-badges">Also: '
                                      f'{_h(" · ".join(secondary[:4]))}</div>')
                pp_rate = _number(card.get("team_pp_xgf60"))
                if support_tier["crit"] and move.get("environment_enhanced"):
                    pp_crit_note = ('<div class="wn-crit-env">TEAM PP ACTUAL GF/60 ≥5.7'
                                    ' + OPP 5v5 HDCA/60 ≤2.2 · ENVIRONMENT UPGRADE</div>')
                elif move["kind"] == "CRIT" and pp_rate is not None:
                    pp_crit_note = (f'<div class="wn-crit-env">TEAM PP xGF/60 '
                                    f'{pp_rate:.2f} · PP UPGRADE GATE 8.3</div>')
            move_rows = []
            goal_record_groups = {}
            for fired_move in fired:
                fired_wins, fired_picks = int(fired_move["wins"]), int(fired_move["picks"])
                fired_late_wins = int(fired_move["later_wins"])
                fired_late_picks = int(fired_move["later_picks"])
                fired_pct = 100 * fired_wins / fired_picks if fired_picks else 0
                fired_late_pct = 100 * fired_late_wins / fired_late_picks if fired_late_picks else 0
                fired_rule = str(fired_move.get("rule", fired_move.get("condition", "")))
                # Historical move keys stay frozen for comparable records.
                # Show their actual source meaning rather than implying xA.
                fired_rule = (fired_rule.replace("PP_iXA60", "weighted PP assists/60 (actual)")
                                         .replace("PP_Points60", "PP involvement proxy/60")
                                         .replace("iXA%", "observed assist rank %"))
                fired_label = ("TRACK" if fired_move.get("track") else
                               "LAB" if fired_move.get("experimental") else fired_move["kind"])
                if role == 'Jungle' and fired_move.get('sog_retired'):
                    fired_label = 'ARCHIVED · NOT A CARD SIGNAL'
                    fired_rule = 'Retired from featured cards Nov 11 · history preserved. ' + fired_rule
                if role == "Support" and not fired_move.get("track"):
                    fired_label = assist_display_tier(fired_move)["label"]
                later_text = (f"Mar 15+ {fired_late_wins}/{fired_late_picks} · {fired_late_pct:.1f}%"
                              if fired_late_picks else "2026–27 regular season"
                              if fired_move.get("sample_season") == "2026-27" else "No separate later sample")
                if fired_move.get("sample_from") and fired_move.get("sample_through"):
                    later_text += f' · {fired_move["sample_from"]} through {fired_move["sample_through"]}'
                if fired_move.get("sample_book_be_pct") is not None:
                    later_text += (f' · sample BE {fired_move["sample_book_be_pct"]:.1f}%'
                                   f' · {fired_move["sample_players"]} players · {fired_move["sample_dates"]} slate dates')
                move_row = f'''<div class="wn-move-entry">
                  <div class="wn-move-title"><strong>{_h(fired_move["name"])}</strong>{grade_badges(fired_move)}<em>{_h(fired_label)}</em></div>
                  <div class="wn-move-record">{fired_wins}/{fired_picks} · {fired_pct:.1f}% <span>{later_text}</span></div>
                  <div class="wn-move-rule">{_h(fired_rule)}</div>
                  {goal_return_panel(fired_move, compact=False) if role == 'Carry' else ''}
                  {assist_return_panel(fired_move, compact=False) if role == 'Support' else ''}
                  {points_return_panel(fired_move, compact=False) if role == 'Tank' else ''}
                  {sog_return_panel(fired_move, compact=False) if role == 'Jungle' else ''}
                </div>'''
                if role == 'Carry':
                    if fired_move.get('goal_retired'):
                        move_row = move_row.replace('<div class="wn-move-rule">', '<div class="wn-move-rule"><b>RETIRED BOARD ENTRY · record preserved</b><br>', 1)
                    group = next(iter(goal_inspect_groups([fired_move])))
                    goal_record_groups.setdefault(group, []).append(move_row)
                else:
                    move_rows.append(move_row)
            if role == 'Carry':
                move_rows = [f'<details class="wn-goal-record-group"><summary>{_h(group)} ({len(items)})</summary>{"".join(items)}</details>'
                             for group, items in goal_record_groups.items()]
            later_note = (f"MOVE {wins}/{picks}" +
                          (f" · MAR 15+ {late_wins}/{late_picks} ({late_pct:.1f}%)" if late_picks else
                           " · GOAL RECORD WITHOUT POINTS GATE" if card.get("goal_candidate") else "")
                          if picks else "No Goal attack fired · posted line" if role == "Carry" else
                          "No tested move · posted line" if "confidence" in card
                          else "Baseline screen · no upgraded move")
            if points_tier:
                later_note = f'CURRENT MOVE {wins}/{picks}'
            confidence = card.get("confidence")
            confidence_note = (f" · CONF {confidence:.0f} {_h(card.get('matrix') or 'Unknown')}"
                               if confidence is not None else "")
            record_label = ("2026–27 REG MOVE" if role == "Carry" or
                            (move and move.get("sample_season") == "2026-27") else "HISTORICAL MOVE")
            if points_tier:
                record_label = "2026–27 REG MOVE"
            if role == 'Carry' and goal_return_panel(move):
                record_label = "V2 MOVE %" if move.get("goal_v2") else "V1 MOVE %"
            if role == 'Support' and assist_return_panel(move):
                record_label = "V1 MOVE %"
            if role == 'Tank' and points_original_caption(move):
                record_label = points_original_caption(move)
            if role == 'Jungle' and sog_return_panel(move):
                record_label = "V2 MOVE %" if move.get('sog_profile') else "V1 MOVE %"
            if record_label in {"V1 MOVE %","V2 MOVE %"}:
                later_note = f'{record_label.split()[0]} MOVE {wins}/{picks}'
            record_html = (f'<div class="wn-record"><strong>{pct:.1f}%</strong><span>{wins}/{picks}</span><em>{record_label}</em></div>'
                           if picks else '<div class="wn-record"><strong>SCOUT</strong><em>NO GOAL ATTACK</em></div>'
                           if role == "Carry" else '<div class="wn-record"><strong>BASE</strong><em>SCREEN ONLY</em></div>')
            details_label = ('Move families &amp; preserved records' if role == 'Carry' else 'Full fired move list')
            details_html = (f'<details class="wn-details"><summary>{details_label} ({len(fired)})</summary>'
                            f'<div class="wn-move-list">{"".join(move_rows)}</div></details>' if fired else
                            f'<div class="wn-details">{_h((points_tier or {}).get("rule") or card.get("move_block_reason") or card.get("baseline_rule") or (move or {}).get("rule") or "No tested move fired")}</div>')
            move_name = move["name"] if move else (
                "Scouting Target" if role == "Carry" and baseline_only else
                "Green Baseline" if baseline_only else "Scouting pool")
            if role == "Support" and move:
                move_name = assist_display_name(move)
            units.append(f"""<article class="wn-unit{tier_class}"{route_style}>
              {name_backdrop}
              <div class="wn-portrait" aria-hidden="true">{portrait}</div>
              <div class="wn-unit-body">
                <div class="wn-card-heading"><div class="wn-unit-head"><span class="wn-rank">{rank:02d}</span>{tier_badge}<strong>{_h(card['player'])}</strong><span class="wn-match">{_h(matchup)}</span></div>
                <div class="wn-attack"><span class="wn-attack-name">{_h(move_name)}</span><span class="wn-badge">{_h(status + sample)}</span></div>
                {support_route_note}
                <div class="wn-unit-foot"><span>{_h(line)} <b>{_h(price)}</b>{book_note}</span><span>{_h(later_note)}{confidence_note}</span></div>
                {book_rate_note}
                {beta_summary(card.get('pp_assist_beta')) if role == 'Support' else ''}</div>
                {goal_return_panel(move) if role == 'Carry' else ''}
                {assist_return_panel(move) if role == 'Support' else ''}
                {points_return_panel(move) if role == 'Tank' else ''}
                {sog_return_panel(move) if role == 'Jungle' else ''}
              </div>
              {record_html}
              {_card_quick_summary(role, card)}
              <details class="wn-card-inspect"><summary>Inspect player · profile, matchup &amp; recent form</summary>
              <div class="wn-unit-info">
                {beta_details(card.get('pp_assist_beta')) if role == 'Support' else ''}
                {pp_crit_note}
                {assist_facts}
                {sog_facts}
                {points_facts}
                {points_power_note}
                {points_tier_note if not points_facts else ""}
                {points_research_note}
                {support_badges}
                {support_route_note if role == 'Support' else ""}
                {goal_signals}
                {goal_profile_note}
                <div class="wn-context">
                  {tonight_note}
                  {position_note}
                  {goalie_note}
                  {lineup_note}
                </div>
              </div>
              {_form_html(card, expanded=True)}
              {details_html}
              </details>
            </article>""")
        if not units:
            units = ['<div class="wn-empty">No featured move for this class on this slate. '
                     'Check the complete priced slate below for all posted lines.</div>']
        backdrop = f'<img class="wn-gorilla" src="{character_uri}" alt="" />' if character_uri else ""
        backdrop = _class_roster_art(role)
        count_label = "FEATURED" if cards and "confidence" in cards[0] else "PRICED"
        lane_subtitle = ("GOALS V2 ✦ 6 PLAYER FAMILIES ✦ 8 LEVELS ✦ OPTIONAL BUFFS ✦ ARCHIVED V1 IN INSPECT"
                         if role == "Carry" else
                         "POINTS ✦ CURRENT-SEASON MOVE % ✦ BOOK BREAK-EVEN %" if role == "Tank" else
                         f"{market.upper()} ✦ HISTORICAL MOVE % ✦ BOOK BREAK-EVEN % FROM POSTED ODDS")
        if role == "Jungle" and any((card.get("move") or {}).get("sog_current") for card in cards):
            lane_subtitle = "SOG2.5 ✦ 2026–27 MOVE % ✦ BASE67 · STRONGER71 · PK BUFF ≤104s/OPPORTUNITY"
        if role == "Support":
            lane_subtitle += " · CRIT I 75–79.9% · CRIT II 80%+ · MINIMUM 30 PLAYS; SMALLER SAMPLES MONITOR"
        lanes.append(f"""<section class="wn-lane wn-lane--{role.lower()}" style="--accent:{color}">
          <header class="wn-lane-head">{backdrop}<div class="wn-class-icon" aria-hidden="true">{class_icon}</div>
            <div class="wn-class-text"><span class="wn-kicker">{_h(descriptions[role])}</span><h2>{_h(role)}</h2></div>
            <div class="wn-count"><strong>{len(cards)}</strong><span>{count_label}</span></div></header>
          <div class="wn-lane-sub">{_h(lane_subtitle)}<div class="wn-grade-legend">S / A / B rank updated returns · Watch flags recent losses, deterioration or limited confirmation · every matched route stays in Inspect</div></div>
          <div class="wn-units">{''.join(units)}</div></section>""")
    total = sum(len(cards) for cards in boards.values())
    styles = """<style>
      .wn-board,.wn-board *{box-sizing:border-box}
      .wn-board{font-family:Inter,system-ui,sans-serif;color:#edf1f8}
      .wn-hero{position:relative;overflow:hidden;background:radial-gradient(circle at 89% 4%,#663c274d,transparent 36%),linear-gradient(115deg,#111a2a,#1b1a2b 64%,#261b23);border:1px solid #53516b;border-radius:14px;padding:22px 25px;margin:10px 0 15px;box-shadow:inset 0 0 50px #0005}
      .wn-hero:after{content:'⚔';position:absolute;right:28px;top:-36px;font-size:145px;line-height:1;color:#ffffff0e;transform:rotate(-18deg)}
      .wn-eyebrow,.wn-hero-foot{font-size:10px;letter-spacing:.23em;color:#e9b875;font-weight:800}
      .wn-hero h1{font-size:clamp(27px,3vw,43px);letter-spacing:.06em;line-height:1.05;margin:7px 0 8px;font-weight:950;color:#fff;text-shadow:0 3px 18px #0009}
      .wn-hero p{margin:0 0 14px;color:#cbd2df;font-size:13px}.wn-hero-foot{color:#aebbd1;letter-spacing:.11em}
      .wn-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;align-items:start}
      .wn-lane{min-width:0;background:#111a2a;border:1px solid #38435a;border-radius:14px;overflow:hidden;box-shadow:0 8px 28px #1018282e}
      .wn-lane-head{position:relative;isolation:isolate;display:flex;align-items:center;gap:12px;min-height:96px;padding:16px;background:linear-gradient(100deg,color-mix(in srgb,var(--accent) 23%,#111a2a),#111a2a 82%);border-bottom:1px solid #ffffff16;overflow:hidden}
      .wn-gorilla{position:absolute;z-index:-1;right:8px;top:0;height:230px;width:auto;opacity:.34;pointer-events:none;mask-image:linear-gradient(90deg,transparent,#000 35%)}
      .wn-lane .wn-support-roster{position:absolute;z-index:-1;right:0;top:0;height:100%;width:66%;display:flex;opacity:.55;pointer-events:none;mask-image:linear-gradient(90deg,transparent,#000 24%)}
      .wn-lane .wn-support-roster img{width:33.333%;height:100%;object-fit:cover;object-position:center 20%}
      .wn-class-icon{width:42px;height:42px;flex:none;display:grid;place-items:center;border:1px solid color-mix(in srgb,var(--accent) 50%,transparent);border-radius:9px;background:#0a1425e8;font-size:25px}
      .wn-class-icon svg{width:29px;height:29px;max-width:29px;max-height:29px;fill:var(--accent)}
      .wn-class-text{flex:1}.wn-kicker{font-size:10px;text-transform:uppercase;letter-spacing:.15em;color:#c5c6d3}.wn-class-text h2{font-size:23px;line-height:1;margin:3px 0 0;color:var(--accent);font-weight:950}
      .wn-count{display:flex;flex-direction:column;align-items:center;color:var(--accent);line-height:1;background:#0a1425b8;border:1px solid #ffffff16;border-radius:8px;padding:6px 8px}.wn-count strong{font-size:25px}.wn-count span{font-size:9px;letter-spacing:.12em;margin-top:3px}
      .wn-lane-sub{font-size:11px;line-height:1.6;font-weight:750;letter-spacing:.06em;color:#c8d5e8;padding:10px 16px;background:#0d1625}.wn-lane-sub span{color:var(--accent);padding:0 4px}
      .wn-units{padding:8px}.wn-unit{position:relative;isolation:isolate;overflow:hidden;display:flex;flex-wrap:wrap;gap:4px 10px;min-height:104px;align-items:center;background:#1b2739;border:1px solid #3b4b64;border-left:3px solid var(--accent);border-radius:9px;padding:10px;margin-bottom:7px}
      .wn-unit--s{border-left:4px solid #f4bf55;background:linear-gradient(100deg,#3d3026,#1b2739 39%);box-shadow:inset 0 0 0 1px #f4bf5538}
      .wn-unit--a{border-left:4px solid #9bbcff;background:linear-gradient(100deg,#263653,#1b2739 39%);box-shadow:inset 0 0 0 1px #9bbcff30}
      .wn-unit--b{border-left:4px solid #82d5b0;background:linear-gradient(100deg,#203b36,#1b2739 39%);box-shadow:inset 0 0 0 1px #82d5b030}
      .wn-unit--crit{border-left:4px solid #f4bf55;background:linear-gradient(100deg,#413015,#1b2739 43%);box-shadow:inset 0 0 0 1px #f4bf5540}
      .wn-tier{flex:none;border-radius:5px;padding:4px 7px;font-size:11px;font-weight:900;letter-spacing:.04em}
      .wn-tier--s{color:#211803;background:#f4bf55}.wn-tier--a{color:#0a1832;background:#9bbcff}.wn-tier--b{color:#09251c;background:#82d5b0}
      .wn-unit-ghost{position:absolute;z-index:0;left:65px;top:-28px;height:190px;width:auto;opacity:.16;pointer-events:none;mask-image:linear-gradient(90deg,#000 25%,transparent 95%)}
      .wn-unit:last-child{margin-bottom:0}.wn-portrait{position:relative;z-index:1;width:46px;height:46px;flex:none;display:grid;place-items:center;border:1px solid #ffffff2e;border-radius:9px;background:radial-gradient(circle at top left,color-mix(in srgb,var(--accent) 34%,#162035),#162035 75%);font-size:24px}
      .wn-portrait svg{width:29px;height:29px;max-width:29px;max-height:29px;fill:var(--accent)}
      .wn-portrait img{width:100%;height:100%;object-fit:cover;object-position:center top}
      .wn-unit-body{position:relative;z-index:1;flex:1;min-width:0}
      .wn-unit-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 8px;min-width:0}
      .wn-rank{font-size:12px;color:var(--accent);font-weight:850}
      .wn-unit-head strong{font-size:18px;line-height:1.35;color:#fff;overflow-wrap:anywhere;text-shadow:0 1px 8px #091321}
      .wn-match{font-size:12px;line-height:1.5;color:#cfdbeb;flex:none}
      .wn-attack{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin-top:7px;min-width:0}
      .wn-attack-name{font-size:15px;line-height:1.4;font-weight:800;color:#f5d28e;overflow-wrap:anywhere}
      .wn-badge{font-size:10px;line-height:1.4;letter-spacing:.02em;color:#e9ddff;border:1px solid color-mix(in srgb,var(--accent) 65%,transparent);border-radius:5px;padding:3px 5px;background:#0e1728}
      .wn-unit-foot{display:flex;flex-wrap:wrap;gap:7px 14px;align-items:center;margin-top:9px;padding:9px 11px;background:#111c2e;border:1px solid #465875;border-radius:7px;font-size:13px;line-height:1.6;color:#dbe7f8;letter-spacing:0}
      .wn-unit-foot b{color:#fff;margin-left:3px;font-size:15px}
      .wn-book{font-size:12px;line-height:1.6;color:#f5d28e;font-weight:800;white-space:normal;margin-left:6px}
      .wn-book-rate{font-size:12px;line-height:1.6;color:#e2ecfa;margin-top:7px;font-weight:800;letter-spacing:.025em}
      .wn-crit-env{font-size:12px;line-height:1.6;color:#ffe5a8;margin-top:8px;padding:8px 10px;background:#302611;border:1px solid #8d743d;border-radius:7px;font-weight:800;letter-spacing:.025em;overflow-wrap:anywhere}
      .wn-research{font-size:13px;line-height:1.65;color:#d8e9ff;margin-top:8px;padding:9px 11px;border:1px solid #5977a4;border-radius:7px;background:#111f35}
      .wn-support-badges{font-size:12px;color:#d4e1f5;margin-top:8px;padding:8px 10px;border:1px solid #40516b;border-radius:7px;background:#111c2e;line-height:1.65;overflow-wrap:anywhere}
      .wn-support-route{font-size:13px;color:#f3d38f;margin-top:8px;font-weight:800;line-height:1.6}
      .wn-support-route span{display:block;font-size:12px;font-weight:550;color:#cbd9ee;margin-top:3px;overflow-wrap:anywhere}
      .wn-lane--support .wn-unit--crit{animation:wn-support-glow 3s ease-in-out infinite}
      .wn-lane--support .wn-unit{align-items:flex-start;padding:14px;gap:10px 16px}
      .wn-lane--support .wn-unit-ghost{display:none}
      .wn-lane--support .wn-unit-body{order:0}
      .wn-lane--support .wn-portrait{order:2;width:160px;height:190px;overflow:hidden;border-color:var(--accent)}
      .wn-lane--support .wn-form,.wn-lane--support .wn-form-empty,.wn-lane--support .wn-details{order:4}
      .wn-lane--support .wn-unit-head strong{font-size:18px}
      .wn-lane--support .wn-unit-foot,.wn-lane--support .wn-goalie{font-size:13px;line-height:1.65}
      .wn-lane--support .wn-record{order:1;padding-top:5px}
      .wn-assist-facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr));gap:9px;margin:10px 0}
      .wn-assist-fact{font-size:13px;line-height:1.75;border:1px solid #465875;border-radius:7px;padding:11px 12px;background:#111c2e;color:#dbe7f8;overflow-wrap:anywhere}
      .wn-assist-fact small{display:block;font-size:10px;line-height:1.5;letter-spacing:.08em;color:#f2cf83;font-weight:850;margin-bottom:6px}
      .wn-assist-fact b{color:#fff;font-weight:800}.wn-assist-fact span{color:#c7d7ed}
      .wn-assist-sample{font-size:12px;line-height:1.6;color:#cfdef2;margin-top:7px}.wn-assist-sample b{color:#fff;font-weight:800}
      .wn-support-crit-i .wn-portrait{box-shadow:0 0 14px #79afff88;border:2px solid #9cd4ff}
      .wn-support-crit-ii .wn-portrait{box-shadow:0 0 20px #f4bf55aa;border:2px solid #f4bf55}
      @keyframes wn-support-glow{50%{box-shadow:0 0 15px #f4bf554d,inset 0 0 0 1px #f4bf5566}}
      @media(prefers-reduced-motion:reduce){.wn-lane--support .wn-unit--crit{animation:none}}
      @media(max-width:650px){.wn-lane--support .wn-portrait{order:1;width:92px;height:120px}.wn-lane--support .wn-unit{gap:10px;padding:12px}.wn-lane--support .wn-record{order:3;flex-basis:100%;text-align:left;flex-direction:row;gap:10px;align-items:center}.wn-lane--support .wn-unit-head{white-space:normal;flex-wrap:wrap}.wn-lane--support .wn-unit-head strong{font-size:17px}}
      .wn-context{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr));gap:8px;margin-top:10px}
      .wn-goalie{font-size:13px;line-height:1.65;color:#dbe7f8;padding:9px 11px;margin:0;background:#111c2e;border:1px solid #465875;border-radius:7px;overflow-wrap:anywhere}
      .wn-projection{grid-column:1/-1;color:#e1edff;border-color:#5977a4;background:#111f35}
      .wn-goal-profile{font-size:13px;line-height:1.7;color:#f1dba5;margin-top:8px;padding:9px 11px;border:1px solid #685631;border-radius:7px;background:#211f1a;font-weight:650;overflow-wrap:anywhere}
      .wn-goal-signals{display:grid;gap:4px;margin-top:7px}
      .wn-goal-signal{display:flex;flex-wrap:wrap;justify-content:space-between;gap:5px 10px;padding:8px 10px;border:1px solid #465875;border-radius:6px;font-size:13px;line-height:1.6;overflow-wrap:anywhere;background:#111c2e}
      .wn-goal-signal strong{color:#f5d99f}.wn-goal-signal span{color:#d8e5f6}
      .wn-goal-signal--buff{border-color:#59bcb46b;background:#14313c}.wn-goal-signal--buff strong{color:#a8f3dc}
      .wn-goal-signal--crit{border-color:#f4bf55aa;background:#49331e}.wn-goal-signal--crit strong{color:#ffe09b}
      .wn-lane--carry .wn-unit--crit{animation:wn-goal-glow 3s ease-in-out infinite}
      @keyframes wn-goal-glow{0%,100%{box-shadow:inset 0 0 0 1px #f4bf5540}50%{box-shadow:inset 0 0 0 1px #f4bf55aa,0 0 12px #f4bf5530}}
      @media(prefers-reduced-motion:reduce){.wn-lane--carry .wn-unit--crit{animation:none}}
      .wn-position-sog{font-size:13px;line-height:1.65;color:#c4efd9;padding:9px 11px;margin:0;background:#10251f;border:1px solid #3e6e59;border-radius:7px;overflow-wrap:anywhere}.wn-position-sog b{color:#f0fff7}
      .wn-record{position:relative;z-index:1;text-align:right;flex:none;min-width:80px;display:flex;flex-direction:column;line-height:1.2}
      .wn-record strong{font-size:27px;color:#fff}.wn-record span{color:#f6d587;font-size:15px;font-weight:900;margin-top:4px}.wn-record em{font-style:normal;color:#c4d2e7;font-size:10px;letter-spacing:.04em;margin-top:5px}
      .wn-details{position:relative;z-index:1;flex:0 0 100%;font-size:13px;line-height:1.65;color:#cfdcf0;border-top:1px solid #485971;padding-top:9px}
      .wn-details summary{cursor:pointer;color:#f1d18d;font-weight:750;font-size:13px;padding:4px 0}.wn-move-list{max-height:480px;overflow:auto;display:grid;gap:9px;margin-top:10px;padding-right:3px}
      .wn-form,.wn-form-empty{position:relative;z-index:1;flex:0 0 100%;font-size:13px;line-height:1.65;color:#d5e1f3;border-top:1px solid #485971;padding-top:9px}
      .wn-form summary{cursor:pointer;color:#f1d18d;font-weight:800;font-size:13px;padding:4px 0}.wn-form p{margin:9px 0;color:#c7d7ed}
      .wn-form-tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin:10px 0}.wn-form-tiles>div{border:1px solid #465875;border-radius:7px;padding:10px;background:#111c2e}
      .wn-form-tiles small{display:block;color:#c6d6ec;font-size:10px;letter-spacing:.05em;font-weight:700}.wn-form-tiles strong{display:block;color:#fff;font-size:14px;margin-top:5px}
      .wn-form-meta{margin:7px 0;padding:8px 10px;background:#111c2e;border:1px solid #40516b;border-radius:6px;color:#e3ecfa;font-weight:550}
      .wn-form-strip{display:flex;flex-wrap:wrap;gap:7px;margin:10px 0}.wn-form-strip span{border-radius:6px;padding:8px 9px;font-weight:800;text-align:center;min-width:52px;border:1px solid transparent}
      .wn-form-strip span b{display:block;font-size:14px}.wn-form-strip span small{display:block;font-size:11px;font-weight:650;margin-top:4px;opacity:1}
      .wn-form-strip .hit{background:#174e38;border-color:#4b9c75;color:#effff5}.wn-form-strip .miss{background:#602f3b;border-color:#a96779;color:#fff2f5}
      .wn-move-entry{border:1px solid #465875;border-radius:7px;background:#111c2e;padding:10px 12px}.wn-move-title{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:6px 10px}
      .wn-move-title strong{font-size:14px;color:#f7dfb2}.wn-move-title em{font-size:10px;line-height:1.5;font-style:normal;color:#d7c3ff;text-align:right}
      .wn-move-record{font-size:13px;line-height:1.65;font-weight:800;color:#fff;margin-top:6px}.wn-move-record span{color:#c7d7ed;margin-left:5px;font-weight:550}
      .wn-move-rule{font-size:12px;line-height:1.7;color:#d2e0f2;overflow-wrap:anywhere;margin-top:7px}
      .wn-empty{padding:26px 12px;text-align:center;color:#aebbd0;font-size:12px}
      .wn-board--compact .wn-grid{grid-template-columns:1fr}
      .wn-board--compact .wn-lane-head{display:none}
      @media(max-width:1050px){.wn-grid{grid-template-columns:1fr}}
      @media(max-width:540px){.wn-unit{gap:10px;padding:12px}.wn-unit-ghost{left:45px;opacity:.12}.wn-portrait{width:40px;height:40px}.wn-portrait svg{width:27px;height:27px}.wn-record{min-width:65px}.wn-record strong{font-size:24px}.wn-match{display:block;flex-basis:100%;font-size:12px}.wn-assist-facts,.wn-context{grid-template-columns:1fr}}

      .wn-lane{container-type:inline-size}
      .wn-lane .wn-unit{display:grid;grid-template-columns:minmax(0,1fr) 95px 170px;grid-template-areas:"body record portrait" "form form form" "details details details";align-items:start;gap:10px 16px;padding:14px}
      .wn-lane .wn-unit-ghost{display:none}
      .wn-lane .wn-unit-body{grid-area:body;min-width:0}
      .wn-lane .wn-portrait{grid-area:portrait;order:initial;width:170px;height:220px;overflow:hidden;border:1px solid var(--accent);background:radial-gradient(ellipse at center,color-mix(in srgb,var(--accent) 20%,#102039),#0b1424 80%)}
      .wn-lane .wn-portrait img{object-fit:cover;object-position:center top;padding:0}
      .wn-lane .wn-record{grid-area:record;order:initial;padding-top:5px;min-width:0}
      .wn-lane .wn-form,.wn-lane .wn-form-empty{grid-area:form;order:initial}
      .wn-lane .wn-details{grid-area:details;order:initial}
      @container(max-width:680px){.wn-lane .wn-unit{display:block;padding:14px}.wn-lane .wn-card-heading{padding-right:140px;min-height:245px}.wn-lane .wn-portrait{position:absolute;right:14px;top:14px;width:120px;height:165px}.wn-lane .wn-record{position:absolute;top:188px;right:14px;width:120px;display:flex;flex-direction:column;text-align:center;gap:3px}.wn-lane .wn-record span,.wn-lane .wn-record em{margin:0}.wn-lane .wn-form,.wn-lane .wn-form-empty,.wn-lane .wn-details{margin-top:10px}}
      @container(max-width:430px){.wn-lane .wn-unit{padding:11px}.wn-lane .wn-card-heading{padding-right:104px;min-height:215px}.wn-lane .wn-portrait{right:11px;top:11px;width:92px;height:130px}.wn-lane .wn-record{right:11px;top:150px;width:92px}.wn-lane .wn-record strong{font-size:24px}.wn-lane .wn-record em{font-size:8px}.wn-lane .wn-unit-head strong{font-size:17px}.wn-assist-facts{grid-template-columns:1fr}}

      .wn-lane .wn-unit{padding:11px;gap:7px 12px;grid-template-columns:minmax(0,1fr) 90px 140px;grid-template-areas:"body record portrait" "quick quick quick" "inspect inspect inspect"}
      .wn-lane .wn-portrait{width:140px;height:180px}
      .wn-lane .wn-unit-foot{margin-top:6px;padding:6px 8px;font-size:13px;line-height:1.5;gap:3px 10px}
      .wn-lane .wn-attack{margin-top:5px}.wn-lane .wn-support-route{margin-top:5px;font-size:12px}
      .wn-lane .wn-unit-body>.wn-card-heading>.wn-support-route span{display:none}
      .wn-quick-summary{grid-area:quick;color:#d7e5f7;font-size:13px;line-height:1.5;padding:5px 0}.wn-quick-summary b{color:#fff}
      .wn-card-inspect{grid-area:inspect;width:100%;min-width:0;border-top:1px solid #485971;padding-top:5px;color:#dbe7f8;font-size:13px}
      .wn-card-inspect>summary{cursor:pointer;color:#f1d18d;font-weight:750;padding:5px 0;font-size:13px;line-height:1.4}
      .wn-card-inspect[open]>summary{margin-bottom:7px}
      .wn-unit-info{width:100%;min-width:0}
      .wn-card-inspect .wn-assist-facts{grid-template-columns:repeat(auto-fit,minmax(min(100%,190px),1fr));gap:7px;margin:7px 0}
      .wn-card-inspect .wn-assist-fact{padding:8px 10px;font-size:13px;line-height:1.55}
      .wn-card-inspect .wn-context{gap:7px;margin-top:7px}.wn-card-inspect .wn-goalie{padding:7px 9px;line-height:1.5}
      .wn-card-inspect .wn-form,.wn-card-inspect .wn-form-empty,.wn-card-inspect .wn-details{margin-top:8px}
      @container(max-width:680px){.wn-lane .wn-unit{display:block;padding:11px}.wn-lane .wn-card-heading{padding-right:132px;min-height:230px}.wn-lane .wn-portrait{right:11px;top:11px;width:115px;height:150px}.wn-lane .wn-record{right:11px;top:172px;width:115px}.wn-lane .wn-quick-summary{margin-top:5px}.wn-card-inspect .wn-assist-facts{grid-template-columns:repeat(auto-fit,minmax(min(100%,190px),1fr))}}
      @container(max-width:430px){.wn-lane .wn-card-heading{padding-right:100px;min-height:200px}.wn-lane .wn-portrait{width:88px;height:120px}.wn-lane .wn-record{top:140px;width:88px}.wn-card-inspect .wn-assist-facts,.wn-card-inspect .wn-context{grid-template-columns:1fr}}
    </style>"""
    goal_count = len(boards.get("Carry", []))
    other_count = total - goal_count
    hero_text = ("Goal cards show named 2026–27 attacks. Points 0.5 cards show the current-season Tank move kit; "
                 "legacy Points rules remain frozen for grading. Other featured props require a Green baseline and a 50%+ historical move. "
                 "Compare records with posted book break-even; past rates are not tonight's probabilities."
                 if goal_count else
                 "Points 0.5 cards show the current-season Tank move kit. Other featured cards require a Green baseline and a 50%+ historical move. "
                 "Compare past results with the posted book break-even percentage.")
    hero_foot = (f"⚔ {goal_count} NAMED GOAL ATTACKS · {other_count} FEATURED OTHER PROP ENTRIES"
                 if goal_count else f"⚔ {total} FEATURED PLAYER PROP ENTRIES ACROSS FOUR CLASSES")
    hero = f"""<div class="wn-hero"><span class="wn-eyebrow">WARLORDS OF THE NIGHT · 2026</span>
      <h1>THE NIGHT RAID</h1><p>{_h(hero_text)}</p>
      <div class="wn-hero-foot">{_h(hero_foot)} · HISTORICAL MOVE RATE IS NOT A FORECAST</div></div>"""
    board_class = "wn-board" if show_hero else "wn-board wn-board--compact"
    goal_return_styles = '<style>' + GOAL_RETURN_CSS + '</style>' if any(item[0] in ('Carry', 'Support', 'Tank') for item in selected) else ''
    return BETA_CSS + '<style>' + GRADE_CSS + '</style>' + styles + goal_return_styles + f'<div class="{board_class}">' + (hero if show_hero else "") + '<div class="wn-grid">' + ''.join(lanes) + '</div></div>'








