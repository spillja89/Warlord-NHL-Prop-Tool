"""Dated Points0.5/1.5 return display; never changes rules, ranks or tiers."""
from functools import lru_cache
from pathlib import Path
from html import escape
import json
from goal_return_monitor import render_return_panel

def records():
    from card_return_store import load_records
    return load_records(Path(__file__).with_name('points_return_records.json'))


def record_for(move):
    if not isinstance(move,dict):return None
    legacy=move.get('name') in {'DPS Tank stance','Backbone'} and not move.get('points_current') and not move.get('kit_version') and move.get('market') in (None,'Points') and not move.get('points15_current')
    if not legacy and (move.get('market')!='Points' or not move.get('points_current')):return None
    key='legacy-points15|'+move['name'] if legacy else str(move.get('kit_version'))+'|'+str(move.get('id'))
    row=records().get('by_key',{}).get(key)
    if not row or row['name']!=move.get('name') or row['rule']!=move.get('rule'):return None
    if (row['original_hits'],row['original_plays'])!=(move.get('wins'),move.get('picks')):return None
    if not legacy and bool(move.get('points15_current'))!=(row['line']==1.5):return None
    return row

def original_caption(move):
    row=record_for(move)
    return (row.get('baseline_label') or 'V1 MOVE %') if row and not row.get('legacy') else ''

def display_stats(move):
    row=record_for(move)
    return row['full'] if row and not row.get('legacy') else {'hits':int(move.get('wins',0)), 'plays':int(move.get('picks',0)), 'be':move.get('sample_book_be_pct')}

def return_panel(move, *, compact=True):
    row=record_for(move)
    if row is None:return ''
    if row.get('legacy'):
        data={**records(),'previous_through':row['previous_through'],'since_from':row['since_from'],'move_version_label':row.get('move_version_label'),'wired_on':row.get('wired_on')}
        result=render_return_panel(row,data,compact=compact).replace('FULL MOVE RECORD','FULL 2026–27 RECORD',1)
        if not compact:
            result+='<div class="wn-gr-note">'+escape(data['quote_policy15'])+'</div>'
            result+='<div class="wn-gr-note">These returns replay the unchanged legacy rule on 2026–27 games. Its separate large historical hit-rate record is from 2025–26.</div>'
        return result
    from goal_return_monitor import _metric
    data={**records(),'previous_through':row['previous_through'],'since_from':row['since_from'],'move_version_label':'Moves V5','wired_on':'2026-11-18'}
    result=render_return_panel(row,data,compact=compact).replace('vs V1','vs V5').replace('SINCE CHECK %','SINCE V5 CHECK %')
    # Retain the same already-observed health window, including a losing trend.
    health={**row,'previous':row['health_previous'],'since_check':row['health_since_check']}
    health_html=render_return_panel(health,data,compact=True)
    marker=health_html.find('<div class="wn-gr-change ')
    if marker>=0:
        health_html=health_html[marker:].removesuffix('</div>').replace('vs V1','vs prior baseline')
        stats=row['health_since_check']
        if stats['plays']:
            result+='<div class="wn-gr-note"><b>RETAINED HEALTH · '+escape(row['health_since_from'])+' onward · '+f"{stats['hits']}/{stats['plays']} · {stats['roi']:+.1f}% ROI"+'</b>'+health_html+'</div>'
    if not compact:
        start=result.find('<div class="wn-gr-note">Full record:');end=result.find('</div>',start)
        note='V5 combined baseline through Nov 16; patch wired Nov 18. Since-check starts Nov 17, the first date outside that baseline; it is not all live V5 performance. Earlier appearances are already included, never added twice. Flat 1u at saved pregame prices.'
        if start>=0 and end>=0:result=result[:start]+'<div class="wn-gr-note">'+note+'</div>'+result[end+6:]
        for old in row.get('archived_versions',[]):
            label=old.get('move_version_label') or 'Previous version'
            result+='<details class="wn-details"><summary>Archived '+escape(label)+' · original &amp; prior check</summary><div class="wn-goal-returns">'
            for title,key in [('ORIGINAL RECORD','previous'),('PRIOR SINCE CHECK','since_check'),('COMBINED AT V5 START','full')]:
                result+='<div class="wn-gr-box"><small>'+title+'</small>'+_metric(old[key])+'</div>'
            result+='<div class="wn-gr-note">Already included in V5; these are not additional bets.</div></div></details>'
            recent=old.get('research_recent') or {}
            if recent.get('plays'):
                result+='<div class="wn-gr-note">PRIOR RESEARCH RECENT · Nov 6–15 · '+f"{recent['hits']}/{recent['plays']} · {recent['hit_rate']:.1f}% · BE {recent['be']:.1f}% · {recent['players']} players · {recent['units']:+.2f}u · {recent['roi']:+.1f}% ROI"+'. Included in the baseline; not additional plays.</div>'
        if row['line']==1.5:result+='<div class="wn-gr-note">'+escape(data['quote_policy15'])+'</div>'
    return result
