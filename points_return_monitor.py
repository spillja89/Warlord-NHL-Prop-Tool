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

def return_panel(move, *, compact=True):
    row=record_for(move)
    if row is None:return ''
    data={**records(),'previous_through':row['previous_through'],'since_from':row['since_from'],
          'move_version_label':row.get('move_version_label'),'wired_on':row.get('wired_on')}
    if move.get('points15_v2'):
        data={**data,'full_from':'2026-10-13','previous_through':'2026-11-13','since_from':'2026-11-14'}
    result=render_return_panel(row,data,compact=compact)
    if move.get('points15_v2'):
        result=result.replace('vs V1','vs V2')
        if not compact and not row['since_check']['plays']:
            start=result.find('<div class="wn-gr-note">Full record:')
            end=result.find('</div>',start)
            if start>=0 and end>=0:
                result=result[:start]+'<div class="wn-gr-note">V2 research baseline through Nov 13. No new settled plays since wiring Nov 14. Earlier research is not live V2 performance; flat 1u at saved pregame prices.</div>'+result[end+6:]
    if not compact and row['line']==1.5:
        result+='<div class="wn-gr-note">'+escape(data['quote_policy15'])+'</div>'
    if row.get('legacy'):
        result=result.replace('FULL MOVE RECORD','FULL 2026–27 RECORD',1)
        if not compact:result+='<div class="wn-gr-note">These returns replay the unchanged legacy rule on 2026–27 games. Its separate large historical hit-rate record is from 2025–26.</div>'
    return result
