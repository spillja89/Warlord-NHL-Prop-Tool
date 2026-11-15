"""Dated Assist return display; never changes eligibility, ranking or Crit labels."""
from functools import lru_cache
from pathlib import Path
import json
from goal_return_monitor import render_return_panel

def records():
    from card_return_store import load_records
    return load_records(Path(__file__).with_name('assist_return_records.json'))


def record_for(move):
    if not isinstance(move,dict) or move.get('market')!='Assists':return None
    key=str(move.get('kit_version'))+'|'+str(move.get('move_id') or move.get('name'))
    row=records().get('by_key',{}).get(key)
    if not row or row['name']!=move.get('name') or row['rule']!=move.get('rule'):return None
    if (row['original_hits'],row['original_plays'])!=(move.get('wins'),move.get('picks')):return None
    return row

def return_panel(move, *, compact=True):
    row=record_for(move)
    if not row:return ''
    data=records()
    if move.get('assist_v2'):
        data={**data,'move_version_label':row['move_version_label'],'wired_on':row['wired_on'],
              'full_from':'2026-10-13','previous_through':'2026-11-13','since_from':'2026-11-14'}
    result=render_return_panel(row,data,compact=compact)
    if move.get('assist_v2'):
        result=result.replace('vs V1','vs V2')
        if not compact and not row['since_check']['plays']:
            start=result.find('<div class="wn-gr-note">Full record:');end=result.find('</div>',start)
            if start>=0 and end>=0:
                result=result[:start]+'<div class="wn-gr-note">V2 research baseline through Nov 13; wired Nov 15. No new settled plays since the baseline. Earlier research is not live V2 performance; flat1u at saved pregame prices.</div>'+result[end+6:]
    return result
