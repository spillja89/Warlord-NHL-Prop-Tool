"""Exact current-season SOG/PPP return identities; display never changes eligibility."""
from functools import lru_cache
from pathlib import Path
import json
from goal_return_monitor import render_return_panel

@lru_cache(maxsize=1)
def records():
    try:
        data=json.loads(Path(__file__).with_name('special_return_records.json').read_text(encoding='utf-8'))
        data['by_key']={r['key']:r for r in data['moves']}
        return data
    except (OSError,ValueError,KeyError,TypeError):return {}

def record_for(move,market):
    if not isinstance(move,dict):return None
    if market=='SOG':
        if move.get('market')!='SOG' or not move.get('sog_current'):return None
        key='SOG|'+str(move.get('name'));original=(move.get('wins'),move.get('picks'));kit=move.get('kit_version');rule=move.get('rule')
    elif market=='PPP':
        key='PPP|'+str(move.get('id'));r=move.get('record',{});original=(r.get('hits'),r.get('plays'))
        kit='pp-card-kit-2026-11-07-v2';rule=' + '.join(f'{field} {op} {cut:g}' for field,op,cut in move.get('entry',[])+move.get('conditions',[]))
    else:return None
    row=records().get('by_key',{}).get(key)
    if not row or row['name']!=move.get('name') or row['kit_version']!=kit or row['rule']!=rule:return None
    if (row['original_hits'],row['original_plays'])!=original:return None
    return row

def _panel(move,market,compact):
    row=record_for(move,market)
    if row is None:return ''
    data={**records(),**{f:row[f] for f in ('previous_through','since_from','move_version_label','wired_on')}}
    result=render_return_panel(row,data,compact=compact)
    if market=='PPP' and not compact:
        result+='<div class="wn-gr-note">PPP additions: Nov 7 uses reconstructed pregame inputs; Nov 8 uses captured pregame tags. Each player-game is counted once per unchanged rule.</div>'
    return result

def sog_return_panel(move,*,compact=True):return _panel(move,'SOG',compact)
def pp_return_panel(move,*,compact=True):return _panel(move,'PPP',compact)
