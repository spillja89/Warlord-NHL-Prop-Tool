"""Exact current-season SOG/PPP return identities; display never changes eligibility."""
from functools import lru_cache
from pathlib import Path
import json
from goal_return_monitor import render_return_panel

def records():
    from card_return_store import load_records
    return load_records(Path(__file__).with_name('special_return_records.json'))


def record_for(move,market):
    if not isinstance(move,dict):return None
    if market=='SOG':
        if move.get('market')!='SOG' or not move.get('sog_current'):return None
        key='SOG|'+str(move.get('name'));original=(move.get('wins'),move.get('picks'));kit=move.get('kit_version');rule=move.get('rule')
    elif market=='PPP':
        key='PPP|'+str(move.get('id'));r=move.get('record',{});original=(r.get('hits'),r.get('plays'))
        kit=move.get('kit_version','pp-card-kit-2026-11-07-v2');rule=' + '.join(f'{field} {op} {cut:g}' for field,op,cut in move.get('entry',[])+move.get('conditions',[]))
    else:return None
    row=records().get('by_key',{}).get(key)
    if not row or row['name']!=move.get('name') or row['kit_version']!=kit or row['rule']!=rule:return None
    if (row['original_hits'],row['original_plays'])!=original:return None
    return row

def _panel(move,market,compact):
    row=record_for(move,market)
    if row is None:return ''
    if market=='PPP' and move.get('pp_v3'):
        from pp_v3_display import return_panel
        return return_panel(row,compact=compact)
    if market=='PPP' and move.get('pp_v2'):
        from pp_v2_display import return_panel
        return return_panel(row,compact=compact)
    data={**records(),**{f:row[f] for f in ('previous_through','since_from','move_version_label','wired_on')}}
    if market=='SOG' and row.get('record_patch')=='sog-v3-2026-11-19':
        from sog_record_display import return_panel as v3_panel
        return v3_panel(row,data,compact)
    result=render_return_panel(row,data,compact=compact)
    if market=='PPP' and not compact:
        result+='<div class="wn-gr-note">PPP results retain captured or explicitly labeled pregame-replay provenance. Each player-game is counted once per unchanged rule.</div>'
    return result

def sog_return_panel(move,*,compact=True):return _panel(move,'SOG',compact)
def pp_return_panel(move,*,compact=True):return _panel(move,'PPP',compact)

