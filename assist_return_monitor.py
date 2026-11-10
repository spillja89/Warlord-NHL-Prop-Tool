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
    return render_return_panel(row,records(),compact=compact) if row else ''
