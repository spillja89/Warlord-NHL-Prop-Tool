"""Assist V2 display records; frozen rule identities and eligibility stay intact."""
from pathlib import Path
from html import escape
from goal_return_monitor import render_return_panel, _metric

def records():
    from card_return_store import load_records
    return load_records(Path(__file__).with_name('assist_return_records.json'))

def record_for(move):
    if not isinstance(move, dict) or move.get('market') != 'Assists':return None
    key = str(move.get('kit_version')) + '|' + str(move.get('move_id') or move.get('name'))
    row = records().get('by_key', {}).get(key)
    if not row or row['name'] != move.get('name') or row['rule'] != move.get('rule'):return None
    if (row['original_hits'], row['original_plays']) != (move.get('wins'), move.get('picks')):return None
    return row

def display_stats(move):
    """One current record for headline, Crit tier, tracker and fired-list details."""
    row = record_for(move)
    return row['full'] if row else {'hits': int(move.get('wins', 0)), 'plays': int(move.get('picks', 0))}

def return_panel(move, *, compact=True):
    row = record_for(move)
    if not row:return ''
    data = records()
    result = render_return_panel(row, data, compact=compact).replace('vs V1', 'vs V2').replace('SINCE CHECK %', 'SINCE V2 CHECK %')
    if not compact:
        start = result.find('<div class="wn-gr-note">Full record:');end = result.find('</div>', start)
        if start >= 0 and end >= 0:
            note = (f"V2 combined baseline through {data['previous_through']}; wired {data['wired_on']}. "
                    f"New-period tracking starts {data['since_from']}, the first date outside that baseline. "
                    "Return tiers retain the prior health window; the new V2 counter does not erase it. Earlier replay is not live V2 performance. Flat 1u at saved pregame prices; overlapping moves are separate records.")
            if not row['since_check']['plays']:note += ' No new settled plays to judge yet.'
            result = result[:start] + '<div class="wn-gr-note">' + escape(note) + '</div>' + result[end+6:]
        old = row.get('archived_v1')
        if old:
            result += ('<details class="wn-details wn-assist-v1-archive"><summary>Archived V1 &amp; prior since-check record</summary><div class="wn-goal-returns">'
                       '<div class="wn-gr-box"><small>ORIGINAL V1 %</small>' + _metric(old['baseline']) + '</div>'
                       '<div class="wn-gr-box"><small>V1 SINCE CHECK %</small>' + _metric(old['since_check']) + '</div>'
                       '<div class="wn-gr-box"><small>COMBINED AT V2 START</small>' + _metric(old['full']) + '</div>'
                       '<div class="wn-gr-note">Original V1 through ' + escape(old['previous_through'])
                       + '; prior since-check ' + escape(old['since_from']) + '–' + escape(old['through'])
                       + '. These appearances are already included in V2, not additional bets.</div></div></details>')
    return result
