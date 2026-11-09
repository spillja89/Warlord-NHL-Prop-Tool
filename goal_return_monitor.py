"""Dated Goal return display only; never ranks, filters or mutates a move."""
from functools import lru_cache
from pathlib import Path
from html import escape
import json,math

@lru_cache(maxsize=1)
def records():
    try:
        data=json.loads(Path(__file__).with_name('goal_return_records.json').read_text(encoding='utf-8'))
        data['by_key']={r['key']:r for r in data['moves']}
        return data
    except (OSError,ValueError,KeyError,TypeError):
        return {}

def record_for(move):
    if not isinstance(move,dict) or move.get('market')!='Goal':return None
    data=records()
    if move.get('kit_version')!=data.get('kit'):return None
    if move.get('route_family')=='xg':
        key=('xg:defense' if move.get('matchup_buff') else
             f"xg:{move.get('research_name')}:{move.get('level')}:{bool(move.get('xg_boost'))}")
    else:key=move.get('id')
    row=data.get('by_key',{}).get(key)
    # Same names from another lane/record generation must never inherit returns.
    if not row or row['name']!=move.get('name') or row['rule']!=move.get('rule'):return None
    if (row['original_hits'],row['original_plays'])!=(move.get('wins'),move.get('picks')):return None
    return row

def _date(value):
    month,day=value[5:7],int(value[8:10])
    return {'10':'Oct','11':'Nov'}.get(month,month)+f' {day}'

def _metric(stats):
    if not stats['plays']:return '<b>—</b><span>No new plays · ROI —</span>'
    tone='gain' if stats['units']>0 else 'loss' if stats['units']<0 else 'flat'
    return (f'<b class="wn-gr-{tone}">{stats["units"]:+.2f}u <span>· {stats["roi"]:+.1f}% ROI</span></b>'
            f'<span>{stats["hits"]}/{stats["plays"]} · {stats["plays"]} × 1u bets</span>')

def return_panel(move, *, compact=True):
    row=record_for(move)
    if row is None:return ''
    data=records();full,recent,prior=row['full'],row['since_check'],row['previous']
    if recent['plays']:
        delta=full['roi']-prior['roi']
        arrow='↑' if delta>1e-9 else '↓' if delta<-1e-9 else '→'
        trend='up' if delta>1e-9 else 'down' if delta<-1e-9 else 'neutral'
        change=f'{arrow} {delta:+.1f} pp ROI vs previous full record'
        status='WINNING UNITS' if recent['units']>0 else 'LOSING UNITS' if recent['units']<0 else 'FLAT UNITS'
        tone='gain' if recent['units']>0 else 'loss' if recent['units']<0 else 'flat'
    else:
        change='— No new plays to judge';trend='neutral';status='NO NEW PLAYS';tone='flat'
    detail=('' if compact else
            f'<div class="wn-gr-note">Previous record through {_date(data["previous_through"])}: '
            f'{prior["units"]:+.2f}u · {prior["roi"]:+.1f}% ROI. '
            'Flat 1u at saved pregame prices. Overlapping moves are separate records.</div>')
    return (f'<div class="wn-goal-returns" data-return-version="{escape(data["version"])}">'
            f'<div class="wn-gr-box"><small>FULL · {_date(data["full_from"])}–{_date(data["full_through"])}</small>{_metric(full)}</div>'
            f'<div class="wn-gr-box"><small>SINCE LAST CHECK · {_date(data["since_from"])}–{_date(data["since_through"])}</small>{_metric(recent)}</div>'
            f'<div class="wn-gr-change"><strong class="wn-gr-{tone}">{status}</strong>'
            f'<span class="wn-gr-{trend}">{change}</span></div>{detail}</div>')

RETURN_CSS='''
.wn-goal-returns{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;margin:8px 0 2px;max-width:680px;font-size:12px;line-height:1.4}
.wn-gr-box{background:#101a2bcc;border:1px solid #4a5b73;border-radius:7px;padding:7px 9px;display:flex;flex-direction:column;gap:2px;min-width:0}
.wn-gr-box small{font-size:9px;font-weight:850;letter-spacing:.04em;color:#eac77e}
.wn-gr-box>b{font-size:15px;font-weight:850}.wn-gr-box>b>span{font-size:13px}.wn-gr-box>span{font-size:11px;color:#d0ddef}
.wn-gr-change{grid-column:1/-1;display:flex;gap:8px;flex-wrap:wrap;font-size:10px;align-items:center}.wn-gr-change>strong{font-size:10px}
.wn-goal-returns .wn-gr-gain{color:#82edb5}.wn-goal-returns .wn-gr-loss{color:#ff9c97}.wn-goal-returns .wn-gr-flat{color:#d6deeb}
.wn-goal-returns .wn-gr-up{color:#82edb5}.wn-goal-returns .wn-gr-down{color:#f2cd86}.wn-goal-returns .wn-gr-neutral{color:#c4d2e7}
.wn-gr-note{grid-column:1/-1;color:#bdcde3;font-size:11px}.wn-move-entry .wn-goal-returns{max-width:680px}
@media(max-width:420px){.wn-gr-box{padding:6px}.wn-gr-box>b{font-size:13px}.wn-gr-box>b>span{font-size:12px}.wn-gr-box small{font-size:8px}}
'''
