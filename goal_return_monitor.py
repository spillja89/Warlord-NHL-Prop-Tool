"""Dated Goal return display only; never ranks, filters or mutates a move."""
from functools import lru_cache
from pathlib import Path
from html import escape
import json,math

def records():
    from card_return_store import load_records
    return load_records(Path(__file__).with_name('goal_return_records.json'))


@lru_cache(maxsize=1)
def archived_records():
    data=json.loads(Path(__file__).with_name('goal_v1_return_records.json').read_text(encoding='utf-8'))
    return {**data,'by_key':{r['key']:r for r in data['moves']}}

def _data_for(move):return records() if move.get('goal_v2') else archived_records()

def record_for(move):
    if not isinstance(move,dict) or move.get('market')!='Goal':return None
    data=_data_for(move)
    if move.get('goal_v2'):key=move.get('goal_record_key')
    elif move.get('route_family')=='xg':
        key=('xg:defense' if move.get('matchup_buff') else f"xg:{move.get('research_name')}:{move.get('level')}:{bool(move.get('xg_boost'))}")
    else:key=move.get('id')
    row=data.get('by_key',{}).get(key)
    if not row or move.get('kit_version')!=row.get('kit_version',data.get('kit')):return None
    if not row or row['name']!=move.get('name') or row['rule']!=move.get('rule'):return None
    if (row['original_hits'],row['original_plays'])!=(move.get('wins'),move.get('picks')):return None
    return row


def _date(value):
    month,day=value[5:7],int(value[8:10])
    return {'10':'Oct','11':'Nov'}.get(month,month)+f' {day}'

def _metric(stats):
    if not stats['plays']:return '<strong class="wn-gr-record">—</strong><span>0 graded plays · ROI —</span>'
    tone='gain' if stats['units']>0 else 'loss' if stats['units']<0 else 'flat'
    record=(f'<strong class="wn-gr-record">{stats["hits"]}/{stats["plays"]} · '
            f'{100*stats["hits"]/stats["plays"]:.1f}%</strong>')
    return (record+f'<b class="wn-gr-{tone}">{stats["units"]:+.2f}u <span>· {stats["roi"]:+.1f}% ROI</span></b>'
            f'<span>{stats["plays"]} × 1u bets</span>')

def return_panel(move, *, compact=True):
    row=record_for(move)
    if row is None:return ''
    data=_data_for(move)
    if move.get('goal_v3'):
        data={**data,**{k:row[k] for k in ('move_version_label','wired_on','full_from','since_from','previous_through') if k in row},'kit':move['kit_version']}
    panel=render_return_panel(row,data,compact=compact)
    if move.get('goal_v3') and not compact:
        health=row.get('health_since_check',{});n=health.get('plays',0)
        if n:
            panel+=(f'<div class="wn-gr-note">RECENT HEALTH · from {escape(row["health_since_from"])} · '
                    f'{health["hits"]}/{n} · {health["hit_rate"]:.1f}% · avg posted BE {health["be"]:.1f}% · '
                    f'gap {health["gap"]:+.1f} pp · {health["players"]} players · '
                    f'{health["units"]:+.2f}u · {health["roi"]:+.1f}% ROI. '
                    'This subset is already included in the total; earlier research is not post-patch performance.</div>')
    return panel


def render_return_panel(row,data, *, compact=True):
    """Shared presentation; caller owns the exact market/rule/record identity."""
    full,recent,prior=row['full'],row['since_check'],row['previous']
    alert=''
    if recent['plays']:
        delta=full['roi']-prior['roi']
        arrow='↑' if delta>1e-9 else '↓' if delta<-1e-9 else '→'
        trend='up' if delta>1e-9 else 'down' if delta<-1e-9 else 'neutral'
        comparison='base' if row.get('legacy') else (row.get('baseline_label') or ('V2 MOVE %' if data.get('kit')=='goal-2026-27-moves-v2-2026-11-11' else 'V1 MOVE %')).split()[0]
        change=f'{arrow} Total ROI {delta:+.1f} pp vs {comparison}'
        status='RECENT PROFIT' if recent['units']>0 else 'RECENT LOSS' if recent['units']<0 else 'RECENT FLAT'
        tone='gain' if recent['units']>0 else 'loss' if recent['units']<0 else 'flat'
        if delta < -1e-9:
            alert='<strong class="wn-gr-alert">⚠ ROI DECLINING</strong>'
    else:
        change='— No new plays to judge';trend='neutral';status='NO NEW PLAYS';tone='flat'
    version=data.get('move_version_label')
    wired=data.get('wired_on')
    version_note=(f'<div class="wn-gr-version">{escape(version)}'
                  +(f' · wired {_date(wired)}' if wired else '')+'</div>') if version else ''
    detail=('' if compact else
            f'<div class="wn-gr-note">Full record: {_date(data["full_from"])}–{_date(data["full_through"])}. '
            f'Since last check: {_date(data["since_from"])}–{_date(data["since_through"])}. '
            f'Previous record through {_date(data["previous_through"])}: '
            f'{prior["units"]:+.2f}u · {prior["roi"]:+.1f}% ROI. '
            'Flat 1u at saved pregame prices. Overlapping moves are separate records.</div>')
    if data.get('kit') in ('goal-2026-27-moves-v2-2026-11-11','goal-2026-27-moves-v3-2026-11-23','goal-2026-27-buffs-v4-2026-11-25') and not recent['plays'] and not compact:
        patch=str(data.get('move_version_label') or ('Moves V3' if data['kit']=='goal-2026-27-moves-v3-2026-11-23' else 'Moves V2')).split()[-1]
        detail=('<div class="wn-gr-note">'+patch+' research baseline: '+_date(data['full_from'])+'–'+_date(data['previous_through'])+
                '. No new settled plays since wiring. Earlier research is not live '+patch+' performance. Flat1u at saved pregame prices.</div>')
    return (f'<div class="wn-goal-returns" data-return-version="{escape(data["version"])}">'
            +version_note+
            f'<div class="wn-gr-box"><small>{escape(row.get("baseline_label") or ("BASE RECORD %" if row.get("legacy") else "V1 MOVE %"))}</small>{_metric(prior)}</div>'
            f'<div class="wn-gr-box"><small>SINCE CHECK %</small>{_metric(recent)}</div>'
            f'<div class="wn-gr-box"><small>TOTAL MOVE %</small>{_metric(full)}</div>'
            f'<div class="wn-gr-change wn-gr-change-{trend}">{alert}<strong class="wn-gr-{tone}">{status}</strong>'
            f'<span class="wn-gr-{trend}">{change}</span></div>{detail}</div>')

RETURN_CSS='''
.wn-goal-returns{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin:8px 0 2px;max-width:760px;font-size:12px;line-height:1.4}
.wn-gr-version{grid-column:1/-1;font-size:10px;font-weight:800;color:#c4d2e7;letter-spacing:.03em}
.wn-gr-box{background:#101a2bcc;border:1px solid #4a5b73;border-radius:7px;padding:7px 9px;display:flex;flex-direction:column;gap:2px;min-width:0}
.wn-gr-box small{font-size:9px;font-weight:850;letter-spacing:.04em;color:#eac77e}
.wn-gr-box>.wn-gr-record{font-size:15px;font-weight:850;color:#f1f6ff}
.wn-gr-box>b{font-size:13px;font-weight:850}.wn-gr-box>b>span{font-size:11px}.wn-gr-box>span{font-size:10px;color:#d0ddef}
.wn-gr-change{grid-column:1/-1;display:flex;gap:8px;flex-wrap:wrap;font-size:10px;align-items:center}.wn-gr-change>strong{font-size:10px}
.wn-gr-change-down{background:#65420d40;border:1px solid #c8953d;border-radius:5px;padding:6px 8px}
.wn-gr-change>.wn-gr-alert{color:#ffdb80;font-size:11px;font-weight:900;letter-spacing:.03em}
.wn-goal-returns .wn-gr-gain{color:#82edb5}.wn-goal-returns .wn-gr-loss{color:#ff9c97}.wn-goal-returns .wn-gr-flat{color:#d6deeb}
.wn-goal-returns .wn-gr-up{color:#82edb5}.wn-goal-returns .wn-gr-down{color:#f2cd86}.wn-goal-returns .wn-gr-neutral{color:#c4d2e7}
.wn-gr-note{grid-column:1/-1;color:#bdcde3;font-size:11px}.wn-move-entry .wn-goal-returns{max-width:680px}
@media(max-width:420px){.wn-gr-box{padding:6px}.wn-gr-box>.wn-gr-record{font-size:12px}.wn-gr-box>b{font-size:11px}.wn-gr-box>b>span{font-size:10px}.wn-gr-box small{font-size:8px}}
'''

