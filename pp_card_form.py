"""Display-only PPP last-ten dressed-game chart from the saved pregame log."""
from html import escape
from datetime import date as calendar_date
import json,math

def recent_ppp(row):
    raw=row.get('Form_Log')
    try:data=json.loads(raw) if isinstance(raw,str) else raw
    except (ValueError,TypeError):return None
    if not isinstance(data,dict) or data.get('season')!='2026-27' or not isinstance(data.get('games'),list):return None
    day=str(row.get('Date') or '')[:10]
    try:calendar_date.fromisoformat(day)
    except ValueError:return None
    games={}
    for g in data['games']:
        if not isinstance(g,dict):continue
        date=str(g.get('d') or '')[:10]
        try:calendar_date.fromisoformat(date)
        except ValueError:continue
        if len(date)!=10 or not('2026-07-01'<=date<'2027-07-01') or date>=day:continue
        try:
            value=float(g.get('pp'))
            value=int(value) if math.isfinite(value) and value>=0 and value.is_integer() else None
        except (ValueError,TypeError):value=None
        games.setdefault(date,dict(date=date,value=value))
    recent=sorted(games.values(),key=lambda g:g['date'],reverse=True)[:10]
    known=[g for g in recent if g['value'] is not None]
    hits=sum(g['value']>=1 for g in known)
    return dict(games=recent,hits=hits,known=len(known),missing=len(recent)-len(known),rate=100*hits/len(known) if known else None)

def ppp_l10_label(row):
    form=recent_ppp(row)
    if not form or not form['games']:return 'Unavailable'
    if not form['known']:return f'Unavailable · {len(form["games"])} games'
    result=f'{form["hits"]}/{form["known"]} · {form["rate"]:.0f}%'
    return result+(f' · {form["missing"]} unavailable' if form['missing'] else '')

def ppp_l10_chart(row):
    form=recent_ppp(row)
    if not form or not form['games']:
        return '<section class="pp-w-form"><h4>PPP · LAST 10 GAMES</h4><p>Current-season PPP game log unavailable.</p></section>'
    tiles=[]
    for g in reversed(form['games']):
        v=g['value'];tone='unknown' if v is None else 'hit' if v>=1 else 'miss'
        mark='—' if v is None else f'✓ {v}' if v>=1 else '× 0'
        title=f'{g["date"]} · '+('PPP unavailable' if v is None else f'{v} PPP')
        tiles.append(f'<span class="{tone}" title="{escape(title,quote=True)}"><b>{mark}</b><small>{g["date"][5:].replace("-","/")}</small></span>')
    missing=(f' · {form["missing"]} missing PPP result'+('s' if form['missing']!=1 else '') if form['missing'] else '')
    return (f'<section class="pp-w-form" aria-label="Player PPP last 10 games"><h4>PPP · LAST {len(form["games"])} GAMES '
            f'<span>{escape(ppp_l10_label(row))}</span></h4><p>Over 0.5 PPP · oldest → newest{missing}</p>'
            f'<div class="pp-w-form-strip">{"".join(tiles)}</div></section>')

FORM_CSS='''
.pp-w-form{margin:9px 0;padding:9px;background:#101b2c;border:1px solid #52607c;border-radius:7px}
.pp-w-form h4{font-size:13px;color:#efd095;margin:0 0 5px}.pp-w-form h4>span{color:#eff6ff;margin-left:8px;font-size:14px}
.pp-w-card .pp-w-form p{font-size:11px;color:#cbdaf0;margin:3px 0 7px}
.pp-w-form-strip{display:flex;flex-wrap:wrap;gap:5px}.pp-w-form-strip>span{min-width:40px;padding:6px 7px;border-radius:5px;text-align:center;background:#29354b}
.pp-w-form-strip>span.hit{background:#245744}.pp-w-form-strip>span.miss{background:#663540}.pp-w-form-strip>span.unknown{background:#3a4559;color:#d8dfeb}
.pp-w-form-strip b{display:block;font-size:13px;color:#fff}.pp-w-form-strip small{display:block;font-size:10px;margin-top:4px;color:#e1e8f3}
'''
