"""Display priority for exact saved props; never changes eligibility or returns."""
from html import escape
import math

def number(value):
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (TypeError,ValueError):
        return None

def book_be(odds):
    odds=number(odds)
    if odds is None or not odds:return None
    return 10000/(odds+100) if odds>0 else -100*odds/(100-odds)

def priority_key(pick,metadata):
    meta=metadata.get(pick['pick_id'],{})
    rank=meta.get('card_rank')
    if not rank:
        rank=({'S':4,'A':3,'B':2,'Review':1}.get(meta.get('tier'),0),not meta.get('watch',False),None,None,None)
    # First five terms are the existing card ranking. Only exact performance
    # ties reach the saved quote/profile tie-breakers. Missing is never zero.
    values=[number(x) for x in rank[:5]]
    values += [number(meta.get('saved_gap')),number(meta.get('confidence'))]
    return tuple(-v if v is not None else math.inf for v in values)+(str(meta.get('move_identity') or ''),str(pick.get('player') or '').casefold(),str(pick['pick_id']))

def assign_priorities(picks,metadata):
    result={k:dict(v) for k,v in metadata.items()}
    for rank,pick in enumerate(sorted(picks,key=lambda p:priority_key(p,metadata)),1):
        result.setdefault(pick['pick_id'],{})['priority']=rank
    return result

def priority_html(meta):
    """Short evidence for the same move that leads the full card."""
    if not meta:return ''
    h=lambda x:escape(str(x),quote=True)
    tier=meta.get('tier','Unrated');rank=meta.get('priority')
    rank_note=f'#{rank:02d} · ' if isinstance(rank,int) else ''
    badge=f'<span class="battle-priority battle-tier-{h(tier.lower())}">{rank_note}{h(tier)}'+(' TIER' if tier in {'S','A','B'} else '')+'</span>'
    if meta.get('watch'):badge+='<span class="battle-watch">⚠ WATCH</span>'
    move=meta.get('move_name');parts=[badge]
    if move:parts.append(f'<span class="battle-move">{h(move)}</span>')
    full=meta.get('full') or {};recent=meta.get('recent') or {}
    n=number(full.get('plays'));hits=number(full.get('hits'))
    roi=number(full.get('roi'));units=number(full.get('units'))
    if n and hits is not None:
        parts.append(f'<span class="battle-evidence">MOVE {int(hits)}/{int(n)} · <b>{100*hits/n:.1f}%</b></span>')
    else:parts.append('<span class="battle-evidence">Return record unavailable</span>')
    if roi is not None and units is not None:
        parts.append(f'<span class="battle-evidence">TOTAL <b>{units:+.2f}u · {roi:+.1f}% ROI</b></span>')
    recent_n=number(recent.get('plays'));recent_roi=number(recent.get('roi'))
    if recent_n and recent_roi is not None:
        tone='battle-loss' if recent_roi<0 else 'battle-profit'
        parts.append(f'<span class="battle-evidence {tone}">RECENT <b>{recent_roi:+.1f}% ROI</b> · {int(recent_n)} plays</span>')
    elif n:parts.append('<span class="battle-evidence">RECENT · no new graded plays</span>')
    be=number(meta.get('saved_be'));gap=number(meta.get('saved_gap'))
    if be is not None:
        suffix=f' · {gap:+.1f} pp vs move rate' if gap is not None else ''
        parts.append(f'<span class="battle-price">SAVED PRICE BE <b>{be:.1f}%</b>{suffix}</span>')
    return '<div class="battle-priority-summary">'+''.join(parts)+'</div>'
