"""Distinct optional Goal buff emblems; no changes to eligibility or ranking."""
from html import escape
from base64 import b64encode
FLAIR={
 'vital_breach':('#ff8a8a','M12 3v4m0 10v4M3 12h4m10 0h4M8 8l8 8m0-8-8 8M12 6a6 6 0 1 0 0 12 6 6 0 0 0 0-12','Crimson vital-point mark'),
 'war_chant':('#ffcc79','M3 10h5l10-6v16L8 14H3zM8 14l2 6h4l-3-7M21 8v8','Amber war horn'),
 'battleforged_fury':('#ffae72','M12 3 3 7v6c0 5 9 9 9 9s9-4 9-9V7ZM9 15v-4l2-1V7l3 3h2v5l-3 3z','Molten battle shield'),
 'fractured_guard':('#d0a4ff','M7 3h10l4 6-2 10-7 3-7-3-2-10ZM7 10h3m4 0h3M12 4l-2 8 4 2-2 7M7 16h2m6 0h2','Fractured opposing goalie mask'),
 'counterstrike_sigil':('#8cdcff','M5 4l14 14-3 3L2 7zM19 3l2 2-7 7M5 19l4-4M15 3h4v4M4 15v4h4','Ice-blue counterstrike rune')}
def family(move):return (move or {}).get('goal_buff_family')
def _emblem(move):
    color,path,title=FLAIR.get(family(move),('#ffd177','M12 3 3 12l9 9 9-9z','Optional Goal buff'))
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"><path d="{path}"/></svg>'
    return f'--gb-accent:{color};--gb-soft:{color}18',f'<img width="24" height="24" alt="" src="data:image/svg+xml;base64,{b64encode(svg.encode()).decode()}">',title
def condition(move):
    from goal_routes_v4 import LABELS
    checks=(move or {}).get('checks') or []
    if not checks:return ''
    field,op,cut=checks[-1];sign={'>=':'≥','<=':'≤'}.get(op,op)
    return LABELS.get(field,field.replace('_',' '))+' '+sign+f'{cut:g}'
def lead_context(move):
    if not (move or {}).get('goal_v4'):return ''
    style,icon,title=_emblem(move);kind='GOAL MOVE + GOALIE BUFF' if family(move)=='fractured_guard' else 'GOAL MOVE + TEAM BUFF'
    return f'<div class="wn-goal-buff-lead" style="{style}" title="{escape(title,quote=True)}">{icon}{kind}</div><div class="wn-goal-parent">Base move: <b>{escape(move.get("goal_parent_name") or "Goal route")}</b><span>{escape(condition(move))}</span></div>'
def badges(moves,lead=None):
    from move_return_grades import grade_rank
    from goal_return_monitor import record_for
    selected={}
    for m in moves:
        key=family(m)
        if key not in FLAIR:continue
        prior=selected.get(key)
        if prior is None or (grade_rank(m) or ())>(grade_rank(prior) or ()):selected[key]=m
    parts=[]
    for m in selected.values():
        if lead and m.get('id')==lead.get('id'):continue
        row=record_for(m);s=row['full'] if row else None
        h,n=(s['hits'],s['plays']) if s else (m['wins'],m['picks'])
        style,icon,title=_emblem(m);parent=m.get('goal_parent_name') or 'Exact named Goal route'
        tip=parent+' + '+condition(m)+'. Combined historical route+buff record. Full and recent returns are in Inspect; this is not a percentage-point bonus.'
        parts.append(f'<span class="wn-goal-buff-badge" style="{style}" title="{escape(tip,quote=True)}">{icon}<span>{escape(m["name"])}</span><b>{h}/{n} · {100*h/n:.1f}%</b></span>')
    return '<div class="wn-goal-buff-strip">'+''.join(parts)+'</div>' if parts else ''
CSS='''
.wn-goal-buff-lead{display:flex;align-items:center;gap:7px;font-size:10px;font-weight:900;letter-spacing:.07em;color:var(--gb-accent);margin:5px 0}.wn-goal-buff-lead img{width:26px;height:26px}
.wn-goal-parent{font-size:12px;color:#deebff;margin:3px 0 7px}.wn-goal-parent span{display:block;font-size:11px;color:#c7d7ed}.wn-goal-parent b{color:#fff}
.wn-goal-buff-strip{display:flex;flex-wrap:wrap;gap:6px;margin:7px 0}.wn-goal-buff-badge{display:inline-flex;align-items:center;flex-wrap:wrap;gap:6px;border:1px solid var(--gb-accent);background:linear-gradient(110deg,var(--gb-soft),#142037);color:var(--gb-accent);padding:5px 8px;border-radius:6px;font-size:11px;font-weight:800;line-height:1.4}.wn-goal-buff-badge img{width:24px;height:24px}.wn-goal-buff-badge b{color:#f5f8ff}.wn-goal-buff-badge:hover{box-shadow:0 0 10px var(--gb-soft)}
'''
