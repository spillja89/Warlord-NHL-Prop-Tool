"""Compact V3 environment family display; complete source/return detail in Inspect."""
from html import escape
from goal_return_monitor import _metric

def return_panel(row,compact=True):
    html=('<div class="wn-goal-returns"><div class="wn-gr-version">Moves V3 · wired Nov 21</div>'
      '<div class="wn-gr-box"><small>V3 BASELINE %</small>'+_metric(row['previous'])+'</div>'
      '<div class="wn-gr-box"><small>SINCE V3 CHECK %</small>'+_metric(row['since_check'])+'</div>'
      '<div class="wn-gr-box"><small>TOTAL MOVE %</small>'+_metric(row['full'])+'</div></div>')
    if not compact:
        s=row.get('health_since_check',{})
        html+='<div class="pp-w-info"><b>RESEARCH HEALTH · Nov 1–19</b><br>'+_metric(s)
        if s.get('plays'):html+=f'<br>Average posted book BE {s["be"]:.1f}% · gap {s["gap"]:+.1f} pp · {s["players"]} players.'
        html+='</div><div class="pp-w-info">V3 baseline through Nov 19; this health subset is included in that baseline. New V3 capture begins Nov 21. Zero new plays is not confirmation. Flat1u at saved ordinary pregame quotes. Shared environment families and nested levels are overlapping records.</div>'
        if row.get('context_note'):html+='<div class="pp-w-info">'+escape(row['context_note'])+'</div>'
    return html

def badge(card):
    from pp_cards import display_record
    from move_return_grades import pp_move_rank
    from pp_v2_display import card_qualified
    moves=[m for m in card['moves'] if m.get('pp_v3') and m.get('optional_buff') and card_qualified(m)]
    if not moves or card['best'].get('pp_v3') and card['best'].get('optional_buff'):return ''
    m=max(moves,key=pp_move_rank);s=display_record(m)
    title=f"Strongest matching V3 environment route: {s['hits']}/{s['plays']} · {s['rate']:.1f}%; sample BE {s['BE']:.1f}%. Other families and recent returns are in Inspect. Optional context, not a gate or additive probability bonus."
    return '<div class="pp-v3-buff" style="margin:7px 0;padding:6px 9px;border:1px solid #e8ca7c;border-radius:5px;background:#3b3344;color:#ffe3a0;font-weight:800;box-shadow:0 0 10px #e9c76b35" title="'+escape(title,quote=True)+'">✦ '+escape(m['name'])+f" · {s['hits']}/{s['plays']} · {s['rate']:.1f}%</div>"
