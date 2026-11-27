"""Compact optional PP buffs and consistent V2 baseline/health provenance."""
from html import escape
from goal_return_monitor import _metric

MIN_HISTORICAL_GAP_PP=15.0

def card_qualified(move):
    from pp_cards import display_record
    from board_move_policy import board_eligible
    s=display_record(move);gap=s.get('gap')
    return board_eligible(move) and gap is not None and gap>=MIN_HISTORICAL_GAP_PP

def goalie_summary(row):
    from pp_v2_moves import number,normal_name
    name=str(row.get('Opp_Goalie') or '').strip();key=normal_name(name)
    if key in {'','none','nan','unknown'}:return '<p class="pp-w-goalie">OPP GOALIE · Unavailable</p>'
    source=str(row.get('Opp_Goalie_Source') or '')
    match=normal_name(row.get('Opp_Goalie_Stats_Name',''));gp=number(row.get('Opp_GP'))
    valid=row.get('Goalie_Stats_Season')=='2026-2027' and source.endswith('name_and_stats') and match in {'','none','nan',key} and gp is not None and gp>0
    status=str(row.get('Opp_Goalie_Status') or 'Unconfirmed')
    values=[]
    for label,field,digits in [('GAA','Opp_GAA',2),('SV','Opp_SV',3),('GP','Opp_GP',0)]:
        v=number(row.get(field)) if valid else None
        values.append(f'{label} <b>{v:.{digits}f}</b>' if v is not None else label+' unavailable')
    return '<p class="pp-w-goalie" style="padding:6px 8px;background:#111c2f;border:1px solid #51617b;border-radius:5px">OPP GOALIE · <b>'+escape(name)+'</b> · '+escape(status)+' · '+' · '.join(values)+'</p>'

def return_panel(row,compact=True):
    f,p,s=row['full'],row['previous'],row['since_check']
    html=('<div class="wn-goal-returns"><div class="wn-gr-version">Moves V2 · wired Nov 19</div>'
          '<div class="wn-gr-box"><small>V2 BASELINE %</small>'+_metric(p)+'</div>'
          '<div class="wn-gr-box"><small>SINCE V2 CHECK %</small>'+_metric(s)+'</div>'
          '<div class="wn-gr-box"><small>TOTAL MOVE %</small>'+_metric(f)+'</div></div>')
    if not compact:
        h=row.get('health_since_check',{})
        if h.get('plays'):
            html+='<div class="pp-w-info"><b>RESEARCH HEALTH · Nov 1 onward</b><br>'+_metric(h)
            html+=f'<br>Average posted book BE {h["be"]:.1f}% · gap {h["gap"]:+.1f} pp · {h["players"]} players.</div>'
        html+=('<div class="pp-w-info">Baseline through Nov 17; V2 wired Nov 19. The research health subset is already included in the baseline. '
               'New V2 captures begin Nov 19. Zero new plays is not confirmation. Flat1u at saved ordinary pregame quotes; this is not the owner’s bet ledger.</div>')
    return html

def buff_badges(moves,row):
    from pp_cards import display_families,display_record,_rule_checks,onice_xg
    buffs=display_families([m for m in moves if m.get('pp_v2') and m.get('optional_buff') and card_qualified(m)])
    if not buffs:return ''
    result='<div class="pp-v2-buffs" style="display:flex;gap:5px;flex-wrap:wrap;margin:7px 0">'
    for m in buffs:
        s=display_record(m)
        title=f"Optional {m['name']}: {s['hits']}/{s['plays']} · {s['rate']:.1f}%. Exact full rule and actual values are in Inspect. Not an added probability bonus."
        result+='<span style="padding:5px 8px;border:1px solid #e5c16e;border-radius:5px;color:#ffe6a7;background:#493a2580;box-shadow:0 0 9px #dbb55735;font-size:12px;font-weight:800" title="'+escape(title,quote=True)+'">✦ '+escape(m['name'])+'</span>'
    return result+'</div>'
