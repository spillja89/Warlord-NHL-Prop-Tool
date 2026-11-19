"""SOG combined records and V3 provenance. Never changes eligibility."""
from html import escape

def record_for(move):
    from special_return_monitor import record_for as lookup
    return lookup(move,'SOG')

def display_stats(move):
    row=record_for(move)
    return row['full'] if row else {'hits':int(move.get('wins',0)),'plays':int(move.get('picks',0)),
                                    'be':move.get('sample_book_be_pct')}

def caption(move):
    row=record_for(move)
    if row and row.get('record_patch')=='sog-v3-2026-11-19':return 'V3 MOVE %'
    return 'V2 MOVE %' if move.get('sog_profile') else 'V1 MOVE %'

def return_panel(row,data,compact=True):
    from goal_return_monitor import render_return_panel,_metric
    result=render_return_panel(row,data,compact=compact).replace('vs V1','vs V3').replace('SINCE CHECK %','SINCE V3 CHECK %')
    if not compact:
        result+='<div class="wn-gr-note">SOG V3 wired Nov 19. Baseline through Nov 18; new tracking begins Nov 19. Research health is included in that baseline, not additional bets. Flat1u at saved pregame prices.</div>'
        h=row.get('health_since_check',{})
        if h.get('plays'):
            result+='<details class="wn-details"><summary>Research health · '+escape(row['health_since_from'])+' onward (included above)</summary><div class="wn-goal-returns"><div class="wn-gr-box"><small>RECENT HEALTH</small>'+_metric(h)+'</div>'
            result+=f'<div class="wn-gr-note">{h["hits"]}/{h["plays"]} · {h["hit_rate"]:.1f}% · BE {h["be"]:.1f}% · gap {h["gap"]:+.1f} pp · {h["players"]} players. Retrospective diagnostic, not independent validation.</div></div></details>'
        for old in row.get('archived_versions',[]):
            result+='<details class="wn-details"><summary>Archived '+escape(old.get('move_version_label','previous record'))+' · original and prior check</summary><div class="wn-goal-returns">'
            for label,key in [('ORIGINAL RECORD','previous'),('PRIOR SINCE CHECK','since_check'),('COMBINED AT PATCH','full')]:
                result+='<div class="wn-gr-box"><small>'+label+'</small>'+_metric(old[key])+'</div>'
            result+='<div class="wn-gr-note">Historical provenance preserved. Overlapping replays are not additional bets.</div></div></details>'
    return result
