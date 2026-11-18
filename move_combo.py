"""Compact, literal leading-move conditions; no eligibility or record logic."""
import re
from html import escape

LABELS = {
 'Conf_Assists':'Assist conf', 'Conf_Points':'Points conf', 'Conf_Goal':'Goal conf', 'Conf_SOG':'SOG conf',
 'iXA%':'Observed assist rank', 'iXG%':'Individual xG rank', 'PP_iXA60':'Weighted PP assists/60 (actual)',
 'i5v5_SOG60':'5v5 SOG/60', 'NHL_Team_PPO_PG':'Own PP opportunities/game',
 'i5v5_primaryAssists60':'5v5 primary assists/60', 'PP_ActualPoints60':'Actual PP points/60',
 'Current_Goal_Profile':'Goal profile', 'Log_L10_P_2plus_pct':'L10 2+ point frequency %',
 'NHL_Team_PK_Seconds_per_opportunity':'Own PK sec/opportunity', 'NHL_Team_SA_Venue_PG':'Own venue SOG allowed/game',
 'Goalie_GSAx_per_game':'Opp goalie GSAx/game', 'PP_OnIce_xG_per_game':'PP on-ice xG/game',
}
SIGNS={'>=':'≥','<=':'≤','==':'=','>':'>','<':'<'}

def combo_parts(move, labels=None):
    if not move:return []
    names={**(labels or {}),**LABELS}
    checks=move.get('entry',[])+move.get('conditions',[]) if isinstance(move.get('entry'),list) else None
    if checks is not None:
        parts=[f"{names.get(f,f.replace('_',' '))} {SIGNS.get(op,op)}{cut:g}" for f,op,cut in checks if f!='PPP line']
        return (['Green matrix'] if move.get('points15_v4') else [])+parts
    parts=[]
    for token in re.split(r'\s+\+\s+',str(move.get('rule') or move.get('condition') or '')):
        token=token.strip()
        if not token:continue
        if re.match(r'^(?:Assist|Assists|Goal|Points|SOG)\s*\d',token):continue # Already beside the posted market.
        matrix=re.fullmatch(r'Matrix_\w+\s+(Green|Yellow|Red)',token,re.I)
        if matrix:parts.append(matrix[1].capitalize()+' matrix');continue
        match=re.fullmatch(r'([\w%]+)\s*(>=|<=|==|>|<)\s*(-?\d+(?:\.\d+)?)',token)
        if match:
            field,op,cut=match.groups()
            label=names.get(field,field.replace('_',' '))
            parts.append(f'{label} {SIGNS[op]}{cut}')
        else:parts.append(token)
    return list(dict.fromkeys(parts))

def combo_html(move, labels=None):
    parts=combo_parts(move,labels)
    if not parts:return ''
    chips=''.join('<span style="display:inline-block;margin:2px 4px 2px 0;padding:2px 6px;background:#111b2b;border:1px solid #53647b;border-radius:4px">'+escape(p)+'</span>' for p in parts)
    return ('<div class="wn-move-combo" style="margin:8px 0 4px;font-size:11px;line-height:1.6;color:#e3edfc">'
            '<strong style="display:block;font-size:10px;color:#f2d388;letter-spacing:.05em">MOVE COMBO · ALL REQUIRED</strong>'+chips+'</div>')
