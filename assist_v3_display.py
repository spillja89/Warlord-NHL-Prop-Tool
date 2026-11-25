"""Distinguish the main Assist move, team buffs and goalie buffs visually."""
from html import escape
from base64 import b64encode

FLAIR={
 'Countermarch':('#ffa594','M4 7h13l-4-4m7 14H7l4 4M17 7l3 3M7 17l-3-3','Crossed counterattack banners'),
 'Siegebrand':('#ffb45b','M13 3c1 5-5 5-2 9 1-2 3-2 3-5 4 4 6 7 4 11-3 6-13 3-12-3 0-3 3-5 4-7-1 5 3 5 3 1','Burning siege brand'),
 'Rift Breach':('#c1a4ff','M9 3 4 7v10l5 4m6-18 5 4v10l-5 4M13 3l-3 7 5 3-4 8','Fractured arcane portal'),
 'Stormfront':('#83ddff','M6 15a4 4 0 0 1-1-8 6 6 0 0 1 11-1 4 4 0 0 1 2 8M13 10l-4 7h4l-2 5 6-8h-4z','Storm cloud and lightning'),
 'Counterspell Echo':('#ffb5e2','M8 7 3 12l5 5M16 7l5 5-5 5M10 4h4v4h-4zm0 12h4v4h-4zM8 12h8','Echoing counterspell rune'),
 'Mana Spring':('#83f0ce','M12 3C9 8 5 11 5 15a7 7 0 0 0 14 0c0-4-4-7-7-12ZM12 10v9m-4-5h8','Renewing mana crystal'),
 'Lunar Haven':('#c5d6ff','M17 3a9 9 0 1 0 4 14A9 9 0 0 1 17 3ZM18 7v5m-2-2h4','Sheltering lunar crescent'),
 'Iron Veil':('#c8d4e4','M12 3 3 7v6c0 5 9 9 9 9s9-4 9-9V7ZM7 8v7m5-9v12m5-10v7','Layered iron ward'),
 'Sundered Ward':('#ffdc83','M12 3 3 7v6c0 5 9 9 9 9s9-4 9-9V7ZM13 5l-4 6 5 2-3 7M18 3v5m-2-2h4','Cracked goalie shield'),
}

def _flair(move):
    family=move.get('display_family') or move.get('family')
    accent,path,title=FLAIR.get(family,('#a6e5e0','M12 3 3 12l9 9 9-9ZM7 12h10','Arcane matchup rune'))
    style=f'--buff-accent:{accent};--buff-soft:{accent}18;--buff-dim:{accent}66'
    # Streamlit sanitizes inline SVG out of st.html; a fixed local data image
    # preserves these code-owned emblems without loading any outside asset.
    svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="23" height="23" fill="none" stroke="'+accent+'" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="'+path+'"/></svg>'
    emblem='<span class="wn-assist-emblem" aria-hidden="true"><img alt="" width="23" height="23" src="data:image/svg+xml;base64,'+b64encode(svg.encode()).decode()+'"></span>'
    return style,emblem,title

def buff_type(move):
    condition=(move or {}).get('additional_condition') or []
    return 'goalie' if condition and str(condition[0]).startswith('Goalie_') else 'team'

def _identity(move):
    return ((move or {}).get('kit_version'), (move or {}).get('move_id') or (move or {}).get('id') or (move or {}).get('name'))

def _condition_text(move):
    from assist_routes_v3 import LABELS
    c=move.get('additional_condition') or []
    if len(c)!=3:return str(c)
    field,op,value=c
    label=LABELS.get(field,str(field).replace('_',' '))
    sign={'>=':'≥','<=':'≤','==':'='}.get(op,op)
    value=f'{value:g}' if isinstance(value,(float,int)) else str(value)
    return f'{label} {sign}{value}'

def lead_context(move):
    """Display only: the leading record and its selected rule stay identical."""
    if not move:return ''
    if move.get('assist_v3') and move.get('optional_buff'):
        kind=buff_type(move)
        label='ASSIST MOVE + GOALIE BUFF' if kind=='goalie' else 'ASSIST MOVE + TEAM BUFF'
        parent=move.get('parent_name') or 'Assist baseline'
        style,emblem,title=_flair(move)
        return ('<div class="wn-assist-lead-kind wn-assist-lead-kind--'+kind+'" style="'+style+'" title="'+escape(title,quote=True)+'">'+emblem+label+'</div>'
                '<div class="wn-assist-base-route">Base move: <b>'+escape(parent)+'</b>'
                '<span>Added buff: '+escape(_condition_text(move))+'</span></div>')
    return '<div class="wn-assist-lead-kind">ASSIST MOVE</div>'

def badges(moves, lead=None):
    from move_return_grades import grade_rank
    from assist_return_monitor import display_stats
    groups={}
    for m in moves:
        if not m.get('assist_v3') or not m.get('optional_buff'):continue
        family=m['display_family'];prior=groups.get(family)
        if prior is None or (grade_rank(m) or ())>(grade_rank(prior) or ()):groups[family]=m
    if not groups:return ''
    items={'team':[],'goalie':[]}
    for m in groups.values():
        # The leading buffed route is already named above with its own record.
        if lead and _identity(m)==_identity(lead):continue
        s=display_stats(m);label=m['name'];parent=m.get('parent_name') or 'Original .5 Green Conf81 baseline'
        kind=buff_type(m)
        tip=parent+' + '+_condition_text(m)+'. Optional buff on this specific route; this is the combined historical record, not a bonus added to the main percentage. Full/recent evidence in Inspect.'
        cls='wn-assist-buff wn-assist-buff--'+kind+(' wn-goalie-buff' if kind=='goalie' else '')
        style,emblem,flair_title=_flair(m)
        items[kind].append('<span class="'+cls+'" style="'+style+'" title="'+escape(flair_title+'. '+tip,quote=True)+'">'+emblem+'<span>'+escape(label)+f"</span> <b>{s['hits']}/{s['plays']} · {100*s['hits']/s['plays']:.1f}%</b></span>")
    sections=[]
    for kind,heading in [('team','OTHER TEAM MATCHUP BUFFS'),('goalie','OTHER GOALIE BUFFS')]:
        if items[kind]:sections.append('<div class="wn-assist-buff-group wn-assist-buff-group--'+kind+'"><span class="wn-assist-buff-heading">'+heading+'</span>'+''.join(items[kind])+'</div>')
    return '<div class="wn-v3-buffs">'+''.join(sections)+'</div>' if sections else ''

CSS="""
.wn-lane--support .wn-assist-lead-kind{display:flex;align-items:center;gap:7px;font-size:10px;font-weight:900;letter-spacing:.09em;color:#c9b9ff;margin:7px 0 2px}
.wn-lane--support .wn-assist-lead-kind--team{color:#80e2dc}.wn-lane--support .wn-assist-lead-kind--goalie{color:#ffe3a0}
.wn-lane--support .wn-assist-base-route{font-size:12px;line-height:1.5;color:#dbe6f8;margin:3px 0 6px}
.wn-lane--support .wn-assist-base-route b{color:#fff}.wn-lane--support .wn-assist-base-route span{display:block;font-size:11px;color:#c2d3e8}
.wn-lane--support .wn-assist-buff-group{display:flex;flex-wrap:wrap;align-items:center;gap:5px;margin:7px 0 3px;padding:7px 8px;border:1px solid #41646c;border-radius:6px;background:#10262e}
.wn-lane--support .wn-assist-buff-group--goalie{border-color:#75618d;background:#231d34}
.wn-lane--support .wn-assist-buff-heading{flex:0 0 100%;font-size:9px;font-weight:900;letter-spacing:.08em;color:#8ee9e0}
.wn-lane--support .wn-assist-buff-group--goalie .wn-assist-buff-heading{color:#ffe3a0}
.wn-lane--support .wn-assist-buff{display:inline-flex;align-items:center;flex-wrap:wrap;gap:6px;line-height:1.5;font-size:11px;font-weight:750;padding:4px 7px;border-radius:4px}
.wn-lane--support .wn-assist-buff--team{color:var(--buff-accent);border:1px solid var(--buff-dim);background:linear-gradient(110deg,var(--buff-soft),#12202b);box-shadow:none;animation:none}
.wn-lane--support .wn-assist-buff--team b{color:#fff}.wn-lane--support .wn-assist-buff--goalie{margin:0;letter-spacing:0}
.wn-lane--support .wn-assist-emblem{display:inline-flex;align-items:center;justify-content:center;flex:0 0 29px;width:29px;height:29px;border:1px solid var(--buff-dim);border-radius:7px;background:var(--buff-soft);color:var(--buff-accent);box-shadow:inset 0 0 8px var(--buff-soft);padding:2px}
.wn-lane--support .wn-assist-emblem img{display:block;width:23px;height:23px;object-fit:contain}
.wn-lane--support .wn-assist-buff .wn-assist-emblem{font-size:inherit;text-shadow:none}
.wn-lane--support .wn-assist-buff:hover{border-color:var(--buff-accent);box-shadow:0 0 10px var(--buff-soft)}
.wn-lane--support .wn-assist-buff--goalie>span:not(.wn-assist-emblem){font-size:11px;text-shadow:none}
"""
