"""Display recent and season PK together, using frozen official counts only."""
from fractions import Fraction
from html import escape
from functools import lru_cache
from pathlib import Path
import json
import math

VERSION='pp-pk-window-display-2026-11-25'
PERIODS=('L5','L10','season')
COLUMNS=tuple(k for w in PERIODS for k in (f'Opp PK {w} %',f'Opp PK {w} record',f'Opp PK {w} opportunities/g'))
COLUMN_HELP={f'Opp PK {w} %':f'Opponent penalty kill over {"the last five completed games" if w=="L5" else "the last ten completed games" if w=="L10" else "the current season before this slate"}. Kills divided by total opposing PP opportunities. Higher means a stronger kill, which is tougher for your PP pick. Counts and opportunities/game appear beneath. Arrows compare recent PK with season. A separate PPP flag compares historical player hit rate with the sample average book break-even; it is not tonight\'s player probability or a new move rule.' for w in PERIODS}

@lru_cache(maxsize=1)
def research_catalog():
    return json.loads(Path(__file__).with_name('pp_pk_research.json').read_text(encoding='utf-8'))

def research_flag(row,window):
    value=num(row.get(f'Opp PK {window} %'))
    if value is None or not 0<=value<=100:return ''
    catalog=research_catalog()
    for record in catalog['records']:
        if record['field']!=f'Opp_{window}_Success_pct':continue
        cutoff=record['cutoff'];direction=record['direction']
        if not (value>=cutoff if direction=='>=' else value<=cutoff):continue
        full,recent=record['full'],record['recent']
        warning=full['gap']<0 and recent['gap']<0
        positive=full['gap']>0 and recent['gap']>0
        if not (warning or positive):continue
        label=f"{recent['gap']:+.1f} points"
        title=(f"Recent PPP hit rate {recent['rate']:.1f}% vs sample BE {recent['BE']:.1f}%: {recent['gap']:+.1f} percentage points. "
               +catalog['parent']+'. Recent is included in full; not an independent sample. '
               'Sample BE is the average posted-price break-even in this historical group, not tonight\'s price. '
               'These overlapping research flags are not independent signals or entry gates.')
        rows=''
        for name,s in [('Full',full),('Recent',recent)]:
            rows+=(f'<div class="pp-pk-research-period"><b>{name} · {s["hits"]}/{s["plays"]} · {s["rate"]:.1f}%</b>'
                   f'<span>Sample BE {s["BE"]:.1f}% · gap {s["gap"]:+.1f} pp · {s["players"]} players</span>'
                   f'<span>{s["units"]:+.2f}u · {s["ROI"]:+.1f}% ROI · 1u per bet</span></div>')
        return ('<div class="pp-pk-research pp-pk-research--'+('warning' if warning else 'positive')+'" title="'+escape(title,quote=True)+'">'
                '<strong>'+label+'</strong>'
                '<details class="pp-pk-research-details"><summary>PPP research record</summary><div>'
                f'<p>Opp {window} PK {escape(direction)}{cutoff:g}% · research through Nov 24. Signed badge is recent hit rate minus sample BE.</p>'
                '<p>All priced player PPP over 0.5. No PP1, xG or Layer 1 filter. Historical matchup sample; not tonight’s player probability.</p>'
                +rows+
                '<p>Full Oct 13–Nov 24 · recent Nov 1–Nov 24, included in full. Overlapping flags must not be added together.</p>'
                '</div></details></div>')
    return ''

def num(x):
    try:return float(x) if math.isfinite(float(x)) else None
    except (ValueError,TypeError):return None

def integer(x):
    v=num(x)
    return int(round(v)) if v is not None and v>=0 and abs(v-round(v))<1e-6 else None

def columns(row):
    out={k:None for k in COLUMNS};day=str(row.get('Date') or '')[:10]
    if row.get('Model_Stats_Season')!='2026-2027' or row.get('Team_Stats_Season')!='2026-2027':return out
    def put(window,opps,goals,gp):
        if opps is None or goals is None or gp is None or gp<=0 or opps<=0 or not 0<=goals<=opps:return
        out[f'Opp PK {window} %']=100*(opps-goals)/opps
        out[f'Opp PK {window} record']=f'{opps-goals}/{opps} killed · {goals} GA'
        out[f'Opp PK {window} opportunities/g']=opps/gp
    # These frozen complete-window reports are built from the same official games.
    # Require matching as-of and last-game dates before joining their denominators.
    v3last=str(row.get('PP_V3_Opp_Last_Date') or '')[:10];slast=str(row.get('SOGV3_Opp_Last_Date') or '')[:10]
    common=(str(row.get('PP_V3_Context_AsOf'))==day and str(row.get('SOGV3_Context_AsOf'))==day and v3last==slast and len(v3last)==10 and v3last<day)
    if common:
        minutes=num(row.get('SOGV3_NHL_Opp_PK_L5_Time_PG'));seconds=num(row.get('PPV3_Opp_PK_L5_Seconds_per_O'));pk=num(row.get('NHL_Opp_PK_L5_Success_pct'))
        if minutes is not None and seconds is not None and seconds>0 and pk is not None and 0<=pk<=100:
            opps=integer(minutes*60*5/seconds)
            goals=integer(opps*(1-pk/100)) if opps is not None else None
            put('L5',opps,goals,5)
    l10last=str(row.get('PP_V2_Opp_L10_Last_Date') or '')[:10]
    if str(row.get('PP_V2_Context_AsOf'))==day and str(row.get('SOGV3_Context_AsOf'))==day and l10last==slast and len(l10last)==10 and l10last<day:
        opg=num(row.get('SOGV3_NHL_Opp_PK_L10_O_PG'));gpg=num(row.get('Official_Opp_PK_L10_Goals_PG'))
        put('L10',integer(opg*10) if opg is not None else None,integer(gpg*10) if gpg is not None else None,10)
    gp=integer(row.get('Opp_PKO_GP'));opg=num(row.get('NHL_Opp_PKO_PG'));gpg=num(row.get('NHL_Opp_PK_Goals_PG'))
    if gp is not None:
        put('season',integer(opg*gp) if opg is not None else None,integer(gpg*gp) if gpg is not None else None,gp)
    if out['Opp PK season %'] is None:
        measured=num(row.get('NHL_Opp_PK_Success_pct'))
        if measured is not None and 0<=measured<=100:out['Opp PK season %']=measured
    return out

def cell(row,window):
    value=num(row.get(f'Opp PK {window} %'));season=num(row.get('Opp PK season %'))
    if value is None:return '—'
    label='';css=''
    if window!='season' and season is not None:
        change=value-season
        if change>1e-6:label=f'↑ stronger kill · +{change:.1f} pp';css=' pp-pk-window--stronger'
        elif change < -1e-6:label=f'↓ weaker kill · {change:.1f} pp';css=' pp-pk-window--weaker'
        else:label='Same as season'
    record=row.get(f'Opp PK {window} record') or ''
    opg=num(row.get(f'Opp PK {window} opportunities/g'))
    detail=escape(record)+(f'<br>{opg:.2f} opp/g' if opg is not None else '')
    return '<div class="pp-pk-window'+css+'"><b>'+f'{value:.1f}%</b><small>'+detail+'</small>'+(f'<em>{escape(label)}</em>' if label else '')+research_flag(row,window)+'</div>'

CSS='''
.pp-scout .pp-pk-window{min-width:125px;padding:5px 7px;border-left:3px solid #73869c;background:#edf2f7;border-radius:4px;line-height:1.5}
.pp-scout .pp-pk-window>b{font-size:15px;color:#14283f}.pp-scout .pp-pk-window>small{display:block;color:#40556c;font-size:10px}
.pp-scout .pp-pk-window>em{display:block;font-size:10px;font-style:normal;font-weight:800}
.pp-scout .pp-pk-window--stronger{border-color:#d38b28;background:#fff1d8}.pp-scout .pp-pk-window--stronger>em{color:#86500b}
.pp-scout .pp-pk-window--weaker{border-color:#24976f;background:#e1f5eb}.pp-scout .pp-pk-window--weaker>em{color:#126a4b}
.pp-scout .pp-pk-research{margin-top:6px;padding:5px 7px;border:1px solid;border-radius:5px;font-size:10px;line-height:1.5;white-space:normal}
.pp-scout .pp-pk-research>strong,.pp-scout .pp-pk-research>span,.pp-scout .pp-pk-research>small{display:block}
.pp-scout .pp-pk-research--warning{background:#491b22;border-color:#ec7788;color:#fff0f2;box-shadow:0 0 7px #c1354933}
.pp-scout .pp-pk-research--warning>strong{color:#ffafbb;font-size:11px}
.pp-scout .pp-pk-research--positive{background:#103f35;border-color:#67cfaa;color:#edfff6}
.pp-scout .pp-pk-research--positive>strong{color:#9af5ca;font-size:11px}
.pp-scout .pp-pk-research>small{font-size:9px;opacity:.85}
.pp-scout .pp-pk-research-details>summary{cursor:pointer;text-decoration:underline dotted;color:inherit;font-size:10px}
.pp-scout .pp-pk-research-details>div{max-width:540px;overflow:auto;padding-top:4px}
.pp-scout .pp-pk-research-details p{font-size:10px;margin:4px 0}
.pp-scout .pp-pk-research-period{border-top:1px solid #ffffff33;padding:5px 0}
.pp-scout .pp-pk-research-period>b,.pp-scout .pp-pk-research-period>span{display:block}
'''
