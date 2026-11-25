"""Display recent and season PK together, using frozen official counts only."""
from fractions import Fraction
from html import escape
import math

VERSION='pp-pk-window-display-2026-11-25'
PERIODS=('L5','L10','season')
COLUMNS=tuple(k for w in PERIODS for k in (f'Opp PK {w} %',f'Opp PK {w} record',f'Opp PK {w} opportunities/g'))
COLUMN_HELP={f'Opp PK {w} %':f'Opponent penalty kill over {"the last five completed games" if w=="L5" else "the last ten completed games" if w=="L10" else "the current season before this slate"}. Kills divided by total opposing PP opportunities. Higher means a stronger kill, which is tougher for your PP pick. Counts and opportunities/game appear beneath. Recent arrows compare with season; they are context, not new move rules.' for w in PERIODS}

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
    return '<div class="pp-pk-window'+css+'"><b>'+f'{value:.1f}%</b><small>'+detail+'</small>'+(f'<em>{escape(label)}</em>' if label else '')+'</div>'

CSS='''
.pp-scout .pp-pk-window{min-width:125px;padding:5px 7px;border-left:3px solid #73869c;background:#edf2f7;border-radius:4px;line-height:1.5}
.pp-scout .pp-pk-window>b{font-size:15px;color:#14283f}.pp-scout .pp-pk-window>small{display:block;color:#40556c;font-size:10px}
.pp-scout .pp-pk-window>em{display:block;font-size:10px;font-style:normal;font-weight:800}
.pp-scout .pp-pk-window--stronger{border-color:#d38b28;background:#fff1d8}.pp-scout .pp-pk-window--stronger>em{color:#86500b}
.pp-scout .pp-pk-window--weaker{border-color:#24976f;background:#e1f5eb}.pp-scout .pp-pk-window--weaker>em{color:#126a4b}
'''
