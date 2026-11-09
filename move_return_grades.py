"""Approved return grades select a display lead; they never evaluate move eligibility."""
from html import escape
from functools import lru_cache
from pathlib import Path
import json,math

from goal_return_monitor import record_for as goal_record
from assist_return_monitor import record_for as assist_record
from points_return_monitor import record_for as points_record
from special_return_monitor import record_for as special_record

VERSION='return-grades-2026-11-09-v1'
PRIORITY={'S':4,'A':3,'B':2,'Review':1}

# User-approved disposition of the Nov9 Goal review; not a searched cutoff or
# permanent rule. Exact record fingerprints prevent stale labels after regrading.
GOAL_REVIEW={
    'goal_star:controlled':('A',False,(39,68,35.7795652173913),(14,32,4.489565217391305),86.91666666666667),
    'goal_star:measured':('A',False,(58,108,46.79348463131072),(19,44,5.289565217391305),64.84987408424908),
    'goal_star:pk_breach:reaper':('A',False,(19,31,20.47153846153846),(6,14,2.02),108.53846153846153),
    'goal_star:barrage:reaper':('S',True,(27,37,36.08348463131072),(7,13,5.6695652173913045),126.72466422466422),
}

def _reviewed_goal_grade(row,grade):
    review=GOAL_REVIEW.get(row.get('key'))
    if not review or grade['tier']!='S':return grade
    tier,early,full,recent,prior_roi=review
    for saved,expected in ((row.get('full',{}),full),(row.get('since_check',{}),recent)):
        actual=tuple(_number(saved.get(k)) for k in ('hits','plays','units'))
        if any(a is None or not math.isclose(a,e,rel_tol=0,abs_tol=1e-9) for a,e in zip(actual,expected)):return grade
    actual_roi=_number(row.get('previous',{}).get('roi'))
    if actual_roi is None or not math.isclose(actual_roi,prior_roi,rel_tol=0,abs_tol=1e-9):return grade
    reason=('Only 13 recent bets; strong returns, early confirmation.' if early else
            f'Recent ROI {grade["recent_roi"]:+.1f}% versus V1 {prior_roi:+.1f}%; grade capped at A after review.')
    return {**grade,'tier':tier,'watch':True,'watch_reason':reason,
            'review_early':early,'reviewed_on':'2026-11-09'}

@lru_cache(maxsize=1)
def _pp_kit_version():
    try:return json.loads(Path(__file__).with_name('pp_card_records.json').read_text(encoding='utf-8')).get('version')
    except (OSError,ValueError,TypeError):return None

def _number(value):
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (TypeError,ValueError):return None

def return_record(move):
    if not isinstance(move,dict):return None
    market=move.get('market')
    try:
        if market=='Goal':row=goal_record(move)
        elif market=='Assists':row=assist_record(move)
        elif market=='Points':row=points_record(move)
        elif market=='SOG':row=special_record(move,'SOG')
        elif all(k in move for k in ('entry','conditions','record')):
            row=special_record(move,'PPP')
            if row and row.get('kit_version')!=_pp_kit_version():return None
        else:return None
    except (TypeError,ValueError,KeyError):return None
    # Old-season fallback headlines cannot inherit a current-season performance grade.
    return row if row and not row.get('legacy') else None

def grade_record(row):
    if not isinstance(row,dict):return None
    full,recent=row.get('full',{}),row.get('since_check',{})
    plays,units,roi=(_number(full.get(k)) for k in ('plays','units','roi'))
    new_plays=_number(recent.get('plays'))
    new_units,new_roi=(_number(recent.get(k)) for k in ('units','roi'))
    if None in (plays,units,roi,new_plays) or plays<=0 or new_plays<0:return None
    if new_plays and (new_units is None or new_roi is None):return None
    if plays>=30 and units>=10 and roi>=25 and new_plays>=10 and new_roi>=10:
        tier='S'
    elif plays>=30 and units>=5 and roi>=10 and (new_plays<10 or new_roi>=-5):
        tier='A'
    elif roi>0:tier='B'
    else:tier='Review'
    grade=dict(tier=tier,watch=bool(new_plays and new_units<0),roi=roi,units=units,
                plays=int(plays),recent_plays=int(new_plays),recent_roi=new_roi,
                early=new_plays<10,record_key=row.get('key',''),version=VERSION)
    return _reviewed_goal_grade(row,grade)

def move_grade(move):return grade_record(return_record(move))

def stable_identity(move):
    return '|'.join(str(move.get(k,'')) for k in ('market','kit_version','id','move_id','name','rule'))

def grade_rank(move):
    """Higher is preferred; unknown identities retain the caller's historical fallback."""
    grade=move_grade(move)
    if grade is None:return None
    return (PRIORITY[grade['tier']],not grade['watch'],grade['roi'],grade['units'],
            grade['plays'],stable_identity(move))

def pp_move_rank(move):
    rank=grade_rank(move)
    if rank is not None:return rank
    record=move.get('record',{})
    return (0,False,_number(record.get('rate')) or -1,0,
            _number(record.get('plays')) or 0,stable_identity(move))

def grade_badges(move):
    if not isinstance(move,dict):return ''
    grade=move_grade(move)
    if grade is None:
        return '<span class="rg-badge rg-unrated" title="No matching comparable current-season return record">RETURNS PENDING</span>'
    tier=grade['tier'];label=tier+' TIER' if tier in ('S','A','B') else 'REVIEW'
    explanation=(f'Total {grade["roi"]:+.1f}% ROI · {grade["units"]:+.2f}u · {grade["plays"]} plays. '
                 +(f'Since check {grade["recent_roi"]:+.1f}% ROI · {grade["recent_plays"]} plays.'
                   if grade['recent_plays'] else 'No new graded plays.'))
    if grade['early']:explanation+=' Fewer than10 new plays; S confirmation pending.'
    if grade.get('watch_reason'):explanation+=' '+grade['watch_reason']
    watch_reason=grade.get('watch_reason','Recent graded plays have negative net units; the move remains eligible')
    watch_label='⚠ WATCH · EARLY' if grade.get('review_early') else '⚠ WATCH'
    watch=(f'<span class="rg-watch" title="{escape(watch_reason,quote=True)}">{watch_label}</span>' if grade['watch'] else '')
    return (f'<span class="rg-badges" data-return-grade="{escape(tier)}" '
            f'data-return-watch="{str(grade["watch"]).lower()}">'
            f'<span class="rg-badge rg-{tier.lower()}" title="{escape(explanation,quote=True)}">{label}</span>{watch}</span>')

GRADE_CSS='''
.rg-badges{display:inline-flex;gap:5px;align-items:center;flex-wrap:wrap;vertical-align:middle}
.rg-badge,.rg-watch{display:inline-block;font-family:inherit;font-size:11px;line-height:1.3;font-weight:900;letter-spacing:.04em;border-radius:4px;padding:4px 6px;white-space:nowrap}
.rg-s{background:#f3ce6c;color:#211b0c;border:1px solid #ffe8a7;box-shadow:0 0 9px #e1ae3738}
.rg-a{background:#8bdbc9;color:#102b29;border:1px solid #b4f3e4}
.rg-b{background:#b7c5d9;color:#172337;border:1px solid #d2dceb}
.rg-review,.rg-watch{background:#4d341d;color:#ffe0a0;border:1px solid #d49b4d}
.rg-unrated{background:#28354a;color:#c8d3e4;border:1px solid #66758b;font-size:9px}
.wn-move-title .rg-badges{margin-left:7px}.pp-w-move>.rg-badges{float:none;margin-left:7px}
.pp-w-heading>.rg-badges{margin-right:3px}.wn-grade-legend{color:#d4e0f0;font-size:11px;line-height:1.5;margin-top:4px}
.wn-unit--return-s,.pp-w-card.rg-card-s{border-left:4px solid #f3ce6c;background:linear-gradient(110deg,#3d3324,#1b2739 50%);box-shadow:inset 0 0 0 1px #f3ce6c40,0 0 10px #e1ae3714}
.wn-unit--return-a,.pp-w-card.rg-card-a{border-left:4px solid #8bdbc9;background:linear-gradient(110deg,#233c3c,#1b2739 50%)}
.wn-unit--return-b,.pp-w-card.rg-card-b{border-left:4px solid #b7c5d9;background:linear-gradient(110deg,#293443,#1b2739 50%)}
.wn-unit--return-review,.wn-unit--return-watch,.pp-w-card.rg-card-watch{border-left-color:#d49b4d}
'''

