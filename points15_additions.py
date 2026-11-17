"""Approved additive Points1.5 V2 profiles; V1 rules remain independent."""
from pathlib import Path
import json,math,statistics
from datetime import date as calendar_date

def number(x):
    try:
        x=float(x)
        return x if math.isfinite(x) else None
    except (ValueError,TypeError):return None

def prepare(row):
    r=dict(row)
    # Rebuild only the actually used log measures, strictly before the slate.
    r['Pace_Gap_G10']=None;r['Log_L20_S_median']=None;r['Avg5_SOG']=None
    try:
        log=json.loads(r['Form_Log']) if isinstance(r.get('Form_Log'),str) else r['Form_Log']
        if isinstance(log,dict) and log.get('season')=='2026-27' and isinstance(log.get('games'),list) and all(isinstance(g,dict) and isinstance(g.get('d'),str) for g in log['games']):
            games=sorted([g for g in log['games'] if str(g['d'])<str(r['Date'])[:10]],key=lambda g:g['d'],reverse=True)
            if games and len({g['d'] for g in games})==len(games):
                shots=[number(g.get('s')) for g in games]
                if all(x is not None for x in shots[:20]):r['Log_L20_S_median']=float(statistics.median(shots[:20]))
                if all(x is not None for x in shots[:5]):r['Avg5_SOG']=sum(shots[:5])/len(shots[:5])
                expected=number(r.get('Exp_G_10'));g=games[:10]
                goals=[number(x.get('g')) for x in g]
                if len(g)>=5 and expected is not None and all(x is not None for x in goals):r['Pace_Gap_G10']=round(expected-10*sum(goals)/len(g),2)
    except (TypeError,ValueError,KeyError):pass
    r['Goalie_GSAA_per_game']=None
    gp=number(r.get('Opp_GP'));shots=number(r.get('Opp_Goalie_Shots_Faced'));ga=number(r.get('Opp_Goalie_GA'));league=number(r.get('Goalie_League_SV'))
    if r.get('Goalie_Stats_Season')=='2026-2027' and str(r.get('Opp_Goalie_Status','')).casefold()=='confirmed' and gp is not None and gp>0 and shots is not None and ga is not None and league is not None:
        r['Goalie_GSAA_per_game']=(shots*(1-league)-ga)/gp
    return r

def match(row,conditions):
    if number(row.get('Points_Line'))!=1.5 or row.get('Model_Stats_Season')!='2026-2027' or str(row.get('Matrix_Points','')).strip().casefold() not in ('green','🟢'):return False
    if number(row.get('Conf_Points')) is None or number(row['Conf_Points'])<82:return False
    for field,op,cut in conditions:
        if field in ['PP_Points60','PP_SOG60','PP_OnIce_xG_per_game']:
            source=row.get('PP_Points60_Source')
            if not isinstance(source,str) or not source.strip() or source.strip().casefold() in ('nan','none'):return False
        if field=='SOG_Opportunity_Base':
            if any(number(row.get(k)) is None for k in ['SOG_Opportunity_EV5','SOG_Opportunity_PP']):return False
        if field.startswith(('opp_','Team_','team_','Opp_PK_')) and row.get('Team_Stats_Season')!='2026-2027':return False
        if field.startswith('NHL_'):
            side='Opp' if field.startswith('NHL_Opp_') else 'Team'
            date=row.get('NHL_'+side+'_Latest_source_date')
            try:
                if not isinstance(date,str) or calendar_date.fromisoformat(date[:10])>=calendar_date.fromisoformat(str(row.get('Date',''))[:10]):return False
            except (ValueError,TypeError):return False
        v=number(row.get(field))
        if v is None or not (v>=float(cut)-1e-10 if op=='>=' else v<=float(cut)+1e-10):return False
    return True

KIT=json.loads(Path(__file__).with_name('points15_additions_rules.json').read_text(encoding='utf-8'))
VERSION=KIT['version']

def rule_text(spec):
    return 'Points1.5 + Green + Conf_Points >= 82 + '+' + '.join(f'{f} {op} {cut:g}' for f,op,cut in spec['conditions'])

def fired(row,catalog=None):
    catalog=catalog or KIT
    row=prepare(row)
    return [{**m,'rule':rule_text(m),'market':'Points','points_current':True,
             'points15_current':True,'points15_v2':True,'track':False,
             'later_wins':0,'later_picks':0,'kit_version':VERSION,'record_version':VERSION}
            for m in catalog['moves'] if match(row,m['conditions'])]
