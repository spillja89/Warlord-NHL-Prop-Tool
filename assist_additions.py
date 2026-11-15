"""Approved additive Assist V2 profiles. Existing V1 rules remain independent."""
from pathlib import Path
from datetime import date
import json, math

KIT=json.loads(Path(__file__).with_name('assist_additions_rules.json').read_text(encoding='utf-8'))
VERSION=KIT['version']
ACCENTS={'Runesmith':'#79afff','Oracle':'#c9b6ff','Stormcaller':'#63e1cf'}
ICONS={'Runesmith':'runic_infusion.svg','Oracle':'odins_arcane_orb.svg','Stormcaller':'arcane_channel_ii.svg'}
LABELS={'i5v5_SOG60':'Individual 5v5 SOG/60','PP_ActualPoints60':'Actual PP points/60',
 'Current_Goal_Profile':'Goal-profile score','Log_L10_P_2plus_pct':'L10 multi-point-game %',
 'iXA%':'Observed assist percentile','TOI_per_game':'Ice time min/game',
 'NHL_Team_PK_Seconds_per_opportunity':'Own PK seconds/opportunity',
 'NHL_Team_SA_Venue_PG':'Own venue SOG allowed/game','NHL_Team_PPO_PG':'Own PPO/game',
 'NHL_Team_PP_G60':'Own actual PP goals/60','Goalie_GSAx_per_game':'Opp goalie GSAx/game',
 'NHL_Opp_Goal_conversion_pct':'Opponent shooting conversion %','Conf_Assists':'Assist confidence'}

def number(value):
    try:
        value=float(value)
        return value if math.isfinite(value) else None
    except (ValueError,TypeError):return None

def goalie_valid(row):
    gp=number(row.get('Opp_GP'))
    return (row.get('Goalie_Stats_Season')=='2026-2027'
            and str(row.get('Opp_Goalie_Status','')).strip().casefold() in ('confirmed','official')
            and str(row.get('Opp_Goalie_Source','')).endswith('name_and_stats')
            and gp is not None and gp>0)

def prepare(row):
    """Rebuild only used derivatives, preserving unknown values as unknown."""
    r=dict(row)
    xg,toi=number(r.get('iXG%')),number(r.get('TOI_Pct'))
    r['Current_Goal_Profile']=round(min(100,max(0,.82*xg+.18*toi))) if None not in (xg,toi) else None
    r['Log_L10_P_2plus_pct']=None
    try:
        log=json.loads(r['Form_Log']) if isinstance(r.get('Form_Log'),str) else r['Form_Log']
        before=date.fromisoformat(str(r.get('Date',''))[:10])
        if isinstance(log,dict) and log.get('season')=='2026-27':
            games=sorted([g for g in log['games'] if date.fromisoformat(str(g['d'])[:10])<before],key=lambda g:g['d'],reverse=True)
            if games and len(games)==len({g['d'] for g in games}):
                values=[(number(g.get('g')),number(g.get('a'))) for g in games[:10]]
                if all(None not in v for v in values):
                    r['Log_L10_P_2plus_pct']=100*sum(g+a>=2 for g,a in values)/len(values)
    except (KeyError,TypeError,ValueError):pass
    r['Goalie_GSAx_per_game']=None
    ga,xga=number(r.get('Opp_Goalie_GA')),number(r.get('Opp_Goalie_xGA'))
    if goalie_valid(r) and None not in (ga,xga):r['Goalie_GSAx_per_game']=(xga-ga)/number(r['Opp_GP'])
    return r

def match(row,conditions):
    line=number(row.get('Assists_Line'))
    if line is None:line=number(row.get('Line_Assists'))
    if line!=.5 or row.get('Model_Stats_Season')!='2026-2027':return False
    if str(row.get('Matrix_Assists',row.get('Matrix_A',''))).strip().casefold() not in ('green','🟢'):return False
    conf=number(row.get('Conf_Assists'))
    if conf is None or conf<82:return False
    for field,op,cut in conditions:
        if field=='i5v5_SOG60' and number(row.get('EV5_TOI_per_game')) is None:return False
        if field=='PP_ActualPoints60':
            source=row.get('PP_Points60_Source')
            if not isinstance(source,str) or source.strip().casefold() in ('','nan','none'):return False
        if field.startswith('NHL_'):
            side='Opp' if field.startswith('NHL_Opp_') else 'Team'
            try:
                latest=date.fromisoformat(str(row.get('NHL_'+side+'_Latest_source_date',''))[:10])
                if latest>=date.fromisoformat(str(row.get('Date',''))[:10]):return False
            except (ValueError,TypeError):return False
        value=number(row.get(field))
        if value is None or not (value>=cut-1e-10 if op=='>=' else value<=cut+1e-10):return False
    return True

def rule_text(spec):
    return 'Assist0.5 + Matrix_Assists Green + Conf_Assists >=82' + ''.join(f' + {f} {op} {cut:g}' for f,op,cut in spec['conditions'])

def fired(row):
    row=prepare(row);result=[]
    for spec in KIT['moves']:
        if not match(row,spec['conditions']):continue
        variant=spec['route_variant'];full=spec['full']
        result.append({**spec,'move_id':spec['id'],'market':'Assists','rule':rule_text(spec),
          'wins':full['hits'],'picks':full['plays'],'later_wins':0,'later_picks':0,
          'sample_players':full['players'],'unique_players':full['players'],'sample_dates':full['dates'],
          'sample_book_be_pct':full['be'],'sample_from':'2026-10-13','sample_through':'2026-11-13',
          'sample_season':'2026-27','kit_version':VERSION,'record_version':VERSION,
          'assist_v2':True,'track':False,'icon':ICONS[variant],'route_accent':ACCENTS[variant],
          'level_label':spec['display_level'],'display_priority':4,
          'environment_enhanced':any(f.startswith(('NHL_','Goalie_')) for f,_,_ in spec['conditions'])})
    return result
