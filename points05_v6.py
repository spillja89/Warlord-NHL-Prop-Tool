"""Approved Points0.5 V6 player routes; independent optional goalie context."""
from functools import lru_cache
from pathlib import Path
from html import escape
import json,math,statistics

VERSION='points05-v6-2026-11-18'
ENTRY='Points0.5+Matrix_PointsGreen+Conf_Points>=77'
OWNED=('Player_L20_PPP_fraction_pct','Player_L20_PPP_per60','Player_L20_G_total','Player_L10_TOI_sd','Player_L5_P_2plus_pct','Player_Pace_Gap_A10','Current_SOG_Profile')
LABELS={'Player_L20_PPP_fraction_pct':'L20 points earned on PP %','Player_L20_PPP_per60':'L20 PPP/60 total ice time','Player_L20_G_total':'L20 goals','Player_L10_TOI_sd':'L10 ice-time variation (min)','Player_L5_P_2plus_pct':'L5 multi-point games %','Player_Pace_Gap_A10':'Assist pace gap','Current_SOG_Profile':'Current SOG confidence','PP_ActualPoints60':'Actual PP points/60','EV5_OZS_Pct':'5v5 offensive-zone starts %','PP_Points60':'PP xG/assist proxy /60'}
TOOLTIPS={'Player_L20_PPP_fraction_pct':'Percentage of the player’s points earned on the power play, using up to20 prior games. Not PP ice-time share.', 'Player_L20_PPP_per60':'Power-play points per60 minutes of TOTAL player ice time, using up to20 prior games.', 'Player_L10_TOI_sd':'How much the player’s ice time varies from game to game over up to10 prior games. In minutes; 2.70 means2 minutes42 seconds of standard deviation, not average ice time.', 'Player_Pace_Gap_A10':'Saved expected10-game assist pace minus the player’s actual recent assist pace. Positive means actual pace was lower. Not a guaranteed rebound.', 'Current_SOG_Profile':'The existing current shot-confidence formula, consistently calculated. Includes player ice time, shot share, opponent shot volume and the existing drought adjustment.'}
def number(x):
 try:
  v=float(x);return v if math.isfinite(v) else None
 except (ValueError,TypeError):return None
@lru_cache(maxsize=1)
def catalog():return json.loads(Path(__file__).with_name('points05_v6_rules.json').read_text(encoding='utf-8'))
@lru_cache(maxsize=4096)
def _log_values(raw,before):
 out={f:None for f in OWNED if f!='Current_SOG_Profile'}
 try:
  obj=json.loads(raw)
  if not isinstance(obj,dict) or obj.get('season')!='2026-27' or not isinstance(obj.get('games'),list):return out
  games=sorted([g for g in obj['games'] if isinstance(g,dict) and isinstance(g.get('d'),str) and g['d']<before],key=lambda g:g['d'],reverse=True)
  if not games or len({g['d'] for g in games})!=len(games):return out
  def toi(g):
   bits=str(g.get('toi','')).split(':')
   if len(bits)!=2 or not all(x.isdigit() for x in bits) or int(bits[1])>=60:return None
   return int(bits[0])+int(bits[1])/60
  last20=games[:20];last10=games[:10];last5=games[:5]
  gg=[number(g.get('g')) for g in last20];aa=[number(g.get('a')) for g in last20];pp=[number(g.get('pp')) for g in last20]
  if all(v is not None and v>=0 for v in gg+aa):
   points=sum(gg)+sum(aa);out['Player_L20_G_total']=sum(gg)
   if all(v is not None and v>=0 for v in pp) and all(p<=g+a for p,g,a in zip(pp,gg,aa)):
    out['Player_L20_PPP_fraction_pct']=100*sum(pp)/points if points>0 else None
    ts=[toi(g) for g in last20]
    if all(t is not None and t>0 for t in ts):out['Player_L20_PPP_per60']=60*sum(pp)/sum(ts)
  ts=[toi(g) for g in last10]
  if all(t is not None and t>0 for t in ts):out['Player_L10_TOI_sd']=statistics.pstdev(ts)
  p5=[(number(g.get('g')),number(g.get('a'))) for g in last5]
  if all(a is not None and b is not None and a>=0 and b>=0 for a,b in p5):out['Player_L5_P_2plus_pct']=100*sum(a+b>=2 for a,b in p5)/len(p5)
  a10=[number(g.get('a')) for g in last10]
  if len(games)>=5 and all(a is not None and a>=0 for a in a10):out['_actual_assist_pace10']=10*sum(a10)/len(a10)
 except (ValueError,TypeError,KeyError,ZeroDivisionError):pass
 return out
def prepare(row):
 out=dict(row)
 for f in OWNED:out[f]=None
 raw=out.get('Form_Log');before=str(out.get('Date',''))[:10]
 if isinstance(raw,dict):raw=json.dumps(raw,sort_keys=True)
 if isinstance(raw,str):out.update(_log_values(raw,before))
 actual=out.pop('_actual_assist_pace10',None);expected=number(out.get('Exp_A_10_Gap_Basis'))
 if actual is not None and expected is not None:out['Player_Pace_Gap_A10']=round(expected-actual,2)
 from sog_confidence import sog_profile_confidence
 out['Current_SOG_Profile']=sog_profile_confidence(out.get('TOI_Pct'),out.get('Player_5v5_SOG_Share'),out.get('Opp_SOG_Against_L10'),out.get('Talent_Tier'),out.get('Drought_SOG'))
 return out
def value(row,field):
 if field in {'Assist_Dagger','PP_Points60','PP_ActualPoints60'} and str(row.get('PP_Points60_Source') or '').strip().casefold() in {'','none','nan','<na>'}:return None
 return number(row.get(field))
def goalie_valid(row):
 n=number(row.get('Opp_GAA'));gp=number(row.get('Opp_GP'));name=str(row.get('Opp_Goalie') or '').strip().casefold()
 return bool(row.get('Goalie_Stats_Season')=='2026-2027' and str(row.get('Opp_Goalie_Source','')).endswith('name_and_stats') and name not in {'','none','nan','unknown'} and str(row.get('Opp_Goalie_Status','')).casefold() in {'confirmed','official'} and n is not None and 0<=n<=15 and gp is not None and gp>0)
def moves(row):
 if number(row.get('Points_Line'))!=.5 or row.get('Model_Stats_Season')!='2026-2027' or str(row.get('Matrix_Points','')).strip().casefold() not in {'green','🟢'} or (number(row.get('Conf_Points')) or 0)<77:return []
 row=prepare(row);result=[]
 for s in catalog()['moves']:
  if not all(value(row,f) is not None and (value(row,f)>=c-1e-10 if op=='>=' else value(row,f)<=c+1e-10) for f,op,c in s['conditions']):continue
  rule=ENTRY+''.join(f' + {f} {op} {c:g}' for f,op,c in s['conditions'])
  result.append({**s,'rule':rule,'market':'Points','points_current':True,'points05_v6':True,'kit_version':VERSION,'record_version':VERSION,'sample_season':'2026-27','sample_from':'2026-10-13','sample_through':'2026-11-16','later_wins':0,'later_picks':0,'track':False})
 if goalie_valid(row):
  for s in catalog().get('goalie_tracking',[]):
   if not all(value(row,f) is not None and (value(row,f)>=c-1e-10 if op=='>=' else value(row,f)<=c+1e-10) for f,op,c in s['conditions']):continue
   result.append({**s,'rule':s['rule'],'market':'Points','points_current':True,'points05_v6':True,'points05_aegis_context':True,'kit_version':VERSION,'record_version':VERSION,'sample_season':'2026-27','sample_from':'2026-10-13','sample_through':'2026-11-16','later_wins':0,'later_picks':0,'track':False})
 return result
def goalie_context(row,moves):
 n=number(row.get('Opp_GAA'));gp=number(row.get('Opp_GP'));status=str(row.get('Opp_Goalie_Status','')).casefold();name=str(row.get('Opp_Goalie') or '').strip()
 valid=(row.get('Goalie_Stats_Season')=='2026-2027' and str(row.get('Opp_Goalie_Source','')).endswith('name_and_stats') and name.casefold() not in {'','none','nan','unknown'} and status in {'confirmed','official'} and n is not None and 0<=n<=15 and gp is not None and gp>0)
 if not valid or n<2.95 or not moves:return None
 r=prepare(row);roots=[]
 for s in catalog()['goalie_buff']['records']:
  if all(value(r,f) is not None and (value(r,f)>=c-1e-10 if op=='>=' else value(r,f)<=c+1e-10) for f,op,c in s['root_conditions']):
   item=dict(s)
   from points_return_monitor import records as return_records
   rows=return_records().get('by_key',{})
   a=rows.get(VERSION+'|'+s['available_id']);b=rows.get(VERSION+'|'+s['buff_id'])
   if a and b:
    item['available_parent']={'full':a['full'],'recent':a['health_since_check']};item['record']={'full':b['full'],'recent':b['health_since_check']}
   roots.append(item)
 return dict(name=name,gaa=n,gp=gp,roots=roots,through=return_records().get('full_through','2026-11-16')) if roots else None
def goalie_badge(ctx):
 if not ctx:return ''
 return '<div class="wn-v6-aegis" title="Optional researched matchup buff; does not change the move or admission">✦ SHATTERED AEGIS <b>'+escape(ctx['name'])+f' · GAA {ctx["gaa"]:.2f} ≥2.95</b></div>'
def goalie_inspect(ctx):
 if not ctx:return ''
 html='<details class="wn-details"><summary>✦ Shattered Aegis · supporting route records</summary><div class="wn-context">'
 for r in ctx['roots']:
  html+='<b>'+escape(r['root_name'])+'</b><br>'
  for title,key in [('Comparable parent','available_parent'),('With GAA≥2.95','record')]:
   x=r[key];html+=escape(title)+': '
   for period in ['full','recent']:
    s=x[period];html+=f'{period}: {s["hits"]}/{s["plays"]} · {s["hit_rate"]:.1f}% · BE {s["be"]:.1f}% · {s["players"]} players · {s["units"]:+.2f}u · {s["roi"]:+.1f}% ROI; ' if s['plays'] else period+': 0/0 · unavailable; '
   html+='<br>'
 html+='Research baseline through Nov16; totals graded through '+escape(ctx.get('through','2026-11-16'))+'. Health from Nov5 onward is included in full. New captures begin Nov18. Other matched advanced moves do not inherit these percentages. No fixed confidence/probability boost.</div></details>'
 return html
CSS='.wn-v6-aegis{padding:8px 11px;margin:7px 0;border:1px solid #e6b859;border-left:4px solid #ffce68;border-radius:7px;background:linear-gradient(100deg,#432e14,#182d3c);color:#ffe5a1;font-weight:800;box-shadow:0 0 12px #d59f4250}.wn-v6-aegis b{display:inline-block;margin-left:7px;color:#fff3d2}'
