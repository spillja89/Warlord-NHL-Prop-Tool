"""Read-only live display outcomes. Never settles bets or writes move returns."""
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import math
from grade_daily_tracker import _actuals,_players,_power_play_points

FINAL={'FINAL','OFF'}
ACTIVE={'LIVE','CRIT'}
LABELS={'Goal':'goals','Assists':'assists','Points':'points','SOG':'shots','PPP':'PPP'}

def number(v):
    try:
        n=float(v)
        return n if math.isfinite(n) else None
    except (TypeError,ValueError):return None

def identifier(v):
    n=number(v)
    return str(int(n)) if n is not None and n>0 and n.is_integer() else ''

def utc(v):
    try:
        d=datetime.fromisoformat(str(v).replace('Z','+00:00'))
        return d.astimezone(timezone.utc) if d.tzinfo else None
    except (TypeError,ValueError):return None

def scheduled(pick,now):
    start=utc(pick.get('start_utc'))
    return bool(start and start>now)

def result(state,label,*,value=None,final=False,game_state=None):
    return dict(state=state,label=label,value=value,final=final,game_state=game_state)

def identity_matches(pick,payload):
    if not isinstance(payload,dict) or identifier(payload.get('id'))!=identifier(pick.get('game_id')):
        return False
    start=utc(payload.get('startTimeUTC'))
    day=start.astimezone(ZoneInfo('America/Chicago')).date().isoformat() if start else payload.get('gameDate')
    if day!=pick.get('date'):return False
    away=str((payload.get('awayTeam') or {}).get('abbrev') or '').upper()
    home=str((payload.get('homeTeam') or {}).get('abbrev') or '').upper()
    if not away or not home or str(pick.get('team') or '').upper() not in {away,home}:return False
    game=str(pick.get('game') or '').split(' · ')[0].replace(' ','').upper()
    return not game or game==f'{away}@{home}'

def evaluate(pick,box,landing=None):
    """Exact player/game/date/market/line. Missing stats never become zero."""
    line=number(pick.get('line'));market=pick.get('market')
    if line is None or line<0 or market not in LABELS or str(pick.get('side') or 'Over').casefold()!='over':
        return result('unavailable','Live check unavailable')
    if not identity_matches(pick,box):return result('unavailable','Live check unavailable')
    state=str(box.get('gameState') or '').upper()
    if state in {'FUT','PRE'}:return result('scheduled','Awaiting puck drop',game_state=state)
    if state not in FINAL|ACTIVE:return result('unavailable','Game status pending',game_state=state)
    if box.get('limitedScoring') is True:return result('unavailable','Official stats pending',game_state=state)
    players=_players(box).get(str(pick.get('team') or '').upper(),[])
    matched=[p for p in players if identifier(p.get('playerId'))==identifier(pick.get('player_id'))]
    if len(matched)!=1:return result('unavailable','Player stats pending',game_state=state)
    player=matched[0];actual=_actuals(player);is_final=state in FINAL
    field={'Goal':'goals','Assists':'assists','Points':'points','SOG':'sog'}.get(market)
    if market=='PPP':
        if not identity_matches(pick,landing) or str(landing.get('gameState') or '').upper() not in FINAL|ACTIVE:
            return result('unavailable','PP scoring credits pending',game_state=state)
        points=_power_play_points(landing)
        if points is None:return result('unavailable','PP scoring credits pending',game_state=state)
        value=points.get(int(identifier(pick.get('player_id'))),0) if identifier(pick.get('player_id')) else None
        if value is not None and (actual.get('points') is None or value>actual['points']):
            return result('unavailable','PP scoring credits updating',game_state=state)
        # Landing can finalize before/after the box score. Call PPP final only
        # when both official responses are final; live victories stay live.
        is_final=is_final and str(landing.get('gameState') or '').upper() in FINAL
    else:value=actual.get(field)
    if value is None or value<0 or value!=int(value):return result('unavailable','Official stats pending',game_state=state)
    if is_final and str(player.get('toi') or '') in {'0:00','00:00'} and all((actual.get(k) or 0)==0 for k in ['goals','assists','sog']):
        return result('did_not_play','Did not play · check book',final=True,game_state=state)
    value=int(value);stat=f'{value} {LABELS[market]}'
    if value>line:return result('victory',f'⚔ VICTORIOUS · {"FINAL" if is_final else "LIVE"} · {stat}',value=value,final=is_final,game_state=state)
    if is_final:
        return result('push' if value==line else 'loss',f'{"PUSH" if value==line else "MISSED"} · FINAL · {stat}',value=value,final=True,game_state=state)
    needed=math.floor(line)+1
    return result('live',f'IN BATTLE · {value}/{needed} {LABELS[market]}',value=value,game_state=state)

def apply_results(picks,bundles,now=None,previous=None):
    now=now or datetime.now(timezone.utc);previous=previous or {};out={}
    for pick in picks:
        pid=str(pick.get('pick_id') or '')
        if scheduled(pick,now):
            out[pid]={**result('scheduled','Awaiting puck drop'),'checked_utc':now.isoformat()};continue
        bundle=bundles.get(identifier(pick.get('game_id'))) or {}
        fresh=evaluate(pick,bundle.get('box'),bundle.get('landing'))
        if fresh['state']=='unavailable' and previous.get(pid,{}).get('state') in {'victory','live','loss','push','did_not_play'}:
            old=previous[pid]
            out[pid]={**old,'stale':True,'label':old['label'].split(' · update delayed')[0]+' · update delayed'}
        else:
            out[pid]={**fresh,'stale':False,'checked_utc':bundle.get('checked_utc') or now.isoformat()}
    return out
