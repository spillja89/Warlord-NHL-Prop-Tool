"""Dated starter-report confirmation fallback. Never supplies goalie statistics."""
from datetime import date,datetime,timedelta,timezone
from zoneinfo import ZoneInfo
import json,re,unicodedata
from bs4 import BeautifulSoup

TEAMS=set('ANA BOS BUF CAR CBJ CGY CHI COL DAL DET EDM FLA LAK MIN MTL NJD NSH NYI NYR OTT PHI PIT SEA SJS STL TBL TOR UTA VAN VGK WPG WSH'.split())

def name_key(value):
    return ''.join(c for c in unicodedata.normalize('NFKD',str(value or '')).casefold() if c.isalnum() and not unicodedata.combining(c))

def numeric_id(value):
    try:return str(int(float(value)))
    except (TypeError,ValueError,OverflowError):return ''

def parse_goaliepost(html,day,now=None):
    """Parse only exact-game CONFIRMED entries, with dated report provenance."""
    now=now or datetime.now(timezone.utc)
    day=date.fromisoformat(str(day));url=f'https://goaliepost.com/probable-goalies/{day.isoformat()}'
    chunks=[]
    for script in BeautifulSoup(html,'html.parser').find_all('script'):
        text=script.string or script.get_text()
        if 'self.__next_f.push(' not in text:continue
        try:
            payload=json.loads(text.split('self.__next_f.push(',1)[1].rsplit(')',1)[0])
            if len(payload)>1 and isinstance(payload[1],str):chunks.append(payload[1])
        except (ValueError,TypeError,IndexError):continue
    stream=''.join(chunks);games={};decoder=json.JSONDecoder()
    for match in re.finditer(r'"games"\s*:\s*(\[)',stream):
        try:items,_=decoder.raw_decode(stream,match.start(1))
        except ValueError:continue
        for game in items:
            if isinstance(game,dict) and numeric_id(game.get('externalId')):games[numeric_id(game['externalId'])]=game
    out={};ambiguous=set()
    for game_id,game in games.items():
        try:
            start=datetime.fromisoformat(str(game['gameDate']).removeprefix('$D').replace('Z','+00:00'))
            season=str(day.year if day.month>=7 else day.year-1)+str(day.year+1 if day.month>=7 else day.year)
            if start.tzinfo is None or start.astimezone(ZoneInfo('America/Chicago')).date()!=day or str(game.get('season'))!=season or game.get('gameType')!='R':continue
            for side,other in [('HOME','AWAY'),('AWAY','HOME')]:
                team=game['teams'][side]['team']['abbreviation'];opp=game['teams'][other]['team']['abbreviation']
                if team not in TEAMS or opp not in TEAMS:continue
                entries=[x for x in game['predictedGoalies'].get(side,[]) if x.get('likeliness')=='CONFIRMED']
                if len(entries)!=1:continue
                entry=entries[0];goalie=entry['goalie'];gid=numeric_id(goalie.get('externalId'));name=str(goalie.get('fullName') or '').strip()
                reported=datetime.fromtimestamp(float(entry['timestamp']),timezone.utc)
                if not gid or not name or reported>now+timedelta(minutes=5) or reported<start-timedelta(days=2):continue
                row=dict(goalie=name,goalie_id=gid,status='Confirmed',source='GoaliePost',url=url,
                         reported_at=reported.isoformat(),game_id=game_id,opponent=opp,start_utc=start.isoformat())
                if team in out and out[team]!=row:ambiguous.add(team)
                else:out[team]=row
        except (KeyError,TypeError,ValueError,OverflowError):continue
    return {k:v for k,v in out.items() if k not in ambiguous}

def fetch_goaliepost(day,session=None):
    import requests
    client=session or requests.Session()
    response=client.get(f'https://goaliepost.com/probable-goalies/{day.isoformat()}',timeout=(4,8),headers={'User-Agent':'Mozilla/5.0'})
    response.raise_for_status()
    return parse_goaliepost(response.text,day)

def merge_reports(primary,secondary):
    """Keep official/confirmed primary reports; supplement provisional reports."""
    out={k:{**v,'source':v.get('source') or 'DailyFaceoff'} for k,v in primary.items()}
    for team,report in secondary.items():
        if report.get('status')!='Confirmed':continue
        old=out.get(team,{})
        if str(old.get('status','')).lower() in ('official','confirmed'):continue
        out[team]=dict(report)
    return out

def fetch_reports(day,primary_fetcher):
    try:primary=primary_fetcher(day)
    except Exception:primary={}
    try:secondary=fetch_goaliepost(day)
    except Exception:secondary={}
    return merge_reports(primary,secondary)

def apply_live_reports(frame,reports,day):
    """Status-only update for same day, game, team and named/identified goalie."""
    out=frame.copy();updated=0;changed=[]
    for team,report in reports.items():
        name=report.get('goalie');status=str(report.get('status') or '')
        if not name or status.lower() not in ('confirmed','official','likely','expected'):continue
        team_mask=out['Opp'].astype(str).eq(team)&out['Date'].astype(str).str[:10].eq(str(day))
        if report.get('game_id'):
            team_mask &= out['Game_ID'].map(numeric_id).eq(numeric_id(report['game_id']))
        same=out['Opp_Goalie'].map(name_key).eq(name_key(name))
        if report.get('goalie_id') and 'Opp_Goalie_ID' in out:
            saved_ids=out['Opp_Goalie_ID'].map(numeric_id)
            same = saved_ids.eq(numeric_id(report['goalie_id'])) | (saved_ids.eq('') & same)
        mask=team_mask&same
        # A provisional source must not downgrade a saved confirmed/official starter.
        if status.lower()!='official':mask &= ~out['Opp_Goalie_Status'].astype(str).str.lower().eq('official')
        if status.lower() not in ('confirmed','official'):mask &= ~out['Opp_Goalie_Status'].astype(str).str.lower().eq('confirmed')
        if mask.any():
            if status.lower()=='confirmed' and not out.loc[mask,'Opp_Goalie_Status'].astype(str).str.lower().eq('confirmed').all():updated+=1
            out.loc[mask,'Opp_Goalie_Status']=status
            for field,key in [('Opp_Goalie_Report_Source','source'),('Opp_Goalie_Report_URL','url'),('Opp_Goalie_Report_UTC','reported_at')]:
                if field not in out:out[field]=''
                out.loc[mask,field]=report.get(key,'')
        elif team_mask.any() and not same[team_mask].any():changed.append(team)
    return out,updated,changed
