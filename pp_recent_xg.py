"""Recent PP on-ice xG/game from the same MoneyPuck game-log download.

The window is the player's last5/10 completed regular-season appearances,
including zero-PP games. Missing PP rows stay missing, never a season proxy.
"""
from datetime import date
import math
import pandas as pd

VERSION='pp-onice-recent-2026-11-25-v1'
FIELDS=tuple('PP_OnIce_xG_L'+str(w)+'_'+suffix for w in (5,10) for suffix in ('per_game','Games','Status'))+('PP_OnIce_xG_Recent_SourceDate','PP_OnIce_xG_Recent_AsOf','PP_OnIce_xG_Recent_Version')

def from_game_frame(frame,slate_date,season_start):
    before=date.fromisoformat(str(slate_date)[:10]).isoformat()
    required={'playerId','gameId','gameDate','situation','OnIce_F_xGoals','icetime'}
    if not required.issubset(frame):return {}
    f=frame[list(required)].copy()
    for k in ('playerId','gameId','gameDate'):f[k]=pd.to_numeric(f[k],errors='coerce')
    f=f.dropna(subset=['playerId','gameId','gameDate'])
    f=f[f.playerId.gt(0)&f.playerId.mod(1).eq(0)&f.gameId.mod(1).eq(0)&f.gameDate.mod(1).eq(0)]
    f['playerId']=f.playerId.astype('int64');f['gameId']=f.gameId.astype('int64')
    f['day']=pd.to_datetime(f.gameDate.astype('int64').astype(str),format='%Y%m%d',errors='coerce').dt.strftime('%Y-%m-%d')
    f=f[f.day.notna()&f.day.lt(before)&f.gameId.astype(str).str.startswith(str(int(season_start))+'02')]
    all_games=f[f.situation.eq('all')].copy();pp=f[f.situation.eq('5on4')].copy()
    invalid=set(all_games.loc[all_games.duplicated(['playerId','gameId'],keep=False),'playerId'])|set(pp.loc[pp.duplicated(['playerId','gameId'],keep=False),'playerId'])
    for k in ['OnIce_F_xGoals','icetime']:pp[k]=pd.to_numeric(pp[k],errors='coerce')
    pp=pp.set_index(['playerId','gameId']);result={}
    for pid,g in all_games.groupby('playerId',sort=False):
        g=g.sort_values(['day','gameId']);last=str(g.day.iloc[-1]);out={'PP_OnIce_xG_Recent_SourceDate':last,'PP_OnIce_xG_Recent_AsOf':before,'PP_OnIce_xG_Recent_Version':VERSION}
        for w in (5,10):
            h=g.tail(w);values=[];status='Ready'
            for row in h.itertuples():
                key=(pid,row.gameId)
                if pid in invalid or key not in pp.index:status='Missing game-level PP data';break
                z=pp.loc[key];value=float(z.OnIce_F_xGoals);seconds=float(z.icetime)
                if str(z.day)!=str(row.day) or not math.isfinite(value) or value<0 or not math.isfinite(seconds) or seconds<0 or seconds==0 and value!=0:
                    status='Invalid game-level PP data';break
                values.append(value)
            complete=len(values)==len(h) and len(h)>0
            out[f'PP_OnIce_xG_L{w}_per_game']=round(sum(values)/len(h),4) if complete else None
            out[f'PP_OnIce_xG_L{w}_Games']=len(h) if complete else len(values)
            out[f'PP_OnIce_xG_L{w}_Status']=('Ready' if len(h)==w else f'Available{len(h)}of{w} games') if complete else status
        result[int(pid)]=out
    return result

def display_value(row,window):
    """Only show a saved recent value with matching dated provenance."""
    try:
        before=date.fromisoformat(str(row.get('Date',''))[:10]).isoformat()
        last=date.fromisoformat(str(row.get('PP_OnIce_xG_Recent_SourceDate',''))[:10]).isoformat()
        n=float(row.get(f'PP_OnIce_xG_L{window}_Games'));v=float(row.get(f'PP_OnIce_xG_L{window}_per_game'))
        if row.get('PP_OnIce_xG_Recent_Version')!=VERSION or row.get('PP_OnIce_xG_Recent_AsOf')!=before or last>=before:return None
        if not math.isfinite(v) or v<0 or not math.isfinite(n) or n<1 or n>window or int(n)!=n:return None
        return v
    except (TypeError,ValueError):return None
