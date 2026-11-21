"""Independent approved environment routes on the .55 xG plus Layer1 parent."""
from functools import lru_cache
from pathlib import Path
import json
from pp_v2_moves import number,goalie_valid
from pp_v3_context import VERSION as CONTEXT

VERSION='pp-moves-v3-2026-11-21'
GOALIE_FIELD='PPV3_Goalie_Expected_SV'

@lru_cache(maxsize=1)
def catalog():return json.loads(Path(__file__).with_name('pp_v3_rules.json').read_text(encoding='utf-8'))

def prepare(row):
    out=dict(row);out[GOALIE_FIELD]=None
    if row.get('Model_Stats_Season')!='2026-2027':return out
    if goalie_valid(row):
        ga,shots=number(row.get('Opp_Goalie_xGA')),number(row.get('Opp_Goalie_Shots_Faced'))
        # Reject a check explicitly captured after the game's scheduled start.
        from card_returns import stamp
        check,start=stamp(row.get('Goalie_Check_UTC')),stamp(row.get('StartTimeUTC'))
        if check and start and check>=start:return out
        if ga is not None and ga>=0 and shots is not None and shots>0:
            v=1-ga/shots
            if 0<=v<=1:out[GOALIE_FIELD]=v
    return out

def value(row,field,line=.5):
    if field=='PPP_Line':return number(line)
    if field in {'Drought_PPP','PPP10_total'}:
        v=number(row.get(field));return v if v is not None and v>=0 and v.is_integer() else None
    if field.startswith('PPV3_') and field!=GOALIE_FIELD:
        if row.get('PP_V3_Context_Version')!=CONTEXT or str(row.get('PP_V3_Context_AsOf'))!=str(row.get('Date'))[:10]:return None
    if field=='opp_5v5_xGA60':
        v=number(row.get(field));return v if row.get('Team_Stats_Season')=='2026-2027' and v is not None and 0<=v<50 else None
    if field=='PP_OnIce_xG_per_game':
        from pp_cards import onice_xg
        return onice_xg(row)
    return number(row.get(field))

def fired(row,line=.5):
    if number(line)!=.5 or row.get('Model_Stats_Season')!='2026-2027' or row.get('Team_Stats_Season')!='2026-2027':return []
    row=prepare(row);found=[]
    for move in catalog()['moves']:
        if all((v:=value(row,f,line)) is not None and (v>=c if op=='>=' else v<=c if op=='<=' else v==c) for f,op,c in move['entry']+move['conditions']):found.append(move)
    return found

def attach_fields(frame):
    out=frame.copy();out[GOALIE_FIELD]=[prepare(r).get(GOALIE_FIELD) for r in out.to_dict('records')]
    out['PP_V3_Player_Version']=VERSION
    return out
