"""Freeze PPP move identities/quotes; settle them without rerunning future rules."""
from __future__ import annotations
from datetime import datetime, timezone
from hashlib import sha256
import json, math
import pandas as pd

VERSION = 'pp-tracking-2026-11-08-v1'
CAPTURED = 'PREGAME_CAPTURED'
RECONSTRUCTED = 'RECONSTRUCTED_ASOF'

def _num(v):
    try:
        v=float(v)
        return v if math.isfinite(v) else None
    except (TypeError,ValueError):return None

def _json(v, default):
    if not isinstance(v,str) or not v.strip():return default
    result=json.loads(v)
    if not isinstance(result,type(default)):raise ValueError('Invalid frozen PP tracking JSON')
    return result

def _rule(move):
    return ' + '.join(f'{field} {op} {cut:g}' for field,op,cut in move['entry']+move['conditions'])

def attach_pp_tracking(frame, *, mode=CAPTURED, evaluated_utc=None, source_sha256='', source_file=''):
    """Capture the approved rules and best exact-.5 quote; never change other moves."""
    from pp_cards import records,rank_pp_cards,VERSION as KIT,onice_xg
    from power_play_quotes import priced_ppp_quotes
    if mode not in (CAPTURED,RECONSTRUCTED):raise ValueError('Unknown PP tracking mode')
    out=frame.copy()
    if out.empty:return out
    if out.Date.astype(str).str[:10].nunique()!=1:raise ValueError('PP tracking needs one slate date')
    keys=['Player','Team','Game']
    # Same-name teammates are intentionally retained as roster rows. Their
    # name-only book prices are quarantined rather than assigned to either ID.
    ambiguous=out.duplicated(keys,keep=False)
    # Team provenance is finalized after the official context adapter in the model.
    # Rebuild this same-source gap here so its availability reflects that final provenance.
    if {'Team_Stats_Season','Team_PP_xGF60','Team_PP_GF60'}.issubset(out.columns):
        def gap(r):
            e,a=_num(r.get('Team_PP_xGF60')),_num(r.get('Team_PP_GF60'))
            return e-a if r.get('Team_Stats_Season')=='2026-2027' and e is not None and a is not None and min(e,a)>=0 else None
        out['MP_Team_PP_xG_minus_G60']=[gap(r) for r in out.to_dict('records')]
    catalog=records()['moves']
    identity=json.dumps([dict(id=m['id'],entry=m['entry'],conditions=m['conditions'],record=m['record']) for m in catalog],sort_keys=True,separators=(',',':'),ensure_ascii=False)
    record_version=KIT+':'+sha256(identity.encode('utf-8')).hexdigest()[:16]
    manifest=json.dumps([dict(id=m['id'],name=m['name'],kind=m['kind']) for m in catalog],ensure_ascii=False,separators=(',',':'))
    eligible=out.loc[~ambiguous]
    cards=rank_pp_cards(eligible,priced_ppp_quotes(eligible,line_filter=.5))
    frozen={}
    for card in cards:
        row,quote=card['row'],card['quote'];tags=[]
        for m in card['moves']:
            values={field:(_num(quote['PPP line']) if field=='PPP line' else onice_xg(row) if field=='PP_OnIce_xG_per_game' else _num(row.get(field))) for field,_,_ in m['entry']+m['conditions']}
            tags.append(dict(market='PPP',name=m['name'],id=m['id'],kind=m['kind'],rule=_rule(m),icon='ϟ',kit_version=KIT,record_version=record_version,
                             line=float(quote['PPP line']),odds=float(quote['Over odds']),book=str(quote['Book']),odds_checked_utc=str(row.get('Odds_Checked_UTC') or ''),
                             pp_unit=str(quote.get('PP unit') or ''),tracking_mode=mode,historical_record=m['record'],inputs=values,
                             context_asof=str(row.get('PP_Card_Context_AsOf') or ''),lower_context_asof=str(row.get('PP_Lower_Context_AsOf') or '')))
        frozen[tuple(str(row.get(k)) for k in keys)]=json.dumps(tags,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    out['PP_Fired_Moves']=[frozen.get(tuple(str(row.get(k)) for k in keys),'[]') for row in out.to_dict('records')]
    out['PP_Move_Kit_Version']=KIT
    out['PP_Move_Record_Version']=record_version
    out['PP_Tracking_Version']=VERSION
    out['PP_Tracking_Mode']=mode
    out['PP_Move_Catalog']=manifest
    out['PP_Tracking_Evaluated_UTC']=evaluated_utc or datetime.now(timezone.utc).isoformat()
    out['PP_Tracking_Source_SHA256']=source_sha256
    out['PP_Tracking_Source_File']=source_file
    out['PP_Tracking_Identity_Status']=['AMBIGUOUS_NAME_WITHHELD' if v else 'OK' for v in ambiguous]
    return out

def frozen_pp_tags(row):
    """Read only captured definitions. Missing tags never trigger a current-rule replay."""
    tags=_json(row.get('PP_Fired_Moves'),[])
    seen=set()
    for tag in tags:
        if not isinstance(tag,dict) or tag.get('market')!='PPP' or not all(tag.get(k) for k in ('id','name','kit_version','record_version')):
            raise ValueError('Invalid frozen PPP move identity')
        if tag['id'] in seen:raise ValueError('Duplicate frozen PPP move')
        seen.add(tag['id'])
        if _num(tag.get('line'))!=.5 or _num(tag.get('odds')) in (None,0):raise ValueError('Missing exact-.5 PPP quote')
        if tag.get('tracking_mode') not in (CAPTURED,RECONSTRUCTED):raise ValueError('Unknown frozen PPP capture provenance')
    return tags

def pp_move_results(row):
    results=[]
    for tag in frozen_pp_tags(row):
        line=_num(tag['line']);actual=_num(row.get('Actual_PPP')) if row.get('Grade_Status')=='FINAL' else None
        outcome='W' if actual is not None and actual>line else 'L' if actual is not None and actual<line else 'P' if actual is not None else ''
        odds=_num(tag['odds']);valid_price=odds is not None and abs(odds)>=100
        profit=(odds/100 if odds>0 else 100/-odds) if outcome=='W' and valid_price else -1.0 if outcome=='L' and valid_price else 0.0 if outcome=='P' and valid_price else None
        be=100*(100/(odds+100) if odds>0 else -odds/(-odds+100)) if valid_price else None
        hist=tag.get('historical_record') or {}
        results.append(dict(Date=row.get('Date'),Game=row.get('Game'),Game_ID=row.get('Resolved_Game_ID',row.get('Game_ID')),Player=row.get('Player'),
            Player_ID=row.get('Resolved_Player_ID',row.get('Player_ID')),Team=row.get('Team'),Market='PPP',Move=tag['name'],Kind=tag['kind'],Rule=tag['rule'],Icon=tag.get('icon',''),Kit_Version=tag['kit_version'],
            Line=line,Actual=actual,Outcome=outcome,Grade_Status=row.get('Grade_Status'),Match_Status='OK' if actual is not None else row.get('Grade_Status'),
            Move_ID=tag['id'],Tracking_Mode=tag['tracking_mode'],Record_Version=tag['record_version'],Odds=odds,Book=tag.get('book',''),Odds_Checked_UTC=tag.get('odds_checked_utc',''),
            PP_Unit=tag.get('pp_unit',''),Book_BE=be,Stake_Units=1.0 if profit is not None else None,Profit_Units=profit,
            Sample_Hits=hist.get('hits'),Sample_Plays=hist.get('plays'),Sample_Hit_Pct=hist.get('rate'),Sample_Avg_Book_BE=hist.get('BE'),
            Source_Snapshot_SHA256=row.get('PP_Tracking_Source_SHA256','')))
    return results

def pp_price_records(graded):
    result=[]
    for row in graded.to_dict('records'):
        for r in pp_move_results(row):
            if r['Outcome'] not in ('W','L','P') or r['Profit_Units'] is None:continue
            be=r['Book_BE'];band='Under 40%' if be<40 else '40–49%' if be<50 else '50–59%' if be<60 else '60%+'
            result.append(dict(Date=str(r['Date']),Game=r['Game'],Player=r['Player'],Market='PPP',Move=r['Move'],Kit=r['Kit_Version'],Line=r['Line'],Odds=r['Odds'],Book=r['Book'],
                Captured_UTC=r['Odds_Checked_UTC'],Break_even_pct=round(be,2),Price_band=band,Outcome=r['Outcome'],Profit_units=round(r['Profit_Units'],4),Tracking_Mode=r['Tracking_Mode'],Record_Version=r['Record_Version']))
    return result

def tracking_summary(graded,moves):
    """Include all frozen catalog entries, including rules with zero appearances."""
    catalog={}
    for row in graded.to_dict('records'):
        for m in _json(row.get('PP_Move_Catalog'),[]):
            mode=row.get('PP_Tracking_Mode')
            key=(m.get('kit_version',row.get('PP_Move_Kit_Version')),m.get('record_version',row.get('PP_Move_Record_Version')),mode,m['id'])
            catalog[key]=m
    pp=moves.loc[moves.Market.eq('PPP')] if not moves.empty else pd.DataFrame()
    result=[]
    for (kit,record_version,mode,ident),m in catalog.items():
        group=pp.loc[pp.Move_ID.eq(ident)&pp.Kit_Version.eq(kit)&pp.Record_Version.eq(record_version)&pp.Tracking_Mode.eq(mode)] if not pp.empty else pp
        wins=int(group.Outcome.eq('W').sum()) if len(group) else 0
        losses=int(group.Outcome.eq('L').sum()) if len(group) else 0
        pushes=int(group.Outcome.eq('P').sum()) if len(group) else 0
        settled=group.loc[group.Outcome.isin(['W','L'])&group.Profit_Units.notna()] if len(group) else group
        n=len(settled);net=float(settled.Profit_Units.sum()) if n else None
        rate=100*wins/(wins+losses) if wins+losses else None
        be=float(settled.Book_BE.mean()) if n else None
        result.append(dict(Move=m['name'],Move_ID=ident,Kit_Version=kit,Record_Version=record_version,Tracking_Mode=mode,
            Appearances=len(group),Wins=wins,Losses=losses,Pushes=pushes,Unresolved=len(group)-wins-losses-pushes,Hit_Pct=rate,Average_Book_BE=be,Gap_pp=rate-be if rate is not None and be is not None else None,
            Unique_Players=int(group.Player_ID.nunique()) if len(group) else 0,Games=int(group.Game_ID.nunique()) if len(group) else 0,Stake_Units=n,Profit_Units=net,ROI_Pct=100*net/n if n else None))
    return dict(version=VERSION,tagged_rows=int(graded.get('PP_Fired_Moves',pd.Series(dtype=str)).fillna('[]').ne('[]').sum()),
                missing_capture_rows=int(graded.get('PP_Tracking_Version',pd.Series(index=graded.index,dtype=str)).isna().sum()),
                catalog_moves=len(catalog),modes=sorted({k[2] for k in catalog}),moves=result,
                unit_policy='Flat1u at each rule\'s saved exact-.5 best quote; overlapping rules are separate reports, not an additive portfolio.')
