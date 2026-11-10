"""Frozen-price card returns: immutable V1, replaceable settled player-game results.

This module tracks existing rules. It never changes eligibility or fits a model.
"""
from __future__ import annotations
import argparse,base64,copy,csv,gzip,hashlib,json,math
from collections import Counter,defaultdict
from datetime import datetime,timezone
from decimal import Decimal
from pathlib import Path
from functools import lru_cache

VERSION='card-return-tracking-2026-11-10-v1'
FILES=('goal_return_records.json','assist_return_records.json','points_return_records.json','special_return_records.json')
ROOT=Path(__file__).resolve().parent
MARKET_ACTUAL={'Goal':'Actual_Goal','Assists':'Actual_Assists','Points':'Actual_Points','SOG':'Actual_SOG','PPP':'Actual_PPP'}

def number(v):
    try:
        n=float(v);return n if math.isfinite(n) else None
    except (TypeError,ValueError):return None

def stamp(v):
    try:
        t=datetime.fromisoformat(str(v).replace('Z','+00:00'))
        return t.astimezone(timezone.utc) if t.tzinfo else None
    except (TypeError,ValueError):return None

def ident(v):
    n=number(v)
    return str(int(n)) if n is not None and n>0 and n==int(n) else ''

def stats(rows):
    n=len(rows);h=sum(bool(r[6]) for r in rows)
    empty=dict(hits=0,plays=0,hit_rate=None,be=None,gap=None,players=0,dates=0,games=0,units=None,roi=None)
    if not n:return empty
    units=Decimal(0);be=Decimal(0)
    for r in rows:
        o=Decimal(str(r[7]))
        if abs(o)<100:raise ValueError('Invalid saved American odds')
        units+=(o/100 if o>0 else 100/-o) if r[6] else -1
        be+=(10000/(o+100) if o>0 else -100*o/(100-o))
    pct=100*h/n;b=float(be/n)
    return dict(hits=h,plays=n,hit_rate=pct,be=b,gap=pct-b,players=len({r[4] for r in rows}),dates=len({r[2] for r in rows}),games=len({r[3] for r in rows}),units=float(units),roi=float(units*100/n))

def catalog(root=ROOT):
    root=Path(root)
    signature=tuple((root/f).stat().st_mtime_ns for f in FILES)
    return _catalog(str(root),signature)

@lru_cache(maxsize=8)
def _catalog(directory,signature):
    root=Path(directory)
    result={}
    for filename in FILES:
        doc=json.loads((root/filename).read_text(encoding='utf-8'))
        for spec in doc['moves']:
            market=spec.get('market') or ('Goal' if filename.startswith('goal_') else 'Assists' if filename.startswith('assist_') else 'Points')
            line=spec.get('line',2.5 if market=='SOG' else .5)
            kit=spec.get('kit_version',doc.get('kit'))
            result[(filename,spec['key'])]={'file':filename,'market':market,'line':line,'kit':kit,**spec}
    return result

def _quote(row,market,line):
    quotes=[]
    def add(l,o,b):
        o=number(o);b=str(b or '').strip()
        if number(l)==line and o is not None and abs(o)>=100 and b and b.lower() not in ('none','nan'):
            quotes.append((o,b))
    add(row.get(market+'_Line'),row.get(market+'_Odds_Over'),row.get(market+'_Book'))
    if market=='Goal' and not quotes:add(row.get('ATG_Line'),row.get('ATG_Odds_Over'),row.get('ATG_Book'))
    if market=='Points' and line==1.5:
        for i in range(1,5):add(row.get(f'BDL_Points_Line_{i}'),row.get(f'BDL_Points_Odds_{i}'),row.get(f'BDL_Points_Book_{i}'))
    return max(quotes,key=lambda q:q[0]) if quotes else None

def capture_row(row,*,root=ROOT):
    """Called before saving the model tracker; does not modify its selected lines."""
    if row.get('Model_Stats_Season')!='2026-2027':return []
    if str(row.get('Roster_Watch','')).strip().lower() in ('true','1'):return []
    from goal_moves_current import goals_moves
    from assist_moves_current import assists_moves
    from sog_moves_current import sog_moves
    from sog_profiles_2026 import profile_moves
    from points_moves_current import points_moves
    from warlord_moves_2026 import legacy_points_moves
    from pp_tracking import frozen_pp_tags
    specs=catalog(root);lookup=defaultdict(list)
    for spec in specs.values():lookup[(spec['market'],spec['name'],spec['rule'])].append(spec)
    result=[]
    def add(market,moves,quote,line):
        if not quote:return
        for m in moves:
            for s in lookup[(market,m['name'],m.get('rule',''))]:
                if s['line']!=line:continue
                if not s.get('legacy') and m.get('kit_version')!=s['kit']:continue
                result.append(dict(file=s['file'],key=s['key'],market=market,line=line,odds=quote[0],book=quote[1],rule=s['rule'],kit_version=s['kit']))
    for market,fn,line in [('Goal',goals_moves,.5),('Assists',assists_moves,.5),('SOG',sog_moves,2.5)]:
        quote=_quote(row,market,line)
        if quote:add(market,fn(row)+(profile_moves(row,monitors=True) if market=='SOG' else []),quote,line)
    for line in (.5,1.5):
        quote=_quote(row,'Points',line)
        if not quote:continue
        r={**row,'Points_Line':line,'Points_Odds_Over':quote[0],'Points_Book':quote[1]}
        add('Points',points_moves(r),quote,line)
        if line==1.5:
            legacy=dict(r)
            if number(legacy.get('Conf_Goal_Move')) is not None:legacy['Conf_Goal']=legacy['Conf_Goal_Move']
            add('Points',[m for m in legacy_points_moves(legacy) if m['name'] in ('Backbone','DPS Tank stance')],quote,line)
    for tag in frozen_pp_tags(row):
        s=specs.get(('special_return_records.json','PPP|'+tag['id']))
        if s and s['rule']==tag['rule'] and s['kit']==tag['kit_version']:
            result.append(dict(file=s['file'],key=s['key'],market='PPP',line=.5,odds=tag['odds'],book=tag['book'],rule=s['rule'],kit_version=s['kit']))
    if len({(x['file'],x['key']) for x in result})!=len(result):raise ValueError('Duplicate return capture identity')
    return result

def attach_return_tracking(frame,now=None):
    now=now or datetime.now(timezone.utc)
    output=frame.copy()
    output['Return_Fired_Moves']=[json.dumps(capture_row(r),ensure_ascii=False,separators=(',',':')) for r in frame.to_dict('records')]
    output['Return_Tracking_Version']=VERSION
    output['Return_Evaluated_UTC']=now.isoformat()
    return output

def validate_tags(row,root=ROOT):
    if row.get('Return_Tracking_Version')!=VERSION:raise ValueError('Unknown return capture version')
    start=stamp(row.get('StartTimeUTC'));evaluated=stamp(row.get('Return_Evaluated_UTC'));frozen=stamp(row.get('Frozen_At_UTC'))
    quote_time=stamp(row.get('Odds_Checked_UTC'))
    if not start or not evaluated or evaluated>=start or (frozen and evaluated>frozen):raise ValueError('Return tags were not captured before game start/freezing')
    if not quote_time or quote_time>=start:raise ValueError('Missing or postgame price capture time')
    tags=json.loads(row.get('Return_Fired_Moves') or '[]')
    if not isinstance(tags,list):raise ValueError('Invalid return tags')
    specs=catalog(root);seen=set()
    for tag in tags:
        key=(tag.get('file'),tag.get('key'));s=specs.get(key)
        if key in seen:raise ValueError('Duplicate return tag')
        seen.add(key)
        if s is None:continue # obsolete kit remains in its own archive, never transferred
        if any(tag.get(k)!=v for k,v in [('rule',s['rule']),('kit_version',s['kit']),('market',s['market']),('line',s['line'])]):raise ValueError('Return tag identity/line mismatch')
        if number(tag.get('odds')) is None or abs(float(tag['odds']))<100 or not tag.get('book'):raise ValueError('Invalid frozen return price')
    return tags

def _rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def _write(path,value):
    content=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text(encoding='utf-8')==content:return False
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(content,encoding='utf-8');temp.replace(path);return True

def backfill_day(root,day):
    """Explicit one-time migration only; never the routine grading path."""
    source=root/'output/slates'/f'tracker_{day}.csv'
    rows=_rows(source);tags={}
    for r in rows:
        start,frozen=stamp(r.get('StartTimeUTC')),stamp(r.get('Frozen_At_UTC'))
        if not start or not frozen or frozen>=start:continue
        r={k:v for k,v in r.items() if not k.startswith(('Actual_','Outcome_','Grade_','Match_Status_'))}
        # Only frozen pregame fields are evaluated in this explicitly labeled migration.
        key=ident(r.get('Game_ID'))+'|'+ident(r.get('Player_ID'))
        if '|'==key or key in tags:raise ValueError('Ambiguous migration player/game')
        tags[key]=capture_row(r,root=root)
    payload=dict(version=VERSION,mode='EXPLICIT_SAVED_PREGAME_REPLAY',date=day,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),tags=tags)
    _write(root/'output/card_returns'/f'backfill_{day}.json',payload)
    return payload

def refresh(root=ROOT):
    compressed=root/'card_return_seed.b64'
    seed=json.loads(gzip.decompress(base64.b64decode(compressed.read_bytes()))) if compressed.exists() else json.loads((root/'card_return_seed.json').read_text(encoding='utf-8'))
    specs=catalog(root)
    if {(f,r['key']) for f,d in seed['files'].items() for r in d['moves']}!=set(specs):raise ValueError('New move version requires a new approved return seed')
    members={tuple(x[:5]):list(x) for x in seed['members']};counts=Counter();through=seed['through'];sources={};settlement_stamp=''
    for path in sorted((root/'output/graded').glob('tracker_*_GRADED.csv')):
        day=path.name[8:18]
        rows=_rows(path);lookup={}
        for r in rows:
            gid=ident(r.get('Resolved_Game_ID')) or ident(r.get('Game_ID'));pid=ident(r.get('Resolved_Player_ID')) or ident(r.get('Player_ID'))
            if not gid or not pid:continue
            if (gid,pid) in lookup:raise ValueError('Duplicate graded player/game')
            lookup[gid,pid]=r
        # Corrections in the post-V1 seed interval replace old outcomes; V1 stays fixed.
        for key,m in list(members.items()):
            if m[2]!=day or day>seed['through']:continue
            s=specs[m[0],m[1]];doc=seed['files'][m[0]]
            cutoff=s.get('previous_through',doc.get('previous_through'))
            if day<=cutoff:continue
            r=lookup.get((m[3],m[4]));actual=number(r.get(MARKET_ACTUAL[s['market']])) if r else None
            if r and r.get('Model_Stats_Season')=='2026-2027' and r.get('Grade_Status')=='FINAL' and actual is not None:
                m[6]=actual>s['line'];counts['seed_outcomes_checked']+=1
        if day<=seed['through']:continue
        sources[day]=hashlib.sha256(path.read_bytes()).hexdigest()
        migration_path=root/'output/card_returns'/f'backfill_{day}.json'
        migration=json.loads(migration_path.read_text()) if migration_path.exists() else None
        if migration:
            raw=root/'output/slates'/f'tracker_{day}.csv'
            if migration['source_sha256']!=hashlib.sha256(raw.read_bytes()).hexdigest():raise ValueError('Migration frozen source changed')
        summary_path=path.with_name(f'tracker_{day}_summary.json')
        if summary_path.exists():
            summary=json.loads(summary_path.read_text());settlement_stamp=max(settlement_stamp,str(summary.get('graded_at_utc') or summary.get('graded_at') or ''))
        for r in rows:
            if r.get('Model_Stats_Season')!='2026-2027':counts['noncurrent_rows']+=1;continue
            if r.get('Grade_Status')!='FINAL':counts['unsettled_rows']+=1;continue
            start,frozen=stamp(r.get('StartTimeUTC')),stamp(r.get('Frozen_At_UTC'))
            if not start or not frozen or frozen>=start:counts['not_pregame']+=1;continue
            gid=ident(r.get('Resolved_Game_ID')) or ident(r.get('Game_ID'));pid=ident(r.get('Resolved_Player_ID')) or ident(r.get('Player_ID'))
            if not gid or not pid:counts['missing_ids']+=1;continue
            if not gid.startswith('202602'):counts['not_current_regular_season']+=1;continue
            if r.get('Return_Tracking_Version')==VERSION:
                tags=validate_tags(r,root);provenance='CAPTURED_PREGAME'
            elif migration:
                tags=migration['tags'].get(gid+'|'+pid,[]);provenance=migration['mode']
            else:counts['missing_return_capture_rows']+=1;continue
            through=max(through,day)
            for tag in tags:
                s=specs.get((tag['file'],tag['key']))
                if not s:counts['other_kit_tags']+=1;continue
                if tag['rule']!=s['rule'] or tag['kit_version']!=s['kit'] or tag['line']!=s['line']:raise ValueError('Captured rule identity changed')
                actual=number(r.get(MARKET_ACTUAL[s['market']]))
                if actual is None:counts['missing_actuals']+=1;continue
                m=[tag['file'],tag['key'],day,gid,pid,r['Player'],actual>tag['line'],float(tag['odds']),tag['book'],provenance]
                key=tuple(m[:5])
                if key in members:raise ValueError('Duplicate settled return identity')
                members[key]=m;counts['new_memberships']+=1
    groups=defaultdict(list)
    for m in members.values():groups[m[0],m[1]].append(m)
    documents=copy.deepcopy(seed['files'])
    for filename,doc in documents.items():
        current={s['key']:s for s in specs.values() if s['file']==filename}
        for row in doc['moves']:
            s=current.get(row['key'])
            if not s or any(s.get(k)!=row.get(k) for k in ('name','rule','original_hits','original_plays')):raise ValueError('Seed does not match current move kit')
            cutoff=row.get('previous_through',doc.get('previous_through'));group=groups[filename,row['key']]
            row['full']=stats(group);row['since_check']=stats([m for m in group if m[2]>cutoff])
            # row['previous'] is the exact immutable approved V1 object.
        doc.update(full_through=through,since_through=through,auto_update_version=VERSION,seed_version=seed['version'],baseline_record_version=seed['files'][filename]['version'],version=seed['files'][filename]['version']+'+graded-through-'+through)
        _write(root/filename,doc)
    manifest={'version':VERSION,'seed_version':seed['version'],'through':through,'source_files':sources,'counts':dict(counts),'memberships':len(members),'V1_fixed':True,'rules_unchanged':True}
    manifest['generation']=hashlib.sha256(json.dumps(documents,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    previous_path=root/'output/card_returns/latest.json'
    previous=json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else {}
    old=previous.get('manifest',{})
    manifest['generated_utc']=(old.get('generated_utc') if old.get('generation')==manifest['generation'] and old.get('generated_utc') else datetime.now(timezone.utc).isoformat())
    bundle={'manifest':manifest,'records':documents}
    _write(root/'output/card_returns/latest.json',bundle)
    _write(root/'output/card_returns/status.json',manifest)
    return bundle

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',type=Path,default=ROOT);parser.add_argument('--backfill-date',action='append',default=[])
    args=parser.parse_args()
    for day in args.backfill_date:backfill_day(args.repo,day)
    bundle=refresh(args.repo);print(json.dumps(bundle['manifest'],indent=2))

if __name__=='__main__':main()

