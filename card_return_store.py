"""Read one validated card-return generation; refresh independently of odds runs."""
from functools import lru_cache
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import requests
from card_returns import FILES,VERSION,_write

URL='https://raw.githubusercontent.com/spillja89/Warlord-NHL-Prop-Tool/launch/2026-warlord-readiness/output/card_returns/latest.json'

def _identity(doc):
    return {r['key']:(r['name'],r['rule'],r.get('kit_version'),r.get('line'),r['original_hits'],r['original_plays'],r['previous']) for r in doc['moves']}

def validate_bundle(bundle,root):
    if bundle.get('manifest',{}).get('version')!=VERSION or set(bundle.get('records',{}))!=set(FILES):raise ValueError('Unknown return bundle')
    docs=bundle['records']
    digest=hashlib.sha256(json.dumps(docs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=bundle['manifest'].get('generation'):raise ValueError('Incomplete return generation')
    for filename,doc in docs.items():
        local=json.loads((root/filename).read_text(encoding='utf-8'))
        if _identity(doc)!=_identity(local):raise ValueError('Return bundle changed a move identity or V1 record')
        for row in doc['moves']:
            a,b,c=row['previous'],row['since_check'],row['full']
            if a['plays']+b['plays']!=c['plays'] or a['hits']+b['hits']!=c['hits']:raise ValueError('Overlapping return periods')
            for s in (a,b,c):
                if s['plays'] and abs(s['units']*100/s['plays']-s['roi'])>1e-7:raise ValueError('Return ROI arithmetic mismatch')
    return bundle

def sync_published(root,session=requests):
    root=Path(root)
    try:
        response=session.get(URL,params={'checked_at':int(datetime.now(timezone.utc).timestamp())},timeout=8)
        response.raise_for_status()
        if len(response.content)>3*1024*1024:raise ValueError('Return bundle too large')
        bundle=validate_bundle(response.json(),root)
        # One atomic file supplies all props; readers never see half a generation.
        _write(root/'cache/card_returns/latest.json',bundle)
        return {**bundle['manifest'],'sync_ok':True}
    except (requests.RequestException,ValueError,OSError,KeyError,TypeError):
        current=_bundle(root)
        return {**(current.get('manifest',{}) if current else {}),'sync_ok':False}

@lru_cache(maxsize=16)
def _read(path,mtime,size):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def _json(path):
    s=path.stat();return _read(str(path),s.st_mtime_ns,s.st_size)

def _bundle(root):
    candidates=[]
    for p in (root/'output/card_returns/latest.json',root/'cache/card_returns/latest.json'):
        try:
            bundle=_json(p)
            # validate only on load of a new generation; current file stats key the cache.
            validated=_validate_cached(str(root),str(p),p.stat().st_mtime_ns,p.stat().st_size,tuple((root/f).stat().st_mtime_ns for f in FILES))
            candidates.append(validated)
        except (OSError,ValueError,KeyError,TypeError):continue
    return max(candidates,key=lambda b:(b['manifest']['through'],b['manifest'].get('generated_utc',''))) if candidates else None

@lru_cache(maxsize=16)
def _validate_cached(root,path,mtime,size,identity_signature):
    return validate_bundle(_read(path,mtime,size),Path(root))

def load_records(path):
    path=Path(path);bundle=_bundle(path.parent)
    try:doc=bundle['records'][path.name] if bundle else _json(path)
    except (OSError,ValueError,KeyError,TypeError):return {}
    return {**doc,'by_key':{r['key']:r for r in doc['moves']}}
