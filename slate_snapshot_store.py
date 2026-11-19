"""Durable completed NHL slates, isolated from private Played/bet tables.

Only server-generated raw trackers are saved. No account settings, secrets,
uploaded files, Played marks or ledger records are included in these payloads.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date,datetime,timezone
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import csv,gzip,io,json,os,tempfile,threading

VERSION='slate-snapshot-2026-11-19-v1'
MAX_BYTES=25*1024*1024
TABLE='warlord_slate_snapshots'
_LOCAL_COMMIT_LOCK=threading.RLock()

def stamp(value):
    try:
        v=datetime.fromisoformat(str(value).replace('Z','+00:00'))
        return v.astimezone(timezone.utc) if v.tzinfo else None
    except (TypeError,ValueError):return None

def _metadata(row):
    day=date.fromisoformat(str(row.get('Date',''))[:10])
    start=day.year if day.month>=7 else day.year-1
    expected=f'{start}-{start+1}'
    season=str(row.get('Model_Stats_Season',''))
    if season not in (expected,'','Unavailable'):raise ValueError('Snapshot season mismatch')
    season=expected
    quoted=next((t for f in ('Odds_Checked_UTC','Roster_Check_UTC','Injury_Check_UTC') if (t:=stamp(row.get(f)))),None)
    if quoted is None:raise ValueError('Snapshot has no source timestamp')
    built=stamp(row.get('Snapshot_Built_UTC')) or max([quoted]+[t for f in ('Return_Evaluated_UTC','PP_Tracking_Evaluated_UTC','Roster_Check_UTC') if (t:=stamp(row.get(f)))])
    return day,season,built,quoted

@dataclass(frozen=True)
class Snapshot:
    content:bytes
    slate_date:date
    season:str
    built_at:datetime
    quoted_at:datetime
    sha:str
    rows:int

    @property
    def rank(self):return self.slate_date,self.built_at,self.quoted_at
    def public_info(self):
        return dict(date=self.slate_date.isoformat(),built_at=self.built_at.isoformat(),quoted_at=self.quoted_at.isoformat(),rows=self.rows,sha=self.sha,version=VERSION)

def _inspect_snapshot(content):
    if not isinstance(content,bytes) or not content or len(content)>MAX_BYTES:raise ValueError('Invalid snapshot size')
    csv.field_size_limit(MAX_BYTES)
    text=content.decode('utf-8-sig')
    if not text.endswith(('\n','\r')):raise ValueError('Incomplete CSV ending')
    reader=csv.DictReader(io.StringIO(text,newline=''),strict=True)
    if not {'Date','Player','Model_Stats_Season'}.issubset(reader.fieldnames or []):raise ValueError('Incomplete snapshot schema')
    forbidden={'warlord_admin_password','warlord_ledger_database_url','odds_api_key','balldontlie_api_key','password','private_pick'}
    if forbidden.intersection(f.strip().casefold() for f in reader.fieldnames):raise ValueError('Not a raw public model tracker')
    first=None;n=0;identity=None;current_season_rows=0
    for row in reader:
        if None in row or any(v is None for v in row.values()) or not str(row.get('Player','')).strip():raise ValueError('Incomplete snapshot row')
        if first is None:first=row;identity=_metadata(row)
        if str(row.get('Date'))[:10]!=identity[0].isoformat() or row.get('Model_Stats_Season') not in (identity[1],'','Unavailable'):raise ValueError('Mixed slate snapshot')
        current_season_rows+=row.get('Model_Stats_Season')==identity[1]
        n+=1
    if first is None:raise ValueError('Empty snapshot')
    if not current_season_rows:raise ValueError('Snapshot has no current-season source rows')
    return Snapshot(content,*identity,sha256(content).hexdigest(),n)

def inspect_snapshot(content):
    try:return _inspect_snapshot(content)
    except (UnicodeError,csv.Error,OverflowError) as error:
        raise ValueError('Unreadable completed snapshot') from None

def _connect(database_url):
    from ledger_store import _connect as connect
    return connect(database_url)

def _ensure(cursor):
    cursor.execute("SET LOCAL lock_timeout = '3s'")
    cursor.execute("SET LOCAL statement_timeout = '6s'")
    cursor.execute('''CREATE TABLE IF NOT EXISTS warlord_slate_snapshots (
        slate_date date NOT NULL, model_season text NOT NULL,
        built_at timestamptz NOT NULL, quoted_at timestamptz NOT NULL,
        sha256 text NOT NULL, row_count integer NOT NULL CHECK(row_count>0),
        payload bytea NOT NULL, saved_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (slate_date,model_season))''')

class SnapshotStore:
    def __init__(self,database_url,connect=None):
        self._connect=connect or (lambda:_connect(database_url))

    def save(self,snapshot):
        """Per-day monotonic upsert: a late older refresh cannot replace newer data."""
        payload=gzip.compress(snapshot.content,mtime=0)
        with self._connect() as conn,conn.cursor() as cur:
            _ensure(cur)
            cur.execute('''INSERT INTO warlord_slate_snapshots
                (slate_date,model_season,built_at,quoted_at,sha256,row_count,payload)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (slate_date,model_season) DO UPDATE SET
                    built_at=EXCLUDED.built_at,quoted_at=EXCLUDED.quoted_at,
                    sha256=EXCLUDED.sha256,row_count=EXCLUDED.row_count,
                    payload=EXCLUDED.payload,saved_at=now()
                WHERE (EXCLUDED.built_at,EXCLUDED.quoted_at)>
                    (warlord_slate_snapshots.built_at,warlord_slate_snapshots.quoted_at)''',
                (snapshot.slate_date,snapshot.season,snapshot.built_at,snapshot.quoted_at,snapshot.sha,snapshot.rows,payload))
            cur.execute('SELECT sha256 FROM warlord_slate_snapshots WHERE slate_date=%s AND model_season=%s',
                        (snapshot.slate_date,snapshot.season))
            stored=cur.fetchone()
            return bool(stored and stored[0]==snapshot.sha)

    def latest(self):
        with self._connect() as conn,conn.cursor() as cur:
            _ensure(cur)
            cur.execute('''SELECT slate_date,model_season,built_at,quoted_at,sha256,row_count,payload
                FROM warlord_slate_snapshots ORDER BY slate_date DESC,built_at DESC,quoted_at DESC LIMIT 1''')
            row=cur.fetchone()
        if not row:return None
        day,season,built,quoted,digest,n,payload=row
        if len(payload)>MAX_BYTES:raise ValueError('Stored snapshot too large')
        with gzip.GzipFile(fileobj=io.BytesIO(bytes(payload))) as zipped:content=zipped.read(MAX_BYTES+1)
        snap=inspect_snapshot(content)
        if (snap.slate_date,snap.season,snap.built_at,snap.quoted_at,snap.sha,snap.rows)!=(day,season,built,quoted,digest,n):
            raise ValueError('Stored snapshot checksum or metadata mismatch')
        return snap

@lru_cache(maxsize=512)
def _file_rank(path,size,mtime):
    try:
        if not 0<size<=MAX_BYTES:return None
        csv.field_size_limit(MAX_BYTES)
        with open(path,encoding='utf-8-sig',newline='') as f:first=next(csv.DictReader(f))
        day,season,built,quoted=_metadata(first)
        return day,built,quoted
    except (OSError,ValueError,StopIteration,UnicodeError,csv.Error):return None

def local_snapshot(output_dir,extra_paths=()):
    """Read model timestamps, not mtime. In-progress .tmp files are never candidates."""
    root=Path(output_dir).resolve();paths=set(root.glob('tracker_*.csv'))
    for p in extra_paths:
        if p:
            q=Path(p).resolve()
            if q.parent==root:paths.add(q)
    ranked=[]
    for p in paths:
        if p.name.endswith('_GRADED.csv'):continue
        try:s=p.stat();rank=_file_rank(str(p),s.st_size,s.st_mtime_ns)
        except OSError:continue
        if rank:ranked.append((rank,p))
    for rank,p in sorted(ranked,key=lambda x:(x[0],str(x[1])),reverse=True):
        try:
            before=p.stat();content=p.read_bytes();after=p.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):continue
            return inspect_snapshot(content),str(p)
        except (OSError,ValueError,UnicodeError,csv.Error):continue
    return None,None

def best_snapshot(*candidates):
    valid=[(source,s) for source,s in candidates if isinstance(s,Snapshot)]
    return max(valid,key=lambda item:item[1].rank) if valid else (None,None)

def restore_local(output_dir,*,durable=None,published=None,extra_paths=()):
    with _LOCAL_COMMIT_LOCK:
        local,path=local_snapshot(output_dir,extra_paths)
        source,selected=best_snapshot(('local',local),('durable',durable),('published',published))
        if selected:
            from published_output import save_if_changed
            save_if_changed(Path(output_dir)/'tracker_latest.csv',selected.content)
            return selected,source
        return None,None

def promote_completed(path,stable_path):
    """Stable pointer advances under the same lock used by reboot restoration."""
    with _LOCAL_COMMIT_LOCK:
        # Comparing all completed locals also prevents a historical/manual run
        # from replacing an already generated newer slate's stable pointer.
        snapshot,_=local_snapshot(Path(stable_path).parent,extra_paths=(path,))
        if snapshot:
            from published_output import save_if_changed
            save_if_changed(Path(stable_path),snapshot.content)

def atomic_tracker_csv(frame,path):
    """Expose a timestamped .csv only after the entire write succeeds."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',newline='',dir=path.parent,prefix='.tracker-',suffix='.tmp',delete=False) as handle:
            temporary=Path(handle.name);frame.to_csv(handle,index=False);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return str(path)

def backup_completed(path,database_url,store=None):
    if not database_url:return dict(ok=False,message='Backup storage is not connected.')
    try:
        snap=inspect_snapshot(Path(path).read_bytes())
        ok=(store or SnapshotStore(database_url)).save(snap)
        return dict(ok=ok,info=snap.public_info(),message='Saved securely; survives reboot.' if ok else 'A different newer snapshot is already saved. Current run was not substituted.')
    except Exception:
        # Never echo a database URL, database exception or raw payload into the UI/log.
        return dict(ok=False,message='The slate loaded, but its secure backup was not confirmed. Retry backup before rebooting.')
