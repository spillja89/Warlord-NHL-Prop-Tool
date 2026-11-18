"""Compact release notes for every prop, separate from routine grading dates."""
from pathlib import Path
import json,re
from functools import lru_cache
ROOT=Path(__file__).resolve().parent
FILES=('goal_return_records.json','assist_return_records.json','points_return_records.json','special_return_records.json')
def entries():return _entries(tuple((ROOT/f).stat().st_mtime_ns for f in FILES))
@lru_cache(maxsize=8)
def _entries(signature):
 docs={f:json.loads((ROOT/f).read_text(encoding='utf-8')) for f in FILES};out=[]
 groups=[('Goals .5',FILES[0],lambda r:True,'Current Goal player routes and optional environment buffs.'),('Assists .5',FILES[1],lambda r:True,'Established Support routes plus Assist V2 additions; optional goalie context.'),('Points .5',FILES[2],lambda r:r.get('line')==.5,'V6 Tank profiles and combos; Shattered Aegis optional GAA2.95+ buff. Healthy existing moves preserved.'),('Points 1.5',FILES[2],lambda r:r.get('line')==1.5 and not r.get('legacy'),'Existing 1.5 Tank routes preserved; V5 combined-record patch.'),('Shots 2.5',FILES[3],lambda r:r.get('market')=='SOG' and not r.get('research_monitor'),'Current shot routes; separate research monitors remain labelled.'),('Power Play .5',FILES[3],lambda r:r.get('market')=='PPP','Layer1 matchup and four player signals; table research and tracked rule records. No new PP move patch in V6.')]
 for label,file,select,note in groups:
  doc=docs[file];rs=[r for r in doc['moves'] if select(r)]
  versions=[]
  for r in rs:
   x=str(r.get('move_version_label') or doc.get('move_version_label') or '');m=re.search(r'V(\d+)',x)
   if m:versions.append(int(m.group(1)))
  v=max(versions,default=None);dates=[r.get('wired_on') or doc.get('wired_on') for r in rs];dates=[d for d in dates if d]
  out.append({'Prop':label,'Version':f'V{v}' if v else 'Tracked research','Updated':max(dates) if dates else 'Date unavailable','Update':note,'Results graded through':doc.get('full_through','—')})
 return out
def render(st):
 st.caption('⚔️ Move updates · Points .5 V6 — new Tank paths + Shattered Aegis · Nov18')
 with st.expander('Version & move updates · all props',expanded=False):
  st.table(entries())
  st.caption('Move versions identify approved patches. Results continue updating after grading; a grading refresh alone does not replace the move rules. Full rules and return periods remain in Inspect.')
