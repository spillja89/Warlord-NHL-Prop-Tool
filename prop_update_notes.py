"""Compact release notes for every prop, separate from routine grading dates."""
from pathlib import Path
import json,re
from html import escape
from datetime import date
from functools import lru_cache
ROOT=Path(__file__).resolve().parent
FILES=('goal_return_records.json','assist_return_records.json','points_return_records.json','special_return_records.json')
def entries():return _entries(tuple((ROOT/f).stat().st_mtime_ns for f in FILES))
@lru_cache(maxsize=8)
def _entries(signature):
 docs={f:json.loads((ROOT/f).read_text(encoding='utf-8')) for f in FILES};out=[]
 groups=[('Goals .5',FILES[0],lambda r:True,'Goals V3: eight independent attacks, seven exact-parent optional buffs, plus approved .47 routes. Healthy existing moves preserved.'),('Assists .5',FILES[1],lambda r:True,'Established Support routes plus Assist V2 additions; optional goalie context.'),('Points .5',FILES[2],lambda r:r.get('line')==.5,'V6 Tank profiles and combos; Shattered Aegis optional GAA2.95+ buff. Healthy existing moves preserved.'),('Points 1.5',FILES[2],lambda r:r.get('line')==1.5 and not r.get('legacy'),'Existing 1.5 Tank routes preserved; V5 combined-record patch.'),('Shots 2.5',FILES[3],lambda r:r.get('market')=='SOG' and not r.get('research_monitor'),'SOG V3: steadier-minutes routes, Deadeye Precision and optional buffs. Healthy profiles preserved; Swipe/Rage/Paralysis archived from leads.'),('Power Play .5',FILES[3],lambda r:r.get('market')=='PPP','PP V3: two player routes + seven environment families on .55xG + Layer1;15point card floor, grouped levels, V1/V2 preserved. Table first, compact cards below.')]
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
 st.caption('⚔️ Move updates · Goals V3 — eight attacks + seven optional buffs; healthy existing and .47 routes preserved · Nov23')
 with st.expander('Version & move updates · all props',expanded=False):
  rows=entries()
  def day(v):
   try:return date.fromisoformat(v).strftime('%b %d').replace(' 0',' ')
   except (ValueError,TypeError):return v
  body=''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in [r['Prop'],r['Version'],day(r['Updated']),r['Update'],day(r['Results graded through'])])+'</tr>' for r in rows)
  st.html('<style>.wv-patch-scroll{overflow-x:auto}.wv-patch-table{width:100%;min-width:690px;border-collapse:collapse;font-size:13px;line-height:1.5}.wv-patch-table th,.wv-patch-table td{padding:9px 12px;border-bottom:1px solid #8593a133;text-align:left;vertical-align:top}.wv-patch-table th{font-weight:750;white-space:nowrap}.wv-patch-table td:nth-child(1),.wv-patch-table td:nth-child(2),.wv-patch-table td:nth-child(3),.wv-patch-table td:nth-child(5){white-space:nowrap}.wv-patch-table th:nth-child(2){min-width:55px}</style><div class="wv-patch-scroll"><table class="wv-patch-table"><thead><tr><th>Prop</th><th>Version</th><th>Updated</th><th>Move update</th><th>Graded through</th></tr></thead><tbody>'+body+'</tbody></table></div>')
  st.caption('Move versions identify approved patches. Results continue updating after grading; a grading refresh alone does not replace the move rules. Full rules and return periods remain in Inspect.')

