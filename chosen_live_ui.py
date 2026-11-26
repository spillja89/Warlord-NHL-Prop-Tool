"""Isolated live-badge polling; never rerenders the roster or accesses pick storage."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from html import escape
import requests
import streamlit as st
from chosen_live import apply_results,identifier,scheduled

LIVE_JS=r'''
export default function({data}) {
  // A fragment can mount fresh renderer instances before the previous one is
  // unmounted. Only the newest result set may observe a particular roster.
  const registry=document.__warlordLiveBadges||(document.__warlordLiveBadges=new Map());
  const previous=registry.get(data.scope);if(previous)previous();
  let disposed=false,queued=false;
  const results=data.results||{};
  function text(node,value){if(node.textContent!==value)node.textContent=value;}
  function update(){
    if(disposed)return;
    document.querySelectorAll('[data-chosen-live-pick]').forEach(node=>{
      if(node.dataset.chosenLiveScope!==data.scope)return;
      const r=results[node.dataset.chosenLivePick];if(!r)return;
      text(node,r.label);node.dataset.liveState=r.state;
      node.classList.toggle('chosen-live-stale',!!r.stale);
      node.title=r.stale?'Last verified result; the latest official check is delayed.':
        (r.state==='victory'&&!r.final?'Live official stat; a later scoring correction can update this badge.':'Official NHL game stats; betting settlement remains with your book.');
    });
    document.querySelectorAll('[data-chosen-live-champion]').forEach(node=>{
      if(node.dataset.chosenLiveScope!==data.scope)return;
      let ids=[];try{ids=JSON.parse(node.dataset.chosenLivePicks||'[]');}catch{return;}
      const rows=ids.map(id=>results[id]);const wins=rows.filter(r=>r&&r.state==='victory').length;
      const stale=rows.some(r=>r&&r.stale);
      const label=wins?(wins===ids.length?'⚔ VICTORIOUS · ':'⚔ ')+wins+'/'+ids.length+' props victorious'+(stale?' · update delayed':''):
        (rows.some(r=>r&&r.state==='live')?'⚔ IN BATTLE':'');
      text(node,label);
      const card=node.closest('.battle-champion');
      if(card){card.classList.toggle('battle-all-victorious',wins>0&&wins===ids.length);
        card.classList.toggle('battle-some-victorious',wins>0&&wins<ids.length);}
    });
  }
  function schedule(){if(!queued&&!disposed){queued=true;queueMicrotask(()=>{queued=false;update();});}}
  const observer=new MutationObserver(schedule);
  function cleanup(){disposed=true;observer.disconnect();if(registry.get(data.scope)===cleanup)registry.delete(data.scope);}
  registry.set(data.scope,cleanup);
  observer.observe(document.body,{childList:true,subtree:true});update();
  return cleanup;
}
'''
_live_bridge=st.components.v2.component('warlord_live_victory',js=LIVE_JS)

CSS='''
.chosen-live-prop{display:block;margin-top:3px;color:#bbcce2;font-size:10px;line-height:1.45;font-weight:750}
.chosen-live-prop[data-live-state="victory"]{color:#b1ffd1;background:#134b35;border:1px solid #5bbd89;border-radius:5px;padding:4px 6px}
.chosen-live-prop[data-live-state="loss"]{color:#e6a39d}.chosen-live-prop[data-live-state="push"]{color:#f3d58d}
.chosen-live-stale{opacity:.8}.chosen-live-champion{display:block;color:#8effbe;font-size:11px;font-weight:900;margin-top:5px}
.battle-champion.battle-some-victorious{border-color:#59c18e;border-left-color:#81efb1}
.battle-champion.battle-all-victorious{border-color:#f5dc87;border-left-color:#ffe298;box-shadow:inset 0 0 18px #3cca8b25,0 0 12px #e8cb6440;background:linear-gradient(125deg,#173c32,#282b36)}
'''

def pick_placeholder(pick,scope='chosen_champion_details'):
    return f'<span class="chosen-live-prop" data-chosen-live-pick="{escape(str(pick["pick_id"]),quote=True)}" data-chosen-live-scope="{escape(scope,quote=True)}">Awaiting live check</span>'

@st.cache_data(ttl=45,max_entries=128,show_spinner=False)
def _game(gid,need_pp):
    """Bounded, read-only official requests. No retries inside the live loop."""
    result={};base=f'https://api-web.nhle.com/v1/gamecenter/{gid}/'
    for name,path in [('box','boxscore')]+([('landing','landing')] if need_pp else []):
        try:
            response=requests.get(base+path,timeout=(1.5,2.5),headers={'User-Agent':'Warlord-NHL-Live-Display/1.0'})
            response.raise_for_status();payload=response.json()
            if isinstance(payload,dict) and identifier(payload.get('id'))==gid:result[name]=payload
        except (requests.RequestException,ValueError):pass
    result['checked_utc']=datetime.now(timezone.utc).isoformat()
    return result

def fetch_bundles(picks,now):
    games={}
    for p in picks:
        gid=identifier(p.get('game_id'))
        if gid and not scheduled(p,now):games[gid]=games.get(gid,False) or p.get('market')=='PPP'
    if not games:return {}
    def one(item):return item[0],_game(*item)
    with ThreadPoolExecutor(max_workers=min(8,len(games))) as pool:
        return dict(pool.map(one,games.items()))

@st.fragment(run_every=60)
def render_live_badges(picks,scope='chosen_champion_details'):
    now=datetime.now(timezone.utc)
    bundles=fetch_bundles(picks,now)
    key='chosen_live_results_'+scope
    results=apply_results(picks,bundles,now,st.session_state.get(key,{}))
    st.session_state[key]=results
    _live_bridge(data={'scope':scope,'results':results},key='chosen_live_bridge_'+scope)
