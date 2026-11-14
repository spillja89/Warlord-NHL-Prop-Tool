"""Read-only, privacy-aware deployment showcase for owner-selected picks."""
from html import escape
import streamlit as st
from matchup_display import matchup_label

ROLES={'Goal':'Carry','Assists':'Support','Points':'Tank','SOG':'Jungle','PPP':'Support'}

_CHAMPION_JS=r"""
export default function({data,setTriggerValue}) {
  const controller=new AbortController();
  let timer,attempts=0;
  function attach() {
    if(controller.signal.aborted)return;
    const buttons=Array.from(document.querySelectorAll('[data-battle-champion]'))
      .filter(button=>button.dataset.battleScope===data.scope);
    if(!buttons.length&&data.has_champions&&attempts++<20){timer=setTimeout(attach,50);return;}
    buttons.forEach(button=>button.addEventListener('click',()=>{
      setTriggerValue('selection',{champion:button.dataset.battleChampion,nonce:crypto.randomUUID()});
    },{signal:controller.signal}));
  }
  attach();
  return()=>{clearTimeout(timer);controller.abort();};
}
"""
_champion_bridge=st.components.v2.component('warlord_champion_details',js=_CHAMPION_JS)


def champion_key(pick):
    return str(pick.get('game_id') or '')+':'+str(pick.get('player_id') or '')


def selected_card(pick, boards, frame):
    """Bind details to authoritative game/player/date IDs before resolving a card."""
    from played_picks import numeric_id
    if frame is None:
        return None,None
    rows=[row for row in frame.to_dict('records')
          if numeric_id(row.get('Game_ID'))==numeric_id(pick.get('game_id'))
          and numeric_id(row.get('Player_ID'))==numeric_id(pick.get('player_id'))
          and str(row.get('Date') or '')[:10]==pick.get('date')]
    if not rows:
        return None,None
    row=rows[0]
    card=next((card for lane in boards.values() for card in lane
               if card.get('player')==str(row.get('Player') or '').strip()
               and card.get('team')==str(row.get('Team') or '').strip()
               and card.get('game')==str(row.get('Game') or '').strip()
               and card.get('market')==pick.get('market')
               and card.get('line')==pick.get('line')),None)
    return card,row


def champion_details_html(pick, boards, frame):
    from warlords_night_board import render_warlords,render_power_play_form,_form_html
    card,row=selected_card(pick,boards,frame)
    if row is None:
        return '<p>Details for this saved selection are not in the loaded slate. The saved pick remains on the roster.</p>'
    if card:
        return render_warlords({ROLES[pick['market']]:[card]},roles=[ROLES[pick['market']]],
                               limit=1,show_hero=False).replace('<details class="wn-card-inspect">','<details class="wn-card-inspect" open>')
    if pick.get('market')=='PPP':
        import pandas as pd
        from power_play_quotes import priced_ppp_quotes,format_pk_minutes_per_game,team_pp_goal_research_signals
        from pp_card_form import ppp_l10_label
        from pp_assist_beta import beta_details
        quotes=priced_ppp_quotes(pd.DataFrame([row]),line_filter=.5)
        quote=quotes.iloc[0].to_dict() if not quotes.empty else {}
        view=dict(player=pick['player'],game=matchup_label(pick.get('game'),pick.get('start_utc')),
                  odds=quote.get('Over odds',pick['odds']),book=quote.get('Book',pick.get('book')),
                  form_log=row.get('Form_Log'),form_season=row.get('Form_Season'),
                  pp_unit=quote.get('PP unit'),pp_research=quote.get('PP research signals'),
                  pp_player_signal_details=quote.get('PP player signal details'),
                  pp_onice_xg_game=quote.get('PP on-ice xG/game'),pp_ixg_game=quote.get('PP shooter xG/game'),
                  team_pp_xgf60=row.get('Team_PP_xGF60'),opp_pk_xga60=row.get('Opp_PK_xGA60'),
                  opp_pk_ga60=row.get('Opp_PK_GA60'),
                  opp_pk_time_pg=format_pk_minutes_per_game(row.get('Opp_PK_Time_PG')),
                  ppp_l10=ppp_l10_label(row),ppp_drought=row.get('Drought_PPP'),
                  team_research_signals=team_pp_goal_research_signals(row))
        return render_power_play_form(view)+beta_details(quote.get('PP Assist beta'))
    return ('<p>The selected line is not offered on this snapshot. Current move details are unavailable for that exact line.</p>'
            +_form_html(dict(player=pick['player'],market=pick['market'],line=pick['line'],
                             form_log=row.get('Form_Log'),form_season=row.get('Form_Season')),expanded=True))


@st.dialog('Champion details',width='large')
def _open_champion(selections, boards, frame):
    by_id={pick['pick_id']:pick for pick in selections}
    selected=st.selectbox('Selected prop',list(by_id),format_func=lambda key:
                          f'Over {by_id[key]["line"]:g} {by_id[key]["market"]} · {by_id[key]["odds"]:+g} · {by_id[key].get("book","")}')
    pick=by_id[selected]
    st.caption(f'{pick["player"]} · Played selection at the saved quote above. Card details use the loaded slate snapshot.')
    st.html(champion_details_html(pick,boards,frame))


def roster_picks(store, day, *, owner_access):
    # One shared deployment view for everyone. Never publish older private history.
    rows=store.public_rows()
    return [row for row in rows if row.get('date')==day]


def roster_html(picks, boards, *, scope='chosen_champion_details'):
    from warlords_night_board import _class_identity, _character_uri
    cards=[card for lane in boards.values() for card in lane]
    champions={}
    for pick in picks:
        key=(pick.get('game_id'),pick.get('player_id'))
        champions.setdefault(key,[]).append(pick)
    tiles=[]
    for selections in champions.values():
        lead=selections[0]
        role=ROLES.get(lead.get('market'),'Support')
        card=next((card for card in cards if card.get('player')==lead.get('player')
                   and card.get('game')==lead.get('game') and card.get('market')==lead.get('market')
                   and card.get('line')==lead.get('line')),None)
        variant='Stormcaller' if lead.get('market')=='PPP' else ''
        if card:
            _,variant=_class_identity(role,card,card.get('points_current_tier') or card.get('move'))
        elif lead.get('market')!='PPP':
            _,variant=_class_identity(role,{'line':lead.get('line')},None)
        image=_character_uri(role,variant)
        art=f'<img src="{image}" alt="{escape(role)} champion" loading="lazy">' if image else ''
        props=[]
        for pick in selections:
            market='Power-play point' if pick['market']=='PPP' else pick['market']
            props.append(f'<span class="battle-prop">Over {float(pick["line"]):g} {escape(market)}'
                         f' <b>{float(pick["odds"]):+g}</b> · {escape(str(pick.get("book") or ""))}</span>')
        tiles.append(f'<button type="button" class="battle-champion" data-battle-champion="{escape(champion_key(lead),quote=True)}"'
                     f' data-battle-scope="{escape(scope,quote=True)}" aria-label="Open {escape(str(lead.get("player") or "champion"),quote=True)} champion card">'+art+'<div>'
                     f'<strong>{escape(str(lead.get("player") or "Champion"))}</strong>'
                     f'<small>{escape(matchup_label(lead.get("game"),lead.get("start_utc")))}</small>'
                     +''.join(props)+'<span class="battle-open">Inspect champion →</span></div></button>')
    return '''<style>
      .battle-roster{display:grid;grid-template-columns:repeat(auto-fit,minmax(285px,1fr));gap:10px;margin:8px 0 18px}
      .battle-champion{display:flex;gap:12px;align-items:center;background:linear-gradient(125deg,#172436,#2c2634);border:1px solid #88734c;border-left:3px solid #e8bb55;border-radius:10px;padding:10px;color:#f1f5fb;font:13px/1.45 system-ui,sans-serif;min-width:0}
      .battle-champion img{width:82px;height:100px;object-fit:cover;object-position:center top;border:1px solid #b69b62;border-radius:7px;flex:none}
      .battle-champion strong{display:block;font-size:17px;color:#fff}.battle-champion small{display:block;color:#c2d1e5;margin:3px 0 7px}
      .battle-prop{display:block;color:#e1e9f7;font-size:12px}.battle-prop b{color:#ffd982}
      .battle-champion{width:100%;text-align:left;cursor:pointer}.battle-champion:hover,.battle-champion:focus-visible{border-color:#ffda7d;box-shadow:0 0 14px #e9ba5540;outline:2px solid #ffda7d;outline-offset:2px}
      .battle-open{display:block;margin-top:7px;font-size:11px;font-weight:800;color:#f4d78f}
      @media(max-width:450px){.battle-roster{grid-template-columns:1fr}}
    </style><div class="battle-roster">'''+''.join(tiles)+'</div>'


@st.fragment
def render_battle_roster(day, boards, frame=None, *, owner_access, setting):
    from played_cards import storage_url
    from played_picks import PickStore
    st.subheader("⚔️ The Warlord’s Chosen for Battle")
    st.caption('Champions deployed to cook the books.')
    st.button('↻ Update the Chosen',key='refresh_battle_roster',help='Refreshes only this roster, without rebuilding the board or running the slate.')
    url=storage_url(setting)
    if not url:
        st.info('Your chosen champions will appear here once pick storage is connected.')
        return
    try:
        picks=roster_picks(PickStore(url),day,owner_access=owner_access)
    except Exception:
        st.info('The chosen roster is temporarily unavailable. Your saved picks have not changed.')
        return
    if not picks:
        st.caption('Mark Played on a card, then Update the Chosen to deploy your champions here.'
                   if owner_access else 'No champions shared for this slate yet.')
        return
    unique={(pick.get('game_id'),pick.get('player_id')) for pick in picks}
    st.caption(f'{len(unique)} champion' + ('s' if len(unique)!=1 else '')
               + f' deployed · {len(picks)} selected prop' + ('s' if len(picks)!=1 else '')
               + ' · Saved reference prices')
    st.html(roster_html(picks,boards))
    registry={}
    for pick in picks:
        registry.setdefault(champion_key(pick),[]).append(pick)
    bridge_key='chosen_champion_details'
    def on_selection():
        action=st.session_state.get(bridge_key,{}).get('selection')
        if not isinstance(action,dict):
            return
        selections=registry.get(action.get('champion'))
        if selections:
            _open_champion(selections,boards,frame)
    _champion_bridge(data={'scope':bridge_key,'has_champions':bool(registry)},
                     key=bridge_key,on_selection_change=on_selection)
