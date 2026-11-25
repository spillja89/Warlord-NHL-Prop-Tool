"""Read-only, privacy-aware deployment showcase for owner-selected picks."""
from html import escape
import streamlit as st
from matchup_display import matchup_label
from slate_browsing import CHOSEN_ORDERS, TIER_ORDER, browse_picks, prop_label, slot_key, slot_options
from chosen_priority import assign_priorities, priority_key, priority_html, book_be, number

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
      setTriggerValue('selection',{champion:button.dataset.battleChampion,pick_id:button.dataset.battlePick,nonce:crypto.randomUUID()});
    },{signal:controller.signal}));
  }
  attach();
  return()=>{clearTimeout(timer);controller.abort();};
}
"""
_champion_bridge=st.components.v2.component('warlord_champion_details',js=_CHAMPION_JS)


def champion_key(pick):
    return str(pick.get('game_id') or '')+':'+str(pick.get('player_id') or '')


def default_pick(selections):
    """Open a selected regular-prop card before a PP research panel."""
    return next((pick for pick in selections if pick.get('market')!='PPP'),selections[0])


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
                  portrait_variant='Celestial Sage',
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
        from pp_cards import rank_pp_cards,render_pp_cards
        from warlords_night_board import _character_uri
        cards=rank_pp_cards(pd.DataFrame([row]),quotes)
        art=_character_uri('Support','Celestial Sage')
        details=render_pp_cards(cards,art,character_override='Panda · Celestial Sage · Power Play Support').replace('<details class="pp-w-inspect">','<details class="pp-w-inspect" open>') if cards else ''
        return (details if cards else render_power_play_form(view))+beta_details(quote.get('PP Assist beta'))
    return ('<p>The selected line is not offered on this snapshot. Current move details are unavailable for that exact line.</p>'
            +_form_html(dict(player=pick['player'],market=pick['market'],line=pick['line'],
                             form_log=row.get('Form_Log'),form_season=row.get('Form_Season')),expanded=True))


@st.dialog('Champion details',width='large')
def _open_champion(selections, boards, frame, initial_pick_id=None, request_key='default'):
    by_id={pick['pick_id']:pick for pick in selections}
    initial=initial_pick_id if initial_pick_id in by_id else default_pick(selections)['pick_id']
    selected=st.radio('Played prop',list(by_id),index=list(by_id).index(initial),horizontal=True,
                      key='chosen_prop_'+request_key,format_func=lambda key:
                      f'Over {by_id[key]["line"]:g} {"Power-play point" if by_id[key]["market"]=="PPP" else by_id[key]["market"]} · {by_id[key]["odds"]:+g} · {by_id[key].get("book","")}')
    pick=by_id[selected]
    st.caption(f'{pick["player"]} · Played selection at the saved quote above. Card details use the loaded slate snapshot.')
    st.html(champion_details_html(pick,boards,frame))


def roster_picks(store, day, *, owner_access):
    # One shared deployment view for everyone. Never publish older private history.
    rows=store.public_rows()
    return [row for row in rows if row.get('date')==day]


def pick_metadata(picks, boards, frame):
    from move_return_grades import return_record,grade_record,stable_identity,pp_move_rank
    from warlords_night_board import _move_rank
    pp_by_id={}
    if frame is not None and any(p.get('market')=='PPP' for p in picks):
        from power_play_quotes import priced_ppp_quotes
        from pp_cards import rank_pp_cards
        from played_picks import numeric_id
        quotes=priced_ppp_quotes(frame,line_filter=.5)
        for card in rank_pp_cards(frame,quotes):
            row=card['row']
            pp_by_id[(numeric_id(row.get('Game_ID')),numeric_id(row.get('Player_ID')),str(row.get('Date') or '')[:10])]=card
    metadata = {}
    records={}
    for pick in picks:
        card, _ = selected_card(pick, boards, frame) if frame is not None else (None, None)
        move = (card.get('move') or card.get('points_current_tier')) if card else None
        if pick.get('market')=='PPP':
            from played_picks import numeric_id
            card=pp_by_id.get((numeric_id(pick.get('game_id')),numeric_id(pick.get('player_id')),pick.get('date')))
            move=card.get('best') if card and number(pick.get('line'))==.5 else None
        identity=stable_identity(move) if move else ''
        if move and identity not in records:
            records[identity]=return_record(move)
        record=records.get(identity)
        grade=grade_record(record) if record else None
        full=record.get('full') if record else None
        recent=record.get('since_check') if record else None
        be=book_be(pick.get('odds'))
        hits=number((full or {}).get('hits'));n=number((full or {}).get('plays'))
        rate=100*hits/n if hits is not None and n else None
        rank=(pp_move_rank(move) if pick.get('market')=='PPP' else _move_rank(move)) if move else None
        metadata[pick['pick_id']] = {'tier': grade['tier'] if grade else 'Unrated',
            'watch': bool(grade and grade['watch']),'card_rank':rank,'move_identity':identity,
            'move_name':move.get('name') if move else None,'full':full,'recent':recent,
            'saved_be':be,'saved_gap':rate-be if rate is not None and be is not None else None,
            'confidence':card.get('confidence') if card and pick.get('market')!='PPP' else None}
    return assign_priorities(picks,metadata)


def roster_html(picks, boards, *, scope='chosen_champion_details', metadata=None):
    from warlords_night_board import _class_identity, _character_uri
    cards=[card for lane in boards.values() for card in lane]
    champions={}
    for pick in picks:
        key=(pick.get('game_id'),pick.get('player_id'))
        champions.setdefault(key,[]).append(pick)
    tiles=[]
    for selections in champions.values():
        selections=sorted(selections,key=lambda p:priority_key(p,metadata)) if metadata else selections
        lead=selections[0] if metadata else default_pick(selections)
        role=ROLES.get(lead.get('market'),'Support')
        card=next((card for card in cards if card.get('player')==lead.get('player')
                   and card.get('game')==lead.get('game') and card.get('market')==lead.get('market')
                   and card.get('line')==lead.get('line')),None)
        variant='Celestial Sage' if lead.get('market')=='PPP' else ''
        if card:
            _,variant=_class_identity(role,card,card.get('points_current_tier') or card.get('move'))
        elif lead.get('market')!='PPP':
            _,variant=_class_identity(role,{'line':lead.get('line')},None)
        image=_character_uri(role,variant)
        art=f'<img src="{image}" alt="{escape(role)} champion" loading="lazy">' if image else ''
        props=[]
        for pick in selections:
            market='Power-play point' if pick['market']=='PPP' else pick['market']
            info = (metadata or {}).get(pick['pick_id'], {})
            tier = info.get('tier', 'Unrated')
            priority=info.get('priority')
            number_label=f'#{priority:02d} · ' if isinstance(priority,int) else ''
            badge = (f'<span class="battle-tier battle-tier-{escape(tier.lower())}">{number_label}{escape(tier)}'
                     + (' · WATCH' if info.get('watch') else '') + '</span>') if pick['pick_id']!=lead['pick_id'] or not metadata else ''
            props.append(f'<button type="button" class="battle-prop" data-battle-champion="{escape(champion_key(pick),quote=True)}"'
                         f' data-battle-pick="{escape(pick["pick_id"],quote=True)}" data-battle-scope="{escape(scope,quote=True)}"'
                         f' aria-label="Open {escape(str(pick.get("player") or "champion"),quote=True)} over {float(pick["line"]):g} {escape(market,quote=True)}">'
                         f'Over {float(pick["line"]):g} {escape(market)}'
                         f' <b>{float(pick["odds"]):+g}</b> · {escape(str(pick.get("book") or ""))} {badge}</button>')
        lead_info=(metadata or {}).get(lead['pick_id'],{})
        tiles.append(f'<article class="battle-champion" data-chosen-priority="{lead_info.get("priority", "")}"><button type="button" class="battle-champion-main" data-battle-champion="{escape(champion_key(lead),quote=True)}"'
                     f' data-battle-pick="{escape(lead["pick_id"],quote=True)}" data-battle-scope="{escape(scope,quote=True)}"'
                     f' aria-label="Open {escape(str(lead.get("player") or "champion"),quote=True)} champion card"></button>'+art+'<div class="battle-body">'
                     f'<strong>{escape(str(lead.get("player") or "Champion"))}</strong>'
                     f'<small>{escape(matchup_label(lead.get("game"),lead.get("start_utc")))}</small>'
                     +priority_html(lead_info)
                     +''.join(props)+'<span class="battle-open">Click a prop to inspect →</span></div></article>')
    return '''<style>
      .battle-roster{display:grid;grid-template-columns:repeat(auto-fit,minmax(285px,1fr));gap:10px;margin:8px 0 18px}
      .battle-champion{display:flex;gap:12px;align-items:center;background:linear-gradient(125deg,#172436,#2c2634);border:1px solid #88734c;border-left:3px solid #e8bb55;border-radius:10px;padding:10px;color:#f1f5fb;font:13px/1.45 system-ui,sans-serif;min-width:0}
      .battle-champion img{width:82px;height:100px;object-fit:cover;object-position:center top;border:1px solid #b69b62;border-radius:7px;flex:none}
      .battle-champion strong{display:block;font-size:17px;color:#fff}.battle-champion small{display:block;color:#c2d1e5;margin:3px 0 7px}
      .battle-prop{display:block;width:100%;text-align:left;background:none;border:0;padding:2px 0;color:#e1e9f7;font:12px/1.45 system-ui,sans-serif;cursor:pointer;pointer-events:auto}.battle-prop b{color:#ffd982}
      .battle-prop:hover,.battle-prop:focus-visible{color:#fff2c9;text-decoration:underline;outline:1px solid #f4d78f;outline-offset:2px}
      .battle-tier{display:inline-block;margin:3px 0 0 4px;padding:1px 5px;border-radius:4px;background:#34455d;color:#e3eaf6;font-size:10px;font-weight:850}
      .battle-tier-s{background:#f3ce6c;color:#211b0c}.battle-tier-a{background:#8bdbc9;color:#102b29}.battle-tier-b{background:#b7c5d9;color:#172337}.battle-tier-review{background:#4d341d;color:#ffe0a0}
      .battle-priority-summary{display:grid;gap:3px;margin:5px 0 7px;font-size:11px;line-height:1.4}
      .battle-priority{justify-self:start;border:1px solid #ffffff40;border-radius:4px;padding:3px 7px;font-size:12px;font-weight:900;color:#eff5ff;background:#34455d}
      .battle-priority.battle-tier-s{background:#f3ce6c;color:#211b0c}.battle-priority.battle-tier-a{background:#8bdbc9;color:#102b29}.battle-priority.battle-tier-b{background:#b7c5d9;color:#172337}
      .battle-move{font-weight:800;color:#ffe2a0;font-size:12px}.battle-evidence{color:#dce7f8}.battle-evidence b{color:#fff}
      .battle-price{color:#b8cbe4;font-size:10px}.battle-profit,.battle-profit b{color:#8de6b5}.battle-loss,.battle-loss b{color:#ffaaa2}
      .battle-watch{color:#ffdc99;font-weight:850}.battle-body{min-width:0;flex:1}
      .battle-champion{position:relative;width:100%;text-align:left}.battle-champion:hover,.battle-champion:focus-within{border-color:#ffda7d;box-shadow:0 0 14px #e9ba5540}
      .battle-champion-main{position:absolute;inset:0;z-index:1;width:100%;height:100%;border:0;border-radius:10px;background:none;cursor:pointer}
      .battle-champion-main:focus-visible{outline:2px solid #ffda7d;outline-offset:2px}
      .battle-body,.battle-champion img{position:relative;z-index:2;pointer-events:none}
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
    metadata = pick_metadata(picks, boards, frame)
    first, second, third = st.columns(3)
    with first:
        props = st.multiselect('Chosen props', sorted({prop_label(pick) for pick in picks}),
                               key='chosen_props', placeholder='All props')
    with second:
        tiers = st.multiselect('Chosen tiers', [tier for tier in TIER_ORDER if any(info['tier'] == tier for info in metadata.values())],
                               key='chosen_tiers', placeholder='All tiers',
                               help='The current tier for that exact prop. Unrated includes PP research picks and picks without a matching current move.')
    with third:
        labels = slot_options(slot_key(pick.get('start_utc')) for pick in picks)
        slots = st.multiselect('Chosen start times (CT)', list(labels), format_func=labels.get,
                               key='chosen_slots', placeholder='All start times')
    first, second = st.columns(2)
    with first:
        group = st.selectbox('Group champions by', ('Champion', 'Prop', 'Tier', 'Start time'), key='chosen_group_v2')
    with second:
        order = st.selectbox('Sort champions by', CHOSEN_ORDERS, key='chosen_order_v2')
    st.caption('Priority follows the full card ranking: tier, recent health, total move ROI, units and sample. '
               'Ties use the saved-price gap, then profile confidence. Numbers rank all chosen props; filters and groups keep those numbers. '
               'Records describe move history, not tonight’s hit probability or your personal returns.')
    groups = browse_picks(picks, metadata, props=props, tiers=tiers, slots=slots, order=order, group=group)
    visible = [pick for selections in groups.values() for pick in selections]
    unique={(pick.get('game_id'),pick.get('player_id')) for pick in visible}
    st.caption(f'{len(unique)} champion' + ('s' if len(unique)!=1 else '')
               + f' shown · {len(visible)} of {len(picks)} selected props · Saved reference prices · Independent of board filters')
    if not visible:
        st.info('No chosen props match these filters. Clear a filter to see more; your saved picks are unchanged.')
    for label, selections in groups.items():
        if group != 'Champion':
            st.markdown(f'#### {label} · {len(selections)} prop' + ('s' if len(selections) != 1 else ''))
        st.html(roster_html(selections, boards, metadata=metadata))
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
            pick_id=action.get('pick_id')
            request_key=action.get('nonce')
            if pick_id not in {pick['pick_id'] for pick in selections} or not isinstance(request_key,str) or len(request_key)>64:
                return
            st.session_state['chosen_open_request']={'champion':action['champion'],'pick_id':pick_id,'nonce':request_key}
    _champion_bridge(data={'scope':bridge_key,'has_champions':bool(registry)},
                     key=bridge_key,on_selection_change=on_selection)
    request=st.session_state.pop('chosen_open_request',None)
    if request and request['champion'] in registry:
        _open_champion(registry[request['champion']],boards,frame,request['pick_id'],request['nonce'])

