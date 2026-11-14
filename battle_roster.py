"""Read-only, privacy-aware deployment showcase for owner-selected picks."""
from html import escape
import streamlit as st
from matchup_display import matchup_label

ROLES={'Goal':'Carry','Assists':'Support','Points':'Tank','SOG':'Jungle','PPP':'Support'}


def roster_picks(store, day, *, owner_access):
    # One shared deployment view for everyone. Never publish older private history.
    rows=store.public_rows()
    return [row for row in rows if row.get('date')==day]


def roster_html(picks, boards):
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
        tiles.append('<article class="battle-champion">'+art+'<div>'
                     f'<strong>{escape(str(lead.get("player") or "Champion"))}</strong>'
                     f'<small>{escape(matchup_label(lead.get("game"),lead.get("start_utc")))}</small>'
                     +''.join(props)+'</div></article>')
    return '''<style>
      .battle-roster{display:grid;grid-template-columns:repeat(auto-fit,minmax(285px,1fr));gap:10px;margin:8px 0 18px}
      .battle-champion{display:flex;gap:12px;align-items:center;background:linear-gradient(125deg,#172436,#2c2634);border:1px solid #88734c;border-left:3px solid #e8bb55;border-radius:10px;padding:10px;color:#f1f5fb;font:13px/1.45 system-ui,sans-serif;min-width:0}
      .battle-champion img{width:82px;height:100px;object-fit:cover;object-position:center top;border:1px solid #b69b62;border-radius:7px;flex:none}
      .battle-champion strong{display:block;font-size:17px;color:#fff}.battle-champion small{display:block;color:#c2d1e5;margin:3px 0 7px}
      .battle-prop{display:block;color:#e1e9f7;font-size:12px}.battle-prop b{color:#ffd982}
      @media(max-width:450px){.battle-roster{grid-template-columns:1fr}}
    </style><div class="battle-roster">'''+''.join(tiles)+'</div>'


@st.fragment
def render_battle_roster(day, boards, *, owner_access, setting):
    from played_cards import storage_url
    from played_picks import PickStore
    st.subheader("⚔️ Tonight’s Warband")
    st.caption('Chosen to battle the books · Played champions, deployed for tonight.')
    st.button('↻ Update warband',key='refresh_battle_roster',help='Refreshes only this roster, without rebuilding the board or running the slate.')
    url=storage_url(setting)
    if not url:
        st.info('Your warband will appear here once pick storage is connected.')
        return
    try:
        picks=roster_picks(PickStore(url),day,owner_access=owner_access)
    except Exception:
        st.info('The warband is temporarily unavailable. Your saved picks have not changed.')
        return
    if not picks:
        st.caption('Mark Played on a card, then Update warband to deploy your champions here.'
                   if owner_access else 'No champions shared for this slate yet.')
        return
    unique={(pick.get('game_id'),pick.get('player_id')) for pick in picks}
    st.caption(f'{len(unique)} champion' + ('s' if len(unique)!=1 else '')
               + f' deployed · {len(picks)} selected prop' + ('s' if len(picks)!=1 else '')
               + ' · Saved reference prices')
    st.html(roster_html(picks,boards))
