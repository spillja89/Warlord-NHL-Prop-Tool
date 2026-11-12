"""Small card checkboxes; authenticated writes and isolated public Tail list."""
from html import escape
from pathlib import Path
import json

import streamlit as st
from played_picks import PickStore, make_pick, pregame, results_for_picks

_JS = r"""
export default function({data, setTriggerValue}) {
  const controller = new AbortController();
  function attach() {
    document.querySelectorAll('[data-warlord-pick]').forEach(box => {
      if (box.dataset.bridge === data.token) return;
      box.dataset.bridge = data.token;
      box.querySelectorAll('input[data-pick-field]').forEach(input => {
        input.addEventListener('change', () => {
          box.querySelectorAll('input').forEach(el => el.disabled = true);
          const note = box.querySelector('[role="status"]');
          if (note) note.textContent = 'Saving…';
          setTriggerValue('action', {pick_id: box.dataset.warlordPick,
            field: input.dataset.pickField, value: input.checked,
            revision: Number(box.dataset.revision)});
        }, {signal: controller.signal});
      });
    });
  }
  attach();
  const observer = new MutationObserver(attach);
  observer.observe(document.body, {childList:true, subtree:true});
  return () => { controller.abort(); observer.disconnect(); };
}
"""
_bridge = st.components.v2.component('warlord_pick_marks', js=_JS)
_CSS = """<style>
.wn-pick-controls{display:flex;flex-wrap:wrap;gap:8px 17px;align-items:center;margin-top:8px;padding:7px 9px;border:1px solid #52657e;border-radius:7px;background:#102136;color:#edf4ff;font-size:13px}
.wn-pick-controls label{display:flex;align-items:center;gap:7px;cursor:pointer;font-weight:700}
.wn-pick-controls input{width:18px;height:18px;accent-color:#67dfad;cursor:pointer}
.wn-pick-controls input:disabled{cursor:default}.wn-pick-controls [role=status]{color:#b9cadd;font-size:11px}
</style>"""


def storage_url(setting):
    url = setting('WARLORD_LEDGER_DATABASE_URL')
    return url if url.startswith(('postgres://', 'postgresql://')) else ''


def apply_card_action(store, registry, states, action, *, owner_access):
    """Resolve the quote on the server; browser events carry only key/action."""
    if not owner_access or not isinstance(action, dict) or action.get('pick_id') not in registry:
        raise ValueError('This card is no longer available. Refresh the board.')
    pick = registry[action['pick_id']]
    expected = int(states.get(pick['pick_id'], {}).get('revision', 0))
    if type(action.get('revision')) is not int or action['revision'] != expected:
        raise ValueError('The mark changed. Please try again on the refreshed card.')
    store.change(pick, action.get('field'), action.get('value'), expected)


def render_markable_boards(boards, frame, *, render, owner_access, setting, key, **kwargs):
    """Public rendering never reads private marks or mounts owner controls."""
    if not owner_access:
        st.html(render(boards, **kwargs))
        return
    url = storage_url(setting)
    store, states, error = (PickStore(url) if url else None), {}, ''
    if store:
        try:
            states = store.private_rows()
        except Exception:
            error = 'Pick storage is unavailable. Nothing was marked or shared. Try again shortly.'
    else:
        error = 'Played marks need private storage connected before they can save across reboots.'
    registry = {}
    rows = frame.to_dict('records')

    def controls(card):
        candidates = [row for row in rows if str(row.get('Player', '')).strip() == card['player']
                      and str(row.get('Team', '')).strip() == card['team']
                      and str(row.get('Game', '')).strip() == card['game']]
        picks = []
        for row in candidates:
            try:
                picks.append(make_pick(card, row))
            except (ValueError, TypeError, KeyError):
                continue
        if not picks or len({p['pick_id'] for p in picks}) != 1:
            return '<div class="wn-pick-controls">Played unavailable · missing player/game identity</div>'
        pick = picks[0]
        registry[pick['pick_id']] = pick
        state = states.get(pick['pick_id'], {})
        played, shared = bool(state.get('played')), bool(state.get('shared'))
        revision = int(state.get('revision', 0))
        disabled = ' disabled' if error else ''
        share_disabled = ' disabled' if error or not played or (not shared and not pregame(pick)) else ''
        note = 'Saved privately' if played else 'Private mark'
        if shared:
            note = 'Saved · shared on Tail list'
        if error:
            note = 'Saving unavailable'
        return (f'<div class="wn-pick-controls" data-warlord-pick="{pick["pick_id"]}" data-revision="{revision}">'
                f'<label><input type="checkbox" data-pick-field="played" aria-label="Played: {escape(pick["player"])} {pick["market"]} over {pick["line"]}"'
                f'{" checked" if played else ""}{disabled}>✓ Played</label>'
                f'<label title="Publish this pick and displayed price. Your personal stakes stay private."><input type="checkbox" data-pick-field="shared" aria-label="Share to Tail list: {escape(pick["player"])} {pick["market"]} over {pick["line"]}"'
                f'{" checked" if shared else ""}{share_disabled}>Share to Tail list</label>'
                f'<span role="status">{note}</span></div>')

    st.html(_CSS + render(boards, card_controls=controls, **kwargs))
    if error:
        st.warning(error)
        if not store:
            st.caption('Use the existing private PostgreSQL ledger connection. No private picks are stored in the public repository.')
        return
    flash = st.session_state.pop('_pick_mark_notice', '')
    if flash:
        st.info(flash)
    # Reruns may remove/recreate st.html nodes, so bind to the new control nodes.
    import uuid
    result = _bridge(data={'token': uuid.uuid4().hex}, key=key, on_action_change=lambda: None)
    action = result.action
    if action:
        try:
            apply_card_action(store, registry, states, action, owner_access=owner_access)
        except ValueError as exc:
            st.session_state['_pick_mark_notice'] = str(exc)
        except Exception:
            st.session_state['_pick_mark_notice'] = 'Save failed. Your previous mark is unchanged; please try again.'
        st.rerun()


def _return(odds, status):
    if status == 'Win':
        return odds / 100 if odds > 0 else 100 / abs(odds)
    return -1.0 if status == 'Loss' else 0.0


def render_pick_list(*, public, owner_access, setting, output_dir):
    st.subheader('📣 Tail list' if public else '✓ My Played picks')
    if not public and not owner_access:
        st.info('Unlock Owner controls to see your private Played picks.')
        return
    url = storage_url(setting)
    if not url:
        st.info('The Tail list is not connected yet.' if public else 'Private pick storage needs to be connected before marks can save.')
        return
    store = PickStore(url)
    try:
        picks = store.public_rows() if public else [r for r in store.private_rows().values() if r['played']]
    except Exception:
        st.info('Pick storage is temporarily unavailable. Please try again shortly.')
        return
    st.caption('Only picks explicitly shared by the owner appear here. Prices are the saved published quotes; no bets are placed by this page.' if public else
               'Private marks only. The saved card price is a reference quote, not a claim about the odds or stake you actually bet. Unchecking Played also removes it from the Tail list.')
    if not picks:
        st.info('No shared picks yet.' if public else 'Check Played on a card to keep it here.')
        return
    # Date filter is deliberate: older picks remain saved and can be reviewed.
    dates = sorted({p['date'] for p in picks}, reverse=True)
    chosen = st.selectbox('Slate', ['All saved dates', *dates], index=1,
                          key='tail_date' if public else 'played_date')
    selected = [p for p in picks if chosen == 'All saved dates' or p['date'] == chosen]
    outcomes = results_for_picks(selected, Path(output_dir) / 'graded')
    wins = sum(outcomes.get(p['pick_id']) == 'Win' for p in selected)
    losses = sum(outcomes.get(p['pick_id']) == 'Loss' for p in selected)
    pending = sum(outcomes.get(p['pick_id'], 'Pending') == 'Pending' for p in selected)
    st.write(f'**{wins} wins · {losses} losses · {pending} pending**')
    st.caption('Results use saved official grades. Unmatched or conflicting player results stay pending. Personal profit is not inferred from a checkmark.')
    table = []
    for p in selected:
        status = outcomes.get(p['pick_id'], 'Pending')
        table.append({'Player': p['player'], 'Game': p['game'], 'Date': p['date'],
                      'Pick': f'Over {p["line"]:g} {p["market"]}',
                      'Saved quote': f'{p["odds"]:+g} · {p["book"]}', 'Result': status,
                      **({'Shared at (UTC)': p['shared_at']} if public else
                         {'On Tail list': 'Yes' if p['shared'] else 'No', 'Marked at (UTC)': p.get('marked_at', '')})})
    import pandas as pd
    st.dataframe(pd.DataFrame(table), hide_index=True, width='stretch')
    settled = [p for p in selected if outcomes.get(p['pick_id']) in {'Win', 'Loss'}]
    if settled:
        units = sum(_return(p['odds'], outcomes[p['pick_id']]) for p in settled)
        st.caption(f'Saved-price reference only · flat 1u per settled pick: {units:+.2f}u · {100*units/len(settled):+.1f}% ROI. This is not the owner’s actual stake or return.')
    if public:
        st.caption('This view tracks the picks currently shared on this list; withdrawn picks are not included.')
    elif selected:
        with st.expander('Manage a saved pick', expanded=False):
            by_id = {p['pick_id']:p for p in selected}
            pick_id = st.selectbox('Saved pick',list(by_id),format_func=lambda k:
                                  f'{by_id[k]["player"]} · Over {by_id[k]["line"]:g} {by_id[k]["market"]} · {by_id[k]["date"]}')
            pick = by_id[pick_id]
            remove_tail, unmark = st.columns(2)
            action = None
            if remove_tail.button('Remove from Tail list', disabled=not pick['shared']):
                action = 'shared'
            if unmark.button('Unmark Played'):
                action = 'played'
            if action:
                try:
                    store.change(pick, action, False, pick['revision'])
                except Exception:
                    st.error('Could not update this pick. Reload and try again.')
                else:
                    st.rerun()
