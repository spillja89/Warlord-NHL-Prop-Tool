"""Small card checkboxes; authenticated writes and isolated public Tail list."""
from html import escape
from pathlib import Path
import json

import streamlit as st
from matchup_display import matchup_label
from played_picks import PickStore, make_pick, pregame, results_for_picks

_JS = r"""
export default function({data, setTriggerValue}) {
  const controller = new AbortController();
  let timer;
  let recoveryTimer;
  let attempts = 0;
  function attach() {
    if (controller.signal.aborted) return;
    const boxes = Array.from(document.querySelectorAll('[data-warlord-pick]'))
      .filter(box => box.dataset.pickScope === data.scope);
    // st.html can mount just after the bridge. Retry briefly, never observe the
    // whole document: Streamlit's own rendering must not trigger more scans.
    if (!boxes.length && data.has_cards && attempts++ < 20) {
      timer = setTimeout(attach, 50);
      return;
    }
    // Acknowledgements update only existing controls, never the card DOM.
    boxes.forEach(box => {
      if (box.dataset.savedPlayed === undefined) {
        box.dataset.savedPlayed = String(box.querySelector('[data-pick-field="played"]').checked);
        box.dataset.savedShared = String(box.querySelector('[data-pick-field="shared"]').checked);
      }
      const reply = data.replies[box.dataset.warlordPick];
      if (!reply || box.dataset.ack === reply.request_id) return;
      box.dataset.ack = reply.request_id;
      box.dataset.revision = String(reply.revision);
      box.dataset.canShare = String(reply.can_share);
      box.querySelector('[data-pick-field="played"]').checked = reply.played;
      box.querySelector('[data-pick-field="shared"]').checked = reply.shared;
      box.dataset.savedPlayed = String(reply.played);
      box.dataset.savedShared = String(reply.shared);
      box.querySelector('[role="status"]').textContent = reply.message;
      delete box.dataset.pending;
      delete box.dataset.pendingSince;
    });
    // A dropped acknowledgement must not permanently lock every pick. Restore
    // the last acknowledged display, never claim the uncertain write succeeded.
    const pending = boxes.filter(box => box.dataset.pending);
    if (pending.length) {
      const remaining = Math.max(0, 12000 - (Date.now() - Math.min(...pending.map(box => Number(box.dataset.pendingSince) || Date.now()))));
      recoveryTimer = setTimeout(() => {
        if (controller.signal.aborted) return;
        pending.forEach(box => {
          if (!box.dataset.pending) return;
          box.querySelector('[data-pick-field="played"]').checked = box.dataset.savedPlayed === 'true';
          box.querySelector('[data-pick-field="shared"]').checked = box.dataset.savedShared === 'true';
          box.querySelector('[role="status"]').textContent = 'Save not confirmed · retry';
          delete box.dataset.pending;
          delete box.dataset.pendingSince;
        });
        boxes.forEach(box => box.querySelectorAll('input[data-pick-field]').forEach(input => {
          input.disabled = input.dataset.pickField === 'shared' && box.dataset.canShare !== 'true';
        }));
      }, remaining);
    }
    const busy = boxes.some(box => box.dataset.pending);
    boxes.forEach(box => {
      box.querySelectorAll('input[data-pick-field]').forEach(input => {
        input.disabled = busy || (input.dataset.pickField === 'shared' && box.dataset.canShare !== 'true');
      });
      if (box.dataset.bridge === data.token) return;
      box.dataset.bridge = data.token;
      box.querySelectorAll('input[data-pick-field]').forEach(input => {
        input.addEventListener('change', () => {
          if (boxes.some(item => item.dataset.pending)) return;
          box.dataset.pending = crypto.randomUUID();
          box.dataset.pendingSince = String(Date.now());
          clearTimeout(recoveryTimer);
          recoveryTimer = setTimeout(() => {
            if (controller.signal.aborted || !box.dataset.pending) return;
            box.querySelector('[data-pick-field="played"]').checked = box.dataset.savedPlayed === 'true';
            box.querySelector('[data-pick-field="shared"]').checked = box.dataset.savedShared === 'true';
            box.querySelector('[role="status"]').textContent = 'Save not confirmed · retry';
            delete box.dataset.pending;
            delete box.dataset.pendingSince;
            boxes.forEach(item => item.querySelectorAll('input[data-pick-field]').forEach(el => {
              el.disabled = el.dataset.pickField === 'shared' && item.dataset.canShare !== 'true';
            }));
          }, 12000);
          // Serialize writes so rapid clicks cannot replace an in-flight event.
          boxes.forEach(item => item.querySelectorAll('input').forEach(el => el.disabled = true));
          const note = box.querySelector('[role="status"]');
          if (note) note.textContent = 'Saving…';
          setTriggerValue('action', {pick_id: box.dataset.warlordPick,
            field: input.dataset.pickField, value: input.checked,
            revision: Number(box.dataset.revision), request_id: box.dataset.pending});
        }, {signal: controller.signal});
      });
    });
  }
  attach();
  return () => { clearTimeout(timer); clearTimeout(recoveryTimer); controller.abort(); };
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
    return store.change(pick, action.get('field'), action.get('value'), expected)


def save_card_mark(store, registry, states, replies, action, *, owner_access):
    """Save once, then return only control state; never expose credentials/payloads."""
    if not owner_access or not isinstance(action, dict) or action.get('pick_id') not in registry:
        return
    pick_id = action['pick_id']
    request_id = action.get('request_id')
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 64:
        return
    if replies.get(pick_id, {}).get('request_id') == request_id:
        return
    try:
        states[pick_id] = apply_card_action(store, registry, states, action, owner_access=owner_access)
        message = 'Saved ✓'
    except Exception as exc:
        # A stale tab must adopt the database state, rather than overwrite it.
        if owner_access:
            try:
                latest = store.private_rows()
                states[pick_id] = latest.get(pick_id, {})
            except Exception:
                pass
        message = ('The mark changed or is no longer available. Please try again.'
                   if isinstance(exc, ValueError) else 'Save not confirmed. Please try again.')
        if isinstance(exc, ValueError) and str(exc) == 'Mark Played before sharing this pick.':
            message = 'Mark Played before sharing this pick.'
    state = states.get(pick_id, {})
    played, shared = bool(state.get('played')), bool(state.get('shared'))
    replies[pick_id] = dict(request_id=request_id, played=played, shared=shared,
                           revision=int(state.get('revision', 0)),
                           can_share=played, message=message)


@st.fragment
def _save_bridge(store, registry, states, replies, *, owner_access, key):
    """Only this tiny fragment reruns on a checkbox event; cards stay mounted."""
    import uuid

    def on_action():
        component_state = st.session_state.get(key, {})
        save_card_mark(store, registry, states, replies, component_state.get('action'),
                       owner_access=owner_access)

    _bridge(data={'scope': key, 'token': uuid.uuid4().hex, 'replies': replies,
                  'has_cards': bool(registry)},
            key=key, on_action_change=on_action)


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
        share_disabled = ' disabled' if error or not played else ''
        note = 'Saved ✓' if played else 'Private mark'
        if error:
            note = 'Saving unavailable'
        can_share = played
        return (f'<div class="wn-pick-controls" data-warlord-pick="{pick["pick_id"]}" data-revision="{revision}"'
                f' data-pick-scope="{escape(key, quote=True)}" data-can-share="{str(can_share).lower()}">'
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
    _save_bridge(store, registry, states, {}, owner_access=owner_access, key=key)


def pp_pick_for_quote(quote, rows):
    """Bind a displayed PPP quote to exactly one trusted tracker identity."""
    candidates = [row for row in rows
                  if str(row.get('Player', '')).strip() == str(quote.get('Player', '')).strip()
                  and str(row.get('Team', '')).strip() == str(quote.get('Team', '')).strip()
                  and str(row.get('Game', '')).strip() == str(quote.get('Game', '')).strip()]
    card = dict(player=quote.get('Player'), team=quote.get('Team'), game=quote.get('Game'),
                market='PPP', line=quote.get('PPP line'), odds=quote.get('Over odds'),
                book=quote.get('Book'), move={})
    picks = []
    for row in candidates:
        try:
            picks.append(make_pick(card, row))
        except (ValueError, TypeError, KeyError):
            continue
    if not picks or len({pick['pick_id'] for pick in picks}) != 1:
        raise ValueError('Missing or ambiguous player/game identity')
    return picks[0]


def pp_controls_html(pick, state, error, key):
    """The same scoped, fragment-backed controls used on cards, inside a PP row."""
    played, shared = bool(state.get('played')), bool(state.get('shared'))
    revision = int(state.get('revision', 0))
    can_share = played
    disabled = ' disabled' if error else ''
    share_disabled = ' disabled' if error or not can_share else ''
    note = 'Saving unavailable' if error else 'Saved ✓' if played else 'Private mark'
    player = escape(pick['player'], quote=True)
    return (f'<div class="wn-pick-controls" data-warlord-pick="{pick["pick_id"]}" data-revision="{revision}"'
            f' data-pick-scope="{escape(key, quote=True)}" data-can-share="{str(can_share).lower()}">'
            f'<label><input type="checkbox" data-pick-field="played" aria-label="Played: {player} PPP over {pick["line"]}"'
            f'{" checked" if played else ""}{disabled}>✓ Played</label>'
            '<label title="Publish this pick and displayed price. Your personal stakes stay private.">'
            f'<input type="checkbox" data-pick-field="shared" aria-label="Share to Tail list: {player} PPP over {pick["line"]}"'
            f'{" checked" if shared else ""}{share_disabled}>Share to Tail list</label>'
            f'<span role="status">{note}</span></div>')


def render_markable_pp_table(quotes, frame, signals_by_team, *, render, owner_access, setting, key):
    """One bridge for the whole sortable table. Public viewers never load marks."""
    if not owner_access:
        st.html(render(quotes, signals_by_team), unsafe_allow_javascript=True)
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

    def controls(quote):
        try:
            pick = pp_pick_for_quote(quote, rows)
        except (ValueError, TypeError, KeyError):
            return '<div class="wn-pick-controls">Played unavailable · missing player/game identity</div>'
        registry[pick['pick_id']] = pick
        return pp_controls_html(pick, states.get(pick['pick_id'], {}), error, key)

    st.html(_CSS + render(quotes, signals_by_team, row_controls=controls), unsafe_allow_javascript=True)
    if error:
        st.warning(error)
        return
    _save_bridge(store, registry, states, {}, owner_access=owner_access, key=key)


def _return(odds, status):
    if status == 'Win':
        return odds / 100 if odds > 0 else 100 / abs(odds)
    return -1.0 if status == 'Loss' else 0.0


def share_timing(pick):
    from datetime import datetime
    try:
        shared = datetime.fromisoformat(pick['shared_at'].replace('Z', '+00:00'))
        start = datetime.fromisoformat(pick['start_utc'].replace('Z', '+00:00'))
        if shared.tzinfo is None or start.tzinfo is None:
            return 'Start timing unavailable'
        return 'Shared before scheduled start' if shared < start else 'Shared after scheduled start'
    except (KeyError, TypeError, ValueError):
        return 'Start timing unavailable'


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
    st.caption('Only picks explicitly shared by the owner appear here. Prices are saved reference quotes, not live odds. Shares after the scheduled start are labeled; no bets are placed by this page.' if public else
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
        table.append({'Player': p['player'], 'Game': matchup_label(p['game'], p.get('start_utc')), 'Date': p['date'],
                      'Pick': f'Over {p["line"]:g} {p["market"]}',
                      'Saved quote': f'{p["odds"]:+g} · {p["book"]}', 'Result': status,
                      **({'Shared timing': share_timing(p), 'Shared at (UTC)': p['shared_at']} if public else
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
