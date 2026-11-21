"""Keep deep move histories inactive until their native disclosure is opened."""
from html import escape
import streamlit as st

_JS = r"""
export default function({data}) {
  function hydrate(detail) {
    if (!(detail instanceof HTMLDetailsElement) || !detail.open ||
        !detail.classList.contains('wn-deferred-records')) return;
    const scope = detail.closest('[data-warlord-inspect-scope]');
    if (!scope || scope.dataset.warlordInspectScope !== data.scope) return;
    const template = detail.querySelector(':scope > template.wn-record-template');
    if (!template) return;
    detail.appendChild(template.content);
    template.remove();
  }
  function onToggle(event) { hydrate(event.target); }
  // Native toggle stays entirely in the browser; no Streamlit trigger or fetch.
  document.addEventListener('toggle', onToggle, true);
  document.querySelectorAll('details.wn-deferred-records[open]').forEach(hydrate);
  return () => document.removeEventListener('toggle', onToggle, true);
}
"""
_bridge = st.components.v2.component('warlord_inspect_records', js=_JS)


def wrap(html, scope):
    return '<div data-warlord-inspect-scope="' + escape(scope, quote=True) + '">' + html + '</div>'


def mount(scope):
    _bridge(data={'scope': scope}, key=scope + '_inspect_records')
