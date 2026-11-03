"""The Warlord verse masthead; purely visual, no model or clock logic."""
from html import escape


def render_header(*, state, timer, strike_label, status):
    phase = {'pre':'STRIKES IN', 'live':'SLATE LIVE', 'post':'NEXT STRIKE'}[state]
    return f'''<style>
    .wv-masthead{{position:relative;isolation:isolate;overflow:hidden;display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:24px;padding:30px 34px 25px;margin:8px 0 23px;border:1px solid #a994635e;border-radius:6px 24px 6px 24px;background:radial-gradient(ellipse at 73% 0%,#49317777,transparent 58%),linear-gradient(115deg,#0b1120,#182136 62%,#101626);color:#f6f1e7;box-shadow:0 12px 28px #10152530,inset 0 1px #e3c77e55}}
    .wv-masthead::before{{content:"";position:absolute;inset:7px;pointer-events:none;border:1px solid #e2c47826;border-radius:3px 18px 3px 18px;z-index:-1}}
    .wv-masthead::after{{content:"";position:absolute;left:34px;right:34px;bottom:0;height:2px;background:linear-gradient(90deg,transparent,#e5c674,#a481eb,#63e1cf,transparent)}}
    .wv-eyebrow{{font-size:10px;letter-spacing:3.8px;font-weight:800;color:#dcc084;text-transform:uppercase;margin:0 0 14px}}
    .wv-title{{font-family:Georgia,'Times New Roman',serif!important;font-size:clamp(27px,3.4vw,53px)!important;font-weight:700!important;letter-spacing:1.3px!important;line-height:1.02!important;color:#f9efd4!important;margin:0!important;padding:0!important;text-shadow:0 2px 14px #9b7edc35}}
    .wv-title span{{display:block;font-family:Arial,sans-serif;font-size:13px;letter-spacing:5px;color:#a7b9d6;margin-top:12px;font-weight:700}}
    .wv-motto{{font-size:13px;line-height:1.6;color:#bfc8d9;margin:18px 0 0;max-width:520px}}
    .wv-command{{display:flex;flex-direction:column;justify-content:center;text-align:right}}
    .wv-phase{{font-size:10px;letter-spacing:2.6px;color:#bda9e5;font-weight:800}}
    .wv-clock{{font-size:clamp(28px,3.1vw,43px);line-height:1.2;font-family:Arial,sans-serif;font-weight:800;letter-spacing:2px;font-variant-numeric:tabular-nums;color:#f7d58a;margin:4px 0 9px}}
    .wv-strike{{font-size:11px;color:#aebbd0}}
    .wv-status{{display:inline-block;border:1px solid #b7a4dd44;border-radius:4px;padding:5px 9px;margin-top:12px;font-size:9px;letter-spacing:1.5px;color:#cfbaf6;background:#a886de12}}
    .wv-live .wv-clock{{color:#7ce5c5}} .wv-live .wv-status{{color:#7ce5c5;border-color:#7ce5c544}}
    @media(max-width:640px){{.wv-masthead{{grid-template-columns:1fr;padding:25px 23px;gap:20px}}.wv-command{{text-align:left;border-top:1px solid #dcc08433;padding-top:18px}}.wv-title{{font-size:32px!important}}.wv-motto{{font-size:12px}}}}
    </style><header class="wv-masthead wv-{escape(state)}">
    <div><div class="wv-eyebrow">✦ THE WARLORD VERSE ✦</div>
    <h1 class="wv-title">THE WARLORD’S<span>NHL PROP TOOL</span></h1>
    <p class="wv-motto">Weapons forged by the gods.<br>Warlords prepare to wage war on the books.</p></div>
    <div class="wv-command"><div class="wv-phase">{phase}</div>
    <div class="wv-clock">{escape(timer)}</div><div class="wv-strike">{escape(strike_label)} · Central time</div>
    <div><span class="wv-status">{escape(status)}</span></div></div></header>'''
