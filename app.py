
# -------------------------
# Back-compat SVG helpers (used by player-card tags / older HUD snippets)
# -------------------------
if "_svg_get" not in globals():
    def _svg_get(fname: str) -> str:
        """Return raw SVG markup from assets/icons/<fname>."""
        try:
            return _load_svg_icon(fname)
        except Exception:
            return ""

if "_svg_inline" not in globals():
    def _svg_inline(svg: str, size: int = 20, title: str = "") -> str:
        """Wrap raw SVG markup in a span for inline rendering (safe, non-breaking)."""
        try:
            if not svg:
                return ""
            ttl = (title or "").replace('"', "&quot;")
            # keep sizing stable; global CSS clamps icon size to prevent 'giant icon' rerun bug
            return f'<span class="wl-ico wl-mono" title="{ttl}">{svg}</span>'
        except Exception:
            return ""

import os
import glob
import math
import re
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st

def render_odds_implied_reference(location="main", title="Odds → Implied% (break-even)"):
    """Simple reference block: American odds (+odds ladder) → implied break-even probability.

    location: "main" (st) or "sidebar" (st.sidebar)
    """
    try:
        import streamlit as st
    except Exception:
        return

    host = st if location == "main" else st.sidebar

    odds_list = [100,110,120,130,140,150,160,170,180,190,200,210,220,230]

    # Build a compact markdown table (no pandas)
    lines = []
    lines.append("| Odds | Implied% |")
    lines.append("|---:|---:|")
    for o in odds_list:
        try:
            imp = implied_prob_from_american(float(o)) * 100.0
        except Exception:
            imp = 100.0 / (float(o) + 100.0) * 100.0
        lines.append(f"| +{int(o)} | {imp:.2f}% |")

    with host.expander(title, expanded=False):
        host.caption(
            "Implied% shown is the break-even rate from the listed odds (not true two-sided no-vig). "
            "Use this to compare: Edge = Our Hit% − Implied%."
        )
        host.markdown("\n".join(lines))


# --- GLOBAL ICON CSS (always inject; prevents oversized SVGs on reruns) ---
st.markdown("""
<style>
/* All inline SVG icons injected via _svg_icon() */
.wl-ico { display:inline-flex; align-items:center; justify-content:center; line-height:0; overflow:hidden; }
.wl-ico svg { width:20px !important; height:20px !important; max-width:20px !important; max-height:20px !important; }


      /* -------------------------
               VENGEANCE BANNER
               ------------------------- */
            .vengeance-wrap{
              color: rgba(255,255,255,0.96);
              text-shadow: 0 1px 2px rgba(0,0,0,0.55);
              border-radius: 18px;
              padding: 14px 16px;
              border: 1px solid rgba(255,255,255,0.12);
              box-shadow: 0 8px 22px rgba(0,0,0,0.28);
              margin: 10px 0 14px 0;
            }
            .vengeance-pre{
              background: radial-gradient(1200px 120px at 10% 0%, rgba(255,255,255,0.14), rgba(0,0,0,0.0)),
                          linear-gradient(90deg, rgba(25,25,25,0.95), rgba(45,18,18,0.92));
            }
            .vengeance-live{
              background: radial-gradient(900px 140px at 15% 0%, rgba(255,255,255,0.16), rgba(0,0,0,0.0)),
                          linear-gradient(90deg, rgba(80,0,0,0.95), rgba(20,10,10,0.92));
              animation: vengeancePulse 1.6s ease-in-out infinite;
            }
            .vengeance-post{
              background: radial-gradient(1200px 120px at 10% 0%, rgba(255,255,255,0.10), rgba(0,0,0,0.0)),
                          linear-gradient(90deg, rgba(20,25,32,0.95), rgba(12,18,20,0.92));
            }
            @keyframes vengeancePulse{
              0%{ filter: brightness(1.00); transform: translateY(0px); }
              50%{ filter: brightness(1.08); transform: translateY(-1px); }
              100%{ filter: brightness(1.00); transform: translateY(0px); }
            }
            .vengeance-kicker{
              color: rgba(255,255,255,0.78);
              font-weight: 900;
              letter-spacing: 1.2px;
              font-size: 12px;
              opacity: 0.9;
              text-transform: uppercase;
            }
            .vengeance-head{
              color: rgba(255,255,255,0.98);
              font-weight: 950;
              letter-spacing: 0.6px;
              font-size: 26px;
              line-height: 1.05;
              margin-top: 2px;
            }
            .vengeance-sub{
              color: rgba(255,255,255,0.86);
              margin-top: 4px;
              font-size: 13px;
              opacity: 0.92;
            }
            .vengeance-timer{
              color: rgba(255,255,255,0.96);
              font-weight: 950;
              font-variant-numeric: tabular-nums;
              letter-spacing: 1px;
              font-size: 30px;
              text-align: right;
              line-height: 1.05;
            }
            .vengeance-pill{
              display: inline-block;
              padding: 3px 10px;
              border-radius: 999px;
              border: 1px solid rgba(255,255,255,0.18);
              font-size: 12px;
              font-weight: 800;
              opacity: 0.95;
            }
      
          
</style>
""", unsafe_allow_html=True)
# --- END GLOBAL ICON CSS ---

def _safe_float(v, default=None):
    """Safely convert value to float. Returns default (None if not set) on failure."""
    try:
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        x = float(v)
        if math.isnan(x):
            return default
        return x
    except Exception:
        return default





# -------------------------------------------------------------------
# Safety: ensure rich "Why it fires" renderer exists (prevents NameError
# if a partial merge / paste removed the function definition).
# -------------------------------------------------------------------
if "_render_why_it_fires_rich" not in globals():
    def _render_why_it_fires_rich(mkt: str, r, tags: str = "") -> None:
        """Fallback renderer: keeps the app running if rich renderer is missing."""
        mk = str(mkt or "").strip().upper()
        st.caption(f"{mk} — Why it fires")
        if tags:
            st.write(tags)
        # best-effort: print a few key fields
        try:
            for k in ("Player","Game","Pos","Conf_Goal","Conf_Points","Conf_SOG","Conf_Assists",
                      "Matrix_Goal","Matrix_Points","Matrix_SOG","Matrix_Assists",
                      "Goal_Line","Goal_Line","SOG_Line","Points_Line","Assists_Line",
                      "Avg5_SOG","Med10_SOG","ShotIntent","ShotIntent_Pct","opp_5v5_xGA60","Goalie_Weak"):
                if k in getattr(r, "keys", lambda: [])():
                    v = r.get(k, None)
                    if v not in (None, "", np.nan):
                        st.caption(f"{k}: {v}")
        except Exception:
            pass




# =========================
# Ledger helpers (append-only bet tracking)
# =========================
UNIT_VALUE_USD = 50.0   # 1u = $50 (user-defined)
MAX_STAKE_U = 3.0       # cap per play

# CSV headers (append-only)
BETSLIP_HEADERS = [
    'bet_id','date','datetime_placed','game','player','market','line','odds_taken','book','stake_u',
    'earned_green','ev_flag','lock_flag','conf','matrix','model_pct','imp_pct','ev_pct','tier','proof_count','why_tags',
    'opp','opp_goalie','notes'
]

BET_EVENTS_HEADERS = [
    'bet_id','event_type','event_datetime','event_period','event_game_minute','units_net','source','event_notes'
]


def _ledger_paths(output_dir: str) -> tuple[str, str, str]:
    ledger_dir = os.path.join(output_dir, "ledger")
    return ledger_dir, os.path.join(ledger_dir, "betslip.csv"), os.path.join(ledger_dir, "bet_events.csv")


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def american_to_decimal(odds: float) -> float:
    try:
        o = float(odds)
    except Exception:
        return 1.0
    if o == 0:
        return 1.0
    if o > 0:
        return 1.0 + (o / 100.0)
    return 1.0 + (100.0 / abs(o))


def implied_prob_from_american(odds: float) -> float:
    try:
        o = float(odds)
    except Exception:
        return 0.5
    if o == 0:
        return 0.5
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def calc_ev_pct_and_kelly(model_prob: float, odds: float) -> tuple[float, float, float, float]:
    # returns: (imp_prob, ev_pct, kelly_full, dec_odds)
    p = max(0.0001, min(0.9999, float(model_prob)))
    dec = american_to_decimal(float(odds))
    imp = implied_prob_from_american(float(odds))
    b = dec - 1.0
    q = 1.0 - p
    ev_per_dollar = (p * b) - q
    ev_pct = ev_per_dollar * 100.0
    kelly = max(0.0, (b * p - q) / b) if b > 0 else 0.0
    return imp, ev_pct, kelly, dec


def calc_ev_per_dollar(model_prob: float, odds_american: float) -> float | None:
    """Return expected profit (or loss) per $1 staked. Positive = +EV."""
    try:
        p = float(model_prob)
        if p > 1.0:  # accept percent inputs
            p = p / 100.0
        odds = float(odds_american)
        if p <= 0 or p >= 1:
            return None
        if odds == 0:
            return None
        if odds > 0:
            profit_if_win = odds / 100.0
        else:
            profit_if_win = 100.0 / abs(odds)
        ev_per_dollar = p * profit_if_win - (1.0 - p) * 1.0
        return ev_per_dollar
    except Exception:
        return None

def _append_csv_row(path: str, row: dict, headers: list[str]) -> None:
    _ensure_dir(os.path.dirname(path))
    file_exists = os.path.exists(path)
    # ensure all headers exist
    safe_row = {h: row.get(h, "") for h in headers}
    df1 = pd.DataFrame([safe_row], columns=headers)
    if not file_exists:
        df1.to_csv(path, index=False)
    else:
        df1.to_csv(path, mode='a', header=False, index=False)


def _slug(s: str) -> str:
    s = str(s or '').strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "", s)
    return s


def make_bet_id(date_str: str, player: str, market: str, line: float, odds_taken: float) -> str:
    d = str(date_str or '').replace('-', '')
    return f"{d}_{_slug(player)}_{_slug(market)}_{_slug(line)}_{_slug(int(odds_taken) if float(odds_taken).is_integer() else odds_taken)}"

def render_market_filter_bar(default_min_conf: int = 60, key_prefix: str = "m"):
    c1, c2, c3, c4, c5, c6 = st.columns([1.1,1.1,1.2,1.2,1.1,1.6])
    with c1:
        greens_only = st.toggle("🟢 Greens", value=False, key=f"{key_prefix}_greens")
    with c2:
        ev_only = st.toggle("💰 +EV", value=False, key=f"{key_prefix}_ev")
    with c3:
        locks_only = st.toggle("🔒 Locks", value=False, key=f"{key_prefix}_locks")
    with c4:
        plays_first = st.toggle("⭐ Plays first", value=True, key=f"{key_prefix}_playsfirst")
    with c5:
        hide_reds = st.toggle("Hide 🔴", value=True, key=f"{key_prefix}_hidered")
    with c6:
        min_conf = st.slider("Min Conf", 0, 100, int(default_min_conf), 1, key=f"{key_prefix}_minconf")
    return {
        "greens_only": greens_only,
        "ev_only": ev_only,
        "locks_only": locks_only,
        "plays_first": plays_first,
        "hide_reds": hide_reds,
        "min_conf": min_conf,
    }
def legend_signals():
    with st.expander("Legend (signals)", expanded=False):
        st.markdown(
            '''
<div class="wl-card">
  <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;">
    <span class="wl-pill wl-green">🟢 Earned Green</span>
    <span class="wl-pill wl-gold">💰 +EV</span>
    <span class="wl-pill wl-purple">🔒 LOCK</span>
    <span class="wl-pill wl-blue">EV_Signal = best EV% shown</span>
  </div>
  <div style="opacity:0.85;font-size:13px;margin-top:8px;">
    Use <b>🟢</b> when the play is earned by proofs, <b>💰</b> when price is +EV, and <b>🔒</b> when both align.
  </div>
</div>
            ''',
            unsafe_allow_html=True,
        )


def _why_sections_header(mkt: str = ""):
    """Presentation-only ledger for MAIN / SUPPORT / TONIGHT (no gates/logic)."""
    return  # Legend hidden (Combat HUD replaces it)
    mk = str(mkt or "").strip().upper()
    # Market color matching (same palette used by pills)
    if "ASSIST" in mk:
        bg = "rgba(168,85,247,0.14)"
        border = "#a855f7"
    elif "SOG" in mk or "SHOT" in mk:
        bg = "rgba(34,197,94,0.14)"
        border = "#22c55e"
    elif "GOAL" in mk or "ATG" in mk:
        bg = "rgba(239,68,68,0.14)"
        border = "#ef4444"
    else:  # POINTS default
        bg = "rgba(59,130,246,0.14)"
        border = "#3b82f6"

    st.markdown(
        f"""
        <div style="padding:12px 14px;border-radius:14px;border:1px solid rgba(0,0,0,0.10);background:{bg};border-left:6px solid {border};margin-bottom:10px;">
          <div style="font-size:18px;font-weight:900;color:#000;margin-bottom:6px;">Why it fires — legend</div>
          <div style="font-size:14px;font-weight:800;color:#000;line-height:1.45;">
            <b>MAIN</b>: primary trigger(s) for this market<br/>
            <b>SUPPORT</b>: independent proofs backing the MAIN trigger (X / 5)<br/>
            <b>TONIGHT</b>: matchup context (goalie / defense / PP env / pace)
          </div>
          <div style="margin-top:10px;font-size:13px;font-weight:800;color:#000;">SUPPORT proofs (max 5)</div>
          <ul style="margin:6px 0 0 18px;padding:0;font-size:13px;line-height:1.35;color:#000;">
            <li><b>Regression pressure</b> (HOT / DUE / OVERDUE, gap)</li>
            <li><b>PP role / distributor</b> (PP1 share, involvement)</li>
            <li><b>Talent tier</b> (STAR / ELITE)</li>
            <li><b>Volume / involvement</b> (TOI, touches/attempts/usage)</li>
            <li><b>Linemate / stack quality</b> (line context helps conversion)</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_jackpot_meter_ui(reg_gap: float, reg_threshold: float, tier_clean: str) -> None:
    """🎰 Jackpot Meter (UI only).
    Jackpot reflects how strongly current conditions favor regression resolution.
    Talent (ELITE) increases confidence only when regression is already valid.

    NOTE: presentation-only — no gates, no EV logic.
    """
    try:
        rg = float(reg_gap)
    except Exception:
        rg = 0.0
    try:
        thr = float(reg_threshold)
    except Exception:
        thr = 0.0

    reg_valid = (thr > 0) and (rg >= thr)
    tier_u = str(tier_clean or "").strip().upper()
    elite_active = reg_valid and (tier_u == "ELITE")

    # normalize 0-100 for display
    pct = 0.0
    if thr > 0:
        pct = max(0.0, min(100.0, (rg / thr) * 100.0))

    bar_color = "#f5c542" if reg_valid else "#9ca3af"
    glow = "box-shadow: 0 0 14px rgba(245,197,66,0.85);" if elite_active else ""

    st.markdown(
        f"""
        <div class="wl-card" style="margin-top:8px;">
          <div style="display:flex;align-items:center;justify-content:space-between;">
            <div style="font-size:18px;font-weight:900;">🎰 Jackpot Meter</div>
            <div style="font-size:13px;font-weight:800;opacity:0.85;">
              Reg_Gap {rg:.2f} / {thr:.2f}
            </div>
          </div>

          <div style="margin-top:8px;background:#111;border-radius:10px;overflow:hidden;">
            <div style="width:{pct:.0f}%;height:14px;background:{bar_color};{glow}transition:width 0.35s ease;"></div>
          </div>

          <div style="margin-top:8px;font-size:13px;line-height:1.35;">
            <b>Jackpot reflects how strongly current conditions favor regression resolution.</b><br/>
            <span style="opacity:0.85;">
              Talent (ELITE) increases confidence only when regression is already valid.
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _trend_badge_score(mkt: str, d10, d20, d40):
    """
    Presentation-only: derive a compact Window Signal from L10/L20/L40 diffs vs line.

    Returns (badge, score_int_0_100, trend_word).
      - badge: STABLE / DUE / HEATER / VOLATILE / DECLINE / NEUTRAL
      - trend_word: HEATING / COOLING / FLAT

    NOTE (accuracy-first):
      - Treat NaN/blank as missing (do NOT silently coerce to 0).
      - If any of the three diffs are missing, return ("", 0, "") so the UI doesn't lie.
    """
    mk = str(mkt or "").upper().strip()

    # market epsilon: "material" diff threshold (per game)
    eps_map = {"POINTS": 0.15, "ASSISTS": 0.12, "SOG": 0.30, "SHOTS": 0.30}
    eps = float(eps_map.get(mk, 0.20))

    def _num_or_none(v):
        try:
            if v is None:
                return None
            if isinstance(v, str) and not v.strip():
                return None
            x = float(v)
            if math.isnan(x):
                return None
            return x
        except Exception:
            return None

    d10 = _num_or_none(d10)
    d20 = _num_or_none(d20)
    d40 = _num_or_none(d40)
    if d10 is None or d20 is None or d40 is None:
        return "", 0, ""

    def _sgn(x: float) -> int:
        if x > eps:
            return 1
        if x < -eps:
            return -1
        return 0

    s10, s20, s40 = _sgn(d10), _sgn(d20), _sgn(d40)

    # dispersion / stability: how much the window diffs move around
    disp = max(abs(d10 - d20), abs(d20 - d40))

    # score: 100 when windows align tightly; penalize sign flips and dispersion
    # scale dispersion by eps so the score feels comparable per market
    score = 100.0 - 25.0 * (disp / eps) if eps > 0 else 50.0
    if (s10 != s20) or (s20 != s40):
        score -= 15.0
    score = max(0.0, min(100.0, score))
    score_i = int(round(score))

    # trend vs baseline (L10 compared to L40)
    if (d10 - d40) > eps:
        trend = "HEATING"
    elif (d10 - d40) < -eps:
        trend = "COOLING"
    else:
        trend = "FLAT"

    # badge classification (tight + readable)
    if (d10 < -eps) and (d20 < -eps) and (d40 >= 0.0):
        badge = "DUE"
    elif (d10 > eps) and (d20 > eps) and (d40 < 0.0):
        badge = "HEATER"
    elif (d10 >= 0.0) and (d20 >= 0.0) and (d40 >= 0.0) and (disp <= 2.0 * eps):
        badge = "STABLE"
    elif (d40 < -eps) and (d10 < -eps):
        badge = "DECLINE"
    elif (s10 != s20) or (s20 != s40) or (disp > 3.0 * eps):
        badge = "VOLATILE"
    else:
        badge = "NEUTRAL"

    return badge, score_i, trend





# =========================
# SVG Icon system (assets/icons/*.svg)
# =========================
@st.cache_data(show_spinner=False)
def _load_svg_icon(fname: str) -> str:
    """Load an SVG and return raw markup for inline embedding.

    Search order (first hit wins):
      1) <app_dir>/assets/icons/<fname>
      2) <cwd>/assets/icons/<fname>
      3) <app_dir>/<fname>
      4) <cwd>/<fname>

    This makes local dev resilient when icons aren't placed in assets/ yet.
    """
    try:
        app_dir = Path(__file__).parent
    except Exception:
        app_dir = Path.cwd()

    candidates = [
        app_dir / "assets" / "icons" / fname,
        Path.cwd() / "assets" / "icons" / fname,
        app_dir / fname,
        Path.cwd() / fname,
    ]
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        return ""

    svg = p.read_text(encoding="utf-8", errors="ignore")

    # Strip XML/doctype noise for safer embedding
    svg = re.sub(r"<\?xml[^>]*>", "", svg, flags=re.I).strip()
    svg = re.sub(r"<!DOCTYPE[^>]*>", "", svg, flags=re.I).strip()

    # Strip explicit width/height on root <svg> so CSS always controls sizing
    svg = re.sub(r'(<svg\b[^>]*?)\swidth="[^"]*"', r'\1', svg, flags=re.I)
    svg = re.sub(r'(<svg\b[^>]*?)\sheight="[^"]*"', r'\1', svg, flags=re.I)

    # Remove width/height inside inline style on the <svg ...> tag
    svg = re.sub(r'(<svg\b[^>]*?\sstyle="[^"]*?)\bwidth\s*:\s*[^;"]+;?', r'\1', svg, flags=re.I)
    svg = re.sub(r'(<svg\b[^>]*?\sstyle="[^"]*?)\bheight\s*:\s*[^;"]+;?', r'\1', svg, flags=re.I)

    return svg.strip()


    return svg

def _svg_icon(fname: str, title: str = "", market_cls: str = "wl-goals") -> str:
    """Return inline SVG wrapped in a span. market_cls controls color via CSS."""
    svg = _load_svg_icon(fname)
    if not svg:
        return ""
    ttl = (title or "").replace('"', "&quot;")
    return f'<span class="wl-ico wl-mono {market_cls}" title="{ttl}">{svg}</span>'

def render_valhalla_gate(mkt: str) -> None:
    """Presentation-only Valhalla Gate card. No logic. No gating."""
    mk = str(mkt or "").strip().upper()
    role = _role_for_market(mk)

    # Baseline text per market
    if mk == "ASSISTS":
        baseline = "🟢 Matrix Green • Line 0.5 • Conf ≥ 80 • EV ignored"
    elif mk == "GOALS":
        baseline = "🟢 Green • 0.5 • OppSOG_L10 ≥ 29 + xGA ≥ 2.49 • EV ignored"
    elif mk == "POINTS":
        baseline = "🟢 Matrix Green • Line 0.5 • Conf ≥ 70 • EV ignored"
    elif mk in ("SOG", "SHOTS"):
        baseline = "🟢 Matrix Green • Line ≤ 2.5 • Conf ≥ 75 • EV ignored"
    else:
        baseline = "🟢 Matrix Green • Market baseline rules apply"

    market_cls = role.get("cls", "wl-neutral")
    icon_html = _svg_icon("valhalla.svg", "Valhalla Gate", market_cls)

    st.markdown(
        f"""
        <div class="wl-gate-card">
            <div class="wl-gate-header">
                <div class="wl-gate-icon">{icon_html}</div>
                <div class="wl-gate-title">
                    THE WARLORD’S GATE TO VALHALLA — {mk}
                </div>
            </div>
            <div class="wl-gate-baseline">
                <b>Entry requires:</b> {baseline}
            </div>
            <div class="wl-gate-note">
                Passing the Gate only allows entry. Moves trigger inside the board.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Role system (Carry / Support / Tank / Jungle)
# =========================
_ROLE_INFO = {
    "GOALS":   {"role": "Carry",   "svg": "role_carry.svg",   "cls": "wl-goals",   "emoji": "⚔️"},
    "ASSISTS": {"role": "Support", "svg": "role_support.svg", "cls": "wl-assists", "emoji": "🪄"},
    "POINTS":  {"role": "Tank",    "svg": "role_tank.svg",    "cls": "wl-points",  "emoji": "🛡️"},
    "SOG":     {"role": "Jungle",  "svg": "role_jungle.svg",  "cls": "wl-sog",     "emoji": "🌿"},
}

def _role_for_market(mkt: str) -> dict:
    key = str(mkt or "").strip().upper()
    if key in {"GOAL", "GOALS (0.5)", "GOALS1P", "GOALS_1P", "GOALS 1+", "GOALS (1+)"}:
        key = "GOALS"
    return _ROLE_INFO.get(key, {"role": "Role", "svg": "", "cls": "wl-points", "emoji": "⭐"})

def _role_icon_html(mkt: str) -> str:
    info = _role_for_market(mkt)
    svg = info.get("svg") or ""
    cls = info.get("cls") or "wl-points"
    ico = _svg_icon(svg, info.get("role",""), cls) if svg else ""
    if not ico:
        ico = f'<span style="margin-right:6px;">{info.get("emoji","⭐")}</span>'
    return ico

def _page_title_html(page_name: str, mkt: str) -> str:
    info = _role_for_market(mkt)
    ico = _role_icon_html(mkt)
    role = info.get("role","")
    return (
        f'<h2 style="margin:0.2rem 0 0.6rem 0;">'
        f'{page_name} <span style="font-weight:600;opacity:0.85;">({ico}{role})</span>'
        f'</h2>'
    )

def _wl_why_line(icon_svg: str, text: str) -> None:
    """Render one 'Why it fires' line with optional SVG icon."""
    if icon_svg:
        st.markdown(f"{icon_svg}<span>{text}</span>", unsafe_allow_html=True)
    else:
        st.markdown(text)



def _wl_market_color(mk: str) -> str:
    mk = str(mk or "").strip().upper()
    # Keep these in sync with your market colors
    if mk == "GOALS":
        return "#ef4444"  # red
    if mk == "ASSISTS":
        return "#a855f7"  # purple
    if mk == "POINTS":
        return "#2563eb"  # blue
    if mk in ("SOG", "SHOTS", "SHOTS ON GOAL"):
        return "#22c55e"  # green
    return "#111827"      # slate

def _wl_dps_bar(pct: float, mk: str, *, height_px: int = 8) -> None:
    """Tiny progress bar under a WHY line to visually represent DPS (% win)."""
    try:
        p = float(pct)
    except Exception:
        return
    if math.isnan(p):
        return
    p = max(0.0, min(100.0, p))
    color = _wl_market_color(mk)

    st.markdown(
        f"""
        <div style='margin-left:26px;margin-top:2px;margin-bottom:10px;'>
          <div style='height:{height_px}px;max-width:280px;width:70%;
                      background:rgba(17,24,39,0.10);
                      border-radius:999px;overflow:hidden;'>
            <div style='width:{p:.1f}%;height:{height_px}px;
                        background:{color};
                        border-radius:999px;'></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _adj_win(win: float, n: int, k: int = 20) -> float:
    """Shrink DPS win% toward 50 as n gets smaller (presentation only)."""
    try:
        w = float(win)
        nn = int(n) if n is not None else 0
        if nn <= 0:
            return 50.0
        return 50.0 + (w - 50.0) * (nn / (nn + float(k)))
    except Exception:
        return 50.0

def _render_rank_line(best_title: str, win: float, n: int, mk: str) -> None:
    """Small rank label shown on cards/HUDs. Presentation only."""
    if not best_title:
        return
    aw = _adj_win(win, n, k=20)
    try:
        w = float(win)
    except Exception:
        w = win
    st.caption(f"🏆 Best proc: **{best_title}** • AdjWin **{aw:.1f}** • Win **{w:.1f}%** (n={int(n)})")


# =========================
# Board-style DPS filters (shared by market pages) — presentation only
# =========================
def add_best_proc_cols(df: pd.DataFrame, mk: str) -> pd.DataFrame:
    """Add DPS_* columns (Title/Win/N/Adj) using the existing probe functions.
    Presentation-only: does not change any eligibility logic.
    """
    if df is None or len(df) == 0:
        return df
    mk_u = str(mk or "").strip().upper()
    out = df.copy()

    # Compute best proc per row (safe)
    def _best_row(row):
        try:
            b = _probe_best_proc(mk_u, row.to_dict())
            if not b:
                return ("", 0.0, 0, 0.0)
            return (str(b.get("title","")), float(b.get("win",0.0)), int(b.get("n",0)), float(b.get("adj",0.0)))
        except Exception:
            return ("", 0.0, 0, 0.0)

    vals = out.apply(_best_row, axis=1, result_type="expand")
    vals.columns = ["DPS_Title", "DPS_Win", "DPS_N", "DPS_Adj"]
    out = pd.concat([out.reset_index(drop=True), vals.reset_index(drop=True)], axis=1)
    return out

def _odds_value_for_row(row: dict, mk: str) -> float | None:
    mk_u = str(mk or "").strip().upper()

    # Market-specific odds columns (best-effort). We prefer explicit *Over columns when present.
    cand: list[str] = ["Odds", "Best_Odds", "Odds_Taken"]

    if mk_u == "POINTS":
        cand = [
            "Points_Odds_Over",
            "Odds_Points", "Odds_PTS",
            "BDL_Points_Odds", "BDL_Points_Odds_1", "BDL_Points_Odds_2", "BDL_Points_Odds_3", "BDL_Points_Odds_4",
        ] + cand
    elif mk_u == "ASSISTS":
        cand = [
            "Assists_Odds_Over",
            "Odds_Assists", "Odds_AST",
            "BDL_Assists_Odds", "BDL_Assists_Odds_1", "BDL_Assists_Odds_2", "BDL_Assists_Odds_3", "BDL_Assists_Odds_4",
        ] + cand
    elif mk_u == "SOG":
        cand = [
            "SOG_Odds_Over",
            "Odds_SOG", "Odds_Sh", "Odds_Shots",
            "BDL_SOG_Odds", "BDL_SOG_Odds_1", "BDL_SOG_Odds_2", "BDL_SOG_Odds_3", "BDL_SOG_Odds_4",
        ] + cand
    elif mk_u in ("GOALS", "GOAL"):
        cand = [
            "Goal_Odds_Over",
            "Odds_Goals", "Odds_Goal",
            "BDL_Goal_Odds", "BDL_Goal_Odds_1", "BDL_Goal_Odds_2", "BDL_Goal_Odds_3", "BDL_Goal_Odds_4",
        ] + cand
    elif mk_u == "ATG":
        cand = [
            "ATG_Odds_Over",
            "Odds_ATG",
            "BDL_ATG_Odds", "BDL_ATG_Odds_1", "BDL_ATG_Odds_2", "BDL_ATG_Odds_3", "BDL_ATG_Odds_4",
        ] + cand

    for k in cand:
        if k in row:
            v = _safe_float(row.get(k, None), None)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                return float(v)

    return None


def _line_value_for_row(row: dict, mk: str) -> float | None:
    mk_u = str(mk or "").strip().upper()
    cand = []
    if mk_u == "POINTS":
        cand = ["Points_Line","Line_Points"]
    elif mk_u == "ASSISTS":
        cand = ["Assists_Line","Line_Assists"]
    elif mk_u == "SOG":
        cand = ["SOG_Line","Line_SOG","Shots_Line"]
    elif mk_u in ("GOALS","GOAL","ATG"):
        cand = ["Goal_Line","Goals_Line","Line_Goals"]
    for k in cand:
        if k in row:
            v = _safe_float(row.get(k, None), None)
            if v is not None:
                return float(v)
    return None

def apply_dps_filters_ui(df: pd.DataFrame, mk: str, key_prefix: str = "m") -> pd.DataFrame:
    """Board-style filter bar for a single market page.
    Filters: Line, Move/Tier, Min DPS win, Min DPS n, Max favorite odds, Search.
    Sorting: DPS_Adj desc, DPS_N desc (ranking only).
    """
    if df is None or len(df) == 0:
        return df

    mk_u = str(mk or "").strip().upper()
    out = df.copy()

    # Ensure DPS columns exist
    if "DPS_Adj" not in out.columns:
        out = add_best_proc_cols(out, mk_u)

    # Build line + move option lists
    try:
        line_vals = sorted({lv for lv in (out.apply(lambda r: _line_value_for_row(r.to_dict(), mk_u), axis=1).tolist()) if lv is not None})
    except Exception:
        line_vals = []
    move_vals = sorted({str(x) for x in out.get("DPS_Title","").fillna("").astype(str).tolist() if str(x).strip()})

    st.sidebar.subheader(f"{mk_u} — Filters")
    line_sel = st.sidebar.multiselect("Line", line_vals, default=line_vals, key=f"{key_prefix}_line") if line_vals else []
    move_sel = st.sidebar.multiselect("Move / Tier", move_vals, default=move_vals, key=f"{key_prefix}_move") if move_vals else []
    min_win = float(st.sidebar.slider("Min DPS win%", 0.0, 100.0, 50.0, 0.5, key=f"{key_prefix}_minwin"))
    min_n = int(st.sidebar.number_input("Min DPS n", min_value=0, max_value=500, value=20, step=1, key=f"{key_prefix}_minn"))
    max_fav_odds = int(st.sidebar.number_input("Max favorite odds (e.g. -200)", min_value=-1000, max_value=300, value=-200, step=5, key=f"{key_prefix}_maxfav"))
    q = st.sidebar.text_input("Search", value="", key=f"{key_prefix}_q").strip().lower()

    # Compute helper columns for filtering
    out["_Line"] = out.apply(lambda r: _line_value_for_row(r.to_dict(), mk_u), axis=1)
    out["_Odds"] = out.apply(lambda r: _odds_value_for_row(r.to_dict(), mk_u), axis=1)

    if q and "Player" in out.columns:
        out = out[out["Player"].astype(str).str.lower().str.contains(re.escape(q), na=False)]
    if line_sel:
        out = out[out["_Line"].isin(line_sel)]
    if move_sel:
        out = out[out["DPS_Title"].astype(str).isin(move_sel)]

    # DPS-based filters (rows with no proc naturally drop if min_win/min_n > 0)
    out = out[(pd.to_numeric(out.get("DPS_Win", 0), errors="coerce").fillna(0) >= min_win)]
    out = out[(pd.to_numeric(out.get("DPS_N", 0), errors="coerce").fillna(0).astype(int) >= min_n)]

    # Odds filter (favorites only): keep if odds >= max_fav_odds (e.g., -140 passes when max_fav=-150; -200 fails)
    def _odds_ok(v):
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return True  # missing odds => allow
            v = float(v)
            if v < 0:
                return v >= float(max_fav_odds)
            return True
        except Exception:
            return True
    out = out[out["_Odds"].apply(_odds_ok)]

    # Sort by DPS ranking (presentation only)
    out["_dps_adj"] = pd.to_numeric(out.get("DPS_Adj", 0), errors="coerce").fillna(0.0)
    out["_dps_n"] = pd.to_numeric(out.get("DPS_N", 0), errors="coerce").fillna(0).astype(int)
    out = out.sort_values(["_dps_adj", "_dps_n"], ascending=[False, False]).drop(columns=["_dps_adj","_dps_n"], errors="ignore")

    return out


# =========================
# DPS ranking probes (presentation only)
# =========================
def _probe_points_best(r: dict) -> dict | None:
    """Return best active POINTS proc: {title, win, n, adj}. Presentation only."""
    line = _safe_float(r.get("Points_Line"))
    line = 0.0 if line is None else float(line)
    conf_p = _safe_float(r.get("Conf_Points")) or 0.0
    conf_a = _safe_float(r.get("Conf_Assists")) or 0.0
    ppp = _safe_float(r.get("PPP10_total")) or 0.0
    pp_ixg = _safe_float(r.get("PP_iXG60")) or 0.0
    pp_ixa = _safe_float(r.get("PP_iXA60")) or 0.0
    assists_mu = _safe_float(r.get("Assists_mu")) or 0.0
    points_mu = _safe_float(r.get("Points_mu")) or 0.0
    drought_p = _safe_float(r.get("Drought_P"))
    drought_p = 0.0 if drought_p is None else float(drought_p)
    opp_gaa = _safe_float(r.get("Opp_GAA"))
    team_pp_xgf = _safe_float(r.get("Team_PP_xGF60")) or 0.0
    opp_defweak = _safe_float(r.get("Opp_DefWeak")) or 0.0
    opp_xga = _safe_float(r.get("opp_5v5_xGA60", r.get("Opp_5v5_xGA60")))
    team_gf_l5 = _safe_float(r.get("Team_GF_Avg_L5", r.get("Team_GF_L5")))
    team_gf_l5 = 0.0 if team_gf_l5 is None else float(team_gf_l5)
    opp_sog_l10 = _safe_float(r.get("Opp_SOG_Against_L10", r.get("Opp_SA_Avg_L10", r.get("OppSOG_L10"))))
    opp_sog_l10 = 0.0 if opp_sog_l10 is None else float(opp_sog_l10)

    procs = []
    is_fortress = (line <= 0.75)  # 0.5 build
    if is_fortress:
        procs += [
            ("Hammer Fists", 69.1, 55, (conf_p >= 70)),
            ("Echo Stomp I", 75.8, 33, (conf_p >= 70 and ppp >= 3)),
            ("Echo Stomp II", 75.9, 29, (conf_p >= 70 and points_mu >= 1.5)),
            ("Gaia’s Blessing", 70.85, 223, (conf_p >= 75 and assists_mu >= 0.7)),
            ("Enraged Engine", 92.3, 13, (conf_p >= 78 and points_mu >= 1.5)),
            ("Enraged Fury", 90.0, 10, (conf_p >= 78 and assists_mu >= 1.0)),
            ("Blood Stomp", 90.9, 11, (conf_p >= 78 and (opp_gaa is not None) and (2.5 <= float(opp_gaa) <= 3.0))),
            ("Hammer Fists II", 85.0, 20, (conf_p >= 70 and assists_mu >= 1.1)),
            ("Gaia’s Blessing+ (Press)", 73.60, 178, (conf_p >= 77 and assists_mu >= 0.7)),
            ("Gaia’s Blessing++ (Smash)", 78.31, 83, (conf_p >= 82 and assists_mu >= 0.7)),
        ]
        gaia_core = (conf_p >= 75 and assists_mu >= 0.7)
        favor_on = gaia_core and (team_gf_l5 >= 3.5)
        wrath_on = gaia_core and (team_gf_l5 >= 3.7)
        ascension_on = gaia_core and (team_gf_l5 >= 3.9)
        floodgate_on = wrath_on and (opp_sog_l10 >= 27.5)
        if floodgate_on:
            procs.append(("Gaia’s Floodgate", 81.0, 58, True))
        elif ascension_on:
            procs.append(("Gaia’s Ascension", 80.0, 75, True))
        elif wrath_on:
            procs.append(("Gaia’s Wrath", 76.7, 103, True))
        elif favor_on:
            procs.append(("Gaia’s Favor", 73.0, 152, True))
        # Label-only still eligible as a proc for ordering (keeps beta consistent)
        procs.append(("Bleed ENV (Label Only)", 76.9, 26, (conf_p >= 70 and pp_ixg >= 1.5)))
    else:
        # 1.5 kit (cond list copied from HUD)
        procs += [
            ("Backbone", 54.3, 116, (conf_a >= 89 and points_mu >= 1.7)),
            ("Blade Impale (Power Tier)", 60.8, 51, (conf_a >= 89 and points_mu >= 2.2)),
            ("Blade Slash (Monster)", 77.8, 18, (conf_a >= 89 and points_mu >= 2.2 and (opp_xga is not None) and float(opp_xga) >= 2.6)),
            ("Delayed Hammer Smash", 67.6, 34, (conf_a >= 89 and drought_p >= 1 and points_mu >= 1.7)),
            ("Enchanted Hammer (Legacy)", 61.1, 18, (conf_p >= 80 and pp_ixg >= 1.7)),
            ("Blade Impale (Legacy PP)", 49.2, 61, (conf_p >= 80 and pp_ixa >= 4.0)),
            ("Blade Slash (Legacy PP)", 48.1, 81, (conf_p >= 80 and team_pp_xgf >= 7.0)),
            ("Blood Exposure (Legacy)", 54.5, 44, (conf_p >= 80 and team_pp_xgf >= 7.0 and opp_defweak >= 60)),
            ("Blood Exposure II (Legacy)", 54.7, 64, (conf_p >= 80 and opp_defweak >= 60)),
            ("Polarizing Smash (Legacy)", 54.5, 33, (conf_p >= 80 and team_pp_xgf >= 7.0 and opp_defweak >= 70)),
            ("Eternal Smash (Legacy)", 53.2, 47, (conf_p >= 80 and opp_defweak >= 70)),
        ]

    best = None
    for title, win, n, cond in procs:
        if not cond:
            continue
        # Board eligibility guard: Env Mix requires Goal_Odds_Over >= +150 to qualify as best-proc/board pick.
        if title == "Berserker Aggression (Env Mix)" and not envmix_odds_ok:
            continue
        adj = _adj_win(win, n, k=20)
        if (best is None) or (adj > best["adj"]) or (abs(adj - best["adj"]) < 1e-9 and n > best["n"]):
            best = {"title": title, "win": float(win), "n": int(n), "adj": float(adj)}
    return best

def _probe_assists_best(r: dict) -> dict | None:
    import math
    line = _safe_float(r.get("Assists_Line"))
    if line is None:
        line = _safe_float(r.get("Line_Assists"))
    mat = str(r.get("Matrix_Assists", r.get("Matrix_A", "")) or "").strip().upper()
    conf = _safe_float(r.get("Conf_Assists", r.get("Conf_A", 0))) or 0.0
    stance_ok = (mat in {"GREEN", "🟢"}) and ((line == 0.5) or (line is None)) and (conf >= 80)

    ixa_pct    = _safe_float(r.get("iXA%"), default=float("nan"))
    pp_ix      = _safe_float(r.get("PP_iXA60", r.get("PP_iXA_60")), default=float("nan"))
    team_gf_l5 = _safe_float(r.get("Team_GF_L5"), default=float("nan"))
    ppp10      = _safe_float(r.get("PPP10_total"), default=float("nan"))
    pp_toi_pct = _safe_float(r.get("PP_TOI_Pct", r.get("PP_TOI%")), default=float("nan"))
    assists_mu = _safe_float(r.get("Assists_mu"), default=float("nan"))
    goalie_weak = _safe_float(r.get("Goalie_Weak"), default=float("nan"))

    staff_on = stance_ok
    if not staff_on:
        return None

    procs = []
    procs.append(("Staff (Base Shell)", 51.2, 733, True))
    procs.append(("Odin’s Staff", 54.3, 392, (not math.isnan(ixa_pct) and ixa_pct >= 95.0)))

    # Arcane ladder (PPP10)
    if not math.isnan(ppp10):
        if ppp10 >= 6:
            procs.append(("Arcane Channel VI", 60.6, 99, True))
        elif ppp10 >= 5:
            procs.append(("Arcane Channel V", 61.2, 201, True))
        elif ppp10 >= 4:
            procs.append(("Arcane Channel IV", 58.0, 319, True))
        elif ppp10 >= 3:
            procs.append(("Arcane Channel III", 56.3, 439, True))

    # Runic ladder (PP_iXA60)
    if not math.isnan(pp_ix):
        if pp_ix >= 4.0:
            procs.append(("Rune Orchestration (PP_iXA60≥4.0)", 63.3, 210, True))
        elif pp_ix >= 3.5:
            procs.append(("Rune Orchestration (PP_iXA60≥3.5)", 59.6, 317, True))

    # Silent Assassin ladder (mu)
    if not math.isnan(assists_mu):
        if assists_mu >= 1.62:
            procs.append(("Silent Assassin III", 69.4, 49, True))
        elif assists_mu >= 1.30:
            procs.append(("Silent Assassin II", 64.0, 178, True))
        elif assists_mu >= 1.15:
            procs.append(("Silent Assassin I", 61.1, 275, True))

    # ENV
    procs.append(("Odin’s Blessing (Goalie_Weak≥90)", 77.5, 40, (not math.isnan(goalie_weak) and goalie_weak >= 90)))

    # Magic (iXA%>=99)
    procs.append(("Magic (iXA%≥99)", 63.8, 130, (not math.isnan(ixa_pct) and ixa_pct >= 99.0)))

    # Arcane Transcendence (new core)
    arcane_transcendence_on = staff_on and (conf >= 88) and (not math.isnan(ppp10) and ppp10 >= 3) and (not math.isnan(pp_toi_pct) and pp_toi_pct >= 17)
    procs.append(("Arcane Transcendence", 77.8, 45, arcane_transcendence_on))

    # Arcane Supernova (heater upgrade; GF_Avg_L5 ≥ 3.9 ≈ Team_GF_L5 ≥ 20)
    arcane_supernova_on = arcane_transcendence_on and (not math.isnan(team_gf_l5) and team_gf_l5 >= 20)
    procs.append(("Arcane Supernova", 84.0, 25, arcane_supernova_on))

    # Supernova (convergence)
    supernova_on = staff_on and (conf >= 80) and (not math.isnan(ixa_pct) and ixa_pct >= 95.0) and (not math.isnan(pp_ix) and pp_ix >= 3.7) and (not math.isnan(team_gf_l5) and team_gf_l5 >= 20)
    procs.append(("Supernova Overdrive", 75.0, 64, supernova_on))

    # Stars aligned tiers
    if staff_on and (not math.isnan(ixa_pct)):
        if (conf >= 88) and (ixa_pct >= 96):
            procs.append(("Stars Aligned (Tier A)", 65.6, 163, True))
        if (conf >= 90) and (ixa_pct >= 95):
            procs.append(("Stars Aligned (Tier B)", 65.3, 118, True))

    best = None
    for title, win, n, cond in procs:
        if not cond:
            continue
        adj = _adj_win(win, n, k=20)
        if (best is None) or (adj > best["adj"]) or (abs(adj - best["adj"]) < 1e-9 and n > best["n"]):
            best = {"title": title, "win": float(win), "n": int(n), "adj": float(adj)}
    return best

def _probe_goals_best(r: dict) -> dict | None:
    # mirror the GOALS HUD key procs (exclude BASE from ranking)
    line = _safe_float(r.get("Goal_Line", None), 0.0) or 0.0
    mat = str(r.get("Matrix_Goal", "") or "").strip().lower()
    conf = _safe_float(r.get("Conf_Goal", None), None)
    stance_ok = bool(line == 0.5 and mat.startswith("g") and (conf is not None and conf >= 85))

    if not stance_ok:
        return None

    xga   = _safe_float(r.get("opp_5v5_xGA60", None), None)
    oppsog = _safe_float(r.get("Opp_SOG_Against_L10", None), None)

    ixg = None
    for k in ("iXG%", "iXG_pct", "iXG_Pct", "ixg_pct", "ixg%"):
        if k in r:
            ixg = _safe_float(r.get(k, None), None)
            if ixg is not None:
                break
    share = None
    for k in ("Player_5v5_SOG_Share", "Player_5v5_SOG_Share_Pct", "Player_5v5_SOGShare"):
        if k in r:
            share = _safe_float(r.get(k, None), None)
            if share is not None:
                break
    drought_g = _safe_float(r.get("Drought_G", None), None)
    team_gf = None
    for k in ("Team_GF_Avg_L5", "Team_GF_L5", "Team_GF_Avg5", "Team_GF_L5_Avg"):
        if k in r:
            team_gf = _safe_float(r.get(k, None), None)
            if team_gf is not None:
                break

    team_gf = None
    for k in ("Team_GF_Avg_L5", "Team_GF_L5", "Team_GF_Avg5", "Team_GoalsFor_Avg_L5"):
        if k in r:
            team_gf = _safe_float(r.get(k, None), None)
            if team_gf is not None:
                break

    team_gf = _safe_float(r.get("Team_GF_Avg_L5", None), None)

    team_gf = _safe_float(r.get("Team_GF_Avg_L5", None), None)

    # DPS anchors (from HUD)
    DPS = {
        "armor_shred": (41.8, 189),
        "hot_team_press": (51.9, 81),
        "envmix_50": (50.0, 64),
        "envmix_elite": (58.5, 41),
        "fenrir_34": (40.0, 200),
        "fenrir_36": (60.7, 28),
        "fury_35": (47.8, 113),
        "fury_37": (52.6, 76),
        "fury_40": (63.5, 52),
        "tyrs_wrath_unleashed": (72.0, 25),
        "armor_annihilation": (54.5, 66),
        "smash": (57.7, 52),
        "valhalla": (61.4, 44),
        "fury_shredder": (73.3, 15),
    }

    opp_lane = bool(oppsog is not None and oppsog >= 29)
    env_249  = bool(xga is not None and xga >= 2.49)
    env_252  = bool(xga is not None and xga >= 2.52)

    procs = []

    # Tyr’s Wrath Unleashed
    tyr_on = bool(opp_lane and env_252 and (share is not None and share >= 15) and (ixg is not None and ixg >= 97))
    procs.append(("Tyr’s Wrath Unleashed", *DPS["tyrs_wrath_unleashed"], tyr_on))

    # Armor Annihilation stack
    aa_on = bool((ixg is not None and ixg >= 97) and env_252)
    procs.append(("Armor Annihilation", *DPS["armor_annihilation"], aa_on))
    procs.append(("Smash", *DPS["smash"], bool(aa_on and conf is not None and conf >= 91)))
    procs.append(("Valhalla", *DPS["valhalla"], bool(aa_on and conf is not None and conf >= 95)))

    # Fury ladder (shot funnel)
    procs.append(("Fury Shredder", *DPS["fury_shredder"], bool(env_252 and (ixg is not None and ixg >= 94) and opp_lane and (drought_g is not None and drought_g >= 2))))
    procs.append(("Fury 40", *DPS["fury_40"], bool(opp_lane and env_249 and (ixg is not None and ixg >= 94))))
    procs.append(("Fury 37", *DPS["fury_37"], bool(opp_lane and env_249)))
    procs.append(("Fury 35", *DPS["fury_35"], bool(opp_lane)))

    # Fenrir
    procs.append(("Fenrir’s Fury 99+", *DPS["fenrir_36"], bool((ixg is not None and ixg >= 99) and (xga is not None and xga >= 2.55))))
    procs.append(("Fenrir’s Fury 97+", *DPS["fenrir_34"], bool(ixg is not None and ixg >= 97)))

    # Armor Shred alone
    procs.append(("Armor Shred", *DPS["armor_shred"], bool(env_249)))

    # Press the Attack (xGA>=2.49 & iXG>=96.5 & Team_GF_Avg_L5>=3.0)
    procs.append(("Press the Attack", *DPS["hot_team_press"], bool(env_249 and (ixg is not None and ixg >= 96.5) and (team_gf is not None and team_gf >= 3.0))))

    # Berserker Aggression (Env Mix) — xGA>=2.55 & iXG>=92 & Team_GF>=3.0 (requires Goal_Odds >= +150 for board eligibility)
    goal_odds = _safe_float(r.get("Goal_Odds_Over", r.get("Goal_Odds", None)), None)
    envmix_on = bool((xga is not None and xga >= 2.55) and (ixg is not None and ixg >= 92) and (team_gf is not None and team_gf >= 3.0))
    envmix_odds_ok = bool(goal_odds is not None and goal_odds >= 150)
    procs.append(("Berserker Aggression (Env Mix)", *DPS["envmix_50"], bool(envmix_on)))

    # Berserker Aggression (Env Mix • ELITE) — xGA>=2.55 & iXG>=92 & Team_GF>=3.8 (no odds requirement)
    procs.append(("Berserker Aggression (Env Mix • ELITE)", *DPS["envmix_elite"], bool((xga is not None and xga >= 2.55) and (ixg is not None and ixg >= 92) and (team_gf is not None and team_gf >= 3.8))))

    best = None
    for title, win, n, cond in procs:
        if not cond:
            continue
        # Board eligibility guard: Env Mix requires Goal_Odds_Over >= +150 to qualify as best-proc/board pick.
        if title == "Berserker Aggression (Env Mix)" and not envmix_odds_ok:
            continue
        adj = _adj_win(win, n, k=20)
        if (best is None) or (adj > best["adj"]) or (abs(adj - best["adj"]) < 1e-9 and n > best["n"]):
            best = {"title": title, "win": float(win), "n": int(n), "adj": float(adj)}
    return best

def _probe_sog_best(r: dict) -> dict | None:
    # Use the SOG HUD procs (2.5 and 3.5). Exclude 3.5 BASE.
    line = _safe_float(r.get("SOG_Line"))
    line = 0.0 if line is None else float(line)
    conf = _safe_float(r.get("Conf_SOG")) or 0.0
    mat_green = _is_matrix_green(str(r.get("Matrix_SOG", "") or ""))

    # 3.5 sniper spec
    if line >= 3.5:
        l40 = _safe_float(r.get("L40_Rate_SOG", r.get("L40_Rate_SOG", 0))) or 0.0
        xga = _safe_float(r.get("opp_5v5_xGA60", r.get("Opp_5v5_xGA60", r.get("opp_xGA60", 0)))) or 0.0
        hdca = _safe_float(r.get("opp_5v5_HDCA60", r.get("Opp_5v5_HDCA60", r.get("opp_HDCA60", 0)))) or 0.0
        share = _safe_float(r.get("Player_5v5_SOG_Share", r.get("SOG_Share_5v5", r.get("Player_SOG_Share_5v5", 0)))) or 0.0
        opp_l50 = _safe_float(r.get("Opp_SOG_Against_L50", r.get("OppSOG_L50", r.get("Opp_SOG_L50", 0)))) or 0.0

        if not (mat_green and conf >= 75):
            return None

        permission_shatter = (xga >= 2.50) or (hdca >= 2.20)
        enraged = (l40 >= 3.0) and (xga >= 2.50)
        elite_enraged = enraged and (share >= 20.0)
        enraged_shatter = (opp_l50 >= 29.5) and permission_shatter

        procs = [
            ("SNIPER CRIT", 71.4, 28, elite_enraged),
            ("STRONG", 60.4, 53, enraged),
            ("PERMISSION SPECIAL", 60.7, 28, enraged_shatter),
            ("Enhanced Enraged (Share ≥ 18)", 64.1, 39, (enraged and share >= 18.0)),
        ]
        best=None
        for title, win, n, cond in procs:
            if not cond:
                continue
            adj=_adj_win(win,n,k=20)
            if best is None or adj>best["adj"] or (abs(adj-best["adj"])<1e-9 and n>best["n"]):
                best={"title": title, "win": float(win), "n": int(n), "adj": float(adj)}
        return best

    # 2.5 kit
    xga = _safe_float(r.get("opp_5v5_xGA60", r.get("Opp_5v5_xGA60", 0))) or 0.0
    drought = _safe_float(r.get("Drought_SOG", r.get("Drought_S", 0))) or 0.0
    avg5 = _safe_float(r.get("Avg5_SOG", r.get("Avg5", 0))) or 0.0
    sipct = _safe_float(r.get("ShotIntent_Pct", r.get("ShotIntentPct", r.get("SI_Pct", 0)))) or 0.0
    l20 = _safe_float(r.get("L20_Rate_SOG", r.get("L20_Rate", r.get("L20_SOG_Rate", 0)))) or 0.0
    share = _safe_float(r.get("Player_5v5_SOG_Share", r.get("SOG_5v5_Share", 0))) or 0.0
    opp_sog50 = _safe_float(r.get("Opp_SOG_Against_L50", r.get("Opp_SOG_Against_50", r.get("Opp_SOG_Against_50g", 0)))) or 0.0

    base25 = (mat_green and (line > 0 and line <= 2.5) and (conf >= 75))
    if not base25:
        return None

    # DPS anchors (from HUD)
    procs = [
        ("Siege (Ultimate)", 90.9, 11, (l20 >= 3.0 and share >= 16 and conf >= 83 and xga >= 2.50)),
        ("Assassin’s Overdrive (Elite)", 81.0, 21, (l20 >= 3.4 and share >= 16)),
        ("Berserker’s Rage (Strong)", 73.0, 37, (l20 >= 3.0 and share >= 16)),
        ("Berserker Volley (Role)", 61.8, 55, (share >= 16)),
        ("Berserker Swipe (Backbone)", 63.1, 65, (l20 >= 3.0)),
        ("Locked & Loaded (Conf Spike)", 59.2, 49, (conf >= 82)),
        ("Berserker’s Patience", 60.5, 43, (drought >= 1)),
        ("Bloodthirst", 66.7, 27, (drought >= 1 and xga >= 2.48)),
        ("Shattered Armor (Crit)", 71.4, 35, (xga >= 2.55)),
        ("Shattered Ice II", 65.9, 41, (xga >= 2.50)),
        ("Shattered Ice I", 58.0, 69, (xga >= 2.46)),
        ("Berserker’s Barrage (Shots Allowed)", 58.9, 73, (opp_sog50 >= 27.5)),
        ("Enraged Strike", 57.1, 21, (sipct >= 96.0)),
        ("Elite Enraged Strike", 66.7, 18, (sipct >= 96.5)),
    ]

    best=None
    for title, win, n, cond in procs:
        if not cond:
            continue
        adj=_adj_win(win,n,k=20)
        if best is None or adj>best["adj"] or (abs(adj-best["adj"])<1e-9 and n>best["n"]):
            best={"title": title, "win": float(win), "n": int(n), "adj": float(adj)}
    return best

def _probe_best_proc(mkt: str, r: dict) -> dict | None:
    mk = str(mkt or "").strip().upper()
    if mk == "POINTS":
        return _probe_points_best(r)
    if mk == "ASSISTS":
        return _probe_assists_best(r)
    if mk in ("GOALS","GOAL","ATG"):
        return _probe_goals_best(r)
    if mk == "SOG":
        return _probe_sog_best(r)
    return None

def _render_sog_combat_hud(r):
    """SOG COMBAT HUD (Berserker kit) — EV ignored.

    Locked spec source: SOG_Balance_Notes_v1.0 (v1.0 → v1.1):
      - Global Guardrails: Line≤2.5, Matrix=Green, ShotIntent≥3.4, Conf_SOG≥75
      - Moves: Swipe (basic), Volley (mu), Rage (proc), Frenzy (state), Paralysis (ENV),
               Siege (ultimate, display-only), Locked & Loaded (timing), Barrage (engine-on),
               Assassin's Overdrive (execution cliff).
    """
    def _num(v, default=0.0):
        try:
            x = float(v)
            if math.isnan(x):
                return default
            return x
        except Exception:
            return default

    # Core inputs
    line = _num(r.get("SOG_Line", 0), 0.0)
    conf = _num(r.get("Conf_SOG", 0), 0.0)

    # --- SOG 3.5: Jungle — Sniper Spec HUD (separate from 2.5 Jungle) ---
    if line >= 3.5:
        # Core 3.5 inputs (robust defaults)
        l40 = _num(r.get("L40_Rate_SOG", r.get("L40_Rate_SOG", 0)), 0.0)
        xga = _num(r.get("opp_5v5_xGA60", r.get("Opp_5v5_xGA60", r.get("opp_xGA60", 0))), 0.0)
        hdca = _num(r.get("opp_5v5_HDCA60", r.get("Opp_5v5_HDCA60", r.get("opp_HDCA60", 0))), 0.0)
        share = _num(r.get("Player_5v5_SOG_Share", r.get("SOG_Share_5v5", r.get("Player_SOG_Share_5v5", 0))), 0.0)
        opp_l50 = _num(r.get("Opp_SOG_Against_L50", r.get("OppSOG_L50", r.get("Opp_SOG_L50", 0))), 0.0)

        permission_shatter = (xga >= 2.50) or (hdca >= 2.20)
        enraged = (l40 >= 3.0) and (xga >= 2.50)
        elite_enraged = enraged and (share >= 20.0)
        enhanced_enraged_1 = enraged and (share >= 18.0)
        enraged_shatter = (opp_l50 >= 29.5) and permission_shatter

        # Tier resolver (top-down)
        if elite_enraged:
            tier = "SNIPER CRIT"
            n, winp = 28, 71.4
        elif enraged:
            tier = "STRONG"
            n, winp = 53, 60.4
        elif enraged_shatter:
            tier = "PERMISSION SPECIAL"
            n, winp = 28, 60.7
        else:
            tier = "BASE"
            n, winp = 200, 42.5

        st.markdown("**Combat HUD (SOG 3.5): Jungle — Sniper Spec**")

        st.markdown("**STANCE**")
        st.markdown("- Gate: Matrix = Green • Line ≥ 3.5 • Conf ≥ 75 • EV ignored")
        st.markdown("- Permission Shatter: xGA ≥ 2.50 **OR** HDCA ≥ 2.20")

        st.markdown("**MOVES (≥50% DPS anchors)**")
        # Only show ACTIVE 50%+ moves (with formulas + DPS bars). If none are active, do not spam anchor bullets.
        moves = [
            ("SNIPER CRIT", 28, 71.4, elite_enraged,
             f"L40 {l40:.2f} ≥ 3.0 • xGA {xga:.2f} ≥ 2.50 • Share {share:.1f} ≥ 20"),
            ("STRONG", 53, 60.4, enraged,
             f"L40 {l40:.2f} ≥ 3.0 • xGA {xga:.2f} ≥ 2.50"),
            ("PERMISSION SPECIAL", 28, 60.7, enraged_shatter,
             f"OppSOG_L50 {opp_l50:.1f} ≥ 29.5 • (xGA {xga:.2f} ≥ 2.50 OR HDCA {hdca:.2f} ≥ 2.20)"),
            ("Enhanced Enraged (Share ≥ 18)", 39, 64.1, enhanced_enraged_1,
             f"L40 {l40:.2f} ≥ 3.0 • xGA {xga:.2f} ≥ 2.50 • Share {share:.1f} ≥ 18"),
        ]

        active_any = False
        for name, nn, wp, active, formula in moves:
            if not active:
                continue
            active_any = True
            st.markdown(f"- ✅ **{name}** — {formula} — n={nn} • Win%={wp}")
            try:
                _wl_dps_bar(float(wp), "SOG")
            except Exception:
                pass

        if not active_any:
            st.markdown("- **No 50%+ moves active — RESOLVED: BASE**")
        else:
            st.markdown(f"- **RESOLVED:** {tier} — n={n} • Win%={winp}")

        st.markdown("**SUPPORT**")

        st.markdown(
            f"- Conf {conf:.0f} • L40_Rate_SOG {l40:.2f} • Share {share:.1f} "
            f"• opp_xGA60 {xga:.2f} • opp_HDCA60 {hdca:.2f} • Opp_SOG_L50 {opp_l50:.1f}"
        )
        return

    # ShotIntent / SI (column can vary)
    si = _num(r.get("ShotIntent", r.get("SI", r.get("ShotIntent_SOG", r.get("SI_SOG", 0)))), 0.0)

    # Timing
    rg = _num(r.get("Reg_Gap_S10", r.get("RegGap_S10", r.get("Reg_Gap_S", 0))), 0.0)

    # Heavy expectation (mu)
    mu = _num(r.get("SOG_mu", r.get("SOG_Mu", r.get("mu_sog", 0))), 0.0)

    # Proc / state / env helpers
    sipct = _num(r.get("ShotIntent_Pct", r.get("ShotIntentPct", r.get("SI_Pct", 0))), 0.0)
    actual = _num(r.get("Actual_SOG", r.get("SOG_Actual", r.get("SOG_Last", 0))), 0.0)
    xga = _num(r.get("opp_5v5_xGA60", r.get("Opp_5v5_xGA60", 0)), 0.0)

    mat_green = _is_matrix_green(str(r.get("Matrix_SOG", "") or ""))

    # -------------------------
    # NEW JUNGLE ADDITIONS (Regression + Engine stacks)
    #   - These are presentation-only procs for the SOG HUD/player card.
    #   - EV ignored; we key only off Matrix/Line and the relevant signals.
    # -------------------------
    drought = _num(r.get("Drought_SOG", r.get("Drought_S", 0)), 0.0)
    avg5 = _num(r.get("Avg5_SOG", r.get("Avg5", 0)), 0.0)
    hdca = _num(r.get("opp_5v5_HDCA60", r.get("Opp_5v5_HDCA60", 0)), 0.0)

    # Base universe for SOG Smash (EV ignored): Matrix Green + line≤2.5 + Conf≥75
    # NOTE: 2.5 Jungle now supports multiple independent paths:
    #   - Shooter ladder (L20 + Share)
    #   - Conf Spike (Conf) as a standalone proc
    #   - Timing (Drought) snapback
    #   - Armor (opp xGA) tiers
    drought = _num(r.get("Drought_SOG", r.get("Drought_S", 0)), 0.0)
    avg5 = _num(r.get("Avg5_SOG", r.get("Avg5", 0)), 0.0)

    # New 2.5 backbone inputs
    l20 = _num(r.get("L20_Rate_SOG", r.get("L20_Rate", r.get("L20_SOG_Rate", 0))), 0.0)
    share = _num(
        r.get(
            "Player_5v5_SOG_Share",
            r.get("Player_5v5_SOG_SOG_Share", r.get("Player_5v5_SOGShare", r.get("SOG_5v5_Share", 0))),
        ),
        0.0,
    )
    opp_sog50 = _num(
        r.get("Opp_SOG_Against_L50", r.get("Opp_SOG_Against_50", r.get("Opp_SOG_Against_50g", 0))),
        0.0,
    )

    base25 = (mat_green and (line > 0 and line <= 2.5) and (conf >= 75))

    # -------------------------
    # 2.5 SHOOTER LADDER (NEW)
    # -------------------------
    swipe_on = (base25 and (l20 >= 3.0))                       # Backbone
    volley_on = (base25 and (share >= 16))                     # Role
    rage_on = (base25 and (l20 >= 3.0) and (share >= 16))       # Strong
    overdrive_on = (base25 and (l20 >= 3.4) and (share >= 16))  # Elite

    # -------------------------
    # 2.5 MACRO (NEW)
    # -------------------------
    locked_loaded_on = (base25 and (conf >= 82))               # Macro-on bar (volume entry)
    # 2.5 Siege (Ultimate): Ultra permission stack
    siege_on = (base25 and (l20 >= 3.0) and (share >= 16) and (conf >= 83) and (xga >= 2.50))

    # -------------------------
    # 2.5 TIMING (keep — proven)
    # -------------------------
    patience_on = (base25 and (drought >= 1))
    surge_on = (patience_on and (avg5 >= 2.5))
    bloodthirst_on = (patience_on and (xga >= 2.48))

    # -------------------------
    # 2.5 ARMOR (NEW 3-tier xGA mapping)
    # -------------------------
    shattered_ice_1_on = (base25 and (xga >= 2.46))            # Armor I
    shattered_ice_2_on = (base25 and (xga >= 2.50))            # Armor II
    shattered_armor_on = (base25 and (xga >= 2.55))            # Armor III (crit)
    paralysis_on = shattered_ice_2_on                          # align paralysis with “real” armor

    # -------------------------
    # Shots allowed badge (2.5)
    # -------------------------
    barrage_on = (base25 and (opp_sog50 >= 27.5))

    # -------------------------
    # Legacy SI% crit lane (keep until 3.5 decision)
    # -------------------------
    enraged_on = (base25 and (sipct >= 96.0))
    elite_enraged_on = (base25 and (sipct >= 96.5))
    enhanced_enraged_1_on = (enraged_on and (xga >= 2.48))
    enhanced_enraged_2_on = (enraged_on and (xga >= 2.51))
    killing_blow_on = False  # retired (superseded by Enraged Shatter)

    # Legacy bridge keys (keep until retest under new armor tiers)
    shattered_ice_swipe_248_on = (shattered_ice_1_on and (si >= 3.4))
    shattered_ice_swipe_251_on = (shattered_ice_2_on and (si >= 3.4))

    # ULT chain (legacy SI% + xGA) — keep
    enraged_shatter_on = (base25 and (xga >= 2.41) and (sipct >= 97.0))

    # DPS anchors

    # DPS anchors (presentation only) — pulled from your Balance Notes
    DPS = {
        # -------------------------
        # CORE LADDER (2.5) — NEW
        # -------------------------
        "swipe": {"n": 65, "win": 63.1},        # L20>=3.0
        "volley": {"n": 55, "win": 61.8},       # Share>=16
        "rage": {"n": 37, "win": 73.0},         # L20>=3.0 + Share>=16
        "overdrive": {"n": 21, "win": 81.0},    # L20>=3.4 + Share>=16

        # -------------------------
        # MACRO (Conf) — NEW
        # -------------------------
        "locked_loaded": {"n": 49, "win": 59.2},  # Conf>=82 (macro-on BAR)

        # -------------------------
        # TIMING / DROUGHT — NEW where proven
        # -------------------------
        "patience": {"n": 43, "win": 60.5},       # Drought_SOG>=1
        "bloodthirst": {"n": 27, "win": 66.7},    # Drought>=1 + xGA>=2.48
        "surge": {"n": 14, "win": 64.3},          # keep anchor (not re-tested in new set)

        # -------------------------
        # ARMOR (xGA) 3 tiers — NEW
        # -------------------------
        "shattered_ice_1": {"n": 69, "win": 58.0},  # xGA>=2.46  (Armor I / volume)
        "shattered_ice_2": {"n": 41, "win": 65.9},  # xGA>=2.50  (Armor II / real)
        "shattered_armor": {"n": 35, "win": 71.4},  # xGA>=2.55  (Armor III / crit)
        "paralysis": {"n": 41, "win": 65.9},        # align paralysis with xGA>=2.50

        # -------------------------
        # SHOTS ALLOWED badge — NEW
        # -------------------------
        "barrage": {"n": 73, "win": 58.9},          # Opp_SOG_Against_L50>=27.5

        # -------------------------
        # SIEGE (ULTIMATE) — NOW OFFICIAL
        # -------------------------
        "siege": {"n": 11, "win": 90.9},            # L20>=3 + Share>=16 + Conf>=83 + xGA>=2.50

        # -------------------------
        # LEGACY 3.5 / SI% lane — KEEP for now (no breakage)
        # -------------------------
        "enraged": {"n": 21, "win": 57.1},
        "elite_enraged": {"n": 18, "win": 66.7},
        "enhanced_enraged_1": {"n": 12, "win": 83.3},
        "enhanced_enraged_2": {"n": 9, "win": 88.9},
        "enraged_shatter": {"n": 9, "win": 100.0},

        # Legacy bridge keys referenced by current HUD (keep)
        "shattered_ice_swipe_248": {"n": 15, "win": 73.3},
        "shattered_ice_swipe_251": {"n": 12, "win": 75.0},
    }
    base_win = 53.5


    # Tier resolver (2.5) — single identity label
    tier = "BASE"
    if siege_on:
        tier = "SIEGE"
    elif (base25 and (l20 >= 3.0) and (share >= 16) and (conf >= 83)):
        tier = "MACRO"
    elif overdrive_on:
        tier = "ELITE"
    elif rage_on:
        tier = "STRONG"
    elif (base25 and (l20 >= 3.0) and (conf >= 80)):
        tier = "BACKBONE+MACRO"
    elif swipe_on:
        tier = "BACKBONE"


    
    st.markdown("**Combat HUD (SOG):**")

    # Big class emblems (presentation-only)
    st.markdown("""
<style>
.sog-class-header{display:flex;align-items:center;gap:14px;margin:10px 0 6px 0;}
.sog-class-icon{width:28px;height:28px;display:inline-flex;align-items:center;justify-content:center;flex:0 0 28px;}
.sog-class-icon svg{width:28px !important;height:28px !important;max-width:28px;max-height:28px;}
.sog-class-title{font-size:20px;font-weight:900;line-height:1.1;margin:0;}
.sog-class-passive{opacity:0.75;margin-top:2px;font-size:13px;font-weight:600;}
.sog-class-divider{height:1px;background:rgba(17,24,39,0.12);margin:10px 0 8px 0;}
</style>
        """, unsafe_allow_html=True)
    # -------------------------
    # Class UI (presentation-only)
    # -------------------------
    shooter_active = any([swipe_on, volley_on, rage_on, overdrive_on, enraged_on, elite_enraged_on])
    timing_active = any([locked_loaded_on, siege_on, patience_on, surge_on, bloodthirst_on])
    env_active = any([shattered_ice_1_on, shattered_ice_2_on, shattered_armor_on, paralysis_on, barrage_on, enhanced_enraged_1_on, enhanced_enraged_2_on])

    def _pill(label: str, on: bool) -> str:
        bg = "#22c55e" if on else "#e5e7eb"
        fg = "white" if on else "#111827"
        return f"""<span style='display:inline-block;padding:2px 10px;border-radius:999px;
                         background:{bg};color:{fg};font-size:12px;font-weight:700;margin-right:6px;'>{label}</span>"""

    st.markdown(
        _pill("SHOOTER", shooter_active) + _pill("TIMING", timing_active) + _pill("ENV", env_active),
        unsafe_allow_html=True,
    )


    # Stance / eligibility (Guardrails)
    if base25:
        _wl_why_line(
            _svg_icon("sog_basic_swipe.svg", "Jungle Stance (SOG)", "wl-sog"),
            f"Guardrails met — Green / line≤2.5 / Conf≥75  •  EV ignored",
        )
        _wl_why_line(
            _svg_icon("sog_basic_swipe.svg", "Tier", "wl-sog"),
            f"Tier: {tier}",
        )
    else:
        _wl_why_line(
            _svg_icon("sog_basic_swipe.svg", "Jungle Stance (SOG)", "wl-sog"),
            "Guardrails NOT met — needs Matrix Green + line≤2.5 + Conf≥75",
        )
    # Rank label tracking (presentation only)
    _best_title = ""
    _best_win = None
    _best_n = 0
    _best_aw = -1.0
    def _track_best(title: str, win: float, n: int) -> None:
        nonlocal _best_title, _best_win, _best_n, _best_aw
        aw = _adj_win(win, n, k=20)
        nn = int(n) if n is not None else 0
        if (aw > _best_aw) or (abs(aw - _best_aw) < 1e-9 and nn > _best_n):
            _best_aw = aw
            _best_title = title
            _best_win = float(win)
            _best_n = nn


    def _render_move(icon_file: str, title: str, body: str, win: float, n: int, show_bar: bool = True) -> None:
        _track_best(title, win, n)
        _wl_why_line(_svg_icon(icon_file, title, "wl-sog"), body)
        if show_bar:
            _wl_dps_bar(win, "SOG")

    def _section(title: str, passive: str, active: bool, icon_file: str = "") -> None:
        icon_svg = _load_svg_icon(icon_file) if icon_file else ""
        st.markdown(
            f"""
<div class='sog-class-header'>
  <div class='sog-class-icon'>{icon_svg}</div>
  <div>
    <div class='sog-class-title'>{title}</div>
    <div class='sog-class-passive'>{passive}</div>
  </div>
</div>
<div class='sog-class-divider'></div>
            """,
            unsafe_allow_html=True,
        )
        if not active:
            st.caption("No procs active.")
            st.markdown("---")

    # -------------------------
    # SHOOTER CLASS — Rageborn Marksman / Shooter
    # -------------------------

    # -------------------------
    # MARKSMAN / Shooter (2.5 ladder)
    # -------------------------
    _section(
        "Rageborn Marksman (Shooter)",
        "Passive: Shooter ladder is L20 backbone + role Share. (Highest rung only.)",
        shooter_active,
        icon_file="sog_class_shooter.svg",
    )

    # Highest-rung-only ladder render
    best_key = None
    if overdrive_on:
        best_key = "overdrive"
    elif rage_on:
        best_key = "rage"
    elif volley_on:
        best_key = "volley"
    elif swipe_on:
        best_key = "swipe"

    if best_key == "overdrive":
        _render_move(
            "sog_assassins_overdrive.svg",
            "Assassin’s Overdrive (Elite)",
            f"Overdrive — L20 {l20:.2f} ≥ 3.4 | Share {share:.1f} ≥ 16  •  DPS {DPS['overdrive']['win']}% (n={DPS['overdrive']['n']})  (Δ {DPS['overdrive']['win']-base_win:+.1f})",
            DPS["overdrive"]["win"],
            DPS["overdrive"]["n"],
        )
    elif best_key == "rage":
        _render_move(
            "sog_berserkers_rage.svg",
            "Berserker’s Rage (Strong)",
            f"Rage — L20 {l20:.2f} ≥ 3.0 | Share {share:.1f} ≥ 16  •  DPS {DPS['rage']['win']}% (n={DPS['rage']['n']})  (Δ {DPS['rage']['win']-base_win:+.1f})",
            DPS["rage"]["win"],
            DPS["rage"]["n"],
        )
    elif best_key == "volley":
        _render_move(
            "sog_berserker_volley.svg",
            "Berserker Volley (Role)",
            f"Volley — Share {share:.1f} ≥ 16  •  DPS {DPS['volley']['win']}% (n={DPS['volley']['n']})  (Δ {DPS['volley']['win']-base_win:+.1f})",
            DPS["volley"]["win"],
            DPS["volley"]["n"],
        )
    elif best_key == "swipe":
        _render_move(
            "sog_basic_swipe.svg",
            "Berserker Swipe (Backbone)",
            f"Swipe — L20 {l20:.2f} ≥ 3.0  •  DPS {DPS['swipe']['win']}% (n={DPS['swipe']['n']})  (Δ {DPS['swipe']['win']-base_win:+.1f})",
            DPS["swipe"]["win"],
            DPS["swipe"]["n"],
        )

    # Legacy SI% ladder (kept until 3.5 kit is finalized)
    if base25 and enraged_on:
        _render_move(
            "sog_enraged_strike.svg",
            "Enraged Strike",
            f"Enraged Strike — SI% {sipct:.1f} ≥ 96  •  DPS {DPS['enraged']['win']}% (n={DPS['enraged']['n']})",
            DPS["enraged"]["win"],
            DPS["enraged"]["n"],
        )
    if base25 and elite_enraged_on:
        _render_move(
            "sog_elite_enraged_strike.svg",
            "Elite Enraged Strike",
            f"Elite Enraged Strike — SI% {sipct:.1f} ≥ 96.5  •  DPS {DPS['elite_enraged']['win']}% (n={DPS['elite_enraged']['n']})",
            DPS["elite_enraged"]["win"],
            DPS["elite_enraged"]["n"],
        )

    st.markdown("---")

    # -------------------------
    # TIMING CLASS — Warfield of the Damned / Timing
    # -------------------------

    # -------------------------
    # TIMING CLASS — Conf Spike / Drought / Specials
    # -------------------------
    _section(
        "Regression Master (Timing)",
        "Passive: Conf Spike (Conf) + snapback (Drought) layer onto the shooter ladder. Specials live here.",
        timing_active,
        icon_file="sog_class_timing.svg",
    )

    # Macro proc (bar)
    if locked_loaded_on:
        _render_move(
            "sog_locked_loaded.svg",
            "Locked & Loaded (Conf Spike)",
            f"Locked & Loaded — Conf {conf:.0f} ≥ 82  •  DPS {DPS['locked_loaded']['win']}% (n={DPS['locked_loaded']['n']})  (Δ {DPS['locked_loaded']['win']-base_win:+.1f})",
            DPS["locked_loaded"]["win"],
            DPS["locked_loaded"]["n"],
        )

    # Macro Grade (single line; best-looking / clean)
    def _pick_ladder(ladder, conf_val):
        for thr, n, win in ladder:
            if conf_val >= thr:
                return thr, n, win
        return None

    CONF_SOLO_LADDER = [(85, 25, 72.0), (84, 32, 68.8), (83, 43, 65.1), (82, 49, 59.2)]
    L20_CONF_LADDER = [(86, 12, 83.3), (85, 19, 78.9), (84, 24, 75.0), (83, 33, 72.7), (80, 45, 66.7)]

    macro_line = None
    if swipe_on:
        sel = _pick_ladder(L20_CONF_LADDER, conf)
        if sel:
            thr, n, win = sel
            macro_line = f"Conf Spike Grade: L20+Conf {thr} • {win:.1f}% (n{n})"
    else:
        sel = _pick_ladder(CONF_SOLO_LADDER, conf)
        if sel:
            thr, n, win = sel
            macro_line = f"Conf Spike Grade: Conf {thr} • {win:.1f}% (n{n})"

    if macro_line:
        _wl_why_line(
            _svg_icon("sog_locked_loaded.svg", "Conf Spike Grade", "wl-sog"),
            macro_line,
        )

    # Siege (Ultimate) — official 2.5 Ultra Permission
    if siege_on:
        _render_move(
            "sog_berserker_siege.svg",
            "Siege (Ultimate)",
            f"Siege — L20 {l20:.2f} ≥ 3.0 | Share {share:.1f} ≥ 16 | Conf {conf:.0f} ≥ 83 | xGA {xga:.2f} ≥ 2.50  •  DPS {DPS['siege']['win']}% (n={DPS['siege']['n']})",
            DPS["siege"]["win"],
            DPS["siege"]["n"],
        )

    # Specials (text-only; show strongest only)
    mythic_on = (base25 and (l20 >= 3.2) and (share >= 16) and (conf >= 83) and (xga >= 2.50))
    shots_siege_on = (base25 and (opp_sog50 >= 27.5) and (share >= 16) and (xga >= 2.50) and (l20 >= 3.0))

    special_line = None
    if mythic_on:
        special_line = "SPECIALS: Mythic Siege • 100.0% (n8)"
    elif shots_siege_on:
        special_line = "SPECIALS: Shots-Allowed Siege • 92.3% (n13)"

    if special_line:
        _wl_why_line(
            _svg_icon("sog_berserker_siege.svg", "Specials", "wl-sog"),
            special_line,
        )

    # Drought / snapback procs
    if base25 and patience_on:
        _render_move(
            "sog_berserkers_patience.svg",
            "Berserker’s Patience",
            f"Patience — Drought {drought:.0f} ≥ 1  •  DPS {DPS['patience']['win']}% (n={DPS['patience']['n']})",
            DPS["patience"]["win"],
            DPS["patience"]["n"],
        )
    if base25 and surge_on:
        _render_move(
            "sog_berserkers_surge.svg",
            "Berserker’s Surge",
            f"Surge — Patience + Avg5 {avg5:.2f} ≥ 2.5  •  DPS {DPS['surge']['win']}% (n={DPS['surge']['n']})",
            DPS["surge"]["win"],
            DPS["surge"]["n"],
        )
    if base25 and bloodthirst_on:
        _render_move(
            "sog_bloodthirst.svg",
            "Bloodthirst",
            f"Bloodthirst — Patience + opp xGA {xga:.2f} ≥ 2.48  •  DPS {DPS['bloodthirst']['win']}% (n={DPS['bloodthirst']['n']})",
            DPS["bloodthirst"]["win"],
            DPS["bloodthirst"]["n"],
        )

    st.markdown("---")

    # -------------------------
    # ENVIRONMENT CLASS — Opportunist / Armor Breaker
    # -------------------------

    _section(
        "Armor Breaker (Weak Def Environment)",
        "Passive: 3-tier Armor ladder (opp xGA) + shots-allowed badge. (Highest armor tier only.)",
        env_active,
        icon_file="sog_class_env.svg",
    )

    # Highest armor tier only
    armor_key = None
    armor_thr = None
    if shattered_armor_on:
        armor_key, armor_thr = "shattered_armor", 2.55
    elif shattered_ice_2_on:
        armor_key, armor_thr = "shattered_ice_2", 2.50
    elif shattered_ice_1_on:
        armor_key, armor_thr = "shattered_ice_1", 2.46

    if base25 and armor_key:
        icon_map = {
            "shattered_ice_1": "sog_shattered_ice_1.svg",
            "shattered_ice_2": "sog_shattered_ice_2.svg",
            "shattered_armor": "sog_shattered_armor.svg",
        }
        title_map = {
            "shattered_ice_1": "Shattered Ice I",
            "shattered_ice_2": "Shattered Ice II",
            "shattered_armor": "Shattered Armor (Crit)",
        }
        _render_move(
            icon_map[armor_key],
            title_map[armor_key],
            f"Armor — opp xGA {xga:.2f} ≥ {armor_thr:.2f}  •  DPS {DPS[armor_key]['win']}% (n={DPS[armor_key]['n']})",
            DPS[armor_key]["win"],
            DPS[armor_key]["n"],
        )
        _wl_why_line(
            _svg_icon(icon_map[armor_key], "Armor Grade", "wl-sog"),
            f"Armor Grade: xGA {armor_thr:.2f} • {DPS[armor_key]['win']:.1f}% (n{DPS[armor_key]['n']})",
        )

    # Paralysis (env proc aligned to Armor II)
    if base25 and paralysis_on:
        _render_move(
            "sog_env_paralysis.svg",
            "Paralysis (Armor II)",
            f"Paralysis — opp xGA {xga:.2f} ≥ 2.50  •  DPS {DPS['paralysis']['win']}% (n={DPS['paralysis']['n']})",
            DPS["paralysis"]["win"],
            DPS["paralysis"]["n"],
        )

    # Shots Allowed badge
    if base25 and barrage_on:
        _render_move(
            "sog_berserkers_barrage.svg",
            "Berserker’s Barrage (Shots Allowed)",
            f"Barrage — Opp SOG Against L50 {opp_sog50:.1f} ≥ 27.5  •  DPS {DPS['barrage']['win']}% (n={DPS['barrage']['n']})",
            DPS["barrage"]["win"],
            DPS["barrage"]["n"],
        )

    # Legacy bridges / crit ladder (kept until 3.5 decision)
    if base25 and shattered_ice_swipe_248_on:
        _render_move(
            "sog_shattered_ice_swipe.svg",
            "Armor Fracture I",
            f"Armor Fracture I — Armor I + SI {si:.2f} ≥ 3.4  •  DPS {DPS['shattered_ice_swipe_248']['win']}% (n={DPS['shattered_ice_swipe_248']['n']})",
            DPS["shattered_ice_swipe_248"]["win"],
            DPS["shattered_ice_swipe_248"]["n"],
        )
    if base25 and shattered_ice_swipe_251_on:
        _render_move(
            "sog_shattered_ice_swipe.svg",
            "Armor Fracture II",
            f"Armor Fracture II — Armor II + SI {si:.2f} ≥ 3.4  •  DPS {DPS['shattered_ice_swipe_251']['win']}% (n={DPS['shattered_ice_swipe_251']['n']})",
            DPS["shattered_ice_swipe_251"]["win"],
            DPS["shattered_ice_swipe_251"]["n"],
        )

    if base25 and enraged_shatter_on:
        _render_move(
            "sog_enraged_shatter.svg",
            "Enraged Shatter (Legacy ULT)",
            f"Enraged Shatter — opp xGA {xga:.2f} ≥ 2.41 + SI% {sipct:.1f} ≥ 97.0  •  DPS {DPS['enraged_shatter']['win']}% (n={DPS['enraged_shatter']['n']})",
            DPS["enraged_shatter"]["win"],
            DPS["enraged_shatter"]["n"],
        )

    if base25 and enhanced_enraged_1_on:
        _render_move(
            "sog_enhanced_enraged_1.svg",
            "Enhanced Enraged I (Legacy)",
            f"Enhanced Enraged I — Enraged + xGA {xga:.2f} ≥ 2.48  •  DPS {DPS['enhanced_enraged_1']['win']}% (n={DPS['enhanced_enraged_1']['n']})",
            DPS["enhanced_enraged_1"]["win"],
            DPS["enhanced_enraged_1"]["n"],
        )
    if base25 and enhanced_enraged_2_on:
        _render_move(
            "sog_enhanced_enraged_2.svg",
            "Enhanced Enraged II (Legacy)",
            f"Enhanced Enraged II — Enraged + xGA {xga:.2f} ≥ 2.51  •  DPS {DPS['enhanced_enraged_2']['win']}% (n={DPS['enhanced_enraged_2']['n']})",
            DPS["enhanced_enraged_2"]["win"],
            DPS["enhanced_enraged_2"]["n"],
        )
    # Rank label (best active proc)
    if _best_title and _best_win is not None:
        _render_rank_line(_best_title, _best_win, _best_n, "SOG")


def _render_points_combat_hud(r: dict) -> None:
    """Render POINTS combat HUD (Fortress 0.5 & DPS 1.5) in the same style as GOALS/ASSISTS/SOG.

    NOTE: Presentation-only. Does NOT change eligibility or EV logic — it only explains signals.
    """
    # Safe pulls
    line = _safe_float(r.get("Points_Line"))
    line = 0.0 if line is None else float(line)

    conf_p = _safe_float(r.get("Conf_Points")) or 0.0
    conf_a = _safe_float(r.get("Conf_Assists")) or 0.0

    ppp = _safe_float(r.get("PPP10_total")) or 0.0
    pp_ixg = _safe_float(r.get("PP_iXG60")) or 0.0
    pp_ixa = _safe_float(r.get("PP_iXA60")) or 0.0

    assists_mu = _safe_float(r.get("Assists_mu")) or 0.0
    points_mu = _safe_float(r.get("Points_mu")) or 0.0

    drought_p = _safe_float(r.get("Drought_P"))
    drought_p = 0.0 if drought_p is None else float(drought_p)

    opp_gaa = _safe_float(r.get("Opp_GAA"))
    team_pp_xgf = _safe_float(r.get("Team_PP_xGF60")) or 0.0
    opp_defweak = _safe_float(r.get("Opp_DefWeak")) or 0.0
    opp_xga = _safe_float(r.get("opp_5v5_xGA60", r.get("Opp_5v5_xGA60")))

    team_gf_l5 = _safe_float(r.get("Team_GF_Avg_L5", r.get("Team_GF_L5")))
    team_gf_l5 = 0.0 if team_gf_l5 is None else float(team_gf_l5)
    opp_sog_l10 = _safe_float(r.get("Opp_SOG_Against_L10", r.get("Opp_SA_Avg_L10", r.get("OppSOG_L10"))))
    opp_sog_l10 = 0.0 if opp_sog_l10 is None else float(opp_sog_l10)

    is_fortress = (line <= 0.75)  # 0.5 build
    _title = "POINTS — Fortress Tank (0.5)" if is_fortress else "POINTS — DPS Tank (1.5)"
    st.markdown(f"### {_title}")

    # Local helper: icon + label line
    def _icon(name: str, title: str) -> str:
        try:
            return _svg_inline(_svg_get(name), size=18, title=title)
        except Exception:
            return ""

    def _row(icon_name: str, label: str, cond: bool, win: float, n: int, note: str = "") -> None:
        if not cond:
            return
        ico = _icon(icon_name, label)
        suffix = f" — {note}" if note else ""
        st.markdown(f"- {ico} **{label}**{suffix}  •  DPS **{win:.1f}%** (n={n})", unsafe_allow_html=True)
        # Match GOALS/ASSISTS/SOG: show a visual DPS bar under each active proc line.
        _wl_dps_bar(win, "POINTS")

    # =========================
    # 0.5 — Fortress Tank (UPDATED: POINTS_MOVES_V2)
    # =========================
    if is_fortress:
        # Keepers (still hit)
        st.markdown("**FLOOR**")
        _row(
            "PTS05_HAMMER_FISTS.svg",
            "Hammer Fists",
            cond=(conf_p >= 70),
            win=69.1,
            n=55,
            note="Conf≥70",
        )

        st.markdown("**TIER**")
        _row(
            "PTS05_HAMMER_STOMP.svg",
            "Echo Stomp I",
            cond=(conf_p >= 70 and ppp >= 3),
            win=75.8,
            n=33,
            note="Conf≥70 + PPP10≥3",
        )
        _row(
            "PTS05_HAMMER_STOMP.svg",
            "Echo Stomp II",
            cond=(conf_p >= 70 and points_mu >= 1.5),
            win=75.9,
            n=29,
            note="Conf≥70 + Points_mu≥1.5",
        )

        # NEW: Gaia’s Blessing ladder (replaces old PP_iXG Gaia variants)
        _row(
            "PTS05_GAIAS_BLESSING.svg",
            "Gaia’s Blessing",
            cond=(conf_p >= 75 and assists_mu >= 0.7),
            win=70.85,
            n=223,
            note="Conf≥75 + Assists_mu≥0.7",
        )

        st.markdown("**CRITS**")
        _row(
            "PTS05_ENRAGED_FURY.svg",
            "Enraged Engine",
            cond=(conf_p >= 78 and points_mu >= 1.5),
            win=92.3,
            n=13,
            note="Conf≥78 + Points_mu≥1.5",
        )
        _row(
            "PTS05_ENRAGED_FURY.svg",
            "Enraged Fury",
            cond=(conf_p >= 78 and assists_mu >= 1.0),
            win=90.0,
            n=10,
            note="Conf≥78 + Assists_mu≥1.0",
        )
        _row(
            "PTS05_BLEED_ENV.svg",
            "Blood Stomp",
            cond=(conf_p >= 78 and (opp_gaa is not None) and (2.5 <= float(opp_gaa) <= 3.0)),
            win=90.9,
            n=11,
            note="Conf≥78 + Opp_GAA 2.5–3.0",
        )
        _row(
            "PTS05_BLEED_ENV.svg",
            "Hammer Fists II",
            cond=(conf_p >= 70 and assists_mu >= 1.1),
            win=85.0,
            n=20,
            note="Conf≥70 + Assists_mu≥1.1",
        )

        # Gaia cliffs (Press / Valhalla)
        _row(
            "PTS05_GAIAS_BLESSING.svg",
            "Gaia’s Blessing+ (Press)",
            cond=(conf_p >= 77 and assists_mu >= 0.7),
            win=73.60,
            n=178,
            note="Conf≥77 + Assists_mu≥0.7",
        )

        st.markdown("**VALHALLA**")
        _row(
            "PTS05_GAIAS_BLESSING.svg",
            "Gaia’s Blessing++ (Smash)",
            cond=(conf_p >= 82 and assists_mu >= 0.7),
            win=78.31,
            n=83,
            note="Conf≥82 + Assists_mu≥0.7",
        )


        # Gaia Heat Ladder (label-only bonus; Gaia must already be active)
        gaia_core = (conf_p >= 75 and assists_mu >= 0.7)
        favor_on = gaia_core and (team_gf_l5 >= 3.5)
        wrath_on = gaia_core and (team_gf_l5 >= 3.7)
        ascension_on = gaia_core and (team_gf_l5 >= 3.9)
        floodgate_on = wrath_on and (opp_sog_l10 >= 27.5)

        st.markdown("**BONUS (Gaia Heat Ladder)**")
        # Show highest active rung only (clean HUD)
        if floodgate_on:
            _row(
                "PTS05_GAIAS_BLESSING.svg",
                "Gaia’s Floodgate",
                cond=True,
                win=81.0,
                n=58,
                note="Gaia + Team_GF_L5≥3.7 + Opp_SOG_L10≥27.5",
            )
        elif ascension_on:
            _row(
                "PTS05_GAIAS_BLESSING.svg",
                "Gaia’s Ascension",
                cond=True,
                win=80.0,
                n=75,
                note="Gaia + Team_GF_L5≥3.9",
            )
        elif wrath_on:
            _row(
                "PTS05_GAIAS_BLESSING.svg",
                "Gaia’s Wrath",
                cond=True,
                win=76.7,
                n=103,
                note="Gaia + Team_GF_L5≥3.7",
            )
        elif favor_on:
            _row(
                "PTS05_GAIAS_BLESSING.svg",
                "Gaia’s Favor",
                cond=True,
                win=73.0,
                n=152,
                note="Gaia + Team_GF_L5≥3.5",
            )

        # Label-only lane (soft bomb) — keep as context, not a booster
        st.markdown("**LABELS (context only)**")
        _row(
            "PTS05_BLEED_ENV.svg",
            "Bleed ENV (Label Only)",
            cond=(conf_p >= 70 and pp_ixg >= 1.5),
            win=76.9,
            n=26,
            note="Conf≥70 + PP_iXG60≥1.5 (label-only)",
        )
        return

    # =========================
    # 1.5 — DPS Tank (UPDATED: POINTS_MOVES_V2)
    # =========================
    st.markdown("**FLOOR**")
    _row(
        "PTS15_TWO_HANDED_HAMMER.svg",
        "Backbone",
        cond=(conf_a >= 89 and points_mu >= 1.7),
        win=54.3,
        n=116,
        note="Conf_Assists≥89 + Points_mu≥1.7",
    )

    st.markdown("**TIER**")
    _row(
        "PTS15_BLADE_IMPALE.svg",
        "Blade Impale (Power Tier)",
        cond=(conf_a >= 89 and points_mu >= 2.2),
        win=60.8,
        n=51,
        note="Conf_Assists≥89 + Points_mu≥2.2",
    )

    st.markdown("**CRITS**")
    _row(
        "PTS15_BLADE_SLASH.svg",
        "Blade Slash (Monster)",
        cond=(conf_a >= 89 and points_mu >= 2.2 and (opp_xga is not None) and float(opp_xga) >= 2.6),
        win=77.8,
        n=18,
        note="Conf_Assists≥89 + Points_mu≥2.2 + opp_xGA≥2.6",
    )
    _row(
        "PTS15_ETERNAL_SMASH.svg",
        "Delayed Hammer Smash",
        cond=(conf_a >= 89 and drought_p >= 1 and points_mu >= 1.7),
        win=67.6,
        n=34,
        note="Conf_Assists≥89 + Drought_P≥1 + Points_mu≥1.7",
    )

    # Optional legacy kit (kept as alternate path; only shows if it procs)
    st.markdown("**LEGACY KIT (optional path)**")
    _row(
        "PTS15_ENCHANTED_HAMMER.svg",
        "Enchanted Hammer (Legacy)",
        cond=(conf_p >= 80 and pp_ixg >= 1.7),
        win=61.1,
        n=18,
        note="Conf_Points≥80 + PP_iXG60≥1.7",
    )
    _row(
        "PTS15_BLADE_IMPALE.svg",
        "Blade Impale (Legacy PP)",
        cond=(conf_p >= 80 and pp_ixa >= 4.0),
        win=49.2,
        n=61,
        note="Conf_Points≥80 + PP_iXA60≥4.0",
    )
    _row(
        "PTS15_BLADE_SLASH.svg",
        "Blade Slash (Legacy PP)",
        cond=(conf_p >= 80 and team_pp_xgf >= 7.0),
        win=48.1,
        n=81,
        note="Conf_Points≥80 + Team_PP_xGF60≥7",
    )
    _row(
        "PTS15_BLOOD_EXPOSURE.svg",
        "Blood Exposure (Legacy)",
        cond=(conf_p >= 80 and team_pp_xgf >= 7.0 and opp_defweak >= 60),
        win=54.5,
        n=44,
        note="Conf_Points≥80 + Team_PP_xGF60≥7 + Opp_DefWeak≥60",
    )
    _row(
        "PTS15_BLOOD_EXPOSURE.svg",
        "Blood Exposure II (Legacy)",
        cond=(conf_p >= 80 and opp_defweak >= 60),
        win=54.7,
        n=64,
        note="Conf_Points≥80 + Opp_DefWeak≥60",
    )
    _row(
        "PTS15_POLARIZING_SMASH.svg",
        "Polarizing Smash (Legacy)",
        cond=(conf_p >= 80 and team_pp_xgf >= 7.0 and opp_defweak >= 70),
        win=54.5,
        n=33,
        note="Conf_Points≥80 + Team_PP_xGF60≥7 + Opp_DefWeak≥70",
    )
    _row(
        "PTS15_ETERNAL_SMASH.svg",
        "Eternal Smash (Legacy)",
        cond=(conf_p >= 80 and opp_defweak >= 70),
        win=53.2,
        n=47,
        note="Conf_Points≥80 + Opp_DefWeak≥70",
    )


def _render_assists_combat_hud(r) -> None:
    """ASSISTS Combat HUD (presentation-only).
    Does NOT change eligibility or EV logic — only explains signals.
    """
    import math
    pp_toi_pct = float('nan')  # SAFETY: prevent NameError if feature pull block is edited

    mk = "ASSISTS"
    market_cls = "wl-assists"

    # ---- stance (Gate to Valhalla) ----
    line = _safe_float(r.get("Assists_Line"))
    if line is None:
        line = _safe_float(r.get("Line_Assists"))
    mat = str(r.get("Matrix_Assists", r.get("Matrix_A", "")) or "").strip().upper()
    conf = _safe_float(r.get("Conf_Assists", r.get("Conf_A", 0))) or 0.0

    stance_ok = (mat in {"GREEN", "🟢"}) and ((line == 0.5) or (line is None)) and (conf >= 80)

    # ---- feature pulls (safe) ----
    ixa_pct    = _safe_float(r.get("iXA%"), default=float("nan"))
    pp_ix      = _safe_float(r.get("PP_iXA60", r.get("PP_iXA_60")), default=float("nan"))
    team_gf_l5 = _safe_float(r.get("Team_GF_L5"), default=float("nan"))
    ppp10      = _safe_float(r.get("PPP10_total"), default=float("nan"))
    assists_mu = _safe_float(r.get("Assists_mu"), default=float("nan"))
    goalie_weak = _safe_float(r.get("Goalie_Weak"), default=float("nan"))
    opp_sv     = _safe_float(r.get("Opp_SV"), default=float("nan"))

    # ---- DPS anchors (your tracker) ----
    DPS = {
        "staff": {"n": 733, "win": 51.2},
        "odin": {"n": 392, "win": 54.3},

        "arcane_3": {"n": 439, "win": 56.3},
        "arcane_4": {"n": 319, "win": 58.0},
        "arcane_5": {"n": 201, "win": 61.2},
        "arcane_6": {"n": 99,  "win": 60.6},

        "runic": {"n": 317, "win": 59.6},      # PP_iXA60 >= 3.5
        "runic_2": {"n": 210, "win": 63.3},    # PP_iXA60 >= 4.0

        "silent_1": {"n": 275, "win": 61.1},   # mu >= 1.15
        "silent_2": {"n": 178, "win": 64.0},   # mu >= 1.30
        "silent_3": {"n": 49,  "win": 69.4},   # mu >= 1.62

        "valhalla": {"n": 140, "win": 60.7},   # Conf >= 90 label (inside base shell)
        "odins_blessing": {"n": 40, "win": 77.5},  # Goalie_Weak >= 90 (ENV)

        "arcane_alignment": {"n": 62, "win": 72.6},    # mu + heater offense
        "rune_orchestration": {"n": 72, "win": 72.2},  # PP + mu stack (locked heuristic)

        "magic": {"n": 130, "win": 63.8},              # iXA% >= 99
        "supernova_overdrive": {"n": 64, "win": 75.0}, # convergence

        "arcane_transcendence": {"n": 45, "win": 77.8},  # Conf≥88 + PPP10≥3 + PP_TOI%≥17
        "arcane_supernova": {"n": 25, "win": 84.0},       # Arcane Transcendence + Team_GF_L5≥20 (≈ GF_Avg_L5≥3.9)

        "stars_aligned_a": {"n": 163, "win": 65.6},       # Conf≥88 + iXA%≥96
        "stars_aligned_b": {"n": 118, "win": 65.3},       # Conf≥90 + iXA%≥95
    }

    # ---- proc logic (signals only; NOT gates) ----
    staff_on = stance_ok

    odin_on = staff_on and (not math.isnan(ixa_pct)) and (ixa_pct >= 95.0)

    # Arcane Channel ladder (PPP10_total) — highest tier only
    arcane_key = None
    if staff_on and (not math.isnan(ppp10)):
        if ppp10 >= 6:
            arcane_key = "arcane_6"
        elif ppp10 >= 5:
            arcane_key = "arcane_5"
        elif ppp10 >= 4:
            arcane_key = "arcane_4"
        elif ppp10 >= 3:
            arcane_key = "arcane_3"

    # Runic ladder (PP_iXA60) — highest tier only
    runic_key = None
    if staff_on and (not math.isnan(pp_ix)):
        if pp_ix >= 4.0:
            runic_key = "runic_2"
        elif pp_ix >= 3.5:
            runic_key = "runic"

    # Silent ladder (mu) — highest tier only
    silent_key = None
    if staff_on and (not math.isnan(assists_mu)):
        if assists_mu >= 1.62:
            silent_key = "silent_3"
        elif assists_mu >= 1.30:
            silent_key = "silent_2"
        elif assists_mu >= 1.15:
            silent_key = "silent_1"

    valhalla_on = staff_on and (conf >= 90)
    odins_blessing_on = staff_on and (not math.isnan(goalie_weak)) and (goalie_weak >= 90)

    arcane_alignment_on = (
        staff_on
        and (not math.isnan(assists_mu)) and (assists_mu >= 1.30)
        and (not math.isnan(team_gf_l5)) and (team_gf_l5 >= 20)
    )

    rune_orchestration_on = (
        staff_on
        and (not math.isnan(pp_ix)) and (pp_ix >= 4.0)
        and (not math.isnan(assists_mu)) and (assists_mu >= 1.30)
    )

    magic_on = staff_on and (not math.isnan(ixa_pct)) and (ixa_pct >= 99.0)

    # Stars Aligned — adaptive tiers (show highest)
    stars_tier = None
    stars_key = None
    if staff_on and (not math.isnan(ixa_pct)):
        if (conf >= 90) and (ixa_pct >= 95):
            stars_tier = "Tier B: Conf ≥ 90 + iXA% ≥ 95"
            stars_key = "stars_aligned_b"
        elif (conf >= 88) and (ixa_pct >= 96):
            stars_tier = "Tier A: Conf ≥ 88 + iXA% ≥ 96"
            stars_key = "stars_aligned_a"

    arcane_transcendence = (
        staff_on
        and (conf >= 88)
        and (not math.isnan(ppp10)) and (ppp10 >= 3)
        and (not math.isnan(pp_toi_pct)) and (pp_toi_pct >= 17)
    )

    arcane_supernova = (
        arcane_transcendence
        and (not math.isnan(team_gf_l5)) and (team_gf_l5 >= 20)  # ≈ GF_Avg_L5 ≥ 3.9
    )

    supernova_overdrive = (
        staff_on
        and (conf >= 80)
        and (not math.isnan(ixa_pct)) and (ixa_pct >= 95.0)
        and (not math.isnan(pp_ix)) and (pp_ix >= 3.7)
        and (not math.isnan(team_gf_l5)) and (team_gf_l5 >= 20)
    )

    # ---- render ----
    st.markdown("**STANCE**")
    if staff_on:
        st.markdown(
            f"{_svg_icon('valhalla.svg','Valhalla Gate',market_cls)} Gate: Green + 0.5 + Conf ≥ 80 (EV ignored)",
            unsafe_allow_html=True,
        )
    else:
        st.write("Not inside Gate stance (check Matrix/Line/Conf).")

    st.markdown("**MOVES**")

    def _goals_best_tier_tag(b: dict) -> str:

        """Return highest GOALS tier tag (beta), conf-free."""

        opp = _num(b.get("oppsog", b.get("oppsog_l10", b.get("Opp_SOG_Against_L10", 0))), 0)

        xga = _num(b.get("xga", b.get("opp_5v5_xGA60", b.get("opp_xga60", 0))), 0)

        ixg = b.get("ixg", b.get("iXG%", b.get("iXG_Pct", b.get("ixg_pct", 0))))

        ixg = _num(ixg, 0)

        gf = b.get("team_gf_avg_l5", b.get("Team_GF_Avg_L5", 0))

        gf = _num(gf, 0)


        if (opp >= 29) and (gf >= 3.9) and (xga >= 2.52) and (ixg >= 97):

            return "Ragnarök"

        if (opp >= 29) and (gf >= 3.7) and (xga >= 2.55):

            return "Valhalla"

        if (opp >= 29) and (gf >= 3.3) and (xga >= 2.49) and (ixg >= 94):

            return "Smash"

        if (ixg >= 97) and (xga >= 2.49):

            return "Armor Core"

        if (ixg >= 94) and (xga >= 2.55):

            return "Fury Core"

        if (opp >= 29) and (gf >= 2.5):

            return "Funnel Carry"

        return ""




    tier_tag = _goals_best_tier_tag(r)
    if tier_tag:
        st.markdown(f"- **Top Tier:** {tier_tag}")

    moves_rendered = 0

    # Rank label tracking (presentation only)
    _best_title = ""
    _best_win = None
    _best_n = 0
    _best_aw = -1.0
    def _track_best(label: str, win: float, n: int) -> None:
        nonlocal _best_title, _best_win, _best_n, _best_aw
        aw = _adj_win(win, n, k=20)
        nn = int(n) if n is not None else 0
        if (aw > _best_aw) or (abs(aw - _best_aw) < 1e-9 and nn > _best_n):
            _best_aw = aw
            _best_title = label
            _best_win = float(win)
            _best_n = nn



    def _move_line(icon_file: str, label: str, dps_key: str = None, extra: str = ""):
        nonlocal moves_rendered
        ico = _svg_icon(icon_file, label, market_cls)
        tail = f" — {extra}" if extra else ""
        st.markdown(f"{ico} <b>{label}</b>{tail}", unsafe_allow_html=True)
        if dps_key and (dps_key in DPS):
            _track_best(label, DPS[dps_key]["win"], DPS[dps_key]["n"])
            st.caption(f"n={DPS[dps_key]['n']} • Win%={DPS[dps_key]['win']:.1f}")
            _wl_dps_bar(DPS[dps_key]["win"], mk)
        moves_rendered += 1

    # Always show Staff inside Gate so the section never looks empty.
    if staff_on:
        _move_line("staff.svg", "Staff (Base Shell)", "staff", "Matrix Green + Line 0.5 + Conf ≥ 80")

    if odin_on:
        _move_line("odins_arcane_orb.svg", "Odin (Role Engine)", "odin", "iXA% ≥ 95")

    if arcane_key:
        tier_map = {
            "arcane_3": "Arcane Channel I (PPP10 ≥ 3)",
            "arcane_4": "Arcane Channel II (PPP10 ≥ 4)",
            "arcane_5": "Arcane Channel III (PPP10 ≥ 5)",
            "arcane_6": "Arcane Channel CRIT (PPP10 ≥ 6)",
        }
        _move_line("arcane_channel_iii.svg" if arcane_key in {"arcane_5","arcane_6"} else "arcane_channel_ii.svg",
                   tier_map.get(arcane_key, "Arcane Channel"), arcane_key)

    if runic_key:
        tier_map = {"runic": "Runic Infusion (PP_iXA60 ≥ 3.5)", "runic_2": "Runic Infusion II (PP_iXA60 ≥ 4.0)"}
        _move_line("runic_infusion.svg", tier_map.get(runic_key, "Runic Infusion"), runic_key)

    if silent_key:
        tier_map = {
            "silent_1":"Silent Distributor I (μ ≥ 1.15)",
            "silent_2":"Silent Distributor II (μ ≥ 1.30)",
            "silent_3":"Silent Distributor CRIT (μ ≥ 1.62)",
        }
        _move_line("silent_distributor.svg", tier_map.get(silent_key, "Silent Distributor"), silent_key)

    if valhalla_on:
        _move_line("valhalla.svg", "Valhalla (Confidence Spike)", "valhalla", "Conf ≥ 90")

    if odins_blessing_on:
        _move_line("odins_eye.svg", "Odin’s Blessing (Goalie Weak)", "odins_blessing", "Goalie_Weak ≥ 90")

    if arcane_alignment_on:
        _move_line("odins_symbol.svg", "Arcane Alignment", "arcane_alignment", "μ ≥ 1.30 + Team_GF_L5 ≥ 20")

    if rune_orchestration_on:
        _move_line("runic_infusion.svg", "Rune Orchestration", "rune_orchestration", "PP_iXA60 ≥ 4.0 + μ ≥ 1.30")

    if magic_on:
        _move_line("magic_mans_transcendence.svg", "Magic Man (Elite Creator Core)", "magic", "iXA% ≥ 99")

    if stars_tier:
        _move_line("stars.svg", "Stars Aligned", stars_key, stars_tier)

    if arcane_transcendence:
        _move_line("magic_mans_transcendence.svg", "Arcane Transcendence", "arcane_transcendence",
                   "Conf ≥ 88 + PPP10 ≥ 3 + PP_TOI% ≥ 17")

    if arcane_supernova:
        _move_line("supernova.svg", "Arcane Supernova", "arcane_supernova",
                   "Arcane Transcendence + Team_GF_L5 ≥ 20 (≈ GF_Avg_L5 ≥ 3.9)")

    if supernova_overdrive:
        _move_line("supernova.svg", "Supernova Overdrive", "supernova_overdrive",
                   "Conf ≥ 80 + iXA% ≥ 95 + PP_iXA60 ≥ 3.7 + Team_GF_L5 ≥ 20")

    
    # Rank label (best active proc)
    if _best_title and _best_win is not None:
        _render_rank_line(_best_title, _best_win, _best_n, mk)

    if moves_rendered == 0:
        st.caption("No procs fired (inside gate).")


def _render_goals_combat_hud(r) -> None:
    """GOALS Combat HUD (full moves + DPS bars).

    Presentation-only; does not change gating/EV.
    """
    import math

    mk = "GOALS"
    # GOALS combat HUD (Beta). Clean: show only the highest tier per lane.
    # Stance (locked): Line=0.5, Matrix=Green, Conf>=85 (EV ignored)
    line = _safe_float(r.get("Goal_Line", None), 0.0) or 0.0
    mat = str(r.get("Matrix_Goal", "") or "").strip().lower()
    conf = _safe_float(r.get("Conf_Goal", None), None)

    stance_ok = bool(line == 0.5 and mat.startswith("g") and (conf is not None and conf >= 85))

    # Core inputs (new GOALS lanes)
    xga   = _safe_float(r.get("opp_5v5_xGA60", None), None)
    oppsog = _safe_float(r.get("Opp_SOG_Against_L10", None), None)
    # iXG% can come under different column spellings
    ixg = None
    for k in ("iXG%", "iXG_pct", "iXG_Pct", "ixg_pct", "ixg%"):
        if k in r:
            ixg = _safe_float(r.get(k, None), None)
            if ixg is not None:
                break
    share = None
    for k in ("Player_5v5_SOG_Share", "Player_5v5_SOG_Share_Pct", "Player_5v5_SOGShare"):
        if k in r:
            share = _safe_float(r.get(k, None), None)
            if share is not None:
                break
    drought_g = _safe_float(r.get("Drought_G", None), None)

    # --- DPS anchors (final from this chat) ---
    DPS = {
        "base": {"n": 423, "win": 34.3},
        "armor_shred": {"n": 189, "win": 41.8},   # xGA >= 2.49
        "hot_team_press": {"n": 81, "win": 51.9},  # xGA>=2.49 & iXG>=96.5 & Team_GF>=3.0

        # Env+Finisher (mid-iXG) lane — board-eligible only when Goal_Odds >= +150
        "envmix_50": {"n": 64, "win": 50.0},       # xGA>=2.55 & iXG>=92 & Team_GF>=3.0
        "envmix_elite": {"n": 41, "win": 58.5},    # xGA>=2.55 & iXG>=97 & Team_GF>=3.8

        "armor_buff":  {"n": 234, "win": 28.2},   # xGA < 2.49 (derived complement)

        "fenrir_34": {"n": 200, "win": 40.0},     # iXG% >= 97
        "fenrir_36": {"n": 28,  "win": 60.7},     # iXG% >= 99 & xGA >= 2.55

        "fury_35": {"n": 113, "win": 47.8},       # OppSOG_L10 >= 29
        "fury_37": {"n": 76,  "win": 52.6},       # + xGA >= 2.49
        "fury_38": {"n": 57,  "win": 59.6},       # + iXG% >= 93.5
        "fury_40": {"n": 52,  "win": 63.5},       # + iXG% >= 94

        "tyrs_wrath_unleashed": {"n": 25, "win": 72.0},  # OppSOG>=29 & Share>=15 & xGA>=2.52 & iXG>=94

        "armor_annihilation": {"n": 66, "win": 54.5},    # iXG%>=97 & xGA>=2.52
        "smash":             {"n": 52, "win": 57.7},     # armor_annihilation + Conf>=91
        "valhalla":          {"n": 44, "win": 61.4},     # armor_annihilation + Conf>=95

        "fury_shredder": {"n": 15, "win": 73.3},         # xGA>=2.52 & iXG>=94 & OppSOG>=29 & Drought_G>=2
    }

    
    # Rank label tracking (presentation only) — excludes BASE
    _best_title = ""
    _best_win = None
    _best_n = 0
    _best_aw = -1.0
    _key_titles = {
        "tyrs_wrath_unleashed": "Tyr’s Wrath Unleashed",
        "fenrir_36": "Fenrir’s Fury (3.6)",
        "fenrir_34": "Fenrir’s Fury (3.4)",
        "valhalla": "FOR VALHALLA",
        "smash": "SMASH",
        "armor_annihilation": "Armor Annihilation",
        "fury_shredder": "Fury Shredder",
        "hot_team_press": "Press the Attack",
        "envmix_50": "Berserker Aggression (Env Mix)",
        "envmix_elite": "Berserker Aggression (Env Mix • ELITE)",
    }
    def _track_best_key(key: str) -> None:
        nonlocal _best_title, _best_win, _best_n, _best_aw
        if not key or key == "base" or key not in DPS:
            return
        win = DPS[key]["win"]
        n = DPS[key]["n"]
        title = _key_titles.get(key, key)
        aw = _adj_win(win, n, k=20)
        nn = int(n) if n is not None else 0
        if (aw > _best_aw) or (abs(aw - _best_aw) < 1e-9 and nn > _best_n):
            _best_aw = aw
            _best_title = title
            _best_win = float(win)
            _best_n = nn
    base_win = DPS["base"]["win"]

    st.markdown("**Combat HUD (GOALS):**")

    # 1) Stance
    if stance_ok:
        _wl_why_line(
            _svg_icon("base.svg", "Base Attack (Stance)", "wl-goals"),
            f"Base Attack active — Conf≥85 / Green / 0.5  •  DPS {DPS['base']['win']}% (n={DPS['base']['n']})",
        )
        _wl_dps_bar(DPS["base"]["win"], "GOALS")
    else:
        _wl_why_line(
            _svg_icon("base.svg", "Base Attack (Stance)", "wl-goals"),
            "Base Attack NOT active — needs Conf≥85 / Green / 0.5",
        )

    # 2) Enemy armor state (ENV) — show highest tier only
    env_label = None
    env_key = None
    env_icon = None
    if xga is not None:
        if xga >= 2.52:
            env_label = f"Armor Shred (Defense Collapsing) — opp xGA {xga:.2f} ≥ 2.52"
            env_key = "armor_shred"  # DPS anchor is xGA>=2.49; 2.52 is a gate, not separate DPS bar
            env_icon = "armor_shred.svg"
        elif xga >= 2.49:
            env_label = f"Armor Shred — opp xGA {xga:.2f} ≥ 2.49"
            env_key = "armor_shred"
            env_icon = "armor_shred.svg"
        else:
            env_label = f"Enemy Fortified — opp xGA {xga:.2f} < 2.49"
            env_key = "armor_buff"
            env_icon = "armor_buff.svg"

        _wl_why_line(
            _svg_icon(env_icon, "Enemy Armor (ENV)", "wl-goals wl-keep"),
            f"{env_label}  •  DPS {DPS[env_key]['win']}% (n={DPS[env_key]['n']})  (Δ {DPS[env_key]['win']-base_win:+.1f})",
        )
        _wl_dps_bar(DPS[env_key]["win"], "GOALS")
    else:
        _wl_why_line(
            _svg_icon("armor_buff.svg", "Enemy Armor Unknown", "wl-goals"),
            "Enemy armor state unknown — opp xGA missing",
        )


    # Team offense (L5) — used for Press the Attack
    team_gf = None
    for k in ("Team_GF_Avg_L5", "Team_GF_L5_Avg", "Team_GF_L5", "Team_GoalsFor_Avg_L5"):
        if k in r:
            team_gf = _safe_float(r.get(k, None), None)
            if team_gf is not None:
                break
    # Press the Attack (NEW): xGA>=2.49 & iXG>=96.5 & Team_GF_Avg_L5>=3.0
    hot_press_on = bool((xga is not None and xga >= 2.49) and (ixg is not None and ixg >= 96.5) and (team_gf is not None and team_gf >= 3.0))
    if hot_press_on:
        _wl_why_line(
            _svg_icon("smash.svg", "Press the Attack", "wl-goals wl-keep"),
            f"Press the Attack — xGA {xga:.2f} ≥ 2.49 • iXG {ixg:.1f} ≥ 96.5 • Team GF {team_gf:.1f} ≥ 3.0  •  DPS {DPS['hot_team_press']['win']}% (n={DPS['hot_team_press']['n']})  (Δ {DPS['hot_team_press']['win']-base_win:+.1f})",
        )
        _wl_dps_bar(DPS["hot_team_press"]["win"], "GOALS")
        _track_best_key("hot_team_press")


    # Berserker Aggression (Env Mix) — xGA>=2.55 & iXG>=92 & Team_GF>=3.0 (board requires Goal_Odds >= +150)
    goal_odds = _safe_float(r.get("Goal_Odds_Over", r.get("Goal_Odds", None)), None)
    envmix_on = bool((xga is not None and xga >= 2.55) and (ixg is not None and ixg >= 92) and (team_gf is not None and team_gf >= 3.0))
    envmix_odds_ok = bool(goal_odds is not None and goal_odds >= 150)
    envmix_board_ok = bool(envmix_on and envmix_odds_ok)
    if envmix_on:
        odds_txt = f" • Odds +{int(goal_odds)}" if goal_odds is not None else " • Odds n/a"
        extra = "" if envmix_odds_ok else " (board needs ≥ +150)"
        _wl_why_line(
            _svg_icon("fury.svg", "Berserker Aggression (Env Mix)", "wl-goals wl-keep"),
            f"Berserker Aggression (Env Mix) — xGA {xga:.2f} ≥ 2.55 • iXG {ixg:.1f} ≥ 92 • Team GF {team_gf:.1f} ≥ 3.0{odds_txt}{extra}  •  DPS {DPS['envmix_50']['win']}% (n={DPS['envmix_50']['n']})  (Δ {DPS['envmix_50']['win']-base_win:+.1f})",
        )
        _wl_dps_bar(DPS["envmix_50"]["win"], "GOALS")
        if envmix_board_ok:
            _track_best_key("envmix_50")

    # Berserker Aggression (Env Mix • ELITE) — xGA>=2.55 & iXG>=92 & Team_GF>=3.8 (no odds requirement)
    envmix_elite_on = bool((xga is not None and xga >= 2.55) and (ixg is not None and ixg >= 92) and (team_gf is not None and team_gf >= 3.8))
    if envmix_elite_on:
        _wl_why_line(
            _svg_icon("smash.svg", "Berserker Aggression (Env Mix • ELITE)", "wl-goals wl-keep"),
            f"Berserker Aggression (Env Mix • ELITE) — xGA {xga:.2f} ≥ 2.55 • iXG {ixg:.1f} ≥ 92 • Team GF {team_gf:.1f} ≥ 3.8  •  DPS {DPS['envmix_elite']['win']}% (n={DPS['envmix_elite']['n']})  (Δ {DPS['envmix_elite']['win']-base_win:+.1f})",
        )
        _wl_dps_bar(DPS["envmix_elite"]["win"], "GOALS")
        _track_best_key("envmix_elite")

    # Lane flags
    opp_lane = bool(oppsog is not None and oppsog >= 29)
    env_249  = bool(xga is not None and xga >= 2.49)
    env_252  = bool(xga is not None and xga >= 2.52)

    # Tyr’s Wrath Unleashed (signature) — if active, it replaces Fury lines for cleanliness
    tyr_on = bool(opp_lane and env_252 and (share is not None and share >= 15) and (ixg is not None and ixg >= 97))

    if tyr_on:
        _wl_why_line(
            _svg_icon("fury.svg", "Tyr’s Wrath Unleashed", "wl-goals"),
            f"Tyr’s Wrath Unleashed — OppSOG≥29 + Share≥15 + xGA≥2.52 + iXG≥97  •  DPS {DPS['tyrs_wrath_unleashed']['win']}% (n={DPS['tyrs_wrath_unleashed']['n']})  (Δ {DPS['tyrs_wrath_unleashed']['win']-base_win:+.1f})",
        )
        _track_best_key("tyrs_wrath_unleashed")
        _wl_dps_bar(DPS["tyrs_wrath_unleashed"]["win"], "GOALS")

    # 3) Fury lane (Opp shot funnel) — show highest tier only (unless Tyr is active)
    if (not tyr_on) and opp_lane:
        fury_key = "fury_35"
        fury_lbl = f"Warlord Fury — OppSOG_L10 {oppsog:.0f} ≥ 29"
        if env_249:
            fury_key = "fury_37"
            fury_lbl = f"Warlord Fury (Charged) — OppSOG_L10 {oppsog:.0f} ≥ 29 + xGA≥2.49"
            if ixg is not None and ixg >= 94:
                fury_key = "fury_40"
                fury_lbl = f"Warlord Fury (Potent) — + iXG {ixg:.1f} ≥ 94"
            elif ixg is not None and ixg >= 93.5:
                fury_key = "fury_38"
                fury_lbl = f"Warlord Fury (Surging) — + iXG {ixg:.1f} ≥ 93.5"

        _wl_why_line(
            _svg_icon("fury.svg", "Warlord Fury", "wl-goals"),
            f"{fury_lbl}  •  DPS {DPS[fury_key]['win']}% (n={DPS[fury_key]['n']})  (Δ {DPS[fury_key]['win']-base_win:+.1f})",
        )
        _wl_dps_bar(DPS[fury_key]["win"], "GOALS")

    # 4) Fenrir lane (Finisher identity) — show highest tier only
    fenrir_on = bool(ixg is not None and ixg >= 97)
    fenrir_potent = bool(ixg is not None and ixg >= 99 and (xga is not None and xga >= 2.55))
    if fenrir_on:
        if fenrir_potent:
            _wl_why_line(
                _svg_icon("fenrir_claw.svg", "Fenrir’s Claw (Potent)", "wl-goals"),
                f"Fenrir’s Claw (Potent) — iXG {ixg:.1f} ≥ 99 & xGA≥2.55  •  DPS {DPS['fenrir_36']['win']}% (n={DPS['fenrir_36']['n']})  (Δ {DPS['fenrir_36']['win']-base_win:+.1f})",
            )
            _track_best_key("fenrir_36")
            _wl_dps_bar(DPS["fenrir_36"]["win"], "GOALS")
        else:
            _wl_why_line(
                _svg_icon("fenrir_claw.svg", "Fenrir’s Claw", "wl-goals"),
                f"Fenrir’s Claw — iXG {ixg:.1f} ≥ 92  •  DPS {DPS['fenrir_34']['win']}% (n={DPS['fenrir_34']['n']})  (Δ {DPS['fenrir_34']['win']-base_win:+.1f})",
            )
            _track_best_key("fenrir_34")
            _wl_dps_bar(DPS["fenrir_34"]["win"], "GOALS")

    # 5) Premium tiers (Special / Ultimate) — show highest only
    armor_annihilation = bool((ixg is not None and ixg >= 97) and env_252)
    smash = bool(armor_annihilation and (conf is not None and conf >= 91))
    valhalla = bool(armor_annihilation and (conf is not None and conf >= 95))

    if valhalla:
        _wl_why_line(
            _svg_icon("valhalla.svg", "FOR VALHALLA! (Ultimate)", "wl-goals"),
            f"FOR VALHALLA! — Armor Annihilation + Conf≥95  •  DPS {DPS['valhalla']['win']}% (n={DPS['valhalla']['n']})",
        )
        _track_best_key("valhalla")
        _wl_dps_bar(DPS["valhalla"]["win"], "GOALS")
    elif smash:
        _wl_why_line(
            _svg_icon("smash.svg", "Warlord Smash Attack (Special)", "wl-goals"),
            f"Warlord Smash Attack — Armor Annihilation + Conf≥91  •  DPS {DPS['smash']['win']}% (n={DPS['smash']['n']})  (Δ {DPS['smash']['win']-base_win:+.1f})",
        )
        _track_best_key("smash")
        _wl_dps_bar(DPS["smash"]["win"], "GOALS")

    # -------------------------
    # STACK PROCS (clean): show only the procs that are truly active
    # -------------------------
    fury_shredder = bool(env_252 and opp_lane and (ixg is not None and ixg >= 94) and (drought_g is not None and drought_g >= 2))

    if armor_annihilation or fury_shredder:
        st.markdown(
            "<div style='margin-top:6px;font-size:13px;font-weight:900;opacity:0.85;'>⚡ STACK PROCS</div>",
            unsafe_allow_html=True,
        )

    if armor_annihilation:
        _wl_why_line(
            _svg_icon("stack_armor_annihilation.svg", "Armor Annihilation", "wl-goals"),
            f"Armor Annihilation — iXG≥97 + xGA≥2.52  •  DPS {DPS['armor_annihilation']['win']}% (n={DPS['armor_annihilation']['n']})  (Δ {DPS['armor_annihilation']['win']-base_win:+.1f})",
        )
        _track_best_key("armor_annihilation")
        _wl_dps_bar(DPS["armor_annihilation"]["win"], "GOALS")

    if fury_shredder:
        _wl_why_line(
            _svg_icon("stack_fury_shredder.svg", "Fury Shredder", "wl-goals"),
            f"Fury Shredder — Funnel core + Drought_G≥2  •  DPS {DPS['fury_shredder']['win']}% (n={DPS['fury_shredder']['n']})  (Δ {DPS['fury_shredder']['win']-base_win:+.1f})",
        )
        _track_best_key("fury_shredder")
        _wl_dps_bar(DPS["fury_shredder"]["win"], "GOALS")


    # Rank label (best active proc, BASE excluded)
    if _best_title and _best_win is not None:
        _render_rank_line(_best_title, _best_win, _best_n, "GOALS")


def _render_why_it_fires_rich(mkt: str, r, tags: str = "") -> None:
    """Presentation-only rich WHY block. Does not change any gates/logic."""

    mk = str(mkt or "").strip().upper()

    # Role line (replaces legacy Tags/Badges)
    if mk == "POINTS":
        _pl = _safe_float(r.get("Points_Line"))
        _pl = 0.0 if _pl is None else float(_pl)
        _role = "Fortress Tank" if _pl <= 0.75 else "DPS Tank"
        _ico = _role_icon_html("POINTS")
        st.markdown(
            f"**Role:** <span style='display:inline-flex;align-items:center;'>{_ico}<b>{_role}</b></span>",
            unsafe_allow_html=True
        )
    else:
        _info = _role_for_market(mkt)
        _ico = _role_icon_html(mkt)
        st.markdown(
            f"**Role:** <span style='display:inline-flex;align-items:center;'>{_ico}<b>{_info.get('role','')}</b></span>",
            unsafe_allow_html=True
        )

    # SOG: show Role then Combat HUD (SOG) and stop (SOG HUD prints its own block)
    if mk == "SOG":
        _render_sog_combat_hud(r)
        return

    if mk == "POINTS":
        _render_points_combat_hud(r)
        return

    if mk == "ASSISTS":
        _render_assists_combat_hud(r)
        return

    if mk == "GOALS":
        _render_goals_combat_hud(r)
        return

    # Legacy tags/badges hidden (keep var defined to avoid NameError)
    tags_s = ""

# --- Engine checklist (market-pure; presentation only) ---
    mk = str(mkt or "").strip().upper()

    
    
    if mk == "GOALS":
        # GOALS combat HUD (Beta). Clean: show only the highest tier per lane.
        # Stance (locked): Line=0.5, Matrix=Green, Conf>=85 (EV ignored)
        line = _safe_float(r.get("Goal_Line", None), 0.0) or 0.0
        mat = str(r.get("Matrix_Goal", "") or "").strip().lower()
        conf = _safe_float(r.get("Conf_Goal", None), None)

        stance_ok = bool(line == 0.5 and mat.startswith("g") and (conf is not None and conf >= 85))

        # Core inputs (new GOALS lanes)
        xga   = _safe_float(r.get("opp_5v5_xGA60", None), None)
        oppsog = _safe_float(r.get("Opp_SOG_Against_L10", None), None)
        # iXG% can come under different column spellings
        ixg = None
        for k in ("iXG%", "iXG_pct", "iXG_Pct", "ixg_pct", "ixg%"):
            if k in r:
                ixg = _safe_float(r.get(k, None), None)
                if ixg is not None:
                    break
        share = None
        for k in ("Player_5v5_SOG_Share", "Player_5v5_SOG_Share_Pct", "Player_5v5_SOGShare"):
            if k in r:
                share = _safe_float(r.get(k, None), None)
                if share is not None:
                    break
        drought_g = _safe_float(r.get("Drought_G", None), None)

        # --- DPS anchors (final from this chat) ---
        DPS = {
            "base": {"n": 423, "win": 34.3},
            "armor_shred": {"n": 189, "win": 41.8},   # xGA >= 2.49
            "armor_buff":  {"n": 234, "win": 28.2},   # xGA < 2.49 (derived complement)

            "fenrir_34": {"n": 200, "win": 40.0},     # iXG% >= 97
            "fenrir_36": {"n": 28,  "win": 60.7},     # iXG% >= 99 & xGA >= 2.55

            "fury_35": {"n": 113, "win": 47.8},       # OppSOG_L10 >= 29
            "fury_37": {"n": 76,  "win": 52.6},       # + xGA >= 2.49
            "fury_38": {"n": 57,  "win": 59.6},       # + iXG% >= 93.5
            "fury_40": {"n": 52,  "win": 63.5},       # + iXG% >= 94

            "tyrs_wrath_unleashed": {"n": 25, "win": 72.0},  # OppSOG>=29 & Share>=15 & xGA>=2.52 & iXG>=94

            "armor_annihilation": {"n": 66, "win": 54.5},    # iXG%>=97 & xGA>=2.52
            "smash":             {"n": 52, "win": 57.7},     # armor_annihilation + Conf>=91
            "valhalla":          {"n": 44, "win": 61.4},     # armor_annihilation + Conf>=95

            "fury_shredder": {"n": 15, "win": 73.3},         # xGA>=2.52 & iXG>=94 & OppSOG>=29 & Drought_G>=2
        }

        base_win = DPS["base"]["win"]

        st.markdown("**Combat HUD (GOALS):**")

        # 1) Stance
        if stance_ok:
            _wl_why_line(
                _svg_icon("base.svg", "Base Attack (Stance)", "wl-goals"),
                f"Base Attack active — Conf≥85 / Green / 0.5  •  DPS {DPS['base']['win']}% (n={DPS['base']['n']})",
            )
            _wl_dps_bar(DPS["base"]["win"], "GOALS")
        else:
            _wl_why_line(
                _svg_icon("base.svg", "Base Attack (Stance)", "wl-goals"),
                "Base Attack NOT active — needs Conf≥85 / Green / 0.5",
            )

        # 2) Enemy armor state (ENV) — show highest tier only
        env_label = None
        env_key = None
        env_icon = None
        if xga is not None:
            if xga >= 2.52:
                env_label = f"Armor Shred (Defense Collapsing) — opp xGA {xga:.2f} ≥ 2.52"
                env_key = "armor_shred"  # DPS anchor is xGA>=2.49; 2.52 is a gate, not separate DPS bar
                env_icon = "armor_shred.svg"
            elif xga >= 2.49:
                env_label = f"Armor Shred — opp xGA {xga:.2f} ≥ 2.49"
                env_key = "armor_shred"
                env_icon = "armor_shred.svg"
            else:
                env_label = f"Enemy Fortified — opp xGA {xga:.2f} < 2.49"
                env_key = "armor_buff"
                env_icon = "armor_buff.svg"

            _wl_why_line(
                _svg_icon(env_icon, "Enemy Armor (ENV)", "wl-goals wl-keep"),
                f"{env_label}  •  DPS {DPS[env_key]['win']}% (n={DPS[env_key]['n']})  (Δ {DPS[env_key]['win']-base_win:+.1f})",
            )
            _wl_dps_bar(DPS[env_key]["win"], "GOALS")
        else:
            _wl_why_line(
                _svg_icon("armor_buff.svg", "Enemy Armor Unknown", "wl-goals"),
                "Enemy armor state unknown — opp xGA missing",
            )

        # Lane flags
        opp_lane = bool(oppsog is not None and oppsog >= 29)
        env_249  = bool(xga is not None and xga >= 2.49)
        env_252  = bool(xga is not None and xga >= 2.52)

        # Tyr’s Wrath Unleashed (signature) — if active, it replaces Fury lines for cleanliness
        tyr_on = bool(opp_lane and env_252 and (share is not None and share >= 15) and (ixg is not None and ixg >= 97))

        if tyr_on:
            _wl_why_line(
                _svg_icon("fury.svg", "Tyr’s Wrath Unleashed", "wl-goals"),
                f"Tyr’s Wrath Unleashed — OppSOG≥29 + Share≥15 + xGA≥2.52 + iXG≥97  •  DPS {DPS['tyrs_wrath_unleashed']['win']}% (n={DPS['tyrs_wrath_unleashed']['n']})  (Δ {DPS['tyrs_wrath_unleashed']['win']-base_win:+.1f})",
            )
            _wl_dps_bar(DPS["tyrs_wrath_unleashed"]["win"], "GOALS")

        # 3) Fury lane (Opp shot funnel) — show highest tier only (unless Tyr is active)
        if (not tyr_on) and opp_lane:
            fury_key = "fury_35"
            fury_lbl = f"Warlord Fury — OppSOG_L10 {oppsog:.0f} ≥ 29"
            if env_249:
                fury_key = "fury_37"
                fury_lbl = f"Warlord Fury (Charged) — OppSOG_L10 {oppsog:.0f} ≥ 29 + xGA≥2.49"
                if ixg is not None and ixg >= 94:
                    fury_key = "fury_40"
                    fury_lbl = f"Warlord Fury (Potent) — + iXG {ixg:.1f} ≥ 94"
                elif ixg is not None and ixg >= 93.5:
                    fury_key = "fury_38"
                    fury_lbl = f"Warlord Fury (Surging) — + iXG {ixg:.1f} ≥ 93.5"

            _wl_why_line(
                _svg_icon("fury.svg", "Warlord Fury", "wl-goals"),
                f"{fury_lbl}  •  DPS {DPS[fury_key]['win']}% (n={DPS[fury_key]['n']})  (Δ {DPS[fury_key]['win']-base_win:+.1f})",
            )
            _wl_dps_bar(DPS[fury_key]["win"], "GOALS")

        # 4) Fenrir lane (Finisher identity) — show highest tier only
        fenrir_on = bool(ixg is not None and ixg >= 97)
        fenrir_potent = bool(ixg is not None and ixg >= 99 and (xga is not None and xga >= 2.55))
        if fenrir_on:
            if fenrir_potent:
                _wl_why_line(
                    _svg_icon("fenrir_claw.svg", "Fenrir’s Claw (Potent)", "wl-goals"),
                    f"Fenrir’s Claw (Potent) — iXG {ixg:.1f} ≥ 99 & xGA≥2.55  •  DPS {DPS['fenrir_36']['win']}% (n={DPS['fenrir_36']['n']})  (Δ {DPS['fenrir_36']['win']-base_win:+.1f})",
                )
                _wl_dps_bar(DPS["fenrir_36"]["win"], "GOALS")
            else:
                _wl_why_line(
                    _svg_icon("fenrir_claw.svg", "Fenrir’s Claw", "wl-goals"),
                    f"Fenrir’s Claw — iXG {ixg:.1f} ≥ 92  •  DPS {DPS['fenrir_34']['win']}% (n={DPS['fenrir_34']['n']})  (Δ {DPS['fenrir_34']['win']-base_win:+.1f})",
                )
                _wl_dps_bar(DPS["fenrir_34"]["win"], "GOALS")

        # 5) Premium tiers (Special / Ultimate) — show highest only
        armor_annihilation = bool((ixg is not None and ixg >= 97) and env_252)
        smash = bool(armor_annihilation and (conf is not None and conf >= 91))
        valhalla = bool(armor_annihilation and (conf is not None and conf >= 95))

        if valhalla:
            _wl_why_line(
                _svg_icon("valhalla.svg", "FOR VALHALLA! (Ultimate)", "wl-goals"),
                f"FOR VALHALLA! — Armor Annihilation + Conf≥95  •  DPS {DPS['valhalla']['win']}% (n={DPS['valhalla']['n']})",
            )
            _wl_dps_bar(DPS["valhalla"]["win"], "GOALS")
        elif smash:
            _wl_why_line(
                _svg_icon("smash.svg", "Warlord Smash Attack (Special)", "wl-goals"),
                f"Warlord Smash Attack — Armor Annihilation + Conf≥91  •  DPS {DPS['smash']['win']}% (n={DPS['smash']['n']})  (Δ {DPS['smash']['win']-base_win:+.1f})",
            )
            _wl_dps_bar(DPS["smash"]["win"], "GOALS")

        # -------------------------
        # STACK PROCS (clean): show only the procs that are truly active
        # -------------------------
        fury_shredder = bool(env_252 and opp_lane and (ixg is not None and ixg >= 94) and (drought_g is not None and drought_g >= 2))

        if armor_annihilation or fury_shredder:
            st.markdown(
                "<div style='margin-top:6px;font-size:13px;font-weight:900;opacity:0.85;'>⚡ STACK PROCS</div>",
                unsafe_allow_html=True,
            )

        if armor_annihilation:
            _wl_why_line(
                _svg_icon("stack_armor_annihilation.svg", "Armor Annihilation", "wl-goals"),
                f"Armor Annihilation — iXG≥97 + xGA≥2.52  •  DPS {DPS['armor_annihilation']['win']}% (n={DPS['armor_annihilation']['n']})  (Δ {DPS['armor_annihilation']['win']-base_win:+.1f})",
            )
            _wl_dps_bar(DPS["armor_annihilation"]["win"], "GOALS")

        if fury_shredder:
            _wl_why_line(
                _svg_icon("stack_fury_shredder.svg", "Fury Shredder", "wl-goals"),
                f"Fury Shredder — Funnel core + Drought_G≥2  •  DPS {DPS['fury_shredder']['win']}% (n={DPS['fury_shredder']['n']})  (Δ {DPS['fury_shredder']['win']-base_win:+.1f})",
            )
            _wl_dps_bar(DPS["fury_shredder"]["win"], "GOALS")

    elif mk == "POINTS":
        try:
            rg = float(r.get('Reg_Gap_P10', 0) or 0)
        except Exception:
            rg = 0.0
        st.markdown(f"**MAIN:** 🔥 REG GAP 2.5+ (Reg_Gap_P10 {rg:.2f})")
        if tags_s:
            pass  # tags hidden
    elif mk == "SOG":
        try:
            rg = float(r.get('Reg_Gap_S10', 0) or 0)
        except Exception:
            rg = 0.0
        try:
            line = float(r.get('SOG_Line', 0) or 0)
        except Exception:
            line = 0.0
        st.markdown(f"**MAIN:** 🎯 REG GAP SWEET SPOT (Reg_Gap_S10 {rg:.2f})")
        st.markdown(f"**Line:** {line}")
        if tags_s:
            pass  # tags hidden
    else:
        st.markdown(f"**MAIN:** {tags_s if tags_s else '—'}")

    # SUPPORT: show the compact model context line (matrix/conf/ev/heat/gap/drought) if present
    try:
        ctx = []
        # generic picks, mkt-specific keys are passed via r already
        # (caller provides the right row for the mkt page)
        # We'll try common fields used across the app; missing cols are fine.
        # Matrix / Conf / EV / Heat
        _mx = None
        _cp = None
        _ev = None
        _ht = None
        _rg = None
        _dr = None

        if mkt.upper() == "POINTS":
            _mx = str(r.get("Matrix_Points", "") or "").strip()
            _cp = r.get("Conf_Points", None)
            _ev = r.get("Points_EV%", None)
            _ht = str(r.get("Reg_Heat_P", "") or "").strip()
            _rg = r.get("Reg_Gap_P10", None)
            _dr = r.get("Drought_P", None)

        elif mkt.upper() == "ASSISTS":
            _mx = str(r.get("Matrix_Assists", "") or "").strip()
            _cp = r.get("Conf_Assists", None)
            _ev = r.get("Assists_EV%", None)
            _ht = str(r.get("Reg_Heat_A", "") or "").strip()
            _rg = r.get("Reg_Gap_A10", None)
            _dr = r.get("Drought_A", None)
        elif mkt.upper() in ("SOG", "SHOTS"):
            _mx = str(r.get("Matrix_SOG", "") or "").strip()
            _cp = r.get("Conf_SOG", None)
            _ev = r.get("SOG_EV%", None)
            _ht = str(r.get("Reg_Heat_S", "") or "").strip()
            _rg = r.get("Reg_Gap_S10", None)
            _dr = r.get("Drought_SOG", None)
            if _dr is None or _dr == "":
                _dr = r.get("Drought_S", None)

        if _mx:
            ctx.append(f"Matrix: {_mx}")
        if _cp is not None and _cp != "":
            try:
                ctx.append(f"Conf: {float(_cp):.0f}")
            except Exception:
                ctx.append(f"Conf: {_cp}")
        if _ev is not None and _ev != "":
            try:
                ctx.append(f"EV%: {float(_ev):.1f}")
            except Exception:
                ctx.append(f"EV%: {_ev}")
        if _ht:
            ctx.append(f"Heat: {_ht}")
        if _rg is not None and _rg != "" and not (isinstance(_rg, float) and math.isnan(_rg)):
            try:
                ctx.append(f"Gap10: {float(_rg):.2f}")
            except Exception:
                ctx.append(f"Gap10: {_rg}")
        if _dr is not None and _dr != "" and not (isinstance(_dr, float) and math.isnan(_dr)):
            try:
                ctx.append(f"Drought: {int(float(_dr))}")
            except Exception:
                ctx.append(f"Drought: {_dr}")

        st.markdown("**SUPPORT:**")
        st.caption(" | ".join(ctx) if ctx else "—")
        # Support windows (presentation-only; columns come from tracker)
        try:
            _suffix = {"POINTS":"Points","ASSISTS":"Assists","SOG":"SOG","SHOTS":"SOG"}.get(mkt.upper(), "")
            if _suffix:
                # L10 / L20 / L40 tier + rate + diff
                for _w in (10, 20, 40):
                    _tier = str(r.get(f"L{_w}_Tier_{_suffix}", "") or "").strip()
                    _rate = r.get(f"L{_w}_Rate_{_suffix}", None)
                    _diff = r.get(f"L{_w}_Diff_{_suffix}", None)

                    parts = []
                    if _tier:
                        parts.append(f"Tier {_tier}")

                    try:
                        if _rate is not None and _rate != "" and not (isinstance(_rate, float) and math.isnan(_rate)):
                            parts.append(f"Rate {float(_rate):.2f}")
                    except Exception:
                        pass

                    try:
                        if _diff is not None and _diff != "" and not (isinstance(_diff, float) and math.isnan(_diff)):
                            parts.append(f"Diff {float(_diff):+.2f}")
                    except Exception:
                        pass

                    if parts:
                        st.caption(f"L{_w}: " + " | ".join(parts))

                # Window Signal: stability + trend across diffs
                try:
                    def _num_or_none(v):
                        try:
                            if v is None:
                                return None
                            if isinstance(v, str) and not v.strip():
                                return None
                            x = float(v)
                            if math.isnan(x):
                                return None
                            return x
                        except Exception:
                            return None

                    def _calc_diff_from_rate(_w: int):
                        # Prefer explicit Diff column; otherwise derive from Rate - Line (accuracy-first).
                        d = _num_or_none(r.get(f"L{_w}_Diff_{_suffix}", None))
                        if d is not None:
                            return d
                        rate = _num_or_none(r.get(f"L{_w}_Rate_{_suffix}", None))
                        line_col = {"Points": "Points_Line", "Assists": "Assists_Line", "SOG": "SOG_Line"}.get(_suffix, "")
                        line = _num_or_none(r.get(line_col, None)) if line_col else None
                        if rate is None or line is None:
                            return None
                        return float(rate) - float(line)

                    _d10 = _calc_diff_from_rate(10)
                    _d20 = _calc_diff_from_rate(20)
                    _d40 = _calc_diff_from_rate(40)

                    _badge, _score, _trend = _trend_badge_score(mkt, _d10, _d20, _d40)

                    if _badge:
                        if _trend:
                            st.caption(f"Window Signal: {_badge} ({_score}/100) • Trend: {_trend}")
                        else:
                            st.caption(f"Window Signal: {_badge} ({_score}/100)")
                except Exception:
                    pass
        except Exception:
            pass

    except Exception:
        st.markdown("**SUPPORT:**")
        st.caption("—")

    # TONIGHT: surface matchup context if present (non-breaking)
    try:
        tonight = []
        if bool(r.get("Opp_DefWeak", False)):
            tonight.append("Weak Defense")
        _gw = r.get("Goalie_Weak", None)
        try:
            if _gw is not None and float(_gw) >= 65:
                tonight.append("Weak Goalie")
        except Exception:
            pass
        _ppm = r.get("PP_Matchup", None)
        try:
            if _ppm is not None and float(_ppm) >= 60:
                tonight.append("PP Matchup")
        except Exception:
            pass
        st.markdown(f"**TONIGHT:** {' • '.join(tonight) if tonight else '—'}")
    except Exception:
        st.markdown("**TONIGHT:** —")
def _calc_market_map(market: str) -> dict:
    """
    Maps calculator market -> relevant df columns.
    Returns dict with keys: line_col, odds_col, p_model_col, ev_col, conf_col, matrix_col, green_col, ev_icon_col
    """
    m = (market or "").strip().lower()
    if m.startswith("point"):
        return dict(
            line_col="Points_Line",
            odds_col="Points_Odds_Over",
            p_model_col="Points_p_model_over",
            modelpct_col="Points_Model%",
            evpct_col="Points_EV%",
            conf_col="Conf_Points",
            matrix_col="Matrix_Points",
            green_col="Green_Points",
            ev_icon_col="Plays_EV_Points",
        )
    if m.startswith("sog"):
        return dict(
            line_col="SOG_Line",
            odds_col="SOG_Odds_Over",
            p_model_col="SOG_p_model_over",
            modelpct_col="SOG_Model%",
            evpct_col="SOG_EV%",
            conf_col="Conf_SOG",
            matrix_col="Matrix_SOG",
            green_col="Green_SOG",
            ev_icon_col="Plays_EV_SOG",
        )
    if m.startswith("assist"):
        return dict(
            line_col="Assists_Line",
            odds_col="Assists_Odds_Over",
            p_model_col="Assists_p_model_over",
            modelpct_col="Assists_Model%",
            evpct_col="Assists_EV%",
            conf_col="Conf_Assists",
            matrix_col="Matrix_Assists",
            green_col="Green_Assists",
            ev_icon_col="Plays_EV_Assists",
        )
    # Goal / ATG
    return dict(
        line_col="Goal_Line",
        odds_col="Goal_Odds_Over",
        p_model_col="ATG_p_model_over",
        modelpct_col="ATG_Model%",
        evpct_col="ATG_EV%",
        conf_col="Conf_Goal",
        matrix_col="Matrix_Goal",
        green_col="Green_Goal",
        ev_icon_col="Plays_EV_ATG",
    )


def build_ladder_alerts(
    df_in: pd.DataFrame,
    market: str = "SOG",
    min_line: float = 3.5,
    min_ev: float = 8.0,
    min_model_pct: float = 12.0,
    top_k: int = 6,
    start_from_baseline: bool = True,
) -> pd.DataFrame:
    """Scan Top-K BDL alt lines and return ladder alerts.

    Uses schema from odds_ev_bdl.py:
      - Lines/Odds:  BDL_{M}_Line_{i}, BDL_{M}_Odds_{i}, BDL_{M}_Book_{i}
      - Model prob:  {M}_p_model_over_{i} (0-1) or {M}_Model%_{i} (0-100)
      - EV:          {M}_EVpct_over_{i} (pct) or {M}_EV%_{i} (pct)

    UX behavior:
      - If start_from_baseline=True (default), we treat the player's "baseline" as {M}_Line
        (fallback: BDL_{M}_Line, fallback: min available BDL line) and **only show rungs
        at/above that baseline**, ordered from baseline upward.
      - This prevents the table from feeling like it's "starting at the top".
    """
    if df_in is None or df_in.empty:
        return pd.DataFrame(columns=["Player","Team","Game","Market","Alt#","Line","Odds","Book","Model%","EV%","Rung","Why"])

    M = str(market).strip()
    df0 = df_in

    def _num(v):
        try:
            if v is None:
                return None
            if isinstance(v, str) and not v.strip():
                return None
            x = float(v)
            if math.isnan(x):
                return None
            return x
        except Exception:
            return None

    # A short "why" string so you can see what is driving the ladder ranking.
    def _why(r):
        parts = []
        # Common
        if "Opp_DefWeak" in df0.columns:
            odv = _num(r.get("Opp_DefWeak"))
            if odv is not None:
                parts.append(f"DefWeak {odv:.0f}")
        if f"{M}_mu" in df0.columns:
            mu = _num(r.get(f"{M}_mu"))
            if mu is not None:
                parts.append(f"μ {mu:.2f}")
        # Extra ladder proof (only if present)
        if "v2_defense_vulnerability" in df0.columns:
            dv = _num(r.get("v2_defense_vulnerability"))
            if dv is not None:
                parts.append(f"DefV {dv:.0f}")
        if "opp_5v5_SlotSA60" in df0.columns:
            sv = _num(r.get("opp_5v5_SlotSA60"))
            # If we had to fill neutral (50) league-wide, don't spam the UI with it.
            if sv is not None and abs(float(sv) - 50.0) > 0.01:
                parts.append(f"SlotSA60 {sv:.2f}")
        # Prefer true 5v5 share when available; otherwise fall back to our proxy share.
        sh = None
        if "Player_5v5_SOG_Share" in df0.columns:
            sh = _num(r.get("Player_5v5_SOG_Share"))
        if sh is None and "Player_SOG_Share_Proxy" in df0.columns:
            sh = _num(r.get("Player_SOG_Share_Proxy"))
        if sh is not None:
            parts.append(f"5v5Share {sh:.1f}%")

        # Market-specific
        if M == "SOG":
            for c, lab in [("Med10_SOG","Med10"), ("Avg5_SOG","Avg5"), ("ShotIntent_Pct","SI%"), ("TOI_Pct_Game","TOI%")]:
                if c in df0.columns:
                    v = _num(r.get(c))
                    if v is not None:
                        parts.append(f"{lab} {v:.1f}")
        elif M == "Points":
            for c, lab in [("P10_total","P10"), ("A10_total","A10"), ("G10_total","G10"), ("TOI_Pct_Game","TOI%")]:
                if c in df0.columns:
                    v = _num(r.get(c))
                    if v is not None:
                        parts.append(f"{lab} {v:.0f}")
        elif M == "Assists":
            for c, lab in [("A10_total","A10"), ("iXA%","iXA%"), ("TOI_Pct_Game","TOI%")]:
                if c in df0.columns:
                    v = _num(r.get(c))
                    if v is not None:
                        parts.append(f"{lab} {v:.1f}")
        elif M == "Goal":
            for c, lab in [("G10_total","G10"), ("iXG%","iXG%"), ("Goalie_Weak","GWeak")]:
                if c in df0.columns:
                    v = _num(r.get(c))
                    if v is not None:
                        parts.append(f"{lab} {v:.1f}")
        return " | ".join(parts)

    rows = []
    K = max(1, int(top_k))
    for _, r in df0.iterrows():
        player = str(r.get("Player","") or "").strip()
        if not player:
            continue
        game = str(r.get("Game","") or "").strip()
        team = str(r.get("Team","") or "").strip()

        # Baseline line (mainline) for anchoring
        base = None
        if f"{M}_Line" in df0.columns:
            base = _num(r.get(f"{M}_Line"))
        if base is None and f"BDL_{M}_Line" in df0.columns:
            base = _num(r.get(f"BDL_{M}_Line"))
        if base is None:
            # fallback: minimum available alt line
            mins = []
            for i in range(1, K + 1):
                lc = f"BDL_{M}_Line_{i}"
                if lc not in df0.columns:
                    break
                lv = _num(r.get(lc))
                if lv is not None:
                    mins.append(float(lv))
            if mins:
                base = min(mins)

        for i in range(1, K + 1):
            lc = f"BDL_{M}_Line_{i}"
            if lc not in df0.columns:
                break  # no more ladder cols

            line = _num(r.get(lc))
            if line is None:
                continue

            # Anchor: start at baseline, not at the sky
            if start_from_baseline and base is not None and float(line) < float(base):
                continue

            if float(line) < float(min_line):
                continue

            odds = _num(r.get(f"BDL_{M}_Odds_{i}"))
            book = str(r.get(f"BDL_{M}_Book_{i}", "") or "").strip()

            # model prob
            p = _num(r.get(f"{M}_p_model_over_{i}"))
            if p is None:
                mp = _num(r.get(f"{M}_Model%_{i}"))
                if mp is not None:
                    p = float(mp) / 100.0
            if p is None:
                continue

            model_pct = float(p) * 100.0
            if model_pct < float(min_model_pct):
                continue

            # EV
            ev = _num(r.get(f"{M}_EVpct_over_{i}"))
            if ev is None:
                ev = _num(r.get(f"{M}_EV%_{i}"))
            if ev is not None and float(ev) < float(min_ev):
                continue

            rung = None
            if base is not None:
                rung = float(line) - float(base)

            rows.append({
                "Player": player,
                "Team": team,
                "Game": game,
                "Market": M,
                "Alt#": i,
                "Line": float(line),
                "Odds": (None if odds is None else float(odds)),
                "Book": book,
                "Model%": round(model_pct, 1),
                "EV%": (None if ev is None else round(float(ev), 1)),
                "Rung": (None if rung is None else round(float(rung), 1)),
                "Why": _why(r),
                "DefV": (None if _num(r.get("v2_defense_vulnerability")) is None else round(float(_num(r.get("v2_defense_vulnerability"))), 1)),
                "SlotSA60": (None if _num(r.get("opp_5v5_SlotSA60")) is None or abs(float(_num(r.get("opp_5v5_SlotSA60"))) - 50.0) <= 0.01 else round(float(_num(r.get("opp_5v5_SlotSA60"))), 2)),
                "5v5Share%": (
                    None
                    if (
                        _num(r.get("Player_5v5_SOG_Share")) is None
                        and _num(r.get("Player_SOG_Share_Proxy")) is None
                    )
                    else round(
                        float(
                            _num(r.get("Player_5v5_SOG_Share"))
                            if _num(r.get("Player_5v5_SOG_Share")) is not None
                            else _num(r.get("Player_SOG_Share_Proxy"))
                        ),
                        1,
                    )
                ),
                "OppSOG_L10": (None if _num(r.get("Opp_SOG_Against_L10")) is None else round(float(_num(r.get("Opp_SOG_Against_L10"))), 2)),
                "OppSOG_L50": (None if _num(r.get("Opp_SOG_Against_L50")) is None else round(float(_num(r.get("Opp_SOG_Against_L50"))), 2)),

            })

    out = pd.DataFrame(rows)
    if not out.empty:
        # Show baseline-up ladders first, then best EV/model within each rung.
        sort_cols = ["Rung","EV%","Model%","Line"]
        for c in sort_cols:
            if c not in out.columns:
                sort_cols.remove(c)
        out = out.sort_values(sort_cols, ascending=[True, False, False, True][:len(sort_cols)], na_position="last")
    return out


def warlord_call(ev_pct: float, kelly: float) -> tuple[str, str, str]:
    """
    Returns (label, emoji, why) based on EV% and Kelly%.
    Tune thresholds to taste.
    """
    k = max(0.0, float(kelly)) * 100.0
    e = float(ev_pct)

    if e >= 12 and k >= 6:
        return ("PRESS THE ATTACK", "⚔️", "Big price edge + strong sizing support")
    if e >= 8 and k >= 4:
        return ("STRONG EDGE", "🔥", "Good EV + meaningful sizing support")
    if e >= 5 and k >= 2:
        return ("PLAYABLE", "✅", "Positive EV; size it responsibly")
    if e >= 0:
        return ("SMALL EDGE / PRICE CHECK", "🟡", "Edge is thin; consider smaller stake or pass")
    return ("PASS", "🛑", "Negative EV at this price")

# -------------------------
# Number formatting helpers
# -------------------------

def _icon_is_money(v) -> bool:
    """
    Treat either the literal icon "💰" OR boolean-ish values as +EV.
    This keeps compatibility whether the tracker writes icons or True/False.
    """
    try:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return float(v) >= 1.0
        s = str(v).strip()
        if s == "💰":
            return True
        return s.lower() in ("true", "1", "yes", "y", "t")
    except Exception:
        return False

def _col_bool(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    if df[col].dtype == bool:
        return df[col].fillna(False)
    return df[col].astype(str).str.strip().str.lower().isin(["true","1","yes","y","t","🟢"])

def apply_market_filters(
    df_in: pd.DataFrame,
    f: dict,
    green_col: str,
    ev_icon_col: str,
    conf_col: str | None = None,
    matrix_col: str | None = None,
    lock_col: str = "LOCK",
) -> pd.DataFrame:
    df = df_in.copy()

    if conf_col and conf_col in df.columns:
        df = df[pd.to_numeric(df[conf_col], errors="coerce").fillna(0) >= float(f.get("min_conf", 0))]

    if f.get("hide_reds") and matrix_col and matrix_col in df.columns:
        df = df[~df[matrix_col].astype(str).str.lower().str.contains("red", na=False)]


    if f.get("greens_only"):
        # Prefer explicit boolean green flag if present; otherwise fall back to Matrix=Green
        if green_col in df.columns:
            df = df[_col_bool(df, green_col)]
        else:
            # Fallbacks: common legacy names or matrix color
            alt_cols = [c for c in ("Green_A", "Green", "IsGreen", "Is_Green") if c in df.columns]
            if alt_cols:
                df = df[_col_bool(df, alt_cols[0])]
            elif matrix_col and matrix_col in df.columns:
                df = df[df[matrix_col].astype(str).str.strip().str.upper().isin(["GREEN", "🟢"])]
            else:
                # No green signal column available; do not filter everything away
                pass


    if f.get("ev_only"):
        if ev_icon_col in df.columns:
            df = df[df[ev_icon_col].astype(str).apply(_icon_is_money)]

    if f.get("locks_only"):
        if lock_col in df.columns:
            df = df[df[lock_col].astype(str).str.strip() == "🔒"]
        else:
            g = _col_bool(df, green_col)
            e = df[ev_icon_col].astype(str).apply(_icon_is_money) if ev_icon_col in df.columns else False
            df = df[g & e]

    if f.get("plays_first"):
        tmp = df.copy()
        tmp["_lock_sort"] = (tmp[lock_col].astype(str).str.strip() == "🔒").astype(int) if lock_col in tmp.columns else 0
        tmp["_ev_sort"] = tmp[ev_icon_col].astype(str).apply(_icon_is_money).astype(int) if ev_icon_col in tmp.columns else 0
        sort_cols = ["_lock_sort", "_ev_sort"]
        if conf_col and conf_col in tmp.columns:
            sort_cols.append(conf_col)
        tmp = tmp.sort_values(by=sort_cols, ascending=[False]*len(sort_cols), kind="mergesort")
        tmp = tmp.drop(columns=[c for c in ["_lock_sort","_ev_sort"] if c in tmp.columns])
        df = tmp

    return df

def _is_nan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return True
def snap_half(x):
    """Snap a numeric value to the nearest 0.5 (prop lines should look like 2.5, 3.0, etc.)."""
    try:
        if _is_nan(x):
            return np.nan
        v = float(x)
        return round(v * 2.0) / 2.0
    except Exception:
        return np.nan
def snap_int(x):
    """Cast odds to int-ish (American odds should be -110, +120, etc.)."""
    try:
        if _is_nan(x):
            return np.nan
        return int(round(float(x)))
    except Exception:
        return np.nan



# -------------------------
# UI helpers
# -------------------------
def _promote_call_cols(cols):
    order=[
        'SOG_Call','Points_Call','Assists_Call','ATG_Call',
        'Player','Team','Opp','Time','Game','Pos','Tier_Tag','🔥','💰',
    ]
    out=[]
    for c in order:
        if c in cols and c not in out:
            out.append(c)
    for c in cols:
        if c not in out:
            out.append(c)
    return out

COLUMN_WIDTHS = {
    # identity
    "Game": "small",
    "Time": "small",
    "Pos": "small",
    "Team": "small",
    "Opp": "small",
    "Player": "medium",

    # core decision columns
    "Matrix_Points": "small",
    "Matrix_SOG": "small",
    "Matrix_Goal": "small",
    "Matrix_Assists": "small",

    "Conf_Points": "small",
    "Conf_SOG": "small",
    "Conf_Goal": "small",
    "Conf_Assists": "small",
    "Best_Conf": "small",

    # indicators
    "Green": "small",
    "GF_Gate_Badge": "small",
    "Tier_Tag": "small",
    "🔥": "small",
    "💰": "small",

    # drought
    "Drought_P": "small",
    "Drought_A": "small",
    "Drought_G": "small",
    "Drought_SOG": "small",
    "Best_Drought": "small",

    "SOG_Line": "small",
    "SOG_Book": "small",
    "SOG_Odds_Over": "small",
    "SOG_EVpct_over": "small",
    "SOG_Call": "medium",
    "SOG_p_model_over": "small",
    "SOG_p_imp_over": "small",
    "Plays_EV_SOG": "small",

    # EV / odds for other markets
    "Points_Line": "small",
    "Points_Book": "small",
    "Points_Odds_Over": "small",
    "Points_p_model_over": "small",
    "Points_p_imp_over": "small",
    "Points_EVpct_over": "small",
    "Points_Call": "medium",
    "Plays_EV_Points": "small",

    "Goal_Line": "small",
    "Goal_Book": "small",
    "Goal_Odds_Over": "small",
    "Goal_p_model_over": "small",
    "Goal_p_imp_over": "small",
    "Goal_EVpct_over": "small",
    "Plays_EV_Goal": "small",

    "Goal_Line": "small",
    "Goal_Book": "small",
    "Goal_Odds_Over": "small",
    "ATG_p_model_over": "small",
    "ATG_p_imp_over": "small",
    "ATG_EVpct_over": "small",
    "ATG_Call": "medium",
    "Plays_EV_ATG": "small",

    "Assists_Line": "small",
    "Assists_Book": "small",
    "Assists_Odds_Over": "small",
    "Assists_p_model_over": "small",
    "Assists_p_imp_over": "small",
    "Assists_EVpct_over": "small",
    "Assists_Call": "medium",
    "Plays_EV_Assists": "small",


    # goalie / defense
    "Opp_Goalie": "medium",
    "Opp_SV": "small",
    "Opp_GAA": "small",
    "Goalie_Weak": "small",
    "Opp_DefWeak": "small",
    "Opp_SA_Avg_L10": "small",
    "Opp_GA_Avg_L10": "small",

    # misc
    "Line": "small",
    "Odds": "small",
    "Result": "small",
    "Markets": "medium",
    "EV_Signal": "medium",
    "LOCK": "small",
}


# Column tooltips (market engines & key signals)
COLUMN_HELP = {
    # Assists
    "PP_PROOF": "PP Proof = PP usage + PP iXA creation + matchup aligned. Required for Assists engine (EV+Green+0.5).",
    # Points
    "Reg_Gap_P10": "Points engine: Reg_Gap_P10 ≥ 2.5 is the verified regression cliff (EV+Green+0.5).",
    # SOG
    "Reg_Gap_S10": "SOG engine sweet spot: Reg_Gap_S10 in [2.6, 4.3] (EV+Green+Line≤2.5).",
    # Points (UI)
    "REG_LABEL": "Regression status label for Points (NO REG / BUILDING / READY).",
    "REG_PRESSURE": "Visual pressure meter from Reg_Gap_P10 (text bar). Pops when >=2.5.",
    "REG_DROUGHT": "🎰 Jackpot badge when Points engine is READY and Drought_P ≥ 2 (amplifier; not a gate).",
}
def build_column_config(df: pd.DataFrame, cols: list[str]) -> dict:
    cfg = {}

    for c in cols:
        width = COLUMN_WIDTHS.get(c, "small")
        help_txt = COLUMN_HELP.get(c)

        if c not in df.columns:
            cfg[c] = st.column_config.TextColumn(width=width, help=help_txt)
            continue

        if pd.api.types.is_numeric_dtype(df[c]):
            # Betting-friendly numeric formats
            if c.endswith("_Line") or c == "Line":
                cfg[c] = st.column_config.NumberColumn(width=width, format="%.1f", help=help_txt)
            elif c.endswith("_Odds_Over") or c == "Odds":
                cfg[c] = st.column_config.NumberColumn(width=width, format="%.0f", help=help_txt)
            elif c.endswith("_Model%") or c.endswith("_Imp%") or c.endswith("_EV%"):
                cfg[c] = st.column_config.NumberColumn(width=width, format="%.1f", help=help_txt)
            else:
                cfg[c] = st.column_config.NumberColumn(width=width, help=help_txt)
        else:
            cfg[c] = st.column_config.TextColumn(width=width, help=help_txt)

    return cfg


def _text_bar(value: float, vmin: float = 0.0, vmax: float = 8.0, width: int = 10) -> str:
    """Text-only progress bar for st.dataframe (emoji/blocks)."""
    try:
        v = float(value)
    except Exception:
        v = 0.0
    v = max(vmin, min(vmax, v))
    filled = int(round((v - vmin) / (vmax - vmin) * width)) if vmax > vmin else 0
    filled = max(0, min(width, filled))
    return ("█" * filled) + ("░" * (width - filled))



# -------------------------
# Safe getters
# -------------------------

def _to_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, str):
            x = x.strip().replace('%', '')
            if x == "":
                return default
        return float(x)
    except Exception:
        return default

def _get(row, *keys, default=""):
    """Safe getter for dict-like rows (pandas Series or dict).

    Usage:
      _get(r, "A", "A_alt", default="")
    Returns first non-null value among keys, else default.
    """
    # pandas Series supports `get`, dict supports `get`
    for key in keys:
        try:
            v = row.get(key, None)
        except Exception:
            v = None
        # treat NaN as missing
        if v is None:
            continue
        try:
            # pandas NaN
            if isinstance(v, float) and v != v:
                continue
        except Exception:
            pass
        return v
    return default
def _is_hot(reg_scored: str) -> bool:
    """Treat these as Hot regression tiers."""
    if not reg_scored:
        return False
    s = str(reg_scored).strip().lower()
    return s in ("hot", "due", "overdue", "very hot")


# -------------------------
# CONFIG
# -------------------------
OUTPUT_DIR = "output"
st.set_page_config(layout='wide', page_title="The Warlord's NHL Prop Tool")

def inject_warlord_css():
    import streamlit as st
    st.markdown("""
    <style>
      .wl-card{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 12px 14px;
        margin-bottom: 10px;
      }
      .wl-pill{
        display:inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.12);
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .2px;
      }

      /* market pills */
      .wl-purple{ background: rgba(168,85,247,0.14); border-color: rgba(168,85,247,0.35); } /* Assists */
      .wl-blue  { background: rgba(59,130,246,0.14);  border-color: rgba(59,130,246,0.35);} /* Points */
      .wl-orange{ background: rgba(34,197,94,0.14); border-color: rgba(34,197,94,0.35);} /* SOG (Jungle) */

      /* icon system (inline SVG) */
      .wl-ico{ display:inline-flex; align-items:center; justify-content:center; width:18px; height:18px; margin-right:6px; }
      .wl-ico svg{ width:18px; height:18px; display:block; }
      .wl-ico.wl-mono svg *{ fill: currentColor !important; stroke: none !important; }
.wl-ico.wl-keep svg *{ stroke: none !important; }
      .wl-ico.wl-goals{ color: rgba(239,68,68,0.95); }
      .wl-ico.wl-sog{ color: rgba(34,197,94,0.95); }
      .wl-ico.wl-points{ color: rgba(59,130,246,0.95); }
      .wl-ico.wl-assists{ color: rgba(168,85,247,0.95); }

      .wl-red   { background: rgba(239,68,68,0.14);  border-color: rgba(239,68,68,0.35);}  /* Goals */

      /* accent stripe */
      .wl-accent-purple{ background: rgba(168,85,247,0.18); border-left: 5px solid #a855f7; }
      .wl-accent-blue{ background: rgba(59,130,246,0.18); border-left: 5px solid #3b82f6; }
      .wl-accent-orange{ background: rgba(34,197,94,0.18); border-left: 5px solid #22c55e; }
      .wl-accent-red{ background: rgba(239,68,68,0.18); border-left: 5px solid #ef4444; }
    </style>
    """, unsafe_allow_html=True)

def _fmt_hms(delta_seconds: int) -> str:
    delta_seconds = max(0, int(delta_seconds))
    h = delta_seconds // 3600
    m = (delta_seconds % 3600) // 60
    s = delta_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def render_vengeance_banner():
    """
    Top-of-app theme banner:
      - Pre-slate: countdown + "Vengeance is coming"
      - Live: "Clock strikes vengeance" + "Cook the books" + elapsed
      - Post: manual "Slate Complete" -> countdown to next strike
    Timezone: America/Chicago (per user).
    """
    import streamlit as st
    from datetime import datetime, timedelta
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("America/Chicago")
    except Exception:
        tz = None  # fallback: naive

    now = datetime.now(tz) if tz else datetime.now()

    presets = {
        "Weeknight Main (6:00 PM CT)": (18, 0),
        "Weeknight Alt (6:30 PM CT)": (18, 30),
        "Weeknight Late (7:00 PM CT)": (19, 0),
        "Weekend Early (11:30 AM CT)": (11, 30),
        "Weekend Midday (12:30 PM CT)": (12, 30),
        "Custom…": None,
    }

    # Default preset heuristics
    if "vengeance_preset" not in st.session_state:
        dow = now.weekday()  # 0=Mon
        st.session_state.vengeance_preset = "Weeknight Main (6:00 PM CT)" if dow < 5 else "Weekend Early (11:30 AM CT)"
    if "vengeance_custom_h" not in st.session_state:
        st.session_state.vengeance_custom_h = 18
    if "vengeance_custom_m" not in st.session_state:
        st.session_state.vengeance_custom_m = 0
    if "vengeance_completed_for" not in st.session_state:
        st.session_state.vengeance_completed_for = ""  # key like YYYY-MM-DD@HH:MM

    # Controls row (compact)
    c1, c2, c3, c4 = st.columns([2.2, 1.1, 1.1, 1.1], gap="small")
    with c1:
        st.session_state.vengeance_preset = st.selectbox(
            "Tonight's strike",
            list(presets.keys()),
            index=list(presets.keys()).index(st.session_state.vengeance_preset) if st.session_state.vengeance_preset in presets else 0,
            label_visibility="collapsed",
        )
    # Determine target hour/minute
    if st.session_state.vengeance_preset == "Custom…":
        with c2:
            st.session_state.vengeance_custom_h = st.number_input("Hr", min_value=0, max_value=23, value=int(st.session_state.vengeance_custom_h), step=1, label_visibility="collapsed")
        with c3:
            st.session_state.vengeance_custom_m = st.number_input("Min", min_value=0, max_value=59, value=int(st.session_state.vengeance_custom_m), step=1, label_visibility="collapsed")
        target_h, target_m = int(st.session_state.vengeance_custom_h), int(st.session_state.vengeance_custom_m)
    else:
        hm = presets.get(st.session_state.vengeance_preset) or (18, 0)
        target_h, target_m = hm
        with c2:
            st.markdown(f"<span class='vengeance-pill'>Strike: {target_h%12 or 12}:{target_m:02d} {'PM' if target_h>=12 else 'AM'} CT</span>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<span class='vengeance-pill'>TZ: CT</span>", unsafe_allow_html=True)

    with c4:
        if st.button("🔄", help="Refresh the clock"):
            st.rerun()

    # Compute strike datetime (today if upcoming, else tomorrow)
    strike_dt = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
    if now >= strike_dt and (now - strike_dt).total_seconds() > 0:
        # if already past strike today, treat as "today's strike" only if not completed and we want live;
        # else roll to next day for countdown
        pass

    strike_key = f"{strike_dt.date().isoformat()}@{target_h:02d}:{target_m:02d}"
    completed_key = st.session_state.vengeance_completed_for

    # If completed_key is different and now is after strike_dt by a lot, we may be looking at next day.
    # We'll treat "live" as: now >= strike_dt and not completed for this strike_key.
    is_completed = (completed_key == strike_key)

    # If now is past today's strike and it's completed, next strike is tomorrow.
    # If now is past today's strike and it's NOT completed, we are LIVE until user completes.
    # If now is before today's strike, we are PRE.
    if now < strike_dt:
        state = "pre"
        t_delta = int((strike_dt - now).total_seconds())
        big_timer = _fmt_hms(t_delta)
        kicker = "VENGEANCE IS COMING"
        head = "VENGEANCE IS COMING"
        sub = f"Clock strikes at {target_h%12 or 12}:{target_m:02d} {'PM' if target_h>=12 else 'AM'} CT"
        right_timer = f"STRIKES IN {big_timer}"
        pill = "MODELS: ARMING"
        wrap_class = "vengeance-wrap vengeance-pre"
        action_line = "Sharpening the blades…"
    else:
        if not is_completed:
            state = "live"
            elapsed = int((now - strike_dt).total_seconds())
            kicker = "THE CLOCK STRIKES VENGEANCE"
            head = "COOK THE BOOKS."
            sub = "Slate is live — build the board."
            right_timer = f"LIVE {_fmt_hms(elapsed)}"
            pill = "LEDGER: RECORDING"
            wrap_class = "vengeance-wrap vengeance-live"
            action_line = "Punish the lines."
        else:
            state = "post"
            next_dt = strike_dt + timedelta(days=1)
            remaining = int((next_dt - now).total_seconds())
            kicker = "VENGEANCE HAS BEEN SERVED"
            head = "TALLY THE DAMAGE."
            sub = f"Next strike at {next_dt.hour%12 or 12}:{next_dt.minute:02d} {'PM' if next_dt.hour>=12 else 'AM'} CT"
            right_timer = f"NEXT {_fmt_hms(remaining)}"
            pill = "REVIEW: ACTIVE"
            wrap_class = "vengeance-wrap vengeance-post"
            action_line = "Post-mortem underway."

    # Banner layout
    left, right = st.columns([3.2, 1.2], gap="small")
    with left:
        st.markdown(
            f"""
            <div class="{wrap_class}">
              <div class="vengeance-kicker">{kicker}</div>
              <div class="vengeance-head">{head}</div>
              <div class="vengeance-sub">{sub} <span style="opacity:.7">•</span> {action_line} <span style="opacity:.7">•</span> <span class="vengeance-pill">{pill}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="{wrap_class}" style="display:flex; flex-direction:column; justify-content:center;">
              <div class="vengeance-timer">{right_timer}</div>
              <div style="text-align:right; margin-top:6px; opacity:.85; font-size:12px;">
                Strike key: {strike_key}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Manual slate complete (Option 2)
    if state == "live":
        b1, b2 = st.columns([1, 5])
        with b1:
            if st.button("✅ Slate Complete", help="Ends the live state until the next strike time"):
                st.session_state.vengeance_completed_for = strike_key
                st.rerun()
        with b2:
            st.caption("Live until you mark it complete. Keeps the vibe right even when real slate end-times vary.")








# -------------------------------------------------------------------
# Safety: ensure rich "Why it fires" renderer exists (prevents NameError
# if a partial merge / paste removed the function definition).
# -------------------------------------------------------------------
if "_render_why_it_fires_rich" not in globals():
    def _render_why_it_fires_rich(mkt: str, r, tags: str = "") -> None:
        """Fallback renderer: keeps the app running if rich renderer is missing."""
        mk = str(mkt or "").strip().upper()
        st.caption(f"{mk} — Why it fires")
        if tags:
            st.write(tags)
        # best-effort: print a few key fields
        try:
            for k in ("Player","Game","Pos","Conf_Goal","Conf_Points","Conf_SOG","Conf_Assists",
                      "Matrix_Goal","Matrix_Points","Matrix_SOG","Matrix_Assists",
                      "Goal_Line","Goal_Line","SOG_Line","Points_Line","Assists_Line",
                      "Avg5_SOG","Med10_SOG","ShotIntent","ShotIntent_Pct","opp_5v5_xGA60","Goalie_Weak"):
                if k in getattr(r, "keys", lambda: [])():
                    v = r.get(k, None)
                    if v not in (None, "", np.nan):
                        st.caption(f"{k}: {v}")
        except Exception:
            pass




# =========================
# Ledger helpers (append-only bet tracking)
# =========================
UNIT_VALUE_USD = 50.0   # 1u = $50 (user-defined)
MAX_STAKE_U = 3.0       # cap per play

# CSV headers (append-only)
BETSLIP_HEADERS = [
    'bet_id','date','datetime_placed','game','player','market','line','odds_taken','book','stake_u',
    'earned_green','ev_flag','lock_flag','conf','matrix','model_pct','imp_pct','ev_pct','tier','proof_count','why_tags',
    'opp','opp_goalie','notes'
]

BET_EVENTS_HEADERS = [
    'bet_id','event_type','event_datetime','event_period','event_game_minute','units_net','source','event_notes'
]


def inject_vengeance_css() -> None:
    """Inject Vengeance banner CSS (idempotent)."""
    import streamlit as st
    st.markdown("""<style>
/* -------------------------
         VENGEANCE BANNER
         ------------------------- */
      .vengeance-wrap{
        color: rgba(255,255,255,0.96);
        text-shadow: 0 1px 2px rgba(0,0,0,0.55);
        border-radius: 18px;
        padding: 14px 16px;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 8px 22px rgba(0,0,0,0.28);
        margin: 10px 0 14px 0;
      }
      .vengeance-pre{
        background: radial-gradient(1200px 120px at 10% 0%, rgba(255,255,255,0.14), rgba(0,0,0,0.0)),
                    linear-gradient(90deg, rgba(25,25,25,0.95), rgba(45,18,18,0.92));
      }
      .vengeance-live{
        background: radial-gradient(900px 140px at 15% 0%, rgba(255,255,255,0.16), rgba(0,0,0,0.0)),
                    linear-gradient(90deg, rgba(80,0,0,0.95), rgba(20,10,10,0.92));
        animation: vengeancePulse 1.6s ease-in-out infinite;
      }
      .vengeance-post{
        background: radial-gradient(1200px 120px at 10% 0%, rgba(255,255,255,0.10), rgba(0,0,0,0.0)),
                    linear-gradient(90deg, rgba(20,25,32,0.95), rgba(12,18,20,0.92));
      }
      @keyframes vengeancePulse{
        0%{ filter: brightness(1.00); transform: translateY(0px); }
        50%{ filter: brightness(1.08); transform: translateY(-1px); }
        100%{ filter: brightness(1.00); transform: translateY(0px); }
      }
      .vengeance-kicker{
        color: rgba(255,255,255,0.78);
        font-weight: 900;
        letter-spacing: 1.2px;
        font-size: 12px;
        opacity: 0.9;
        text-transform: uppercase;
      }
      .vengeance-head{
        color: rgba(255,255,255,0.98);
        font-weight: 950;
        letter-spacing: 0.6px;
        font-size: 26px;
        line-height: 1.05;
        margin-top: 2px;
      }
      .vengeance-sub{
        color: rgba(255,255,255,0.86);
        margin-top: 4px;
        font-size: 13px;
        opacity: 0.92;
      }
      .vengeance-timer{
        color: rgba(255,255,255,0.96);
        font-weight: 950;
        font-variant-numeric: tabular-nums;
        letter-spacing: 1px;
        font-size: 30px;
        text-align: right;
        line-height: 1.05;
      }
      .vengeance-pill{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.18);
        font-size: 12px;
        font-weight: 800;
        opacity: 0.95;
      }

    
</style>""", unsafe_allow_html=True)

inject_warlord_css()




# -----------------------------





# =========================

# =========================
# HELPERS
# =========================
def find_latest_tracker_csv(output_dir: str) -> str | None:
    files = glob.glob(os.path.join(output_dir, "tracker_*.csv"))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]
def to_bool_series(s: pd.Series) -> pd.Series:
    # Handles True/False, 1/0, "true"/"false", etc.
    if s is None:
        return pd.Series([False] * 0)
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )
def safe_num(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)



# -------------------------
# Signals-first helpers
# -------------------------
def _is_money(x) -> bool:
    return _icon_is_money(x)

def _fmt_ev_pct(x) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return ""
        v = float(x)
        return f"{v:+.1f}%"
    except Exception:
        return ""
def build_markets_pills(row) -> str:
    pills = []
    for key, label in [
        ("Matrix_Points", "PTS"),
        ("Matrix_SOG", "SOG"),
        ("Matrix_Goal", "G"),
        ("Matrix_Assists", "A"),
    ]:
        v = str(row.get(key, "")).lower()
        if not v:
            continue
        if "green" in v:
            pills.append(f"🟢{label}")
        elif "yellow" in v:
            pills.append(f"🟡{label}")
        elif "red" in v:
            pills.append(f"🔴{label}")
        else:
            pills.append(f"⚪{label}")
    return " ".join(pills)
def build_ev_signal(green_bool, money_icon, ev_pct) -> str:
    g = bool(green_bool) if green_bool is not None else False
    m = _is_money(money_icon)
    icons = ("🟢" if g else "") + ("💰" if m else "")
    evs = _fmt_ev_pct(ev_pct)
    if icons and evs:
        return f"{icons} {evs}"
    if icons:
        return icons
    return evs
def build_lock_badge(green_bool, money_icon) -> str:
    g = bool(green_bool) if green_bool is not None else False
    m = _is_money(money_icon)
    return "🔒" if (g and m) else ""
def board_best_market_ev(row) -> tuple[str, str]:
    bm = str(row.get("Best_Market", "")).strip().lower()
    mapping = [
        ("point", "Green_Points", "Plays_EV_Points", "Points_EV%"),
        ("sog", "Green_SOG", "Plays_EV_SOG", "SOG_EV%"),
        ("goal", "Green_Goal", "Plays_EV_ATG", "ATG_EV%"),
        ("assist", "Green_Assists", "Plays_EV_Assists", "Assists_EV%"),
    ]
    for token, gcol, ecol, pcol in mapping:
        if token and token in bm:
            g = row.get(gcol, False)
            e = row.get(ecol, "")
            p = row.get(pcol, None)
            return build_ev_signal(g, e, p), build_lock_badge(g, e)
    return "", ""
def safe_str(df: pd.DataFrame, col: str, default="") -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return df[col].astype(str).fillna(default)
def style_df(df: pd.DataFrame, cols: list[str]) -> "pd.io.formats.style.Styler":
    # Market color pill (Ninja Turtles palette)
    mkt_bg = {
        "ASSISTS": "#a855f7",   # purple
        "A": "#a855f7",
        "POINTS": "#0b1b3a",    # dark blue
        "PTS": "#0b1b3a",
        "SOG": "#22c55e",       # green
        "SHOTS": "#22c55e",
        "GOALS": "#ef4444",     # red
        "G": "#ef4444",
    }
    def _mkt_style(v):
        key = str(v).upper().strip()
        bg = mkt_bg.get(key, "")
        if not bg:
            return ""
        return f"background-color: {bg}; color: white; font-weight: 700;"

    # --- Pandas Styler REQUIRES unique index + unique columns ---
    cols = [c for c in dict.fromkeys(cols) if c in df.columns]

    view = df.loc[:, cols].copy().reset_index(drop=True)

    if view.columns.duplicated().any():
        view = view.loc[:, ~view.columns.duplicated()].copy()

    def matrix_style(v):
        s = str(v).strip().lower()
        if s == "green":
            return "background-color:#1f7a1f;color:white;font-weight:700;"
        if s == "yellow":
            return "background-color:#b38f00;color:white;font-weight:700;"
        if s == "red":
            return "background-color:#8b1a1a;color:white;font-weight:700;"
        return ""

    def heat_style(v):
        s = str(v).strip().upper()
        if s == "HOT":
            return "background-color:#b30000;color:white;font-weight:700;"
        if s == "WARM":
            return "background-color:#e67300;color:white;font-weight:700;"
        if s == "COOL":
            return "background-color:#1f5aa6;color:white;font-weight:700;"
        return ""

    def _conf_style_for(thr_green: float):
        # Per-market confidence thresholds (UI only):
        #   Points: 70+ green
        #   SOG:    75+ green
        #   Assists:80+ green
        #   Goals:  85+ green
        # Yellow = (green - 10). Red otherwise.
        try:
            tg = float(thr_green)
        except Exception:
            tg = 80.0
        ty = tg - 10.0
        def _style(v):
            try:
                x = float(v)
            except Exception:
                return ""
            if x >= tg:
                return "background-color:#1f7a1f;color:white;font-weight:700;"
            if x >= ty:
                return "background-color:#b38f00;color:white;font-weight:700;"
            return "background-color:#8b1a1a;color:white;font-weight:700;"
        return _style


    def ev_style(v):
        try:
            x = float(v)
        except Exception:
            return ""
        if x >= 10:
            return "background-color:#1f7a1f;color:white;font-weight:700;"
        if x >= 5:
            return "background-color:#b38f00;color:white;font-weight:700;"
        if x < 0:
            return "background-color:#8b1a1a;color:white;font-weight:700;"
        return ""

    def ev_signal_style(v):
        s = str(v)
        if not s or s.strip() == "":
            return ""
        if "%" in s:
            return "background-color: rgba(0, 180, 0, 0.20);color: #0b4f0b;font-weight: 700;"
        return ""

    def play_ev_style(v):
        return "background-color:#1f7a1f;color:white;font-weight:700;" if str(v).strip() == "💰" else ""

    def weak_style(v):
        try:
            x = float(v)
        except Exception:
            return ""
        if x >= 75:
            return "background-color:#b30000;color:white;font-weight:700;"
        return ""

    sty = view.style

    if "EV_Signal" in view.columns:
        sty = sty.applymap(ev_signal_style, subset=["EV_Signal"])

    for c in ["Matrix_Points", "Matrix_SOG", "Matrix_Assists", "Matrix_Goal"]:
        if c in view.columns:
            sty = sty.applymap(matrix_style, subset=[c])

    for c in ["Reg_Heat_P", "Reg_Heat_S", "Reg_Heat_G", "Reg_Heat_A"]:
        if c in view.columns:
            sty = sty.applymap(heat_style, subset=[c])

    # Per-market Conf coloring (keeps each market's own green threshold)
    conf_thr = {
        "Conf_Points": 70,
        "Conf_SOG": 75,
        "Conf_Assists": 80,
        "Conf_Goal": 85,
        "Best_Conf": 80,
    }
    for c, thr in conf_thr.items():
        if c in view.columns:
            sty = sty.applymap(_conf_style_for(thr), subset=[c])

    for c in [c for c in view.columns if c.endswith("EVpct_over")]:
        sty = sty.applymap(ev_style, subset=[c])

    # 🗡️ Dagger highlight
    def dagger_tag_style(v):
        return "background-color:#5a00b3;color:white;font-weight:800;" if str(v).strip() == "🗡️" else ""

    def dagger_score_style(v):
        try:
            x = float(v)
        except Exception:
            return ""
        if x >= 65:
            return "background-color:#1f7a1f;color:white;font-weight:800;"
        if x >= 55:
            return "background-color:#b38f00;color:white;font-weight:800;"
        return ""

    if "🗡️" in view.columns:
        sty = sty.applymap(dagger_tag_style, subset=["🗡️"])
    if "Assist_Dagger" in view.columns:
        sty = sty.applymap(dagger_score_style, subset=["Assist_Dagger"])

    for c in [c for c in view.columns if c.startswith("Plays_EV_")]:
        sty = sty.applymap(play_ev_style, subset=[c])

    for c in ["Goalie_Weak", "Opp_DefWeak"]:
        if c in view.columns:
            sty = sty.applymap(weak_style, subset=[c])

    fmt2_cols = [
        "Exp_A_10", "Reg_Gap_A10",
        "Exp_P_10", "Reg_Gap_P10",
        "Exp_G_10", "Reg_Gap_G10",
        "Exp_S_10", "Reg_Gap_S10",
        "TalentMult", "TOI_per_game",
        "Opp_SV", "Opp_GAA",
    ]

    fmt1_cols = [
        "iXA%", "iXG%",
        "Goalie_Weak", "Opp_DefWeak","L10_P","L10_A","L10_G","L10_SOG",
        "TOI_Pct", "StarScore","Med10_SOG","ShotIntent","Avg5_SOG","Drought_SOG",
        "ShotIntent_Pct","Drought_A","Drought_P","Drought_G",
        "v2_player_stability",
        "team_5v5_SF60_pct",
        "team_5v5_xGF60_pct",
    ]

    format_dict = {}
    for c in fmt2_cols:
        if c in view.columns:
            format_dict[c] = "{:.2f}"
    for c in fmt1_cols:
        if c in view.columns:
            format_dict[c] = "{:.1f}"

    for c in view.columns:
        if c.endswith("_Line") or c in ("Line",):
            format_dict.setdefault(c, "{:.1f}")

    for c in view.columns:
        if c.endswith("_Odds_Over") or c in ("Odds",):
            format_dict.setdefault(c, "{:.0f}")

    for c in view.columns:
        if c.endswith("_p_model_over") or c.endswith("_p_imp_over"):
            format_dict.setdefault(c, "{:.1%}")

    if format_dict:
        sty = sty.format(format_dict, na_rep="")
    if "Market" in view.columns:
        sty = sty.applymap(_mkt_style, subset=["Market"])


    return sty

def add_ui_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Preserve an existing 💰 column from the tracker (some trackers provide 💰 directly)
    _money_existing = out['💰'].copy() if '💰' in out.columns else None

    # Ensure these exist
    if "Play_Tag" not in out.columns:
        out["Play_Tag"] = ""
    if "Plays_Points" not in out.columns:
        out["Plays_Points"] = False

    plays_points = to_bool_series(out["Plays_Points"]) if "Plays_Points" in out.columns else pd.Series(False, index=out.index)

    # Fire indicator
    out["🔥"] = plays_points.map(lambda x: "🔥" if x else "")

    # 💰 EV indicator (any market): show when Plays_EV_* is true
    ev_cols = [
        "Plays_EV_SOG", "Plays_EV_Points", "Plays_EV_Goal", "Plays_EV_ATG", "Plays_EV_Assists",
    ]
    ev_any = pd.Series(False, index=out.index)
    for c in ev_cols:
        if c in out.columns:
            ev_any = ev_any | to_bool_series(out[c])
    out["💰"] = ev_any.map(lambda x: "💰" if bool(x) else "")

    # If no Plays_EV_* columns existed but the tracker already had 💰, keep it.
    if _money_existing is not None:
        have_ev_cols = any((c in out.columns) for c in [
            "Plays_EV_SOG", "Plays_EV_Points", "Plays_EV_Goal", "Plays_EV_ATG", "Plays_EV_Assists"
        ])
        if not have_ev_cols:
            out["💰"] = _money_existing

    return out
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # -------------------------
    # Tier alias (schema drift guard)
    # -------------------------
    if "Tier_Tag" not in df.columns:
        if "Tier" in df.columns:
            df["Tier_Tag"] = df["Tier"]
        elif "Talent_Tier" in df.columns:
            df["Tier_Tag"] = df["Talent_Tier"]
        else:
            df["Tier_Tag"] = ""

    # --- SNAP BETTING LINES AFTER df EXISTS (fix NameError) ---
    def snap_half_down_sog(x):
        try:
            if x is None:
                return x
            v = float(x)
            snapped = round(v * 2.0) / 2.0
            if abs(snapped - round(snapped)) < 1e-6:
                return max(0.5, snapped - 0.5)
            return snapped
        except Exception:
            return x

    for _c in [c for c in df.columns if c.endswith('_Line') or c == 'Line']:
        if _c == 'SOG_Line':
            df[_c] = df[_c].apply(snap_half_down_sog)
        else:
            df[_c] = df[_c].apply(snap_half)

    df.columns = [c.strip() for c in df.columns]

    # Add local game time (for table + matchup filter)
    if "StartTimeLocal" in df.columns and "Time" not in df.columns:
        dt = pd.to_datetime(df["StartTimeLocal"], errors="coerce")
        # Use a portable format and strip leading zero (07:00 PM -> 7:00 PM)
        df["Time"] = dt.dt.strftime("%I:%M %p").astype(str).str.lstrip("0")
        df.loc[dt.isna(), "Time"] = ""

    return df
def filter_common(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    st.sidebar.subheader("Filters")

    # Search player
    q = st.sidebar.text_input("Search player", value="").strip().lower()
    if q:
        out = out[safe_str(out, "Player").str.lower().str.contains(q, na=False)]

    # Team filter
    if "Team" in out.columns:
        teams = sorted([t for t in out["Team"].dropna().astype(str).unique().tolist() if t.strip()])
        sel_teams = st.sidebar.multiselect("Team", teams, default=[])
        if sel_teams:
            out = out[out["Team"].astype(str).isin(sel_teams)]
    # Matchup filter (show time when available)
    if "Game" in out.columns:
        games = [g for g in out["Game"].dropna().astype(str).unique().tolist() if g.strip()]

        if "Time" in out.columns:
            # Build label -> game mapping like "7:00 PM — DAL@STL"
            tmp = out[["Game", "Time"]].copy()
            tmp["Time"] = tmp["Time"].astype(str).fillna("")
            # Prefer earliest time per game if duplicates exist
            best = (
                tmp.sort_values(["Game", "Time"])
                .drop_duplicates(subset=["Game"], keep="first")
                .set_index("Game")["Time"]
                .to_dict()
            )

            labels = []
            for g in games:
                t = best.get(g, "")
                label = f"{t} — {g}" if t else g
                labels.append(label)

            labels = sorted(labels, key=lambda x: x.split("—")[-1].strip())
            sel_labels = st.sidebar.multiselect("Matchup", labels, default=[])

            if sel_labels:
                sel_games = [lab.split("—")[-1].strip() for lab in sel_labels]
                out = out[out["Game"].astype(str).isin(sel_games)]
        else:
            games = sorted(games)
            sel_games = st.sidebar.multiselect("Matchup", games, default=[])
            if sel_games:
                out = out[out["Game"].astype(str).isin(sel_games)]
    # Only flagged plays
    only_fire = st.sidebar.checkbox("Only 🔥 plays", value=False)
    if only_fire and "🔥" in out.columns:
        out = out[out["🔥"] == "🔥"]

    only_ev = st.sidebar.checkbox("Only 💰 plays", value=False)
    if only_ev and "💰" in out.columns:
        out = out[out["💰"] == "💰"]

    return out
def sort_board(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_bc"] = safe_num(out, "Best_Conf", 0)
    out["_gw"] = safe_num(out, "Goalie_Weak", 50)
    out["_dw"] = safe_num(out, "Opp_DefWeak", 50)
    out = out.sort_values(["_bc", "_gw", "_dw"], ascending=[False, False, False]).drop(columns=["_bc", "_gw", "_dw"], errors="ignore")
    return out
def show_games_times(df: pd.DataFrame):
    if "Game" not in df.columns:
        return

    have_local = "StartTimeLocal" in df.columns
    have_utc = "StartTimeUTC" in df.columns
    if not (have_local or have_utc):
        return

    tmp = df.copy()
    tmp["Game"] = tmp["Game"].astype(str).fillna("").str.strip()
    tmp = tmp[tmp["Game"] != ""]

    if have_local:
        tmp["StartTimeLocal"] = tmp["StartTimeLocal"].astype(str).fillna("").str.strip()
    if have_utc:
        tmp["StartTimeUTC"] = tmp["StartTimeUTC"].astype(str).fillna("").str.strip()

    def first_nonempty(series: pd.Series) -> str:
        for v in series.tolist():
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    agg = {"Game": "first"}
    if have_local:
        agg["StartTimeLocal"] = first_nonempty
    if have_utc:
        agg["StartTimeUTC"] = first_nonempty

    g = tmp.groupby("Game", as_index=False).agg(agg)

    if have_utc:
        g = g.sort_values("StartTimeUTC")
    elif have_local:
        g = g.sort_values("StartTimeLocal")

    st.subheader("Games & Start Times")
    st.dataframe(g, width="stretch", hide_index=True)


# ---------------------------------
# Smash Plays gating (strict)
# conf/matrix + EV% + (Drought>=2 OR Regression hot/gap)
# ---------------------------------
THR_CONF_DEFAULT = 70
THR_EV_DEFAULT = 0.0
THR_DROUGHT_DEFAULT = 0
THR_REG_GAP_DEFAULT = 0.0   # Reg_Gap_*10
THR_REG_HEAT_DEFAULT = 0.0  # Reg_Heat_*10

# HARD Board Auth Gate (do NOT override with UI sliders)
HARD_BOARD_DROUGHT_GATE = 2  # must be >=2 games drought
HARD_BOARD_HEAT_LEVELS = ("HOT", "DUE", "OVERDUE")  # at least HOT

# --- Market engines (data-verified Feb 2026) ---
POINTS_ENGINE_REG_GAP = 2.5  # Reg_Gap_P10
SOG_ENGINE_LINE_MAX = 2.5
SOG_ENGINE_REG_GAP_MIN = 2.6
SOG_ENGINE_REG_GAP_MAX = 4.3
ASSISTS_ENGINE_LINE = 0.5

def _truthy(v) -> bool:
    s = str(v).strip().lower()
    return s in ("1","true","t","yes","y","💰","✅")


def _num(v, default=0.0):
    try:
        if v is None:
            return float(default)
        if isinstance(v, str) and v.strip() == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)

def _get_first(row, *cols, default=""):
    for c in cols:
        try:
            if c in row and pd.notna(row.get(c)):
                return row.get(c)
        except Exception:
            continue
    return default

def _is_matrix_green(matrix_str: str) -> bool:
    """STRICT: accept only fully-green matrix values."""
    s = str(matrix_str or "").strip()
    if not s:
        return False
    return (s == "🟢") or (s.upper() == "GREEN")

def _bundle_for_market(row, market_key: str) -> dict:
    """Normalize column naming differences across tracker versions."""
    mk = str(market_key).lower().strip()

    def _first_num(*keys, default=0.0):
        return _num(_get_first(row, *keys, default=default))

    def _first_str(*keys, default=""):
        return str(_get_first(row, *keys, default=default) or "")

    # helper: compute EV% from model prob + american odds if not already provided
    def _ev_from_model_odds(model_pct: float, odds_amer: float) -> float:
        try:
            p = float(model_pct) / 100.0
            o = float(odds_amer)
            if p <= 0 or p >= 1 or o == 0:
                return 0.0
            if o > 0:
                b = o / 100.0
            else:
                b = 100.0 / abs(o)
            ev = p * b - (1 - p) * 1.0  # profit expectation per $1 risk
            return float(ev * 100.0)
        except Exception:
            return 0.0

    if mk in ("sog", "shots"):
        label = "SOG"
        conf = _first_num("Conf_SOG", "Conf_Shots", default=0)
        matrix = _first_str("Matrix_SOG", "Matrix_Shots", default="")
        # tracker variants: SOG_EVpct_over (preferred), SOG_EV%, EV_SOG
        ev = _first_num("SOG_EV%", "SOG_EVpct_over", "EV_SOG", default=0)
        model = _first_num("SOG_Model%", "Model%_SOG", "Model_SOG", default=0)
        odds = _first_num("SOG_Odds_Over", "Odds_SOG", "Odds_Amer_SOG", default=0)
        if ev == 0 and model and odds:
            ev = _ev_from_model_odds(model, odds)
        return {
            "label": label,
            "conf": conf,
            "matrix": matrix,
            "ev": ev,
            "model": model,
            "odds": odds,
            "line": _first_num("SOG_Line", "Line_SOG", default=0),
            "drought": _first_num("Drought_S", "Drought_SOG", default=0),
            # tracker variants: Reg_Gap_S10 or Reg_Gap_S, heat: Reg_Heat_S
            "reg_gap": _first_num("Reg_Gap_S10", "Reg_Gap_S", default=0),
            "reg_heat": _first_str("Reg_Heat_S", default=""),
            # U / mu variants (you called this the "U")
            "mu": _first_num("SOG_mu", "MU_SOG", "U_SOG", "U", "MU", default=0),
            "share_5v5": _first_num("5v5_Share", "Share_5v5", "ShotShare_5v5", default=0),
            "lsa10": _first_num("L10_Team_SA", "L10SA", "TeamSA10", "LSA10", default=0),
            "plays_ev": _truthy(_get_first(row, "Plays_EV_SOG", "Plays_EV_Shots", "Plays_EV_S", default=False)),
            "env_hdca": _first_num("opp_5v5_HDCA60", default=0),
            "opp_gaa": _first_num("Opp_GAA", default=0),
            "opp_sv": _first_num("Opp_SV", default=0),

        }

    if mk in ("assists", "a"):
        label = "ASSISTS"
        conf = _first_num("Conf_Assists", "Conf_A", default=0)
        matrix = _first_str("Matrix_Assists", "Matrix_A", default="")
        ev = _first_num("Assists_EV%", "Assists_EVpct_over", "A_EV%", "EV_Assists", default=0)
        model = _first_num("Assists_Model%", "Model%_Assists", default=0)
        odds = _first_num("Assists_Odds_Over", "Odds_Assists", default=0)
        if ev == 0 and model and odds:
            ev = _ev_from_model_odds(model, odds)
        return {
            "label": label,
            "conf": conf,
            "matrix": matrix,
            "ev": ev,
            "model": model,
            "odds": odds,
            "line": _first_num("Assists_Line", "Line_Assists", default=0),
            "ppixa": _first_num("PP_iXA60", "PP_iXA_60", default=0),
            "ppshare": _first_num("PP_TeamShare_pct", "PP_TeamShare%", default=0),
            "xga": _first_num("opp_5v5_xGA60", default=0),
            "goalie_weak": _first_num("Goalie_Weak", default=0),
            "opp_sv": _first_num("Opp_SV", default=0),
            "drought": _first_num("Drought_A", "Drought_Assists", default=0),
            "reg_gap": _first_num("Reg_Gap_A10", "Reg_Gap_A", default=0),
            "reg_heat": _first_str("Reg_Heat_A", default=""),
            "mu": _first_num("Assists_mu", "MU_Assists", "U_Assists", "U", "MU", default=0),
            "share_5v5": _first_num("5v5_Share", "Share_5v5", default=0),
            "lsa10": _first_num("L10_Team_SA", "L10SA", "TeamSA10", default=0),
            "plays_ev": _truthy(_get_first(row, "Plays_EV_Assists", "Plays_EV_A", default=False)),
            "pp_proof": int(_truthy(_get_first(row, "Assist_PP_Proof", default=False))),
            "env_hdca": _first_num("opp_5v5_HDCA60", default=0),
            "opp_gaa": _first_num("Opp_GAA", default=0),
            "opp_sv": _first_num("Opp_SV", default=0),

        }

    if mk in ("points", "pts", "p"):
        label = "POINTS"
        conf = _first_num("Conf_Points", "Conf_P", default=0)
        matrix = _first_str("Matrix_Points", "Matrix_P", default="")
        ev = _first_num("Points_EV%", "Points_EVpct_over", "P_EV%", "EV_Points", default=0)
        model = _first_num("Points_Model%", "Model%_Points", default=0)
        odds = _first_num("Points_Odds_Over", "Odds_Points", default=0)
        if ev == 0 and model and odds:
            ev = _ev_from_model_odds(model, odds)
        return {
            "label": label,
            "conf": conf,
            "matrix": matrix,
            "ev": ev,
            "model": model,
            "odds": odds,
            "line": _first_num("Points_Line", "Line_Points", default=0),
            "l10_rate": _first_num("L10_Rate_Points", default=0),
            "l10_diff": _first_num("L10_Diff_Points", default=0),
            "xga": _first_num("opp_5v5_xGA60", default=0),
            "hdca": _first_num("opp_5v5_HDCA60", default=0),
            "drought": _first_num("Drought_P", "Drought_Points", default=0),
            "reg_gap": _first_num("Reg_Gap_P10", "Reg_Gap_P", default=0),
            "reg_heat": _first_str("Reg_Heat_P", default=""),
            "mu": _first_num("Points_mu", "MU_Points", "U_Points", "U", "MU", default=0),
            "share_5v5": _first_num("5v5_Share", "Share_5v5", default=0),
            "lsa10": _first_num("L10_Team_SA", "L10SA", "TeamSA10", default=0),
            "plays_ev": _truthy(_get_first(row, "Plays_EV_Points", "Plays_EV_P", default=False)),
            "env_hdca": _first_num("opp_5v5_HDCA60", default=0),
            "opp_gaa": _first_num("Opp_GAA", default=0),
            "opp_sv": _first_num("Opp_SV", default=0),

        }

    if mk in ("goals", "goal", "g"):
        label = "GOALS"
        conf = _first_num("Conf_Goal", "Conf_Goals", "Conf_G", default=0)
        matrix = _first_str("Matrix_Goal", "Matrix_Goals", "Matrix_G", default="")
        ev = _first_num("Goal_EV%", "Goal_EVpct_over", "Goals_EVpct_over", "G_EV%", "EV_Goal", "EV_Goals", default=0)
        model = _first_num("Goal_Model%", "Goals_Model%", "Model%_Goal", "Model%_Goals", default=0)
        odds = _first_num("Goal_Odds_Over", "Goal_Odds", "Goals_Odds_Over", "Odds_Goal", "Odds_Goals", "Odds", "Odds_Amer", default=0)
        if ev == 0 and model and odds:
            ev = _ev_from_model_odds(model, odds)
        return {
            "label": label,
            "conf": conf,
            "matrix": matrix,
            "ev": ev,
            "model": model,
            "odds": odds,
            "line": _first_num("Goal_Line","Goals_Line","Goal_Line","Line_Goal","Line_Goals", default=0),
            "avg5_sog": _first_num("Avg5_SOG", "Avg5_Shots", default=0),
            "shotintent": _first_num("ShotIntent", default=0),
            "shotintent_pct": _first_num("ShotIntent_Pct", default=0),
            "xga": _first_num("opp_5v5_xGA60", default=0),
            "drought": _first_num("Drought_G", "Drought_Goal", "Drought_Goals", default=0),
            "reg_gap": _first_num("Reg_Gap_G10", "Reg_Gap_G", default=0),
            "reg_heat": _first_str("Reg_Heat_G", default=""),
            "mu": _first_num("Goals_mu", "MU_Goal", "MU_Goals", "U_Goal", "U_Goals", "U", "MU", default=0),
            "share_5v5": _first_num("5v5_Share", "Share_5v5", default=0),
            "lsa10": _first_num("L10_Team_SA", "L10SA", "TeamSA10", default=0),
        }

    # fallback
    return {
        "label": str(market_key).upper(),
        "conf": 0.0,
        "matrix": "",
        "ev": 0.0,
        "model": 0.0,
        "odds": 0.0,
        "line": 0.0,
        "drought": 0.0,
        "reg_gap": 0.0,
        "reg_heat": "",
        "mu": 0.0,
        "share_5v5": 0.0,
        "lsa10": 0.0,
    }


def _passes_smash(b: dict, thr_conf: int, thr_ev: float, thr_drought: int, thr_gap: float, thr_heat: float) -> bool:
    """HARD Board gate:
    - CONF must pass
    - MATRIX must be green (strict)
    - EV must pass
    - AND (Drought OR Regression) must pass
    """
    # CONF (hard)
    try:
        conf = int(float(b.get("conf", 0) or 0))
    except Exception:
        conf = 0
    if conf < int(thr_conf):
        return False

    # MATRIX (hard, strict green)
    matrix = str(b.get("matrix", "") or "").strip()
    if not _is_matrix_green(matrix):
        return False

    # EV (hard)
    try:
        ev = float(b.get("ev", 0.0) or 0.0)
    except Exception:
        ev = 0.0
    if ev < float(thr_ev):
        return False

    # SECONDARY: Drought OR Regression Heat (tight gate)
    # Rule: player must satisfy ONE of:
    #   - drought >= thr_drought
    #   - reg_heat in {HOT, DUE, OVERDUE} (enabled when thr_heat >= 1)
    try:
        drought = int(float(b.get("drought", 0) or 0))
    except Exception:
        drought = 0

    reg_heat = str(b.get("reg_heat", "") or "").strip().upper()
    heat_ok = reg_heat in HARD_BOARD_HEAT_LEVELS

    drought_ok = drought >= int(HARD_BOARD_DROUGHT_GATE)
    heat_pass = heat_ok  # at least HOT

    return drought_ok or heat_pass


def _passes_engine(b: dict) -> bool:
    """Market-pure Valhalla gates (IGNORE EV everywhere).

    These gates are intentionally strict. Anything not listed here is display-only.
    """
    label = str(b.get("label", "") or "").upper().strip()

    # MATRIX (hard for all Valhalla gates)
    if not _is_matrix_green(str(b.get("matrix", "") or "").strip()):
        return False

    # -------------------------
    # ASSISTS (0.5)
    # -------------------------
    if label == "ASSISTS":
        line = _num(b.get("line", 0), 0)
        conf = _num(b.get("conf", 0), 0)
        if abs(line - 0.5) > 1e-6:
            return False
        if conf < 80:
            return False
        return True

    # -------------------------
    # POINTS (0.5)
    # -------------------------
    if label == "POINTS":
        line = _num(b.get("line", 0), 0)
        if abs(line - 0.5) > 1e-6:
            return False

        # Reg-valid (your tested definition)
        heat = str(b.get("reg_heat", "") or "").strip().upper()
        heat_ok = heat in ("HOT", "DUE", "OVERDUE")
        gap_ok = _num(b.get("reg_gap", 0), 0) >= 2.5
        drt_ok = _num(b.get("drought", 0), 0) >= 2

        # Conf is supportive, but we keep a floor for Valhalla quality
        conf = _num(b.get("conf", 0), 0)
        if conf < 70:
            return False

        return bool(heat_ok or gap_ok or drt_ok)

    # -------------------------
    # GOALS (0.5)
    # -------------------------
    if label in ("GOALS","GOAL","ATG"):
        line = _num(b.get("line", 0), 0)
        conf = _num(b.get("conf", 0), 0)
        avg5 = _num(b.get("avg5_sog", 0), 0)
        if abs(line - 0.5) > 1e-6:
            return False
        if conf < 85:
            return False
        if avg5 < 3.4:
            return False
        return True

    # -------------------------
    # SOG (<=2.5) — keep conservative
    # -------------------------
    if label == "SOG":
        line = _num(b.get("line", 0), 0)
        if line > 2.5:
            return False
        # keep the proven sweet spot as the structural driver
        rg = _num(b.get("reg_gap", 0), 0)
        if not (float(SOG_ENGINE_REG_GAP_MIN) <= rg <= float(SOG_ENGINE_REG_GAP_MAX)):
            return False
        return True

    return False



def select_all_market_rows(row, thr_conf: int, thr_ev: float, thr_drought: int, thr_gap: float, thr_heat: float) -> list[dict]:
    """Return ALL market bundles that pass the hard gate (multi-market allowed).

    Ranking is DPS-first (AdjWin shrunk by n), then DPS n, then odds nudge.
    EV is NOT used for ordering (may still exist as a filter elsewhere).
    """
    cands = [
        _bundle_for_market(row, "sog"),
        _bundle_for_market(row, "assists"),
        _bundle_for_market(row, "points"),
        _bundle_for_market(row, "goal"),
    ]
    elig = [c for c in cands if _passes_engine(c)]

    # attach best DPS proc (presentation only)
    for b in elig:
        p = _probe_best_proc(str(b.get("label","") or "").upper().strip(), row)
        if p:
            b["dps_title"] = p["title"]
            b["dps_win"] = p["win"]
            b["dps_n"] = p["n"]
            b["dps_adj"] = p["adj"]
        else:
            b["dps_title"] = ""
            b["dps_win"] = 0.0
            b["dps_n"] = 0
            b["dps_adj"] = 0.0

    # sort strongest first (DPS AdjWin, then n, then odds (less-favorite / higher odds wins ties))
    elig.sort(
        key=lambda x: (
            float(x.get("dps_adj", 0.0) or 0.0),
            int(x.get("dps_n", 0) or 0),
            float(x.get("odds", 0.0) or 0.0),
        ),
        reverse=True,
    )
    return elig

def select_best_market_row(row, thr_conf: int, thr_ev: float, thr_drought: int, thr_gap: float, thr_heat: float):
    cands = [
        _bundle_for_market(row, "sog"),
        _bundle_for_market(row, "assists"),
        _bundle_for_market(row, "points"),
        _bundle_for_market(row, "goal"),
    ]
    elig = [c for c in cands if _passes_engine(c)]
    if not elig:
        return None
    elig.sort(key=lambda x: (x["ev"], x["conf"], x["model"]), reverse=True)
    return elig[0]

def _flames_from_heat(heat: str) -> str:
    h = str(heat or "").strip().upper()
    if h in ("OVERDUE", "BONKERS", "NUCLEAR"):
        return "🔥🔥🔥"
    if h in ("HOT", "DUE"):
        return "🔥🔥"
    return ""

def _count_markets_from_pills(html: str) -> int:
    s = str(html or "")
    hits = set()
    for key, tag in [("SOG", "SOG"), ("PTS", "PTS"), ("POINT", "PTS"), ("ASSIST", "AST"), ("AST", "AST"), ("GOAL", "G"), ("ATG", "G")]:
        if key in s.upper():
            hits.add(tag)
    return len(hits)

def _derive_badges(row: dict) -> tuple[str, str]:
    """Return (explosion_badge, critical_badge) based on *visible* multi-market signals.
    We intentionally avoid changing gates/math here; this is presentation only.
    """
    markets_html = row.get("Markets","")
    mcount = _count_markets_from_pills(markets_html)
    explosion = "🧨" if mcount >= 2 else ""
    tier = str(row.get("Tier_Tag","") or "")
    # Critical Strike: multi-market + STAR/ELITE + strong confidence on best market OR bonkers heat somewhere
    best_conf = 0
    try:
        best_conf = int(float(row.get("Best_Conf", 0) or 0))
    except Exception:
        best_conf = 0
    # bonkers if any heat column says OVERDUE
    heats = " ".join([str(row.get(c,"") or "") for c in ["Reg_Heat_P","Reg_Heat_S","Reg_Heat_A","Reg_Heat_G"]]).upper()
    bonkers = ("OVERDUE" in heats) or ("BONKERS" in heats)
    critical = "⚔️" if (mcount >= 2 and ("ELITE" in tier.upper() or "STAR" in tier.upper()) and (best_conf >= 85 or bonkers)) else ""
    return explosion, critical


def _engine_badge(mkt: str, r: dict) -> str:
    """Return an engine badge (✅ENG) if the market-specific engine criteria are met."""
    mk = str(mkt or "").strip().upper()
    try:
        if mk == "ASSISTS":
            ev_ok = bool(r.get("Plays_EV_Assists", False))
            line = float(r.get("Assists_Line", 0) or 0)
            mx = _is_matrix_green(str(r.get("Matrix_Assists", "") or ""))
            pp = bool(r.get("Assist_PP_Proof", False))
            return "✅ENG" if (ev_ok and mx and abs(line - 0.5) < 1e-6 and pp) else ""

        if mk == "POINTS":
            ev_ok = bool(r.get("Plays_EV_Points", False))
            line = float(r.get("Points_Line", 0) or 0)
            mx = _is_matrix_green(str(r.get("Matrix_Points", "") or ""))
            rg = _num(r.get("Reg_Gap_P10", 0), 0.0)
            return "✅ENG" if (ev_ok and mx and abs(line - 0.5) < 1e-6 and rg >= POINTS_ENGINE_REG_GAP) else ""

        if mk == "SOG":
            ev_ok = bool(r.get("Plays_EV_SOG", False))
            line = _num(r.get("SOG_Line", 0), 0.0)
            mx = _mat_green  # already computed above
            rg = _num(r.get("Reg_Gap_S10", 0), 0.0)
            in_band = (rg >= SOG_ENGINE_REG_GAP_MIN) and (rg <= SOG_ENGINE_REG_GAP_MAX)
            return "✅ENG" if (ev_ok and mx and line <= SOG_ENGINE_LINE_MAX and in_band) else ""
    except Exception:
        return ""
    return ""







def _render_badge_legend_inline() -> None:
    """Inline legend for 🧨 / ⚔️. Presentation-only."""
    st.markdown(
        """
        <div style="padding:10px 14px;border-radius:12px;
                    background:#f8f8f8;border:1px solid #ddd;
                    margin:6px 0 12px 0;font-size:14px;line-height:1.35;">
          <b>Badges</b><br>
          🧨 <b>Dynamite</b> — explosive ceiling edge (multi-market signal)<br>
          ⚔️ <b>Critical Strike</b> — STAR/ELITE + multi-market + strong confidence/heat alignment
        </div>
        """,
        unsafe_allow_html=True,
    )
def show_table(df: pd.DataFrame, cols: list[str], title: str):
    st.subheader(title)

    # Styler requires unique index + columns; filtering a df can preserve a non-unique index.
    # Also de-dupe the requested column list (some views may accidentally include repeats).
    df = df.copy().reset_index(drop=True)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # de-dupe requested cols while preserving order
    cols = list(dict.fromkeys(cols))

    # Ensure Time column is displayed right next to Game (if available)
    if "Game" in cols and "Time" in df.columns:
        if "Time" not in cols:
            gi = cols.index("Game")
            cols.insert(gi + 1, "Time")
        else:
            gi = cols.index("Game")
            ti = cols.index("Time")
            if abs(ti - gi) != 1:
                cols.pop(ti)
                gi = cols.index("Game")
                cols.insert(gi + 1, "Time")


    existing = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]

    if missing:
        with st.expander("Missing columns (safe to ignore)"):
            st.write(missing)

    styled = style_df(df, existing)

    # ✅ Option A: keeps your Styler colors
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        column_config=build_column_config(df, existing),
    )


# =========================
# APP
# =========================
inject_vengeance_css()
render_vengeance_banner()

st.title("⚔️The Warlord's NHL Prop Tool⚔️")
st.markdown(
    """
    <div style="
        padding: 14px 16px;
        border-radius: 14px;
        font-weight: 800;
        font-size: 22px;
        text-align: center;
        letter-spacing: 0.5px;
        background: #b30000;
        color: white;
        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
        margin-bottom: 12px;
    ">
        As winter ends, the war begins.
        Weapons forged by the gods. Warlords prepare to wage war on the books.
        Vengeance is coming — cook the books!
     
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Data source (no more forced uploads)
# -------------------------
# Optional manual upload (still supported)
uploaded = st.sidebar.file_uploader("Upload tracker CSV (optional)", type=["csv"], key="uploader_tracker_csv_sidebar")

# Preferred stable path written by nhl_edge.py
latest_stable = os.path.join(OUTPUT_DIR, "tracker_latest.csv")
latest_path = latest_stable if os.path.exists(latest_stable) else find_latest_tracker_csv(OUTPUT_DIR)
# If we ran the model this session, prefer that exact path (prevents reverting to yesterday on rerun)
if "latest_path_override" in st.session_state:
    _p = st.session_state.get("latest_path_override")
    if _p and os.path.exists(str(_p)):
        latest_path = str(_p)


# Quick-run inside Streamlit (works on Streamlit Cloud)
st.sidebar.markdown("---")
slate_date = st.sidebar.date_input("Slate date", value=datetime.now().date(), key="date_slate_date")
run_now = st.sidebar.button("Run / Refresh slate", help="Runs nhl_edge.py for the selected date and loads the fresh tracker.", key="btn_run_refresh_slate")

def _run_model_cached(d: date, code_stamp: float) -> str:
    # Import + reload so Streamlit Cloud picks up new engine code
    import importlib
    import nhl_edge
    importlib.reload(nhl_edge)
    return str(nhl_edge.build_tracker(d, debug=False))

source = None
if uploaded is not None:
    source = "upload"
    df = pd.read_csv(uploaded)
else:
    # If user presses run, generate fresh tracker.
    if run_now:
        with st.spinner("Running model…"):
            try:
                # Cache-buster: if nhl_edge.py changed, re-run the model
                try:
                    engine_path = os.path.join(os.path.dirname(__file__), 'nhl_edge.py') if '__file__' in globals() else 'nhl_edge.py'
                    code_stamp = os.path.getmtime(engine_path) if os.path.exists(engine_path) else 0.0
                except Exception:
                    code_stamp = 0.0
                latest_path = _run_model_cached(slate_date, code_stamp)
                st.session_state["latest_path_override"] = str(latest_path)
            except Exception as e:
                st.error(f"Model run failed: {e}")
                st.stop()

    source = "latest"
    if latest_path is None or not os.path.exists(str(latest_path)):
        st.warning(
            "No tracker CSV found yet. Click **Run / Refresh slate** in the sidebar (or run `python nhl_edge.py` locally)."
        )
        st.stop()

    df = load_csv(str(latest_path))

# -------------------------
# FIX: Styler requires unique index + columns
# -------------------------
# 1) reset index (unique)
df = df.reset_index(drop=True)

# 2) de-dupe column names (keep first occurrence)
if df.columns.duplicated().any():
    dupes = df.columns[df.columns.duplicated()].tolist()
    st.warning(f"Duplicate columns detected and removed: {dupes}")
    df = df.loc[:, ~df.columns.duplicated()].copy()


# -------------------------
# Ensure injury columns exist (older CSV safe)
# -------------------------
if "Injury_Badge" not in df.columns:
    df["Injury_Badge"] = ""
if "Injury_Status" not in df.columns:
    df["Injury_Status"] = "Healthy"
# -------------------------
# Ensure drought columns exist (older CSV safe)
# -------------------------
for c in ["Best_Drought", "Drought_P", "Drought_A", "Drought_G", "Drought_SOG"]:
    if c not in df.columns:
        df[c] = ""
# -------------------------
# Ensure tier columns exist (older CSV safe)
# -------------------------
if "Talent_Tier" not in df.columns:
    df["Talent_Tier"] = "NONE"
if "Tier_Tag" not in df.columns:
    df["Tier_Tag"] = ""

# (Removed global Tier_Tag overwrite — keep tracker Tier_Tag as-is)

# -------------------------
# Ensure TEAM GF gate columns exist (older CSV safe)
# -------------------------
if "Team_GF_Gate" not in df.columns:
    df["Team_GF_Gate"] = True  # default "passes" if old CSV
if "Team_GF_Avg_L5" not in df.columns:
    df["Team_GF_Avg_L5"] = np.nan
if "Team_GF_L5" not in df.columns:
    df["Team_GF_L5"] = np.nan

# Create badge if missing (or overwrite if you want consistency)
if "GF_Gate_Badge" not in df.columns:
    # Normalize gate to bool (handles True/False, 1/0, "true"/"false")
    gate_bool = df["Team_GF_Gate"].astype(str).str.strip().str.lower().isin(["true","1","yes","y","t"])
    df["GF_Gate_Badge"] = np.where(
        gate_bool,
        "",  # passed gate = no badge
        "⛔ GF GATE"  # failed gate badge
    )




df = add_ui_columns(df)

# =========================
# ODDS / EV UI DERIVED COLS (readable)
# =========================
# Convert p_model / p_imp into human % columns and create a global 💰 marker.
for m in ["Points","GOALS (0.5)","Assists","ATG","SOG"]:
    pm = f"{m}_p_model_over"
    pi = f"{m}_p_imp_over"
    ev = f"{m}_EVpct_over"
    if pm in df.columns:
        df[f"{m}_Model%"] = (pd.to_numeric(df[pm], errors="coerce") * 100).round(1)
    if pi in df.columns:
        df[f"{m}_Imp%"] = (pd.to_numeric(df[pi], errors="coerce") * 100).round(1)
    if ev in df.columns:
        df[f"{m}_EV%"] = pd.to_numeric(df[ev], errors="coerce").round(1)

# --- Back-compat: some trackers provide EV% but not Plays_EV_* flags.
# If Plays_EV_* columns are missing, derive them from *_EVpct_over (>0) so 🔒 works.
if "Plays_EV_Points" not in df.columns and "Points_EVpct_over" in df.columns:
    df["Plays_EV_Points"] = pd.to_numeric(df["Points_EVpct_over"], errors="coerce").fillna(-999) > 0
if "Plays_EV_SOG" not in df.columns and "SOG_EVpct_over" in df.columns:
    df["Plays_EV_SOG"] = pd.to_numeric(df["SOG_EVpct_over"], errors="coerce").fillna(-999) > 0
if "Plays_EV_Assists" not in df.columns and "Assists_EVpct_over" in df.columns:
    df["Plays_EV_Assists"] = pd.to_numeric(df["Assists_EVpct_over"], errors="coerce").fillna(-999) > 0
if "Plays_EV_ATG" not in df.columns and "ATG_EVpct_over" in df.columns:
    df["Plays_EV_ATG"] = pd.to_numeric(df["ATG_EVpct_over"], errors="coerce").fillna(-999) > 0
if "Plays_EV_Goal" not in df.columns and "Goal_EVpct_over" in df.columns:
    df["Plays_EV_Goal"] = pd.to_numeric(df["Goal_EVpct_over"], errors="coerce").fillna(-999) > 0

# Replace Plays_EV_* booleans with a 💰 icon for readability (keep the original name)
for c in ["Plays_EV_Points","Plays_EV_Goal","Plays_EV_Assists","Plays_EV_ATG","Plays_EV_SOG"]:
    if c in df.columns:
        df[c] = df[c].apply(lambda x: "💰" if bool(x) else "")

# Global 💰 if any EV-play is active
_ev_play_cols = [c for c in ["Plays_EV_Points","Plays_EV_Goal","Plays_EV_Assists","Plays_EV_ATG","Plays_EV_SOG"] if c in df.columns]
if _ev_play_cols:
    df["💰"] = (df[_ev_play_cols].astype(str).apply(lambda r: any(v=="💰" for v in r), axis=1)).map(lambda x: "💰" if x else "")
else:
    # Keep existing 💰 if tracker provided it and no Plays_EV_* columns are present
    if "💰" not in df.columns:
        df["💰"] = ""


# =========================
# BETTING DISPLAY CLEANUP
# =========================
# Snap all *_Line columns to .0/.5 so you never see ugly 2.49999997 style floats.
for c in list(df.columns):
    if c.endswith("_Line") or c == "Line":
        df[c] = pd.to_numeric(df[c], errors="coerce").apply(snap_half)

# American odds should be whole numbers (no decimals)
for c in list(df.columns):
    if c.endswith("_Odds_Over") or c == "Odds":
        df[c] = pd.to_numeric(df[c], errors="coerce").apply(snap_int)


# --- slate size (safe)
try:
    slate_games = int(df["Game"].nunique())
except Exception:
    slate_games = 8
def _tier_color(conf):
    try:
        x = float(conf)
    except Exception:
        return "red"
    if x >= 76:
        return "green"
    if x >= 65:
        return "yellow"
    if x >= 55:
        return "blue"
    return "red"
def _green_conf_threshold(market: str, slate_games: int) -> int:
    # Normalize market aliases
    m = market.strip()

    if m.upper() in ("GOALS (0.5)", "GOAL 1+", "ATG", "ANYTIME GOAL"):
        m = "Goal"

    # Hard floor: GOALS earned-green starts at 85 (fixed, not slate-size dependent)
    if m == "Goal":
        return 85
    if slate_games >= 8:
        return {"SOG": 75, "Points": 70, "Goal": 85, "Assists": 80}[m]
    elif slate_games >= 5:
        return {"SOG": 75, "Points": 70, "Goal": 85, "Assists": 80}[m]
    else:
        return {"SOG": 75, "Points": 70, "Goal": 85, "Assists": 80}[m]




# --- Earned greens (match YOUR columns)
thr_s = _green_conf_threshold("SOG", slate_games)
thr_p = _green_conf_threshold("Points", slate_games)
thr_s = _green_conf_threshold("SOG", slate_games)
# =========================
# GOAL — earned green (v2 proof-count + tier-aware drought)
# =========================

thr_g = _green_conf_threshold("GOALS (0.5)", slate_games)

# numeric safety
for c in ["Conf_Goal", "iXG%", "Med10_SOG", "Avg5_SOG", "Goalie_Weak", "Opp_DefWeak", "Reg_Gap_G10", "Drought_G"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

tier_g = safe_str(df, "Talent_Tier", "NONE").str.upper()
is_star_g = tier_g.isin(["ELITE", "STAR"])

# Tier-aware drought trigger:
# ELITE: >=2, STAR: >=3, NONE: >=4
goal_drought_ok = (
    ((tier_g == "ELITE") & (safe_num(df, "Drought_G", 0) >= 2)) |
    ((tier_g == "STAR")  & (safe_num(df, "Drought_G", 0) >= 3)) |
    (~tier_g.isin(["ELITE", "STAR"]) & (safe_num(df, "Drought_G", 0) >= 4))
)

# Proofs
proof_ixg = (safe_num(df, "iXG%", 0) >= 92)
proof_volume = (
    (safe_num(df, "Med10_SOG", 0) >= 3.0) |
    (safe_num(df, "Avg5_SOG", 0) >= 3.0)
)
proof_env = (
    (safe_num(df, "Goalie_Weak", 0) >= 70) |
    (safe_num(df, "Opp_DefWeak", 0) >= 70)
)
proof_due = (
    (safe_str(df, "Reg_Heat_G", "").str.strip().str.upper() == "HOT") |
    (safe_num(df, "Reg_Gap_G10", 0) >= 0.80) |
    goal_drought_ok
)

goal_proofs = pd.concat([proof_ixg, proof_volume, proof_env, proof_due], axis=1).fillna(False)
df["Goal_ProofCount"] = goal_proofs.sum(axis=1)

needed_g = np.where(is_star_g, 2, 3)

df["Green_Goal"] = (
    (safe_num(df, "Conf_Goal", 0) >= thr_g)
    & (safe_str(df, "Matrix_Goal", "").str.strip().str.lower() == "green")
    & (df["Goal_ProofCount"] >= needed_g)
)

# optional: debug why
def _goal_why(r):
    reasons = []
    if _get(r, "iXG%", 0) >= 92:
        reasons.append("iXG")
    if (_get(r, "Med10_SOG", 0) >= 3.0) or (_get(r, "Avg5_SOG", 0) >= 3.0):
        reasons.append("VOL")
    if (_get(r, "Goalie_Weak", 0) >= 70) or (_get(r, "Opp_DefWeak", 0) >= 70):
        reasons.append("ENV")
    if str(_get(r, "Reg_Heat_G", "")).strip().upper() == "HOT" or _get(r, "Reg_Gap_G10", 0) >= 0.80:
        reasons.append("DUE")
    if _get(r, "Drought_G", 0) >= 2:
        reasons.append("DRT")
    return ",".join(reasons)

df["Goal_Why"] = ""
m = df["Green_Goal"].fillna(False)
df.loc[m, "Goal_Why"] = df.loc[m].apply(_goal_why, axis=1)


sog_volume_proof = (
    (safe_num(df, "Med10_SOG", 0) >= 3.0)
    | (safe_num(df, "Avg5_SOG", 0) >= 3.0)
)

# Regression/timing engine path:
# Your new finding: Drought_SOG == 1 is a meaningful "right now" timing state,
# and should allow SOG to be playable even when ShotIntent_Pct / volume proof aren't there yet.
sog_regression_proof = (
    (safe_num(df, "Drought_SOG", safe_num(df, "Drought_S", 0)) == 1)
)

df["Green_SOG"] = (
    (safe_num(df, "Conf_SOG", 0) >= thr_s)
    & (safe_str(df, "Matrix_SOG", "").str.strip().str.lower() == "green")
    & (
        (safe_num(df, "ShotIntent_Pct", 0) >= 90)
        | sog_volume_proof
        | sog_regression_proof
    )
)



# Make SOG usable everywhere (market views + smash picks)
df["Plays_SOG"] = df["Green_SOG"].fillna(False)

# Optional: why string (helps debugging + "Why it fires" on SOG)
def _sog_why(r):
    reasons = []
    # Always call out when it is an earned/matrix green
    if str(_get(r, "Matrix_SOG", "")).strip().lower() == "green":
        reasons.append("MATRIX")
    if _get(r, "ShotIntent_Pct", 0) >= 90:
        reasons.append("INT")
    if _get(r, "Med10_SOG", 0) >= 3.0 or _get(r, "Avg5_SOG", 0) >= 3.0:
        reasons.append("VOL")
    if _get(r, "iXG%", 0) >= 90:
        reasons.append("iXG")
    if _get(r, "Goalie_Weak", 0) >= 70 or _get(r, "Opp_DefWeak", 0) >= 70:
        reasons.append("ENV")
    if _get(r, "Reg_Gap_S10", 0) >= 1.0 or str(_get(r, "Reg_Heat_S", "")).upper() in ["HOT", "DUE"]:
        reasons.append("REG")
    d_sog = _get(r, "Drought_SOG", None)
    if d_sog is None or d_sog == "":
        d_sog = _get(r, "Drought_S", 0)

    try:
        d_sog = float(d_sog)
    except Exception:
        d_sog = 0.0

    if d_sog == 1:
        reasons.append("DRT1")
    elif d_sog >= 3:
        reasons.append("DRT")

    return ",".join(reasons)


# Preserve any existing SOG_Why from tracker, but backfill when blank (common for earned greens)
if "SOG_Why" not in df.columns:
    df["SOG_Why"] = ""
m = df["Green_SOG"].fillna(False) & (df["SOG_Why"].isna() | (df["SOG_Why"].astype(str).str.strip() == ""))
df.loc[m, "SOG_Why"] = df.loc[m].apply(_sog_why, axis=1)


# =========================
# POINTS — earned green (v3 proof-count, more accurate)
# =========================

thr_p = _green_conf_threshold("Points", slate_games)

# numeric safety
for c in [
    "Conf_Points",
    "iXG%", "iXA%",
    "Med10_SOG", "Avg5_SOG",
    "Goalie_Weak", "Opp_DefWeak",
        "Opp_SA_Avg_L10", "Opp_GA_Avg_L10",
    "team_5v5_xGF60_pct",
    "Reg_Gap_P10", "Drought_P",
    "TOI_Pct",
    "Assist_Volume", "i5v5_primaryAssists60",
]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ---- Proofs (4 lanes) ----
# 1) Finisher involvement
proof_finisher = (
    (safe_num(df, "iXG%", 0) >= 90)
    | (safe_num(df, "Med10_SOG", 0) >= 3.0)
    | (safe_num(df, "Avg5_SOG", 0) >= 3.0)
)

# 2) Playmaking involvement
proof_playmaker = (
    (safe_num(df, "iXA%", 0) >= 90)
    | (safe_num(df, "Assist_Volume", 0) >= 6)
    | (safe_num(df, "i5v5_primaryAssists60", 0) >= 0.50)
)

# 3) Environment (you need events to get points)
proof_env = (
    (safe_num(df, "team_5v5_xGF60_pct", 0) >= 65)
    | (safe_num(df, "Goalie_Weak", 0) >= 70)
    | (safe_num(df, "Opp_DefWeak", 0) >= 70)
)

# 4) Due lane (regression/drought)
proof_due = (
    (safe_str(df, "Reg_Heat_P", "").str.strip().str.upper() == "HOT")
    | (safe_num(df, "Reg_Gap_P10", 0) >= 1.25)
    | (safe_num(df, "Drought_P", 0) >= 3)
)

points_proofs = pd.concat(
    [proof_finisher, proof_playmaker, proof_env, proof_due],
    axis=1
).fillna(False)

df["Points_ProofCount"] = points_proofs.sum(axis=1)

# Tier-aware gate (ELITE/STAR can pass with 2 proofs; others need 3)
tier = safe_str(df, "Talent_Tier", "NONE").str.upper()
is_star = tier.isin(["ELITE", "STAR"])

needed = np.where(is_star, 2, 3)

df["Green_Points"] = (
    (safe_num(df, "Conf_Points", 0) >= thr_p)
    & (safe_str(df, "Matrix_Points", "").str.strip().str.lower() == "green")
    & (df["Points_ProofCount"] >= needed)
)

# Make Points usable everywhere + revive 🔥
df["Plays_Points"] = df["Green_Points"].fillna(False)
# refresh 🔥 now that Plays_Points is defined in-streamlit
df["🔥"] = df["Plays_Points"].map(lambda x: "🔥" if bool(x) else "")


# Optional: why string (helps debugging)
def _points_why(r):
    reasons = []
    if _get(r, "iXG%", 0) >= 90 or _get(r, "Med10_SOG", 0) >= 3.0 or _get(r, "Avg5_SOG", 0) >= 3.0:
        reasons.append("FIN")
    if _get(r, "iXA%", 0) >= 90 or _get(r, "Assist_Volume", 0) >= 6 or _get(r, "i5v5_primaryAssists60", 0) >= 0.50:
        reasons.append("PLY")
    if _get(r, "team_5v5_xGF60_pct", 0) >= 65 or _get(r, "Goalie_Weak", 0) >= 70 or _get(r, "Opp_DefWeak", 0) >= 70:
        reasons.append("ENV")
    if str(_get(r, "Reg_Heat_P", "")).strip().upper() == "HOT" or _get(r, "Reg_Gap_P10", 0) >= 1.25 or _get(r, "Drought_P", 0) >= 3:
        reasons.append("DUE")
    return ",".join(reasons)

df["Points_Why"] = ""
mask = df["Green_Points"].fillna(False)
df.loc[mask, "Points_Why"] = df.loc[mask].apply(_points_why, axis=1)

df["Color_SOG"] = safe_num(df, "Conf_SOG", 0).apply(_tier_color) if "Conf_SOG" in df.columns else "red"
df["Color_Points"] = safe_num(df, "Conf_Points", 0).apply(_tier_color) if "Conf_Points" in df.columns else "red"
df["Color_Goal"] = safe_num(df, "Conf_Goal", 0).apply(_tier_color) if "Conf_Goal" in df.columns else "red"


# =========================
# ASSISTS — earned green rule (v1 FINAL)  ✅ ADDED HERE
# =========================
# Ensure required columns exist (older CSV safe)
if "Conf_Assists" not in df.columns:
    df["Conf_Assists"] = 0
if "Matrix_Assists" not in df.columns:
    df["Matrix_Assists"] = ""
if "iXA%" not in df.columns:
    df["iXA%"] = np.nan
if "v2_player_stability" not in df.columns:
    df["v2_player_stability"] = np.nan
if "team_5v5_xGF60_pct" not in df.columns:
    df["team_5v5_xGF60_pct"] = np.nan
if "Assist_Volume" not in df.columns:
    df["Assist_Volume"] = np.nan
if "i5v5_primaryAssists60" not in df.columns:
    df["i5v5_primaryAssists60"] = np.nan

# Newer trackers
if "Assists_Line" not in df.columns:
    df["Assists_Line"] = np.nan
if "PP_iXA60" not in df.columns:
    df["PP_iXA60"] = np.nan
if "Team_GF_L5" not in df.columns:
    df["Team_GF_L5"] = np.nan

# Optional columns
if "Talent_Tier" not in df.columns:
    df["Talent_Tier"] = ""
if "Plays_Assists" not in df.columns:
    df["Plays_Assists"] = False

# numeric safety
for c in ["iXA%", "Conf_Assists", "v2_player_stability", "team_5v5_xGF60_pct", "Assist_Volume", "i5v5_primaryAssists60"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["Assist_ProofCount"] = 0
df["Assist_Why"] = ""

proof_ixA = (df["iXA%"] >= 92)
proof_v2 = (df["v2_player_stability"] >= 65)
proof_team = (df["team_5v5_xGF60_pct"] >= 65)
proof_vol = (
    (df["Assist_Volume"] >= 6)
    | (df["i5v5_primaryAssists60"] >= 0.50)
)

proofs = pd.concat([proof_ixA, proof_v2, proof_team, proof_vol], axis=1).fillna(False)
df["Assist_ProofCount"] = proofs.sum(axis=1)

tier = df["Talent_Tier"].astype(str).str.upper()
is_star = tier.isin(["ELITE", "STAR"])

# Base shell: Assists_Line=0.5 | Matrix Green | Conf ≥ 80 (EV ignored)
assists_green_earned = (
    (safe_str(df, "Matrix_Assists", "").str.strip().str.lower() == "green")
    & (safe_num(df, "Conf_Assists", 0) >= 80)
    & (np.isclose(safe_num(df, "Assists_Line", 0).astype(float), 0.5))
)

df["Plays_Assists"] = assists_green_earned.fillna(False)
def _assist_why(r):
    reasons = []
    if _get(r, "iXA%", 0) >= 92:
        reasons.append("iXA")
    if _get(r, "v2_player_stability", 0) >= 65:
        reasons.append("v2")
    if _get(r, "team_5v5_xGF60_pct", 0) >= 65:
        reasons.append("xGF")
    if (_get(r, "Assist_Volume", 0) >= 6) or (_get(r, "i5v5_primaryAssists60", 0) >= 0.50):
        reasons.append("VOL")
    return ",".join(reasons)

df.loc[assists_green_earned, "Assist_Why"] = df.loc[assists_green_earned].apply(_assist_why, axis=1)

# append Play_Tag
df.loc[assists_green_earned, "Play_Tag"] = np.where(
    df.loc[assists_green_earned, "Play_Tag"].astype(str).str.len() > 0,
    df.loc[assists_green_earned, "Play_Tag"].astype(str) + " | 🅰️ ASSISTS EARNED",
    "🅰️ ASSISTS EARNED"
)

df["Color_Assists"] = safe_num(df, "Conf_Assists", 0).apply(_tier_color)
df["Green_Assists"] = df["Plays_Assists"].fillna(False)
# =========================
# 🔥 GLOBAL PLAY FLAG (any market)
# =========================
df["🔥"] = (
    df.get("Plays_Points", False).fillna(False)
    | df.get("Plays_Assists", False).fillna(False)
    | df.get("Green_SOG", False).fillna(False)
    | df.get("Green_Goal", False).fillna(False)
).map(lambda x: "🔥" if bool(x) else "")



# Header info
left, right = st.columns([3, 2])
with left:
    st.caption(f"Source: **{source}**")
    if source == "latest":
        st.caption(f"Loaded: **{os.path.basename(latest_path)}**")
with right:
    if "Date" in df.columns:
        st.caption(f"Date: **{df['Date'].iloc[0]}**")
    st.caption(f"Rows: **{len(df)}**")

with st.expander("Debug: loaded columns"):
    st.write(list(df.columns))

# Navigation
page = st.sidebar.radio(
    "Page",
    ["Board", "Points", "Assists", "SOG", "GOALS (0.5)", "Power Play", "🧪 Dagger Lab", "🪜 Ladder Alerts", "Guide", "Ledger", "Raw CSV", "📟 Calculator", "🧾 Log Bet"],
    index=0,
    format_func=lambda x: {
        "Points": "Points (🛡️ Tank)",
        "Assists": "Assists (🪄 Support)",
        "SOG": "SOG (🌿 Jungle)",
        "GOALS (0.5)": "GOALS (0.5) (⚔️ Carry)",
    }.get(x, x)
)

df_f = filter_common(df)

# NOTE: Board gating is applied ONLY inside the Board page.
# We do not shrink the global dataframe for other pages.
# Show slate times table
show_games_times(df_f)


# =========================
# BOARD
# =========================
if page == "Board":

    st.markdown(
        """
        <div style="padding:10px 14px;border-radius:12px;
                    background:#f8f8f8;border:1px solid #ddd;
                    margin-bottom:12px;font-size:14px;">
          <b>Board Signals</b><br>
          🧨 <b>Dynamite</b> — explosive ceiling edge (high-impact, volatile)<br>
          ⚔️ <b>Critical Strike</b> — precision edge with multiple proofs aligned
        </div>
        """,
        unsafe_allow_html=True
    )


    # -------------------------
    # Board market color theme (Ninja palette)
    # Purple = Assists, Dark Blue = Points, Green = SOG, Red = Goals
    # (Board is already gated; colors are MARKET identity, not quality.)
    # -------------------------
    st.markdown("""
    <style>
      .wl-board-legend{
        display:flex; flex-wrap:wrap; gap:8px; align-items:center;
        padding:10px 12px;
        border:1px solid rgba(255,255,255,0.10);
        border-radius:14px;
        background: rgba(255,255,255,0.03);
        margin: 6px 0 10px 0;
      }
      .wl-board-pill{
        display:inline-flex; align-items:center; gap:8px;
        padding:6px 10px;
        border-radius:999px;
        border:1px solid rgba(255,255,255,0.12);
        font-size:12px;
        font-weight:800;
        letter-spacing:.2px;
      }
      .wl-board-dot{ width:10px; height:10px; border-radius:50%; display:inline-block; }

      .wl-board-card{
        font-size: 15px;

        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 12px 14px;
        margin-bottom: 10px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.35);
      }
      .wl-accent-purple{ background: rgba(168,85,247,0.18); border-left: 5px solid #a855f7; }
      .wl-accent-blue{ background: rgba(59,130,246,0.18); border-left: 5px solid #3b82f6; }  { border-left: 5px solid #0b1b3a; }
      .wl-accent-orange{ background: rgba(34,197,94,0.18); border-left: 5px solid #22c55e; }
      .wl-accent-red{ background: rgba(239,68,68,0.18); }   { border-left: 5px solid #ef4444; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
      <div class="wl-board-legend">
        <span class="wl-board-pill"><span class="wl-board-dot" style="background:#a855f7"></span> Assists</span>
        <span class="wl-board-pill"><span class="wl-board-dot" style="background:#0b1b3a"></span> Points</span>
        <span class="wl-board-pill"><span class="wl-board-dot" style="background:#22c55e"></span> SOG</span>
        <span class="wl-board-pill"><span class="wl-board-dot" style="background:#ef4444"></span> Goals</span>
      </div>
    """, unsafe_allow_html=True)

    def _board_market_accent(best_market: str) -> str:
        bm0 = str(best_market or "").lower()
        if "assist" in bm0:
            return "wl-accent-purple"
        if "point" in bm0:
            return "wl-accent-blue"
        if "sog" in bm0 or "shot" in bm0:
            return "wl-accent-green"
        if "goal" in bm0:
            return "wl-accent-red"
        return ""

    # -------------------------
    # Board filters (beta polish) — presentation only
    # -------------------------

    df_board_src = df_f.copy()

    # Tier gate: STAR/ELITE only (robust; fall back if Tier_Tag is blank)
    _mask = None
    if "Tier_Tag" in df_board_src.columns:
        _t = df_board_src["Tier_Tag"].astype(str).str.upper()
        _mask = _t.str.contains("STAR") | _t.str.contains("ELITE")

    if (_mask is None or int(_mask.sum()) == 0) and "Talent_Tier" in df_board_src.columns:
        _t2 = df_board_src["Talent_Tier"].astype(str).str.upper()
        _mask = _t2.str.contains("STAR") | _t2.str.contains("ELITE")

    if (_mask is None or int(_mask.sum()) == 0) and "Tier" in df_board_src.columns:
        _t3 = df_board_src["Tier"].astype(str).str.upper()
        _mask = _t3.str.contains("STAR") | _t3.str.contains("ELITE")

    if _mask is not None:
        df_board_src = df_board_src[_mask].copy()

    best_rows = []

    for _, _r in df_board_src.iterrows():
        passing = select_all_market_rows(_r, 0, 0.0, 0, 0.0, 0.0)
        if not passing:
            continue
        for b in passing:
            # Defensive hard-enforcement (board safety net)
            _lbl = str(b.get("label","") or "").upper().strip()
            if _lbl in ("GOALS","GOAL","ATG"):
                _conf = _num(b.get("conf",0), 0)
                _avg5 = _num(b.get("avg5_sog",0), 0)
                _line = _num(b.get("line",0), 0)
                if abs(_line - 0.5) > 1e-6 or _conf < 85 or _avg5 < 3.4:
                    continue
            rr = _r.copy()
            rr["Best_Market"] = b["label"]
            rr["Best_Conf"] = b["conf"]
            rr["Best_EV%"] = b["ev"]
            rr["Best_Model%"] = b["model"]
            rr["Best_Line"] = b["line"]
            rr["Best_Odds"] = b.get("odds", _r.get("Odds", 0))
            rr["Best_Drought"] = b["drought"]
            rr["Best_Reg_Gap10"] = b["reg_gap"]
            rr["Best_Reg_Heat10"] = b["reg_heat"]
            rr["DPS_Title"] = b.get("dps_title","")
            rr["DPS_Win"] = b.get("dps_win",0.0)
            rr["DPS_N"] = b.get("dps_n",0)
            rr["DPS_Adj"] = b.get("dps_adj",0.0)
            best_rows.append(rr)

    df_board = pd.DataFrame(best_rows) if best_rows else df_board_src.iloc[0:0]
    df_b = sort_board(df_board)

    board_cols = [
        "Game",
        "Player", "Pos",
        "Tier_Tag",
        "Markets",
                "LOCK",
        "Best_Market",
        "Best_Conf",
        "🔥",
        "iXG%", "iXA%",
        "Goalie_Weak", "Opp_DefWeak",
        "Opp_Goalie", "Opp_SV", "Opp_GAA",
        "Matrix_Points", "Conf_Points", "Reg_Heat_P", "Reg_Gap_P10",
        "Matrix_SOG", "Conf_SOG", "Reg_Heat_S", "Reg_Gap_S10",
        "Matrix_Goal", "Conf_Goal", "Reg_Heat_G", "Reg_Gap_G10",
        "Matrix_Assists", "Conf_Assists", "Reg_Heat_A", "Reg_Gap_A10",
        "Line", "Odds", "Result",
    ]

    

    # Build Markets pills + best-market EV signal for Board

    

    df_b["Markets"] = df_b.apply(build_markets_pills, axis=1)

    

    _ev_lock = df_b.apply(board_best_market_ev, axis=1, result_type="expand")
    # Robustly normalize apply() output across pandas versions / empty / 1-row cases
    if isinstance(_ev_lock, pd.DataFrame):
        _c0 = _ev_lock.iloc[:, 0] if _ev_lock.shape[1] > 0 else ""
        _c1 = _ev_lock.iloc[:, 1] if _ev_lock.shape[1] > 1 else ""
    elif isinstance(_ev_lock, pd.Series):
        _c0 = _ev_lock.apply(lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else "")
        _c1 = _ev_lock.apply(lambda x: x[1] if isinstance(x, (list, tuple)) and len(x) > 1 else "")
    else:
        _c0, _c1 = "", ""
    
    df_b["EV_Signal"] = _c0
    df_b["EV_Source"] = _c1
    df_b["LOCK"] = _c1

    


    


    # === PICK SHEET (UI ONLY) ===
    st.subheader("🧾 Pick Sheet — signals-first")
    cA, cB, cC, cD = st.columns(4)
    with cA:
        st.metric("Players", int(len(df_b)))
    with cB:
        st.metric("Locks", int((df_b["LOCK"].astype(str).str.len() > 0).sum()) if "LOCK" in df_b.columns else 0)
    with cC:
        st.metric("+EV", int((df_b["EV_Signal"].astype(str).str.contains("💰")).sum()) if "EV_Signal" in df_b.columns else 0)
    with cD:
        st.metric("Top Conf", float(df_b["Best_Conf"].max()) if "Best_Conf" in df_b.columns else 0.0)
    # === Board filter set (beta default) ===
    st.sidebar.subheader("Board Filters (DPS-first)")
    market_sel = st.sidebar.multiselect("Market", ["POINTS","ASSISTS","SOG","GOALS"], default=["POINTS","ASSISTS","SOG","GOALS"], key="board_mkt_sel")
    # Lines available depend on the underlying slate
    line_vals = sorted([x for x in pd.unique(pd.to_numeric(df_b.get("Best_Line", pd.Series([])), errors="coerce")) if not pd.isna(x)])
    line_sel = st.sidebar.multiselect("Line", line_vals, default=line_vals, key="board_line_sel") if len(line_vals) else []
    move_vals = sorted([x for x in pd.unique(df_b.get("DPS_Title", pd.Series([])).astype(str)) if x and x != "nan"])
    move_sel = st.sidebar.multiselect("Move / Tier", move_vals, default=move_vals, key="board_move_sel") if len(move_vals) else []
    min_win = float(st.sidebar.slider("Min DPS win%", 0.0, 100.0, 55.0, 0.5, key="board_min_win"))
    min_n = int(st.sidebar.number_input("Min DPS n", min_value=0, max_value=500, value=20, step=1, key="board_min_n"))
    max_fav_odds = int(st.sidebar.number_input("Max favorite odds (e.g. -200)", min_value=-1000, max_value=300, value=-200, step=5, key="board_max_fav"))
    q = st.sidebar.text_input("Search", value="", key="board_search").strip().lower()

    df_b_filt = df_b.copy()
    if market_sel:
        df_b_filt = df_b_filt[df_b_filt["Best_Market"].astype(str).str.upper().isin([m.upper() for m in market_sel])]
    if line_sel:
        df_b_filt = df_b_filt[pd.to_numeric(df_b_filt.get("Best_Line", 0), errors="coerce").isin(line_sel)]
    if move_sel:
        df_b_filt = df_b_filt[df_b_filt.get("DPS_Title","").astype(str).isin(move_sel)]
    df_b_filt = df_b_filt[pd.to_numeric(df_b_filt.get("DPS_Win", 0), errors="coerce").fillna(0.0) >= min_win]
    df_b_filt = df_b_filt[pd.to_numeric(df_b_filt.get("DPS_N", 0), errors="coerce").fillna(0).astype(int) >= min_n]
    # odds filter: hide ultra-favorites (keep anything >= max_fav_odds)
    df_b_filt = df_b_filt[pd.to_numeric(df_b_filt.get("Best_Odds", df_b_filt.get("Odds", 0)), errors="coerce").fillna(0.0) >= float(max_fav_odds)]
    if q:
        df_b_filt = df_b_filt[df_b_filt.get("Player","").astype(str).str.lower().str.contains(q)]



    # Top candidates: DPS-first (presentation-only)
    _rank = df_b.copy()
    _rank["_dps_adj"] = pd.to_numeric(_rank.get("DPS_Adj", 0), errors="coerce").fillna(0.0)
    _rank["_dps_n"] = pd.to_numeric(_rank.get("DPS_N", 0), errors="coerce").fillna(0).astype(int)
    _rank["_odds"] = pd.to_numeric(_rank.get("Best_Odds", _rank.get("Odds", 0)), errors="coerce").fillna(0.0)
    _rank = _rank.sort_values(["_dps_adj","_dps_n","_odds"], ascending=[False, False, False])

    top_n = st.slider("Show top plays", 5, 30, 12, 1, key="board_topn")
    top = _rank.head(int(top_n)).copy()

    def _best_why(r: pd.Series) -> str:
        # UI-only: tracker exports a unified 'Why' column
        return str(r.get("Why", "") or "")
    # Render as cards (two-column grid)
    grid_left, grid_right = st.columns(2)
    for i, (_, r) in enumerate(top.iterrows()):
        side = grid_left if i % 2 == 0 else grid_right
        with side:
            player = r.get("Player","")
            game = r.get("Game","")
            tier = r.get("Tier_Tag","")
            bm = r.get("Best_Market","")
            accent = _board_market_accent(bm)
            bc = r.get("Best_Conf","")
            evsig = r.get("EV_Signal","")
            lock = r.get("LOCK","")
            markets = r.get("Markets","")
            why = _best_why(r)

            expl, crit = _derive_badges(r)
            # Flames are best-market regression heat (visual only; logic already exists upstream)
            bm_heat = ""
            if str(bm or "").upper().startswith("SOG"):
                bm_heat = r.get("Reg_Heat_S","")
            elif str(bm or "").upper().startswith("ASS"):
                bm_heat = r.get("Reg_Heat_A","")
            elif str(bm or "").upper().startswith("POI") or str(bm or "").upper().startswith("PTS"):
                bm_heat = r.get("Reg_Heat_P","")
            else:
                bm_heat = r.get("Reg_Heat_G","")
            flames = _flames_from_heat(bm_heat)

            dps_t = str(r.get("DPS_Title","") or "").strip()
            dps_w = _safe_float(r.get("DPS_Win"), 0.0) or 0.0
            dps_n = int(_safe_float(r.get("DPS_N"), 0) or 0)
            dps_a = _safe_float(r.get("DPS_Adj"), 0.0) or 0.0
            headline = f"**{player}** — {game}  ·  {expl}{crit} **{bm}**  ·  🏆 {dps_t}  ·  AdjWin **{dps_a:.1f}**  (Win {dps_w:.1f}% • n={dps_n})"
            mb = calc_ev_per_dollar(_to_float(_get(r, "Model%", "Model_Prob", default="")), _to_float(_get(r, "Odds", "Odds_Amer", default="")))
            mb_txt = f"↩ {mb:+.2f}/$1" if mb is not None else ""
            badges = " ".join([str(x) for x in [lock, evsig, mb_txt] if str(x).strip()])
            st.markdown(f"<div class='wl-board-card {accent}'>"
                        f"<div style='display:flex;justify-content:space-between;gap:10px;'>"
                        f"<div style='font-size:16px;line-height:1.2;'>{headline}</div>"
                        f"<div style='font-size:16px;white-space:nowrap;'>{badges}</div>"
                        f"</div>"
                        f"<div style='margin-top:6px;opacity:0.95;'>{markets}</div>"
                        f"</div>", unsafe_allow_html=True)

            with st.expander("🔥 Why it fires", expanded=False):
                # Market-aware combat HUD (presentation only)
                mkt_raw = str(bm or "").strip().upper()
                if mkt_raw.startswith("SOG"):
                    mkt = "SOG"
                    tags = str(r.get("SOG_Why", r.get("Why","")) or "").strip()
                elif mkt_raw.startswith("POINT"):
                    mkt = "POINTS"
                    tags = str(r.get("Points_Why", r.get("Why","")) or "").strip()
                elif mkt_raw.startswith("ASSIST"):
                    mkt = "ASSISTS"
                    tags = str(r.get("Assist_Why", r.get("Why","")) or "").strip()
                elif mkt_raw.startswith("ATG"):
                    mkt = "ATG"
                    tags = str(r.get("Goal_Why", r.get("Why","")) or "").strip()
                elif mkt_raw.startswith("GOAL"):
                    mkt = "GOALS"
                    tags = str(r.get("Goal_Why", r.get("Why","")) or "").strip()
                else:
                    mkt = mkt_raw or "UNKNOWN"
                    tags = str(r.get("Why","") or "").strip()

                _why_sections_header(mkt)
                _render_why_it_fires_rich(mkt, r, tags)

    with st.expander("Full Board Table (all rows)", expanded=False):
        show_table(df_b, board_cols, "Board (sorted by Best_Conf)")



# =========================
# POINTS
# =========================
elif page == "Points":

    st.markdown(_page_title_html("Points", "POINTS"), unsafe_allow_html=True)

    df_p = df_f.copy()
    df_p["_cp"] = safe_num(df_p, "Conf_Points", 0)
    df_p = df_p.sort_values(["_cp"], ascending=[False]).drop(columns=["_cp"], errors="ignore")

    st.sidebar.subheader("Points Filters")
    show_all = st.sidebar.checkbox("Show all players (ignore filters)", value=False, key="show_all_points")
    min_conf = st.sidebar.slider("Min Conf (Points)", 0, 100, 70, 1)
    color_pick = st.sidebar.multiselect(
        "Colors (Points)",
        ["green", "yellow", "blue", "red"],
        # Default excludes yellow (yellow was not part of the green-matrix test regime)
        default=["green", "blue"]
    )
    require_matrix_green = st.sidebar.checkbox(
        "Require Matrix=Green (Points)",
        value=True,
        key="require_matrix_green_points",
        help="Recommended: our Points testing baseline is Matrix=Green only."
    )

    if not show_all:
        df_p = df_p[df_p["Conf_Points"].fillna(0) >= min_conf]
        if require_matrix_green and "Matrix_Points" in df_p.columns:
            df_p = df_p[df_p["Matrix_Points"].astype(str).str.strip().str.lower() == "green"]
        if "Color_Points" in df_p.columns and color_pick:
            df_p = df_p[df_p["Color_Points"].isin(color_pick)]

    df_p["Green"] = df_p["Green_Points"].map(lambda x: "🟢" if bool(x) else "")

    points_cols = [
        "Game","Player","Pos","Tier_Tag",
        "Green","LOCK",

        # --- BOOK FIRST ---
        "Points_Line","Points_Odds_Over","Points_Book",

        # --- SIGNALS ---
        "Conf_Points","Matrix_Points",

        # --- HUD MATH (core) ---
        "Assists_mu","Points_mu",
        "PPP10_total","PP_iXA60","opp_5v5_xGA60",
        # --- TIMING / STRUCTURE ---
        "REG_PRESSURE",
        "Reg_Heat_P","Reg_Gap_P10",
        "Drought_P",
        "L10_Rate_Points","L10_Diff_Points",
        "Exp_P_10","L10_P",

        # --- SUPPORT STATS (creator identity) ---
        "iXA%","PP_Points60","i5v5_points60",

        # --- ENV (context / loss-avoidance) ---
        "opp_5v5_HDCA60",
        "Opp_Goalie","Opp_SV","Opp_GAA","Goalie_Weak","Opp_DefWeak",
    ]

    # Signals-first extras






    df_p["Markets"] = df_p.apply(build_markets_pills, axis=1)






    g = df_p.get("Green_Points", (df_p.get("Green","") == "🟢"))






    e = df_p["Plays_EV_Points"] if "Plays_EV_Points" in df_p.columns else pd.Series([""]*len(df_p), index=df_p.index)






    p = df_p["Points_EV%"] if "Points_EV%" in df_p.columns else pd.Series([None]*len(df_p), index=df_p.index)






    df_p["EV_Signal"] = [build_ev_signal(gg, ee, pp) for gg, ee, pp in zip(g, e, p if hasattr(p, "__iter__") else [p]*len(df_p))]

    # -------------------------
    # 🎰 Points Regression UI (label + gauge + jackpot)
    # -------------------------
    try:
        _rg = pd.to_numeric(df_p.get("Reg_Gap_P10", 0), errors="coerce").fillna(0.0)
    except Exception:
        _rg = pd.Series([0.0] * len(df_p), index=df_p.index)
    try:
        _dr = pd.to_numeric(df_p.get("Drought_P", 0), errors="coerce").fillna(0.0)
    except Exception:
        _dr = pd.Series([0.0] * len(df_p), index=df_p.index)

    def _reg_label(v: float) -> str:
        try:
            x = float(v)
        except Exception:
            x = 0.0
        if x >= float(POINTS_ENGINE_REG_GAP):
            return "🔥 REG READY"
        if x >= 1.5:
            return "🟡 REG BUILD"
        return "⚪ NO REG"

    df_p["REG_LABEL"] = [_reg_label(v) for v in _rg]
    df_p["REG_PRESSURE"] = [f"{_text_bar(v, 0.0, 8.0, 10)} {float(v):.2f}" for v in _rg]
    df_p["REG_DROUGHT"] = [
        "🎰🔥 REG+DROUGHT" if (float(rg) >= float(POINTS_ENGINE_REG_GAP) and float(dr) >= 2.0) else ""
        for rg, dr in zip(_rg, _dr)
    ]





    df_p["LOCK"] = [build_lock_badge(gg, ee) for gg, ee in zip(g, e)]
    legend_signals()
    render_odds_implied_reference(location="main")
    _f = render_market_filter_bar(default_min_conf=60, key_prefix="pts")

    try:
        df_p = apply_market_filters(
            df_p,
            _f,
            green_col="Green_Points",
            ev_icon_col="Plays_EV_Points",
            conf_col="Conf_Points",
            matrix_col="Matrix_Points",
            lock_col="LOCK",
        )
    except Exception:
        pass


    # --- DPS ranking + filters (Board-style; presentation only) ---
    df_p = add_best_proc_cols(df_p, 'POINTS')
    df_p = apply_dps_filters_ui(df_p, 'POINTS', key_prefix='pts')








    # === SMASH PLAYS (POINTS) ===
    _render_badge_legend_inline()


    st.subheader("⭐ Smash Plays — Points")
    render_valhalla_gate("POINTS")


    _p = df_p.copy()
    try:
        _p = _p[
            (_p.get("Matrix_Points", "").astype(str).str.strip().str.upper().isin(["GREEN","🟢"])) &
            (pd.to_numeric(_p.get("Points_Line", 0), errors="coerce") == 0.5) &
            (_p.get("Outcome_Points", "").astype(str).str.upper().isin(["W","L"]) | (_p.get("Match_Status_Points", "").astype(str).str.upper().ne("GRADED")))
        ].copy()
    except Exception:
        pass

    heat = _p.get("Reg_Heat_P", "").astype(str).str.upper().isin(["HOT","DUE","OVERDUE"])
    gap = pd.to_numeric(_p.get("Reg_Gap_P10", np.nan), errors="coerce").fillna(-999) >= 2.5
    drt = pd.to_numeric(_p.get("Drought_P", np.nan), errors="coerce").fillna(-999) >= 2
    reg_valid = heat | gap | drt
    _p = _p[reg_valid].copy()

    _p["_conf"] = pd.to_numeric(_p.get("Conf_Points", 0), errors="coerce").fillna(0)
    _p["_l10r"] = pd.to_numeric(_p.get("L10_Rate_Points", np.nan), errors="coerce")
    _p["_l10d"] = pd.to_numeric(_p.get("L10_Diff_Points", np.nan), errors="coerce")
    _p["_gap"] = pd.to_numeric(_p.get("Reg_Gap_P10", np.nan), errors="coerce")

    _p = _p.sort_values(["_conf","_l10r","_l10d","_gap"], ascending=[False, False, False, False], kind="mergesort")

    top_n_p = st.slider("Show top plays (Points)", 3, 30, 12, 1, key="points_smash_topn")
    topp = _p.head(int(top_n_p))

    for _, r in topp.iterrows():
        player = str(r.get("Player", "") or "").strip()
        game = str(r.get("Game", "") or "").strip()
        line = r.get("Points_Line", "")
        odds = r.get("Points_Odds_Over", r.get("Points_Odds", ""))
        conf = r.get("Conf_Points", "")
        gapv = r.get("Reg_Gap_P10", "")
        heatv = str(r.get("Reg_Heat_P", "") or "").strip()
        l10r = r.get("L10_Rate_Points", "")
        l10d = r.get("L10_Diff_Points", "")
        xga = r.get("opp_5v5_xGA60", "")
        hdca = r.get("opp_5v5_HDCA60", "")

        tags = []
        _conf_p = _safe_float(conf)
        _conf_a = _safe_float(r.get("Conf_Assists"))
        # Conf tags (visual only)
        if _conf_p is not None:
            if _conf_p >= 80:
                tags.append("⚔️ Conf 80+")
            elif _conf_p >= 70:
                tags.append("Conf 70+")
        if _conf_a is not None and _conf_a >= 89:
            tags.append("⚔️ A-Conf 89+")

        # Combat HUD icons (POINTS) — on player card (match other markets)
        try:
            _line = float(line)
        except Exception:
            _line = None

        # NOTE: these are visual-only symbols; they do NOT change ranking/EV/conf.
        if _line is not None:
            # 0.5 Fortress spec
            if _line <= 0.75:
                _ppp = _safe_float(r.get("PPP10_total")) or 0.0
                _ppixg = _safe_float(r.get("PP_iXG60")) or 0.0
                _amu = _safe_float(r.get("Assists_mu")) or 0.0
                _oppgaa = _safe_float(r.get("Opp_GAA"))

                if _conf_p is not None and _conf_p >= 70:
                    tags.append(_svg_inline(_svg_get("PTS05_HAMMER_FISTS.svg"), size=14, title="Hammer Fists"))
                if _conf_p is not None and _conf_p >= 70 and _ppp >= 3:
                    tags.append(_svg_inline(_svg_get("PTS05_HAMMER_STOMP.svg"), size=14, title="Echo Stomp I"))
                if _conf_p is not None and _conf_p >= 78 and _amu >= 1.0:
                    tags.append(_svg_inline(_svg_get("PTS05_ENRAGED_FURY.svg"), size=14, title="Enraged Fury"))
                if _conf_p is not None and _conf_p >= 78 and (_oppgaa is not None) and (2.5 <= float(_oppgaa) <= 3.0):
                    tags.append(_svg_inline(_svg_get("PTS05_ENRAGED_FURY.svg"), size=14, title="Blood Stomp"))

                # Gaia's Blessing v2 ladder (Conf 75/77/82 + Assists_mu>=0.7) — add BOTH icon + descriptor
                gaia_active = (_conf_p is not None) and (_amu >= 0.7) and (_conf_p >= 75)
                if gaia_active:
                    if _conf_p >= 82:
                        tags.append(_svg_inline(_svg_get("PTS05_GAIAS_BLESSING.svg"), size=14, title="Gaia’s Blessing++ (82+)"))
                        tags.append("Gaia’s Blessing++")
                    elif _conf_p >= 77:
                        tags.append(_svg_inline(_svg_get("PTS05_GAIAS_BLESSING.svg"), size=14, title="Gaia’s Blessing+ (77+)"))
                        tags.append("Gaia’s Blessing+")
                    else:  # 75+
                        tags.append(_svg_inline(_svg_get("PTS05_GAIAS_BLESSING.svg"), size=14, title="Gaia’s Blessing (75+)"))
                        tags.append("Gaia’s Blessing")

                    # BONUS: Gaia Heat Ladder (context labels; show highest active rung)
                    _tgf = _safe_float(r.get("Team_GF_Avg_L5", r.get("Team_GF_L5")))
                    _osog = _safe_float(r.get("Opp_SOG_Against_L10"))
                    if (_tgf is not None) and (_tgf >= 3.7) and (_osog is not None) and (_osog >= 27.5):
                        tags.append("Gaia’s Floodgate")
                    elif (_tgf is not None) and (_tgf >= 3.9):
                        tags.append("Gaia’s Ascension")
                    elif (_tgf is not None) and (_tgf >= 3.7):
                        tags.append("Gaia’s Wrath")
                    elif (_tgf is not None) and (_tgf >= 3.5):
                        tags.append("Gaia’s Favor")

                # Bleed ENV (label-only)
                if _conf_p is not None and _conf_p >= 70 and _ppixg >= 1.5:
                    tags.append(_svg_inline(_svg_get("PTS05_BLEED_ENV.svg"), size=14, title="Bleed ENV (Label)"))

            # 1.5 DPS spec
            else:
                _mu = _safe_float(r.get("Points_mu")) or 0.0
                _xga = _safe_float(r.get("opp_5v5_xGA60"))
                _drt = _safe_float(r.get("Drought_P")) or 0.0

                # New ladder (Backbone / Power / Monster) uses Conf_Assists >= 89 — add BOTH icon + descriptor
                if _conf_a is not None and _conf_a >= 89 and _mu >= 2.2 and (_xga is not None) and float(_xga) >= 2.6:
                    tags.append(_svg_inline(_svg_get("PTS15_BLADE_SLASH.svg"), size=14, title="Monster (Blade Slash)"))
                    tags.append("Blade Slash (Monster)")
                elif _conf_a is not None and _conf_a >= 89 and _mu >= 2.2:
                    tags.append(_svg_inline(_svg_get("PTS15_BLADE_IMPALE.svg"), size=14, title="Power Tier (Blade Impale)"))
                    tags.append("Blade Impale (Power Tier)")
                elif _conf_a is not None and _conf_a >= 89 and _mu >= 1.7:
                    tags.append(_svg_inline(_svg_get("PTS15_TWO_HANDED_HAMMER.svg"), size=14, title="Backbone"))
                    tags.append("Backbone")

                if _conf_a is not None and _conf_a >= 89 and _drt >= 1 and _mu >= 1.7:
                    tags.append("Delayed Hammer Smash")

                # Optional legacy kit icons (PP/DefWeak) — only if they proc
                _ppixg = _safe_float(r.get("PP_iXG60")) or 0.0
                _ppixa = _safe_float(r.get("PP_iXA60")) or 0.0
                _teamxgf = _safe_float(r.get("Team_PP_xGF60")) or 0.0
                _defw = _safe_float(r.get("Opp_DefWeak")) or 0.0
                if _conf_p is not None and _conf_p >= 80 and _defw >= 60:
                    tags.append(_svg_inline(_svg_get("PTS15_BLOOD_EXPOSURE.svg"), size=14, title="Blood Exposure II (Legacy)"))
                if _conf_p is not None and _conf_p >= 80 and _defw >= 70:
                    tags.append(_svg_inline(_svg_get("PTS15_POLARIZING_SMASH.svg"), size=14, title="Eternal Smash (Legacy)"))
        try:
            if float(l10r) >= 0.80: tags.append("🔥 L10 Rate ≥0.80")
            elif float(l10r) >= 0.70: tags.append("L10 Rate ≥0.70")
        except Exception:
            pass
        try:
            if float(l10d) >= 0.25: tags.append("🧨 L10 Diff ≥0.25")
        except Exception:
            pass

        warns = []
        try:
            if float(xga) <= 2.43: warns.append("⚠️ Suppressive xGA (≤2.43)")
        except Exception:
            pass
        try:
            if float(hdca) <= 2.33: warns.append("⚠️ Low HDCA (≤2.33)")
        except Exception:
            pass

        def _is_nan(v) -> bool:
            try:
                if v is None:
                    return True
                if isinstance(v, float) and math.isnan(v):
                    return True
                return str(v).strip().lower() == "nan"
            except Exception:
                return True

        meta = []
        if heatv:
            meta.append(f"Heat {heatv}")
        try:
            _g = float(gapv)
            if not math.isnan(_g):
                meta.append(f"Gap {_g:.2f}")
        except Exception:
            pass
        try:
            _c = float(conf)
            if not math.isnan(_c):
                meta.append(f"Conf {_c:.0f}")
        except Exception:
            pass
        try:
            _lr = float(l10r)
            if not math.isnan(_lr):
                meta.append(f"L10Rate {_lr:.2f}")
        except Exception:
            pass
        try:
            _ld = float(l10d)
            if not math.isnan(_ld):
                meta.append(f"L10Diff {_ld:.2f}")
        except Exception:
            pass

                # --- Player card line (match GOALS/ASSISTS style): icons + bold combo + light meta ---
        proc_icons = "".join([t for t in tags if (isinstance(t, str) and ("<svg" in t or "wl-ico" in t))])

        # Separate combo descriptors from meta-like tags (keep L10 tags as meta; keep core move names as combo)
        combo_bits = []
        meta_bits = []
        for t in tags:
            if not isinstance(t, str):
                continue
            if "<svg" in t or "wl-ico" in t:
                continue
            s = t.strip()
            if not s:
                continue
            if s.startswith(("🔥", "🧨", "Heat", "Gap", "Conf", "L10")):
                meta_bits.append(s)
            else:
                combo_bits.append(s)

        combo_s = " • ".join(combo_bits[:3])

        # Build compact meta (no NaNs)
        meta = []
        _mx = str(r.get("Matrix_Points", r.get("Matrix", "")) or "").strip()
        if _mx:
            meta.append(_mx)
        if heatv:
            meta.append(f"Heat {heatv}")
        try:
            _g = float(gapv)
            if not math.isnan(_g):
                meta.append(f"Gap {_g:.2f}")
        except Exception:
            pass
        try:
            _c = float(conf)
            if not math.isnan(_c):
                meta.append(f"Conf {_c:.0f}")
        except Exception:
            pass

        # Key “up-front” stat trio for Points
        try:
            _pmu = float(r.get("Points_mu", float("nan")))
            if not math.isnan(_pmu):
                meta.append(f"μ {_pmu:.2f}")
        except Exception:
            pass
        try:
            _amu = float(r.get("Assists_mu", float("nan")))
            if not math.isnan(_amu):
                meta.append(f"Aμ {_amu:.2f}")
        except Exception:
            pass
        try:
            _tgf = float(r.get("Team_GF_Avg_L5", r.get("Team_GF_L5", float("nan"))))
            if not math.isnan(_tgf):
                meta.append(f"GF_L5 {_tgf:.1f}")
        except Exception:
            pass
        try:
            _dr = float(r.get("Drought_P", float("nan")))
            if not math.isnan(_dr) and _dr >= 1:
                meta.append(f"Drought {int(_dr)}")
        except Exception:
            pass

        meta.extend(meta_bits)
        meta.extend(warns)

        meta_s = " | ".join([m for m in meta if m])

        dash = " — " if combo_s else ""
        card_line = f"{proc_icons} <span style=\"font-weight:800;\">{combo_s}</span><span style=\"opacity:0.8;\">{dash}{meta_s}</span>"

        _line_s = "" if _is_nan(line) else str(line)
        _odds_s = "" if _is_nan(odds) else str(odds)
        betline = (f"PTS {_line_s}" + (f" @ {_odds_s}" if _odds_s else "")) if _line_s else ""
        headline = f"<b>{player}</b> — {game}" if game else f"<b>{player}</b>"

        st.markdown(
            f"""
 <div class=\"wl-card wl-accent-blue\">
   <div style=\"display:flex;justify-content:space-between;gap:10px;\">
     <div style=\"font-size:16px;line-height:1.2;\">
       {headline}
       <div style=\"opacity:0.9;margin-top:4px;\">{betline}</div>
     </div>
     <div style=\"font-size:16px;white-space:nowrap;\">{_engine_badge('POINTS', r)} {str(r.get('LOCK','') or '').strip()}</div>
   </div>
   <div style=\"margin-top:6px;font-size:12px;opacity:0.92;line-height:1.2;\">{card_line}</div>
 </div>
            """,
            unsafe_allow_html=True,
        )

        _why_tags = str(r.get("Points_Why", r.get("Why", "")) or "").strip()
        with st.expander("Why it fires", expanded=False):
            _why_sections_header("POINTS")
            _render_why_it_fires_rich("POINTS", r, _why_tags)

    st.markdown("---")

    show_table(df_p, points_cols, "Points View")


# =========================
# ASSISTS
# =========================
elif page == "Assists":

    st.markdown(_page_title_html("Assists", "ASSISTS"), unsafe_allow_html=True)

    df_a = df_f.copy()
    df_a["_ca"] = safe_num(df_a, "Conf_Assists", 0)
    df_a = df_a.sort_values(["_ca"], ascending=[False]).drop(columns=["_ca"], errors="ignore")

    st.sidebar.subheader("Assists Filters")
    show_all = st.sidebar.checkbox("Show all players (ignore filters)", value=False, key="show_all_assists")
    min_conf = st.sidebar.slider("Min Conf (Assists)", 0, 100, 80, 1)
    color_pick = st.sidebar.multiselect(
        "Colors (Assists)",
        ["green", "yellow", "blue", "red"],
        default=["green", "yellow", "blue"]
    )

    if not show_all:
        df_a = df_a[df_a["Conf_Assists"].fillna(0) >= min_conf]
        if "Color_Assists" in df_a.columns and color_pick:
            df_a = df_a[df_a["Color_Assists"].isin(color_pick)]

    df_a["Green"] = df_a.get("Green_Assists", False).map(lambda x: "🟢" if bool(x) else "")
    df_a["PP_PROOF"] = df_a.get("Assist_PP_Proof", False).map(lambda x: "✅" if bool(x) else "")

    # --- DPS ranking + filters (Board-style; presentation only) ---
    df_a = add_best_proc_cols(df_a, 'ASSISTS')
    df_a = apply_dps_filters_ui(df_a, 'ASSISTS', key_prefix='assists')


    # Valhalla gate columns (Assists) — matches board text
    df_a["Valhalla_OK"] = (
        (df_a.get("Matrix_Assists", "").astype(str).str.strip().str.lower() == "green")
        & (pd.to_numeric(df_a.get("Assists_Line", 0), errors="coerce").fillna(0) == 0.5)
        & (pd.to_numeric(df_a.get("Conf_Assists", 0), errors="coerce").fillna(0) >= 80)
    ).map(lambda x: "✅" if bool(x) else "")

    # MAIN tier from PP_iXA60 (display)
    _pp_ix = pd.to_numeric(df_a.get("PP_iXA60", 0), errors="coerce").fillna(0)
    df_a["PP_iXA60_Tier"] = np.select(
        [_pp_ix >= 4.2, _pp_ix >= 3.0],
        ["ELITE", "STRONG"],
        default=""
    )

    # ENV warnings (display only)
    _opp_sv = pd.to_numeric(df_a.get("Opp_SV", 0), errors="coerce").fillna(0)
    _xga = pd.to_numeric(df_a.get("opp_5v5_xGA60", 0), errors="coerce").fillna(0)
    _gweak = pd.to_numeric(df_a.get("Goalie_Weak", 0), errors="coerce").fillna(0)

    df_a["ENV_BAD_OppSV"] = (_opp_sv >= 0.905).map(lambda x: "⚠️" if bool(x) else "")
    df_a["ENV_GOOD_OppSV"] = ((_opp_sv > 0) & (_opp_sv < 0.885)).map(lambda x: "✅" if bool(x) else "")  # SV% < 88.5% = weak goalie (good for assists)
    df_a["ENV_BAD_xGA"] = ((_xga > 0) & (_xga <= 2.40)).map(lambda x: "⚠️" if bool(x) else "")
    df_a["ENV_BAD_GWeak"] = ((_gweak > 0) & (_gweak <= 35)).map(lambda x: "⚠️" if bool(x) else "")
    df_a["ENV_GOOD_GWeak"] = (_gweak >= 82).map(lambda x: "✅" if bool(x) else "")

    # 🗡️ Dagger indicator (PP assist edge) — HARD GATE (recomputed every time)
    # Goal: daggers are rare and meaningful (PP1/proof-level assist edges only).
    df_a["🗡️"] = ""

    # Safe pulls
    proof_col = "Assist_PP_Proof" if "Assist_PP_Proof" in df_a.columns else None
    proof = df_a[proof_col].astype(bool) if proof_col else False

    apc = pd.to_numeric(df_a.get("Assist_ProofCount", 0), errors="coerce").fillna(0)
    adg = pd.to_numeric(df_a.get("Assist_Dagger", 0), errors="coerce").fillna(0)
    ppt = df_a.get("PP_Tier", "").astype(str).str.upper()

    # HARD gate:
    # 1) Explicit proof, OR
    # 2) 4-of-4 assist proofs, OR
    # 3) Elite dagger score (>=85), OR
    # 4) PP A/B + strong proof (>=3) + decent dagger (>=70)
    mask = (proof if isinstance(proof, pd.Series) else False)

    df_a.loc[mask, "🗡️"] = "🗡️"

    assists_cols = [

        "Game",
        "Player", "Pos",
        "Tier_Tag",
        "Markets",
        "Green",
                "LOCK", "Assists_Odds_Over",
        "Assists_Book",
        "Conf_Assists", "Matrix_Assists", "Assists_Line", "Valhalla_OK", "PP_iXA60", "PP_iXA60_Tier", "PP_TOI_Pct_Game", "PP_Matchup",
        "opp_5v5_xGA60",      "ENV_BAD_OppSV", "ENV_GOOD_OppSV", "ENV_BAD_xGA", "ENV_BAD_GWeak", "ENV_GOOD_GWeak","PP_PROOF", 

       
       
        "Drought_A",
       
        
        "Reg_Heat_A", "Reg_Gap_A10", "Exp_A_10", "L10_A",
        "PP_Tier", "PP_Path", 
        "PP_TOI_Pct_Game",  "PP_Matchup",

        
        "iXA%","iXG%", "v2_player_stability",
        "Opp_Goalie", "Opp_SV",
        "Goalie_Weak", "Opp_DefWeak",

        # --- EV / Odds ---
       
        "Assists_Model%",
        "Assists_Imp%",
        "Assists_EV%",
        "Plays_EV_Assists",

        "Line", "Odds", "Result",
    ]

    # Signals-first extras

    df_a["Markets"] = df_a.apply(build_markets_pills, axis=1)

    g = df_a.get("Green_Assists", (df_a.get("Green","") == "🟢"))

    e = df_a["Plays_EV_Assists"] if "Plays_EV_Assists" in df_a.columns else pd.Series([""]*len(df_a), index=df_a.index)

    p = df_a["Assists_EV%"] if "Assists_EV%" in df_a.columns else pd.Series([None]*len(df_a), index=df_a.index)
    st.subheader("⭐ Smash Plays — Assists")

    # Valhalla Gate presentation (ASSISTS) — baseline only
    render_valhalla_gate("ASSISTS")

    # Eligibility (ignore EV): Matrix Green + Line 0.5 + Conf >= 80
    _a = df_a.copy()

    _mat = _a["Matrix_Assists"] if "Matrix_Assists" in _a.columns else (_a["Matrix_A"] if "Matrix_A" in _a.columns else pd.Series("", index=_a.index))
    _line = _a["Assists_Line"] if "Assists_Line" in _a.columns else (_a["Line_Assists"] if "Line_Assists" in _a.columns else pd.Series([np.nan]*len(_a), index=_a.index))
    _conf = _a["Conf_Assists"] if "Conf_Assists" in _a.columns else (_a["Conf_A"] if "Conf_A" in _a.columns else pd.Series([0]*len(_a), index=_a.index))

    # NOTE: don't block on Outcome/Match_Status for today's slate — Gate is just stance + floor.
    _a = _a[
        (_mat.astype(str).str.strip().str.upper().isin(["GREEN", "🟢"])) &
        ((pd.to_numeric(_line, errors="coerce") == 0.5) | (pd.to_numeric(_line, errors="coerce").isna())) &
        (pd.to_numeric(_conf, errors="coerce").fillna(0) >= 80)
    ].copy()

    # Feature pulls (safe)
    _a["_conf"] = pd.to_numeric(_a.get("Conf_Assists", 0), errors="coerce").fillna(0)
    _a["_ppixa"] = pd.to_numeric(_a.get("PP_iXA60", _a.get("PP_iXA_60", np.nan)), errors="coerce")
    _a["_ppshare"] = pd.to_numeric(_a.get("PP_TeamShare_pct", _a.get("PP_TeamShare%", np.nan)), errors="coerce")
    _a["_ixa_pct"] = pd.to_numeric(_a.get("iXA%", np.nan), errors="coerce")
    _a["_team_gf_l5"] = pd.to_numeric(_a.get("Team_GF_L5", np.nan), errors="coerce")

    _a = _a.sort_values(["_conf","_ppixa","_ppshare"], ascending=[False, False, False], kind="mergesort")

    top_n_a = st.slider("Show top plays (Assists)", 3, 30, 12, 1, key="assist_smash_topn")
    topa = _a.head(int(top_n_a))

    if len(topa) == 0:
        st.info("No assists currently pass the Valhalla Gate (Matrix Green + 0.5 line + Conf ≥ 80).")
    else:
        for _, r in topa.iterrows():
            player = str(r.get("Player", "") or "").strip()
            game = str(r.get("Game", "") or "").strip()

            odds = r.get("Assists_Odds_Over", r.get("Assists_Odds", r.get("Odds", "")))
            conf = float(r.get("Conf_Assists", 0) or 0)

            # Core features for moves
            ixa_pct = float(r.get("_ixa_pct", np.nan)) if not pd.isna(r.get("_ixa_pct", np.nan)) else float("nan")
            pp_ix   = float(r.get("_ppixa", np.nan)) if not pd.isna(r.get("_ppixa", np.nan)) else float("nan")
            team_gf = float(r.get("_team_gf_l5", np.nan)) if not pd.isna(r.get("_team_gf_l5", np.nan)) else float("nan")

            # New move logic (signals only; no gating)
            # Stars Aligned: Conf ≥ 88 + iXA% ≥ 96  (n=163, 65.6% in your test)
            stars_aligned = (conf >= 88) and (not math.isnan(ixa_pct)) and (ixa_pct >= 96)

            # Supernova Overdrive: Conf ≥ 80 + iXA% ≥ 95 + PP_iXA60 ≥ 3.7 + Team_GF_L5 ≥ 20  (n=64, 75.0%)
            supernova_overdrive = (
                (conf >= 80)
                and (not math.isnan(ixa_pct)) and (ixa_pct >= 95)
                and (not math.isnan(pp_ix)) and (pp_ix >= 3.7)
                and (not math.isnan(team_gf)) and (team_gf >= 20)
            )

            # Magic Man: Conf ≥ 87 + iXA% ≥ 99 (kept as Mythic)
            magic_on = (conf >= 87) and (not math.isnan(ixa_pct)) and (ixa_pct >= 99)

                        # --- Card (match other markets: wl-card + accent) ---
            meta = []
            try: meta.append(f"Conf {float(conf):.0f}")
            except Exception: pass
            if not math.isnan(ixa_pct):
                meta.append(f"iXA% {ixa_pct:.1f}")
            if not math.isnan(pp_ix):
                meta.append(f"PP_iXA60 {pp_ix:.2f}")
            if not math.isnan(team_gf):
                meta.append(f"Team_GF_L5 {team_gf:.0f}")

            meta_s = " | ".join([m for m in meta if m])

            # --- Card procs (icons + short combo reads) ---
            ppp10 = _safe_float(r.get("PPP10_total"), default=float("nan"))
            assists_mu = _safe_float(r.get("Assists_mu"), default=float("nan"))
            goalie_weak = _safe_float(r.get("Goalie_Weak"), default=float("nan"))

            creator_role = (not math.isnan(ixa_pct)) and (ixa_pct >= 95.0)
            elite_creator = (not math.isnan(ixa_pct)) and (ixa_pct >= 99.0)
            pp_engine = (not math.isnan(pp_ix)) and (pp_ix >= 4.0)
            pp_hot = (not math.isnan(ppp10)) and (ppp10 >= 5.0)
            on_heater = (not math.isnan(team_gf)) and (team_gf >= 20.0)
            playmaking_pace = (not math.isnan(assists_mu)) and (assists_mu >= 1.30)
            conf_spike = False  # GOALS tiers are conf-free (beta)
            soft_goalie = (not math.isnan(goalie_weak)) and (goalie_weak >= 90.0)

            # Short combo tags (2–3 words)
            combo_tags = []
            if elite_creator: combo_tags.append("Elite Creator")
            elif creator_role: combo_tags.append("Creator Role")
            if pp_engine: combo_tags.append("PP Engine")
            if pp_hot: combo_tags.append("PP Hot")
            if on_heater: combo_tags.append("On Heater")
            if playmaking_pace: combo_tags.append("Playmaking Pace")
            if conf_spike: combo_tags.append("Conf Spike")
            if soft_goalie: combo_tags.append("Soft Goalie")

            combo_s = " • ".join(combo_tags[:6])  # keep it tight

            # Icons (show only what fired)
            icons = []
            icons.append(_svg_icon("staff.svg", "Base Shell", "wl-assists"))
            if creator_role: icons.append(_svg_icon("odins_arcane_orb.svg", "Creator Role", "wl-assists"))
            if pp_engine: icons.append(_svg_icon("runic_infusion.svg", "PP Engine", "wl-assists"))
            if pp_hot: icons.append(_svg_icon("arcane_channel_iii.svg", "PP Hot", "wl-assists"))
            if playmaking_pace: icons.append(_svg_icon("silent_distributor_ii.svg", "Playmaking Pace", "wl-assists"))
            if on_heater: icons.append(_svg_icon("arcane_alignment.svg", "On Heater", "wl-assists"))
            if conf_spike: icons.append(_svg_icon("valhalla.svg", "Conf Spike", "wl-assists"))
            if soft_goalie: icons.append(_svg_icon("odins_eye.svg", "Soft Goalie", "wl-assists"))
            if stars_aligned: icons.append(_svg_icon("stars.svg", "Stars Aligned", "wl-assists"))
            if supernova_overdrive: icons.append(_svg_icon("supernova.svg", "Wombo Stack", "wl-assists"))
            if magic_on: icons.append(_svg_icon("magic_mans_transcendence.svg", "Elite Creator", "wl-assists"))

            proc_icons = "".join([i for i in icons if i])

            betline = f"A 0.5 @ {odds}" if odds != "" else "A 0.5"
            headline = f"<b>{player}</b> — {game}" if game else f"<b>{player}</b>"

            st.markdown(
                f"""
             <div class="wl-card wl-accent-purple">
               <div style="display:flex;justify-content:space-between;gap:10px;">
                 <div style="font-size:16px;line-height:1.2;">
                   {headline}
                   <div style="opacity:0.9;margin-top:4px;">{betline}</div>
                 </div>
                 <div style="font-size:16px;white-space:nowrap;">{_engine_badge('ASSISTS', r)} {str(r.get('LOCK','') or '').strip()}</div>
               </div>
               <div style="margin-top:6px;font-size:12px;opacity:0.95;line-height:1.2;">{proc_icons} <span style="font-weight:800;">{combo_s}</span><span style="opacity:0.8;">{" — " if combo_s else ""}{meta_s}</span></div>
             </div>
                """,
                unsafe_allow_html=True,
            )

            # --- Why it fires (use rich HUD renderer like other markets) ---
            _why_tags = str(r.get("Assists_Why", r.get("Why", "")) or "").strip()
            with st.expander("Why it fires", expanded=False):
                _why_sections_header("ASSISTS")
                _render_why_it_fires_rich("ASSISTS", r, _why_tags)



    # Full table (Assists)
    show_table(df_a, assists_cols, "Assists View")

# =========================
# SOG
# =========================
elif page == "SOG":

    st.markdown(_page_title_html("SOG", "SOG"), unsafe_allow_html=True)

    df_s = df_f.copy()
    df_s["_cs"] = safe_num(df_s, "Conf_SOG", 0)
    df_s = df_s.sort_values(["_cs"], ascending=[False]).drop(columns=["_cs"], errors="ignore")

    st.sidebar.subheader("SOG Filters")
    show_all = st.sidebar.checkbox("Show all players (ignore filters)", value=False, key="show_all_sog")
    min_conf = st.sidebar.slider("Min Conf (SOG)", 0, 100, 75, 1)
    color_pick = st.sidebar.multiselect(
        "Colors (SOG)",
        ["green", "yellow", "blue", "red"],
        default=["green", "yellow", "blue"]
    )

    if not show_all:
        df_s = df_s[df_s["Conf_SOG"].fillna(0) >= min_conf]
        if "Color_SOG" in df_s.columns and color_pick:
            df_s = df_s[df_s["Color_SOG"].isin(color_pick)]

    df_s["Green"] = df_s["Green_SOG"].map(lambda x: "🟢" if bool(x) else "")

    sog_cols = [
       "Game",
       "Player", "Pos",
       "Tier_Tag",
       
       "Green",
              
       "Conf_SOG", "Matrix_SOG",

        # --- EV / Odds ---
        "SOG_Line",
        "SOG_Odds_Over",
        "SOG_Book",
        "L20_Rate_SOG", "L40_Rate_SOG",
        "Opp_SOG_Against_L50", "opp_5v5_xGA60", "Player_5v5_SOG_Share",
        "Drought_SOG", "Best_Drought",
        "Med10_SOG", "Avg5_SOG", "ShotIntent", "ShotIntent_Pct",
        "Reg_Heat_S", "Reg_Gap_S10", "Exp_S_10", "L10_S",
        "Opp_Goalie", "Opp_SV",
        "Goalie_Weak", "Opp_DefWeak",
        "Line", "Odds", "Result",
    ]


    # Signals-first extras


    df_s["Markets"] = df_s.apply(build_markets_pills, axis=1)


    g = df_s.get("Green_SOG", (df_s.get("Green","") == "🟢"))


    e = df_s["Plays_EV_SOG"] if "Plays_EV_SOG" in df_s.columns else pd.Series([""]*len(df_s), index=df_s.index)


    p = df_s["SOG_EV%"] if "SOG_EV%" in df_s.columns else pd.Series([None]*len(df_s), index=df_s.index)


    df_s["EV_Signal"] = [build_ev_signal(gg, ee, pp) for gg, ee, pp in zip(g, e, p if hasattr(p, "__iter__") else [p]*len(df_s))]

    df_s["LOCK"] = [build_lock_badge(gg, ee) for gg, ee in zip(g, e)]
    legend_signals()
    render_odds_implied_reference(location="main")
    _f = render_market_filter_bar(default_min_conf=60, key_prefix="sog")

    try:
        df_s = apply_market_filters(
            df_s,
            _f,
            green_col="Green_SOG",
            ev_icon_col="Plays_EV_SOG",
            conf_col="Conf_SOG",
            matrix_col="Matrix_SOG",
            lock_col="LOCK",
        )
    except Exception:
        pass


    # --- DPS ranking + filters (Board-style; presentation only) ---
    df_s = add_best_proc_cols(df_s, 'SOG')
    df_s = apply_dps_filters_ui(df_s, 'SOG', key_prefix='sog')

    # -------------------------
    # SOG Smash (cards) — Berserker kit (EV ignored)
    # -------------------------

    _render_badge_legend_inline()

    st.subheader("⭐ Smash Plays — SOG")
    st.caption("Gates: (A) Line ≤ 2.5 • Matrix = Green • Conf ≥ 75 • (ShotIntent ≥ 3.4 OR Drought == 1)  OR  (B) Line ≥ 3.5 • Matrix = Green • Conf ≥ 75 (Jungle — Sniper Spec) • EV ignored")

    top_n = st.slider("Show top plays (SOG)", 3, 25, 10, 1, key="sog_smash_topn")

    _rank = df_s.copy()

    # Global gates (always on)
    _mx = _rank.get("Matrix_SOG", "").astype(str).str.strip().str.lower().eq("green")
    _line = pd.to_numeric(_rank.get("SOG_Line", 0), errors="coerce").fillna(0.0)
    _conf = pd.to_numeric(_rank.get("Conf_SOG", 0), errors="coerce").fillna(0.0)

    # ShotIntent / SI (column may vary across builds)
    if "ShotIntent" in _rank.columns:
        _si = pd.to_numeric(_rank.get("ShotIntent", 0), errors="coerce").fillna(0.0)
    elif "SI" in _rank.columns:
        _si = pd.to_numeric(_rank.get("SI", 0), errors="coerce").fillna(0.0)
    elif "ShotIntent_SOG" in _rank.columns:
        _si = pd.to_numeric(_rank.get("ShotIntent_SOG", 0), errors="coerce").fillna(0.0)
    elif "SI_SOG" in _rank.columns:
        _si = pd.to_numeric(_rank.get("SI_SOG", 0), errors="coerce").fillna(0.0)
    else:
        _si = 0.0

    # Drought / regression timing (column may vary across builds)
    if "Drought_SOG" in _rank.columns:
        _dr = pd.to_numeric(_rank.get("Drought_SOG", 0), errors="coerce").fillna(0.0).astype(int)
    elif "Drought_S" in _rank.columns:
        _dr = pd.to_numeric(_rank.get("Drought_S", 0), errors="coerce").fillna(0.0).astype(int)
    else:
        _dr = pd.Series(0, index=_rank.index, dtype=int)

    # ---- Path A (Jungle main) — SOG 2.5 shell
    _base25 = _mx & (_line <= 2.5) & (_conf >= 75) & ((_si >= 3.4) | (_dr == 1))

    # ---- Path B (Jungle — Sniper Spec) — SOG 3.5 shell
    _base35 = _mx & (_line >= 3.5) & (_conf >= 75)

    _rank = _rank[_base25 | _base35].copy()

    # Sniper tiering (only applies to 3.5 lines; used for sorting + label)
    _xga = pd.to_numeric(_rank.get("opp_5v5_xGA60", pd.Series(0.0, index=_rank.index)), errors="coerce").fillna(0.0)
    _hdca = pd.to_numeric(_rank.get("opp_5v5_HDCA60", pd.Series(0.0, index=_rank.index)), errors="coerce").fillna(0.0)
    _l40 = pd.to_numeric(_rank.get("L40_Rate_SOG", pd.Series(0.0, index=_rank.index)), errors="coerce").fillna(0.0)
    _share = pd.to_numeric(_rank.get("Player_5v5_SOG_Share", pd.Series(0.0, index=_rank.index)), errors="coerce").fillna(0.0)
    _opp50 = pd.to_numeric(_rank.get("Opp_SOG_Against_L50", pd.Series(0.0, index=_rank.index)), errors="coerce").fillna(0.0)

    _permission_shatter = (_xga >= 2.50) | (_hdca >= 2.20)
    _enraged = (_l40 >= 3.0) & (_xga >= 2.50)
    _elite_enraged = _enraged & (_share >= 20.0)
    _enraged_shatter = (_opp50 >= 29.5) & _permission_shatter

    _is35 = pd.to_numeric(_rank.get("SOG_Line", 0), errors="coerce").fillna(0.0) >= 3.5
    _rank["_sniper_tier"] = ""
    _rank.loc[_is35 & _elite_enraged, "_sniper_tier"] = "SNIPER CRIT"
    _rank.loc[_is35 & (_rank["_sniper_tier"] == "") & _enraged, "_sniper_tier"] = "STRONG"
    _rank.loc[_is35 & (_rank["_sniper_tier"] == "") & _enraged_shatter, "_sniper_tier"] = "PERMISSION SPECIAL"
    _rank.loc[_is35 & (_rank["_sniper_tier"] == ""), "_sniper_tier"] = "BASE"

    _tier_rank = {"SNIPER CRIT": 4, "STRONG": 3, "PERMISSION SPECIAL": 2, "BASE": 1, "": 0}
    _rank["_tier_rank"] = _rank["_sniper_tier"].map(_tier_rank).fillna(0).astype(int)



    # 3.5 Sniper Spec: do NOT show BASE on the Smash board (only show 50%+ move procs).
    _rank = _rank[~(_is35 & (_rank["_sniper_tier"] == "BASE"))].copy()

    # Rank: Locks (if present) -> Sniper tier (3.5 only) -> Conf (no EV)
    if "LOCK" in _rank.columns:
        _rank["_is_lock"] = (_rank.get("LOCK", "").astype(str).str.strip() == "🔒").astype(int)
        if "_tier_rank" in _rank.columns:
            _rank = _rank.sort_values(["_is_lock", "_tier_rank", "Conf_SOG"], ascending=[False, False, False], kind="mergesort")
        else:
            _rank = _rank.sort_values(["_is_lock", "Conf_SOG"], ascending=[False, False], kind="mergesort")
    else:
        if "_tier_rank" in _rank.columns:
            _rank = _rank.sort_values(["_tier_rank", "Conf_SOG"], ascending=[False, False], kind="mergesort")
        else:
            _rank = _rank.sort_values(["Conf_SOG"], ascending=[False], kind="mergesort")

    top = _rank.head(int(top_n))

    if top.empty:
        st.info("No SOG plays meet the Berserker gates on this slate.")
    else:
        for _, r in top.iterrows():
                player = str(r.get("Player", "") or "").strip()
                game = str(r.get("Game", "") or "").strip()

                line = r.get("SOG_Line", "")
                odds = r.get("SOG_Odds_Over", "")
                call = str(r.get("SOG_Call", "") or "").strip()

                conf = r.get("Conf_SOG", "")
                matrix = str(r.get("Matrix_SOG", "") or "").strip()
                expl, crit = _derive_badges(r)
                eng = _engine_badge("SOG", r)
                badges = f"{eng} {str(r.get('EV_Signal','') or '').strip()} {str(r.get('LOCK','') or '').strip()} {expl} {crit}".strip()

                # Pretty line/odds strings
                try:
                    l_str = "" if line is None or (isinstance(line, float) and math.isnan(line)) else f"{float(line):.1f}"
                except Exception:
                    l_str = str(line)
                try:
                    o_str = "" if odds is None or (isinstance(odds, float) and math.isnan(odds)) else f"{int(round(float(odds))):d}"
                except Exception:
                    o_str = str(odds)

                headline = f"**{player}** — {game}"
                betline = f"SOG {l_str}+  ({o_str})" if (l_str or o_str) else "SOG"

                meta = []
                if matrix:
                    meta.append(matrix)
                if conf != "" and conf is not None:
                    try:
                        meta.append(f"Conf {float(conf):.0f}")
                    except Exception:
                        meta.append(f"Conf {conf}")
                if call:
                    meta.append(call)  # includes Shot Anchor / DUE labels when present

                # SOG 3.5 Sniper tier label
                try:
                    if float(r.get("SOG_Line", 0) or 0) >= 3.5:
                        tier = str(r.get("_sniper_tier", "") or "").strip()
                        if tier:
                            meta.append(tier)
                except Exception:
                    pass

                # Combat HUD icons (SOG) — card line (presentation-only)
                try:
                    _line_s = float(r.get("SOG_Line", float("nan")))
                except Exception:
                    _line_s = float("nan")
                try:
                    _conf_s = float(r.get("Conf_SOG", 0) or 0)
                except Exception:
                    _conf_s = 0.0
                _mat_s = str(r.get("Matrix_SOG", r.get("Matrix", "")) or "").strip().lower()
                try:
                    _si = float(r.get("ShotIntent", r.get("SI", r.get("ShotIntent_SOG", r.get("SI_SOG", float("nan"))))))
                except Exception:
                    _si = float("nan")
                try:
                    _sipct = float(r.get("ShotIntent_Pct", r.get("ShotIntentPct", r.get("SI_Pct", float("nan")))))
                except Exception:
                    _sipct = float("nan")
                try:
                    _mu = float(r.get("SOG_mu", r.get("SOG_Mu", r.get("mu_sog", float("nan")))))
                except Exception:
                    _mu = float("nan")
                try:
                    _xga = float(r.get("opp_5v5_xGA60", r.get("Opp_5v5_xGA60", float("nan"))))
                except Exception:
                    _xga = float("nan")
                try:
                    _rg = float(r.get("Reg_Gap_S10", r.get("RegGap_S10", float("nan"))))
                except Exception:
                    _rg = float("nan")                # Card-line logic mirrors the NEW SOG Jungle (2.5) HUD:
                # Universe: line<=2.5 + Matrix=Green + Conf>=75 (EV ignored)
                _mat_green = (_mat_s == "green")  # strict match for card line

                _base25 = (
                    _mat_green
                    and (not math.isnan(_line_s)) and (_line_s > 0) and (_line_s <= 2.5)
                    and (_conf_s >= 75)
                )

                # Core inputs (safe)
                _l20 = _num(r.get("L20_Rate_SOG", r.get("L20_Rate_S", r.get("L20_Rate", float("nan")))) , float("nan"))
                _share = _num(r.get("Player_5v5_SOG_Share", r.get("Player_5v5_SOG_SOG_Share", r.get("Player_5v5_SOGShare", float("nan")))) , float("nan"))
                _oppsa = _num(r.get("Opp_SOG_Against_L50", r.get("OppSA_L50", r.get("Opp_SA_L50", float("nan")))) , float("nan"))
                _drought = _num(r.get("Drought_SOG", r.get("Drought_S", 0)), 0.0)

                # Moves (presentation-only; same keys as HUD)
                swipe_on = _base25 and (not math.isnan(_l20)) and (_l20 >= 3.0)
                volley_on = _base25 and (not math.isnan(_share)) and (_share >= 16.0)
                rage_on = swipe_on and volley_on
                overdrive_on = _base25 and (not math.isnan(_l20)) and (_l20 >= 3.4) and volley_on

                locked_loaded_on = _base25 and (_conf_s >= 82.0)  # macro bar starts at 82
                patience_on = _base25 and (int(_drought) == 1)
                bloodthirst_on = patience_on and (not math.isnan(_xga)) and (_xga >= 2.48)

                # Armor tiers (xGA)
                armor_iii = (not math.isnan(_xga)) and (_xga >= 2.55)
                armor_ii  = (not math.isnan(_xga)) and (_xga >= 2.50)
                armor_i   = (not math.isnan(_xga)) and (_xga >= 2.46)

                # Shots allowed badge
                barrage_on = _base25 and (not math.isnan(_oppsa)) and (_oppsa >= 27.5)

                # Siege + killer specials (2.5)
                siege_on = _base25 and swipe_on and volley_on and (_conf_s >= 83.0) and (not math.isnan(_xga)) and (_xga >= 2.50)
                mythic_siege_on = _base25 and (not math.isnan(_l20)) and (_l20 >= 3.2) and volley_on and (_conf_s >= 83.0) and (not math.isnan(_xga)) and (_xga >= 2.50)
                shots_allowed_siege_on = _base25 and barrage_on and volley_on and (not math.isnan(_xga)) and (_xga >= 2.50) and swipe_on

                # Icons (keep existing SOG icon set)
                _hud = []
                if swipe_on:
                    _hud.append(_svg_icon("sog_basic_swipe.svg", "Backbone (L20)", "wl-sog"))
                if volley_on:
                    _hud.append(_svg_icon("sog_berserker_volley.svg", "Role (Share)", "wl-sog"))
                if rage_on:
                    _hud.append(_svg_icon("sog_berserkers_rage.svg", "Strong (L20+Share)", "wl-sog"))
                if overdrive_on:
                    _hud.append(_svg_icon("sog_enraged_strike.svg", "Elite Shooter", "wl-sog"))
                if locked_loaded_on:
                    _hud.append(_svg_icon("sog_locked_loaded.svg", "Locked & Loaded (Conf Spike)", "wl-sog"))
                if patience_on:
                    _hud.append(_svg_icon("sog_berserkers_patience.svg", "Drought", "wl-sog"))
                if bloodthirst_on:
                    _hud.append(_svg_icon("sog_bloodthirst.svg", "Drought+Armor", "wl-sog"))
                if armor_iii:
                    _hud.append(_svg_icon("sog_env_paralysis.svg", "Armor III (xGA)", "wl-sog"))
                elif armor_ii:
                    _hud.append(_svg_icon("sog_env_paralysis.svg", "Armor II (xGA)", "wl-sog"))
                elif armor_i:
                    _hud.append(_svg_icon("sog_env_paralysis.svg", "Armor I (xGA)", "wl-sog"))
                if barrage_on:
                    _hud.append(_svg_icon("sog_berserker_siege.svg", "Shots Allowed", "wl-sog"))
                if siege_on:
                    _hud.append(_svg_icon("sog_berserker_siege.svg", "SIEGE", "wl-sog"))

                proc_icons = "".join(_hud)

                # Card descriptors (match GOALS/ASSISTS: PLAYABLE • tags — meta)
                combo_tags = ["PLAYABLE"]

                # Specials first
                if mythic_siege_on:
                    combo_tags.append("Mythic Siege")
                elif shots_allowed_siege_on:
                    combo_tags.append("Shots-Allowed Siege")
                elif siege_on:
                    combo_tags.append("SIEGE")

                # Tier-ish labels (2.5)
                if overdrive_on:
                    combo_tags.append("Elite Shooter")
                elif rage_on:
                    combo_tags.append("Strong")
                elif volley_on:
                    combo_tags.append("Role")
                elif swipe_on:
                    combo_tags.append("Backbone")

                # Macro / timing / env descriptors
                if swipe_on and (_conf_s >= 80.0):
                    combo_tags.append("Backbone+Conf Spike")
                elif locked_loaded_on:
                    combo_tags.append("Conf Spike")

                if bloodthirst_on:
                    combo_tags.append("Drought+Armor")
                elif patience_on:
                    combo_tags.append("Drought")

                # Armor tier tag (highest only)
                if armor_iii:
                    combo_tags.append("Armor III")
                elif armor_ii:
                    combo_tags.append("Armor II")
                elif armor_i:
                    combo_tags.append("Armor I")

                # Keep it tight like GOALS/ASSISTS
                combo_s = " • ".join(combo_tags[:6])

                meta_s = " | ".join([m for m in meta if m])
                card_line = f"{proc_icons} <span style=\"font-weight:800;\">{combo_s}</span><span style=\"opacity:0.8;\">{' — ' if combo_s else ''}{meta_s}</span>"
                st.markdown(
                    f"""
        <div class="wl-card wl-accent-orange">
          <div style="display:flex;justify-content:space-between;gap:10px;">
            <div style="font-size:16px;line-height:1.2;">
              {headline}
              <div style="opacity:0.9;margin-top:4px;">{betline}</div>
            </div>
            <div style="font-size:16px;white-space:nowrap;">{badges}</div>
          </div>
          <div style="margin-top:6px;font-size:12px;opacity:0.92;line-height:1.2;">{card_line}</div>
        </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Why it fires (SOG)
                _v = r.get("SOG_Why", "")
                _why_tags = "" if (_v is None or (isinstance(_v, float) and math.isnan(_v))) else str(_v).strip()
                if not _why_tags:
                    try:
                        _why_tags = _sog_why(r)
                    except Exception:
                        _why_tags = ""
                with st.expander("Why it fires", expanded=False):
                    _render_why_it_fires_rich("SOG", r, _why_tags)






    st.markdown("---")

    show_table(df_s, sog_cols, "SOG View")


# =========================
# GOAL
# =========================
elif page == "GOALS (0.5)":

    st.markdown(_page_title_html("GOALS (0.5)", "GOALS"), unsafe_allow_html=True)

    df_g = df_f.copy()

    # --- Odds aliasing: treat GOALS(0.5) as Anytime Goal (ATG) when Goal_* is missing ---
    for _col in ["Goal_Line", "Goal_Odds_Over", "Goal_Book"]:
        if _col not in df_g.columns:
            df_g[_col] = pd.NA

    # Coerce numeric odds/lines
    df_g["Goal_Line"] = pd.to_numeric(df_g["Goal_Line"], errors="coerce")
    df_g["Goal_Odds_Over"] = pd.to_numeric(df_g["Goal_Odds_Over"], errors="coerce")

    # Fall back from ATG_* (common BDL naming)
    if "ATG_Line" in df_g.columns:
        df_g["Goal_Line"] = df_g["Goal_Line"].fillna(pd.to_numeric(df_g["ATG_Line"], errors="coerce"))
    if "ATG_Odds_Over" in df_g.columns:
        df_g["Goal_Odds_Over"] = df_g["Goal_Odds_Over"].fillna(pd.to_numeric(df_g["ATG_Odds_Over"], errors="coerce"))

    # Book string fill: handle NaN -> empty
    gb = df_g["Goal_Book"].astype("string")
    gb = gb.fillna("").replace("nan", "").replace("None", "")
    if "ATG_Book" in df_g.columns:
        ab = df_g["ATG_Book"].astype("string").fillna("").replace("nan", "").replace("None", "")
        gb = gb.mask(gb.str.strip().eq(""), ab)
    df_g["Goal_Book"] = gb

    df_g["_cg"] = safe_num(df_g, "Conf_Goal", 0)
    df_g = df_g.sort_values(["_cg"], ascending=[False]).drop(columns=["_cg"], errors="ignore")

    st.sidebar.subheader("Goal Filters")
    show_all = st.sidebar.checkbox("Show all players (ignore filters)", value=False)
    min_conf = st.sidebar.slider("Min Conf (Goal)", 0, 100, 80, 1)
    color_pick = st.sidebar.multiselect(
        "Colors (Goal)",
        ["green", "yellow", "blue", "red"],
        default=["green", "yellow", "blue"]
    )

    if not show_all:
        df_g = df_g[df_g["Conf_Goal"].fillna(0) >= min_conf]
        if "Color_Goal" in df_g.columns and color_pick:
            df_g = df_g[df_g["Color_Goal"].isin(color_pick)]

    df_g["Green"] = df_g.get("Green_Goal", False).map(lambda x: "🟢" if bool(x) else "")

    goal_cols = [
        "Game",
        "Player", "Pos",
        "Tier_Tag",

        # --- LINE / ODDS FIRST (bet slip view) ---
        "Goal_Line",
        "Goal_Odds_Over",
        "Goal_Book",

        # --- SIGNALS ---
        "Green",
        
        "Conf_Goal", "Matrix_Goal",

        # --- GOALS HUD MATH (combo core) ---
        "Opp_SOG_Against_L10",
        "iXG%",
        "Opp_DefWeak",
        "Team_GF_Avg_L5",
        "opp_5v5_xGA60",
        "Med10_SOG",
        "Avg5_SOG",
        "ShotIntent",
        "ShotIntent_Pct",
        "Drought_G", "Best_Drought",

        # --- WHY IT FIRES (label) ---
        "ATG_Call",

        # --- EV / MODEL (display only) ---
        "ATG_Model%", "ATG_Imp%", "ATG_EV%", "Plays_EV_ATG",

        # --- GOALIE / CONTEXT ---
        "Opp_Goalie", "Opp_SV", "Opp_GAA", "Goalie_Weak",

        # --- result / bookkeeping ---
        "Line", "Odds", "Result",
    ]
# Signals-first extras

    df_g["Markets"] = df_g.apply(build_markets_pills, axis=1)

    g = df_g.get("Green_Goal", (df_g.get("Green","") == "🟢"))

    e = df_g["Plays_EV_ATG"] if "Plays_EV_ATG" in df_g.columns else pd.Series([""]*len(df_g), index=df_g.index)

    p = df_g["ATG_EV%"] if "ATG_EV%" in df_g.columns else pd.Series([None]*len(df_g), index=df_g.index)

    df_g["EV_Signal"] = [build_ev_signal(gg, ee, pp) for gg, ee, pp in zip(g, e, p if hasattr(p, "__iter__") else [p]*len(df_g))]

    df_g["LOCK"] = [build_lock_badge(gg, ee) for gg, ee in zip(g, e)]
    legend_signals()
    render_odds_implied_reference(location="main")
    _f = render_market_filter_bar(default_min_conf=60, key_prefix="goal")

    try:
        df_g = apply_market_filters(
            df_g,
            _f,
            green_col="Green_Goal",
            ev_icon_col="Plays_EV_ATG",
            conf_col="Conf_Goal",
            matrix_col="Matrix_Goal",
            lock_col="LOCK",
        )
    except Exception:
        pass


    # --- DPS ranking + filters (Board-style; presentation only) ---
    df_g = add_best_proc_cols(df_g, 'GOALS')
    df_g = apply_dps_filters_ui(df_g, 'GOALS', key_prefix='goals')




    # === SMASH PLAYS (GOALS) ===
    _render_badge_legend_inline()
    st.subheader("⭐ Smash Plays — Goals")

    render_valhalla_gate("GOALS")


    _g = df_g.copy()

    # Robust column picks (tracker schema varies)
    line_col = None
    for _c in ["Goal_Line", "Goal_Line", "Goals_Line"]:
        if _c in _g.columns:
            line_col = _c
            break
    if line_col is None:
        line_col = "Goal_Line"  # fallback

    matrix_col = None
    for _c in ["Matrix_Goal", "Matrix_Goals", "Matrix_ATG", "ATG_Matrix", "Matrix"]:
        if _c in _g.columns:
            matrix_col = _c
            break
    if matrix_col is None:
        matrix_col = "Matrix_Goal"

    conf_col = None
    for _c in ["Conf_Goal", "Conf_Goals", "Conf_ATG", "ATG_Conf", "Conf"]:
        if _c in _g.columns:
            conf_col = _c
            break
    if conf_col is None:
        conf_col = "Conf_Goal"

        # Hard gates (GOALS Beta Gate)
    m_matrix = _g[matrix_col].astype(str).str.strip().str.upper().isin(["GREEN", "🟢"])
    m_line   = (pd.to_numeric(_g.get(line_col, 0), errors="coerce") == 0.5)
    m_conf   = (pd.to_numeric(_g.get(conf_col, 0), errors="coerce").fillna(0) >= 85)

    # Pull key GOALS columns (schema varies)
    _oppsog = pd.to_numeric(_g.get("Opp_SOG_Against_L10", np.nan), errors="coerce")
    _xga    = pd.to_numeric(_g.get("opp_5v5_xGA60", np.nan), errors="coerce")
    _ixg    = pd.to_numeric(_g.get("iXG%", _g.get("iXG_pct", _g.get("iXG_Pct", np.nan))), errors="coerce")
    _share  = pd.to_numeric(_g.get("Player_5v5_SOG_Share", np.nan), errors="coerce")
    _drought= pd.to_numeric(_g.get("Drought_G", _g.get("Drought_Goal", _g.get("Drought", np.nan))), errors="coerce")
    _teamgf = pd.to_numeric(_g.get('Team_GF_Avg_L5', _g.get('Team_GF_L5', _g.get('Team_GF_L5_Avg', np.nan))), errors='coerce')

    # Odds for +odds lane (optional)
    odds_col = None
    for _c in ["Goal_Odds","ATG_Odds","Odds_Goal","Odds_Goals","Odds","Price","Goal_Price","ATG_Price"]:
        if _c in _g.columns:
            odds_col = _c
            break
    _odds = pd.to_numeric(_g.get(odds_col, np.nan), errors="coerce") if odds_col else pd.Series(np.nan, index=_g.index)

    # Keep ungraded for tonight, but only W/L for historical (schema-safe)
    _out = _g.get("Outcome_Goals", pd.Series("", index=_g.index)).astype(str).str.upper()
    _ms  = _g.get("Match_Status_Goals", pd.Series("", index=_g.index)).astype(str).str.upper()
    m_grade = (_out.isin(["W", "L"]) | (_ms.ne("GRADED")))

    # Multi-path Beta Gate (GOALS)
    m_opp   = _oppsog.fillna(-999) >= 29
    m_xga49 = _xga.fillna(-999) >= 2.49
    m_gf25  = _teamgf.fillna(-999) >= 2.5

    ixg94    = _ixg.fillna(-999) >= 94
    ixg97    = _ixg.fillna(-999) >= 97
    drought2 = _drought.fillna(0) >= 2

    # Path A: Armor/Env finisher (no OppSOG required)
    path_armor  = m_xga49 & (ixg94 | drought2)

    # Path B: Frenzy lane (OppSOG + Hot Team)
    path_frenzy = m_opp & m_gf25

    # Path C: Funnel Sniper (OppSOG + iXG>=97)
    path_sniper = m_opp & ixg97


    # Path D: Berserker Aggression (Env Mix) — xGA>=2.55 & iXG>=92 & TeamGF>=3.0 (+150 odds requirement)
    #        Elite Team Scoring tier — same but TeamGF>=3.8 (no odds requirement)
    m_xga55 = _xga.fillna(-999) >= 2.55
    ixg92   = _ixg.fillna(-999) >= 92
    gf30    = _teamgf.fillna(-999) >= 3.0
    gf38    = _teamgf.fillna(-999) >= 3.8
    # Odds: prefer Goal_Odds_Over; fallback to ATG_Odds_Over; allow NaN only for elite tier
    _odds_over = pd.to_numeric(_g.get("Goal_Odds_Over", _g.get("ATG_Odds_Over", np.nan)), errors="coerce")
    odds150 = _odds_over.fillna(-999) >= 150
    path_envmix = m_xga55 & ixg92 & gf30 & odds150
    path_envmix_elite = m_xga55 & ixg92 & gf38

    eligible = path_armor | path_frenzy | path_sniper | path_envmix | path_envmix_elite

    _g = _g[m_matrix & m_line & m_conf & m_grade & eligible].copy()

    _g["_valhalla"] = (pd.to_numeric(_g.get("opp_5v5_xGA60", np.nan), errors="coerce").fillna(0) > 2.50).astype(int)
    _g["_avg5"] = pd.to_numeric(_g.get("Avg5_SOG", 0), errors="coerce").fillna(0)
    _g["_conf"] = pd.to_numeric(_g.get("Conf_Goal", 0), errors="coerce").fillna(0)
    _g = _g.sort_values(["_valhalla","_avg5","_conf"], ascending=[False, False, False], kind="mergesort")

    top_n_g = st.slider("Show top plays (Goals)", 3, 25, 10, 1, key="goal_smash_topn")
    topg = _g.head(int(top_n_g))

    for _, r in topg.iterrows():
        player = str(r.get("Player", "") or "").strip()
        game = str(r.get("Game", "") or "").strip()
        line = r.get(line_col, r.get("Goals_Line", ""))
        odds = r.get("Goal_Odds_Over", r.get("Goal_Odds_Over", r.get("Goals_Odds_Over", "")))
        conf = r.get("Conf_Goal", "")
        avg5 = r.get("Avg5_SOG", "")
        xga = r.get("opp_5v5_xGA60", "")

        # Support intent tags (NOT gates)
        si = _num(r.get("ShotIntent", 0), 0.0)
        sip = _num(r.get("ShotIntent_Pct", 0), 0.0)
        intent_tags = []
        if si >= 3.9: intent_tags.append("ShotIntent ELITE")
        elif si >= 3.75: intent_tags.append("ShotIntent STRONG")
        elif si >= 3.5: intent_tags.append("ShotIntent VOL")
        if sip >= 97.5: intent_tags.append("Intent% ELITE")
        elif sip >= 95: intent_tags.append("Intent% STRONG")

        val = bool(pd.to_numeric(pd.Series([xga]), errors="coerce").fillna(0).iloc[0] > 2.50)
        val_tag = "Valhalla xGA>2.50" if val else "xGA≤2.50 (tough)"


        # Combat HUD icons (GOALS) — card line
        try:
            _line_g = float(line)
        except Exception:
            _line_g = None
        _mat_g = str(matrix).strip().lower() if 'matrix' in locals() else str(r.get("Matrix_Goal","")).strip().lower()
        try:
            _conf_g = float(conf)
        except Exception:
            _conf_g = 0.0
        _stance_ok = (_line_g == 0.5) and (_mat_g == "green") and (_conf_g >= 85)

        _hud = []
        if _stance_ok:
            _hud.append(_svg_icon("base.svg", "Base Attack (Stance)", "wl-goals"))
        if xga is not None:
            if float(xga) >= 2.50:
                _hud.append(_svg_icon("armor_shred.svg", "Armor Shred", "wl-goals"))
            else:
                _hud.append(_svg_icon("armor_buff.svg", "Enemy Fortified", "wl-goals wl-keep"))
        if si is not None and float(si) >= 3.4 and _stance_ok:
            _hud.append(_svg_icon("fenrir_claw.svg", "Fenrir’s Claw", "wl-goals"))
        if avg5 is not None and float(avg5) >= 3.5 and _stance_ok:
            _hud.append(_svg_icon("fury.svg", "Warlord Fury", "wl-goals"))

        # Special / Ultimate icons
        if _stance_ok and xga is not None and float(xga) >= 2.50 and ((si is not None and float(si) >= 3.4) or (avg5 is not None and float(avg5) >= 3.5)):
            _hud.append(_svg_icon("smash.svg", "Warlord Smash Attack", "wl-goals"))
        if _stance_ok and xga is not None and float(xga) >= 2.50 and (avg5 is not None and float(avg5) >= 3.5):
            _hud.append(_svg_icon("valhalla.svg", "FOR VALHALLA!", "wl-goals"))

        _hud_html = "".join(_hud)

        meta = []
        try: meta.append(f"Conf {float(conf):.0f}")
        except Exception: pass
        try: meta.append(f"Avg5 {float(avg5):.1f} SOG")
        except Exception: pass
        try: meta.append(f"xGA {float(xga):.2f}")
        except Exception: pass
        # ---- GOALS card descriptors (ASSISTS-style) ----
        meta_s = " | ".join([m for m in (meta + intent_tags + [val_tag]) if m])

        # Pull key signals (safe) - GOALS beta columns
        oppsog = _safe_float(r.get("Opp_SOG_Against_L10", None), None)
        ixg = _safe_float(r.get("iXG%", r.get("iXG_pct", r.get("iXG_Pct", None))), None)
        share = _safe_float(r.get("Player_5v5_SOG_Share", None), None)
        drought_g = _safe_float(r.get("Drought_G", r.get("Drought_Goal", r.get("Drought_Goals", None))), None)

        _xga = xga
        _oppsog = oppsog
        _ixg = ixg
        _share = share
        _drg = drought_g

        # ENV descriptor (locked wording)
        env_tag = ""
        env_icon = ""
        if _xga is not None:
            if _xga >= 2.52:
                env_tag = "Defense Collapsing"
                env_icon = "armor_shred.svg"
            elif _xga >= 2.49:
                env_tag = "Armor Shred"
                env_icon = "armor_shred.svg"
            else:
                env_tag = "Enemy Fortified"
                env_icon = "armor_buff.svg"

        # Core lanes
        shot_funnel = bool(_oppsog is not None and _oppsog >= 29)
        elite_finisher = bool(_ixg is not None and _ixg >= 97)
        finisher_crit = bool(_ixg is not None and _ixg >= 99 and (_xga is not None and _xga >= 2.55))
        driver_share = bool(_share is not None and _share >= 15)
        drought_proc = bool(_drg is not None and _drg >= 2)

        armor_annihilation = bool(elite_finisher and (_xga is not None and _xga >= 2.52))
        conf_spike = False  # GOALS tiers are conf-free (beta)
        valhalla_spike = False  # GOALS tiers are conf-free (beta)

        tyr_unleashed = bool(shot_funnel and driver_share and (_xga is not None and _xga >= 2.52) and (_ixg is not None and _ixg >= 97))

        # Strict beta "playability" signal (presentation only)
        core_gate = bool(shot_funnel and (_xga is not None and _xga >= 2.49))
        proof = bool((_ixg is not None and _ixg >= 93.5) or drought_proc or armor_annihilation or tyr_unleashed)
        playable = bool(core_gate and proof) or armor_annihilation or tyr_unleashed
        longshot = bool(shot_funnel and not playable)  # OppSOG-only lane for +odds

        # Short bold tags (like ASSISTS)
        combo_tags = []
        if playable:
            combo_tags.append("PLAYABLE")
        elif longshot:
            combo_tags.append("Longshot +Odds")

        if tyr_unleashed:
            combo_tags.append("Tyr’s Wrath Unleashed")
        if armor_annihilation:
            combo_tags.append("Armor Annihilation")

        if shot_funnel and ("Tyr’s Wrath Unleashed" not in combo_tags):
            combo_tags.append("Shot Funnel")
        if env_tag:
            combo_tags.append(env_tag)

        if finisher_crit:
            combo_tags.append("Finisher Crit")
        elif elite_finisher:
            combo_tags.append("Elite Finisher")

        if driver_share and ("Tyr’s Wrath Unleashed" not in combo_tags):
            combo_tags.append("Driver Share")
        if drought_proc:
            combo_tags.append("Drought Proc")
        if valhalla_spike:
            combo_tags.append("Valhalla Spike")
        elif conf_spike:
            combo_tags.append("Conf Spike")

        combo_s = " • ".join(combo_tags[:6])

        # Icons (keep board symbols; show only what fired)
        icons = []
        icons.append(_svg_icon("base.svg", "Base Shell", "wl-goals"))
        if env_icon:
            icons.append(_svg_icon(env_icon, env_tag, "wl-goals wl-keep"))
        if shot_funnel:
            icons.append(_svg_icon("fury.svg", "Shot Funnel", "wl-goals"))
        if elite_finisher:
            icons.append(_svg_icon("fenrir_claw.svg", "Elite Finisher", "wl-goals"))
        if tyr_unleashed:
            icons.append(_svg_icon("fury.svg", "Tyr’s Wrath Unleashed", "wl-goals"))
        if armor_annihilation:
            icons.append(_svg_icon("stack_armor_annihilation.svg", "Armor Annihilation", "wl-goals"))
        if drought_proc:
            icons.append(_svg_icon("stack_fury_shredder.svg", "Drought Proc", "wl-goals"))
        if valhalla_spike:
            icons.append(_svg_icon("valhalla.svg", "Valhalla Spike", "wl-goals"))
        elif conf_spike:
            icons.append(_svg_icon("smash.svg", "Conf Spike", "wl-goals"))

        proc_icons = "".join([i for i in icons if i])


        # Meta tail (beta: show the key GOALS columns that matter)
        _meta_parts = []
        if conf is not None:
            _meta_parts.append(f"Conf {conf:.0f}")
        if oppsog is not None:
            _meta_parts.append(f"OppSOG_L10 {oppsog:.0f}")
        if ixg is not None:
            _meta_parts.append(f"iXG% {ixg:.1f}")
        if share is not None:
            _meta_parts.append(f"Share {share:.1f}")
        if drought_g is not None and drought_g >= 1:
            _meta_parts.append(f"Drought {int(drought_g)}")
        if xga is not None:
            _meta_parts.append(f"xGA {xga:.2f}")
        meta_s = " | ".join(_meta_parts)

        betline = f"GOAL {line} @ {odds}" if (line or odds) else ""
        headline = f"<b>{player}</b> — {game}" if game else f"<b>{player}</b>"

        st.markdown(
            f"""
         <div class=\"wl-card wl-accent-red\">
           <div style=\"display:flex;justify-content:space-between;gap:10px;\">
             <div style=\"font-size:16px;line-height:1.2;\">
               {headline}
               <div style=\"opacity:0.9;margin-top:4px;\">{betline}</div>
             </div>
             <div style=\"font-size:16px;white-space:nowrap;\">{_engine_badge('GOALS', r)} {str(r.get('LOCK','') or '').strip()}</div>
           </div>
           <div style=\"margin-top:6px;font-size:12px;opacity:0.95;line-height:1.2;\">{proc_icons} <span style=\"font-weight:800;\">{combo_s}</span><span style=\"opacity:0.8;\">{" — " if combo_s else ""}{meta_s}</span></div>
         </div>
                    """,
            unsafe_allow_html=True,
        )

        _why_tags = str(r.get("Goal_Why", r.get("Why", "")) or "").strip()
        with st.expander("Why it fires", expanded=False):
            _why_sections_header("GOALS")
            _render_why_it_fires_rich("GOALS", r, _why_tags)

    st.markdown("---")

    show_table(df_g, goal_cols, "GOAL (1+) View")




# =========================
# POWER PLAY
# =========================
elif page == "Power Play":
    st.subheader("⚡ Power Play (PPP / 5v4)")
    st.caption("Read-only view: PP usage + PP creation + team PP vs opponent PK + PPP drought. Does not change model probabilities yet.")

    # Aliases (engine naming -> app naming)
    alias_map = {
        "PP_TOI_min": "PP_TOI",
        "PP_TOI_per_game": "PP_TOI_PG",
        "PP_iP60": "PP_Points60",
    }
    for src, dst in alias_map.items():
        if dst not in df_f.columns and src in df_f.columns:
            df_f[dst] = df_f[src]

    # PP unit tag/icon
    if "PP_Role" in df_f.columns:
        def _pp_role_tag(x):
            try:
                v = int(float(x))
            except Exception:
                return "PP0"
            return "PP1" if v >= 2 else ("PP2" if v == 1 else "PP0")
        df_f["PP_UnitTag"] = df_f["PP_Role"].apply(_pp_role_tag)
        df_f["PP_Unit"] = df_f["PP_UnitTag"].map({"PP1": "🔌 PP1", "PP2": "🔋 PP2"}).fillna("")
    else:
        df_f["PP_Unit"] = ""

    st.sidebar.subheader("Power Play Filters")
    unit_sel = st.sidebar.multiselect("PP Unit", ["PP1", "PP2"], default=["PP1", "PP2"], key="pp_unit_sel")
    min_pp_toi = st.sidebar.slider("Min PP TOI / game", 0.0, 10.0, 1.0, 0.25, key="pp_min_toi")
    min_ppp_drought = st.sidebar.slider("Min PPP Drought (games)", 0, 12, 0, 1, key="pp_min_ppp_drought")
    tier_opts = ["A","B","C"] if "PP_Tier" in df_f.columns else []
    tier_sel = st.sidebar.multiselect("PP Tier", tier_opts, default=tier_opts, key="pp_tier_sel") if tier_opts else []
    path_opts = ["Shooter","Distributor","Hybrid","Passenger"] if "PP_Path" in df_f.columns else []
    path_sel = st.sidebar.multiselect("PP Path", path_opts, default=path_opts, key="pp_path_sel") if path_opts else []


    df_pp = df_f.copy()
    if "PP_UnitTag" in df_pp.columns:
        df_pp = df_pp[df_pp["PP_UnitTag"].isin(unit_sel)]

    if "PP_TOI_PG" in df_pp.columns:
        df_pp = df_pp[pd.to_numeric(df_pp["PP_TOI_PG"], errors="coerce").fillna(0.0) >= float(min_pp_toi)]

    if "Drought_PPP" in df_pp.columns:
        df_pp = df_pp[pd.to_numeric(df_pp["Drought_PPP"], errors="coerce").fillna(0).astype(int) >= int(min_ppp_drought)]
    if "PP_Tier" in df_pp.columns and tier_sel:
        df_pp = df_pp[df_pp["PP_Tier"].astype(str).str.upper().isin([t.upper() for t in tier_sel])]
    if "PP_Path" in df_pp.columns and path_sel:
        df_pp = df_pp[df_pp["PP_Path"].astype(str).isin(path_sel)]


    # Sort best-first (only by columns that exist)
    sort_cols = [c for c in ["PP_Matchup", "PP_BOOST", "PP_Points60", "PP_TOI_PG", "Drought_PPP"] if c in df_pp.columns]
    if sort_cols:
        df_pp = df_pp.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    pp_cols = [
        "Game",
        "Player", "Pos", "Team", "Opp",
        "Tier_Tag",
        "PP_Unit",
        "PP_TOI_PG",
        "PP_TeamShare_pct",
        "PP_TOI_stability",
        "PP_Tier",
        "PP_Path",
        "PP_BOOST",
        "PP_TOI_Pct",
        "PP_Points60",
        "PP_iXG60",
        "PP_iXA60",
        "Team_PP_xGF60",
        "Opp_PK_xGA60",

        # Opportunity context (season-to-date team rates)
        "Team_PPO_PG",
        "Opp_TSH_PG",
        "Team_PP_Eff",
        "PP_Opps_Score",
        "Opp_Penalty_Score",
        "PP_Opportunity",

        "PP_Matchup",
        "PPP10_total",
        "Drought_PPP",
    ]

    show_table(df_pp, pp_cols, "Power Play (5v4) — Usage, creation, matchup, PPP drought")


elif page == "🧪 Dagger Lab":
    st.subheader("🧪 Dagger Lab")
    st.caption("Explain *why* the 🗡️ shows up — role, stability, environment, and creation. This page is diagnostic only (does not change EV).")

    # Focus on players with any dagger context available
    df_lab = df_f.copy()

    # Build dagger icon (HARD GATE) — recompute every time (ignore any 🗡️ column in CSV)
    df_lab["🗡️"] = ""

    proof_col = "Assist_PP_Proof" if "Assist_PP_Proof" in df_lab.columns else None
    proof = df_lab[proof_col].astype(bool) if proof_col else pd.Series(False, index=df_lab.index)

    apc = pd.to_numeric(df_lab.get("Assist_ProofCount", 0), errors="coerce").fillna(0)
    adg = pd.to_numeric(df_lab.get("Assist_Dagger", 0), errors="coerce").fillna(0)
    ppt = df_lab.get("PP_Tier", "").astype(str).str.upper()

    # HARD gate:
    # 1) Explicit proof, OR
    # 2) 4-of-4 assist proofs, OR
    # 3) Elite dagger score (>=82), OR
    # 4) PP A/B + strong proof (>=3) + decent dagger (>=60)
    mask = (proof | (apc >= 4) | (adg >= 82) | ((ppt.isin(["A","B"])) & (apc >= 3) & (adg >= 60)))
    df_lab.loc[mask, "🗡️"] = "🗡️"

    # Prefer listing dagger candidates first
    sort_cols = []
    if "🗡️" in df_lab.columns: sort_cols.append("🗡️")
    if "Assist_Dagger" in df_lab.columns: sort_cols.append("Assist_Dagger")
    if "PP_BOOST" in df_lab.columns: sort_cols.append("PP_BOOST")
    if sort_cols:
        df_lab = df_lab.sort_values(by=[c for c in sort_cols if c in df_lab.columns], ascending=[False]*len(sort_cols))

    # Player selector (Game context helps)
    label_col = "Player" if "Player" in df_lab.columns else df_lab.columns[0]
    game_col = "Game" if "Game" in df_lab.columns else None
    def _lab_label(r):
        name = str(r.get(label_col, "")).strip()
        game = str(r.get(game_col, "")).strip() if game_col else ""
        pp_tier = str(r.get("PP_Tier", "")).upper().strip()

        tier_tag = f"[PP {pp_tier}]" if pp_tier and pp_tier not in {"NAN", "NONE"} else  "[PP ?]"

        if game_col:
            return f"{name} {tier_tag}  —  {game}"
        return f"{name} {tier_tag}"


    options = df_lab.apply(_lab_label, axis=1).tolist()
    if not options:
        st.warning("No rows loaded.")
    else:
        pick = st.selectbox("Select a player", options, index=0)
        idx = options.index(pick)
        r = df_lab.iloc[idx]

        # Core fields (safe pulls)
        def g(col, default=None):
            return r.get(col, default) if col in r.index else default

        # Display helpers: convert numpy types / NaN to clean primitives for UI
        def _disp(v):
            if v is None:
                return ""
            try:
                if pd.isna(v):
                    return ""
            except Exception:
                pass
            try:
                import numpy as _np
                if isinstance(v, _np.generic):
                    v = v.item()
            except Exception:
                pass
            return v

        # Back-compat: derive PP fields if missing from the tracker
        if "PP_Tier" not in df_lab.columns and "PP_Role" in df_lab.columns:
            def _pp_tier(v):
                try:
                    x = int(float(v))
                except Exception:
                    return ""
                return "A" if x >= 2 else ("B" if x == 1 else "C")
            df_lab["PP_Tier"] = df_lab["PP_Role"].apply(_pp_tier)

        if "PP_Path" not in df_lab.columns and ("PP_iXG60" in df_lab.columns or "PP_iXA60" in df_lab.columns):
            def _pp_path_row(rr):
                try:
                    role = int(float(rr.get("PP_Role", 0) or 0))
                except Exception:
                    role = 0
                if role <= 0:
                    return "Passenger"
                ixg = pd.to_numeric(rr.get("PP_iXG60", np.nan), errors="coerce")
                ixa = pd.to_numeric(rr.get("PP_iXA60", np.nan), errors="coerce")
                tot = (0.0 if pd.isna(ixg) else float(ixg)) + (0.0 if pd.isna(ixa) else float(ixa))
                if tot <= 0:
                    return "Passenger"
                share = (ixg / tot)
                if share >= 0.60:
                    return "Shooter"
                if share <= 0.40:
                    return "Passer"
                return "Balanced"
            df_lab["PP_Path"] = df_lab.apply(_pp_path_row, axis=1)

        if "PP_TeamShare_pct" not in df_lab.columns and "PP_TOI_Pct_Game" in df_lab.columns:
            df_lab["PP_TeamShare_pct"] = pd.to_numeric(df_lab["PP_TOI_Pct_Game"], errors="coerce")

        if "PP_TOI_stability" not in df_lab.columns and "PP_TOI" in df_lab.columns and "PP_TOI_min" in df_lab.columns:
            toi = pd.to_numeric(df_lab["PP_TOI"], errors="coerce")
            toi_min = pd.to_numeric(df_lab["PP_TOI_min"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                df_lab["PP_TOI_stability"] = (100.0 * (toi_min / toi)).clip(lower=0.0, upper=100.0)

        if "PP_Env_Score" not in df_lab.columns and "PP_Matchup" in df_lab.columns:
            df_lab["PP_Env_Score"] = pd.to_numeric(df_lab["PP_Matchup"], errors="coerce").fillna(50.0)

        if "PP_BOOST" not in df_lab.columns and ("PP_Tier" in df_lab.columns):
            def _boost(rr):
                t = str(rr.get("PP_Tier","")).upper().strip()
                s = pd.to_numeric(rr.get("PP_TeamShare_pct", np.nan), errors="coerce")
                e = pd.to_numeric(rr.get("PP_Env_Score", np.nan), errors="coerce")
                if t == "A" and pd.notna(s) and pd.notna(e) and s >= 20.0 and e >= 60.0:
                    return "ON"
                if t == "B" and pd.notna(s) and pd.notna(e) and s >= 17.5 and e >= 62.0:
                    return "ON"
                return ""
            df_lab["PP_BOOST"] = df_lab.apply(_boost, axis=1)

        # Re-pull the picked row after back-compat mutations
        r = df_lab.iloc[idx]

        st.markdown(f"### {g('Player','(player)')}  {g('Team','')} vs {g('Opp','')}  —  {g('Game','')}")
        cols = st.columns(5)
        cols[0].metric("🗡️", g("🗡️",""))
        cols[1].metric("Assist Dagger", g("Assist_Dagger", None))
        cols[2].metric("PP Tier", g("PP_Tier", ""))
        cols[3].metric("PP Path", g("PP_Path", ""))
        cols[4].metric("PP Boost", g("PP_BOOST", None))

        # Build a non-binding "Dagger Strength" explainer score (0–100)
        base_conf = float(g("Conf_Assists", 0) or 0)
        pp_env = float(g("PP_Env_Score", 50) or 50)
        stab = float(g("PP_TOI_stability", 50) or 50)
        share = float(g("PP_TeamShare_pct", g("PP_TOI_Pct_Game", 0)) or 0)
        tier = str(g("PP_Tier","C") or "C").upper().strip()

        tier_bonus = {"A": 20.0, "B": 10.0, "C": 0.0}.get(tier, 0.0)
        # Normalize base_conf roughly 0–100 (your confs are already 0–100)
        strength = 0.40*base_conf + 0.20*pp_env + 0.15*stab + 0.15*min(100.0, share*4.0) + 0.10*tier_bonus
        strength = max(0.0, min(100.0, round(strength, 1)))

        st.markdown("#### Dagger Strength (explain-only)")
        st.progress(float(strength)/100.0)
        st.caption(f"Strength: **{strength}/100** — for explanation only (does not feed EV).")

        # Breakdown cards
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Base Proof")
            st.write({
                "Conf_Assists": _disp(g("Conf_Assists")),
                "Matrix_Assists": _disp(g("Matrix_Assists")),
                "Assist_ProofCount": _disp(g("Assist_ProofCount")),
                "Assist_Why": _disp(g("Assist_Why")),
            })
        with c2:
            st.markdown("#### PP Layer")
            st.write({
                "PP_Tier": _disp(g("PP_Tier")),
                "PP_Path": _disp(g("PP_Path")),
                "PP_TeamShare_pct": _disp(g("PP_TeamShare_pct")),
                "PP_TOI_Pct_Game": _disp(g("PP_TOI_Pct_Game")),
                "PP_TOI_stability": _disp(g("PP_TOI_stability")),
                "PP_Env_Score": _disp(g("PP_Env_Score")),
                "PP_Matchup": _disp(g("PP_Matchup")),
                "PP_iXA60": _disp(g("PP_iXA60")),
                "PP_BOOST": _disp(g("PP_BOOST")),
            })

        st.markdown("#### Quick read")
        msgs = []
        if tier in ("A","B"):
            msgs.append(f"PP role: **Tier {tier}** (real contributor).")
        else:
            msgs.append("PP role: **Tier C** (passenger / cosmetic).")

        path = str(g("PP_Path","Passenger") or "Passenger")
        if path == "Shooter":
            msgs.append("PP Path: **Shooter** (goals/SOG skew).")
        elif path == "Distributor":
            msgs.append("PP Path: **Distributor** (assists skew).")
        elif path == "Hybrid":
            msgs.append("PP Path: **Hybrid** (assists + goals).")
        else:
            msgs.append("PP Path: **Passenger** (low PP usage impact).")
        if pp_env >= 65:
            msgs.append("Environment: **high PP volume** expected.")
        elif pp_env <= 40:
            msgs.append("Environment: **low PP volume** — beware empty whistles.")
        else:
            msgs.append("Environment: **neutral**.")
        if stab >= 65:
            msgs.append("Deployment: **stable PP minutes**.")
        elif stab <= 40:
            msgs.append("Deployment: **coach blender risk**.")
        if bool(g("Assist_PP_Proof", False)):
            msgs.append("🗡️ Trigger: **ON** (PP assist proof passed).")
        else:
            msgs.append("🗡️ Trigger: **OFF** (didn't meet proof gates).")
        st.write(" ".join(msgs))



elif page == "🪜 Ladder Alerts":
    st.subheader("🪜 Ladder Alerts")
    st.caption("Scan Top-K alt lines (BDL) starting from each player’s baseline line. Use presets to go from normal ladders (2.5→3.5) to rare ‘nuclear’ rungs.")
    legend_signals()

    # Use the filtered slate (sidebar filters apply)
    df_calc = df_f.copy()

    # Market first (so presets can set sane defaults)
    cA, cB, cC, cD, cE = st.columns([1.0, 1.0, 1.0, 1.2, 1.0])
    with cA:
        ladder_market = st.selectbox("Market", ["SOG", "Points", "Assists", "Goal"], index=0, key="ladder_market")

    preset = st.selectbox(
        "Preset",
        ["Baseline+", "Standard", "Elite Volume", "Nuclear"],
        index=1,
        key="ladder_preset",
        help="Baseline+ starts at the player's mainline. Standard/Elite/Nuclear are higher-rung scans (rare-volume modes).",
    )

    # Per-market baseline defaults (most of the slate lives here)
    base_defaults = {"SOG": 2.5, "Points": 1.5, "Assists": 0.5, "Goal": 0.5}
    base_line_default = float(base_defaults.get(ladder_market, 1.5))

    # Preset thresholds
    if preset == "Baseline+":
        _min_line, _min_ev, _min_model = base_line_default, 6.0, 10.0
    elif preset == "Standard":
        _min_line, _min_ev, _min_model = max(base_line_default, 3.5 if ladder_market == "SOG" else base_line_default), 8.0, 12.0
    elif preset == "Elite Volume":
        _min_line, _min_ev, _min_model = max(base_line_default, 5.5), 10.0, 10.0
    else:
        _min_line, _min_ev, _min_model = max(base_line_default, 7.5), 6.0, 10.0

    with cB:
        min_line = st.number_input("Min line", value=float(_min_line), step=0.5, key="ladder_min_line_v2")
    with cC:
        min_ev = st.number_input("Min EV%", value=float(_min_ev), step=0.5, key="ladder_min_ev_v2")
    with cD:
        min_model = st.number_input("Min Model%", value=float(_min_model), step=0.5, key="ladder_min_model_v2")
    with cE:
        start_from_baseline = st.checkbox("Start at baseline", value=True, key="ladder_start_baseline")

    # Detect how many ladders we actually have (up to 8)
    max_k = 0
    for k in range(8, 0, -1):
        if f"BDL_{ladder_market}_Line_{k}" in df_calc.columns:
            max_k = k
            break
    if max_k == 0:
        st.info("No BDL alt lines found in this CSV (BDL_*_Line_i columns missing).")
    else:
        ladd = build_ladder_alerts(
            df_calc,
            market=ladder_market,
            min_line=float(min_line),
            min_ev=float(min_ev),
            min_model_pct=float(min_model),
            top_k=int(max_k),
            start_from_baseline=bool(start_from_baseline),
        )
        if ladd.empty:
            st.write("No ladder alerts met your thresholds.")
        else:
            st.caption(f"Showing {len(ladd)} alerts (Top-K={max_k}).")

            # --- UI signals (no math changes) ---
            try:
                # Basic KPIs
                best_ev = pd.to_numeric(ladd.get("EV%", pd.Series(dtype=float)), errors="coerce").max()
                best_model = pd.to_numeric(ladd.get("Model%", pd.Series(dtype=float)), errors="coerce").max()
                uniq_players = ladd["Player"].nunique() if "Player" in ladd.columns else len(ladd)
                uniq_games = ladd["Game"].nunique() if "Game" in ladd.columns else None

                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.markdown(f'<div class="wl-card"><h4>Alerts</h4><div class="wl-big">{len(ladd)}</div></div>', unsafe_allow_html=True)
                with k2:
                    st.markdown(f'<div class="wl-card"><h4>Players</h4><div class="wl-big">{int(uniq_players)}</div></div>', unsafe_allow_html=True)
                with k3:
                    be = "" if pd.isna(best_ev) else f"{best_ev:.1f}%"
                    st.markdown(f'<div class="wl-card"><h4>Best EV%</h4><div class="wl-big">{be}</div></div>', unsafe_allow_html=True)
                with k4:
                    bm = "" if pd.isna(best_model) else f"{best_model:.1f}%"
                    st.markdown(f'<div class="wl-card"><h4>Best Model%</h4><div class="wl-big">{bm}</div></div>', unsafe_allow_html=True)

                # Quick explain (Why + key drivers) for a selected rung
                if {"Player","Market","Line","Why"}.issubset(set(ladd.columns)):
                    st.markdown("#### 🔥 Why this is fire")
                    ladd["_pick_label"] = (
                        ladd["Player"].astype(str) + " — " +
                        ladd["Market"].astype(str) + " " +
                        ladd["Line"].astype(str) + " (" +
                        ladd.get("Book", "").astype(str) + " " +
                        ladd.get("Odds", "").astype(str) + ")"
                    )
                    pick = st.selectbox("Pick an alert to inspect", options=ladd["_pick_label"].tolist(), index=0, key="ladder_pick")
                    row = ladd[ladd["_pick_label"] == pick].iloc[0]

                    # --- UI-only Earned / Tail verdict (no math changes) ---
                    def _num(x):
                        try:
                            if x is None: return None
                            s = str(x).strip()
                            if s == "" or s.lower() == "none" or s.lower() == "nan": return None
                            return float(s)
                        except Exception:
                            return None

                    evv = _num(row.get("EV%"))
                    mod = _num(row.get("Model%"))
                    rung = _num(row.get("Rung"))
                    defweak = _num(row.get("Opp_DefWeak"))
                    proof = str(row.get("Proof","") or "")
                    alert = str(row.get("Alert","") or "")

                    big3 = 0
                    for k in ["OppSOG_L10","OppSOG_L50","5v5Share%","DefV"]:
                        if _num(row.get(k)) is not None:
                            big3 += 1

                    is_star = ("STAR" in proof.upper()) or ("ELITE" in proof.upper())
                    strong_def = (defweak is not None and defweak <= 25)

                    earned_ui = (("GREAT" in alert.upper()) or
                                 (evv is not None and evv >= 8 and mod is not None and mod >= 55 and is_star and (rung is None or rung <= 1.0) and (big3 >= 2)))

                    tail_ui = (not earned_ui) and (
                                (rung is not None and rung >= 2.0) or
                                (mod is not None and mod < 52 and strong_def) or
                                (evv is not None and evv >= 20 and mod is not None and mod < 55)
                              )

                    verdict = "🪜 EARNED LADDER" if earned_ui else ("⚠️ TAIL LADDER (sprinkle only)" if tail_ui else "👀 WORTH A LOOK")
                    vcls = "wl-pill-green" if earned_ui else ("wl-pill-muted" if tail_ui else "wl-pill-yellow")

                    st.markdown(
                        f"<div class='wl-card { 'wl-card-earned' if earned_ui else ('wl-card-tail' if tail_ui else '') }'>"
                        f"<span class='wl-pill {vcls}'>{verdict}</span>"
                        f"<span class='wl-pill wl-pill-blue'>Model {'' if mod is None else f'{mod:.1f}%'} </span>"
                        f"<span class='wl-pill wl-pill-green'>EV {'' if evv is None else f'{evv:.1f}%'} </span>"
                        f"<span class='wl-pill wl-pill-muted'>Rung {'' if rung is None else f'{rung:.1f}'} </span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )


                    # Pills summarizing the drivers (already computed columns)
                    pills = []
                    if "DefV" in row.index and str(row.get("DefV","")).strip():
                        pills.append(("DefV " + str(row.get("DefV")), "wl-pill-red"))
                    if "SlotSA60" in row.index and str(row.get("SlotSA60","")).strip():
                        pills.append(("SlotSA60 " + str(row.get("SlotSA60")), "wl-pill-blue"))
                    if "5v5Share%" in row.index and str(row.get("5v5Share%","")).strip():
                        pills.append(("5v5 Share " + str(row.get("5v5Share%")) + "%", "wl-pill-green"))
                    if "OppSOG_L10" in row.index and str(row.get("OppSOG_L10","")).strip():
                        pills.append(("Opp SOG L10 " + str(row.get("OppSOG_L10")), "wl-pill-yellow"))
                    if "OppSOG_L50" in row.index and str(row.get("OppSOG_L50","")).strip():
                        pills.append(("Opp SOG L50 " + str(row.get("OppSOG_L50")), "wl-pill-yellow"))

                    pill_html = "".join([f'<span class="wl-pill {cls}">{lab}</span>' for lab, cls in pills]) or '<span class="wl-pill wl-neutral">No drivers found</span>'
                    st.markdown(
                        f'''
<div class="wl-card">
  <div style="margin-bottom:8px;">{pill_html}</div>
  <div style="font-size:14px;line-height:1.35;opacity:0.95;"><b>Why:</b> {str(row.get("Why",""))}</div>
</div>
                        ''',
                        unsafe_allow_html=True,
                    )
                    # Cleanup helper col so it doesn't leak into the table below
                    ladd = ladd.drop(columns=["_pick_label"], errors="ignore")
            except Exception:
                pass

            cols = ["Player","Team","Game","Market","Line","Odds","Book","Model%","EV%","Rung","DefV","SlotSA60","5v5Share%","OppSOG_L10","OppSOG_L50","Why"]
            cols = [c for c in cols if c in ladd.columns]
            cfg = build_column_config(ladd, cols)
            st.dataframe(ladd[cols], width="stretch", hide_index=True, column_config=cfg)


elif page == "📟 Calculator":
    st.subheader("📟 EV + Stake Calculator")
    st.caption("Pick a player from today’s CSV and the calculator will auto-load their line/odds/model%. Override anything if you want.")
    legend_signals()

    # Base dataset (use filtered df so user can narrow by sidebar search/team/game)
    df_calc = df_f.copy()

    # Player dropdown
    players = []
    if "Player" in df_calc.columns:
        players = sorted([p for p in df_calc["Player"].dropna().astype(str).unique().tolist() if p.strip()])

    c1, c2, c3 = st.columns([1.4, 1.0, 1.0])
    with c1:
        player_sel = st.selectbox("Player", options=["(Manual)"] + players, index=0, key="calc_player")
    with c2:
        market = st.selectbox("Market", ["Points", "SOG", "Goal", "Assists"], index=0, key="calc_market")
    with c3:
        bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=1000.0, step=50.0, key="calc_bankroll")

    mcfg = _calc_market_map(market)

    # -------------------------
    # 🚨 Ladder Alerts (UI only)
    # -------------------------
    
    # -------------------------
    # Ladder Alerts (UI only)
    # -------------------------
    st.info("Ladders moved: use the **🪜 Ladder Alerts** page for full ladder scanning.")

    # Pull row for the selected player (first match)
    row = None
    if player_sel != "(Manual)" and "Player" in df_calc.columns:
        hit = df_calc[df_calc["Player"].astype(str) == str(player_sel)]
        if len(hit) > 0:
            row = hit.iloc[0]

    # Resolve auto values (mainline by default)
    auto_line = None
    auto_odds = None
    auto_p = None
    auto_ev = None
    auto_conf = None
    auto_matrix = None
    auto_green = None
    auto_ev_icon = None

    def _get_num_from_row(r, col):
        try:
            if r is None or col not in df_calc.columns:
                return None
            v = r.get(col, None)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            return float(v)
        except Exception:
            return None

    # Helper: pick from Alt-line columns if present
    def _resolve_alt_cols(market_name: str, idx: int) -> tuple[float | None, float | None, float | None]:
        """Return (line, odds, p_model) for alt index idx (1..K) if present.

        Tracker schema (from odds_ev_bdl.py):
          - line/odds: BDL_{M}_Line_{i}, BDL_{M}_Odds_{i}
          - model prob: {M}_p_model_over_{i}  (or {M}_Model%_{i})
        """
        if row is None:
            return (None, None, None)

        M = str(market_name).strip()

        lc = f"BDL_{M}_Line_{idx}"
        oc = f"BDL_{M}_Odds_{idx}"
        pc = f"{M}_p_model_over_{idx}"
        mp = f"{M}_Model%_{idx}"

        l = _get_num_from_row(row, lc)
        o = _get_num_from_row(row, oc)

        p = _get_num_from_row(row, pc)
        if p is None:
            mpp = _get_num_from_row(row, mp)
            if mpp is not None:
                p = float(mpp) / 100.0

        return (l, o, p)

    if row is not None:
        auto_line = _get_num_from_row(row, mcfg["line_col"])
        auto_odds = _get_num_from_row(row, mcfg["odds_col"])
        # model prob: prefer p_model_over (0-1), else Model% (0-100)
        auto_p = _get_num_from_row(row, mcfg.get("p_model_col", ""))
        if auto_p is None:
            mp = _get_num_from_row(row, mcfg.get("modelpct_col", ""))
            if mp is not None:
                auto_p = float(mp) / 100.0
        auto_ev = _get_num_from_row(row, mcfg.get("evpct_col", ""))
        auto_conf = _get_num_from_row(row, mcfg.get("conf_col", ""))
        try:
            auto_matrix = str(row.get(mcfg.get("matrix_col",""), "")).strip()
        except Exception:
            auto_matrix = ""
        try:
            auto_green = bool(row.get(mcfg.get("green_col",""), False))
        except Exception:
            auto_green = False
        try:
            auto_ev_icon = str(row.get(mcfg.get("ev_icon_col",""), "")).strip()
        except Exception:
            auto_ev_icon = ""

    # Unique keys per (player, market) so switching doesn't "carry" stale values
    # (avoid truncation collisions by hashing)
    import hashlib
    key_prefix = "calc_" + hashlib.md5(f"{str(player_sel)}|{market}".encode()).hexdigest()

    # If alt lines exist for this market, allow selecting which line to cash-check
    alt_labels = ["Mainline"]
    if row is not None:
        M = str(market).strip()
        # show only available BDL alt lines for this market
        for i in range(1, 7):
            lc = f"BDL_{M}_Line_{i}"
            if lc in df_calc.columns:
                lv = _get_num_from_row(row, lc)
                if lv is not None:
                    alt_labels.append(f"Alt {i} ({lv:.1f})")

    if len(alt_labels) > 1:
        pick = st.selectbox("Line source", alt_labels, index=0, key=f"{key_prefix}_pick")
        if pick.startswith("Alt"):
            try:
                idx = int(pick.split()[1])
            except Exception:
                idx = None
            if idx:
                l2, o2, p2 = _resolve_alt_cols(market, idx)
                if l2 is not None:
                    auto_line = l2
                if o2 is not None:
                    auto_odds = o2
                if p2 is not None:
                    auto_p = p2

    def _parse_american_odds_text(s: str) -> float | None:
        """Parse American odds from user text. Accepts +120, -110, unicode minus."""
        try:
            if s is None:
                return None
            t = str(s).strip()
            if not t:
                return None
            t = t.replace("−", "-")
            if t.startswith("+"):
                t = t[1:]
            return float(int(t))
        except Exception:
            return None

    st.markdown("### Inputs (auto-filled when player selected)")
    i1, i2, i3, i4 = st.columns([1.0, 1.0, 1.0, 1.2])
    with i1:
        line = st.number_input("Line", value=float(auto_line) if auto_line is not None else 0.5, step=0.5, key=f"{key_prefix}_line")
    with i2:
        odds_str = st.text_input(
            "Odds (American)",
            value=str(int(auto_odds)) if auto_odds is not None else "-110",
            help="Examples: -110, +120",
            key=f"{key_prefix}_odds_str",
        )
        odds = _parse_american_odds_text(odds_str)
        if odds is None:
            st.warning("Invalid odds format. Use -110 or +120.")
            odds = float(int(auto_odds)) if auto_odds is not None else -110.0
    with i3:
        override_model = st.checkbox("Override Model%", value=False, key=f"{key_prefix}_ovp")
        if (auto_p is not None) and (not override_model):
            model_prob = float(auto_p)
            st.metric("Model win probability", f"{model_prob*100.0:.1f}%")
        else:
            model_prob = st.slider(
                "Model win probability (%)",
                1.0, 99.0,
                float(auto_p * 100.0) if auto_p is not None else 55.0,
                0.5,
                key=f"{key_prefix}_p"
            ) / 100.0
    with i4:
        use_manual_ev = st.checkbox("Override EV% manually", value=False, key=f"{key_prefix}_usem")
        manual_ev = st.number_input("Manual EV% (if overriding)", value=float(auto_ev) if auto_ev is not None else 0.0, step=0.5, key=f"{key_prefix}_mev")

    s1, s2, s3 = st.columns([1.0, 1.0, 1.0])
    with s1:
        kelly_frac = st.slider("Kelly Fraction", 0.0, 1.0, 0.25, 0.05, key=f"{key_prefix}_kf")
    with s2:
        max_pct = st.slider("Max Stake cap (% bankroll)", 0.0, 0.20, 0.05, 0.01, key=f"{key_prefix}_cap")
    with s3:
        st.caption("Tip: Best bets are **🟢 + 💰**. Calculator helps size the bet.")

    # Use shared odds math (single source of truth)
    imp, ev_pct_calc, kelly, dec = calc_ev_pct_and_kelly(float(model_prob), float(odds))


    p = float(model_prob)

    # EV% is calculated from (model_prob, odds) via calc_ev_pct_and_kelly above.
    ev_pct = float(ev_pct_calc)
    if use_manual_ev:
        ev_pct = float(manual_ev)

    fair_dec = (1.0 / p) if p > 0 else 999.0

    stake = bankroll * float(kelly) * float(kelly_frac)
    stake = min(stake, bankroll * float(max_pct))


    st.markdown("### Results")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Decimal Odds", f"{dec:.3f}")
    r2.metric("Implied Prob", f"{imp*100:.1f}%")
    r3.metric("Model Prob", f"{p*100:.1f}%")
    r4.metric("EV%", f"{ev_pct:+.1f}%")

    r1, r2, r3 = st.columns(3)
    r1.metric("Fair Decimal", f"{fair_dec:.3f}")
    r2.metric("Edge (Model-Imp)", f"{(p-imp)*100:.1f}%")
    r3.metric("Kelly % (full)", f"{kelly*100:.2f}%")

    r1, r2 = st.columns(2)
    r1.metric("Recommended Stake ($)", f"{stake:.2f}")
    r2.metric("Stake % bankroll", f"{(stake/bankroll*100.0) if bankroll>0 else 0.0:.2f}%")

    # Signal callout
    label, emoji, why = warlord_call(ev_pct, kelly)
    if ev_pct >= 5:
        st.success(f"{emoji} **{label}** — {why}")
    elif ev_pct >= 0:
        st.warning(f"{emoji} **{label}** — {why}")
    else:
        st.error(f"{emoji} **{label}** — {why}")

    # Player context panel (when selected)
    if row is not None:
        st.markdown("### Player context (from today’s CSV)")
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        c1.metric("Conf", f"{auto_conf:.0f}" if auto_conf is not None else "—")
        c2.metric("Matrix", auto_matrix if auto_matrix else "—")
        c3.metric("Earned Green", "🟢" if auto_green else "—")
        c4.metric("+EV", "💰" if str(auto_ev_icon).strip() == "💰" else "—")

        if auto_ev is not None and not use_manual_ev:
            st.caption(f"EV% from CSV: **{auto_ev:+.1f}%** (you can override if shopping a different book/price).")
        st.caption("Remember: **Calculator is sizing + price check**. Your model signals are still king.")



elif page == "🧾 Log Bet":
    st.subheader("🧾 Log Bet — append-only Warlord Ledger")
    st.caption("Enter only what you actually bet. Everything else auto-fills from today’s model CSV.")
    legend_signals()

    df_log = df_f.copy()

    # Paths
    ledger_dir, betslip_path, events_path = _ledger_paths(OUTPUT_DIR)
    st.caption(f"Ledger folder: `{ledger_dir}`")

    # Player dropdown
    players = []
    if "Player" in df_log.columns:
        players = sorted([p for p in df_log["Player"].dropna().astype(str).unique().tolist() if p.strip()])

    c1, c2, c3 = st.columns([1.6, 1.0, 1.0])
    with c1:
        player_sel = st.selectbox("Player", options=players, index=0 if players else None, key="log_player")
    with c2:
        market = st.selectbox("Market", ["Points", "SOG", "Goal", "Assists"], index=0, key="log_market")
    with c3:
        book = st.text_input("Book", value="", placeholder="DK / FD / MGM / CZR...", key="log_book")

    # Find player row
    row = None
    if player_sel and "Player" in df_log.columns:
        hit = df_log[df_log["Player"].astype(str) == str(player_sel)]
        if len(hit) > 0:
            row = hit.iloc[0]

    mcfg = _calc_market_map(market)

    def _get_num_from_row(r, col):
        try:
            if r is None or col not in df_log.columns:
                return None
            v = r.get(col, None)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            return float(v)
        except Exception:
            return None

    # Auto values
    auto_date = str(row.get("Date", "")) if row is not None and "Date" in df_log.columns else ""
    auto_game = str(row.get("Game", "")) if row is not None and "Game" in df_log.columns else ""
    auto_opp = str(row.get("Opp", "")) if row is not None and "Opp" in df_log.columns else ""
    auto_goalie = str(row.get("Opp_Goalie", "")) if row is not None and "Opp_Goalie" in df_log.columns else ""

    auto_line = _get_num_from_row(row, mcfg.get("line_col", "")) if row is not None else None
    auto_odds = _get_num_from_row(row, mcfg.get("odds_col", "")) if row is not None else None

    # model prob: prefer p_model_over (0-1), else Model% (0-100)
    auto_p = _get_num_from_row(row, mcfg.get("p_model_col", "")) if row is not None else None
    if auto_p is None and row is not None:
        mp = _get_num_from_row(row, mcfg.get("modelpct_col", ""))
        if mp is not None:
            auto_p = float(mp) / 100.0

    auto_conf = _get_num_from_row(row, mcfg.get("conf_col", "")) if row is not None else None
    auto_matrix = str(row.get(mcfg.get("matrix_col", ""), "")).strip() if row is not None else ""
    auto_green = bool(row.get(mcfg.get("green_col", ""), False)) if row is not None else False
    auto_ev_icon = str(row.get(mcfg.get("ev_icon_col", ""), "")).strip() if row is not None else ""

    # Extra model context
    auto_tier = str(row.get("Talent_Tier", "")) if row is not None and "Talent_Tier" in df_log.columns else ""
    proof_col = mcfg.get("proof_col", "")
    why_col = mcfg.get("why_col", "")
    auto_proof = int(row.get(proof_col, 0)) if row is not None and proof_col and proof_col in df_log.columns else 0
    auto_why = str(row.get(why_col, "")) if row is not None and why_col and why_col in df_log.columns else ""

    # Inputs you control
    kpref = f"log_{player_sel}_{market}".replace(" ", "_")[:90]
    i1, i2, i3, i4 = st.columns([1.0, 1.0, 1.0, 1.0])
    with i1:
        line = st.number_input("Line", value=float(auto_line) if auto_line is not None else 0.5, step=0.5, key=f"{kpref}_line")
    with i2:
        odds_taken = st.number_input("Odds taken (American)", value=int(auto_odds) if auto_odds is not None else -110, step=5, key=f"{kpref}_odds")
    with i3:
        override_model = st.checkbox("Override Model%", value=False, key=f"{kpref}_ovp")
        if (auto_p is not None) and (not override_model):
            model_prob = float(auto_p)
            st.metric("Model win probability", f"{model_prob*100.0:.1f}%")
        else:
            model_prob = st.slider(
                "Model win probability (%)",
                1.0, 99.0,
                float(auto_p * 100.0) if auto_p is not None else 55.0,
                0.5,
                key=f"{kpref}_p"
            ) / 100.0
    with i4:
        stake_u = st.number_input("Stake (u)", min_value=0.0, max_value=float(MAX_STAKE_U), value=1.0, step=0.25, key=f"{kpref}_u")

    notes = st.text_input("Notes (optional)", value="", key=f"{kpref}_notes")

    # Derived
    imp, ev_pct, kelly, dec = calc_ev_pct_and_kelly(model_prob, odds_taken)
    ev_flag = bool(ev_pct >= 5.0)
    lock_flag = bool(auto_green and ev_flag)

    st.markdown("### Snapshot")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Earned Green", "🟢" if auto_green else "—")
    s2.metric("+EV", "💰" if ev_flag else "—")
    s3.metric("LOCK", "🔒" if lock_flag else "—")
    s4.metric("Stake", f"{stake_u:.2f}u  (${'{:.0f}'.format(stake_u*UNIT_VALUE_USD)})")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Implied %", f"{imp*100:.1f}%")
    s2.metric("Model %", f"{model_prob*100:.1f}%")
    s3.metric("EV% (recalc)", f"{ev_pct:+.1f}%")
    s4.metric("Kelly (full)", f"{kelly*100:.2f}%")

    if row is not None:
        st.caption(f"Model context: Conf={auto_conf:.0f} | Matrix={auto_matrix or '—'} | Tier={auto_tier or '—'} | Proofs={auto_proof} | Why={auto_why or '—'}")
        if auto_ev_icon == '💰':
            st.caption("Note: CSV already flagged this as 💰 at its listed odds — ledger uses the odds you took.")

    # Log button
    if st.button("🧾 Log Bet (append)", use_container_width=True):
        dt_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        date_str = auto_date or datetime.now().strftime('%Y-%m-%d')
        bet_id = make_bet_id(date_str, player_sel, market, line, odds_taken)

        row_out = {
            'bet_id': bet_id,
            'date': date_str,
            'datetime_placed': dt_now,
            'game': auto_game,
            'player': str(player_sel),
            'market': market.upper(),
            'line': float(line),
            'odds_taken': int(odds_taken),
            'book': book.strip() if book.strip() else '',
            'stake_u': float(stake_u),
            'earned_green': int(bool(auto_green)),
            'ev_flag': int(bool(ev_flag)),
            'lock_flag': int(bool(lock_flag)),
            'conf': float(auto_conf) if auto_conf is not None else '',
            'matrix': auto_matrix,
            'model_pct': round(model_prob*100.0, 2),
            'imp_pct': round(imp*100.0, 2),
            'ev_pct': round(ev_pct, 2),
            'tier': auto_tier,
            'proof_count': int(auto_proof),
            'why_tags': auto_why,
            'opp': auto_opp,
            'opp_goalie': auto_goalie,
            'notes': notes,
        }

        _append_csv_row(betslip_path, row_out, BETSLIP_HEADERS)
        st.success(f"Logged: **{bet_id}** → {stake_u:.2f}u")

    # Show recent bets
    try:
        if os.path.exists(betslip_path):
            st.markdown("### Recent logs")
            tail = pd.read_csv(betslip_path).tail(10)
            st.dataframe(tail, use_container_width=True, hide_index=True)
        else:
            st.info("No betslip.csv yet — first log will create it.")
    except Exception as e:
        st.warning(f"Could not read ledger yet: {e}")

elif page == "Guide":
    st.subheader("📘 Guide — How to use")
    st.markdown(r"""
## The 60-second workflow
1) **Start on Board**
   - Sorted best-first (**Best_Conf → Goalie_Weak → Opp_DefWeak**)
   - Look for **Tier_Tag + HOT regression + weak matchup** stacking

2) **Open a market view** (Points / SOG / Goal / Assists)
   - Use **Min Conf** slider to tighten
   - Use **Colors** to hide red
   - Use **Only 🔥** to isolate your shortlist

3) **Use two gates (this is the secret sauce)**
   - **🟢 Earned Green** = “model says playable”
   - **💰 EV Play** = “market is mispriced vs us”
   
**Best bets are when 🟢 and 💰 agree.**

---

## EV / Odds columns — what they mean
**Line** → What must happen to win (threshold)  
**Odds** → Payout price (American odds)  
**Model%** → Our model probability the Over hits  
**Imp%** → Sportsbook implied probability from odds  
**EV%** → Expected value edge (positive = good)  
**💰** → Approved +EV wager (EV% cleared our threshold)

### Line vs Odds (common confusion)
- `Points_Line = 3.0` means **Over 3.0 → 4+ points**
- `Points_Odds_Over = +900` means **risk 1 to win 9**
So “300/900” is **odds**, not the line.

### Milestone mapping (how Overs work)
- 0.5 → **1+**
- 1.0 → **1+**
- 1.5 → **2+**
- 2.0 → **2+**
- 2.5 → **3+**
- 3.0 → **4+**
(Over X.0 = X+1)

### EV% interpretation
- **< 0%** → bad price (-EV)
- **0–4%** → thin edge
- **5–9%** → decent edge
- **10%+** → strong edge (this is where 💰 triggers)

### Why EV fields can be blank
Usually means:
- no odds posted for that player/market yet
- market not offered for that player
- early slate (books post props in waves)

---

## Your key signals
### Matrix_* (Green/Yellow/Red)
- **Green** = baseline conditions met
- **Yellow** = borderline
- **Red** = failed

### Conf_* (0–100)
Confidence after gates/adjustments.

### Earned Green 🟢
This is your strict “playable” rule.

---

## Earned Green rules (plain English)
### 🟢 SOG
Matrix green + confidence gate + volume/intent confirmation.

### 🟢 Points
Matrix green + confidence gate + involvement proofs pass.

### 🟢 Goal
Matrix green + confidence gate + due/env/drought proof hits.

### 🟢 Assists
Matrix green + Conf_Assists ≥ 77 + proof gate passes.

---

## Best daily betting rules
**Safe test phase**
- Only play **🟢 earned greens**
- Prefer ⭐/👑 on big slates
- Prefer HOT regression when choices are close

**A+ stack**
✅ Earned Green 🟢  
✅ HOT regression  
✅ Weak goalie/defense  
✅ Tier ⭐/👑  
✅ 💰 EV% 10+

---

## Troubleshooting
If a market page looks blank:
- Min Conf too high
- Color filters hiding everything
- Odds not posted yet
""")

elif page == "Ledger":
    st.subheader("📜 Ledger — What everything means")

    st.markdown("""
### Core ideas
- **Matrix_*:** quick “signal” (Green/Yellow/Red) based on the model’s conditions.
- **Conf_*:** 0–100 confidence score *after* adjustments (injury, drought bump, etc.).
- **Green_*:** “earned green” rules (your stricter gating) — not just raw confidence.

---
### Badges / Tags
- **🔥** = flagged play tag (your manual/auto “this is a real look” indicator).
- **👑 ELITE / ⭐ STAR** = talent tier tags.
- **⛔ GF GATE** = team scoring environment failed (team’s recent scoring too low).  
  When this triggers, **Goal/Points/Assists confidence is forced to 0** and Matrix becomes **FAIL_GF**.

---
### Colors (how to read them)
- **Matrix colors**
  - **Green** = conditions met
  - **Yellow** = borderline / mixed
  - **Red** = conditions failed
  - **FAIL_GF** = hard fail due to team scoring gate

- **Confidence colors**
  - **Green** = high confidence (your thresholding)
  - **Yellow** = mid
  - **Blue** = lower but usable
  - **Red** = avoid

- **Regression heat**
  - **HOT** = due/overdue (bump potential)
  - **WARM** = mild
  - **COOL** = not due

---
### Key columns (most important)
- **Best_Market / Best_Conf** = which market looks best *for that player*.
- **Reg_Gap_* / Exp_*:** expected vs actual gap (how “due” they are).
- **Goalie_Weak / Opp_DefWeak** = matchup-based vulnerability.
- **ShotIntent / ShotIntent_Pct** = volume + intent proxy for SOG.
- **Assist_ProofCount / Assist_Why** = why assists earned green was triggered.

---
### Injury logic
- **Injury_Status / Injury_Badge / Injury_DFO_Score**
  - GTD knocks confidence down
  - ROLE+ boosts slightly (if you coded it)
  - OUT/IR should be filtered via **Available**
""")

    st.info("If you want, I can generate this ledger automatically from a Python dict so it stays synced when you add columns.")



# =========================
# RAW
# =========================
else:
    st.subheader("Raw CSV (all columns)")
    st.dataframe(df_f, width="stretch", hide_index=True)