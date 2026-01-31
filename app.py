import os
import glob
import math
import re


def render_badge_legend_inline(bg="#eef5ff"):
    """Static inline legend for Dynamite / Critical Strike (presentation-only)."""
    try:
        st.markdown(
            f"""
            <div style="padding:8px 12px;
                        border-radius:10px;
                        background:{bg};
                        border:1px solid rgba(0,0,0,0.12);
                        margin:6px 0 10px 0;
                        font-size:13px;">
              <b>Board Signals</b><br>
              🧨 <b>Dynamite</b> — explosive ceiling edge (high-impact, volatile)<br>
              🗡️ <b>Critical Strike</b> — precision edge with multiple proofs aligned
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass

from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st


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
    mk = str(mkt or "").strip().upper()
    # Market color matching (same palette used by pills)
    if "ASSIST" in mk:
        bg = "rgba(168,85,247,0.14)"
        border = "#a855f7"
    elif "SOG" in mk or "SHOT" in mk:
        bg = "rgba(245,158,11,0.14)"
        border = "#f59e0b"
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


def _render_why_it_fires_rich(mkt: str, r, tags: str = "") -> None:
    """Presentation-only rich WHY block. Does not change any gates/logic."""

    tags_s = str(tags or "").strip()
    if tags_s:
        st.markdown(f"**Tags:** {tags_s}")
    else:
        st.markdown("**Tags:** —")

    # MAIN: for now, show the market tags (human reads them as primary triggers)
    st.markdown(f"**MAIN:** {tags_s if tags_s else '—'}")

    # SUPPORT: show the compact model context line (matrix/conf/ev/heat/gap/drought) if present
    try:
        ctx = []
        # generic picks, market-specific keys are passed via r already
        # (caller provides the right row for the market page)
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
        line_col="ATG_Line",
        odds_col="ATG_Odds_Over",
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
        df = df[_col_bool(df, green_col)]

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

    "ATG_Line": "small",
    "ATG_Book": "small",
    "ATG_Odds_Over": "small",
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
def build_column_config(df: pd.DataFrame, cols: list[str]) -> dict:
    cfg = {}

    for c in cols:
        width = COLUMN_WIDTHS.get(c, "small")

        if c not in df.columns:
            cfg[c] = st.column_config.TextColumn(width=width)
            continue

        if pd.api.types.is_numeric_dtype(df[c]):
            # Betting-friendly numeric formats
            if c.endswith("_Line") or c == "Line":
                cfg[c] = st.column_config.NumberColumn(width=width, format="%.1f")
            elif c.endswith("_Odds_Over") or c == "Odds":
                cfg[c] = st.column_config.NumberColumn(width=width, format="%.0f")
            elif c.endswith("_Model%") or c.endswith("_Imp%") or c.endswith("_EV%"):
                cfg[c] = st.column_config.NumberColumn(width=width, format="%.1f")
            else:
                cfg[c] = st.column_config.NumberColumn(width=width)
        else:
            cfg[c] = st.column_config.TextColumn(width=width)

    return cfg



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
      .wl-orange{ background: rgba(245,158,11,0.14); border-color: rgba(245,158,11,0.35);} /* SOG */
      .wl-red   { background: rgba(239,68,68,0.14);  border-color: rgba(239,68,68,0.35);}  /* Goals */

      /* accent stripe */
      .wl-accent-purple{ background: rgba(168,85,247,0.18); border-left: 5px solid #a855f7; }
      .wl-accent-blue{ background: rgba(59,130,246,0.18); border-left: 5px solid #3b82f6; }
      .wl-accent-orange{ background: rgba(245,158,11,0.18); }{ border-left: 5px solid #f59e0b; }
      .wl-accent-red{ background: rgba(239,68,68,0.18); }{    border-left: 5px solid #ef4444; }
    </style>
    """, unsafe_allow_html=True)

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
        "SOG": "#f59e0b",       # orange
        "SHOTS": "#f59e0b",
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

    def conf_style(v):
        try:
            x = float(v)
        except Exception:
            return ""
        if x >= 80:
            return "background-color:#1f7a1f;color:white;font-weight:700;"
        if x >= 70:
            return "background-color:#b38f00;color:white;font-weight:700;"
        return "background-color:#8b1a1a;color:white;font-weight:700;"

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

    for c in ["Best_Conf", "Conf_Points", "Conf_SOG", "Conf_Goal", "Conf_Assists"]:
        if c in view.columns:
            sty = sty.applymap(conf_style, subset=[c])

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
            "drought": _first_num("Drought_A", "Drought_Assists", default=0),
            "reg_gap": _first_num("Reg_Gap_A10", "Reg_Gap_A", default=0),
            "reg_heat": _first_str("Reg_Heat_A", default=""),
            "mu": _first_num("Assists_mu", "MU_Assists", "U_Assists", "U", "MU", default=0),
            "share_5v5": _first_num("5v5_Share", "Share_5v5", default=0),
            "lsa10": _first_num("L10_Team_SA", "L10SA", "TeamSA10", default=0),
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
            "drought": _first_num("Drought_P", "Drought_Points", default=0),
            "reg_gap": _first_num("Reg_Gap_P10", "Reg_Gap_P", default=0),
            "reg_heat": _first_str("Reg_Heat_P", default=""),
            "mu": _first_num("Points_mu", "MU_Points", "U_Points", "U", "MU", default=0),
            "share_5v5": _first_num("5v5_Share", "Share_5v5", default=0),
            "lsa10": _first_num("L10_Team_SA", "L10SA", "TeamSA10", default=0),
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
            "line": _first_num("Goals_Line", "Line_Goal", "Line_Goals", default=0),
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


def select_all_market_rows(row, thr_conf: int, thr_ev: float, thr_drought: int, thr_gap: float, thr_heat: float) -> list[dict]:
    """Return ALL market bundles that pass the hard gate (multi-market allowed)."""
    cands = [
        _bundle_for_market(row, "sog"),
        _bundle_for_market(row, "assists"),
        _bundle_for_market(row, "points"),
        _bundle_for_market(row, "goal"),
    ]
    elig = [c for c in cands if _passes_smash(c, thr_conf, thr_ev, thr_drought, thr_gap, thr_heat)]
    # sort strongest first
    elig.sort(key=lambda x: (x.get("ev", 0.0), x.get("conf", 0), x.get("model", 0.0)), reverse=True)
    return elig

def select_best_market_row(row, thr_conf: int, thr_ev: float, thr_drought: int, thr_gap: float, thr_heat: float):
    cands = [
        _bundle_for_market(row, "sog"),
        _bundle_for_market(row, "assists"),
        _bundle_for_market(row, "points"),
        _bundle_for_market(row, "goal"),
    ]
    elig = [c for c in cands if _passes_smash(c, thr_conf, thr_ev, thr_drought, thr_gap, thr_heat)]
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
        Vengeance is Coming!
        Cook the Books!
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
for m in ["Points","GOAL (1+)","Assists","ATG","SOG"]:
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

    if m.upper() in ("GOAL (1+)", "GOAL 1+", "ATG", "ANYTIME GOAL"):
        m = "Goal"

    if slate_games >= 8:
        return {"SOG": 77, "Points": 77, "Goal": 78, "Assists": 77}[m]
    elif slate_games >= 5:
        return {"SOG": 77, "Points": 77, "Goal": 80, "Assists": 77}[m]
    else:
        return {"SOG": 77, "Points": 77, "Goal": 83, "Assists": 77}[m]




# --- Earned greens (match YOUR columns)
thr_s = _green_conf_threshold("SOG", slate_games)
thr_p = _green_conf_threshold("Points", slate_games)
thr_s = _green_conf_threshold("SOG", slate_games)
# =========================
# GOAL — earned green (v2 proof-count + tier-aware drought)
# =========================

thr_g = _green_conf_threshold("GOAL (1+)", slate_games)

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

df["Green_SOG"] = (
    (safe_num(df, "Conf_SOG", 0) >= thr_s)
    & (safe_str(df, "Matrix_SOG", "").str.strip().str.lower() == "green")
    & (
        (safe_num(df, "ShotIntent_Pct", 0) >= 90)
        | sog_volume_proof
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
    if _get(r, "Drought_SOG", 0) >= 3:
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

earned_gate = (df["Assist_ProofCount"] >= 2) | (is_star & (df["Assist_ProofCount"] >= 1))

assists_green_earned = (
    (safe_str(df, "Matrix_Assists", "").str.strip().str.lower() == "green")
    & (safe_num(df, "Conf_Assists", 0) >= 77)
    & earned_gate
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
    ["Board", "Points", "Assists", "SOG", "GOAL (1+)", "Power Play", "🧪 Dagger Lab", "🪜 Ladder Alerts", "Guide", "Ledger", "Raw CSV", "📟 Calculator", "🧾 Log Bet"],
    index=0
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
    # Purple = Assists, Dark Blue = Points, Orange = SOG, Red = Goals
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
      .wl-accent-orange{ background: rgba(245,158,11,0.18); }{ border-left: 5px solid #f59e0b; }
      .wl-accent-red{ background: rgba(239,68,68,0.18); }   { border-left: 5px solid #ef4444; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
      <div class="wl-board-legend">
        <span class="wl-board-pill"><span class="wl-board-dot" style="background:#a855f7"></span> Assists</span>
        <span class="wl-board-pill"><span class="wl-board-dot" style="background:#0b1b3a"></span> Points</span>
        <span class="wl-board-pill"><span class="wl-board-dot" style="background:#f59e0b"></span> SOG</span>
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
            return "wl-accent-orange"
        if "goal" in bm0:
            return "wl-accent-red"
        return ""

    # -------------------------
    # Board-only gating (does NOT affect other pages)
    # -------------------------
    thr_conf = int(st.sidebar.number_input("Smash min Confidence", min_value=0, max_value=100, value=int(THR_CONF_DEFAULT), step=1, key="board_thr_conf"))
    thr_ev = float(st.sidebar.number_input("Smash min EV%", value=float(THR_EV_DEFAULT), step=0.5, key="board_thr_ev"))
    thr_drought = int(st.sidebar.number_input("Smash min Drought", min_value=0, max_value=20, value=int(THR_DROUGHT_DEFAULT), step=1, key="board_thr_drought"))
    thr_gap = float(st.sidebar.number_input("Smash min Reg Gap", value=float(THR_REG_GAP_DEFAULT), step=0.5, key="board_thr_gap"))
    thr_heat = float(st.sidebar.number_input("Smash min Reg Heat", value=float(THR_REG_HEAT_DEFAULT), step=0.5, key="board_thr_heat"))

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
        passing = select_all_market_rows(_r, thr_conf, thr_ev, thr_drought, thr_gap, thr_heat)
        if not passing:
            continue
        for b in passing:
            rr = _r.copy()
            rr["Best_Market"] = b["label"]
            rr["Best_Conf"] = b["conf"]
            rr["Best_EV%"] = b["ev"]
            rr["Best_Model%"] = b["model"]
            rr["Best_Line"] = b["line"]
            rr["Best_Drought"] = b["drought"]
            rr["Best_Reg_Gap10"] = b["reg_gap"]
            rr["Best_Reg_Heat10"] = b["reg_heat"]
            best_rows.append(rr)

    df_board = pd.DataFrame(best_rows) if best_rows else df_board_src.iloc[0:0]
    df_b = sort_board(df_board)

    board_cols = [
        "Game",
        "Player", "Pos",
        "Tier_Tag",
        "Markets",
        "EV_Signal",
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

    # Top candidates: prefer locks, then EV signal, then confidence (presentation-only)
    _rank = df_b.copy()
    _rank["_is_lock"] = _rank["LOCK"].astype(str).str.len().fillna(0).astype(int) if "LOCK" in _rank.columns else 0
    _rank["_is_ev"] = _rank["EV_Signal"].astype(str).str.contains("💰").fillna(False).astype(int) if "EV_Signal" in _rank.columns else 0
    _rank["_bc"] = pd.to_numeric(_rank.get("Best_Conf", 0), errors="coerce").fillna(0)
    _rank = _rank.sort_values(["_is_lock","_is_ev","_bc"], ascending=[False, False, False])

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

            headline = f"**{player}** — {game}  ·  {expl}{crit} **{bm}** {flames} ·  Conf **{bc}**"
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
                # Market-aware WHY renderer (MAIN / SUPPORT / TONIGHT)
                mkt_raw = str(bm or "").strip().upper()
                if mkt_raw.startswith("SOG"):
                    mkt = "SOG"
                elif mkt_raw.startswith("POINT"):
                    mkt = "POINTS"
                elif mkt_raw.startswith("ASSIST"):
                    mkt = "ASSISTS"
                elif mkt_raw.startswith("ATG"):
                    mkt = "ATG"
                elif mkt_raw.startswith("GOAL"):
                    mkt = "GOALS"
                else:
                    mkt = mkt_raw or "UNKNOWN"

                def _f(x, default=None):
                    try:
                        if pd.isna(x):
                            return default
                        return float(x)
                    except Exception:
                        return default

                def _s(x):
                    return str(x).strip() if x is not None and not pd.isna(x) else ""

                def _heat_to_flames(h: str) -> str:
                    h = _s(h).upper()
                    if "OVERDUE" in h:
                        return "🔥🔥🔥"
                    if h in ("HOT","DUE"):
                        return "🔥🔥"
                    return ""

                mu = line = None
                heat = ""
                gap = None
                drought = None
                proof = None

                if mkt == "SOG":
                    mu = _f(r.get("SOG_mu"))
                    line = _f(r.get("SOG_Line"))
                    heat = _s(r.get("Reg_Heat_S"))
                    gap = _f(r.get("Reg_Gap_S10"))
                    drought = _f(r.get("Drought_SOG"))
                    proof = _f(r.get("SOG_ProofCount"))
                elif mkt == "POINTS":
                    mu = _f(r.get("Points_mu"))
                    line = _f(r.get("Points_Line"))
                    heat = _s(r.get("Reg_Heat_P"))
                    gap = _f(r.get("Reg_Gap_P10"))
                    drought = _f(r.get("Drought_P"))
                    proof = _f(r.get("Points_ProofCount"))
                elif mkt == "ASSISTS":
                    mu = _f(r.get("Assists_mu"))
                    line = _f(r.get("Assists_Line"), 0.5)
                    heat = _s(r.get("Reg_Heat_A"))
                    gap = _f(r.get("Reg_Gap_A10"))
                    drought = _f(r.get("Drought_A"))
                    proof = _f(r.get("Assist_ProofCount"))
                elif mkt in ("GOALS","ATG"):
                    if mkt == "ATG":
                        mu = _f(r.get("ATG_mu"))
                        line = _f(r.get("ATG_Line"), 0.5)
                    else:
                        mu = _f(r.get("Goal_mu"))
                        line = _f(r.get("Goal_Line"), 0.5)
                    heat = _s(r.get("Reg_Heat_G"))
                    gap = _f(r.get("Reg_Gap_G10"))
                    drought = _f(r.get("Drought_G"))
                    proof = _f(r.get("Goal_ProofCount"))

                flames = _heat_to_flames(heat)
                med10_sog = _f(r.get("Med10_SOG"))
                avg5_sog = _f(r.get("Avg5_SOG"))
                share_sog = _f(r.get("Player_5v5_SOG_Share"))
                toi = _f(r.get("TOI_per_game"))
                pp_role = _s(r.get("PP_Role"))
                pp_match = _f(r.get("PP_Matchup"))
                opp_def = _f(r.get("Opp_DefWeak"))
                gk_weak = _f(r.get("Goalie_Weak"))
                opp_sog_l10 = _f(r.get("Opp_SOG_Against_L10"))
                team_sf60 = _f(r.get("team_5v5_SF60"))
                tier_raw = (r.get('Tier_Tag_Best') or r.get('Tier_Tag') or r.get('Tier') or r.get('TierTag') or r.get('Tier_Class') or '')
                tier_u = _s(tier_raw).upper()

                main = ""
                if mkt == "SOG":
                    if (med10_sog is not None and line is not None and med10_sog >= max(4.0, line + 1.0)) or (avg5_sog is not None and line is not None and avg5_sog >= max(4.0, line + 1.0)):
                        main = "High Shot Volume"
                    elif (mu is not None and line is not None and mu >= line + 1.5):
                        main = "μ Expectation (High)"
                    elif (toi is not None and toi >= 18) or (share_sog is not None and share_sog >= 0.12):
                        main = "Usage Dominance"
                elif mkt == "POINTS":
                    if (mu is not None and line is not None and mu >= line + 1.5):
                        main = "μ Expectation (High)"
                    elif flames == "🔥🔥🔥":
                        main = "Bonkers Regression 🔥🔥🔥"
                    elif (gk_weak is not None and gk_weak >= 65) and (opp_def is not None and opp_def >= 60) and (mu is not None and line is not None and mu >= line + 0.75):
                        main = "Scoring Environment"
                elif mkt == "ASSISTS":
                    if (proof is not None and proof >= 4):
                        main = "🗡️ Assist Proof (4/6)"
                    elif (mu is not None and line is not None and mu >= line + 1.5) and (proof is not None and proof >= 4):
                        main = "μ Expectation (High)"
                    elif flames == "🔥🔥🔥" and (proof is not None and proof >= 4):
                        main = "Bonkers Regression 🔥🔥🔥"
                elif mkt in ("GOALS","ATG"):
                    if (mu is not None and line is not None and mu >= line + 0.35):
                        main = "μ Expectation (High)"
                    elif flames == "🔥🔥🔥" and ((gk_weak is not None and gk_weak >= 65) or (opp_def is not None and opp_def >= 60)):
                        main = "Conversion Regression 🔥🔥🔥"
                    elif (gk_weak is not None and gk_weak >= 70) and ((med10_sog is not None and med10_sog >= 3.5) or (avg5_sog is not None and avg5_sog >= 3.5)):
                        main = "Finishing Matchup"

                supports = []
                def _add_support(label: str, ok: bool):
                    supports.append((label, bool(ok)))

                if mkt == "SOG":
                    _add_support("μ Moderate+", (mu is not None and line is not None and mu >= line + 0.75))
                    _add_support("Stable Usage", (toi is not None and toi >= 16))
                    _add_support("STAR/ELITE", ("ELITE" in tier_u) or ("STAR" in tier_u))
                    _add_support("Shot Environment", (opp_sog_l10 is not None and opp_sog_l10 >= 30) or (opp_def is not None and opp_def >= 60))
                    _add_support("Pace", (team_sf60 is not None and team_sf60 >= 58))
                elif mkt == "POINTS":
                    _add_support("μ Moderate+", (mu is not None and line is not None and mu >= line + 0.75))
                    _add_support("Regression 🔥🔥", flames in ("🔥🔥","🔥🔥🔥"))
                    _add_support("STAR/ELITE", ("ELITE" in tier_u) or ("STAR" in tier_u))
                    _add_support("Usage Stability", (toi is not None and toi >= 16))
                    _add_support("PP Role", pp_role != "" and pp_role.upper() not in ("NONE","0","N/A"))
                elif mkt == "ASSISTS":
                    _add_support("μ Moderate+", (mu is not None and line is not None and mu >= line + 0.75))
                    _add_support("Regression 🔥🔥", flames in ("🔥🔥","🔥🔥🔥"))
                    _add_support("PP Distributor", bool(r.get("Assist_PP_Proof")) or (pp_role != "" and (("PP1" in pp_role.upper()) or ("PP2" in pp_role.upper()) or pp_role.strip() in ("1","2"))))
                    _add_support("STAR/ELITE", ("ELITE" in tier_u) or ("STAR" in tier_u))
                    _add_support("Linemate Finishing", (opp_def is not None and opp_def >= 60) or (gk_weak is not None and gk_weak >= 65))
                elif mkt in ("GOALS","ATG"):
                    _add_support("μ Moderate+", (mu is not None and line is not None and mu >= line + 0.20))
                    _add_support("Regression 🔥🔥", flames in ("🔥🔥","🔥🔥🔥"))
                    _add_support("Shooter Identity", (med10_sog is not None and med10_sog >= 3.5) or (avg5_sog is not None and avg5_sog >= 3.5))
                    _add_support("STAR/ELITE", ("ELITE" in tier_u) or ("STAR" in tier_u))
                    _add_support("PP Role", pp_role != "" and pp_role.upper() not in ("NONE","0","N/A"))

                support_on = [lab for lab, ok in supports if ok]
                support_score = sum(1 for _, ok in supports if ok)
                support_total = len(supports)

                tonight = []
                if opp_def is not None and opp_def >= 60:
                    tonight.append("Weak Defense")
                if gk_weak is not None and gk_weak >= 65:
                    tonight.append("Weak Goalie")
                if pp_match is not None and pp_match >= 60:
                    tonight.append("PP Matchup")
                if mkt == "SOG" and opp_sog_l10 is not None and opp_sog_l10 >= 30:
                    tonight.append("Shot-Friendly Opponent")
                if team_sf60 is not None and team_sf60 >= 60:
                    tonight.append("Pace")

                st.markdown(f"**MAIN:** {main if main else '—'}")
                st.markdown(f"**SUPPORT:** {support_score} / {support_total}")
                st.caption(" • ".join(support_on) if support_on else "—")
                st.markdown(f"**TONIGHT:** {' • '.join(tonight) if tonight else '—'}")
                if flames:
                    st.caption(f"Regression: {flames} ({heat})")
                if gap is not None or drought is not None:
                    _gap_s = f"{gap:.2f}" if gap is not None else "—"
                    _dr_s = str(int(drought)) if drought is not None and not pd.isna(drought) else "—"
                    st.caption(f"Reg gap: {_gap_s}  |  Drought: {_dr_s}")
    with st.expander("Full Board Table (all rows)", expanded=False):
        show_table(df_b, board_cols, "Board (sorted by Best_Conf)")



# =========================
# POINTS
# =========================
elif page == "Points":
    df_p = df_f.copy()
    df_p["_cp"] = safe_num(df_p, "Conf_Points", 0)
    df_p = df_p.sort_values(["_cp"], ascending=[False]).drop(columns=["_cp"], errors="ignore")

    st.sidebar.subheader("Points Filters")
    show_all = st.sidebar.checkbox("Show all players (ignore filters)", value=False, key="show_all_points")
    min_conf = st.sidebar.slider("Min Conf (Points)", 0, 100, 77, 1)
    color_pick = st.sidebar.multiselect(
        "Colors (Points)",
        ["green", "yellow", "blue", "red"],
        default=["green", "yellow", "blue"]
    )

    if not show_all:
        df_p = df_p[df_p["Conf_Points"].fillna(0) >= min_conf]
        if "Color_Points" in df_p.columns and color_pick:
            df_p = df_p[df_p["Color_Points"].isin(color_pick)]

    df_p["Green"] = df_p["Green_Points"].map(lambda x: "🟢" if bool(x) else "")

    points_cols = [
        "Game","Player","Pos",
        "Tier_Tag",
        "Markets",
        "Green",
        "EV_Signal",
        "LOCK",
        "Conf_Points","Matrix_Points",
        "Points_Line",
        "Points_Odds_Over",

        # --- EV / Odds ---
        "Points_Book",
        "Points_Model%",
        "Points_Imp%",
        "Points_EV%",
        "Plays_EV_Points",

        "Points_Call",
        "Reg_Heat_P","Reg_Gap_P10","Exp_P_10","L10_P",
        "iXG%","iXA%",
        "Opp_Goalie","Opp_SV","Opp_GAA","Goalie_Weak","Opp_DefWeak",
        "Drought_P","Best_Drought",
        "Line","Odds","Result",
]






    # Signals-first extras






    df_p["Markets"] = df_p.apply(build_markets_pills, axis=1)






    g = df_p.get("Green_Points", (df_p.get("Green","") == "🟢"))






    e = df_p["Plays_EV_Points"] if "Plays_EV_Points" in df_p.columns else pd.Series([""]*len(df_p), index=df_p.index)






    p = df_p["Points_EV%"] if "Points_EV%" in df_p.columns else pd.Series([None]*len(df_p), index=df_p.index)






    df_p["EV_Signal"] = [build_ev_signal(gg, ee, pp) for gg, ee, pp in zip(g, e, p if hasattr(p, "__iter__") else [p]*len(df_p))]





    df_p["LOCK"] = [build_lock_badge(gg, ee) for gg, ee in zip(g, e)]
    legend_signals()
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








    # === SMASH PLAYS (POINTS) ===
    st.subheader("⭐ Smash Plays — Points")
    st.markdown(
        """
        <div style="padding:14px 16px;border-radius:14px;border:2px solid #000;background:#fff;">
          <div style="font-size:24px;font-weight:900;color:#000;margin-bottom:6px;">
            Top candidates for this market
          </div>
          <div style="font-size:18px;font-weight:800;color:#000;margin-bottom:10px;">
            Ranked by 🔒 Locks &nbsp;&gt;&nbsp; 💰 +EV &nbsp;&gt;&nbsp; Conf
          </div>
          <div style="font-size:18px;font-weight:800;color:#000;line-height:1.45;">
            Why these fire:
          </div>
          <div style="font-size:17px;font-weight:700;color:#000;line-height:1.45;margin-top:4px;">
            <b>FIN</b> = finishing / converts chances &nbsp;&nbsp;•&nbsp;&nbsp;
            <b>PLY</b> = playmaking hub / involvement &nbsp;&nbsp;•&nbsp;&nbsp;
            <b>ENV</b> = matchup environment &nbsp;&nbsp;•&nbsp;&nbsp;
            <b>DUE</b> = due / regression pressure
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    _rank = df_p.copy()
    try:
        _rank["_is_lock"] = (_rank["LOCK"].astype(str).str.strip() == "🔒").astype(int) if "LOCK" in _rank.columns else 0
    except Exception:
        _rank["_is_lock"] = 0

    try:
        if "EV_Signal" in _rank.columns:
            _rank["_is_ev"] = _rank["EV_Signal"].astype(str).str.contains("💰", na=False).astype(int)
        elif "Plays_EV_Points" in _rank.columns:
            _rank["_is_ev"] = (_rank["Plays_EV_Points"].astype(str).str.strip() == "💰").astype(int)
        else:
            _rank["_is_ev"] = 0
    except Exception:
        _rank["_is_ev"] = 0

    _rank["_conf"] = pd.to_numeric(_rank.get("Conf_Points", 0), errors="coerce").fillna(0)
    _rank = _rank.sort_values(["_is_lock", "_is_ev", "_conf"], ascending=[False, False, False], kind="mergesort")

    top_n = st.slider("Show top plays (Points)", 3, 25, 10, 1, key="pts_smash_topn")
    top = _rank.head(int(top_n))

    for _, r in top.iterrows():
        player = str(r.get("Player", "") or "").strip()
        game = str(r.get("Game", "") or "").strip()

        line = r.get("Points_Line", "")
        odds = r.get("Points_Odds_Over", "")
        call = str(r.get("Points_Call", "") or "").strip()

        conf = r.get("Conf_Points", "")
        matrix = str(r.get("Matrix_Points", "") or "").strip()
        expl, crit = _derive_badges(r)
        badges = f"{str(r.get('EV_Signal','') or '').strip()} {str(r.get('LOCK','') or '').strip()} {expl} {crit}".strip()

        # Pretty line/odds strings
        try:
            l_str = "" if line is None or (isinstance(line, float) and math.isnan(line)) else f"{float(line):.1f}"
        except Exception:
            l_str = str(line) if line is not None else ""
        try:
            o_str = "" if odds is None or (isinstance(odds, float) and math.isnan(odds)) else f"{int(round(float(odds)))}"
        except Exception:
            o_str = str(odds) if odds is not None else ""

        headline = f"<b>{player}</b> — {game}" if game else f"<b>{player}</b>"
        betline = f"PTS {l_str} @ {o_str}" if (l_str or o_str) else ""

        meta = []
        if matrix:
            meta.append(matrix)
        if conf != "" and conf is not None:
            try:
                meta.append(f"Conf {float(conf):.0f}")
            except Exception:
                meta.append(f"Conf {conf}")
        if call:
            meta.append(call)
        meta_s = " | ".join([m for m in meta if m])

        st.markdown(
            f"""
    <div class="wl-card wl-accent-blue">
      <div style="display:flex;justify-content:space-between;gap:10px;">
        <div style="font-size:16px;line-height:1.2;">
          {headline}
          <div style="opacity:0.9;margin-top:4px;">{betline}</div>
        </div>
        <div style="font-size:16px;white-space:nowrap;">{badges}</div>
      </div>
      <div style="margin-top:6px;font-size:12px;opacity:0.92;line-height:1.2;">{meta_s}</div>
    </div>
            """,
            unsafe_allow_html=True,
        )

        # Why it fires (Points)
        _why_tags = str(r.get("Points_Why", "") or "").strip()
        if not _why_tags:
            try:
                _why_tags = _points_why(r)
            except Exception:
                _why_tags = ""
        with st.expander("Why it fires", expanded=False):  # key removed
            render_badge_legend_inline()
            _why_sections_header("POINTS")
            _render_why_it_fires_rich("POINTS", r, _why_tags)

    st.markdown("---")

    show_table(df_p, points_cols, "Points View")


# =========================
# ASSISTS
# =========================
elif page == "Assists":
    df_a = df_f.copy()
    df_a["_ca"] = safe_num(df_a, "Conf_Assists", 0)
    df_a = df_a.sort_values(["_ca"], ascending=[False]).drop(columns=["_ca"], errors="ignore")

    st.sidebar.subheader("Assists Filters")
    show_all = st.sidebar.checkbox("Show all players (ignore filters)", value=False, key="show_all_assists")
    min_conf = st.sidebar.slider("Min Conf (Assists)", 0, 100, 77, 1)
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
    mask = (
        (proof if isinstance(proof, pd.Series) else False)
        | (apc >= 4)
        | (adg >= 82)
        | ((ppt.isin(["A", "B"])) & (apc >= 3) & (adg >= 60))
    )

    df_a.loc[mask, "🗡️"] = "🗡️"

    assists_cols = [

        "Game",
        "Player", "Pos",
        "Tier_Tag",
        "Markets",
        "Green",
        "EV_Signal",
        "LOCK",
        "Conf_Assists", "Matrix_Assists",

        # --- EV / Odds ---
        "Assists_Line",
        "Assists_Odds_Over",
        "Assists_Book",
        "Assists_Model%",
        "Assists_Imp%",
        "Assists_EV%",
        "Plays_EV_Assists",

        "Assists_Call",
        "Drought_A","Best_Drought",
        "Assist_ProofCount", "Assist_Why", "🗡️", "Assist_Dagger",
        
        "Reg_Heat_A", "Reg_Gap_A10", "Exp_A_10", "L10_A",
        "PP_Tier", "PP_Path", "PP_BOOST",
        "PP_TOI_Pct_Game", "PP_iXA60", "PP_Matchup",

        
        "iXA%","iXG%", "v2_player_stability",
        "Opp_Goalie", "Opp_SV",
        "Goalie_Weak", "Opp_DefWeak",
        "Line", "Odds", "Result",
    ]

    # Signals-first extras

    df_a["Markets"] = df_a.apply(build_markets_pills, axis=1)

    g = df_a.get("Green_Assists", (df_a.get("Green","") == "🟢"))

    e = df_a["Plays_EV_Assists"] if "Plays_EV_Assists" in df_a.columns else pd.Series([""]*len(df_a), index=df_a.index)

    p = df_a["Assists_EV%"] if "Assists_EV%" in df_a.columns else pd.Series([None]*len(df_a), index=df_a.index)

    df_a["EV_Signal"] = [build_ev_signal(gg, ee, pp) for gg, ee, pp in zip(g, e, p if hasattr(p, "__iter__") else [p]*len(df_a))]

    df_a["LOCK"] = [build_lock_badge(gg, ee) for gg, ee in zip(g, e)]
    legend_signals()
    _f = render_market_filter_bar(default_min_conf=60, key_prefix="ast")

    try:
        df_a = apply_market_filters(
            df_a,
            _f,
            green_col="Green_Assists",
            ev_icon_col="Plays_EV_Assists",
            conf_col="Conf_Assists",
            matrix_col="Matrix_Assists",
            lock_col="LOCK",
        )
    except Exception:
        pass



    
    # === SMASH PLAYS (ASSISTS) ===
    st.subheader("⭐ Smash Plays — Assists")
    st.markdown(
        """
        <div style="padding:14px 16px;border-radius:14px;border:2px solid #000;background:#fff;">
          <div style="font-size:24px;font-weight:900;color:#000;margin-bottom:6px;">
            Top candidates for this market
          </div>
          <div style="font-size:18px;font-weight:800;color:#000;margin-bottom:10px;">
            Ranked by 🔒 Locks &nbsp;&gt;&nbsp; 💰 +EV &nbsp;&gt;&nbsp; Conf
          </div>
          <div style="font-size:18px;font-weight:800;color:#000;line-height:1.45;">
            Why these fire:
          </div>
          <div style="font-size:17px;font-weight:700;color:#000;line-height:1.45;margin-top:4px;">
            <b>PLY</b> = primary playmaker / distributor &nbsp;&nbsp;•&nbsp;&nbsp;
            <b>ENV</b> = matchup + teammates convert &nbsp;&nbsp;•&nbsp;&nbsp;
            <b>DUE</b> = drought / regression pressure
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _rank = df_a[df_a.get("Plays_Assists", False)].copy()

    # HARD FAIL (Assists): require Assist-specific Drought OR Assist-specific Reg Heat
    _drought_a = pd.to_numeric(_rank.get("Drought_A", 0), errors="coerce").fillna(0).astype(int)
    _heat_a = _rank.get("Reg_Heat_A", "").astype(str).str.strip().str.upper()
    _rank = _rank[(_drought_a >= int(HARD_BOARD_DROUGHT_GATE)) | (_heat_a.isin(HARD_BOARD_HEAT_LEVELS))].copy()


    try:
        _rank["_is_lock"] = (_rank["LOCK"].astype(str).str.strip() == "🔒").astype(int)
    except Exception:
        _rank["_is_lock"] = 0

    try:
        if "EV_Signal" in _rank.columns:
            _rank["_is_ev"] = _rank["EV_Signal"].astype(str).str.contains("💰", na=False).astype(int)
        elif "Plays_EV_Assists" in _rank.columns:
            _rank["_is_ev"] = (_rank["Plays_EV_Assists"].astype(str).str.strip() == "💰").astype(int)
        else:
            _rank["_is_ev"] = 0
    except Exception:
        _rank["_is_ev"] = 0

    _rank["_conf"] = pd.to_numeric(_rank.get("Conf_Assists", 0), errors="coerce").fillna(0)
    _rank = _rank.sort_values(["_is_lock", "_is_ev", "_conf"], ascending=[False, False, False], kind="mergesort")

    top_n = st.slider("Show top plays (Assists)", 3, 25, 10, 1, key="ast_smash_topn")
    top = _rank.head(int(top_n))

    for _, r in top.iterrows():
        player = str(r.get("Player", "") or "").strip()
        game = str(r.get("Game", "") or "").strip()

        line = r.get("Assists_Line", "")
        odds = r.get("Assists_Odds_Over", "")
        call = str(r.get("Assists_Call", "") or "").strip()

        conf = r.get("Conf_Assists", "")
        matrix = str(r.get("Matrix_Assists", "") or "").strip()
        expl, crit = _derive_badges(r)
        badges = f"{str(r.get('EV_Signal','') or '').strip()} {str(r.get('LOCK','') or '').strip()} {expl} {crit}".strip()

        headline = f"<b>{player}</b> — {game}" if game else f"<b>{player}</b>"
        betline = f"AST {line} @ {odds}" if (line or odds) else ""

        meta = []
        if matrix:
            meta.append(matrix)
        if conf != "" and conf is not None:
            try:
                meta.append(f"Conf {float(conf):.0f}")
            except Exception:
                meta.append(f"Conf {conf}")
        if call:
            meta.append(call)
        meta_s = " | ".join([m for m in meta if m])

        st.markdown(
            f"""
    <div class="wl-card wl-accent-purple">
      <div style="display:flex;justify-content:space-between;gap:10px;">
        <div style="font-size:16px;line-height:1.2;">
          {headline}
          <div style="opacity:0.9;margin-top:4px;">{betline}</div>
        </div>
        <div style="font-size:16px;white-space:nowrap;">{badges}</div>
      </div>
      <div style="margin-top:6px;font-size:12px;opacity:0.92;line-height:1.2;">{meta_s}</div>
    </div>
            """,
            unsafe_allow_html=True,
        )

        _why_tags = str(r.get("Assist_Why", "") or "").strip()
        with st.expander("Why it fires", expanded=False):
            render_badge_legend_inline()
            _why_sections_header("ASSISTS")
            _render_why_it_fires_rich("ASSISTS", r, _why_tags)

    st.markdown("---")

    show_table(df_a, assists_cols, "Assists View")


# =========================
# SOG
# =========================
elif page == "SOG":
    df_s = df_f.copy()
    df_s["_cs"] = safe_num(df_s, "Conf_SOG", 0)
    df_s = df_s.sort_values(["_cs"], ascending=[False]).drop(columns=["_cs"], errors="ignore")

    st.sidebar.subheader("SOG Filters")
    show_all = st.sidebar.checkbox("Show all players (ignore filters)", value=False, key="show_all_sog")
    min_conf = st.sidebar.slider("Min Conf (SOG)", 0, 100, 77, 1)
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
       "Markets",
       "Green",
       "EV_Signal",
       "LOCK",
       "Conf_SOG", "Matrix_SOG",

        # --- EV / Odds ---
        "SOG_Line",
        "SOG_Odds_Over",
        "SOG_Book",
        "SOG_Model%",
        "SOG_Imp%",
        "SOG_EV%",
        "Plays_EV_SOG",

        "SOG_Call",
        "SOG_Why",
        "Drought_SOG", "Best_Drought",
        "Reg_Heat_S", "Reg_Gap_S10", "Exp_S_10", "L10_S",
        "Med10_SOG", "Avg5_SOG", "ShotIntent", "ShotIntent_Pct",
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





    # -------------------------
    # SOG Smash (cards)
    # -------------------------
    try:
        _rank = df_s.copy()
        _rank["_is_lock"] = (_rank.get("LOCK", "").astype(str).str.strip() == "🔒").astype(int)
        _rank["_is_ev"] = _rank.get("Plays_EV_SOG", "").astype(str).str.strip().eq("💰").astype(int) if "Plays_EV_SOG" in _rank.columns else 0
        _rank["_conf"] = pd.to_numeric(_rank.get("Conf_SOG", 0), errors="coerce").fillna(0)
        _rank = _rank.sort_values(["_is_lock", "_is_ev", "_conf"], ascending=[False, False, False], kind="mergesort")

        top_n = st.slider("Show top plays (SOG)", 3, 25, 10, 1, key="sog_smash_topn")
        top = _rank.head(int(top_n))

        if "Plays_SOG" in top.columns:
            top = top[_col_bool(top, "Plays_SOG")]

        if top.empty:
            st.info("No smash plays available for SOG on this slate.")
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
                badges = f"{str(r.get('EV_Signal','') or '').strip()} {str(r.get('LOCK','') or '').strip()} {expl} {crit}".strip()

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
                meta_s = " | ".join([m for m in meta if m])

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
          <div style="margin-top:6px;font-size:12px;opacity:0.92;line-height:1.2;">{meta_s}</div>
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
                    render_badge_legend_inline()
                    st.write(_why_tags if _why_tags else "—")

    except Exception:
        pass

    st.markdown("---")

    show_table(df_s, sog_cols, "SOG View")


# =========================
# GOAL
# =========================
elif page == "GOAL (1+)":
    df_g = df_f.copy()
    df_g["_cg"] = safe_num(df_g, "Conf_Goal", 0)
    df_g = df_g.sort_values(["_cg"], ascending=[False]).drop(columns=["_cg"], errors="ignore")

    st.sidebar.subheader("Goal Filters")
    show_all = st.sidebar.checkbox("Show all players (ignore filters)", value=False)
    min_conf = st.sidebar.slider("Min Conf (Goal)", 0, 100, 77, 1)
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
        "Markets",
        "Green",
        "EV_Signal",
        "LOCK",
        "Conf_Goal", "Matrix_Goal",

        # --- EV / Odds ---
        "ATG_Line",
        "ATG_Odds_Over",
        "ATG_Book",
        "ATG_Model%", "ATG_Imp%", "ATG_EV%", "Plays_EV_ATG",

        "ATG_Call",
        "Reg_Heat_G", "Reg_Gap_G10", "Exp_G_10", "L10_G",
        "iXG%", "iXA%",
        "Opp_Goalie", "Opp_SV", "Opp_GAA", "Goalie_Weak", "Opp_DefWeak",
        "Drought_G", "Best_Drought",
    ]

    # Signals-first extras

    df_g["Markets"] = df_g.apply(build_markets_pills, axis=1)

    g = df_g.get("Green_Goal", (df_g.get("Green","") == "🟢"))

    e = df_g["Plays_EV_ATG"] if "Plays_EV_ATG" in df_g.columns else pd.Series([""]*len(df_g), index=df_g.index)

    p = df_g["ATG_EV%"] if "ATG_EV%" in df_g.columns else pd.Series([None]*len(df_g), index=df_g.index)

    df_g["EV_Signal"] = [build_ev_signal(gg, ee, pp) for gg, ee, pp in zip(g, e, p if hasattr(p, "__iter__") else [p]*len(df_g))]

    df_g["LOCK"] = [build_lock_badge(gg, ee) for gg, ee in zip(g, e)]
    legend_signals()
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
        min_line = st.number_input("Min line", value=float(_min_line), step=0.5, key="ladder_min_line")
    with cC:
        min_ev = st.number_input("Min EV%", value=float(_min_ev), step=0.5, key="ladder_min_ev")
    with cD:
        min_model = st.number_input("Min Model%", value=float(_min_model), step=0.5, key="ladder_min_model")
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