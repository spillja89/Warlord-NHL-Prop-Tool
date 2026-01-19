import os
import glob
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


# -------------------------
# Number formatting helpers
# -------------------------
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
        'Player','Team','Opp','Game','Pos','Tier_Tag','🔥','💰',
    ]
    out=[]
    for c in order:
        if c in cols and c not in out:
            out.append(c)
    for c in cols:
        if c not in out:
            out.append(c)
    return out


def _pill(label: str, state) -> str:
    """Return an emoji pill based on Matrix state."""
    if state is None:
        return ""
    s = str(state).strip().lower()
    if not s:
        return ""
    if "green" in s:
        return f"🟢 {label}"
    if "yellow" in s:
        return f"🟡 {label}"
    if "red" in s:
        return f"🔴 {label}"
    return f"⚪ {label}"


def _markets_pills_row(r) -> str:
    """Build a quick market pills string from Matrix_* columns."""
    pills = []
    pills.append(_pill("PTS", r.get("Matrix_Points")))
    pills.append(_pill("SOG", r.get("Matrix_SOG")))
    pills.append(_pill("G", r.get("Matrix_Goal")))
    pills.append(_pill("A", r.get("Matrix_Assists")))
    return "  ".join([p for p in pills if p])


def lock_badge(green_bool, ev_icon) -> str:
    """🔒 when earned green AND +EV (💰) for that market."""
    try:
        g = bool(green_bool)
    except Exception:
        g = False
    e = (str(ev_icon).strip() == "💰")
    return "🔒" if (g and e) else ""


COLUMN_WIDTHS = {
    # identity
    "Game": "small",
    "Pos": "small",
    "Team": "small",
    "Opp": "small",
    "Player": "medium",
    "Markets": "large",

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

    # misc
    "Line": "small",
    "Odds": "small",
    "Result": "small",
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
def _get(row, key, default=0):
    """Safe getter for dict-like rows (pandas Series or dict)."""
    try:
        v = row.get(key, default)
    except Exception:
        v = default
    return default if v is None else v


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
st.set_page_config(page_title="NHL Prop Tool", layout="wide")

# =========================
# GLOBAL TABLE COMPACT CSS (Option A)
# =========================
st.markdown(
    """
    <style>
      /* Tighten the dataframe (works for st.dataframe + Styler) */
      div[data-testid="stDataFrame"] table {
        font-size: 12px;
      }
      div[data-testid="stDataFrame"] thead tr th {
        padding-top: 2px !important;
        padding-bottom: 2px !important;
      }
      div[data-testid="stDataFrame"] tbody tr td {
        padding-top: 1px !important;
        padding-bottom: 1px !important;
        line-height: 1.05 !important;
        white-space: nowrap !important;
      }

      /* Optional: make header slightly tighter too */
      div[data-testid="stDataFrame"] thead tr th div {
        line-height: 1.05 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# VENGEANCE CLOCK (Board)
# =========================
def render_vengeance_clock():
    """Analog ticking clock widget (pure HTML/CSS/JS)."""
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:14px;justify-content:flex-end;margin-top:-8px;margin-bottom:6px;">
          <div style="font-weight:900;letter-spacing:0.4px;font-size:14px;opacity:0.95;">
            🕰️ <span style="text-transform:uppercase;">The clock strikes vengeance</span>
          </div>
          <div id="vengeanceClock" style="position:relative;width:54px;height:54px;border-radius:50%;
               border:2px solid rgba(255,255,255,0.75); box-shadow:0 6px 18px rgba(0,0,0,0.25);
               background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.18), rgba(0,0,0,0.15));">
            <div style="position:absolute;inset:5px;border-radius:50%;border:1px solid rgba(255,255,255,0.18);"></div>
            <div id="vcHour" style="position:absolute;left:50%;top:50%;width:3px;height:16px;background:rgba(255,255,255,0.85);
                 transform-origin:50% 90%;border-radius:2px;transform:translate(-50%,-90%) rotate(0deg);"></div>
            <div id="vcMin" style="position:absolute;left:50%;top:50%;width:2px;height:21px;background:rgba(255,255,255,0.85);
                 transform-origin:50% 92%;border-radius:2px;transform:translate(-50%,-92%) rotate(0deg);"></div>
            <div id="vcSec" style="position:absolute;left:50%;top:50%;width:1px;height:23px;background:rgba(255,0,0,0.85);
                 transform-origin:50% 92%;border-radius:2px;transform:translate(-50%,-92%) rotate(0deg);"></div>
            <div style="position:absolute;left:50%;top:50%;width:6px;height:6px;border-radius:50%;background:rgba(255,255,255,0.9);
                 transform:translate(-50%,-50%);"></div>
          </div>
        </div>

        <script>
          (function() {
            function tickVengeanceClock(){
              const d = new Date();
              const s = d.getSeconds();
              const m = d.getMinutes();
              const h = d.getHours() % 12;

              const secDeg = s * 6;                 // 360/60
              const minDeg = (m + s/60) * 6;
              const hourDeg = (h + m/60) * 30;      // 360/12

              const sec = document.getElementById('vcSec');
              const min = document.getElementById('vcMin');
              const hour = document.getElementById('vcHour');
              if (!sec || !min || !hour) return;

              sec.style.transform = `translate(-50%,-92%) rotate(${secDeg}deg)`;
              min.style.transform = `translate(-50%,-92%) rotate(${minDeg}deg)`;
              hour.style.transform = `translate(-50%,-90%) rotate(${hourDeg}deg)`;
            }
            tickVengeanceClock();
            window.__vcInterval = window.__vcInterval || setInterval(tickVengeanceClock, 1000);
          })();
        </script>
        """,
        unsafe_allow_html=True,
    )



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


def safe_str(df: pd.DataFrame, col: str, default="") -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return df[col].astype(str).fillna(default)


def style_df(df: pd.DataFrame, cols: list[str]) -> "pd.io.formats.style.Styler":
    # --- Pandas Styler REQUIRES unique index + unique columns ---
    # de-dupe requested columns while preserving order
    cols = [c for c in dict.fromkeys(cols) if c in df.columns]

    # safe view with unique index
    view = df.loc[:, cols].copy().reset_index(drop=True)

    # if anything upstream produced duplicate column names, drop them
    if view.columns.duplicated().any():
        view = view.loc[:, ~view.columns.duplicated()].copy()

    # --- Matrix: Green/Yellow/Red ---
    def matrix_style(v):
        s = str(v).strip().lower()
        if s == "green":
            return "background-color:#1f7a1f;color:white;font-weight:700;"
        if s == "yellow":
            return "background-color:#b38f00;color:white;font-weight:700;"
        if s == "red":
            return "background-color:#8b1a1a;color:white;font-weight:700;"
        return ""

    # --- Heat: HOT red, WARM orange, COOL blue ---
    def heat_style(v):
        s = str(v).strip().upper()
        if s == "HOT":
            return "background-color:#b30000;color:white;font-weight:700;"
        if s == "WARM":
            return "background-color:#e67300;color:white;font-weight:700;"
        if s == "COOL":
            return "background-color:#1f5aa6;color:white;font-weight:700;"
        return ""

    # --- Confidence: Green >=80, Yellow 70-79, Red <70 ---
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



    # --- EV%: Green >= +10, Yellow +5..+10, Red <0 ---
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
    # --- Weakness: only flag extreme weakness (red) ---
    def weak_style(v):
        try:
            x = float(v)
        except Exception:
            return ""
        if x >= 75:
            return "background-color:#b30000;color:white;font-weight:700;"
        return ""

    sty = view.style

    # Matrix columns
    for c in ["Matrix_Points", "Matrix_SOG", "Matrix_Assists", "Matrix_Goal"]:
        if c in view.columns:
            sty = sty.applymap(matrix_style, subset=[c])

    # Heat columns
    for c in ["Reg_Heat_P", "Reg_Heat_S", "Reg_Heat_G", "Reg_Heat_A"]:
        if c in view.columns:
            sty = sty.applymap(heat_style, subset=[c])

    # Confidence columns
    for c in ["Best_Conf", "Conf_Points", "Conf_SOG", "Conf_Goal", "Conf_Assists"]:
        if c in view.columns:
            sty = sty.applymap(conf_style, subset=[c])


    # EV columns
    for c in [c for c in view.columns if c.endswith("EVpct_over")]:
        sty = sty.applymap(ev_style, subset=[c])

    # Plays_EV_* booleans (highlight when true)
    def play_ev_style(v):
        try:
            return "background-color:#1f7a1f;color:white;font-weight:700;" if bool(v) else ""
        except Exception:
            return ""

    for c in [c for c in view.columns if c.startswith("Plays_EV_")]:
        sty = sty.applymap(play_ev_style, subset=[c])

    # Weakness columns (only extreme)
    for c in ["Goalie_Weak", "Opp_DefWeak"]:
        if c in view.columns:
            sty = sty.applymap(weak_style, subset=[c])
    # -------------------------
    # FORCE DECIMALS (Styler tables)
    # -------------------------
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

    # --- Betting UI: lines + odds ---
    # Lines: show 1 decimal, and they're already snapped to .0/.5 upstream (or we snap later)
    for c in view.columns:
        if c.endswith("_Line") or c in ("Line",):
            format_dict.setdefault(c, "{:.1f}")

    # American odds: show no decimals
    for c in view.columns:
        if c.endswith("_Odds_Over") or c in ("Odds",):
            format_dict.setdefault(c, "{:.0f}")

    # Probabilities (0-1): show as clean percent (e.g., 53.4%)
    for c in view.columns:
        if c.endswith("_p_model_over") or c.endswith("_p_imp_over"):
            format_dict.setdefault(c, "{:.1%}")

    if format_dict:
        sty = sty.format(format_dict, na_rep="")

  


    return sty


def add_ui_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

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

    return out


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
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
    # Matchup filter
    if "Game" in out.columns:
        games = sorted(
            [g for g in out["Game"].dropna().astype(str).unique().tolist() if g.strip()]
        )
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

    def first_nonempty(s: pd.Series) -> str:
        for v in s.tolist():
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    agg = {"Game": "first"}
    if have_local:
        agg["StartTimeLocal"] = first_nonempty
    if have_utc:
        agg["StartTimeUTC"] = first_nonempty

    g = tmp.groupby("Game", as_index=False).agg(agg)

    # Sort (UTC is best if present)
    if have_utc:
        g = g.sort_values("StartTimeUTC")
    elif have_local:
        g = g.sort_values("StartTimeLocal")

    st.subheader("Games & Start Times")
    st.dataframe(g, width="stretch", hide_index=True)


def show_table(df: pd.DataFrame, cols: list[str], title: str):
    st.subheader(title)

    # Styler requires unique index + columns; filtering a df can preserve a non-unique index.
    # Also de-dupe the requested column list (some views may accidentally include repeats).
    df = df.copy().reset_index(drop=True)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # de-dupe requested cols while preserving order
    cols = list(dict.fromkeys(cols))

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

uploaded = st.sidebar.file_uploader("Upload tracker CSV (optional)", type=["csv"])
latest_path = find_latest_tracker_csv(OUTPUT_DIR)

source = None
if uploaded is not None:
    source = "upload"
    df = pd.read_csv(uploaded)
else:
    source = "latest"
    if latest_path is None:
        st.warning(f"No CSV found in `{OUTPUT_DIR}/`. Run `python nhl_edge.py` first (it writes to output/).")
        st.stop()
    df = load_csv(latest_path)

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

# Build Tier_Tag from Talent_Tier (always overwrite so it stays correct)
tt = df["Talent_Tier"].astype(str).str.upper().fillna("NONE")
df["Tier_Tag"] = np.where(
    tt.eq("ELITE"),
    "👑 ELITE",
    np.where(tt.eq("STAR"), "⭐ STAR", "")
)

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

# Replace Plays_EV_* booleans with a 💰 icon for readability (keep the original name)
for c in ["Plays_EV_Points","Plays_EV_Goal","Plays_EV_Assists","Plays_EV_ATG","Plays_EV_SOG"]:
    if c in df.columns:
        df[c] = df[c].apply(lambda x: "💰" if bool(x) else "")

# Global 💰 if any EV-play is active
_ev_play_cols = [c for c in ["Plays_EV_Points","Plays_EV_Goal","Plays_EV_Assists","Plays_EV_ATG","Plays_EV_SOG"] if c in df.columns]
if _ev_play_cols:
    df["💰"] = (df[_ev_play_cols].astype(str).apply(lambda r: any(v=="💰" for v in r), axis=1)).map(lambda x: "💰" if x else "")
else:
    df["💰"] = ""


# =========================
# LOCK + EV summary columns (signals first)
# =========================
# Per-market LOCK badge (earned green + EV)
if "Green_Points" in df.columns and "Plays_EV_Points" in df.columns:
    df["LOCK_Points"] = np.where(df["Green_Points"].fillna(False).astype(bool) & (df["Plays_EV_Points"].astype(str)=="💰"), "🔒", "")
if "Green_SOG" in df.columns and "Plays_EV_SOG" in df.columns:
    df["LOCK_SOG"] = np.where(df["Green_SOG"].fillna(False).astype(bool) & (df["Plays_EV_SOG"].astype(str)=="💰"), "🔒", "")
if "Green_Goal" in df.columns and "Plays_EV_ATG" in df.columns:
    df["LOCK_Goal"] = np.where(df["Green_Goal"].fillna(False).astype(bool) & (df["Plays_EV_ATG"].astype(str)=="💰"), "🔒", "")
if "Green_Assists" in df.columns and "Plays_EV_Assists" in df.columns:
    df["LOCK_Assists"] = np.where(df["Green_Assists"].fillna(False).astype(bool) & (df["Plays_EV_Assists"].astype(str)=="💰"), "🔒", "")

# Any LOCK on the slate
lock_cols = [c for c in ["LOCK_Points","LOCK_SOG","LOCK_Goal","LOCK_Assists"] if c in df.columns]
df["LOCK"] = np.where(df[lock_cols].astype(str).apply(lambda r: any(v=="🔒" for v in r), axis=1), "🔒", "") if lock_cols else ""

# Best EV% across markets (shown up front on Board)
ev_pct_cols = [c for c in ["Points_EV%","SOG_EV%","ATG_EV%","Assists_EV%","Goal_EV%"] if c in df.columns]
if ev_pct_cols:
    tmp_ev = df[ev_pct_cols].apply(pd.to_numeric, errors="coerce")
    df["EV_Best"] = tmp_ev.max(axis=1).round(1)
else:
    df["EV_Best"] = np.nan


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

df["Green_Goal"] = (
    (safe_num(df, "Conf_Goal", 0) >= thr_g)
    & (safe_str(df, "Matrix_Goal", "").str.strip().str.lower() == "green")
    & (
        (safe_str(df, "Reg_Heat_G", "").str.strip().str.upper() == "HOT")
        | (safe_num(df, "Goalie_Weak", 0) >= 70)
        | (safe_num(df, "Drought_G", 0) >= 2)
    )
)

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
    ["Board", "Points", "Assists", "SOG", "GOAL (1+)","Guide", "Ledger", "Raw CSV"],
    index=0
)

df_f = filter_common(df)

# Show slate times table
show_games_times(df_f)


# =========================
# BOARD
# =========================
if page == "Board":
    # ---- Board quick controls (visual-only) ----
    # Header + clock
    h1, h2 = st.columns([3, 1])
    with h1:
        st.subheader("Board")
    with h2:
        render_vengeance_clock()

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        greens_only = st.toggle("🟢 Greens only", value=False)
    with c2:
        ev_only = st.toggle("💰 +EV only", value=False)
    with c3:
        hide_red_rows = st.toggle("Hide 🔴 rows", value=True)
    with c4:
        min_best_conf = st.slider("Min Best_Conf", 0, 100, 60, 1)

    df_b = df_f.copy().reset_index(drop=True)

    # Markets pills (PTS/SOG/G/A) from Matrix states
    if "Markets" not in df_b.columns:
        df_b["Markets"] = df_b.apply(_markets_pills_row, axis=1)

    # Playable flags
    if "Any_Green" not in df_b.columns:
        df_b["Any_Green"] = (
            df_b.get("Green_Points", False).fillna(False).astype(bool)
            | df_b.get("Green_SOG", False).fillna(False).astype(bool)
            | df_b.get("Green_Goal", False).fillna(False).astype(bool)
            | df_b.get("Green_Assists", False).fillna(False).astype(bool)
        )

    # Respect min Best_Conf
    df_b = df_b[safe_num(df_b, "Best_Conf", 0) >= float(min_best_conf)]

    # Filters
    if greens_only:
        # Prefer the strict Any_Green gate, fall back to 🔥 if needed
        if "Any_Green" in df_b.columns:
            df_b = df_b[df_b["Any_Green"] == True]
        elif "🔥" in df_b.columns:
            df_b = df_b[df_b["🔥"] == "🔥"]

    if ev_only and "💰" in df_b.columns:
        df_b = df_b[df_b["💰"] == "💰"]

    if hide_red_rows:
        mcols = [c for c in ["Matrix_Points","Matrix_SOG","Matrix_Goal","Matrix_Assists"] if c in df_b.columns]
        if mcols:
            def _all_red(r):
                vals = [str(r.get(c, "")).strip().lower() for c in mcols]
                present = [v for v in vals if v]
                return bool(present) and all("red" in v for v in present)
            df_b = df_b[~df_b.apply(_all_red, axis=1)]

    # Sort best-first
    df_b = sort_board(df_b)

    with st.expander("Legend / How to read"):
        st.markdown("""
- **Markets** shows each market's **Matrix** state at a glance.
- **🟢 Earned green** = playable per market rules.
- **💰** = the odds are mispriced vs our model (positive EV threshold passed).
- Use **Min Best_Conf** + **Hide reds** to keep the board tight.
""")

    board_cols_signal = [
        "Game",
        "Player", "Pos", "Team", "Opp",
        "Markets",
        "Tier_Tag",
        "Best_Market",
        "Best_Conf",
        "LOCK", "💰", "EV_Best",
        "Goalie_Weak", "Opp_DefWeak",
    ]

    show_table(df_b, board_cols_signal, "Board — Signals (scan first)")

    with st.expander("Details / Why (click to dig deeper)"):
        board_cols_detail = [
            "Game",
            "Player", "Pos", "Team", "Opp",
            "Markets",
            "Tier_Tag",
            "Best_Market",
            "Best_Conf",
            "LOCK", "💰", "EV_Best",
            "Opp_Goalie", "Opp_SV", "Opp_GAA",
            "Matrix_Points", "Conf_Points", "Reg_Heat_P", "Reg_Gap_P10", "Points_EV%", "Points_Line", "Points_Odds_Over",
            "Matrix_SOG", "Conf_SOG", "Reg_Heat_S", "Reg_Gap_S10", "SOG_EV%", "SOG_Line", "SOG_Odds_Over",
            "Matrix_Goal", "Conf_Goal", "Reg_Heat_G", "Reg_Gap_G10", "ATG_EV%", "ATG_Line", "ATG_Odds_Over",
            "Matrix_Assists", "Conf_Assists", "Reg_Heat_A", "Reg_Gap_A10", "Assists_EV%", "Assists_Line", "Assists_Odds_Over",
            "Line", "Odds", "Result",
        ]
        show_table(df_b, board_cols_detail, "Board — Details")




# =========================
# POINTS
# =========================
elif page == "Points":
    df_p = df_f.copy()
    df_p["_cp"] = safe_num(df_p, "Conf_Points", 0)
    df_p = df_p.sort_values(["_cp"], ascending=[False]).drop(columns=["_cp"], errors="ignore")

    st.subheader("Points")
    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1, 2])
    with c1:
        greens_only = st.toggle("🟢 Greens only", value=False, key="pts_greens_only")
    with c2:
        ev_only = st.toggle("💰 +EV only", value=False, key="pts_ev_only")
    with c3:
        hide_red_rows = st.toggle("Hide 🔴 rows", value=True, key="pts_hide_reds")
    with c4:
        plays_first = st.toggle("Plays first", value=True, key="pts_plays_first")
    with c5:
        show_all = st.toggle("Show all", value=False, key="pts_show_all")
    with c6:
        min_conf = st.slider("Min Conf (Points)", 0, 100, 77, 1, key="pts_min_conf")

    color_pick = st.multiselect(
        "Colors (Points)",
        ["green", "yellow", "blue", "red"],
        default=["green", "yellow", "blue"],
        key="pts_colors",
    )


    if not show_all:
        df_p = df_p[df_p["Conf_Points"].fillna(0) >= min_conf]
        if "Color_Points" in df_p.columns and color_pick:
            df_p = df_p[df_p["Color_Points"].isin(color_pick)]
    # ---- Visual-style filters (Board-like) ----
    if hide_red_rows and "Matrix_Points" in df_p.columns:
        df_p = df_p[safe_str(df_p, "Matrix_Points", "").str.strip().str.lower() != "red"]

    if greens_only and "Green_Points" in df_p.columns:
        df_p = df_p[df_p["Green_Points"].fillna(False).astype(bool)]

    if ev_only:
        if "Plays_EV_Points" in df_p.columns:
            df_p = df_p[df_p["Plays_EV_Points"].astype(str) == "💰"]
        elif "💰" in df_p.columns:
            df_p = df_p[df_p["💰"] == "💰"]

    # Markets pills for quick scan
    if "Markets" not in df_p.columns:
        df_p["Markets"] = df_p.apply(_markets_pills_row, axis=1)

    if plays_first:
        df_p["_g"] = df_p.get("Green_Points", False).fillna(False).astype(bool)
        df_p["_ev"] = (df_p.get("Plays_EV_Points", "").astype(str) == "💰")
        df_p["_c"] = safe_num(df_p, "Conf_Points", 0)
        df_p = df_p.sort_values(["_g", "_ev", "_c"], ascending=[False, False, False]).drop(columns=["_g","_ev","_c"], errors="ignore")


    df_p["Green"] = df_p["Green_Points"].map(lambda x: "🟢" if bool(x) else "")
    df_p["LOCK"] = df_p.apply(lambda r: lock_badge(r.get("Green_Points", False), r.get("Plays_EV_Points", "")), axis=1)

    points_cols_signal = [
        "Game","Player","Pos","Team","Opp","Markets",
        "Tier_Tag",
        "Matrix_Points","Conf_Points","Green",
        "Plays_EV_Points","Points_EV%","LOCK",
        "Points_Line","Points_Odds_Over","Points_Book",
        "Points_Call",
    ]

    show_table(df_p, points_cols_signal, "Points — Signals (simple first)")

    with st.expander("Details / Why (Points)"):
        points_cols_detail = [
            "Game","Player","Pos","Team","Opp","Markets",
            "Tier_Tag",
            "Matrix_Points","Conf_Points","Green","Plays_EV_Points","Points_EV%","LOCK",
            "Points_Line","Points_Book","Points_Odds_Over","Points_Model%","Points_Imp%","Points_EV%",
            "Reg_Heat_P","Reg_Gap_P10","Exp_P_10","L10_P",
            "Points_ProofCount","Points_Why",
            "iXG%","iXA%","TOI_Pct","team_5v5_xGF60_pct",
            "Opp_Goalie","Opp_SV","Opp_GAA","Goalie_Weak","Opp_DefWeak",
            "Drought_P","Best_Drought",
            "Line","Odds","Result",
        ]
        show_table(df_p, points_cols_detail, "Points — Details")



# =========================
# ASSISTS
# =========================
elif page == "Assists":
    df_a = df_f.copy()
    df_a["_ca"] = safe_num(df_a, "Conf_Assists", 0)
    df_a = df_a.sort_values(["_ca"], ascending=[False]).drop(columns=["_ca"], errors="ignore")

    st.subheader("Assists")
    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1, 2])
    with c1:
        greens_only = st.toggle("🟢 Greens only", value=False, key="ast_greens_only")
    with c2:
        ev_only = st.toggle("💰 +EV only", value=False, key="ast_ev_only")
    with c3:
        hide_red_rows = st.toggle("Hide 🔴 rows", value=True, key="ast_hide_reds")
    with c4:
        plays_first = st.toggle("Plays first", value=True, key="ast_plays_first")
    with c5:
        show_all = st.toggle("Show all", value=False, key="ast_show_all")
    with c6:
        min_conf = st.slider("Min Conf (Assists)", 0, 100, 77, 1, key="ast_min_conf")

    color_pick = st.multiselect(
        "Colors (Assists)",
        ["green", "yellow", "blue", "red"],
        default=["green", "yellow", "blue"],
        key="ast_colors",
    )


    if not show_all:
        df_a = df_a[df_a["Conf_Assists"].fillna(0) >= min_conf]
        if "Color_Assists" in df_a.columns and color_pick:
            df_a = df_a[df_a["Color_Assists"].isin(color_pick)]
    # ---- Visual-style filters (Board-like) ----
    if hide_red_rows and "Matrix_Assists" in df_a.columns:
        df_a = df_a[safe_str(df_a, "Matrix_Assists", "").str.strip().str.lower() != "red"]

    if greens_only and "Green_Assists" in df_a.columns:
        df_a = df_a[df_a["Green_Assists"].fillna(False).astype(bool)]

    if ev_only:
        if "Plays_EV_Assists" in df_a.columns:
            df_a = df_a[df_a["Plays_EV_Assists"].astype(str) == "💰"]
        elif "💰" in df_a.columns:
            df_a = df_a[df_a["💰"] == "💰"]

    if "Markets" not in df_a.columns:
        df_a["Markets"] = df_a.apply(_markets_pills_row, axis=1)

    if plays_first:
        df_a["_g"] = df_a.get("Green_Assists", False).fillna(False).astype(bool)
        df_a["_ev"] = (df_a.get("Plays_EV_Assists", "").astype(str) == "💰")
        df_a["_c"] = safe_num(df_a, "Conf_Assists", 0)
        df_a = df_a.sort_values(["_g", "_ev", "_c"], ascending=[False, False, False]).drop(columns=["_g","_ev","_c"], errors="ignore")


    df_a["Green"] = df_a.get("Green_Assists", False).map(lambda x: "🟢" if bool(x) else "")

    assists_cols_signal = [
        "Game","Player","Pos","Team","Opp","Markets",
        "Tier_Tag",
        "Matrix_Assists","Conf_Assists","Green",
        "Plays_EV_Assists","Assists_EV%","LOCK",
        "Assists_Line","Assists_Odds_Over","Assists_Book",
        "Assists_Call",
    ]

    show_table(df_a, assists_cols_signal, "Assists — Signals (simple first)")

    with st.expander("Details / Why (Assists)"):
        assists_cols_detail = [
            "Game","Player","Pos","Team","Opp","Markets",
            "Tier_Tag",
            "Matrix_Assists","Conf_Assists","Green","Plays_EV_Assists","Assists_EV%","LOCK",
            "Assists_Line","Assists_Book","Assists_Odds_Over","Assists_Model%","Assists_Imp%","Assists_EV%",
            "Reg_Heat_A","Reg_Gap_A10","Exp_A_10","L10_A",
            "Assists_ProofCount","Assists_Why",
            "iXA%","iXG%","TOI_Pct","team_5v5_xGF60_pct",
            "Opp_Goalie","Opp_SV","Opp_GAA","Goalie_Weak","Opp_DefWeak",
            "Drought_A","Best_Drought",
            "Line","Odds","Result",
        ]
        show_table(df_a, assists_cols_detail, "Assists — Details")



# =========================
# SOG
# =========================
elif page == "SOG":
    df_s = df_f.copy()
    df_s["_cs"] = safe_num(df_s, "Conf_SOG", 0)
    df_s = df_s.sort_values(["_cs"], ascending=[False]).drop(columns=["_cs"], errors="ignore")

    st.subheader("SOG")
    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1, 2])
    with c1:
        greens_only = st.toggle("🟢 Greens only", value=False, key="sog_greens_only")
    with c2:
        ev_only = st.toggle("💰 +EV only", value=False, key="sog_ev_only")
    with c3:
        hide_red_rows = st.toggle("Hide 🔴 rows", value=True, key="sog_hide_reds")
    with c4:
        plays_first = st.toggle("Plays first", value=True, key="sog_plays_first")
    with c5:
        show_all = st.toggle("Show all", value=False, key="sog_show_all")
    with c6:
        min_conf = st.slider("Min Conf (SOG)", 0, 100, 77, 1, key="sog_min_conf")

    color_pick = st.multiselect(
        "Colors (SOG)",
        ["green", "yellow", "blue", "red"],
        default=["green", "yellow", "blue"],
        key="sog_colors",
    )


    if not show_all:
        df_s = df_s[df_s["Conf_SOG"].fillna(0) >= min_conf]
        if "Color_SOG" in df_s.columns and color_pick:
            df_s = df_s[df_s["Color_SOG"].isin(color_pick)]
    # ---- Visual-style filters (Board-like) ----
    if hide_red_rows and "Matrix_SOG" in df_s.columns:
        df_s = df_s[safe_str(df_s, "Matrix_SOG", "").str.strip().str.lower() != "red"]

    if greens_only and "Green_SOG" in df_s.columns:
        df_s = df_s[df_s["Green_SOG"].fillna(False).astype(bool)]

    if ev_only:
        if "Plays_EV_SOG" in df_s.columns:
            df_s = df_s[df_s["Plays_EV_SOG"].astype(str) == "💰"]
        elif "💰" in df_s.columns:
            df_s = df_s[df_s["💰"] == "💰"]

    if "Markets" not in df_s.columns:
        df_s["Markets"] = df_s.apply(_markets_pills_row, axis=1)

    if plays_first:
        df_s["_g"] = df_s.get("Green_SOG", False).fillna(False).astype(bool)
        df_s["_ev"] = (df_s.get("Plays_EV_SOG", "").astype(str) == "💰")
        df_s["_c"] = safe_num(df_s, "Conf_SOG", 0)
        df_s = df_s.sort_values(["_g", "_ev", "_c"], ascending=[False, False, False]).drop(columns=["_g","_ev","_c"], errors="ignore")


    df_s["Green"] = df_s["Green_SOG"].map(lambda x: "🟢" if bool(x) else "")
    df_s["LOCK"] = df_s.apply(lambda r: lock_badge(r.get("Green_SOG", False), r.get("Plays_EV_SOG", "")), axis=1)

    sog_cols_signal = [
        "Game","Player","Pos","Team","Opp","Markets",
        "Tier_Tag",
        "Matrix_SOG","Conf_SOG","Green",
        "Plays_EV_SOG","SOG_EV%","LOCK",
        "SOG_Line","SOG_Odds_Over","SOG_Book",
        "SOG_Call",
    ]

    show_table(df_s, sog_cols_signal, "SOG — Signals (simple first)")

    with st.expander("Details / Why (SOG)"):
        sog_cols_detail = [
            "Game","Player","Pos","Team","Opp","Markets",
            "Tier_Tag",
            "Matrix_SOG","Conf_SOG","Green","Plays_EV_SOG","SOG_EV%","LOCK",
            "SOG_Line","SOG_Book","SOG_Odds_Over","SOG_Model%","SOG_Imp%","SOG_EV%",
            "Reg_Heat_S","Reg_Gap_S10","Exp_S_10","L10_S",
            "SOG_ProofCount","SOG_Why",
            "iXG%","TOI_Pct","shot_intent_5","shot_intent_10",
            "Opp_Goalie","Opp_SV","Opp_GAA","Goalie_Weak","Opp_DefWeak",
            "Drought_SOG3","Drought_SOG2","Best_Drought",
            "Line","Odds","Result",
        ]
        show_table(df_s, sog_cols_detail, "SOG — Details")



# =========================
# GOAL
# =========================
elif page == "GOAL (1+)":
    df_g = df_f.copy()
    df_g["_cg"] = safe_num(df_g, "Conf_Goal", 0)
    df_g = df_g.sort_values(["_cg"], ascending=[False]).drop(columns=["_cg"], errors="ignore")

    st.subheader("GOAL (1+)")
    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1, 2])
    with c1:
        greens_only = st.toggle("🟢 Greens only", value=False, key="g_greens_only")
    with c2:
        ev_only = st.toggle("💰 +EV only", value=False, key="g_ev_only")
    with c3:
        hide_red_rows = st.toggle("Hide 🔴 rows", value=True, key="g_hide_reds")
    with c4:
        plays_first = st.toggle("Plays first", value=True, key="g_plays_first")
    with c5:
        show_all = st.toggle("Show all", value=False, key="g_show_all")
    with c6:
        min_conf = st.slider("Min Conf (Goal)", 0, 100, 77, 1, key="g_min_conf")

    color_pick = st.multiselect(
        "Colors (Goal)",
        ["green", "yellow", "blue", "red"],
        default=["green", "yellow", "blue"],
        key="g_colors",
    )


    if not show_all:
        df_g = df_g[df_g["Conf_Goal"].fillna(0) >= min_conf]
        if "Color_Goal" in df_g.columns and color_pick:
            df_g = df_g[df_g["Color_Goal"].isin(color_pick)]
    # ---- Visual-style filters (Board-like) ----
    if hide_red_rows and "Matrix_Goal" in df_g.columns:
        df_g = df_g[safe_str(df_g, "Matrix_Goal", "").str.strip().str.lower() != "red"]

    if greens_only and "Green_Goal" in df_g.columns:
        df_g = df_g[df_g["Green_Goal"].fillna(False).astype(bool)]

    if ev_only:
        if "Plays_EV_ATG" in df_g.columns:
            df_g = df_g[df_g["Plays_EV_ATG"].astype(str) == "💰"]
        elif "💰" in df_g.columns:
            df_g = df_g[df_g["💰"] == "💰"]

    if "Markets" not in df_g.columns:
        df_g["Markets"] = df_g.apply(_markets_pills_row, axis=1)

    if plays_first:
        df_g["_g"] = df_g.get("Green_Goal", False).fillna(False).astype(bool)
        df_g["_ev"] = (df_g.get("Plays_EV_ATG", "").astype(str) == "💰")
        df_g["_c"] = safe_num(df_g, "Conf_Goal", 0)
        df_g = df_g.sort_values(["_g", "_ev", "_c"], ascending=[False, False, False]).drop(columns=["_g","_ev","_c"], errors="ignore")


    df_g["Green"] = df_g.get("Green_Goal", False).map(lambda x: "🟢" if bool(x) else "")

    goal_cols_signal = [
        "Game","Player","Pos","Team","Opp","Markets",
        "Tier_Tag",
        "Matrix_Goal","Conf_Goal","Green",
        "Plays_EV_ATG","ATG_EV%","LOCK",
        "ATG_Line","ATG_Odds_Over","ATG_Book",
        "Goal_Call",
    ]

    show_table(df_g, goal_cols_signal, "GOAL (1+) — Signals (simple first)")

    with st.expander("Details / Why (GOAL 1+)"):
        goal_cols_detail = [
            "Game","Player","Pos","Team","Opp","Markets",
            "Tier_Tag",
            "Matrix_Goal","Conf_Goal","Green","Plays_EV_ATG","ATG_EV%","LOCK",
            "ATG_Line","ATG_Book","ATG_Odds_Over","ATG_Model%","ATG_Imp%","ATG_EV%",
            "Reg_Heat_G","Reg_Gap_G10","Exp_G_10","L10_G",
            "Goal_ProofCount","Goal_Why",
            "iXG%","TOI_Pct","shot_intent_5","shot_intent_10",
            "Opp_Goalie","Opp_SV","Opp_GAA","Goalie_Weak","Opp_DefWeak",
            "Drought_G","Best_Drought",
            "Line","Odds","Result",
        ]
        show_table(df_g, goal_cols_detail, "GOAL (1+) — Details")


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







