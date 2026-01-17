import os
import glob
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

COLUMN_WIDTHS = {
    # identity
    "Game": "small",
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

    # drought
    "Drought_P": "small",
    "Drought_A": "small",
    "Drought_G": "small",
    "Drought_SOG": "small",
    "Best_Drought": "small",

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
# HELPERS
# =========================
def find_latest_tracker_csv(output_dir: str) -> str | None:
    # Prefer the file your engine actually writes
    preferred = os.path.join(output_dir, "nhl_edge_tracker_latest.csv")
    if os.path.isfile(preferred):
        return preferred

    # Otherwise fall back to dated engine outputs
    patterns = [
        os.path.join(output_dir, "nhl_edge_tracker_*.csv"),
        os.path.join(output_dir, "*.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
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
    """
    Still used for Raw/optional, but NOT used for the sortable tables.
    (Kept here in case you want it later.)
    """
    view = df[cols].copy()

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

    def weak_style(v):
        try:
            x = float(v)
        except Exception:
            return ""
        if x >= 75:
            return "background-color:#b30000;color:white;font-weight:700;"
        return ""

    sty = view.style

    for c in ["Matrix_Points", "Matrix_SOG", "Matrix_Assists", "Matrix_Goal"]:
        if c in view.columns:
            sty = sty.applymap(matrix_style, subset=[c])

    for c in ["Reg_Heat_P", "Reg_Heat_S", "Reg_Heat_G", "Reg_Heat_A"]:
        if c in view.columns:
            sty = sty.applymap(heat_style, subset=[c])

    for c in ["Best_Conf", "Conf_Points", "Conf_SOG", "Conf_Goal", "Conf_Assists"]:
        if c in view.columns:
            sty = sty.applymap(conf_style, subset=[c])

    for c in ["Goalie_Weak", "Opp_DefWeak"]:
        if c in view.columns:
            sty = sty.applymap(weak_style, subset=[c])

    # ✅ FIXED INDENT: formatting is OUTSIDE loops
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
        "Goalie_Weak", "Opp_DefWeak",
        "L10_P","L10_A","L10_G","L10_SOG","L10_S",
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

    if format_dict:
        sty = sty.format(format_dict, na_rep="")

    return sty


def show_table(df: pd.DataFrame, cols: list[str], title: str, styled: bool = True):
    st.subheader(title)

    existing = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]

    if missing:
        with st.expander("Missing columns (safe to ignore)"):
            st.write(missing)

    if styled:
        # ✅ colored cells again
        sty = style_df(df, existing)
        st.dataframe(sty, use_container_width=True, hide_index=True)
    else:
        # ✅ sortable plain table
        st.dataframe(
            df[existing],
            use_container_width=True,
            hide_index=True,
            column_config=build_column_config(df, existing),
        )


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

# =========================
# ALIASES / COMPAT LAYER (v7.2)
# =========================
def _series_default(df: pd.DataFrame, default=np.nan) -> pd.Series:
    return pd.Series([default] * len(df), index=df.index)

def _first_present(df: pd.DataFrame, names: list[str], numeric: bool = False, default=np.nan) -> pd.Series:
    for n in names:
        if n in df.columns:
            s = df[n]
            return pd.to_numeric(s, errors="coerce") if numeric else s
    return _series_default(df, default=default)

# ---- Last10 aliases (so Board/SOG pages always have L10_*)
df["L10_P"] = _first_present(df, ["L10_P", "P10_total", "P10"], numeric=True)
df["L10_G"] = _first_present(df, ["L10_G", "G10_total", "G10"], numeric=True)
df["L10_A"] = _first_present(df, ["L10_A", "A10_total", "A10"], numeric=True)
df["L10_S"] = _first_present(df, ["L10_S", "S10_total", "L10_SOG", "L10_SOG_total"], numeric=True)

# ---- Expectations aliases
df["Exp_P_10"] = _first_present(df, ["Exp_P_10", "Exp_P10"], numeric=True)
df["Exp_G_10"] = _first_present(df, ["Exp_G_10", "Exp_G10"], numeric=True)
df["Exp_A_10"] = _first_present(df, ["Exp_A_10", "Exp_A10"], numeric=True)
df["Exp_S_10"] = _first_present(df, ["Exp_S_10", "Exp_S10"], numeric=True)

# ---- Regression gaps (compute if missing/blank)
def _fill_gap(col_gap: str, col_exp: str, col_l10: str):
    if (col_gap not in df.columns) or df[col_gap].isna().all():
        df[col_gap] = (pd.to_numeric(df[col_exp], errors="coerce") - pd.to_numeric(df[col_l10], errors="coerce")).round(2)

_fill_gap("Reg_Gap_P10", "Exp_P_10", "L10_P")
_fill_gap("Reg_Gap_G10", "Exp_G_10", "L10_G")
_fill_gap("Reg_Gap_A10", "Exp_A_10", "L10_A")
_fill_gap("Reg_Gap_S10", "Exp_S_10", "L10_S")

# ---- Heat (compute if missing/blank)
def _heat_from_gap(x) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        v = float(x)
    except Exception:
        return ""
    # Use your same vibe as engine: HOT / WARM / COOL
    if v >= 2.0:
        return "HOT"
    if v >= 1.0:
        return "WARM"
    return "COOL"

for gap, heat in [
    ("Reg_Gap_P10", "Reg_Heat_P"),
    ("Reg_Gap_G10", "Reg_Heat_G"),
    ("Reg_Gap_A10", "Reg_Heat_A"),
    ("Reg_Gap_S10", "Reg_Heat_S"),
]:
    if (heat not in df.columns) or df[heat].astype(str).str.strip().eq("").all():
        df[heat] = df[gap].apply(_heat_from_gap)



def add_ui_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Ensure these exist
    if "Play_Tag" not in out.columns:
        out["Play_Tag"] = ""
    if "Plays_Points" not in out.columns:
        out["Plays_Points"] = False

    plays_points = (
        to_bool_series(out["Plays_Points"])
        if "Plays_Points" in out.columns else pd.Series(False, index=out.index)
    )

    # Fire indicator
    out["🔥"] = plays_points.map(lambda x: "🔥" if bool(x) else "")
    return out


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
        games = sorted([g for g in out["Game"].dropna().astype(str).unique().tolist() if g.strip()])
        sel_games = st.sidebar.multiselect("Matchup", games, default=[])
        if sel_games:
            out = out[out["Game"].astype(str).isin(sel_games)]

    # Only 🔥 plays
    only_fire = st.sidebar.checkbox("Only 🔥 plays", value=False)
    if only_fire and "🔥" in out.columns:
        out = out[out["🔥"] == "🔥"]

    return out


def sort_board(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_bc"] = safe_num(out, "Best_Conf", 0)
    out["_gw"] = safe_num(out, "Goalie_Weak", 50)
    out["_dw"] = safe_num(out, "Opp_DefWeak", 50)
    out = out.sort_values(["_bc", "_gw", "_dw"], ascending=[False, False, False])
    return out.drop(columns=["_bc", "_gw", "_dw"], errors="ignore")


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
    st.dataframe(g, use_container_width=True, hide_index=True)


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

# =========================
# v7.2 COMPAT: create replacement / alias columns
# Put this RIGHT AFTER df is loaded
# =========================
def ensure_v72_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    def _num(col, default=np.nan):
        if col not in out.columns:
            return pd.Series([default] * len(out), index=out.index)
        return pd.to_numeric(out[col], errors="coerce")

    def _txt(col, default=""):
        if col not in out.columns:
            return pd.Series([default] * len(out), index=out.index)
        return out[col].astype(str).fillna(default)

    def _first_present(cands: list[str], kind="num", default=np.nan):
        for c in cands:
            if c in out.columns:
                return _num(c, default) if kind == "num" else _txt(c, default)
        return pd.Series([default] * len(out), index=out.index)

    # -------------------------
    # L10 / L5 legacy columns
    # -------------------------
    out["L10_SOG"] = _first_present(["L10_SOG", "L10_S", "S10_total"], kind="num")
    out["L10_S"]   = _first_present(["L10_S", "L10_SOG", "S10_total"], kind="num")

    out["L10_G"]   = _first_present(["L10_G", "G10_total"], kind="num")
    out["L10_A"]   = _first_present(["L10_A", "A10_total", "L10_AST"], kind="num")
    out["L10_P"]   = _first_present(["L10_P", "P10_total"], kind="num")

    out["L5_G"]    = _first_present(["L5_G", "G5_total"], kind="num")
    out["L5_A"]    = _first_present(["L5_A", "A5_total"], kind="num")

    # -------------------------
    # Expected / Regression legacy columns
    # -------------------------
    # Expectations
    out["Exp_P_10"] = _first_present(["Exp_P_10", "Exp_P10"], kind="num")
    out["Exp_G_10"] = _first_present(["Exp_G_10", "Exp_G10"], kind="num")
    out["Exp_A_10"] = _first_present(["Exp_A_10", "Exp_A10"], kind="num")
    out["Exp_S_10"] = _first_present(["Exp_S_10", "Exp_S10"], kind="num")

    # Regression gaps
    out["Reg_Gap_G10"] = _first_present(
        ["Reg_Gap_G10", "RegGap_G10", "Reg_Gap_G"],
        kind="num"
    )
st.write(df.filter(regex="Reg_").head(10))


# --- Regression/L10 alias fixes (engine naming drift) ---
# SOG
if "L10_S" not in df.columns and "L10_SOG" in df.columns:
    df["L10_S"] = pd.to_numeric(df["L10_SOG"], errors="coerce")
if "L10_SOG" not in df.columns and "L10_S" in df.columns:
    df["L10_SOG"] = pd.to_numeric(df["L10_S"], errors="coerce")

# Assists
if "L10_A" not in df.columns and "L10_AST" in df.columns:
    df["L10_A"] = pd.to_numeric(df["L10_AST"], errors="coerce")
if "Reg_Heat_S" not in df.columns and "Reg_Heat_SOG" in df.columns:
    df["Reg_Heat_S"] = df["Reg_Heat_SOG"]
if "Reg_Heat_A" not in df.columns and "Reg_Heat_AST" in df.columns:
    df["Reg_Heat_A"] = df["Reg_Heat_AST"]



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

# -------------------------
# Tier tags (do NOT overwrite if CSV already has tags)
# -------------------------
if "Tier_Tag" not in df.columns:
    df["Tier_Tag"] = ""

# If Tier_Tag is blank everywhere, build it from Talent_Tier
if df["Tier_Tag"].astype(str).str.strip().eq("").all():
    tt = df.get("Talent_Tier", "NONE").astype(str).str.upper().fillna("NONE")
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


def add_display_badges(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def m_badge(v):
        s = str(v).strip().upper()
        if s == "GREEN":
            return "🟢 Green"
        if s == "YELLOW":
            return "🟡 Yellow"
        if s == "RED":
            return "🔴 Red"
        if s.startswith("FAIL"):
            return f"⛔ {s}"
        return s

    def h_badge(v):
        s = str(v).strip().upper()
        if s == "HOT":
            return "🔥 HOT"
        if s == "WARM":
            return "🟠 WARM"
        if s == "COOL":
            return "🧊 COOL"
        return s

    # Matrix badges
    for c in ["Matrix_Points", "Matrix_SOG", "Matrix_Goal", "Matrix_Assists"]:
        if c in out.columns:
            out[c + "_B"] = out[c].map(m_badge)

    # Heat badges
    for c in ["Reg_Heat_P", "Reg_Heat_S", "Reg_Heat_G", "Reg_Heat_A"]:
        if c in out.columns:
            out[c + "_B"] = out[c].map(h_badge)

    return out




df = add_ui_columns(df)

df = add_display_badges(df)


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
    if slate_games >= 8:
        return {"SOG": 80, "Points": 77, "Goal": 85, "Assists": 77}[market]
    elif slate_games >= 5:
        return {"SOG": 80, "Points": 76, "Goal": 84, "Assists": 77}[market]
    else:
        return {"SOG": 80, "Points": 75, "Goal": 83, "Assists": 77}[market]




# --- Earned greens (match YOUR columns)
thr_s = _green_conf_threshold("SOG", slate_games)
thr_p = _green_conf_threshold("Points", slate_games)
thr_s = _green_conf_threshold("SOG", slate_games)
# =========================
# GOAL — earned green (v2 proof-count + tier-aware drought)
# =========================

thr_g = _green_conf_threshold("Goal", slate_games)

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
    df.get("Plays_Points", pd.Series(False, index=df.index)).fillna(False)
    | df.get("Plays_Assists", pd.Series(False, index=df.index)).fillna(False)
    | df.get("Green_SOG", pd.Series(False, index=df.index)).fillna(False)
    | df.get("Green_Goal", pd.Series(False, index=df.index)).fillna(False)
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


# Navigation
page = st.sidebar.radio(
    "Page",
    ["Board", "Points", "Assists", "SOG", "Goal","Guide", "Ledger", "Raw CSV"],
    index=0
)

df_f = filter_common(df)

# Show slate times table
show_games_times(df_f)


# =========================
# BOARD
# =========================
if page == "Board":
    df_b = sort_board(df_f)

    board_cols = [
    "Game","Player","Pos","Team","Opp",
    "Best_Market","Best_Conf","Tier_Tag","iXG%","iXA%",
    "🔥",
    "Goalie_Weak","Opp_DefWeak",
    "Opp_Goalie","Opp_SV","Opp_GAA",
    "Matrix_Points_B","Conf_Points","Reg_Heat_P_B","Reg_Gap_P10",
    "Matrix_SOG_B","Conf_SOG","Reg_Heat_S_B","Reg_Gap_S10",
    "Matrix_Goal_B","Conf_Goal","Reg_Heat_G_B","Reg_Gap_G10",
    "Matrix_Assists_B","Conf_Assists","Reg_Heat_A_B","Reg_Gap_A10",
    "Line","Odds","Result",
]

    

    show_table(df_b, board_cols, "Board (sorted by Best_Conf)")


# =========================
# POINTS
# =========================
elif page == "Points":
    df_p = df_f.copy()
    df_p["_cp"] = safe_num(df_p, "Conf_Points", 0)
    df_p = df_p.sort_values(["_cp"], ascending=[False]).drop(columns=["_cp"], errors="ignore")

    st.sidebar.subheader("Points Filters")
    min_conf = st.sidebar.slider("Min Conf (Points)", 0, 100, 77, 1)

    df_p = df_p[df_p["Conf_Points"].fillna(0) >= min_conf]
    df_p["Green"] = df_p.get("Green_Points", False).map(lambda x: "🟢" if bool(x) else "")

    points_cols = [
        "Game","Player","Pos","Team","Opp",
        "Matrix_Points_B","Conf_Points","Green","GF_Gate_Badge","Tier_Tag",
        "Reg_Heat_P_B","Reg_Gap_P10","Exp_P_10","L10_P",
        "iXG%","iXA%",
        "Opp_Goalie","Opp_SV","Opp_GAA","Goalie_Weak","Opp_DefWeak",
        "Drought_P","Best_Drought",
        "Line","Odds","Result",
    ]

    show_table(df_p, points_cols, "Points View")

    show_table(df_p, points_cols, "Points View")


# =========================
# ASSISTS
# =========================
elif page == "Assists":
    df_a = df_f.copy()
    df_a["_ca"] = safe_num(df_a, "Conf_Assists", 0)
    df_a = df_a.sort_values(["_ca"], ascending=[False]).drop(columns=["_ca"], errors="ignore")

    st.sidebar.subheader("Assists Filters")
    min_conf = st.sidebar.slider("Min Conf (Assists)", 0, 100, 77, 1)
    color_pick = st.sidebar.multiselect(
        "Colors (Assists)",
        ["green", "yellow", "blue", "red"],
        default=["green", "yellow", "blue"]
    )

    df_a = df_a[df_a["Conf_Assists"].fillna(0) >= min_conf]
    if "Color_Assists" in df_a.columns and color_pick:
        df_a = df_a[df_a["Color_Assists"].isin(color_pick)]

    df_a["Green"] = df_a.get("Green_Assists", False).map(lambda x: "🟢" if bool(x) else "")

    assists_cols = [
        "Game","Player","Pos","Team","Opp",
        "Matrix_Assists_B",
        "Conf_Assists","Green","GF_Gate_Badge","Tier_Tag","Drought_A","Best_Drought",
        "Assist_ProofCount","Assist_Why",
        "Reg_Heat_A_B","Reg_Gap_A10","Exp_A_10","L10_A",
        "iXA%","iXG%","v2_player_stability",
        "Opp_Goalie","Opp_SV",
        "Goalie_Weak","Opp_DefWeak",
        "Line","Odds","Result",
    ]

        
    

    show_table(df_a, assists_cols, "Assists View")


# =========================
# SOG
# =========================
elif page == "SOG":
    df_s = df_f.copy()
    df_s["_cs"] = safe_num(df_s, "Conf_SOG", 0)
    df_s = df_s.sort_values(["_cs"], ascending=[False]).drop(columns=["_cs"], errors="ignore")

    st.sidebar.subheader("SOG Filters")
    min_conf = st.sidebar.slider("Min Conf (SOG)", 0, 100, 77, 1)
    color_pick = st.sidebar.multiselect(
        "Colors (SOG)",
        ["green", "yellow", "blue", "red"],
        default=["green", "yellow", "blue"]
    )

    df_s = df_s[df_s["Conf_SOG"].fillna(0) >= min_conf]
    if "Color_SOG" in df_s.columns and color_pick:
        df_s = df_s[df_s["Color_SOG"].isin(color_pick)]

    df_s["Green"] = df_s["Green_SOG"].map(lambda x: "🟢" if bool(x) else "")

    sog_cols = [
        "Game","Player","Pos","Team","Opp",
        "Matrix_SOG_B",
        "Conf_SOG","Green","Tier_Tag","Drought_SOG","Best_Drought",
        "Reg_Heat_S_B","Reg_Gap_S10","Exp_S_10","L10_S",
        "Med10_SOG","Avg5_SOG","ShotIntent","ShotIntent_Pct",
        "Opp_Goalie","Opp_SV",
        "Goalie_Weak","Opp_DefWeak",
        "Line","Odds","Result",
    ]    


    if "Tier_Tag_SOG" in df_s.columns:
        df_s["Tier_Tag"] = df_s["Tier_Tag_SOG"].fillna(df_s.get("Tier_Tag", ""))


    show_table(df_s, sog_cols, "SOG View")


# =========================
# GOAL
# =========================
elif page == "Goal":
    df_g = df_f.copy()
    df_g["_cg"] = safe_num(df_g, "Conf_Goal", 0)
    df_g = df_g.sort_values(["_cg"], ascending=[False]).drop(columns=["_cg"], errors="ignore")

    st.sidebar.subheader("Goal Filters")
    min_conf = st.sidebar.slider("Min Conf (Goal)", 0, 100, 77, 1)
    color_pick = st.sidebar.multiselect(
        "Colors (Goal)",
        ["green", "yellow", "blue", "red"],
        default=["green", "yellow", "blue"]
    )

    df_g = df_g[df_g["Conf_Goal"].fillna(0) >= min_conf]
    if "Color_Goal" in df_g.columns and color_pick:
        df_g = df_g[df_g["Color_Goal"].isin(color_pick)]

    df_g["Green"] = df_g.get("Green_Goal", False).map(lambda x: "🟢" if bool(x) else "")

    goal_cols = [
        "Game",
        "Player", "Pos", "Team", "Opp",
        "Matrix_Goal", 
        "Conf_Goal", "Green","GF_Gate_Badge", "Tier_Tag","Drought_G","Best_Drought",
        "Reg_Heat_G", "Reg_Gap_G10", "Exp_G_10", "L10_G",
        "iXG%", "iXA%", "L5_G", "Opp_Goalie", "Opp_SV",
        "Goalie_Weak", "Opp_DefWeak",
        "Line", "Odds", "Result",
    ]

    show_table(df_g, goal_cols, "Goal View")

elif page == "Guide":
    st.subheader("📘 Guide — How to use The Warlord’s NHL Prop Tool")

    st.markdown("""
## The 60-second workflow
1) **Start on Board**
   - Sort is already best-first (Best_Conf, then Goalie_Weak, then Opp_DefWeak).
   - Look for **Tier_Tag + HOT regression + weak matchup** stacking.

2) **Open the market view** (Points / SOG / Goal / Assists)
   - Use **Min Conf** slider to tighten.
   - Use **Colors** to hide red.

3) **Only bet “earned greens”**
   - Your “🟢” badge inside each market page is the *actual playable* signal.

---

## Your key signals (what to trust most)
### 1) Matrix_*  (Green/Yellow/Red)
- **Green** = model conditions met (baseline signal)
- **Yellow** = mixed / borderline
- **Red** = failed conditions

### 2) Conf_* (0–100)
- Model confidence after adjustments (injury, environment gates, etc.)
- You’re using color tiers via `_tier_color()`.

### 3) Earned Green (🟢)
This is your “final boss” gating.
A player can look good in raw confidence, but **earned green is what you actually play**.

---

## Earned Green rules (current app logic)

### ✅ SOG Earned Green
Requires:
- Matrix_SOG == **Green**
- Conf_SOG >= threshold (slate-size gated)

### ✅ Points Earned Green
Requires:
- Matrix_Points == **Green**
- Conf_Points >= threshold
- AND one of:
  - Reg_Heat_P == **HOT**
  - Play_Tag contains **HOT**
  - 🔥 flagged

    # -------------------------
    # 🥅 GOAL Earned Green (Streamlit boolean) + Drought lane
    # Requires:
    # - Matrix_Goal == Green
    # - Conf_Goal >= threshold
    # - AND one of:
    #   - Reg_Heat_G == HOT
    #   - Goalie_Weak >= 70
    #   - Goal drought qualifies (tier-aware)
    # -------------------------
    GOAL_CONF_GREEN = 77

    # Ensure numeric safety
    for c in ["Conf_Goal", "Goalie_Weak", "Drought_G"]:
        if c in tracker.columns:
            tracker[c] = pd.to_numeric(tracker[c], errors="coerce")

    tier = tracker.get("Talent_Tier", "").astype(str).str.upper()

    # Tier-aware drought trigger:
    # ELITE: drought >= 2
    # STAR:  drought >= 3
    # NONE:  drought >= 4
    goal_drought_ok = (
        ((tier == "ELITE") & (tracker["Drought_G"].fillna(0) >= 2)) |
        ((tier == "STAR")  & (tracker["Drought_G"].fillna(0) >= 3)) |
        (~tier.isin(["ELITE", "STAR"]) & (tracker["Drought_G"].fillna(0) >= 4))
    )

    tracker["Plays_Goal"] = (
        (tracker["Matrix_Goal"].astype(str) == "Green") &
        (tracker["Conf_Goal"].fillna(0) >= GOAL_CONF_GREEN) &
        (
            tracker["Reg_Heat_G"].astype(str).str.upper().eq("HOT") |
            (tracker["Goalie_Weak"].fillna(0) >= 70) |
            goal_drought_ok
        )
    ).fillna(False)

    # Optional: add a tag so it "pops" in Streamlit
    mask = (
        tracker["Plays_Goal"] &
        ~tracker["Play_Tag"].fillna("").str.contains("GOAL EARNED", regex=False)
    )

    tracker.loc[mask, "Play_Tag"] = np.where(
        tracker.loc[mask, "Play_Tag"].fillna("").astype(str).str.len() > 0,
        tracker.loc[mask, "Play_Tag"].fillna("").astype(str) + " | 🥅 GOAL EARNED",
        "🥅 GOAL EARNED"
    )




### ✅ Assists Earned Green (v1 FINAL)
Requires:
- Matrix_Assists == **Green**
- Conf_Assists >= **77**
- AND earned proof gate:
  - ProofCount >= 2
  - OR (STAR/ELITE AND ProofCount >= 1)

**Assist proofs:**
- iXA% >= 92  → `iXA`
- v2_player_stability >= 65 → `v2`
- team_5v5_xGF60_pct >= 65 → `xGF`
- Assist_Volume >= 6 OR i5v5_primaryAssists60 >= 0.50 → `VOL`

You’ll see: **Assist_ProofCount** and **Assist_Why**.

---

## The matchup columns (how to use them)
- **Goalie_Weak**
  - Higher = worse goalie environment (more attackable)
  - Your styling only “screams red” when >= 75

- **Opp_DefWeak**
  - Higher = softer defense environment / more chances allowed

Best spots are when **Best_Conf is high** AND **Goalie_Weak / Opp_DefWeak are elevated**.

---

## Regression columns (how to interpret)
- **Reg_Heat_*:** HOT / WARM / COOL
- **Reg_Gap_*:** expected minus actual (bigger gap = more “due”)
- **Exp_*_10:** expected output over next 10 (or model window)

In your model:
- **HOT** is not “they’re on a heater”
- **HOT means they’re due** (positive regression pressure)

---

## Drought columns (what they mean in practice)
- **Drought_*:** how long since last event in that market
- **Best_Drought:** whichever drought is most relevant for the player

Drought is useful when it aligns with:
- Matrix Green
- Conf high
- Regression HOT
- Matchup weakness

---

## Filters you should use daily
- **Search player**: fast lookup
- **Team / Matchup**: reduce noise
- **Only 🔥 plays**: isolate your flagged list

---

## Recommended rules for real wagers
**The safe “test phase” rule:**
- Only play **🟢 earned greens**
- Prefer **Tier_Tag (STAR/ELITE)** when slate is big
- Prefer **HOT regression** when choices are similar
- Avoid plays with major injury tags unless you’re intentionally fading

---

## Troubleshooting quick hits
- Missing columns expander = normal (older CSV)
- If a market page looks blank:
  - Your **Min Conf slider** may be too high
  - Or **Color filters** are hiding everything
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
    st.dataframe(df_f, use_container_width=True, hide_index=True)







