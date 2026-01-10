import os
import glob
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


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
# HELPERS
# =========================
def find_latest_tracker_csv(output_dir: str) -> str | None:
    patterns = [
        os.path.join(output_dir, "tracker_*.csv"),
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
    view = df[cols].copy()

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
        "Goalie_Weak", "Opp_DefWeak",
        "TOI_Pct", "StarScore",
        "ShotIntent_Pct",
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
    st.dataframe(g, use_container_width=True, hide_index=True)


def show_table(df: pd.DataFrame, cols: list[str], title: str):
    st.subheader(title)

    existing = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]

    if missing:
        with st.expander("Missing columns (safe to ignore)"):
            st.write(missing)

    styled = style_df(df, existing)
    st.dataframe(styled, use_container_width=True, hide_index=True)


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



df = add_ui_columns(df)

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
        return {"SOG": 75, "Points": 77, "Goal": 79}[market]
    elif slate_games >= 5:
        return {"SOG": 74, "Points": 76, "Goal": 78}[market]
    else:
        return {"SOG": 73, "Points": 75, "Goal": 77}[market]


# --- Earned greens (match YOUR columns)
thr_s = _green_conf_threshold("SOG", slate_games)
thr_p = _green_conf_threshold("Points", slate_games)
thr_g = _green_conf_threshold("Goal", slate_games)

df["Green_SOG"] = (
    (safe_num(df, "Conf_SOG", 0) >= thr_s)
    & (safe_str(df, "Matrix_SOG", "").str.strip().str.lower() == "green")
)

df["Green_Points"] = (
    (safe_num(df, "Conf_Points", 0) >= thr_p)
    & (safe_str(df, "Matrix_Points", "").str.strip().str.lower() == "green")
    & (
        (safe_str(df, "Reg_Heat_P", "").str.strip().str.upper() == "HOT")
        | (safe_str(df, "Play_Tag", "").str.contains("HOT", case=False, na=False))
        | (safe_str(df, "🔥", "") == "🔥")
    )
)

df["Green_Goal"] = (
    (safe_num(df, "Conf_Goal", 0) >= thr_g)
    & (safe_str(df, "Matrix_Goal", "").str.strip().str.lower() == "green")
    & (
        (safe_str(df, "Reg_Heat_G", "").str.strip().str.upper() == "HOT")
        | (safe_num(df, "Goalie_Weak", 0) >= 80)
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

proof_ixA = (df["iXA%"] >= 88)
proof_v2 = (df["v2_player_stability"] >= 65)
proof_team = (df["team_5v5_xGF60_pct"] >= 65)
proof_vol = (
    (df["Assist_Volume"] >= 6)
    | (df["i5v5_primaryAssists60"] >= 0.45)
)

proofs = pd.concat([proof_ixA, proof_v2, proof_team, proof_vol], axis=1).fillna(False)
df["Assist_ProofCount"] = proofs.sum(axis=1)

tier = df["Talent_Tier"].astype(str).str.upper()
is_star = tier.isin(["ELITE", "STAR"])

earned_gate = (df["Assist_ProofCount"] >= 2) | (is_star & (df["Assist_ProofCount"] >= 1))

assists_green_earned = (
    (safe_str(df, "Matrix_Assists", "").str.strip().str.lower() == "green")
    & (safe_num(df, "Conf_Assists", 0) >= 72)
    & earned_gate
)

df["Plays_Assists"] = assists_green_earned.fillna(False)

def _assist_why(r):
    reasons = []
    if _get(r, "iXA%", 0) >= 92:
        reasons.append("iXA")
    if _get(r, "v2_player_stability", 0) >= 60:
        reasons.append("v2")
    if _get(r, "team_5v5_xGF60_pct", 0) >= 60:
        reasons.append("xGF")
    if (_get(r, "Assist_Volume", 0) >= 6) or (_get(r, "i5v5_primaryAssists60", 0) >= 0.45):
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
page = st.sidebar.radio("Page", ["Board", "Points", "Assists", "SOG", "Goal", "Raw CSV"], index=0)

# Apply shared filters
df_f = filter_common(df)

# Show slate times table
show_games_times(df_f)


# =========================
# BOARD
# =========================
if page == "Board":
    df_b = sort_board(df_f)

    board_cols = [
        "Game",
        "Player", "Pos", "Team", "Opp",
        "Best_Market",
        "Best_Conf","Tier_Tag", "iXG%", "iXA%",
        "🔥", "Play_Tag",
        "Goalie_Weak", "Opp_DefWeak",
        "Opp_Goalie", "Opp_SV", "Opp_GAA",
        "Matrix_Points", "Conf_Points", "Reg_Heat_P", "Reg_Gap_P10",
        "Matrix_SOG", "Conf_SOG", "Reg_Heat_S", "Reg_Gap_S10",
        "Matrix_Goal", "Conf_Goal", "Reg_Heat_G", "Reg_Gap_G10",
        "Matrix_Assists", "Conf_Assists", "Reg_Heat_A", "Reg_Gap_A10",
        "Line", "Odds", "Result",
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
    min_conf = st.sidebar.slider("Min Conf (Points)", 0, 100, 80, 1)
    color_pick = st.sidebar.multiselect(
        "Colors (Points)",
        ["green", "yellow", "blue", "red"],
        default=["green", "yellow", "blue"]
    )

    df_p = df_p[df_p["Conf_Points"].fillna(0) >= min_conf]
    if "Color_Points" in df_p.columns and color_pick:
        df_p = df_p[df_p["Color_Points"].isin(color_pick)]

    df_p["Green"] = df_p["Green_Points"].map(lambda x: "🟢" if bool(x) else "")

    points_cols = [
        "Game",
        "Player", "Pos", "Team", "Opp", 
        "Matrix_Points",
        "Conf_Points", "Green","Tier_Tag","Best_Drought", "🔥", 
        "Reg_Heat_P", "Reg_Gap_P10", "Exp_P_10", "L10_P",
        "iXA%", "iXG%", "v2_player_stability", "Opp_Goalie", "Opp_SV",
        "Goalie_Weak", "Opp_DefWeak",
        "Line", "Odds", "Result",
    ]

    show_table(df_p, points_cols, "Points View")


# =========================
# ASSISTS
# =========================
elif page == "Assists":
    df_a = df_f.copy()
    df_a["_ca"] = safe_num(df_a, "Conf_Assists", 0)
    df_a = df_a.sort_values(["_ca"], ascending=[False]).drop(columns=["_ca"], errors="ignore")

    st.sidebar.subheader("Assists Filters")
    min_conf = st.sidebar.slider("Min Conf (Assists)", 0, 100, 80, 1)
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
        "Game",
        "Player", "Pos", "Team", "Opp", 
        "Matrix_Assists",
        "Conf_Assists", "Green","Tier_Tag","Best_Drought",
        "Assist_ProofCount", "Assist_Why",
        "Reg_Heat_A", "Reg_Gap_A10", "Exp_A_10", "L10_A",
        "iXA%", "v2_player_stability", 
        "Opp_Goalie", "Opp_SV",
        "Goalie_Weak", "Opp_DefWeak",
        "Play_Tag",
        "Line", "Odds", "Result",
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
    min_conf = st.sidebar.slider("Min Conf (SOG)", 0, 100, 80, 1)
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
        "Game",
        "Player", "Pos", "Team", "Opp", 
        "Matrix_SOG",
        "Conf_SOG", "Green","Tier_Tag","Best_Drought",
        "Reg_Heat_S", "Reg_Gap_S10", "Exp_S_10", "L10_S",
         "Med10_SOG", "Avg5_SOG", "ShotIntent", "ShotIntent_Pct", "Opp_Goalie", "Opp_SV",
        "Goalie_Weak", "Opp_DefWeak",
        "Line", "Odds", "Result",
    ]

    show_table(df_s, sog_cols, "SOG View")


# =========================
# GOAL
# =========================
elif page == "Goal":
    df_g = df_f.copy()
    df_g["_cg"] = safe_num(df_g, "Conf_Goal", 0)
    df_g = df_g.sort_values(["_cg"], ascending=[False]).drop(columns=["_cg"], errors="ignore")

    st.sidebar.subheader("Goal Filters")
    min_conf = st.sidebar.slider("Min Conf (Goal)", 0, 100, 80, 1)
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
        "Conf_Goal", "Green","Tier_Tag","Best_Drought",
        "Reg_Heat_G", "Reg_Gap_G10", "Exp_G_10", "L10_G",
        "iXG%", "iXA%", "L5_G", "Opp_Goalie", "Opp_SV",
        "Goalie_Weak", "Opp_DefWeak",
        "Line", "Odds", "Result",
    ]

    show_table(df_g, goal_cols, "Goal View")


# =========================
# RAW
# =========================
else:
    st.subheader("Raw CSV (all columns)")
    st.dataframe(df_f, use_container_width=True, hide_index=True)
