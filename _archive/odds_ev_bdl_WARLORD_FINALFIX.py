from __future__ import annotations
"""
odds_ev_bdl_VERIFIED.py — Warlord NHL Prop Tool (ATG-style EV engine)

Key points:
- UNIFORM "ATG-style" model probability across markets using Poisson tails:
    lambda = Exp_*_10 / 10  (per-game rate)
    k      = ceil(line)     (milestone threshold)
    p_model = P(X >= k)

- Baseline-only enforcement:
    SOG: keep only >= 2.0
    Points/Assists/Goal/ATG: keep only 1+ (normalize display to 0.5)

- Robust to input column naming:
    Works whether inputs are BDL_*_Line/Odds/Book (pre-display) OR
    *_Line/*_Odds_Over/*_Book (post-display).

- Adds a visible stamp column so you can confirm the engine is actually running:
    BDL_EV_ENGINE = "ATG_POISSON_v1_2026-01-18"
"""

import os
import math
from typing import Dict, Optional, Tuple, List

import requests
import certifi
import pandas as pd

__BDL_EV_ENGINE__ = "MARKETSFULL_POISSON_v1_2026-01-18"

BDL_NHL_BASE = "https://api.balldontlie.io/nhl/v1"


# -----------------------------
# Small utilities
# -----------------------------

def _safe_float(x) -> float | None:
    try:
        if x is None:
            return None
        if isinstance(x, str) and not x.strip():
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _norm_name(x) -> str:
    if x is None:
        return ""
    s = str(x).lower().strip()
    s = s.replace(".", "").replace("'", "")
    s = " ".join(s.split())
    return s


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def american_to_implied_prob(odds) -> float | None:
    """Convert odds to implied probability.
    Supports:
      - American odds: -120, +150
      - Decimal odds leaks: 1.91, 2.10  -> implied = 1/odds
    """
    try:
        o = float(odds)
    except Exception:
        return None
    if o == 0:
        return None

    # decimal odds heuristic
    if 1.01 <= o <= 20.0:
        return 1.0 / o

    # american odds
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)


def american_payout(odds) -> float | None:
    """Profit on 1u stake."""
    try:
        o = float(odds)
    except Exception:
        return None
    if o == 0:
        return None
    if o > 0:
        return o / 100.0
    return 100.0 / (-o)


def _compute_ev_pct(p_model: float, odds) -> float | None:
    payout = american_payout(odds)
    if payout is None:
        return None
    ev = p_model * payout - (1.0 - p_model) * 1.0
    return ev * 100.0


def _milestone_k(line_value: float) -> int:
    """Convert a line to milestone threshold k for P(X >= k)."""
    try:
        lv = float(line_value)
        if math.isnan(lv) or lv <= 0:
            return 1
        return max(1, int(math.ceil(lv)))
    except Exception:
        return 1


def _poisson_tail_ge_k(lam: float, k: int) -> float:
    """P(X >= k) for Poisson(lam)."""
    if lam is None or lam <= 0 or k <= 0:
        return 0.0
    # 1 - CDF(k-1)
    cdf = 0.0
    term = math.exp(-lam)  # i=0
    cdf += term
    for i in range(1, k):
        term *= lam / float(i)
        cdf += term
    return _clamp(1.0 - cdf, 0.0, 0.999)


def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _lambda_from_exp10_row(row: pd.Series, exp_candidates: List[str], fallback_rate_cols: List[Tuple[str, float]]) -> float:
    """
    Prefer Exp_*_10 totals -> lam = Exp/10.
    If missing, fall back to rate derived from other columns:
      fallback_rate_cols: list of (colname, scale_to_lambda)
        e.g. ("PPG", 1.0) means lambda = PPG
             ("L10_P", 1/10) means lambda = L10_P/10
    """
    for c in exp_candidates:
        if c in row.index:
            v = _safe_float(row.get(c))
            if v is not None and v > 0:
                return v / 10.0

    for c, scale in fallback_rate_cols:
        if c in row.index:
            v = _safe_float(row.get(c))
            if v is not None and v > 0:
                return float(v) * float(scale)

    return 0.0


# -----------------------------
# BallDontLie API helpers
# -----------------------------

def _bdl_headers(api_key: str | None) -> Dict[str, str]:
    k = (api_key or os.getenv("BDL_API_KEY") or os.getenv("BALLDONTLIE_KEY") or os.getenv("BALLDONTLIE_API_KEY") or "").strip()
    return {"Authorization": k} if k else {}


def _bdl_get(url: str, params: dict | list | None, api_key: str | None, timeout: int = 25) -> dict:
    r = requests.get(url, params=params, headers=_bdl_headers(api_key), timeout=timeout, verify=certifi.where())
    r.raise_for_status()
    return r.json()


def fetch_bdl_games_for_date(game_date: str, api_key: str | None = None, per_page: int = 100) -> List[dict]:
    url = f"{BDL_NHL_BASE}/games"
    j = _bdl_get(url, params={"dates[]": game_date, "per_page": per_page}, api_key=api_key)
    return list(j.get("data") or [])


def fetch_bdl_props_for_game(game_id: int, api_key: str | None = None, vendors: Optional[List[str]] = None) -> List[dict]:
    url = f"{BDL_NHL_BASE}/odds/player_props"
    params: List[Tuple[str, str | int]] = [("game_id", int(game_id))]
    if vendors:
        for v in vendors:
            params.append(("vendors[]", str(v)))
    j = _bdl_get(url, params=params, api_key=api_key)
    return list(j.get("data") or [])


def fetch_bdl_players_map(player_ids: List[int], api_key: str | None = None, chunk: int = 80, per_page: int = 100) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    if not player_ids:
        return out

    url = f"{BDL_NHL_BASE}/players"
    ids = [int(x) for x in player_ids if x is not None]

    for i in range(0, len(ids), chunk):
        batch = ids[i : i + chunk]
        params: List[Tuple[str, str | int]] = [("per_page", int(per_page))]
        for pid in batch:
            params.append(("player_ids[]", int(pid)))

        try:
            j = _bdl_get(url, params=params, api_key=api_key)
            rows = list(j.get("data") or [])
            for p in rows:
                try:
                    out[int(p.get("id"))] = p
                except Exception:
                    pass
        except Exception:
            params2: List[Tuple[str, str | int]] = [("per_page", int(per_page))]
            for pid in batch:
                params2.append(("player_ids", int(pid)))
            j = _bdl_get(url, params=params2, api_key=api_key)
            rows = list(j.get("data") or [])
            for p in rows:
                try:
                    out[int(p.get("id"))] = p
                except Exception:
                    pass

    return out


def _player_team_tricode(player_obj: dict) -> str:
    try:
        teams = player_obj.get("teams") or []
        if not teams:
            return ""
        t = teams[-1] or {}
        return str(t.get("tricode") or "").upper().strip()
    except Exception:
        return ""


# -----------------------------
# Merge BDL props -> tracker
# -----------------------------

def merge_bdl_props_mainlines(
    tracker: pd.DataFrame,
    game_date: str,
    api_key: str | None = None,
    vendors: Optional[List[str]] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """Merge baseline BDL lines/odds onto tracker using consensus line selection."""
    df = tracker.copy()
    if df.empty:
        return df

    vendors = vendors or ["draftkings", "fanduel", "caesars"]

    try:
        games = fetch_bdl_games_for_date(game_date, api_key=api_key)
    except Exception as e:
        if debug:
            print(f"[odds/ev] BDL games fetch failed: {e}")
        return df

    game_ids = [g.get("id") for g in games if g.get("id") is not None]
    if not game_ids:
        return df

    all_props: List[dict] = []
    for gid in game_ids:
        try:
            all_props.extend(fetch_bdl_props_for_game(int(gid), api_key=api_key, vendors=vendors))
        except Exception as e:
            if debug:
                print(f"[odds/ev] props fetch failed for game_id={gid}: {e}")

    if not all_props:
        return df

    keep_types = {"shots_on_goal", "points", "assists", "goals", "anytime_goal"}
    props = [p for p in all_props if (p.get("prop_type") in keep_types)]
    if not props:
        return df

    player_ids = sorted({int(p.get("player_id")) for p in props if p.get("player_id") is not None})
    try:
        players_map = fetch_bdl_players_map(player_ids, api_key=api_key)
    except Exception as e:
        if debug:
            print(f"[odds/ev] BDL player lookup failed: {e}")
        return df

    def _odds_from_market(mkt: dict) -> float | None:
        if not isinstance(mkt, dict):
            return None
        if mkt.get("type") == "over_under":
            return _safe_float(mkt.get("over_odds"))
        return _safe_float(mkt.get("odds"))

    def _norm_line(ptype: str, line_value: float | None) -> float | None:
        lv = _safe_float(line_value)
        if lv is None:
            return None
        if ptype == "shots_on_goal":
            return lv if lv >= 2.0 else None
        if ptype in {"points", "assists", "goals", "anytime_goal"}:
            return 0.5 if lv <= 1.0 else None
        return None

    buckets: Dict[Tuple[str, str, str], list] = {}
    for pr in props:
        pid = pr.get("player_id")
        pobj = players_map.get(int(pid)) if pid is not None else None
        if not pobj:
            continue
        name = str(pobj.get("full_name") or "").strip()
        if not name:
            continue
        team = _player_team_tricode(pobj)
        ptype = str(pr.get("prop_type") or "")
        nl = _norm_line(ptype, pr.get("line_value"))
        if nl is None:
            continue
        key = (_norm_name(name), team, ptype)
        buckets.setdefault(key, []).append((float(nl), pr))

    if not buckets:
        return df

    best: Dict[Tuple[str, str, str], dict] = {}
    for key, items in buckets.items():
        counts: Dict[float, int] = {}
        for nl, _pr in items:
            counts[nl] = counts.get(nl, 0) + 1
        consensus_line = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        best_pr = None
        best_odds = None
        for nl, pr in items:
            if float(nl) != float(consensus_line):
                continue
            o = _odds_from_market(pr.get("market") or {})
            if o is None:
                continue
            if best_odds is None or float(o) > float(best_odds):
                best_odds = float(o)
                best_pr = pr
        if best_pr is None:
            best_pr = items[0][1]
        best[key] = best_pr

    if "Player_norm" in df.columns:
        df["__player_key"] = df["Player_norm"].map(_norm_name)
    else:
        df["__player_key"] = df["Player"].map(_norm_name)
    df["__team_key"] = df.get("Team_norm", df.get("Team", "")).astype(str).str.upper().str.strip()

    rows: List[dict] = []
    for (pkey, tkey, ptype), pr in best.items():
        market = pr.get("market") or {}
        odds = None
        if isinstance(market, dict):
            odds = market.get("over_odds") if market.get("type") == "over_under" else market.get("odds")

        if ptype in {"points", "assists", "goals", "anytime_goal"}:
            line_value = 0.5
        else:
            line_value = _safe_float(pr.get("line_value"))

        rows.append(
            {
                "__player_key": pkey,
                "__team_key": tkey,
                "prop_type": ptype,
                "line_value": _safe_float(line_value),
                "odds": _safe_float(odds),
                "vendor": str(pr.get("vendor") or ""),
            }
        )

    odds_df = pd.DataFrame(rows)
    if odds_df.empty:
        df.drop(columns=["__player_key", "__team_key"], inplace=True, errors="ignore")
        return df

    def _pivot(ptype: str, out_prefix: str):
        sub = odds_df[odds_df["prop_type"] == ptype].copy()
        if sub.empty:
            return
        sub = sub.drop_duplicates(subset=["__player_key", "__team_key"], keep="first")
        m = sub.set_index(["__player_key", "__team_key"])
        idx = pd.MultiIndex.from_frame(df[["__player_key", "__team_key"]])
        df[f"BDL_{out_prefix}_Line"] = [m["line_value"].get(k, None) for k in idx]
        df[f"BDL_{out_prefix}_Odds"] = [m["odds"].get(k, None) for k in idx]
        df[f"BDL_{out_prefix}_Book"] = [m["vendor"].get(k, "") for k in idx]

    _pivot("shots_on_goal", "SOG")
    _pivot("points", "Points")
    _pivot("assists", "Assists")
    _pivot("goals", "Goal")
    _pivot("anytime_goal", "ATG")

    df.drop(columns=["__player_key", "__team_key"], inplace=True, errors="ignore")
    return df


def merge_bdl_milestones(
    tracker: pd.DataFrame,
    game_date: str,
    vendors: Optional[Tuple[str, ...]] = ("fanduel", "draftkings", "caesars"),
    debug: bool = False,
) -> pd.DataFrame:
    vlist = list(vendors) if vendors else None
    return merge_bdl_props_mainlines(tracker, game_date, api_key=None, vendors=vlist, debug=debug)


# -----------------------------
# Compute EV / model prob
# -----------------------------

def add_bdl_ev_all(tracker: pd.DataFrame) -> pd.DataFrame:
    """Add per-market Line/Odds/Book + p_model/p_imp/EVpct columns, and stamp engine version."""
    df = tracker.copy()
    if df.empty:
        return df

    df["BDL_EV_ENGINE"] = __BDL_EV_ENGINE__

    # per-market configuration:
    # exp candidates: Exp_*_10 first, then some common alternates
    # fallback rate candidates: (col, scale_to_lambda)
    cfgs = {
        "SOG": {
            "bdl_line": "BDL_SOG_Line", "bdl_odds": "BDL_SOG_Odds", "bdl_book": "BDL_SOG_Book",
            "line": "SOG_Line", "odds": "SOG_Odds_Over", "book": "SOG_Book",
            "exp": ["Exp_S_10", "Exp_SOG_10", "Exp_SOG10"],
            "fallback": [("S10_total", 1/10), ("L10_S", 1/10)],
            "baseline_min": 2.0,
        },
        "Points": {
            "bdl_line": "BDL_Points_Line", "bdl_odds": "BDL_Points_Odds", "bdl_book": "BDL_Points_Book",
            "line": "Points_Line", "odds": "Points_Odds_Over", "book": "Points_Book",
            "exp": ["Exp_P_10", "Exp_Points_10", "Exp_P10"],
            "fallback": [("L10_P", 1/10), ("P10_total", 1/10), ("PPG", 1.0)],
            "baseline_min": None,  # milestone baseline 1+
        },
        "Assists": {
            "bdl_line": "BDL_Assists_Line", "bdl_odds": "BDL_Assists_Odds", "bdl_book": "BDL_Assists_Book",
            "line": "Assists_Line", "odds": "Assists_Odds_Over", "book": "Assists_Book",
            "exp": ["Exp_A_10", "Exp_Assists_10", "Exp_A10"],
            "fallback": [("L10_A", 1/10), ("A10_total", 1/10), ("APG", 1.0)],
            "baseline_min": None,
        },
        "Goal": {
            "bdl_line": "BDL_Goal_Line", "bdl_odds": "BDL_Goal_Odds", "bdl_book": "BDL_Goal_Book",
            "line": "Goal_Line", "odds": "Goal_Odds_Over", "book": "Goal_Book",
            "exp": ["Exp_G_10", "Exp_Goals_10", "Exp_G10"],
            "fallback": [("L10_G", 1/10), ("G10_total", 1/10), ("GPG", 1.0)],
            "baseline_min": None,
        },
        "ATG": {
            "bdl_line": "BDL_ATG_Line", "bdl_odds": "BDL_ATG_Odds", "bdl_book": "BDL_ATG_Book",
            "line": "ATG_Line", "odds": "ATG_Odds_Over", "book": "ATG_Book",
            "exp": ["Exp_G_10", "Exp_Goals_10", "Exp_G10"],
            "fallback": [("L10_G", 1/10), ("G10_total", 1/10), ("GPG", 1.0)],
            "baseline_min": None,
        },
    }

    for out_prefix, cfg in cfgs.items():
        # accept either BDL_* inputs (preferred) or already-surfaced inputs
        line_in = cfg["bdl_line"] if cfg["bdl_line"] in df.columns else cfg["line"]
        odds_in = cfg["bdl_odds"] if cfg["bdl_odds"] in df.columns else cfg["odds"]
        book_in = cfg["bdl_book"] if cfg["bdl_book"] in df.columns else cfg["book"]

        # We want model probability even when odds are missing.
        if line_in not in df.columns:
            # create a baseline line so every player gets a model%
            if out_prefix in {"Points","Assists","Goal","ATG"}:
                df[f"{out_prefix}_Line"] = 0.5
            elif out_prefix == "SOG":
                df[f"{out_prefix}_Line"] = 2.5
            else:
                continue

        if f"{out_prefix}_Line" not in df.columns:
            df[f"{out_prefix}_Line"] = pd.to_numeric(df[line_in], errors="coerce")
        # odds are optional; keep NaN if not available
        if odds_in in df.columns:
            df[f"{out_prefix}_Odds_Over"] = pd.to_numeric(df[odds_in], errors="coerce")
        # If odds exist but line is missing, force baseline lines so Model% and EV can compute.
        if out_prefix in {"Points","Assists","Goal","ATG"}:
            m = df[f"{out_prefix}_Odds_Over"].notna() & df[f"{out_prefix}_Line"].isna()
            df.loc[m, f"{out_prefix}_Line"] = 0.5
        elif out_prefix == "SOG":
            m = df[f"{out_prefix}_Odds_Over"].notna() & df[f"{out_prefix}_Line"].isna()
            df.loc[m, f"{out_prefix}_Line"] = 2.5

        else:
            df[f"{out_prefix}_Odds_Over"] = pd.NA
        # If odds exist but line is missing, force baseline lines so Model% and EV can compute.
        if out_prefix in {"Points","Assists","Goal","ATG"}:
            m = df[f"{out_prefix}_Odds_Over"].notna() & df[f"{out_prefix}_Line"].isna()
            df.loc[m, f"{out_prefix}_Line"] = 0.5
        elif out_prefix == "SOG":
            m = df[f"{out_prefix}_Odds_Over"].notna() & df[f"{out_prefix}_Line"].isna()
            df.loc[m, f"{out_prefix}_Line"] = 2.5

        if book_in in df.columns:
            df[f"{out_prefix}_Book"] = df[book_in]
        else:
            df[f"{out_prefix}_Book"] = ""

        # baseline enforcement
        if out_prefix in {"Points", "Assists", "Goal", "ATG"}:
            df.loc[df[f"{out_prefix}_Line"].notna(), f"{out_prefix}_Line"] = 0.5
        elif out_prefix == "SOG":
            df.loc[df[f"{out_prefix}_Line"] < 2.0, f"{out_prefix}_Line"] = pd.NA

        # compute p_model per-row (robust to missing Exp cols)
        p_model = []
        p_model_src = []
        for _, row in df.iterrows():
            lv = _safe_float(row.get(f"{out_prefix}_Line"))
            if lv is None:
                p_model.append(float('nan'))
                p_model_src.append('no_line')
                continue
            # extra enforcement for SOG
            if out_prefix == 'SOG' and lv < 2.0:
                p_model.append(float('nan'))
                p_model_src.append('bad_line')
                continue

            lam = _lambda_from_exp10_row(row, cfg['exp'], cfg['fallback'])
            src_tag = 'exp/fallback'
            # If lambda is missing, use conservative league-average fallback so Model% is always populated.
            if lam <= 0:
                if out_prefix == 'SOG':
                    lam = 2.3
                elif out_prefix == 'Points':
                    lam = 0.60
                elif out_prefix == 'Assists':
                    lam = 0.35
                elif out_prefix in ('Goal','ATG'):
                    lam = 0.30
                src_tag = 'league_avg'
            k = _milestone_k(float(lv))
            p_model.append(_poisson_tail_ge_k(lam, k))
            p_model_src.append(src_tag)

        df[f"{out_prefix}_ModelSrc"] = pd.Series(p_model_src, index=df.index).astype("string")
        df[f"{out_prefix}_p_model_over"] = pd.to_numeric(pd.Series(p_model, index=df.index), errors="coerce")
        df[f"{out_prefix}_p_imp_over"] = pd.to_numeric(df[f"{out_prefix}_Odds_Over"].apply(american_to_implied_prob), errors="coerce")

        evs = []
        for pm, od in zip(df[f"{out_prefix}_p_model_over"].tolist(), df[f"{out_prefix}_Odds_Over"].tolist()):
            if pm is None or pd.isna(pm) or od is None or pd.isna(od):
                evs.append(float("nan"))
                continue
            evs.append(_compute_ev_pct(float(pm), float(od)))
        df[f"{out_prefix}_EVpct_over"] = pd.to_numeric(pd.Series(evs, index=df.index), errors="coerce")

    return df
