
from __future__ import annotations

"""
odds_ev_bdl.py — Warlord NHL Prop Tool (STABLE COLUMN POPULATOR)
---------------------------------------------------------------

Goal:
  - Populate Shots (SOG), Points, Assists columns with sportsbook-realistic .5 mainlines.
  - Never promote 5.0 SOG to mainline if 2.5/3.5/4.5 exist.
  - Keep Top-K alt lines for cash checker.

Exports used by nhl_edge.py:
  - merge_bdl_props_altlines(df, game_date, api_key=None, vendors=None, top_k=4, debug=False) -> pd.DataFrame
  - add_bdl_ev_all(df, top_k=4) -> pd.DataFrame
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


import re as _re_vendor

def _norm_vendor(v: str) -> str:
    """Normalize vendor/book strings to stable tokens (handles dk/DraftKings/DRAFT_KINGS/etc)."""
    s = str(v or "").strip().lower()
    s = _re_vendor.sub(r"[^a-z0-9]+", "", s)
    return s

_MARKET_CANON = {
    "shots_on_goal": "SOG",
    "sog": "SOG",
    "points": "Points",
    "player_points": "Points",
    "assists": "Assists",
    "player_assists": "Assists",
    "goals": "Goal",
    "player_goals": "Goal",
    "anytime_goal_scorer": "ATG",
}

# sportsbook mainline preferences (Jason verified)
_PREF_MAINLINES = {
    "SOG": [2.5, 3.5, 4.5],      # never 5.0 mainline if these exist
    "Points": [0.5, 1.5],        # keep crisp
    "Assists": [0.5, 1.5],       # keep crisp
    "Goal": [0.5, 1.5],
    "ATG": [0.5],
}

def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None

def implied_prob_from_american(odds: Any) -> Optional[float]:
    o = _to_float(odds)
    if o is None or o == 0:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)

def _bdl_session(api_key: Optional[str]) -> requests.Session:
    s = requests.Session()
    k = (api_key or "").strip()
    if k:
        s.headers.update({"Authorization": k})
    return s

def _fetch_games_for_date(game_date: str, api_key: Optional[str], timeout: int = 25):
    url = "https://api.balldontlie.io/nhl/v1/games"
    s = _bdl_session(api_key)
    r = s.get(url, params={"dates[]": game_date, "per_page": 100}, timeout=timeout)
    r.raise_for_status()
    return (r.json() or {}).get("data", []) or []

def _fetch_props_for_game(game_id: int, api_key: Optional[str], timeout: int = 25):
    url = "https://api.balldontlie.io/nhl/v1/odds/player_props"
    s = _bdl_session(api_key)
    out = []
    page = 1
    while True:
        r = s.get(url, params={"game_id": game_id, "per_page": 100, "page": page}, timeout=timeout)
        r.raise_for_status()
        j = r.json() or {}
        data = j.get("data", []) or []
        out.extend(data)
        meta = j.get("meta") or {}
        nxt = meta.get("next_page")
        if not nxt:
            break
        page = int(nxt)
    return out

def _canon_market(prop: Dict[str, Any]) -> Optional[str]:
    pt = (prop.get("prop_type") or "").strip().lower()
    mk = prop.get("market") or {}
    mk_name = (mk.get("name") or mk.get("key") or "").strip().lower()
    cand = pt or mk_name
    return _MARKET_CANON.get(cand)

def _extract_line(prop: Dict[str, Any]) -> Optional[float]:
    # BALLDONTLIE NHL player props use `line_value` for the line.
    # (Older drafts used `line` in some sports.)
    line = prop.get("line_value")
    if line is None:
        line = prop.get("line")
    if line is None:
        line = (prop.get("market") or {}).get("line")
    return _to_float(line)

def _extract_odds(prop: Dict[str, Any]) -> Optional[float]:
    o = prop.get("odds")
    if o is None:
        o = prop.get("price")
    if o is None:
        o = (prop.get("market") or {}).get("odds")
    return _to_float(o)

def _extract_vendor(prop: Dict[str, Any]) -> str:
    v = prop.get("vendor")
    if isinstance(v, dict):
        v = v.get("name") or v.get("key")
    if v is None:
        v = prop.get("book") or prop.get("sportsbook") or ""
    return str(v).strip().lower()

def _extract_player_name(prop: Dict[str, Any]) -> str:
    p = prop.get("player") or {}
    name = p.get("full_name") or p.get("name") or prop.get("player_name") or ""
    return str(name).strip()

def _norm_name(s: str) -> str:
    return " ".join(str(s or "").strip().lower().split())

def _pick_mainline(market: str, avail: List[float]) -> Optional[float]:
    if not avail:
        return None
    uniq = sorted({float(x) for x in avail})
    for lv in _PREF_MAINLINES.get(market, []):
        if lv in uniq:
            return float(lv)
    # fallback: choose lowest half-step, else lowest
    halfs = [x for x in uniq if abs(x * 2 - round(x * 2)) < 1e-9]
    return float(min(halfs)) if halfs else float(min(uniq))

def _pick_topk(avail: List[float], top_k: int) -> List[float]:
    if not avail or top_k <= 0:
        return []
    uniq = sorted({float(x) for x in avail})
    return uniq[:top_k]

def merge_bdl_props_altlines(
    df: pd.DataFrame,
    game_date: str,
    api_key: Optional[str] = None,
    vendors: Optional[List[str]] = None,
    top_k: int = 4,
    debug: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    api_key = api_key if api_key is not None else (os.getenv("BALLDONTLIE_API_KEY") or os.getenv("BDL_API_KEY") or None)
    vendors_set = {_norm_vendor(v) for v in (vendors or []) if str(v).strip()}

    # Fetch
    try:
        games = _fetch_games_for_date(game_date, api_key)
    except Exception as e:
        if debug:
            print("[bdl] games fetch failed:", e)
        return df

    props = []
    for g in games:
        gid = g.get("id")
        if gid is None:
            continue
        try:
            props.extend(_fetch_props_for_game(int(gid), api_key))
        except Exception as e:
            if debug:
                print(f"[bdl] props fetch failed game_id={gid}:", e)

    # Diagnostics counters
    df["BDL_Games"] = len(games) if isinstance(games, list) else 0
    df["BDL_Props"] = len(props)
    df["BDL_Kept"] = 0
    df["BDL_Vendors_Seen"] = ""
    df["BDL_Markets_Seen"] = ""

    # Initialize columns so downstream never breaks
    markets = ["SOG", "Points", "Assists", "Goal"]
    for m in markets:
        df[f"BDL_{m}_Line"] = pd.NA
        df[f"BDL_{m}_Odds"] = pd.NA
        df[f"BDL_{m}_Book"] = ""
        for k in range(1, top_k + 1):
            df[f"BDL_{m}_Line_{k}"] = pd.NA
            df[f"BDL_{m}_Odds_{k}"] = pd.NA
            df[f"BDL_{m}_Book_{k}"] = ""


    # Diagnostics (so CSV shows exactly why odds are missing)
    if "BDL_Status" not in df.columns:
        df["BDL_Status"] = ""
    if "BDL_Error" not in df.columns:
        df["BDL_Error"] = ""
    if not props:
        df["BDL_Status"] = "NO_PROPS"
        return df

    # Build best odds per (player, market, line)
    best: Dict[Tuple[str, str, float], Tuple[float, str]] = {}
    lines_map: Dict[Tuple[str, str], List[float]] = {}

    vendors_seen = set()
    markets_seen = set()
    kept = 0
 Dict[Tuple[str, str], List[float]] = {}

    for p in props:
        # Only use traditional O/U markets for baseline + alts.
        mk = p.get("market") or {}
        mkt = _canon_market(p)
        if mkt not in _PREF_MAINLINES:
            continue

        line = _extract_line(p)
        odds = _extract_odds(p)
        if line is None or odds is None:
            continue

        # enforce 0.5 grid only (Jason request)
        if abs(line * 2 - round(line * 2)) > 1e-6:
            continue

        vendor_raw = _extract_vendor(p)
        vendor = _norm_vendor(vendor_raw)
        if vendor:
            vendors_seen.add(vendor)
        if vendors_set and (vendor not in vendors_set) and (not any(tok in vendor for tok in vendors_set)):
            continue

        player = _norm_name(_extract_player_name(p))
        if not player:
            continue

        kept += 1
        key = (player, mkt, float(line))
        cur = best.get(key)
        if cur is None or float(odds) > float(cur[0]):
            best[key] = (float(odds), vendor)

    if not best:
        df["BDL_Status"] = "NO_MATCHING_PROPS"
        df["BDL_Kept"] = kept
        df["BDL_Vendors_Seen"] = ",".join(sorted(vendors_seen))
        df["BDL_Markets_Seen"] = ",".join(sorted(markets_seen))
        df["BDL_Error"] = "filtered_out: vendor/line_grid/name_match"
        return df

    for (pn, mkt, line) in best.keys():
        lines_map.setdefault((pn, mkt), []).append(float(line))

    # Ensure Player_norm
    if "Player_norm" not in df.columns:
        if "Player" in df.columns:
            df["Player_norm"] = df["Player"].astype(str).map(_norm_name)
        elif "Name" in df.columns:
            df["Player_norm"] = df["Name"].astype(str).map(_norm_name)
        else:
            df["Player_norm"] = df.iloc[:, 0].astype(str).map(_norm_name)

    # Populate rows
    for i, row in df.iterrows():
        pn = _norm_name(row.get("Player_norm") or "")
        if not pn:
            continue
        for m in markets:
            avail = sorted({float(x) for x in lines_map.get((pn, m), [])})
            if not avail:
                continue

            # alt lines (lowest K)
            top_lines = _pick_topk(avail, top_k)
            for k, lv in enumerate(top_lines, start=1):
                o, v = best.get((pn, m, float(lv)), (None, ""))
                df.at[i, f"BDL_{m}_Line_{k}"] = float(lv)
                df.at[i, f"BDL_{m}_Odds_{k}"] = o
                df.at[i, f"BDL_{m}_Book_{k}"] = v

            # mainline by sportsbook convention
            main_lv = _pick_mainline(m, avail)
            if main_lv is None:
                continue
            o, v = best.get((pn, m, float(main_lv)), (None, ""))
            df.at[i, f"BDL_{m}_Line"] = float(main_lv)
            df.at[i, f"BDL_{m}_Odds"] = o
            df.at[i, f"BDL_{m}_Book"] = v

    if debug:
        print("[bdl] merged best keys:", len(best))

    return df

def add_bdl_ev_all(df: pd.DataFrame, top_k: int = 4) -> pd.DataFrame:
    """
    Surface BDL mainline fields into standard columns the app expects.
    Focused on LINE/ODDS/BOOK/IMPLIED% being correct and present.
    """
    if df is None or df.empty:
        return df

    def ensure(prefix: str):
        for c in (f"{prefix}_Line", f"{prefix}_Odds_Over", f"{prefix}_Book", f"{prefix}_Imp%_Over"):
            if c not in df.columns:
                df[c] = pd.NA

    map_m = {"SOG": "SOG", "Points": "Points", "Assists": "Assists", "Goal": "Goal"}

    for mkt, pref in map_m.items():
        ensure(pref)
        df[f"{pref}_Line"] = df.get(f"BDL_{mkt}_Line", pd.NA)
        df[f"{pref}_Odds_Over"] = df.get(f"BDL_{mkt}_Odds", pd.NA)
        df[f"{pref}_Book"] = df.get(f"BDL_{mkt}_Book", "")
        imp = df[f"{pref}_Odds_Over"].map(implied_prob_from_american)
        df[f"{pref}_Imp%_Over"] = imp.map(lambda x: round(float(x) * 100.0, 1) if x is not None else pd.NA)

    return df
