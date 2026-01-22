
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

    "player_shots_on_goal": "SOG",
    "shots": "SOG",
    "player_shots": "SOG",
    "shots_on_goal_alternate": "SOG",
    "player_points_alternate": "Points",
    "player_assists_alternate": "Assists",
    "player_goals_alternate": "Goal",
    "goal": "Goal",
    "points_alternate": "Points",
    "assists_alternate": "Assists",
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
    """Return canonical market label (SOG/Points/Assists/Goal/ATG) or None.

    BDL has changed field names a few times. We try:
      1) prop_type
      2) market.name / market.key
      3) heuristic keyword matching as a fallback
    """
    pt = (prop.get("prop_type") or prop.get("propType") or "").strip().lower()
    mk = prop.get("market") or {}
    if isinstance(mk, str):
        mk = {"name": mk}
    mk_name = (mk.get("name") or mk.get("key") or mk.get("type") or "").strip().lower()

    cand = pt or mk_name
    if cand:
        hit = _MARKET_CANON.get(cand)
        if hit:
            return hit

    # Heuristic fallback for new/unknown keys
    blob = " ".join([str(cand or ""), str(prop.get("stat") or ""), str(prop.get("name") or ""), str(prop.get("type") or "")]).lower()
    # common variants
    if ("shot" in blob and "goal" in blob) or "sog" in blob or "shots_on_goal" in blob:
        return "SOG"
    if "assist" in blob:
        return "Assists"
    if "point" in blob:
        return "Points"
    if "anytime" in blob or "atg" in blob or "goal scorer" in blob:
        return "ATG"
    # Be careful: "goals against" etc shouldn't match; this is player props, so goal is OK.
    if "goal" in blob:
        return "Goal"
    return None



def _deep_find_number(obj: Any, keys: List[str]) -> Optional[float]:
    """Best-effort recursive numeric lookup for any of keys."""
    try:
        if obj is None:
            return None
        if isinstance(obj, (int, float)) and not (isinstance(obj, float) and math.isnan(obj)):
            return float(obj)
        if isinstance(obj, str):
            return _to_float(obj)
        if isinstance(obj, dict):
            for k in keys:
                if k in obj:
                    v = _deep_find_number(obj.get(k), keys)
                    if v is not None:
                        return v
            # search nested
            for v in obj.values():
                hit = _deep_find_number(v, keys)
                if hit is not None:
                    return hit
        if isinstance(obj, list):
            for it in obj:
                hit = _deep_find_number(it, keys)
                if hit is not None:
                    return hit
    except Exception:
        return None
    return None

def _extract_line(prop: Dict[str, Any]) -> Optional[float]:
    # Common keys observed across BDL schema variants
    line = (
        prop.get("line_value")
        if prop.get("line_value") is not None
        else prop.get("line")
    )
    if line is None:
        mk = prop.get("market") or {}
        if isinstance(mk, dict):
            line = mk.get("line") or mk.get("line_value")
    if line is None:
        # Deep search (covers nested/outcomes formats)
        line = _deep_find_number(prop, ["line_value", "line", "total", "points"])
    return _to_float(line)


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

        vendor = _extract_vendor(p)
        if vendors_set and vendor not in vendors_set:
            continue

        player = _norm_name(_extract_player_name(p))
        if not player:
            continue

        key = (player, mkt, float(line))
        cur = best.get(key)
        if cur is None or float(odds) > float(cur[0]):
            best[key] = (float(odds), vendor)


    # If we filtered by vendor list and ended up with nothing, fall back to ALL vendors.
    if not best and vendors_set:
        if debug:
            print("[bdl] vendor-filter produced no matches; retrying with all vendors")
        vendors_set = set()
        for p in props:
            mkt = _canon_market(p)
            if mkt not in _PREF_MAINLINES:
                continue
            line = _extract_line(p)
            odds = _extract_odds(p)
            if line is None or odds is None:
                continue
            if abs(line * 2 - round(line * 2)) > 1e-6:
                continue
            vendor = _extract_vendor(p)
            player = _norm_name(_extract_player_name(p))
            if not player:
                continue
            key = (player, mkt, float(line))
            cur = best.get(key)
            if cur is None or float(odds) > float(cur[0]):
                best[key] = (float(odds), vendor)

    if not best:
        df["BDL_Status"] = "NO_MATCHING_PROPS"
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
