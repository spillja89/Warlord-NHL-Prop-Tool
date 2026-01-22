from __future__ import annotations

"""
odds_ev_bdl.py — robust BDL props merge + implied% + optional EV%

This version is designed to survive BallDontLie NHL schema drift.
- Detects markets via prop_type / market fields + heuristics
- Uses line_value (preferred) and multiple fallbacks
- Normalizes vendor/book strings
- Writes diagnostic columns when nothing matches (so you can see WHY)
"""

import os
import re
import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


# -----------------------------
# Helpers
# -----------------------------
def implied_prob_from_american(odds: Any) -> Optional[float]:
    try:
        if odds is None or (isinstance(odds, float) and math.isnan(odds)):
            return None
        o = float(odds)
        if o == 0:
            return None
        if o > 0:
            return 100.0 / (o + 100.0)
        return abs(o) / (abs(o) + 100.0)
    except Exception:
        return None


def _deep_get(d: Any, keys: List[str]) -> Any:
    """Try several keys at top-level and nested under 'market'/'book' shapes."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and d.get(k) is not None:
            return d.get(k)
    # common nesting
    for container_key in ("market", "book", "sportsbook", "vendor"):
        c = d.get(container_key)
        if isinstance(c, dict):
            for k in keys:
                if k in c and c.get(k) is not None:
                    return c.get(k)
    return None


def _norm_vendor(v: Any) -> str:
    s = str(v or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _canon_market(prop: Dict[str, Any]) -> Optional[str]:
    pt = str(prop.get("prop_type") or prop.get("type") or "").strip().lower()
    mk = prop.get("market") or {}
    mk_name = str(mk.get("name") or mk.get("key") or mk.get("market") or "").strip().lower()

    def _norm(s: str) -> str:
        s = (s or "").lower().strip()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    cand = _norm(pt or mk_name)

    aliases = {
        "shots_on_goal": "SOG",
        "shot_on_goal": "SOG",
        "player_shots_on_goal": "SOG",
        "player_shot_on_goal": "SOG",
        "sog": "SOG",
        "shots": "SOG",
        "points": "Points",
        "player_points": "Points",
        "skater_points": "Points",
        "assists": "Assists",
        "assist": "Assists",
        "player_assists": "Assists",
        "skater_assists": "Assists",
        "goals": "Goal",
        "goal": "Goal",
        "player_goals": "Goal",
        "skater_goals": "Goal",
        "anytime_goal_scorer": "Goal",
        "atg": "Goal",
    }
    if cand in aliases:
        return aliases[cand]

    s = f"{pt} {mk_name}".lower()
    if ("shot" in s and "goal" in s) or "sog" in s:
        return "SOG"
    if "assist" in s:
        return "Assists"
    if "point" in s:
        return "Points"
    if "goal" in s or "anytime" in s or "scorer" in s:
        return "Goal"
    return None


def _extract_line(prop: Dict[str, Any]) -> Optional[float]:
    v = _deep_get(prop, ["line_value", "line", "value", "handicap"])
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _extract_odds(prop: Dict[str, Any]) -> Optional[float]:
    v = _deep_get(prop, ["odds", "price", "american_odds", "americanOdds"])
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _extract_vendor(prop: Dict[str, Any]) -> str:
    v = _deep_get(prop, ["vendor", "book", "sportsbook", "provider", "name", "key"])
    return str(v or "").strip()


def _extract_player_name(prop: Dict[str, Any]) -> str:
    # common shapes: player_name, athlete.name, player.full_name
    for key in ("player_name", "player", "athlete", "competitor"):
        v = prop.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict):
            for k2 in ("full_name", "name", "display_name"):
                if isinstance(v.get(k2), str) and v.get(k2).strip():
                    return v.get(k2).strip()
    # fallback
    v = prop.get("name")
    return str(v or "").strip()


_PREF_MAINLINES = {
    "SOG": [2.5, 3.5, 4.5],
    "Points": [0.5, 1.5],
    "Assists": [0.5, 1.5],
    "Goal": [0.5, 1.5],
    "ATG": [0.5],
}


def _pick_mainline(market: str, avail: List[float]) -> Optional[float]:
    uniq = sorted({float(x) for x in avail if x is not None})
    for lv in _PREF_MAINLINES.get(market, []):
        if lv in uniq:
            return float(lv)
    return float(uniq[0]) if uniq else None


def _fetch_games_for_date(game_date: str, api_key: Optional[str]) -> List[Dict[str, Any]]:
    url = "https://api.balldontlie.io/nhl/v1/games"
    params = {"dates[]": game_date, "per_page": 100}
    headers = {"Authorization": api_key} if api_key else {}
    r = requests.get(url, params=params, headers=headers, timeout=25)
    r.raise_for_status()
    return r.json().get("data", []) or []


def _fetch_props_for_game(game_id: int, api_key: Optional[str]) -> List[Dict[str, Any]]:
    url = "https://api.balldontlie.io/nhl/v1/odds/player_props"
    params = {"game_id": game_id, "per_page": 100}
    headers = {"Authorization": api_key} if api_key else {}
    r = requests.get(url, params=params, headers=headers, timeout=25)
    r.raise_for_status()
    return r.json().get("data", []) or []


# -----------------------------
# Main merge
# -----------------------------
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

    # Diagnostics columns
    for c in ("BDL_Status", "BDL_Error", "BDL_Props_Count", "BDL_Kept_Count", "BDL_Vendors_Seen", "BDL_PropTypes_Sample", "BDL_MarketNames_Sample"):
        if c not in df.columns:
            df[c] = "" if c in ("BDL_Status","BDL_Error","BDL_Vendors_Seen","BDL_PropTypes_Sample","BDL_MarketNames_Sample") else 0

    # Initialize output columns so downstream never breaks
    markets = ["SOG", "Points", "Assists", "Goal"]
    for m in markets:
        for suffix in ["", *[f"_{k}" for k in range(1, top_k + 1)]]:
            for col in (f"BDL_{m}_Line{suffix}", f"BDL_{m}_Odds{suffix}", f"BDL_{m}_Book{suffix}"):
                if col not in df.columns:
                    df[col] = pd.NA

    # Fetch games
    try:
        games = _fetch_games_for_date(game_date, api_key)
    except Exception as e:
        df["BDL_Status"] = "NO_GAMES"
        df["BDL_Error"] = f"games_fetch_failed: {type(e).__name__}: {e}"
        return df

    # Fetch props
    props: List[Dict[str, Any]] = []
    for g in games:
        gid = g.get("id")
        if gid is None:
            continue
        try:
            props.extend(_fetch_props_for_game(int(gid), api_key))
        except Exception as e:
            # keep going; partial data ok
            if debug:
                print("[bdl] props fetch failed for game", gid, e)

    df["BDL_Props_Count"] = int(len(props))
    if not props:
        df["BDL_Status"] = "NO_PROPS"
        df["BDL_Error"] = "no_props_returned"
        return df

    vendors_set = {_norm_vendor(v) for v in (vendors or []) if str(v).strip()}

    vendors_seen, prop_types_seen, market_names_seen = set(), set(), set()
    kept = 0

    # Collect lines per player/market and best odds per line
    lines_map: Dict[Tuple[str, str], List[float]] = {}
    best: Dict[Tuple[str, str, float], Tuple[float, str]] = {}

    for p in props:
        # samples
        try:
            prop_types_seen.add(str(p.get("prop_type") or p.get("type") or "").strip().lower())
        except Exception:
            pass
        try:
            mkx = (p.get("market") or {})
            market_names_seen.add(str(mkx.get("name") or mkx.get("key") or mkx.get("market") or "").strip().lower())
        except Exception:
            pass

        mkt = _canon_market(p)
        if not mkt:
            continue

        player = _extract_player_name(p)
        if not player:
            continue

        line = _extract_line(p)
        if line is None:
            continue

        # Enforce .5 grid to avoid milestone markets (2+, 3+, etc.)
        if abs(line * 2 - round(line * 2)) > 1e-6:
            continue

        odds = _extract_odds(p)
        if odds is None:
            continue

        vendor_raw = _extract_vendor(p)
        vendor = _norm_vendor(vendor_raw)
        if vendor:
            vendors_seen.add(vendor)

        # vendor filtering (if passed)
        if vendors_set and (vendor not in vendors_set) and (not any(tok in vendor for tok in vendors_set)):
            continue

        kept += 1
        lines_map.setdefault((player, mkt), []).append(float(line))
        key = (player, mkt, float(line))
        # keep best odds (highest payout) for over
        if key not in best or float(odds) > float(best[key][0]):
            best[key] = (float(odds), vendor_raw)

    df["BDL_Kept_Count"] = int(kept)
    df["BDL_Vendors_Seen"] = ",".join(sorted(list(vendors_seen))[:20])
    df["BDL_PropTypes_Sample"] = ",".join([x for x in sorted(list(prop_types_seen)) if x][:10])
    df["BDL_MarketNames_Sample"] = ",".join([x for x in sorted(list(market_names_seen)) if x][:10])

    if not best:
        df["BDL_Status"] = "NO_MATCHING_PROPS"
        df["BDL_Error"] = "filtered_out (market/vend/name). Check BDL_*_Sample cols."
        return df

    # Apply to dataframe rows
    def _row_key(row) -> str:
        return str(row.get("Player") or "").strip()

    # Build per-row assignments
    for i, row in df.iterrows():
        player = _row_key(row)
        if not player:
            continue

        for mkt in markets:
            avail = lines_map.get((player, mkt), [])
            if not avail:
                continue
            main = _pick_mainline(mkt, avail)
            if main is None:
                continue

            # mainline
            o, b = best.get((player, mkt, float(main)), (pd.NA, ""))
            df.at[i, f"BDL_{mkt}_Line"] = float(main)
            df.at[i, f"BDL_{mkt}_Odds"] = o
            df.at[i, f"BDL_{mkt}_Book"] = b

            # alt ladder: closest to main, increasing line
            uniq = sorted({float(x) for x in avail})
            ladder = [x for x in uniq if x >= float(main)]
            ladder = ladder[:top_k]
            for k, lv in enumerate(ladder, start=1):
                o2, b2 = best.get((player, mkt, float(lv)), (pd.NA, ""))
                df.at[i, f"BDL_{mkt}_Line_{k}"] = float(lv)
                df.at[i, f"BDL_{mkt}_Odds_{k}"] = o2
                df.at[i, f"BDL_{mkt}_Book_{k}"] = b2

    df["BDL_Status"] = "OK"
    df["BDL_Error"] = ""
    return df


def add_bdl_ev_all(df: pd.DataFrame, top_k: int = 4) -> pd.DataFrame:
    """
    Surface BDL mainline fields into standard columns the app expects.
    Also computes implied% and optional EV% if *_p_model_over exists.
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

        # Optional EV% if model probability column exists
        pcol = f"{pref}_p_model_over"
        if pcol in df.columns:
            def _ev(model_p, odds):
                try:
                    if model_p is None or pd.isna(model_p) or odds is None or pd.isna(odds):
                        return pd.NA
                    p = float(model_p)
                    o = float(odds)
                    dec = 1.0 + (o / 100.0) if o > 0 else 1.0 + (100.0 / abs(o))
                    b = dec - 1.0
                    q = 1.0 - p
                    return round(((p * b) - q) * 100.0, 1)
                except Exception:
                    return pd.NA

            df[f"{pref}_Model%"] = pd.to_numeric(df[pcol], errors="coerce").map(lambda x: round(float(x) * 100.0, 1) if pd.notna(x) else pd.NA)
            df[f"{pref}_EV%"] = [_ev(mp, od) for mp, od in zip(pd.to_numeric(df[pcol], errors="coerce"), pd.to_numeric(df[f"{pref}_Odds_Over"], errors="coerce"))]

    return df
