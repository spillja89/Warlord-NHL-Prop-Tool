from __future__ import annotations

"""Compute EV% for BallDontLie milestone (X+) props using merged BDL_* columns.

Inputs expected on tracker (some subset):
  - BDL_SOG_Line, BDL_SOG_Odds, BDL_SOG_Book
  - BDL_Points_Line, BDL_Points_Odds, BDL_Points_Book
  - BDL_Goal_Line, BDL_Goal_Odds, BDL_Goal_Book
  - BDL_Assists_Line, BDL_Assists_Odds, BDL_Assists_Book
  - BDL_PPP_Line, BDL_PPP_Odds, BDL_PPP_Book
  - BDL_Saves_Line, BDL_Saves_Odds, BDL_Saves_Book

We generate columns per market:
  <MKT>_Line, <MKT>_Book, <MKT>_Odds_Over
  <MKT>_p_model_over, <MKT>_p_imp_over, <MKT>_EVpct_over

Where <MKT> is: SOG, Points, Goal, Assists, PPP, Saves

Probability model:
  - Uses Conf_<MKT> (or closest) as base probability proxy
  - Nudges based on Matrix_<MKT> and Earned flags if present
  - Applies a simple decay for higher milestone lines (2+, 3+, etc.)

This is a pragmatic heuristic meant for highlighting strong overlays, not a full distribution model.
"""

import os
import math
import requests
import certifi
import pandas as pd
from typing import Dict, Optional, Tuple, List


# ============================
# BallDontLie live pull + merge
# ============================

BDL_NHL_BASE = "https://api.balldontlie.io/nhl/v1"


def _bdl_headers(api_key: str | None) -> Dict[str, str]:
    k = (api_key or os.getenv("BDL_API_KEY") or os.getenv("BALLDONTLIE_KEY") or os.getenv("BALLDONTLIE_API_KEY") or "").strip()
    return {"Authorization": k} if k else {}


def _bdl_get(url: str, params: dict | list | None, api_key: str | None, timeout: int = 25) -> dict:
    r = requests.get(
        url,
        params=params,
        headers=_bdl_headers(api_key),
        timeout=timeout,
        verify=certifi.where(),
    )
    r.raise_for_status()
    return r.json()


def fetch_bdl_games_for_date(game_date: str, api_key: str | None = None, per_page: int = 100) -> List[dict]:
    """Return list of BDL games for the given YYYY-MM-DD date."""
    url = f"{BDL_NHL_BASE}/games"
    # docs show dates[]; the requests library will encode this correctly
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
    """Map BDL player_id -> player object (includes full_name + teams)."""
    out: Dict[int, dict] = {}
    if not player_ids:
        return out

    url = f"{BDL_NHL_BASE}/players"

    # The docs list `player_ids` as a filter, with examples like `player_ids[]=123`.
    # To avoid URL-length / API limits and occasional 400s, we chunk.
    ids = [int(x) for x in player_ids if x is not None]
    for i in range(0, len(ids), chunk):
        batch = ids[i : i + chunk]
        params: List[Tuple[str, str | int]] = [("per_page", int(per_page))]

        # Try the documented array form
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
            continue
        except Exception:
            # Fallback: some clients/APIs accept repeated `player_ids` without brackets
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
        # pick the last listed team entry (typically current season)
        t = teams[-1] or {}
        return str(t.get("tricode") or "").upper().strip()
    except Exception:
        return ""


def merge_bdl_props_mainlines(
    tracker: pd.DataFrame,
    game_date: str,
    api_key: str | None = None,
    vendors: Optional[List[str]] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """Fetch live BDL player props for the slate and merge best available lines/odds onto tracker.

    Produces (some subset):
      - BDL_Points_Line/Odds/Book
      - BDL_Assists_Line/Odds/Book
      - BDL_Goal_Line/Odds/Book
      - BDL_ATG_Line/Odds/Book
      - BDL_SOG_Line/Odds/Book (if present in feed)
    """
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
    if debug:
        print(f"[odds/ev] games found for {game_date}: {len(game_ids)}")
    if not game_ids:
        return df

    all_props: List[dict] = []
    for gid in game_ids:
        try:
            props = fetch_bdl_props_for_game(int(gid), api_key=api_key, vendors=vendors)
            if debug:
                print(f"[odds/ev] props fetched for game_id={gid}: {len(props)}")
            all_props.extend(props)
        except Exception as e:
            if debug:
                print(f"[odds/ev] props fetch failed for game_id={gid}: {e}")

    if not all_props:
        return df

    # Only keep prop types we can use right now
    keep_types = {
        "shots_on_goal",
        "points",
        "assists",
        "goals",
        "anytime_goal",
    }
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

    # Build a best-line-per-player record.
    # Rule: pick the lowest line_value for each prop_type (best chance), and if tie choose best payout for the over/milestone.
    best: Dict[Tuple[str, str, str], dict] = {}

    def _odds_from_market(mkt: dict) -> float | None:
        if not isinstance(mkt, dict):
            return None
        if mkt.get("type") == "over_under":
            return _safe_float(mkt.get("over_odds"))
        # milestone
        return _safe_float(mkt.get("odds"))

    def _better_choice(a: dict, b: dict) -> dict:
        """Return better of a/b based on lower line then higher payout."""
        la = _safe_float(a.get("line_value"))
        lb = _safe_float(b.get("line_value"))
        if la is None:
            return b
        if lb is None:
            return a
        if lb < la:
            return b
        if la < lb:
            return a
        oa = _odds_from_market(a.get("market") or {})
        ob = _odds_from_market(b.get("market") or {})
        if oa is None:
            return b
        if ob is None:
            return a
        # higher payout (more positive odds) is better for EV hunting; for negative odds, -110 is better than -150
        return b if float(ob) > float(oa) else a

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
        key = (_norm_name(name), team, ptype)
        if key not in best:
            best[key] = pr
        else:
            best[key] = _better_choice(best[key], pr)

    if not best:
        return df

    # Prep tracker merge keys
    if "Player_norm" in df.columns:
        df["__player_key"] = df["Player_norm"].map(_norm_name)
    else:
        df["__player_key"] = df["Player"].map(_norm_name)
    df["__team_key"] = df.get("Team_norm", df.get("Team", "")).astype(str).str.upper().str.strip()

    # Build merge rows
    rows: List[dict] = []
    for (pkey, tkey, ptype), pr in best.items():
        market = pr.get("market") or {}
        odds = None
        if isinstance(market, dict):
            odds = market.get("over_odds") if market.get("type") == "over_under" else market.get("odds")
        rows.append(
            {
                "__player_key": pkey,
                "__team_key": tkey,
                "prop_type": ptype,
                "line_value": _safe_float(pr.get("line_value")),
                "odds": _safe_float(odds),
                "vendor": str(pr.get("vendor") or ""),
            }
        )

    odds_df = pd.DataFrame(rows)
    if odds_df.empty:
        return df

    def _pivot(ptype: str, out_prefix: str):
        sub = odds_df[odds_df["prop_type"] == ptype].copy()
        if sub.empty:
            return
        sub = sub.drop_duplicates(subset=["__player_key", "__team_key"], keep="first")
        df.loc[:, f"BDL_{out_prefix}_Line"] = df.get(f"BDL_{out_prefix}_Line")
        m = sub.set_index(["__player_key", "__team_key"])
        # merge by index for speed
        idx = pd.MultiIndex.from_frame(df[["__player_key", "__team_key"]])
        df[f"BDL_{out_prefix}_Line"] = [m["line_value"].get(k, None) for k in idx]
        df[f"BDL_{out_prefix}_Odds"] = [m["odds"].get(k, None) for k in idx]
        df[f"BDL_{out_prefix}_Book"] = [m["vendor"].get(k, "") for k in idx]

    _pivot("shots_on_goal", "SOG")
    _pivot("points", "Points")
    _pivot("assists", "Assists")
    _pivot("goals", "Goal")
    _pivot("anytime_goal", "ATG")

    # cleanup
    df.drop(columns=["__player_key", "__team_key"], inplace=True, errors="ignore")
    return df




def _norm_name(x) -> str:
    if x is None:
        return ""
    s=str(x).lower().strip()
    s=s.replace(".", "").replace("'", "")
    s=" ".join(s.split())
    return s


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def american_to_implied_prob(odds) -> float | None:
    try:
        o = float(odds)
    except Exception:
        return None
    if o == 0:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)


def american_payout(odds) -> float | None:
    """Return profit on 1u stake (ex: +150 => 1.5, -120 => 0.8333)."""
    try:
        o = float(odds)
    except Exception:
        return None
    if o == 0:
        return None
    if o > 0:
        return o / 100.0
    return 100.0 / (-o)


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


def _conf_to_prob(conf: float) -> float:
    """Map model confidence (0-100) to a baseline probability proxy (0.05-0.95)."""
    # 50 -> 0.50, 70 -> 0.62, 80 -> 0.68, 90 -> 0.74
    return _clamp(0.50 + (conf - 50.0) * 0.006, 0.05, 0.95)


def _milestone_k(line_value: float) -> int:
    """Convert milestone line_value to integer threshold k for P(X >= k).

    Examples:
      0.5 -> 1+  => k=1
      1.0 -> 1+  => k=1
      1.5 -> 2+  => k=2
      3.0 -> 3+  => k=3
    """
    lv = float(line_value)
    if lv <= 0:
        return 1
    # integer lines are already "k+" for milestone markets
    if abs(lv - round(lv)) < 1e-6:
        k = int(round(lv))
        return max(k, 1)
    # x.5 lines mean (floor(x)+1)+
    return max(int(math.floor(lv)) + 1, 1)


# ------------------------------------------------------------------
# Back-compat helpers
# ------------------------------------------------------------------
# Earlier patch iterations referenced these names. Keep them so the
# EV pipeline can’t crash on a missing helper again.


def _line_to_threshold_k(line_value: float) -> int:
    """Alias of _milestone_k: convert a prop line to the integer k for P(X>=k)."""
    return _milestone_k(line_value)


def _poisson_tail_ge_k(lam: float, k: int) -> float:
    """Alias of _poisson_prob_ge, but accepts (lam, k) for convenience."""
    return _poisson_prob_ge(int(k), float(lam))


def _poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k) for Poisson(lam)."""
    if lam <= 0:
        return 1.0 if k >= 0 else 0.0
    k = int(k)
    if k < 0:
        return 0.0
    s = 0.0
    term = math.exp(-lam)  # i=0
    s += term
    for i in range(1, k + 1):
        term *= lam / float(i)
        s += term
    return _clamp(s, 0.0, 1.0)


def _poisson_prob_ge(k: int, lam: float) -> float:
    """P(X >= k) for Poisson(lam)."""
    return _clamp(1.0 - _poisson_cdf(k - 1, lam), 0.0, 1.0)


def _matrix_adjust(p: float, matrix_val: str | None) -> float:
    m = (matrix_val or "").strip().lower()
    if m == "green":
        return _clamp(p + 0.05, 0.05, 0.97)
    if m == "yellow":
        return _clamp(p - 0.03, 0.05, 0.97)
    if m == "red":
        return _clamp(p - 0.08, 0.05, 0.97)
    return p


def _earned_adjust(p: float, earned_flag) -> float:
    try:
        if bool(earned_flag):
            return _clamp(p + 0.03, 0.05, 0.97)
    except Exception:
        pass
    return p


def _heat_adjust(p: float, heat_val: str | None) -> float:
    h = (heat_val or "").strip().upper()
    if h in {"HOT", "WARM"}:
        return _clamp(p + 0.02, 0.05, 0.97)
    return p


def _line_decay(market: str, line_plus: float) -> float:
    """Decay factor applied when line is above 1+ (or above baseline for SOG/saves)."""
    m = market.upper()
    if m == "SOG":
        # Baseline around 2+; decay beyond that
        steps = max(int(round(line_plus)) - 2, 0)
        return 0.78 ** steps
    if m in {"POINTS", "ASSISTS"}:
        steps = max(int(round(line_plus)) - 1, 0)
        return 0.65 ** steps
    if m == "GOAL":
        steps = max(int(round(line_plus)) - 1, 0)
        return 0.60 ** steps
    if m == "PPP":
        steps = max(int(round(line_plus)) - 1, 0)
        return 0.60 ** steps
    if m == "SAVES":
        # Save milestones are bigger numbers; scale steps by 5 saves above 20
        steps = max(int((line_plus - 20) // 5), 0)
        return 0.90 ** steps
    return 1.0


def _compute_ev_pct(p_model: float, odds) -> float | None:
    payout = american_payout(odds)
    if payout is None:
        return None
    # EV in "units" for 1u stake: win profit payout, lose -1
    ev = p_model * payout - (1.0 - p_model) * 1.0
    return ev * 100.0


def add_bdl_ev_all(tracker: pd.DataFrame) -> pd.DataFrame:
    df = tracker.copy()

    markets: Dict[str, Dict[str, str]] = {
        "SOG": {
            "line": "BDL_SOG_Line",
            "odds": "BDL_SOG_Odds",
            "book": "BDL_SOG_Book",
            "conf": "Conf_SOG",
            "matrix": "Matrix_SOG",
            "earned": "Plays_SOG",
            "heat": "Reg_Heat_S",
        },
        "POINTS": {
            "line": "BDL_Points_Line",
            "odds": "BDL_Points_Odds",
            "book": "BDL_Points_Book",
            "conf": "Conf_Points",
            "matrix": "Matrix_Points",
            "earned": "Plays_Points",
            "heat": "Reg_Heat_P",
        },
        "GOAL": {
            "line": "BDL_Goal_Line",
            "odds": "BDL_Goal_Odds",
            "book": "BDL_Goal_Book",
            "conf": "Conf_Goal",
            "matrix": "Matrix_Goal",
            "earned": "Plays_Goal",
            "heat": "Reg_Heat_G",
        },
        "ATG": {
            "line": "BDL_ATG_Line",
            "odds": "BDL_ATG_Odds",
            "book": "BDL_ATG_Book",
            "conf": "Conf_Goal",
            "matrix": "Matrix_Goal",
            "earned": "Plays_Goal",
            "heat": "Reg_Heat_G",
        },
        "ASSISTS": {
            "line": "BDL_Assists_Line",
            "odds": "BDL_Assists_Odds",
            "book": "BDL_Assists_Book",
            "conf": "Conf_Assists",
            "matrix": "Matrix_Assists",
            "earned": "Plays_Assists",
            "heat": "Reg_Heat_A",
        },
        "PPP": {
            "line": "BDL_PPP_Line",
            "odds": "BDL_PPP_Odds",
            "book": "BDL_PPP_Book",
            "conf": "Conf_Points",  # best available proxy
            "matrix": "Matrix_Points",
            "earned": "Plays_Points",
            "heat": "Reg_Heat_P",
        },
        "SAVES": {
            "line": "BDL_Saves_Line",
            "odds": "BDL_Saves_Odds",
            "book": "BDL_Saves_Book",
            "conf": "Conf_Saves",
            "matrix": "Matrix_Saves",
            "earned": "Plays_Saves",
            "heat": "Reg_Heat_Saves",
        },
    }

    for mkt, cfg in markets.items():
        line_c = cfg["line"]
        odds_c = cfg["odds"]
        book_c = cfg["book"]
        conf_c = cfg["conf"]
        matrix_c = cfg["matrix"]
        earned_c = cfg["earned"]
        heat_c = cfg["heat"]

        if line_c not in df.columns or odds_c not in df.columns:
            continue
        if conf_c not in df.columns:
            continue

        out_prefix = ("SOG" if mkt == "SOG" else ("Goal" if mkt == "GOAL" else ("ATG" if mkt == "ATG" else ("Assists" if mkt == "ASSISTS" else ("Points" if mkt == "POINTS" else mkt.title())))))

        # output columns
        df[f"{out_prefix}_Line"] = pd.to_numeric(df[line_c], errors="coerce")
        df[f"{out_prefix}_Book"] = df.get(book_c, "")
        df[f"{out_prefix}_Odds_Over"] = pd.to_numeric(df[odds_c], errors="coerce")

        # probabilities
        # Prefer a Poisson model when we have an expectation column (Exp_*_10) and a milestone line.
        # This makes p_model interpretable and consistent across markets.
        exp_col = None
        if mkt == "SOG":
            exp_col = "Exp_S_10"
        elif mkt == "POINTS":
            exp_col = "Exp_P_10"
        elif mkt in {"GOAL", "ATG"}:
            exp_col = "Exp_G_10"
        elif mkt == "ASSISTS":
            exp_col = "Exp_A_10"

        line_vals = pd.to_numeric(df[f"{out_prefix}_Line"], errors="coerce")

        p = [None] * len(df)
        # Track whether Poisson was used (so we don't apply heuristic line-decay)
        use_poisson = [False] * len(df)
        if exp_col and exp_col in df.columns:
            exp_v = pd.to_numeric(df[exp_col], errors="coerce")
            for i, (ev10, lv) in enumerate(zip(exp_v.tolist(), line_vals.tolist())):
                if ev10 is None or pd.isna(ev10) or lv is None or pd.isna(lv):
                    continue
                lam = float(ev10) / 10.0
                if lam <= 0:
                    continue
                k = _line_to_threshold_k(float(lv))
                p[i] = _poisson_tail_ge_k(lam, k)
                use_poisson[i] = True

        # Fallback: conf -> probability (if we couldn't compute Poisson)
        if any(pi is None for pi in p):
            conf_v = pd.to_numeric(df[conf_c], errors="coerce")
            p_conf = conf_v.apply(lambda x: _conf_to_prob(float(x)) if x is not None and not pd.isna(x) else None)
            for i in range(len(p)):
                if p[i] is None:
                    p[i] = p_conf.iat[i]

        # matrix/earned/heat adjustments
        if matrix_c in df.columns:
            p = [(_matrix_adjust(pi, mv) if pi is not None else None) for pi, mv in zip(p, df[matrix_c].astype(str))]
        if earned_c in df.columns:
            p = [(_earned_adjust(pi, ev) if pi is not None else None) for pi, ev in zip(p, df[earned_c])]
        if heat_c in df.columns:
            p = [(_heat_adjust(pi, hv) if pi is not None else None) for pi, hv in zip(p, df[heat_c].astype(str))]
        # apply line-decay only when using the confidence fallback (Poisson already accounts for line threshold)
        line_vals = pd.to_numeric(df[f"{out_prefix}_Line"], errors="coerce")
        _use_poisson = use_poisson

        p2 = []
        for pi, lv, up in zip(p, line_vals, _use_poisson):
            # If we have a probability but the line is missing, keep the probability
            # (this prevents odds rows from ending up with p_model=None when a line
            # failed to merge for some reason).
            if pi is None:
                p2.append(None)
                continue
            if lv is None or pd.isna(lv):
                p2.append(_clamp(float(pi), 0.02, 0.95))
                continue
            if bool(up):
                p2.append(_clamp(float(pi), 0.02, 0.95))
                continue
            dec = _line_decay(mkt, float(lv))
            p2.append(_clamp(float(pi) * dec, 0.02, 0.95))

        df[f"{out_prefix}_p_model_over"] = pd.to_numeric(pd.Series(p2), errors="coerce")
        df[f"{out_prefix}_p_imp_over"] = pd.to_numeric(df[f"{out_prefix}_Odds_Over"].apply(american_to_implied_prob), errors="coerce")
        df[f"{out_prefix}_EVpct_over"] = pd.to_numeric(
            [(_compute_ev_pct(float(pm), od) if pm is not None and not pd.isna(pm) else None)
             for pm, od in zip(df[f"{out_prefix}_p_model_over"], df[f"{out_prefix}_Odds_Over"])],
            errors="coerce"
        )

    return df


# ----------------------------
# Merge BallDontLie milestone props (single-odds) onto tracker
# ----------------------------

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def _auth_headers() -> Dict[str, str]:
    k = os.getenv("BALLDONTLIE_API_KEY", "").strip()
    if not k:
        raise RuntimeError("Missing env var BALLDONTLIE_API_KEY")
    return {"Authorization": k}


def _fetch_games_for_date(game_date: str) -> Dict[frozenset, int]:
    url = f"https://api.balldontlie.io/nhl/v1/games"
    r = requests.get(url, headers=_auth_headers(), params={"dates[]": game_date, "per_page": 100}, timeout=25, verify=certifi.where())
    r.raise_for_status()
    data = (r.json() or {}).get("data", []) or []
    out = {}
    for g in data:
        gid = g.get("id")
        home = (g.get("home_team") or {}).get("tricode")
        away_obj = g.get("visitor_team") or g.get("away_team") or {}
        away = (away_obj or {}).get("tricode")
        if gid and home and away:
            out[frozenset([str(home).upper(), str(away).upper()])] = int(gid)
    return out


def _fetch_player_props(game_id: int, prop_type: str, vendors: Optional[Tuple[str, ...]] = None) -> List[dict]:
    url = f"https://api.balldontlie.io/nhl/v1/odds/player_props"
    params = {"game_id": int(game_id), "prop_type": prop_type, "per_page": 100}
    if vendors:
        params["vendors[]"] = list(vendors)
    r = requests.get(url, headers=_auth_headers(), params=params, timeout=30, verify=certifi.where())
    r.raise_for_status()
    return (r.json() or {}).get("data", []) or []


def _fetch_players_by_ids(player_ids: List[int]) -> Dict[int, str]:
    # per_page max is 100; chunk requests
    out: Dict[int, str] = {}
    if not player_ids:
        return out
    url = f"https://api.balldontlie.io/nhl/v1/players"
    for i in range(0, len(player_ids), 80):
        chunk = player_ids[i:i+80]
        params = {"player_ids[]": chunk, "per_page": 100}
        r = requests.get(url, headers=_auth_headers(), params=params, timeout=30, verify=certifi.where())
        r.raise_for_status()
        data = (r.json() or {}).get("data", []) or []
        for pl in data:
            pid = pl.get("id")
            fn = (pl.get("first_name") or "").strip()
            ln = (pl.get("last_name") or "").strip()
            if pid and (fn or ln):
                out[int(pid)] = (fn + " " + ln).strip()
    return out


def _pick_market_odds(it: dict) -> Optional[float]:
    m = it.get("market")
    if isinstance(m, dict):
        # milestone typically: {type: milestone, odds: <american>}
        for k in ("odds", "price", "american_odds", "odds_american", "americanOdds"):
            v = m.get(k)
            if v is not None and str(v).strip() != "":
                try:
                    return float(v)
                except Exception:
                    return None
    return None


def _line_to_threshold_k(line_val: float) -> int:
    """Convert a line value to an integer milestone threshold k.

    BallDontLie NHL props for these markets are typically milestone style
    (e.g., 1+ assist, 2+ points). Feeds sometimes represent this as an integer
    (1, 2, 3) or with a half-step (0.5, 1.5) depending on vendor.

    We normalize to the *minimum integer event count* required to cash:
      - 0.5 -> 1
      - 1.0 -> 1
      - 1.5 -> 2
      - 2.0 -> 2

    In other words: k = ceil(line).
    """
    try:
        lv = float(line_val)
        if math.isnan(lv):
            return 1
        k = int(math.ceil(lv))
        return max(1, k)
    except Exception:
        return 1


def merge_bdl_milestones(
    tracker: pd.DataFrame,
    game_date: str,
    vendors: Optional[Tuple[str, ...]] = ("fanduel", "draftkings", "caesars"),
    debug: bool = False,
) -> pd.DataFrame:
    """Merge milestone odds (single American odds per line) for multiple markets onto tracker.

    Writes BDL_* columns consumed by add_bdl_ev_all():
      - BDL_Points_Line/Odds/Book
      - BDL_Assists_Line/Odds/Book
      - BDL_Goal_Line/Odds/Book
      - BDL_ATG_Line/Odds/Book
      - BDL_PPP_Line/Odds/Book
      - BDL_SOG_Line/Odds/Book  (milestone SOG, NOT O/U)

    Note: BallDontLie NHL player_props are milestone-style markets (no over/under for SOG).
    """
    df = tracker.copy()

    # ensure schema always exists
    bdl_cols = [
        "BDL_Points_Line","BDL_Points_Odds","BDL_Points_Book",
        "BDL_Assists_Line","BDL_Assists_Odds","BDL_Assists_Book",
        "BDL_Goal_Line","BDL_Goal_Odds","BDL_Goal_Book",
        "BDL_ATG_Line","BDL_ATG_Odds","BDL_ATG_Book",
        "BDL_PPP_Line","BDL_PPP_Odds","BDL_PPP_Book",
        "BDL_SOG_Line","BDL_SOG_Odds","BDL_SOG_Book",
    ]
    df = _ensure_cols(df, bdl_cols)

    # Need Team/Opp to map to game_id
    if "Team" not in df.columns or "Opp" not in df.columns:
        return df

    try:
        game_map = _fetch_games_for_date(game_date)
        if debug:
            print(f"[bdl] games found for {game_date}: {len(game_map)}")
    except Exception as e:
        if debug:
            print(f"[bdl] games fetch failed: {e}")
        return df

    # Attach game_id for each row
    def _gid(row) -> Optional[int]:
        t = str(row.get("Team", "")).upper().strip()
        o = str(row.get("Opp", "")).upper().strip()
        return game_map.get(frozenset([t, o]))

    df["_bdl_game_id"] = df.apply(_gid, axis=1)

    # Normalize player names for join
    if "Player" not in df.columns:
        df.drop(columns=["_bdl_game_id"], inplace=True, errors="ignore")
        return df
    df["_pname_norm"] = df["Player"].apply(_norm_name)

    markets = {
        # market_name: (prop_type, output_prefix, prefer_line)
        "POINTS": ("points", "Points", 1.0),
        "ASSISTS": ("assists", "Assists", 1.0),
        "GOAL": ("goals", "Goal", 1.0),
        "ATG": ("anytime_goal", "ATG", 1.0),
        "PPP": ("power_play_points", "PPP", 1.0),
        # SOG is milestone; 3+ is most useful when available
        "SOG": ("shots_on_goal", "SOG", 3.0),
    }

    # Pull props for each game once per prop_type
    all_rows = []
    # Only consider games actually present in df
    game_ids = sorted({int(x) for x in df["_bdl_game_id"].dropna().unique().tolist()})
    if not game_ids:
        df.drop(columns=["_bdl_game_id","_pname_norm"], inplace=True, errors="ignore")
        return df

    for mkt, (ptype, outp, prefer_line) in markets.items():
        rows = []
        for gid in game_ids:
            try:
                items = _fetch_player_props(gid, ptype, vendors=vendors)
                if debug:
                    print(f"[bdl] {ptype} props game_id={gid}: {len(items)}")
            except Exception as e:
                if debug:
                    print(f"[bdl] {ptype} fetch failed game_id={gid}: {e}")
                continue
            for it in items:
                pid = it.get("player_id")
                if pid is None:
                    continue
                odds = _pick_market_odds(it)
                if odds is None:
                    continue
                try:
                    lv = float(it.get("line_value"))
                except Exception:
                    lv = None
                rows.append({
                    "game_id": int(gid),
                    "player_id": int(pid),
                    "vendor": str(it.get("vendor") or ""),
                    "line_value": lv,
                    "odds": float(odds),
                })

        if not rows:
            continue

        # player names
        pids = sorted({r["player_id"] for r in rows})
        name_map = _fetch_players_by_ids(pids)
        for r in rows:
            nm = name_map.get(r["player_id"], "")
            if not nm:
                continue
            r["player_name"] = nm
            r["pname_norm"] = _norm_name(nm)
        rows = [r for r in rows if r.get("pname_norm")]
        if not rows:
            continue

        props_df = pd.DataFrame(rows)

        # Pick preferred line if available; otherwise smallest line
        # Then choose best (highest) payout odds within that line
        props_df["line_value"] = pd.to_numeric(props_df["line_value"], errors="coerce")
        # if prefer_line exists in group, filter to it else min
        best_rows = []
        for (gid, pname), gdf in props_df.groupby(["game_id", "pname_norm"], dropna=True):
            gdf = gdf.dropna(subset=["odds"])
            if gdf.empty:
                continue
            if prefer_line is not None and (gdf["line_value"] == prefer_line).any():
                gg = gdf[gdf["line_value"] == prefer_line]
            else:
                min_line = gdf["line_value"].min()
                gg = gdf[gdf["line_value"] == min_line]
            # pick the best odds (max american)
            pick = gg.sort_values(["odds"], ascending=[False]).iloc[0]
            best_rows.append(pick)

        if not best_rows:
            continue

        best = pd.DataFrame(best_rows)
        best = best[["game_id","pname_norm","vendor","line_value","odds"]].copy()
        best.rename(columns={
            "game_id": "_bdl_game_id",
            "pname_norm": "_pname_norm",
            "vendor": f"BDL_{outp}_Book",
            "line_value": f"BDL_{outp}_Line",
            "odds": f"BDL_{outp}_Odds",
        }, inplace=True)

        df = df.merge(best, on=["_bdl_game_id","_pname_norm"], how="left", suffixes=("", ""))

    df.drop(columns=["_bdl_game_id","_pname_norm"], inplace=True, errors="ignore")
    return df
