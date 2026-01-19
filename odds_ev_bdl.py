from __future__ import annotations

"""odds_ev_bdl.py — BallDontLie odds merge + market-aware model probability + EV

This version turns the BDL integration into the **alt-line cash checker** backbone.

Why this matters:
- Your previous engine normalized Points/Assists/Goals/ATG to **0.5 only**.
  That makes model probability look “static” and makes natural star lines
  (e.g. McDavid/MacK 1.5 points) impossible to evaluate.

What’s new:
1) Alt lines: keep up to Top-K distinct lines per player/market.
   Columns written (stable schema):
     - BDL_{M}_Line_1..K, BDL_{M}_Odds_1..K, BDL_{M}_Book_1..K
   And a selected “mainline”:
     - BDL_{M}_Line / Odds / Book

2) Market-aware probability:
   - Uses **Negative Binomial tails** (Poisson-gamma) per market to allow
     over-dispersion vs Poisson.
   - Falls back to Poisson if dispersion collapses.

3) Dataflow:
   nhl_edge.py merges BDL odds first, then add_bdl_ev_all() computes:
     - {M}_Line, {M}_Odds_Over, {M}_Book
     - {M}_p_model_over, {M}_p_imp_over, {M}_EVpct_over
     - friendly display: {M}_Model%, {M}_EV%
     - and alt variants: {M}_p_model_over_1..K, etc.
"""

import os
import math
from typing import Dict, Optional, Tuple, List, Any

import requests
import certifi
import pandas as pd

__BDL_EV_ENGINE__ = "ALTLINE_NB_v1_2026-01-18"

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


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _norm_name(x) -> str:
    if x is None:
        return ""
    s = str(x).lower().strip()
    s = s.replace(".", "").replace("'", "")
    s = " ".join(s.split())
    return s


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


def _round_to_half(x: float) -> float:
    return round(float(x) * 2.0) / 2.0


def _k_for_over(line_value: float) -> int:
    """Over threshold: line 0.5 -> need >=1, 1.5 -> >=2, 2.5 -> >=3.

    We treat integer lines as "need >= line+1" (push-lines are uncommon in player props).
    """
    try:
        lv = float(line_value)
        if math.isnan(lv) or lv <= 0:
            return 1
        return max(1, int(math.floor(lv)) + 1)
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


def _nb_tail_ge_k(mu: float, alpha: float, k: int) -> float:
    """Negative Binomial tail with mean=mu and variance=mu + alpha*mu^2.

    Uses NB parameterization with real-valued r:
      var = mu + mu^2 / r  -> r = mu^2 / (var-mu)
      p = r/(r+mu)

    Tail = 1 - CDF(k-1), computed via PMF recurrence.
    """
    if mu is None or mu <= 0 or k <= 0:
        return 0.0

    # variance model
    var = mu + max(0.0, alpha) * (mu ** 2)
    if var <= mu + 1e-12:
        return _poisson_tail_ge_k(mu, k)

    r = (mu ** 2) / max(1e-12, (var - mu))
    if r <= 0:
        return _poisson_tail_ge_k(mu, k)

    p = r / (r + mu)  # success prob
    p = _clamp(p, 1e-9, 1.0 - 1e-9)

    # CDF up to k-1 via recurrence
    # pmf(0) = p^r
    try:
        pmf = math.exp(r * math.log(p))
    except Exception:
        pmf = p ** r

    cdf = pmf
    for i in range(0, k - 1):
        # pmf(i+1) from pmf(i)
        pmf = pmf * ((i + r) / (i + 1.0)) * (1.0 - p)
        cdf += pmf
        if cdf >= 0.999999:
            cdf = 0.999999
            break

    return _clamp(1.0 - cdf, 0.0, 0.999)


def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _lambda_from_exp10_row(
    row: pd.Series,
    exp_candidates: List[str],
    fallback_rate_cols: List[Tuple[str, float]],
) -> float:
    """Prefer Exp_*_10 totals -> mu = Exp/10. Else fall back to rate columns."""
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


def _blend_mu_row(market: str, row: pd.Series, mu_recent: float) -> float:
    """Blend mu ONLY for Goal / ATG.

    IMPORTANT (baseline protection):
    - Points / Assists / SOG mu should come straight from nhl_edge.py (or Exp_* / fallbacks),
      and must NOT be re-blended here. Re-blending those markets causes drift vs your
      locked baseline behavior.
    - Goal / ATG are special: higher variance + ATG semantics. Here we allow a light
      season-prior / league anchor blend and a low-shot dampener.

    Returns a per-game mean (mu).
    """

    mu_recent = float(mu_recent or 0.0)
    if mu_recent < 0:
        mu_recent = 0.0

    # 🔒 Preserve baseline behavior for non-goal markets
    if market not in ("Goal", "ATG"):
        return mu_recent

    cfgs = _market_cfgs()
    cfg = cfgs.get(market) or {}
    league_mu = float(cfg.get("league_mu", 0.0))
    alpha = float(cfg.get("alpha", 0.0))

    # Season prior for goals (if present)
    season_mu = 0.0
    for c in ("GPG", "GoalsPerGame"):
        if c in row.index:
            v = _safe_float(row.get(c))
            if v is not None and v > 0:
                season_mu = float(v)
                break

    # If we have season_mu, use a conservative blend; else fall back to alpha shrinkage
    if season_mu > 0:
        # Goal is noisier than ATG: lean more on season + league
        if market == "Goal":
            w_recent, w_season, w_league = 0.25, 0.50, 0.25
        else:  # ATG
            w_recent, w_season, w_league = 0.35, 0.45, 0.20
        mu = (w_recent * mu_recent) + (w_season * season_mu) + (w_league * league_mu)
    else:
        mu = alpha * mu_recent + (1.0 - alpha) * league_mu

    # Low-shot dampener (prevents "sniper noise")
    sog_mu = _safe_float(row.get("SOG_mu"))
    if sog_mu is not None and sog_mu > 0:
        if sog_mu < 1.5:
            mu *= 0.75
        elif sog_mu < 2.0:
            mu *= 0.85

    return max(0.0, float(mu))


# -----------------------------

# BallDontLie API helpers
# -----------------------------

def _bdl_headers(api_key: str | None) -> Dict[str, str]:
    k = (
        api_key
        or os.getenv("BDL_API_KEY")
        or os.getenv("BALLDONTLIE_KEY")
        or os.getenv("BALLDONTLIE_API_KEY")
        or ""
    ).strip()
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
            # fallback variant
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
# Market config
# -----------------------------

_MARKETS = {
    # mapping from BDL prop_type -> (MarketName, baseline_min, valid_range)
    "shots_on_goal": ("SOG", 2.0, (1.5, 8.5)),
    "points": ("Points", 0.5, (0.5, 4.5)),
    "assists": ("Assists", 0.5, (0.5, 3.5)),
    "goals": ("Goal", 0.5, (0.5, 2.5)),
    "anytime_goal": ("ATG", 0.5, (0.5, 2.5)),
}


def _market_cfgs() -> Dict[str, dict]:
    """Central config used by both selection + EV."""
    return {
        "SOG": {
            "exp": ["Exp_S_10", "Exp_SOG_10", "Exp_SOG10"],
            "fallback": [("S10_total", 1 / 10), ("L10_S", 1 / 10)],
            "league_mu": 2.3,
            "alpha": 0.25,
        },
        "Points": {
            "exp": ["Exp_P_10", "Exp_Points_10", "Exp_P10"],
            "fallback": [("L10_P", 1 / 10), ("P10_total", 1 / 10), ("PPG", 1.0)],
            "league_mu": 0.60,
            "alpha": 0.35,
        },
        "Assists": {
            "exp": ["Exp_A_10", "Exp_Assists_10", "Exp_A10"],
            "fallback": [("L10_A", 1 / 10), ("A10_total", 1 / 10), ("APG", 1.0)],
            "league_mu": 0.35,
            "alpha": 0.50,
        },
        "Goal": {
            "exp": ["Exp_G_10", "Exp_Goals_10", "Exp_G10"],
            # Goals are high-variance: trust recent totals lightly, lean on season talent.
            "fallback": [("GPG", 1.0), ("L10_G", 1 / 10)],
            "league_mu": 0.30,
            "alpha": 0.40,
        },
        "ATG": {
            "exp": ["Exp_G_10", "Exp_Goals_10", "Exp_G10"],
            # ATG uses xG/goal-expectation signal (lower variance than raw goals).
            "fallback": [("GPG", 1.0), ("L10_G", 1 / 10)],
            "league_mu": 0.30,
            "alpha": 0.65,
        },
    }


def _target_line(market: str, mu: float) -> float:
    """Heuristic for “natural” mainline selection among available alt lines."""
    m = market
    mu = float(mu or 0.0)

    if m == "SOG":
        if mu >= 4.2:
            return 4.5
        if mu >= 3.3:
            return 3.5
        if mu >= 2.7:
            return 2.5
        return 2.0

    if m == "Points":
        if mu >= 1.45:
            return 2.5
        if mu >= 1.05:
            return 1.5
        return 0.5

    if m == "Assists":
        if mu >= 0.95:
            return 1.5
        return 0.5

    if m == "Goal":
        # Most books keep goals at 0.5; allow 1.5 if a true scorer.
        if mu >= 0.70:
            return 1.5
        return 0.5

    if m == "ATG":
        return 0.5

    return 0.5


# -----------------------------
# Merge BDL props -> tracker (ALT LINES)
# -----------------------------


def merge_bdl_props_altlines(
    tracker: pd.DataFrame,
    game_date: str,
    api_key: str | None = None,
    vendors: Optional[List[str]] = None,
    top_k: int = 4,
    debug: bool = False,
) -> pd.DataFrame:
    """Merge BDL props with Top-K alt lines per market.

    Writes:
      - BDL_{M}_Line_1..K, BDL_{M}_Odds_1..K, BDL_{M}_Book_1..K
      - BDL_{M}_Line/Odds/Book (selected mainline)

    Selection is *per player* using tracker mu (Exp_*_10 / 10) and a market-specific
    target line.
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

    keep_types = set(_MARKETS.keys())
    props = [p for p in all_props if str(p.get("prop_type") or "") in keep_types]
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
        # always treat as over
        if mkt.get("type") == "over_under":
            return _safe_float(mkt.get("over_odds"))
        return _safe_float(mkt.get("odds"))

    def _american_to_imp(odds: float | None) -> float | None:
        if odds is None:
            return None
        try:
            o = float(odds)
        except Exception:
            return None
        if o == 0:
            return None
        if o < 0:
            return (-o) / ((-o) + 100.0)
        return 100.0 / (o + 100.0)

    def _pick_conservative_mainline(
        kp: Tuple[str, str, str],
        market: str,
        arr: List[float],
        tgt: float | None,
    ) -> float:
        """Pick a sportsbook-like mainline:
        - For Points/Assists/Goal/ATG: skip integer (push) lines.
        - Prefer odds closest to even (implied ~50%).
        - Tie-break: prefer LOWER line (more conservative/mainline-like).
        - If odds missing for all lines: fallback to nearest-to-target then lower line.
        """
        candidates = [float(x) for x in arr]
        if market in ("Points", "Assists", "Goal", "ATG"):
            candidates = [x for x in candidates if not float(x).is_integer()]
        if not candidates:
            candidates = [float(x) for x in arr]

        scored = []
        for lv in candidates:
            o = _get_odds(kp, lv)
            imp = _american_to_imp(o)
            if imp is None:
                continue
            scored.append((abs(imp - 0.5), lv))

        if scored:
            scored.sort(key=lambda t: (t[0], t[1]))
            return float(scored[0][1])

        # odds unavailable -> use target distance
        if tgt is None:
            srt = sorted(candidates)
            mid = len(srt) // 2
            tgt = float(srt[mid]) if srt else 0.0
        return float(min(sorted(candidates), key=lambda x: (abs(float(x) - float(tgt)), float(x))))

    # bucket best odds per (player, team, market, line)
    best_per_line: Dict[Tuple[str, str, str, float], Tuple[float, str]] = {}

    for pr in props:
        pid = pr.get("player_id")
        pobj = players_map.get(int(pid)) if pid is not None else None
        if not pobj:
            continue

        full_name = str(pobj.get("full_name") or "").strip()
        if not full_name:
            continue

        team = _player_team_tricode(pobj)
        ptype = str(pr.get("prop_type") or "")
        mkt_name, baseline_min, (lo, hi) = _MARKETS.get(ptype, (None, None, None))
        if not mkt_name:
            continue

        lv = _safe_float(pr.get("line_value"))
        if lv is None:
            continue

        lv = _round_to_half(lv)
        # BDL "anytime_goal_scorer" sometimes reports line_value=1.0; treat as 0.5 (score 1+).
        if mkt_name == "ATG" and lv >= 1.0:
            lv = 0.5

        if baseline_min is not None and lv < float(baseline_min):
            continue
        if lv < lo or lv > hi:
            continue

        odds = _odds_from_market(pr.get("market") or {})
        if odds is None:
            continue

        key = (_norm_name(full_name), team, mkt_name, float(lv))
        curr = best_per_line.get(key)
        vendor = str(pr.get("vendor") or "")
        if curr is None or float(odds) > float(curr[0]):
            best_per_line[key] = (float(odds), vendor)

    if not best_per_line:
        return df

    # attach keys to tracker
    if "Player_norm" in df.columns:
        df["__player_key"] = df["Player_norm"].map(_norm_name)
    else:
        df["__player_key"] = df["Player"].map(_norm_name)

    df["__team_key"] = df.get("Team_norm", df.get("Team", "")).astype(str).str.upper().str.strip()

    cfgs = _market_cfgs()

    # build per-row market -> sorted line list
    for market in cfgs.keys():
        # collect all lines for this market per (player,team)
        lines_map: Dict[Tuple[str, str], List[float]] = {}
        for (pkey, tkey, mkt, lv), (od, vendor) in best_per_line.items():
            if mkt != market:
                continue
            lines_map.setdefault((pkey, tkey), []).append(float(lv))

        if not lines_map:
            continue

        # de-dupe + sort
        for k in list(lines_map.keys()):
            lines_map[k] = sorted({float(x) for x in lines_map[k]})

        # fill alt columns (Line_1..K etc)
        def _get_line_at(idx_key: Tuple[str, str], i: int) -> float | None:
            arr = lines_map.get(idx_key)
            if not arr or i >= len(arr):
                return None
            return float(arr[i])

        def _get_odds(idx_key: Tuple[str, str], lv: float | None) -> float | None:
            if lv is None:
                return None
            t = best_per_line.get((idx_key[0], idx_key[1], market, float(lv)))
            return None if t is None else float(t[0])

        def _get_vendor(idx_key: Tuple[str, str], lv: float | None) -> str:
            if lv is None:
                return ""
            t = best_per_line.get((idx_key[0], idx_key[1], market, float(lv)))
            return "" if t is None else str(t[1] or "")

        idx_pairs = list(zip(df["__player_key"].tolist(), df["__team_key"].tolist()))

        K = max(1, int(top_k))
        for i in range(K):
            lv_i = [_get_line_at(kp, i) for kp in idx_pairs]
            df[f"BDL_{market}_Line_{i+1}"] = lv_i
            df[f"BDL_{market}_Odds_{i+1}"] = [_get_odds(kp, lv) for kp, lv in zip(idx_pairs, lv_i)]
            df[f"BDL_{market}_Book_{i+1}"] = [_get_vendor(kp, lv) for kp, lv in zip(idx_pairs, lv_i)]

        # choose a mainline per player based on mu
        chosen_line: List[float | None] = []
        chosen_odds: List[float | None] = []
        chosen_book: List[str] = []
        for kp, (_, row) in zip(idx_pairs, df.iterrows()):
            arr = lines_map.get(kp) or []
            if not arr:
                chosen_line.append(None)
                chosen_odds.append(None)
                chosen_book.append("")
                continue
            mu = _lambda_from_exp10_row(row, cfgs[market]["exp"], cfgs[market]["fallback"])
            if mu <= 0:
                mu = float(cfgs[market]["league_mu"])
            tgt = _target_line(market, mu)

            # pick a sportsbook-like mainline (conservative; avoids big alt lines)
            arr_sorted = sorted(arr)
            best_lv = _pick_conservative_mainline(kp, market, arr_sorted, tgt)
            chosen_line.append(float(best_lv))
            chosen_odds.append(_get_odds(kp, best_lv))
            chosen_book.append(_get_vendor(kp, best_lv))

        df[f"BDL_{market}_Line"] = chosen_line
        df[f"BDL_{market}_Odds"] = chosen_odds
        df[f"BDL_{market}_Book"] = chosen_book

    df.drop(columns=["__player_key", "__team_key"], inplace=True, errors="ignore")
    return df


# Backwards-compatible alias (older code calls this)

def merge_bdl_props_mainlines(
    tracker: pd.DataFrame,
    game_date: str,
    api_key: str | None = None,
    vendors: Optional[List[str]] = None,
    debug: bool = False,
) -> pd.DataFrame:
    return merge_bdl_props_altlines(tracker, game_date, api_key=api_key, vendors=vendors, top_k=1, debug=debug)


def merge_bdl_milestones(
    tracker: pd.DataFrame,
    game_date: str,
    vendors: Optional[Tuple[str, ...]] = ("fanduel", "draftkings", "caesars"),
    debug: bool = False,
) -> pd.DataFrame:
    vlist = list(vendors) if vendors else None
    return merge_bdl_props_altlines(tracker, game_date, api_key=None, vendors=vlist, top_k=1, debug=debug)


# -----------------------------
# Compute EV / model prob
# -----------------------------


def _model_prob_over(market: str, mu: float, line: float) -> float:
    cfgs = _market_cfgs()
    cfg = cfgs.get(market)
    if cfg is None:
        return float("nan")

    if mu <= 0:
        mu = float(cfg["league_mu"])

    k = _k_for_over(line)
    alpha = float(cfg.get("alpha", 0.0))
    p = _nb_tail_ge_k(mu, alpha, k)
    return p


def add_bdl_ev_all(tracker: pd.DataFrame, top_k: int = 4) -> pd.DataFrame:
    """Add per-market Line/Odds/Book + p_model/p_imp/EVpct columns, and stamp engine version.

    Also computes alt-line model/EV for up to top_k lines if present.
    """

    df = tracker.copy()
    if df.empty:
        return df

    df["BDL_EV_ENGINE"] = __BDL_EV_ENGINE__

    cfgs = _market_cfgs()
    K = max(1, int(top_k))

    for market, cfg in cfgs.items():
        # Prefer selected BDL mainline; fallback to already-surfaced columns
        bdl_line = f"BDL_{market}_Line"
        bdl_odds = f"BDL_{market}_Odds"
        bdl_book = f"BDL_{market}_Book"

        line_in = bdl_line if bdl_line in df.columns else f"{market}_Line"
        odds_in = bdl_odds if bdl_odds in df.columns else f"{market}_Odds_Over"
        book_in = bdl_book if bdl_book in df.columns else f"{market}_Book"

        # Surface mainline
        if f"{market}_Line" not in df.columns:
            df[f"{market}_Line"] = pd.to_numeric(df.get(line_in), errors="coerce")
        else:
            # keep if already exists, but update from BDL if present
            if bdl_line in df.columns:
                m = pd.to_numeric(df[bdl_line], errors="coerce").notna()
                df.loc[m, f"{market}_Line"] = pd.to_numeric(df.loc[m, bdl_line], errors="coerce")

        if f"{market}_Odds_Over" not in df.columns:
            df[f"{market}_Odds_Over"] = pd.to_numeric(df.get(odds_in), errors="coerce")
        else:
            if bdl_odds in df.columns:
                m = pd.to_numeric(df[bdl_odds], errors="coerce").notna()
                df.loc[m, f"{market}_Odds_Over"] = pd.to_numeric(df.loc[m, bdl_odds], errors="coerce")

        if f"{market}_Book" not in df.columns:
            df[f"{market}_Book"] = df.get(book_in, "")
        else:
            if bdl_book in df.columns:
                m = df[bdl_book].astype(str).str.strip().ne("")
                df.loc[m, f"{market}_Book"] = df.loc[m, bdl_book]

        # Compute mu per row
        mus: List[float] = []
        mu_src: List[str] = []
        for _, row in df.iterrows():
            mu = _lambda_from_exp10_row(row, cfg["exp"], cfg["fallback"])
            if mu > 0:
                mu = _blend_mu_row(market, row, mu)
            if mu <= 0:
                mu = float(cfg["league_mu"])
                mu_src.append("league_avg")
            else:
                mu_src.append("exp/fallback")
            mus.append(float(mu))

        df[f"{market}_ModelSrc"] = pd.Series(mu_src, index=df.index).astype("string")
        df[f"{market}_mu"] = pd.to_numeric(pd.Series(mus, index=df.index), errors="coerce")

        # Mainline model prob
        p_model = []
        for mu, lv in zip(df[f"{market}_mu"].tolist(), df[f"{market}_Line"].tolist()):
            lvf = _safe_float(lv)
            muf = _safe_float(mu)
            if lvf is None or muf is None:
                p_model.append(float("nan"))
                continue
            # enforce minimal SOG sanity
            if market == "SOG" and lvf < 1.5:
                p_model.append(float("nan"))
                continue
            p_model.append(_model_prob_over(market, float(muf), float(lvf)))

        df[f"{market}_p_model_over"] = pd.to_numeric(pd.Series(p_model, index=df.index), errors="coerce")
        df[f"{market}_Model%"] = (df[f"{market}_p_model_over"] * 100.0).round(1)

        # implied + EV
        df[f"{market}_p_imp_over"] = pd.to_numeric(df[f"{market}_Odds_Over"].apply(american_to_implied_prob), errors="coerce")

        evs = []
        for pm, od in zip(df[f"{market}_p_model_over"].tolist(), df[f"{market}_Odds_Over"].tolist()):
            if pm is None or pd.isna(pm) or od is None or pd.isna(od):
                evs.append(float("nan"))
                continue
            evs.append(_compute_ev_pct(float(pm), float(od)))
        df[f"{market}_EVpct_over"] = pd.to_numeric(pd.Series(evs, index=df.index), errors="coerce")
        df[f"{market}_EV%"] = df[f"{market}_EVpct_over"].round(1)

        # Alt lines: compute model/EV for _1..K if present
        for i in range(1, K + 1):
            lcol_bdl = f"BDL_{market}_Line_{i}"
            ocol_bdl = f"BDL_{market}_Odds_{i}"

            if lcol_bdl not in df.columns:
                continue

            lv_series = pd.to_numeric(df[lcol_bdl], errors="coerce")
            od_series = pd.to_numeric(df.get(ocol_bdl), errors="coerce")

            p_alt = []
            for mu, lv in zip(df[f"{market}_mu"].tolist(), lv_series.tolist()):
                lvf = _safe_float(lv)
                muf = _safe_float(mu)
                if lvf is None or muf is None:
                    p_alt.append(float("nan"))
                    continue
                if market == "SOG" and lvf < 1.5:
                    p_alt.append(float("nan"))
                    continue
                p_alt.append(_model_prob_over(market, float(muf), float(lvf)))

            df[f"{market}_p_model_over_{i}"] = pd.to_numeric(pd.Series(p_alt, index=df.index), errors="coerce")
            df[f"{market}_Model%_{i}"] = (df[f"{market}_p_model_over_{i}"] * 100.0).round(1)

            ev_alt = []
            for pm, od in zip(df[f"{market}_p_model_over_{i}"].tolist(), od_series.tolist()):
                if pm is None or pd.isna(pm) or od is None or pd.isna(od):
                    ev_alt.append(float("nan"))
                    continue
                ev_alt.append(_compute_ev_pct(float(pm), float(od)))
            df[f"{market}_EVpct_over_{i}"] = pd.to_numeric(pd.Series(ev_alt, index=df.index), errors="coerce")
            df[f"{market}_EV%_{i}"] = df[f"{market}_EVpct_over_{i}"].round(1)

    return df