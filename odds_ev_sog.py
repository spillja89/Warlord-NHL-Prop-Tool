from __future__ import annotations

import os
import math
import requests
from typing import Dict, Optional, Tuple, Iterable
import certifi


import pandas as pd

API_BASE = "https://api.balldontlie.io"
ENV_KEY = "BALLDONTLIE_API_KEY"

# ----------------------------
# helpers
# ----------------------------
def _clamp(x: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, x))

def _norm_name(x) -> str:
    if x is None:
        return ""
    s = str(x).lower().strip()
    s = s.replace(".", "").replace("'", "")
    s = " ".join(s.split())
    return s

def american_to_decimal(odds: Optional[float]) -> Optional[float]:
    if odds is None:
        return None
    try:
        o = float(odds)
    except Exception:
        return None
    if o == 0:
        return None
    if o > 0:
        return 1.0 + (o / 100.0)
    return 1.0 + (100.0 / abs(o))

def fair_prob_from_ou(over_american: Optional[float], under_american: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    dec_o = american_to_decimal(over_american)
    dec_u = american_to_decimal(under_american)
    if not dec_o or not dec_u:
        return None, None
    p_o_raw = 1.0 / dec_o
    p_u_raw = 1.0 / dec_u
    s = p_o_raw + p_u_raw
    if s <= 0:
        return None, None
    return p_o_raw / s, p_u_raw / s

def conf_to_prob_sog(conf_sog: Optional[float], matrix_sog: str = "", reg_heat_s: str = "", earned_green: bool = False) -> Optional[float]:
    if conf_sog is None or (isinstance(conf_sog, float) and math.isnan(conf_sog)):
        return None
    try:
        c = float(conf_sog)
    except Exception:
        return None

    # conservative mapping (you can tune later)
    p = 0.50 + ((c - 50.0) / 50.0) * 0.75

    if str(matrix_sog).lower() == "green":
        p += 0.02
    if str(reg_heat_s).upper() == "HOT":
        p += 0.02
    if earned_green:
        p += 0.01

    return _clamp(p, 0.02, 0.98)

def _auth_headers() -> Dict[str, str]:
    k = os.getenv(ENV_KEY, "").strip()
    if not k:
        raise RuntimeError(f"Missing env var {ENV_KEY}. Set it, reopen terminal, retry.")
    return {"Authorization": k}

# ----------------------------
# API fetchers (BallDontLie NHL)
# ----------------------------
def fetch_games_for_date(game_date: str) -> Dict[frozenset, int]:
    url = f"{API_BASE}/nhl/v1/games"
    r = requests.get(url, params={"dates[]": game_date, "per_page": 100}, headers=_auth_headers(), timeout=25)
    r.raise_for_status()
    data = (r.json() or {}).get("data", []) or []

    out: Dict[frozenset, int] = {}
    for g in data:
        gid = g.get("id")
        home = (g.get("home_team") or {}).get("tricode")
        # Some responses use visitor_team; some use away_team.
        away_obj = g.get("visitor_team") or g.get("away_team") or {}
        away = (away_obj or {}).get("tricode")
        if gid and home and away:
            out[frozenset([str(home).upper(), str(away).upper()])] = int(gid)
    return out

def fetch_sog_props_for_game(
    game_id: int,
    vendors: Optional[Tuple[str, ...]] = None,
) -> pd.DataFrame:
    import certifi

    url = f"{API_BASE}/nhl/v1/odds/player_props"
    params = {
        "game_id": int(game_id),
        "prop_type": "shots_on_goal",
        "per_page": 100,
    }
    # If vendors is None, do not filter (BDL coverage varies heavily by slate/book).
    if vendors:
        params["vendors[]"] = list(vendors)

    r = requests.get(
        url,
        params=params,
        headers=_auth_headers(),
        timeout=30,
        verify=certifi.where(),
    )
    r.raise_for_status()
    items = (r.json() or {}).get("data", []) or []

    def _pick_odds(d: dict) -> Optional[float]:
        """Return an American odds value from a dict with varying key names."""
        if not isinstance(d, dict):
            return None
        for k in ("odds", "price", "american_odds", "odds_american", "oddsAmerican", "americanOdds"):
            v = d.get(k)
            if v is not None and str(v).strip() != "":
                return v
        return None

    rows = []
    for it in items:
        # market can be a dict ("market") or a list ("markets") depending on vendor/feed
        markets = []
        if isinstance(it.get("market"), dict):
            markets = [it.get("market")]
        elif isinstance(it.get("markets"), list):
            markets = [x for x in it.get("markets") if isinstance(x, dict)]

        if not markets:
            continue

        # pick the first over_under market we can parse
        m = next((x for x in markets if str(x.get("type", "")).lower() == "over_under"), None)
        if not m:
            continue

        # --- BDL market shape variants ---
        # Variant A: market has over/under dicts:
        #   market: { type:'over_under', over:{odds:-110}, under:{odds:-110} }
        over_odds = None
        under_odds = None

        # Variant A: market has over/under dicts
        over_odds = _pick_odds(m.get("over"))
        under_odds = _pick_odds(m.get("under"))

        # Variant B: market has outcomes list
        if (over_odds is None or under_odds is None) and isinstance(m.get("outcomes"), list):
            for outc in m.get("outcomes"):
                if not isinstance(outc, dict):
                    continue
                name = str(outc.get("name") or outc.get("type") or "").lower()
                if "over" in name and over_odds is None:
                    over_odds = _pick_odds(outc)
                if "under" in name and under_odds is None:
                    under_odds = _pick_odds(outc)

        # Variant C: odds live on the row
        if over_odds is None:
            over_odds = it.get("over_odds") or it.get("over_price")
        if under_odds is None:
            under_odds = it.get("under_odds") or it.get("under_price")

        # line can live on market or row
        line_value = it.get("line_value")
        if line_value is None:
            line_value = m.get("line_value") or m.get("line") or it.get("line")

        rows.append({
            "game_id": it.get("game_id"),
            "player_name": it.get("player_name"),
            "player_id": it.get("player_id"),
            "vendor": it.get("vendor"),
            "line_value": line_value,
            "over_odds": over_odds,
            "under_odds": under_odds,
            "updated_at": it.get("updated_at"),
        })

    return pd.DataFrame(rows)

def fetch_players_by_ids(player_ids: Iterable[int]) -> Dict[int, str]:
    """
    BallDontLie NHL odds rows can omit player_name but include player_id.
    This fetches names for IDs so we can join reliably.
    """
    ids = sorted({int(x) for x in player_ids if x is not None})
    if not ids:
        return {}

    url = f"{API_BASE}/nhl/v1/players"
    out: Dict[int, str] = {}

    # Keep this conservative to avoid 400s from too-long querystrings.
    CHUNK = 40
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i:i + CHUNK]
        params = {"per_page": 100, "player_ids[]": chunk}
        r = requests.get(url, params=params, headers=_auth_headers(), timeout=25)
        r.raise_for_status()
        data = (r.json() or {}).get("data", []) or []
        for p in data:
            pid = p.get("id")
            name = p.get("full_name") or ""
            if pid is not None and name:
                out[int(pid)] = str(name)
    return out


# ----------------------------
# main merge
# ----------------------------
DEFAULT_SOG_COLS: Tuple[str, ...] = (
    "SOG_Line",
    "SOG_Book",
    "SOG_Odds_Over",
    "SOG_Odds_Under",
    "SOG_p_model_over",
    "SOG_p_imp_over",
    "SOG_p_mkt_over_fair",
    "SOG_EV_over",
    "SOG_EVpct_over",
)

def _ensure_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df

def merge_sog_ev(
    df: pd.DataFrame,
    game_date: str,
    vendors: Optional[Tuple[str, ...]] = None,
    debug: bool = False,
) -> pd.DataFrame:
    # Always make columns exist so downstream / Streamlit doesn't depend on a successful pull
    df = df.copy()
    df = _ensure_cols(df, DEFAULT_SOG_COLS)

    if "Player" not in df.columns or "Team" not in df.columns or "Opp" not in df.columns:
        if debug:
            print("[odds/ev] missing Player/Team/Opp columns; leaving EV cols empty")
        return df

    games = fetch_games_for_date(game_date)
    if debug:
        print(f"[odds/ev] games found for {game_date}: {len(games)}")

    # map each row to game id
    df["_game_id_bdl"] = df.apply(lambda r: games.get(frozenset([str(r["Team"]).upper(), str(r["Opp"]).upper()])), axis=1)
    df["_pname_norm"] = df["Player"].apply(_norm_name)

    gids = sorted(set(int(x) for x in df["_game_id_bdl"].dropna().unique()))
    if not gids:
        if debug:
            print("[odds/ev] no game_ids mapped from Team/Opp; leaving EV cols empty")
        df.drop(columns=["_game_id_bdl","_pname_norm"], errors="ignore", inplace=True)
        return df

    prop_frames = []
    for gid in gids:
        try:
            pf = fetch_sog_props_for_game(gid, vendors=vendors)
            if debug:
                print(f"[odds/ev] props fetched for game_id={gid}: {len(pf)}")
            prop_frames.append(pf)
        except Exception as e:
            if debug:
                print(f"[odds/ev] props fetch failed for game_id={gid}: {e}")
            continue

    if not prop_frames:
        if debug:
            print("[odds/ev] no props fetched; leaving EV cols empty")
        df.drop(columns=["_game_id_bdl","_pname_norm"], errors="ignore", inplace=True)
        return df

    props = pd.concat(prop_frames, ignore_index=True)
    if props.empty:
        if debug:
            print("[odds/ev] props empty after concat; leaving EV cols empty")
        df.drop(columns=["_game_id_bdl", "_pname_norm"], errors="ignore", inplace=True)
        return df

    # convert types (must be OUTSIDE the props.empty return)
    props["line_value"] = pd.to_numeric(props["line_value"], errors="coerce")

    # ----------------------------
    # Ensure player_name exists (BDL may omit it)
    # ----------------------------
    if "player_name" not in props.columns:
        props["player_name"] = None

    # Backfill missing player_name using player_id (BDL sometimes omits names)
    need_name = props["player_name"].isna() | (props["player_name"].astype(str).str.strip() == "")
    if need_name.any() and "player_id" in props.columns:
        ids_need = (
            props.loc[need_name, "player_id"].dropna().astype(int).unique().tolist()
        )
        if ids_need:
            id_to_name = fetch_players_by_ids(ids_need)
            props.loc[need_name, "player_name"] = props.loc[need_name, "player_id"].map(id_to_name)

    props["_pname_norm"] = props["player_name"].apply(_norm_name)


    props["over_dec"] = props["over_odds"].apply(american_to_decimal)
    props["p_over_imp"] = props["over_dec"].apply(lambda d: (1.0 / d) if d else None)
    props["p_over_fair"] = props.apply(lambda r: fair_prob_from_ou(r.get("over_odds"), r.get("under_odds"))[0], axis=1)

    # pick best OVER price (max decimal) per (game_id, player)
    props = props.sort_values(by=["game_id", "_pname_norm", "over_dec"], ascending=[True, True, False])
    best = props.drop_duplicates(subset=["game_id", "_pname_norm"], keep="first").copy()

    best = best.rename(columns={
        "line_value": "SOG_Line",
        "vendor": "SOG_Book",
        "over_odds": "SOG_Odds_Over",
        "under_odds": "SOG_Odds_Under",
        "p_over_imp": "SOG_p_imp_over",
        "p_over_fair": "SOG_p_mkt_over_fair",
    })

    df = df.merge(
        best[["game_id", "_pname_norm", "SOG_Line", "SOG_Book", "SOG_Odds_Over", "SOG_Odds_Under", "SOG_p_imp_over", "SOG_p_mkt_over_fair"]],
        left_on=["_game_id_bdl", "_pname_norm"],
        right_on=["game_id", "_pname_norm"],
        how="left",
    ).drop(columns=["game_id"], errors="ignore")

    # model p + EV
    if "Conf_SOG" in df.columns:
        df["SOG_p_model_over"] = df.apply(
            lambda r: conf_to_prob_sog(
                r.get("Conf_SOG"),
                matrix_sog=str(r.get("Matrix_SOG", "")),
                reg_heat_s=str(r.get("Reg_Heat_S", r.get("Reg_Heat_SOG", ""))),
                earned_green=bool(r.get("Earned_Green_SOG", False) or r.get("Green_SOG", False)),
            ),
            axis=1,
        )
        df["SOG_over_dec"] = df["SOG_Odds_Over"].apply(american_to_decimal)
        df["SOG_EV_over"] = df.apply(
            lambda r: (r["SOG_p_model_over"] * r["SOG_over_dec"] - 1.0)
            if (r.get("SOG_p_model_over") is not None and r.get("SOG_over_dec") is not None)
            else None,
            axis=1,
        )
        df["SOG_EVpct_over"] = df["SOG_EV_over"].apply(lambda x: None if x is None else round(100.0 * float(x), 2))

    df.drop(columns=["_game_id_bdl", "_pname_norm", "SOG_over_dec"], errors="ignore", inplace=True)
    return df
