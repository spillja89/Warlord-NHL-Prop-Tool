from __future__ import annotations

import os
import time
import json
import math
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from typing import Optional, Dict, List, Any, Tuple
from statistics import median
from concurrent.futures import ThreadPoolExecutor, as_completed
from nhl_edge_v2_upgrade_fixed import apply_v2_to_existing_edges


import pandas as pd
import requests

# ============================
# CONFIG (tune here)
# ============================
OUTPUT_DIR = "output"
CACHE_DIR = "cache"

REQUEST_TIMEOUT_SEC = 25
HTTP_SLEEP_SEC = 0.02  # tiny courtesy delay; we also parallelize

# Your local timezone for "today" (fixes missing late games)
LOCAL_TZ = "America/Chicago"

# Parallelism for NHL game logs (bigger -> faster, but don't go crazy)
MAX_WORKERS = 10

# Candidate pool (we only fetch NHL last-10 logs for these)
PROCESS_MIN_PCT = 70  # lower than before because last10 is stable

# Matrix thresholds (anti-skew uses MEDIAN10 SOG)
SOG_MED10_GREEN_FWD = 3.5
SOG_MED10_GREEN_DEF = 2.3

SOG_MED10_YELLOW_FWD = 3.0
SOG_MED10_YELLOW_DEF = 1.9

# Goalie matchup thresholds (team goalie "starter-ish by GP")
MIN_GOALIE_GP = 5
FAV_SV = 0.905
FAV_GAA = 3.10
AVOID_SV = 0.920
AVOID_GAA = 2.60

# Goal tagging (Anytime Goal candidates)
# Two-path system: volume scorers OR quality scorers
GOAL_IXG_VOLUME = 90
GOAL_SOG_MED10_VOLUME_FWD = 4.0
GOAL_SOG_MED10_VOLUME_DEF = 3.0

GOAL_IXG_QUALITY = 85
GOAL_SOG_MED10_QUALITY_FWD = 3.0
GOAL_SOG_MED10_QUALITY_DEF = 2.2

# Terminal output sizes
TOP_GREEN_LIST = 30
TOP_YELLOW_LIST = 25
TOP_GOAL_LIST = 15


# ============================
# Helpers
# ============================
def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        )
    })
    return s


def http_get_json(sess: requests.Session, url: str) -> Any:
    r = sess.get(url, timeout=REQUEST_TIMEOUT_SEC)
    r.raise_for_status()
    return r.json()


def http_get_text(sess: requests.Session, url: str) -> str:
    r = sess.get(url, timeout=REQUEST_TIMEOUT_SEC)
    r.raise_for_status()
    return r.text


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def is_defense(pos: str) -> bool:
    return (pos or "").upper().strip() in {"D", "LD", "RD"}


def is_goalie(pos: str) -> bool:
    return (pos or "").upper().strip() == "G"


def fmt_x_over_5(total: Any) -> str:
    """
    Display last-5 totals as X/5.
    Handles None and NaN safely.
    """
    if total is None:
        return ""
    try:
        if isinstance(total, float) and math.isnan(total):
            return ""
        return f"{int(total)}/5"
    except Exception:
        return ""


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return int(x)
    except Exception:
        return None


def parse_utc_to_local_date(start_time_utc: str, tz_name: str) -> Optional[date]:
    """
    Convert NHL startTimeUTC (ISO string) -> local calendar date in tz_name.
    Uses zoneinfo (Python 3.9+).
    """
    if not start_time_utc or "T" not in start_time_utc:
        return None
    try:
        # Example: "2026-01-05T00:00:00Z"
        dt_utc = datetime.fromisoformat(start_time_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
        try:
            from zoneinfo import ZoneInfo
            dt_local = dt_utc.astimezone(ZoneInfo(tz_name))
        except Exception:
            dt_local = dt_utc
        return dt_local.date()
    except Exception:
        return None


# ============================
# NHL schedule (TODAY) — api-web.nhle.com
# Robust for late games (UTC rollover)
# ============================
def nhl_schedule_today(sess: requests.Session, today_local: date) -> List[Dict[str, str]]:
    """
    Fetch schedule for today_local and tomorrow_local (because schedule uses UTC start times).
    Keep games whose startTimeUTC converted to LOCAL_TZ falls on today_local.
    """
    def fetch_day(d: date) -> Any:
        url = f"https://api-web.nhle.com/v1/schedule/{d.isoformat()}"
        return http_get_json(sess, url)

    data_today = fetch_day(today_local)
    data_tom = fetch_day(today_local + timedelta(days=1))

    out: List[Dict[str, str]] = []

    def consume(data: Any) -> None:
        for day in data.get("gameWeek", []):
            for g in day.get("games", []):
                away = (g.get("awayTeam") or {}).get("abbrev")
                home = (g.get("homeTeam") or {}).get("abbrev")
                start_utc = str(g.get("startTimeUTC", ""))
                local_d = parse_utc_to_local_date(start_utc, LOCAL_TZ)
                if local_d == today_local and away and home:
                    out.append({"away": away, "home": home})

    consume(data_today)
    consume(data_tom)

    # de-dup
    seen = set()
    deduped: List[Dict[str, str]] = []
    for g in out:
        k = (g["away"], g["home"])
        if k not in seen:
            seen.add(k)
            deduped.append(g)
    return deduped


# ============================
# MoneyPuck (seasonSummary)
# ============================
def current_season_start_year(today: date) -> int:
    return today.year if today.month >= 7 else today.year - 1


def moneypuck_skaters_url(season_start_year: int) -> str:
    return f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season_start_year}/regular/skaters.csv"


def moneypuck_goalies_url(season_start_year: int) -> str:
    return f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{season_start_year}/regular/goalies.csv"


def load_moneypuck_csv(sess: requests.Session, url: str) -> pd.DataFrame:
    csv_text = http_get_text(sess, url)
    if not csv_text.strip():
        raise RuntimeError(f"Empty CSV from {url}")
    df = pd.read_csv(StringIO(csv_text))
    df.columns = df.columns.str.strip()
    return df


def load_moneypuck_skaters(sess: requests.Session) -> pd.DataFrame:
    start = current_season_start_year(date.today())
    last_err = None
    for y in (start, start - 1):
        url = moneypuck_skaters_url(y)
        try:
            return load_moneypuck_csv(sess, url)
        except Exception as e:
            last_err = f"{url} -> {type(e).__name__}: {e}"
    raise RuntimeError(f"Could not download MoneyPuck skaters.csv. Last error: {last_err}")


def load_moneypuck_goalies(sess: requests.Session) -> pd.DataFrame:
    start = current_season_start_year(date.today())
    last_err = None
    for y in (start, start - 1):
        url = moneypuck_goalies_url(y)
        try:
            return load_moneypuck_csv(sess, url)
        except Exception as e:
            last_err = f"{url} -> {type(e).__name__}: {e}"
    raise RuntimeError(f"Could not download MoneyPuck goalies.csv. Last error: {last_err}")


def _find_col(df: pd.DataFrame, name: str) -> Optional[str]:
    m = {c.lower(): c for c in df.columns}
    return m.get(name.lower())


def normalize_skaters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal columns only (noise-free):
    - playerId, name, team, position, situation, games_played, icetime
    - iXG proxy: I_F_xGoals
    - iXA proxy: I_F_primaryAssists + I_F_secondaryAssists
    """
    required = {
        "playerId": _find_col(df, "playerId"),
        "Player": _find_col(df, "name"),
        "Team": _find_col(df, "team"),
        "Pos": _find_col(df, "position"),
        "situation": _find_col(df, "situation"),
        "games_played": _find_col(df, "games_played"),
        "icetime": _find_col(df, "icetime"),
        "I_F_xGoals": _find_col(df, "I_F_xGoals"),
        "I_F_primaryAssists": _find_col(df, "I_F_primaryAssists"),
        "I_F_secondaryAssists": _find_col(df, "I_F_secondaryAssists"),
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise KeyError(f"Skaters CSV missing required columns: {missing}")

    out = df[[v for v in required.values()]].copy()
    out.columns = list(required.keys())

    out["situation"] = out["situation"].astype(str).str.lower()
    out = out[out["situation"].isin(["all", "all situations", "all_situations", "all-situations"])].copy()

    out["playerId"] = pd.to_numeric(out["playerId"], errors="coerce")
    out["games_played"] = pd.to_numeric(out["games_played"], errors="coerce")
    out["icetime"] = pd.to_numeric(out["icetime"], errors="coerce")

    out["iXG_raw"] = pd.to_numeric(out["I_F_xGoals"], errors="coerce")
    out["iXA_raw"] = (
        pd.to_numeric(out["I_F_primaryAssists"], errors="coerce").fillna(0)
        + pd.to_numeric(out["I_F_secondaryAssists"], errors="coerce").fillna(0)
    )

    out = out[~out["Pos"].astype(str).apply(is_goalie)].copy()
    return out


def add_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["iXG_pct"] = df["iXG_raw"].rank(pct=True) * 100.0
    df["iXA_pct"] = df["iXA_raw"].rank(pct=True) * 100.0
    return df


def infer_pp_role_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy PP role via TOI rank on team:
      - Top 8 TOI => PP1 proxy
      - Next 8 => PP2 proxy
    """
    df = df.copy()
    df["toi_rank_team"] = df.groupby("Team")["icetime"].rank(ascending=False, method="first")

    def role(r: float) -> str:
        try:
            r = float(r)
        except Exception:
            return ""
        if r <= 8:
            return "PP1"
        if r <= 16:
            return "PP2"
        return ""

    df["PP"] = df["toi_rank_team"].apply(role)
    return df


# ============================
# MoneyPuck goalies -> SV%, GAA, GP
# ============================
def normalize_goalies(df: pd.DataFrame) -> pd.DataFrame:
    col_name = _find_col(df, "name")
    col_team = _find_col(df, "team")
    col_sit = _find_col(df, "situation")
    col_gp = _find_col(df, "games_played")
    col_it = _find_col(df, "icetime")
    col_ongoal = _find_col(df, "ongoal")
    col_goals = _find_col(df, "goals")

    needed = {
        "name": col_name, "team": col_team, "situation": col_sit,
        "games_played": col_gp, "icetime": col_it, "ongoal": col_ongoal, "goals": col_goals
    }
    missing = [k for k, v in needed.items() if v is None]
    if missing:
        raise KeyError(f"Goalies CSV missing columns: {missing}")

    out = df[[col_name, col_team, col_sit, col_gp, col_it, col_ongoal, col_goals]].copy()
    out.columns = ["Goalie", "Team", "situation", "GP", "icetime", "SOG_Against", "GA"]

    out["situation"] = out["situation"].astype(str).str.lower()
    out = out[out["situation"].isin(["all", "all situations", "all_situations", "all-situations"])].copy()

    out["GP"] = pd.to_numeric(out["GP"], errors="coerce")
    out["icetime"] = pd.to_numeric(out["icetime"], errors="coerce")
    out["SOG_Against"] = pd.to_numeric(out["SOG_Against"], errors="coerce")
    out["GA"] = pd.to_numeric(out["GA"], errors="coerce")

    out["SV"] = (out["SOG_Against"] - out["GA"]) / out["SOG_Against"]
    out.loc[out["SOG_Against"] <= 0, "SV"] = pd.NA

    out["GAA"] = pd.NA
    sec_mask = out["icetime"] > 10000
    min_mask = ~sec_mask
    out.loc[min_mask & (out["icetime"] > 0), "GAA"] = out.loc[min_mask, "GA"] * 60.0 / out.loc[min_mask, "icetime"]
    out.loc[sec_mask & (out["icetime"] > 0), "GAA"] = out.loc[sec_mask, "GA"] * 3600.0 / out.loc[sec_mask, "icetime"]

    return out


def build_team_goalie_map(goalies_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    No confirmed starters: pick "starter-ish" = highest GP for each team.
    """
    m: Dict[str, Dict[str, Any]] = {}
    for team, grp in goalies_df.groupby("Team"):
        g2 = grp.dropna(subset=["GP"]).sort_values("GP", ascending=False)
        if g2.empty:
            continue
        top = g2.iloc[0]
        m[team] = {
            "Goalie": str(top["Goalie"]),
            "GP": safe_float(top["GP"]),
            "SV": safe_float(top["SV"]),
            "GAA": safe_float(top["GAA"]),
        }
    return m


def matchup_flag(opp_gp: Optional[float], opp_sv: Optional[float], opp_gaa: Optional[float]) -> str:
    if opp_gp is None or opp_gp < MIN_GOALIE_GP:
        return "Neutral"
    if opp_sv is not None and opp_sv <= FAV_SV:
        return "Favorable"
    if opp_gaa is not None and opp_gaa >= FAV_GAA:
        return "Favorable"
    if (opp_sv is not None and opp_sv >= AVOID_SV) and (opp_gaa is not None and opp_gaa <= AVOID_GAA):
        return "Avoid"
    return "Neutral"


# ============================
# NHL player game logs (anti-skew core)
# ============================
def cache_path_today(today: date) -> str:
    return os.path.join(CACHE_DIR, f"nhle_playerlogs_{today.isoformat()}.json")


def load_cache(today: date) -> Dict[str, Any]:
    p = cache_path_today(today)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(today: date, cache: Dict[str, Any]) -> None:
    p = cache_path_today(today)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def nhle_player_gamelog_now(sess: requests.Session, player_id: int) -> Optional[Dict[str, Any]]:
    url = f"https://api-web.nhle.com/v1/player/{player_id}/game-log/now"
    try:
        return http_get_json(sess, url)
    except Exception:
        return None


def _extract_game_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    for k in ("gameLog", "gameLogs", "games", "gamelog", "game-log"):
        v = payload.get(k)
        if isinstance(v, list):
            return v
    for v in payload.values():
        if isinstance(v, dict):
            for kk in ("gameLog", "gameLogs", "games"):
                vv = v.get(kk)
                if isinstance(vv, list):
                    return vv
    return []


def _pick_stat(d: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[int]:
    for k in keys:
        if k in d:
            return safe_int(d.get(k))
    for nk in ("playerGameStats", "stats", "skaterStats", "summary"):
        v = d.get(nk)
        if isinstance(v, dict):
            for k in keys:
                if k in v:
                    return safe_int(v.get(k))
    return None


def compute_lastN_features(payload: Dict[str, Any], n10: int = 10, n5: int = 5) -> Dict[str, Any]:
    rows = _extract_game_rows(payload)

    shots: List[int] = []
    goals: List[int] = []
    assists: List[int] = []

    for r in rows:
        sog = _pick_stat(r, ("shotsOnGoal", "sog", "shots_on_goal", "shots"))
        g = _pick_stat(r, ("goals", "g"))
        a = _pick_stat(r, ("assists", "a"))

        if sog is None and g is None and a is None:
            continue

        shots.append(int(sog) if sog is not None else 0)
        goals.append(int(g) if g is not None else 0)
        assists.append(int(a) if a is not None else 0)

    v10_shots = shots[:n10]
    v5_shots = shots[:n5]
    v5_goals = goals[:n5]
    v5_assists = assists[:n5]

    def trimmed_mean(vals: List[int], trim_each_side: int = 1) -> Optional[float]:
        if not vals:
            return None
        vv = sorted([float(x) for x in vals])
        if len(vv) <= 2 * trim_each_side:
            return sum(vv) / len(vv)
        vv = vv[trim_each_side: len(vv) - trim_each_side]
        return sum(vv) / len(vv)

    med10 = float(median(v10_shots)) if len(v10_shots) >= 3 else None
    trim10 = trimmed_mean(v10_shots, 1) if len(v10_shots) >= 5 else None
    avg5 = (sum(v5_shots) / len(v5_shots)) if v5_shots else None

    g5_total = sum(v5_goals) if v5_goals else None
    a5_total = sum(v5_assists) if v5_assists else None

    return {
        "Median10_SOG": med10,
        "TrimMean10_SOG": trim10,
        "Avg5_SOG": avg5,
        "G5_total": g5_total,
        "A5_total": a5_total,
        "N_games_found": len(shots),
    }


# ============================
# Matrix + Props + Confidence
# ============================
def sog_med10_threshold(pos: str) -> float:
    return SOG_MED10_GREEN_DEF if is_defense(pos) else SOG_MED10_GREEN_FWD


def sog_med10_yellow_threshold(pos: str) -> float:
    return SOG_MED10_YELLOW_DEF if is_defense(pos) else SOG_MED10_YELLOW_FWD


def assign_matrix(ixg_pct: float, ixa_pct: float, med10: Optional[float], pos: str) -> str:
    elite = (ixg_pct >= 80) or (ixa_pct >= 80)
    if not elite:
        return "Red"
    if med10 is None:
        return "Yellow"
    if med10 >= sog_med10_threshold(pos):
        return "Green"
    if med10 >= sog_med10_yellow_threshold(pos):
        return "Yellow"
    return "Red"


def recommend_prop(ixg_pct: float, ixa_pct: float) -> str:
    return "SOG" if ixg_pct >= ixa_pct else "Point"


def pp_bonus(pp: str) -> float:
    if pp == "PP1":
        return 8.0
    if pp == "PP2":
        return 3.0
    return 0.0


def matchup_bonus(mu: str) -> float:
    if mu == "Favorable":
        return 6.0
    if mu == "Avoid":
        return -10.0
    return 0.0


def sog_scale_for_conf(med10: Optional[float], trim10: Optional[float], pos: str) -> float:
    v = trim10 if trim10 is not None else med10
    if v is None:
        return 50.0
    if is_defense(pos):
        return clamp((v - 1.0) / (3.5 - 1.0) * 100.0)
    return clamp((v - 2.0) / (5.0 - 2.0) * 100.0)


def last5_total_score(total: Optional[int]) -> float:
    if total is None:
        return 50.0
    return clamp((total / 5.0) * 100.0)


def confidence_sog(ixg_pct: float, med10: Optional[float], trim10: Optional[float], avg5: Optional[float], pos: str, pp: str, mu: str) -> int:
    shots_score = sog_scale_for_conf(med10, trim10, pos)
    trend_nudge = 0.0
    if avg5 is not None and med10 is not None:
        if avg5 >= med10 + 0.5:
            trend_nudge = 4.0
        elif avg5 <= med10 - 0.5:
            trend_nudge = -4.0

    base = 0.55 * ixg_pct + 0.35 * shots_score + 0.10 * clamp(shots_score + trend_nudge)
    base += pp_bonus(pp)
    base += matchup_bonus(mu)
    return int(round(clamp(base)))


def confidence_point(ixa_pct: float, a5_total: Optional[int], pp: str, mu: str) -> int:
    base = 0.70 * ixa_pct + 0.30 * last5_total_score(a5_total)
    base += pp_bonus(pp)
    base += (matchup_bonus(mu) * 0.5)
    return int(round(clamp(base)))


def confidence_goal(ixg_pct: float, med10: Optional[float], trim10: Optional[float], g5_total: Optional[int], pos: str, pp: str, mu: str) -> int:
    shots_score = sog_scale_for_conf(med10, trim10, pos)
    base = 0.55 * ixg_pct + 0.30 * shots_score + 0.15 * last5_total_score(g5_total)
    base += (pp_bonus(pp) * 1.2)
    base += (matchup_bonus(mu) * 1.2)
    return int(round(clamp(base)))


def goal_tag(ixg_pct: float, med10: Optional[float], pos: str, pp: str, mu: str) -> str:
    if mu == "Avoid":
        return "No"
    if pp != "PP1":
        return "No"
    if med10 is None:
        return "No"

    if is_defense(pos):
        volume_ok = (ixg_pct >= GOAL_IXG_VOLUME and med10 >= GOAL_SOG_MED10_VOLUME_DEF)
        quality_ok = (ixg_pct >= GOAL_IXG_QUALITY and med10 >= GOAL_SOG_MED10_QUALITY_DEF and mu == "Favorable")
        return "Yes" if (volume_ok or quality_ok) else "No"

    volume_ok = (ixg_pct >= GOAL_IXG_VOLUME and med10 >= GOAL_SOG_MED10_VOLUME_FWD)
    quality_ok = (ixg_pct >= GOAL_IXG_QUALITY and med10 >= GOAL_SOG_MED10_QUALITY_FWD and mu == "Favorable")
    return "Yes" if (volume_ok or quality_ok) else "No"


# ============================
# MAIN
# ============================
def main() -> None:
    today_local = date.today()
    ensure_dirs()
    sess = http_session()

    print(f"\\nNHL EDGE TOOL — NHL API Last10 (Median + TrimMean) — {today_local.isoformat()}\\n")

    # 1) Today’s slate (robust)
    games = nhl_schedule_today(sess, today_local)
    if not games:
        print("No games found for today from api-web.nhle.com (unexpected).")
        return

    print("Matchups:")
    for g in games:
        print(f"  {g['away']}@{g['home']}")
    print("")

    teams_playing = set()
    game_map: Dict[str, str] = {}
    opp_map: Dict[str, str] = {}

    for g in games:
        away, home = g["away"], g["home"]
        matchup = f"{away}@{home}"
        teams_playing.add(away)
        teams_playing.add(home)
        game_map[away] = matchup
        game_map[home] = matchup
        opp_map[away] = home
        opp_map[home] = away

    print(f"Games today: {len(games)}")
    print(f"Teams today: {', '.join(sorted(teams_playing))}\\n")

    # 2) MoneyPuck skaters
    sk_raw = load_moneypuck_skaters(sess)
    sk = normalize_skaters(sk_raw)
    sk = add_percentiles(sk)
    sk = infer_pp_role_proxy(sk)

    # only teams playing today
    sk = sk[sk["Team"].isin(teams_playing)].copy()
    if sk.empty:
        print("No skaters matched today's teams. Possible team-code mismatch.")
        return

    # 3) MoneyPuck goalies
    g_raw = load_moneypuck_goalies(sess)
    gdf = normalize_goalies(g_raw)
    team_goalie = build_team_goalie_map(gdf)

    def add_matchup(row: pd.Series) -> pd.Series:
        team = row["Team"]
        opp = opp_map.get(team, "")
        row["Game"] = game_map.get(team, "")
        row["Opp"] = opp

        og = team_goalie.get(opp, {})
        row["Opp_Goalie"] = og.get("Goalie", "")
        row["Opp_GP"] = og.get("GP", None)
        row["Opp_SV"] = og.get("SV", None)
        row["Opp_GAA"] = og.get("GAA", None)
        row["Matchup"] = matchup_flag(row["Opp_GP"], row["Opp_SV"], row["Opp_GAA"])
        return row

    sk = sk.apply(add_matchup, axis=1)

    # 4) Candidate pool for NHL game logs
    sk["TopPct"] = sk[["iXG_pct", "iXA_pct"]].max(axis=1)
    candidates = sk[(sk["TopPct"] >= PROCESS_MIN_PCT) & (sk["Matchup"] != "Avoid")].copy()
    candidates = candidates.sort_values("TopPct", ascending=False).reset_index(drop=True)

    cache = load_cache(today_local)

    cand_ids: List[int] = []
    for pid in candidates["playerId"].dropna().tolist():
        try:
            cand_ids.append(int(pid))
        except Exception:
            continue
    cand_ids = sorted(list(set(cand_ids)))

    print(f"Fetching NHL game logs (now) for {len(cand_ids)} players (TopPct>= {PROCESS_MIN_PCT}, matchup!=Avoid)\\n")

    def fetch_one(pid: int) -> Tuple[int, Optional[Dict[str, Any]]]:
        key = str(pid)
        if key in cache:
            return pid, cache[key]
        data = nhle_player_gamelog_now(sess, pid)
        time.sleep(HTTP_SLEEP_SEC)
        if data is None:
            return pid, None
        cache[key] = data
        return pid, data

    fetched: Dict[int, Optional[Dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_one, pid): pid for pid in cand_ids}
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                _, data = fut.result()
                fetched[pid] = data
            except Exception:
                fetched[pid] = None

    save_cache(today_local, cache)

    feats_rows: List[Dict[str, Any]] = []
    for pid in cand_ids:
        payload = fetched.get(pid)
        if payload is None:
            feats_rows.append({
                "playerId": pid,
                "Median10_SOG": None,
                "TrimMean10_SOG": None,
                "Avg5_SOG": None,
                "G5_total": None,
                "A5_total": None,
                "N_games_found": 0,
            })
            continue
        feats = compute_lastN_features(payload, 10, 5)
        feats_rows.append({"playerId": pid, **feats})

    feats_df = pd.DataFrame(feats_rows)
    sk = sk.merge(feats_df, on="playerId", how="left")

    # 5) Matrix + Prop + Confidence
    sk["Matrix"] = sk.apply(lambda r: assign_matrix(
        float(r["iXG_pct"]),
        float(r["iXA_pct"]),
        safe_float(r.get("Median10_SOG")),
        str(r["Pos"])
    ), axis=1)

    sk.loc[(sk["Matrix"] == "Green") & (sk["Matchup"] == "Avoid"), "Matrix"] = "Yellow"

    sk["Prop"] = ""
    green_mask = sk["Matrix"] == "Green"
    sk.loc[green_mask, "Prop"] = sk.loc[green_mask].apply(
        lambda r: recommend_prop(float(r["iXG_pct"]), float(r["iXA_pct"])),
        axis=1
    )

    sk["Conf_SOG"] = sk.apply(lambda r: confidence_sog(
        float(r["iXG_pct"]),
        safe_float(r.get("Median10_SOG")),
        safe_float(r.get("TrimMean10_SOG")),
        safe_float(r.get("Avg5_SOG")),
        str(r["Pos"]),
        str(r.get("PP", "")),
        str(r.get("Matchup", "Neutral")),
    ), axis=1)

    sk["Conf_Point"] = sk.apply(lambda r: confidence_point(
        float(r["iXA_pct"]),
        safe_int(r.get("A5_total")),
        str(r.get("PP", "")),
        str(r.get("Matchup", "Neutral")),
    ), axis=1)

    sk["Conf_Goal"] = sk.apply(lambda r: confidence_goal(
        float(r["iXG_pct"]),
        safe_float(r.get("Median10_SOG")),
        safe_float(r.get("TrimMean10_SOG")),
        safe_int(r.get("G5_total")),
        str(r["Pos"]),
        str(r.get("PP", "")),
        str(r.get("Matchup", "Neutral")),
    ), axis=1)

    sk["Goal_Tag"] = sk.apply(lambda r: goal_tag(
        float(r["iXG_pct"]),
        safe_float(r.get("Median10_SOG")),
        str(r["Pos"]),
        str(r.get("PP", "")),
        str(r.get("Matchup", "Neutral")),
    ), axis=1)

    def pick_conf(r: pd.Series) -> Any:
        p = str(r.get("Prop", ""))
        if p == "SOG":
            return int(r["Conf_SOG"])
        if p == "Point":
            return int(r["Conf_Point"])
        return ""

    sk["Confidence"] = sk.apply(pick_conf, axis=1)

    # ============================
    # NHL EDGE v2.0 UPGRADE HOOK
    # (plumbing now; replace placeholders with real i5v5 inputs when ready)
    # ============================
    sk["ixG_pct"] = sk["iXG_pct"]
    sk["ixA_pct"] = sk["iXA_pct"]

    for col in [
        "i5v5_oiGF60",
        "i5v5_shotAssists60",
        "i5v5_points60",
        "i5v5_iCF60",
        "i5v5_HDCA60",
        "i5v5_xGA60",
        "i5v5_slotShotsAgainst60",
        "i5v5_DZturnovers60",
    ]:
        if col not in sk.columns:
            sk[col] = pd.NA

    sk = apply_v2_to_existing_edges(
        sk,
        points_edge_col=None,
        assists_edge_col=None,
        goals_edge_col=None,
        sog_edge_col=None,
        position_col="Pos"
    )

    # 6) Output tracker (minimal columns, readable)
    tracker = pd.DataFrame({
        "Date": today_local.isoformat(),
        "Game": sk["Game"],
        "Player": sk["Player"],
        "Team": sk["Team"],
        "Pos": sk["Pos"],
        "iXG%": sk["iXG_pct"].round(1),
        "iXA%": sk["iXA_pct"].round(1),

        "Med10_SOG": sk["Median10_SOG"],
        "Trim10_SOG": sk["TrimMean10_SOG"],
        "Avg5_SOG": sk["Avg5_SOG"],

        "L5_G": sk["G5_total"].apply(fmt_x_over_5),
        "L5_A": sk["A5_total"].apply(fmt_x_over_5),

        "PP": sk["PP"],
        "Matchup": sk["Matchup"],
        "Opp_Goalie": sk["Opp_Goalie"],
        "Opp_GP": sk["Opp_GP"],
        "Opp_SV": sk["Opp_SV"],
        "Opp_GAA": sk["Opp_GAA"],

        "Matrix": sk["Matrix"],
        "Prop": sk["Prop"],
        "Confidence": sk["Confidence"],
        "Conf_SOG": sk["Conf_SOG"],
        "Conf_Point": sk["Conf_Point"],
        "Conf_Goal": sk["Conf_Goal"],
        "Goal_Tag": sk["Goal_Tag"],

        # v2 columns (new)
        "v2_Best": sk["v2_best_market"],
        "v2_Pivot": sk["v2_pivot"],
        "v2_Stability": sk["v2_player_stability"],
        "v2_DefWeak": sk["v2_defense_vulnerability"],

        "Odds": "",
        "Result": "",
    })

    # Sort: Green -> Yellow -> Red, then confidence desc
    order = {"Green": 0, "Yellow": 1, "Red": 2}
    tracker["_ord"] = tracker["Matrix"].map(order).fillna(9).astype(int)
    tracker = tracker.sort_values(["_ord", "Confidence", "iXG%"], ascending=[True, False, False]).drop(columns=["_ord"])

    out_path = os.path.join(OUTPUT_DIR, f"tracker_{today_local.isoformat()}.csv")
    tracker.to_csv(out_path, index=False)

    # 7) Terminal highlights
    greens = tracker[tracker["Matrix"] == "Green"].copy().head(TOP_GREEN_LIST)
    yellows = tracker[tracker["Matrix"] == "Yellow"].copy().head(TOP_YELLOW_LIST)
    goals = tracker[tracker["Goal_Tag"] == "Yes"].copy().sort_values(["Conf_Goal", "iXG%"], ascending=False).head(TOP_GOAL_LIST)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    if not greens.empty:
        print("GREEN PLAYS (sorted)\\n")
        cols = ["Player", "Pos", "Team", "Game", "iXG%", "iXA%", "Med10_SOG", "Trim10_SOG", "Avg5_SOG",
                "PP", "Matchup", "Opp_Goalie", "Opp_SV", "Opp_GAA", "Prop", "Confidence", "v2_Best", "v2_Pivot"]
        print(greens[cols].to_string(index=False))
        print("")
    else:
        print("No GREEN plays (with current thresholds). Top YELLOW:\\n")
        cols = ["Player", "Pos", "Team", "Game", "iXG%", "iXA%", "Med10_SOG", "Trim10_SOG", "PP", "Matchup", "Prop", "Conf_SOG", "Conf_Point", "v2_Best", "v2_Pivot"]
        print(yellows[cols].to_string(index=False))
        print("")

    if not goals.empty:
        print("BEST ANYTIME GOAL CANDIDATES (Goal_Tag == Yes)\\n")
        cols = ["Player", "Pos", "Team", "Game", "iXG%", "Med10_SOG", "Trim10_SOG", "PP",
                "Matchup", "Opp_Goalie", "Opp_SV", "Opp_GAA", "Conf_Goal"]
        print(goals[cols].to_string(index=False))
        print("")
    else:
        print("No Goal_Tag == Yes candidates (rules are strict). If you want more, we can loosen the quality path.\\n")

    print(f"CSV saved to: {out_path}")
    print(f"Cache saved to: {cache_path_today(today_local)}")
    print("Note: Opp_Goalie is 'starter-ish by GP' (not confirmed starters yet).")
    print("Note: NHL game-log endpoint provides the last games; Median/TrimMean are anti-skew.\\n")


if __name__ == "__main__":
    main()
