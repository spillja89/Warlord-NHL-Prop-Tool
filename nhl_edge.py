from __future__ import annotations

"""
NHL EDGE TOOL — Stable v7.2 (engine) — single-file paste (CLEAN FIXED + SOG + Injury Upgrade)

What’s fixed tonight:
✅ No more NameError crashes (helpers are defined BEFORE use)
✅ SOG inflation fixed (NHL “shots” vs attempts guarded, prefers shotsOnGoal)
✅ Injury upgrades (DFO line-combo injuries parsed, Team_Out_Count correct, GTD penalty applied)
✅ Injury impact now adjusts confidence slightly (OUT/IR excluded; GTD lowers confidence)
✅ Keeps your existing outputs + matrices/confidence/regression + assists earned rule

Run:
  python -u nhl_edge.py
  python -u nhl_edge.py --debug
  python -u nhl_edge.py --date 2026-01-09 --debug
"""

import os
import time
import json
import math
import argparse
import re
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from typing import Optional, Dict, List, Any, Tuple
from statistics import median
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

pd.options.display.float_format = "{:.2f}".format



# ============================
# CONFIG
# ============================
OUTPUT_DIR = "output"
CACHE_DIR = "cache"

REQUEST_TIMEOUT_SEC = 25
HTTP_SLEEP_SEC = 0.02
LOCAL_TZ = "America/Chicago"

MAX_WORKERS = 10

PROCESS_MIN_PCT = 70     # min(max(iXG%, iXA%)) to fetch NHL logs
CAND_CAP = 220           # cap NHL log pulls (avoid 800+)

# Matrix thresholds for SOG based on last-10 median
SOG_MED10_GREEN_FWD = 3.5
SOG_MED10_GREEN_DEF = 2.3
SOG_MED10_YELLOW_FWD = 3.0
SOG_MED10_YELLOW_DEF = 1.9

# Regression heat thresholds (gap over 10)
REG_HOT_GAP = 2.5
REG_WARM_GAP = 1.5

# -------------------------
# TEAM RECENT GOALS FOR HARD GATE (applies to GOAL / POINTS / ASSISTS)
# -------------------------
TEAM_GF_WINDOW = 5
TEAM_GF_MIN_AVG = 2.70   # HARD FAIL threshold


# Goalie weakness thresholds
MIN_GOALIE_GP = 5

# Star prior strength (small nudge)
TALENT_MULT_MAX = 1.10
TALENT_MULT_MIN = 0.94
TALENT_MULT_STRENGTH = 0.18  # +/- ~6% around 50->100, 50->0

# Usage weighting into confidence (small)
USAGE_WEIGHT_SOG = 0.08
USAGE_WEIGHT_POINTS = 0.08

# Injury impact into confidence (small; GTD hurts, ROLE+ helps)
INJURY_CONF_MULT = 2.0  # Injury_DFO_Score * this added into conf (clamped)


# ============================
# TALENT TIERS (ELITE / STAR / NONE)
# ============================
ELITE_SEED = set()
STAR_SEED  = set()

ELITE_STARSCORE = 95.0
STAR_STARSCORE  = 90.0

ELITE_TOI_PCT = 73.0
STAR_TOI_PCT  = 68.0

ELITE_TOPPCT = 95.0
STAR_TOPPCT  = 90.0

ELITE_SHOTINTENT_PCT = 96.0
STAR_SHOTINTENT_PCT  = 90.0

ELITE_MIN_PPG = 1.10  # tune: 0.95–1.10
STAR_MIN_PPG  = 0.90   # tune: 0.70–0.85



# ============================
# TEAM ALIASES (schedule -> moneypuck)
# ============================
TEAM_ALIAS = {
    "VEG": "VGK", "VGK": "VGK",
    "NJ": "NJD", "NJD": "NJD",
    "LA": "LAK", "LAK": "LAK",
    "TB": "TBL", "TBL": "TBL",
    "CLB": "CBJ", "CBJ": "CBJ",
    "SJ": "SJS", "SJS": "SJS",
    "NAS": "NSH", "NSH": "NSH",
    "MON": "MTL", "MTL": "MTL",
    "WAS": "WSH", "WSH": "WSH",
    "WIN": "WPG", "WPG": "WPG",
    "NYI": "NYI", "NYR": "NYR",
    "PHX": "ARI", "ARZ": "ARI", "ARI": "ARI",
    "UTA": "UTA", "UTH": "UTA","DET": "DET"
}

def norm_team(x: Any) -> str:
    s = str(x).strip().upper()
    return TEAM_ALIAS.get(s, s)

def is_defense(pos: str) -> bool:
    return (pos or "").upper().strip() in {"D", "LD", "RD"}


# ============================
# REQUIRED HELPERS (MUST BE ABOVE USAGE)
# ============================
def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x is pd.NA:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None

def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or x is pd.NA:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return int(x)
    except Exception:
        return None

def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))

def safe_ppg(df: pd.DataFrame) -> pd.Series:
    """
    Best-effort PPG:
    - If actual Points/GP exist, use them
    - else return NaN series (won't break tiering; PPG proof just fails)
    """
    gp = pd.to_numeric(df.get("games_played"), errors="coerce").replace(0, np.nan)

    # Try common actual-stat column names if present
    p_col = None
    for cand in ["P", "Points", "points", "I_F_points", "pts"]:
        if cand in df.columns:
            p_col = cand
            break

    # If we have goals+assists, compute points
    g_col = None
    a_col = None
    for cand in ["G", "Goals", "goals", "I_F_goals"]:
        if cand in df.columns:
            g_col = cand
            break
    for cand in ["A", "Assists", "assists", "I_F_assists"]:
        if cand in df.columns:
            a_col = cand
            break

    if p_col is not None:
        pts = pd.to_numeric(df[p_col], errors="coerce")
        return (pts / gp).replace([np.inf, -np.inf], np.nan)

    if g_col is not None and a_col is not None:
        g = pd.to_numeric(df[g_col], errors="coerce").fillna(0)
        a = pd.to_numeric(df[a_col], errors="coerce").fillna(0)
        return ((g + a) / gp).replace([np.inf, -np.inf], np.nan)

    return pd.Series(np.nan, index=df.index, dtype="float64")


def pct_rank(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.rank(pct=True) * 100.0

def parse_utc_to_local_date(start_time_utc: str, tz_name: str) -> Optional[date]:
    if not start_time_utc or "T" not in start_time_utc:
        return None
    try:
        dt_utc = datetime.fromisoformat(start_time_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
        dt_local = dt_utc.astimezone(ZoneInfo(tz_name))
        return dt_local.date()
    except Exception:
        return None

def _norm_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _lower_map(df: pd.DataFrame) -> Dict[str, str]:
    return {str(c).strip().lower(): str(c).strip() for c in df.columns}

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lm = _lower_map(df)
    for nm in candidates:
        c = lm.get(nm.lower())
        if c:
            return c
    return None

def debug_find_like(df: pd.DataFrame, needle: str) -> List[str]:
    n = needle.lower()
    return [c for c in df.columns if n in str(c).lower()]

def per60(stat: Any, toi: Any) -> pd.Series:
    """
    Compute per-60 from stat and icetime (MoneyPuck can be minutes or seconds).
    """
    s = pd.to_numeric(stat, errors="coerce")
    t = pd.to_numeric(toi, errors="coerce")
    out = pd.Series(np.nan, index=t.index, dtype="float64")
    m = t > 0
    sec = m & (t > 10000)
    mins = m & ~sec
    out.loc[sec] = s.loc[sec] * 3600.0 / t.loc[sec]
    out.loc[mins] = s.loc[mins] * 60.0 / t.loc[mins]
    return out


# ============================
# Files/HTTP
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


# ============================
# NHL schedule
# ============================
def nhl_schedule_today(sess: requests.Session, today_local: date) -> List[Dict[str, Any]]:
    def fetch_day(d: date) -> Any:
        return http_get_json(sess, f"https://api-web.nhle.com/v1/schedule/{d.isoformat()}")

    data_today = fetch_day(today_local)
    data_tom = fetch_day(today_local + timedelta(days=1))

    out: List[Dict[str, Any]] = []

    def consume(data: Any) -> None:
        for day in data.get("gameWeek", []):
            for g in day.get("games", []):
                away = (g.get("awayTeam") or {}).get("abbrev")
                home = (g.get("homeTeam") or {}).get("abbrev")
                start_utc = str(g.get("startTimeUTC", ""))
                local_d = parse_utc_to_local_date(start_utc, LOCAL_TZ)
                if local_d == today_local and away and home:
                    out.append({"away": norm_team(away), "home": norm_team(home), "startTimeUTC": start_utc})

    consume(data_today)
    consume(data_tom)

    seen = set()
    dedup: List[Dict[str, Any]] = []
    for g in out:
        k = (g["away"], g["home"])
        if k not in seen:
            seen.add(k)
            dedup.append(g)
    return dedup


# ============================
# MoneyPuck CSV loads
# ============================
def current_season_start_year(today: date) -> int:
    return today.year if today.month >= 7 else today.year - 1

def moneypuck_url(kind: str, y: int) -> str:
    return f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{y}/regular/{kind}.csv"

def load_moneypuck_csv(sess: requests.Session, url: str) -> pd.DataFrame:
    txt = http_get_text(sess, url)
    if not txt.strip():
        raise RuntimeError(f"Empty CSV from {url}")
    df = pd.read_csv(StringIO(txt))
    df.columns = df.columns.str.strip()
    return df

def load_moneypuck_best_effort(sess: requests.Session, kind: str) -> pd.DataFrame:
    start = current_season_start_year(date.today())
    last_err = None
    for y in (start, start - 1):
        url = moneypuck_url(kind, y)
        try:
            return load_moneypuck_csv(sess, url)
        except Exception as e:
            last_err = f"{url} -> {type(e).__name__}: {e}"
    raise RuntimeError(f"Could not download MoneyPuck {kind}.csv. Last error: {last_err}")


# ============================
# DailyFaceoff team mapping
# ============================
DFO_TEAMNAME_TO_ABBR = {
    "Anaheim Ducks": "ANA",
    "Arizona Coyotes": "ARI",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Utah Hockey Club": "UTA", "Utah": "UTA", "Utah HC": "UTA", "Utah Hockey": "UTA",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
    "St Louis Blues": "STL",
}

def _norm_teamname_full(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

DFO_TEAMNAME_TO_ABBR_NORM = { _norm_teamname_full(k): v for k, v in DFO_TEAMNAME_TO_ABBR.items() }

def dfo_team_to_abbr(team_full: str) -> str | None:
    n = _norm_teamname_full(team_full)
    hit = DFO_TEAMNAME_TO_ABBR_NORM.get(n)
    if hit:
        return hit
    best = None
    best_len = 0
    for k, ab in DFO_TEAMNAME_TO_ABBR_NORM.items():
        if k and (k in n or n in k):
            if len(k) > best_len:
                best = ab
                best_len = len(k)
    return best


# ============================
# DailyFaceoff starting goalies
# ============================
def fetch_dailyfaceoff_starters(today_local: date, debug: bool = False) -> dict[str, dict[str, str]]:
    url = "https://www.dailyfaceoff.com/starting-goalies"
    html = http_get_text(http_session(), url)
    soup = BeautifulSoup(html, "html.parser")

    statuses_norm = {
        "confirmed": "Confirmed",
        "unconfirmed": "Unconfirmed",
        "likely": "Likely",
        "expected": "Expected",
        "probable": "Likely",
    }

    def norm_status(s: Any) -> str:
        t = str(s or "").strip().lower()
        return statuses_norm.get(t, str(s or "").strip().title())

    def team_to_abbr(team_obj: Any) -> str:
        if isinstance(team_obj, dict):
            for k in ("abbrev", "abbreviation", "abbr", "code", "shortName", "short_name"):
                v = team_obj.get(k)
                if v:
                    return norm_team(str(v))
            for k in ("name", "fullName", "full_name", "teamName"):
                v = team_obj.get(k)
                if v:
                    ab = DFO_TEAMNAME_TO_ABBR_NORM.get(_norm_teamname_full(str(v)))
                    return norm_team(ab or str(v))
        s = str(team_obj or "").strip()
        if not s:
            return ""
        ab = DFO_TEAMNAME_TO_ABBR_NORM.get(_norm_teamname_full(s))
        return norm_team(ab or s)

    def goalie_name(goalie_obj: Any) -> str:
        if isinstance(goalie_obj, dict):
            for k in ("name", "fullName", "full_name", "playerName"):
                v = goalie_obj.get(k)
                if v:
                    return str(v).strip()
            p = goalie_obj.get("player") if isinstance(goalie_obj.get("player"), dict) else None
            if p:
                for k in ("name", "fullName", "full_name"):
                    v = p.get(k)
                    if v:
                        return str(v).strip()
        return str(goalie_obj or "").strip()

    # --------- 1) Try __NEXT_DATA__ ----------
    out: dict[str, dict[str, str]] = {}
    script = soup.find("script", id="__NEXT_DATA__")
    if script and script.string:
        try:
            data = json.loads(script.string)

            def walk(x: Any):
                if isinstance(x, dict):
                    yield x
                    for v in x.values():
                        yield from walk(v)
                elif isinstance(x, list):
                    for it in x:
                        yield from walk(it)

            candidates = []
            for d in walk(data):
                if any(k in d for k in ("goalie", "startingGoalie", "starter", "goalieStarter")) and any(
                    k in d for k in ("team", "teamAbbrev", "team_abbrev", "teamName", "teamAbbreviation")
                ):
                    candidates.append(d)

            for d in candidates:
                team_obj = d.get("team") or d.get("teamName") or d.get("teamAbbrev") or d.get("team_abbrev")
                g_obj = d.get("goalie") or d.get("startingGoalie") or d.get("starter") or d.get("goalieStarter")
                st = d.get("status") or d.get("starterStatus") or d.get("confirmation") or ""
                team_abbr = team_to_abbr(team_obj)
                gname = goalie_name(g_obj)
                if team_abbr and gname:
                    out[team_abbr] = {"goalie": gname, "status": norm_status(st)}

            if len(out) >= 10:
                if debug:
                    print(f"[DFO] __NEXT_DATA__ parsed starter teams: {len(out)}")
                    print("[DFO] sample:", list(out.items())[:8])
                return out

            if debug:
                print(f"[DFO] __NEXT_DATA__ produced only {len(out)} teams; falling back...")

        except Exception as e:
            if debug:
                print(f"[DFO] __NEXT_DATA__ parse failed: {type(e).__name__}: {e}")

    # --------- 2) Fallback text scan ----------
    raw_lines = soup.get_text("\n", strip=True).splitlines()
    bad_exact = {"Show More", "Line Combos", "News", "Stats", "Schedule", "|"}
    statuses = {"Confirmed", "Unconfirmed", "Likely", "Expected"}

    lines: list[str] = []
    for ln in raw_lines:
        ln = (ln or "").strip()
        if not ln:
            continue
        if ln in bad_exact:
            continue
        if ln.startswith("Schedule "):
            ln = ln.replace("Schedule ", "", 1).strip()
        if ln == "Schedule":
            continue
        lines.append(ln)

    def is_iso_datetime(s: str) -> bool:
        return ("T" in s and "Z" in s and len(s) >= 18 and s[4] == "-" and s[7] == "-")

    def find_goalie_after(idx: int) -> tuple[str, str, int] | None:
        j = idx
        while j < len(lines):
            s = lines[j]
            if s in statuses or is_iso_datetime(s):
                j += 1
                continue
            if s.startswith(("W-L-OTL:", "GAA:", "SV%:", "SO:", "Source:", "#", "NHL Starting Goalies")):
                j += 1
                continue
            goalie = s
            k = j + 1
            while k < len(lines) and (is_iso_datetime(lines[k]) or lines[k] in bad_exact):
                k += 1
            if k < len(lines) and lines[k] in statuses:
                return goalie, lines[k], k + 1
            j += 1
        return None

    matchup_idxs: list[tuple[int, str, str]] = []
    for i in range(len(lines)):
        ln = lines[i]
        if " at " in ln:
            away_full, home_full = [x.strip() for x in ln.split(" at ", 1)]
            matchup_idxs.append((i, away_full, home_full))
        elif " @ " in ln:
            away_full, home_full = [x.strip() for x in ln.split(" @ ", 1)]
            matchup_idxs.append((i, away_full, home_full))
        elif ln == "at" and i - 1 >= 0 and i + 1 < len(lines):
            matchup_idxs.append((i, lines[i - 1].strip(), lines[i + 1].strip()))

    out = {}
    for i, away_full, home_full in matchup_idxs:
        away = dfo_team_to_abbr(away_full)
        home = dfo_team_to_abbr(home_full)
        if not away or not home:
            continue
        g1 = find_goalie_after(i + 1)
        if not g1:
            continue
        goalie_away, status_away, next_idx = g1
        g2 = find_goalie_after(next_idx)
        if not g2:
            continue
        goalie_home, status_home, _ = g2
        out[norm_team(away)] = {"goalie": goalie_away.strip(), "status": status_away.strip()}
        out[norm_team(home)] = {"goalie": goalie_home.strip(), "status": status_home.strip()}

    if debug:
        print(f"[DFO] fallback text parsed starter teams: {len(out)}")
        print("[DFO] sample:", list(out.items())[:8])

    return out


# ============================
# DailyFaceoff injuries (UPGRADED)
# ============================
DFO_TEAM_ABBR_TO_SLUG = {
    "ANA": "anaheim-ducks",
    "ARI": "arizona-coyotes",
    "BOS": "boston-bruins",
    "BUF": "buffalo-sabres",
    "CGY": "calgary-flames",
    "CAR": "carolina-hurricanes",
    "CHI": "chicago-blackhawks",
    "COL": "colorado-avalanche",
    "CBJ": "columbus-blue-jackets",
    "DAL": "dallas-stars",
    "DET": "detroit-red-wings",
    "EDM": "edmonton-oilers",
    "FLA": "florida-panthers",
    "LAK": "los-angeles-kings",
    "MIN": "minnesota-wild",
    "MTL": "montreal-canadiens",
    "NSH": "nashville-predators",
    "NJD": "new-jersey-devils",
    "NYI": "new-york-islanders",
    "NYR": "new-york-rangers",
    "OTT": "ottawa-senators",
    "PHI": "philadelphia-flyers",
    "PIT": "pittsburgh-penguins",
    "SJS": "san-jose-sharks",
    "SEA": "seattle-kraken",
    "STL": "st-louis-blues",
    "TBL": "tampa-bay-lightning",
    "TOR": "toronto-maple-leafs",
    "UTA": "utah-hockey-club",
    "VAN": "vancouver-canucks",
    "VGK": "vegas-golden-knights",
    "WSH": "washington-capitals",
    "WPG": "winnipeg-jets",
}

def _dfo_norm_status(s: str) -> str:
    t = (s or "").strip().upper()
    if not t:
        return "Healthy"
    if "GTD" in t or "DAY" in t or "DTD" in t:
        return "GTD"
    if "IR" in t or "INJURED" in t:
        return "IR"
    if "OUT" in t:
        return "Out"
    if "SCRATCH" in t:
        return "Scratch"
    return t.title()

def _dfo_injury_type(s: str) -> str:
    t = (s or "").strip().lower()
    if not t:
        return "Unknown"
    if "upper" in t:
        return "Upper"
    if "lower" in t:
        return "Lower"
    if "ill" in t or "flu" in t:
        return "Illness"
    return "Unknown"

def _find_injury_table(soup: BeautifulSoup) -> Any:
    tables = soup.find_all("table")
    best = None
    best_score = 0
    for tb in tables:
        header_txt = " ".join([th.get_text(" ", strip=True) for th in tb.find_all("th")]).lower()
        score = 0
        score += 1 if "player" in header_txt else 0
        score += 1 if "status" in header_txt else 0
        score += 1 if "injury" in header_txt else 0
        score += 1 if "return" in header_txt or "expected" in header_txt else 0
        if score > best_score:
            best_score = score
            best = tb
    return best if best_score >= 2 else None

def fetch_dfo_injuries_for_team(team_abbr: str, debug: bool = False) -> List[Dict[str, Any]]:
    team_abbr = norm_team(team_abbr)
    slug = DFO_TEAM_ABBR_TO_SLUG.get(team_abbr)
    if not slug:
        return []

    url = f"https://www.dailyfaceoff.com/teams/{slug}/line-combinations"
    try:
        html = http_get_text(http_session(), url)
    except Exception as e:
        if debug:
            print(f"[DFO INJ] fetch failed {team_abbr}: {type(e).__name__}: {e}")
        return []

    soup = BeautifulSoup(html, "html.parser")

    tb = _find_injury_table(soup)
    if tb is not None:
        rows = tb.find_all("tr")
        out_rows: List[Dict[str, Any]] = []
        for r in rows[1:]:
            tds = r.find_all("td")
            if not tds:
                continue
            cols = [td.get_text(" ", strip=True) for td in tds]
            cols = [c.strip() for c in cols if c is not None]
            player = cols[0] if len(cols) >= 1 else ""
            status = cols[1] if len(cols) >= 2 else ""
            injury = cols[2] if len(cols) >= 3 else ""
            expret = cols[3] if len(cols) >= 4 else ""
            if not player:
                continue
            out_rows.append({
                "Team": team_abbr,
                "Player": player,
                "Status": _dfo_norm_status(status),
                "Injury": injury,
                "Expected_Return": expret,
            })
        if debug:
            print(f"[DFO INJ] {team_abbr} injuries parsed (table): {len(out_rows)}")
        return out_rows

    # List-section parsing fallback
    lines = soup.get_text("\n", strip=True).splitlines()
    lines = [(ln or "").strip() for ln in lines if (ln or "").strip()]
    try:
        start_idx = next(i for i, ln in enumerate(lines) if ln.strip().lower() == "injuries")
    except StopIteration:
        if debug:
            print(f"[DFO INJ] no 'Injuries' section found for {team_abbr}")
        return []

    stop_words = {"badges:", "badge:", "click player jersey for news, stats and more!"}
    status_tokens = {"ir", "out", "dtd", "gtd", "day-to-day", "game-time decision", "scratch"}

    out_rows: List[Dict[str, Any]] = []
    cur_status: Optional[str] = None

    i = start_idx + 1
    while i < len(lines):
        ln = lines[i].strip()
        low = ln.lower()

        if low in stop_words or low.startswith("badges"):
            break

        if low in status_tokens:
            if low == "ir":
                cur_status = "IR"
            elif low == "out":
                cur_status = "Out"
            elif low == "scratch":
                cur_status = "Scratch"
            else:
                cur_status = "GTD"
            i += 1
            continue

        if low in {"injuries", "goalies", "forwards", "defensive pairings"}:
            i += 1
            continue

        if cur_status:
            player = ln
            if len(player) >= 3:
                out_rows.append({
                    "Team": team_abbr,
                    "Player": player,
                    "Status": _dfo_norm_status(cur_status),
                    "Injury": "",
                    "Expected_Return": "",
                })
            cur_status = None
            i += 1
            continue

        i += 1

    if debug:
        print(f"[DFO INJ] {team_abbr} injuries parsed (list): {len(out_rows)}")

    return out_rows

def fetch_dfo_injuries_for_teams(teams: set[str], debug: bool = False) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for t in sorted({norm_team(x) for x in teams}):
        rows.extend(fetch_dfo_injuries_for_team(t, debug=debug))
        time.sleep(HTTP_SLEEP_SEC)

    if not rows:
        return pd.DataFrame(columns=["Team", "Player", "Status", "Injury", "Expected_Return", "Player_norm"])

    df = pd.DataFrame(rows)
    df["Player_norm"] = df["Player"].astype(str).map(_norm_name)
    df["Team"] = df["Team"].astype(str).map(norm_team)
    return df

def apply_injury_dfo(sk: pd.DataFrame, inj_df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    out = sk.copy()

    out["Injury_Status"] = "Healthy"
    out["Injury_Type"] = "Unknown"
    out["Injury_Text"] = ""
    out["Injury_Return"] = ""
    out["Team_Out_Count"] = 0
    out["Injury_DFO_Score"] = 0.0
    out["Injury_Badge"] = ""

    if inj_df is None or inj_df.empty:
        if debug:
            print("[DFO INJ] inj_df empty -> neutral injuries")
        return out

    tmp = inj_df.copy()
    tmp["Status"] = tmp["Status"].astype(str).map(_dfo_norm_status)

    # Team_Out_Count counts Out/IR/Scratch as "missing bodies"
    out_counts = (
        tmp[tmp["Status"].isin(["Out", "IR", "Scratch"])]
        .groupby("Team", as_index=True)
        .size()
        .to_dict()
    )
    out["Team_Out_Count"] = out["Team"].map(lambda t: int(out_counts.get(norm_team(t), 0)))

    j = inj_df[["Team", "Player_norm", "Status", "Injury", "Expected_Return"]].copy()
    j["Team"] = j["Team"].astype(str).map(norm_team)
    j["Player_norm"] = j["Player_norm"].astype(str)

    out["Player_norm"] = out["Player"].astype(str).map(_norm_name)

    out = out.merge(
        j,
        left_on=["Team", "Player_norm"],
        right_on=["Team", "Player_norm"],
        how="left",
        suffixes=("", "_inj"),
    )

    out["Injury_Status"] = out["Status"].apply(lambda x: _dfo_norm_status(x) if pd.notna(x) else "Healthy")
    out["Injury_Text"] = out["Injury"].apply(lambda x: str(x).strip() if pd.notna(x) else "")
    out["Injury_Return"] = out["Expected_Return"].apply(lambda x: str(x).strip() if pd.notna(x) else "")
    out["Injury_Type"] = out["Injury_Text"].apply(_dfo_injury_type)

    def base_score(st: str) -> float:
        if st in ("Out", "IR", "Scratch"):
            return -3.0
        if st == "GTD":
            return -1.0
        return 0.0

    out["Injury_DFO_Score"] = out["Injury_Status"].apply(base_score)

    # ROLE+ bump: if team is missing 2+ and this guy is high-usage, they often get extra run
    toi_pct = pd.to_numeric(out.get("TOI_Pct", 50.0), errors="coerce").fillna(50.0)
    bump = (
        (out["Injury_Status"] == "Healthy") &
        (pd.to_numeric(out["Team_Out_Count"], errors="coerce").fillna(0) >= 2) &
        (toi_pct >= 70.0)
    )
    out.loc[bump, "Injury_DFO_Score"] = out.loc[bump, "Injury_DFO_Score"] + 1.0
    out["Injury_DFO_Score"] = out["Injury_DFO_Score"].map(lambda x: max(-5.0, min(5.0, float(x))))

    def badge(st: str, score: float) -> str:
        if st in ("Out", "IR", "Scratch"):
            return "🔴 OUT"
        if st == "GTD":
            return "🟡 GTD"
        if score >= 1.0:
            return "🟢 ROLE+"
        return ""

    out["Injury_Badge"] = out.apply(
        lambda r: badge(str(r.get("Injury_Status", "")), float(r.get("Injury_DFO_Score", 0.0))),
        axis=1
    )

    out = out.drop(columns=["Status", "Injury", "Expected_Return"], errors="ignore")
    out = out.drop(columns=["Player_norm"], errors="ignore")

    if debug:
        ex = out[["Team", "Player", "Injury_Status", "Injury_Text", "Injury_DFO_Score", "Injury_Badge", "Team_Out_Count"]].head(25)
        print("\n[DFO INJ] sample merged injury fields:\n", ex.to_string(index=False))

    return out


# ============================
# MoneyPuck normalize: skaters (ALL)
# ============================
def normalize_skaters_all(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    col_pid = find_col(df, ["playerid", "playerId"])
    col_name = find_col(df, ["name", "playername", "playerName", "player", "skatername", "skaterName"])
    col_team = find_col(df, ["team", "teamabbrev", "teamAbbrev", "team_abbrev"])
    col_pos  = find_col(df, ["position", "pos", "positioncode", "positionCode"])
    col_sit  = find_col(df, ["situation"])
    col_gp   = find_col(df, ["games_played", "gamesplayed", "gp"])
    col_it   = find_col(df, ["icetime", "timeonice", "toi", "timeOnIce"])

    # expected offense (required)
    col_ixg  = find_col(df, ["I_F_xGoals", "i_f_xgoals", "ixGoals", "ixg", "xGoals"])
    col_a1   = find_col(df, ["I_F_primaryAssists", "i_f_primaryassists", "primaryassists", "primaryAssists"])
    col_a2   = find_col(df, ["I_F_secondaryAssists", "i_f_secondaryassists", "secondaryassists", "secondaryAssists"])

    # try true xAssists/ixA if present
    col_xa = find_col(df, [
        "I_F_xAssists", "i_f_xassists", "xAssists", "xassists",
        "I_F_ixA", "i_f_ixa", "ixA", "iXA", "xA", "xa"
    ])

    # ===== NEW: actual season totals (best-effort) so we can compute REAL PPG =====
    col_g = find_col(df, ["I_F_goals", "goals", "Goals", "G"])
    col_a = find_col(df, ["I_F_assists", "assists", "Assists", "A"])
    col_p = find_col(df, ["I_F_points", "points", "Points", "P", "pts"])

    missing = [k for k, v in {
        "playerId": col_pid, "name": col_name, "team": col_team,
        "situation": col_sit, "games_played": col_gp, "icetime": col_it,
        "I_F_xGoals": col_ixg, "I_F_primaryAssists": col_a1, "I_F_secondaryAssists": col_a2
    }.items() if v is None]
    if missing:
        raise KeyError(f"MoneyPuck skaters.csv missing required columns: {missing}")

    use_cols = [col_pid, col_name, col_team, col_sit, col_gp, col_it, col_ixg, col_a1, col_a2]
    if col_pos:
        use_cols.insert(3, col_pos)
    if col_xa:
        use_cols.append(col_xa)

    # add actual totals if present
    if col_g: use_cols.append(col_g)
    if col_a: use_cols.append(col_a)
    if col_p: use_cols.append(col_p)

    out = df[use_cols].copy()

    rename_map = {
        col_pid: "playerId",
        col_name: "Player",
        col_team: "Team",
        col_sit: "situation",
        col_gp: "games_played",
        col_it: "icetime",
        col_ixg: "I_F_xGoals",
        col_a1: "I_F_primaryAssists",
        col_a2: "I_F_secondaryAssists",
    }
    if col_pos:
        rename_map[col_pos] = "Pos"
    if col_xa:
        rename_map[col_xa] = "I_F_xAssists"

    if col_g: rename_map[col_g] = "G"
    if col_a: rename_map[col_a] = "A"
    if col_p: rename_map[col_p] = "P"

    out = out.rename(columns=rename_map)

    if "Pos" not in out.columns:
        out["Pos"] = "F"

    out["Team"] = out["Team"].apply(norm_team)
    out["situation"] = out["situation"].astype(str).str.lower()
    out = out[out["situation"].isin(["all", "all situations", "all_situations", "all-situations"])].copy()

    out["playerId"] = pd.to_numeric(out["playerId"], errors="coerce")
    out["games_played"] = pd.to_numeric(out["games_played"], errors="coerce")
    out["icetime"] = pd.to_numeric(out["icetime"], errors="coerce")

    # expected xG / xA
    out["iXG_raw"] = pd.to_numeric(out["I_F_xGoals"], errors="coerce")

    if "I_F_xAssists" in out.columns:
        out["iXA_raw"] = pd.to_numeric(out["I_F_xAssists"], errors="coerce")
        out["iXA_Source"] = "xAssists"
    else:
        out["iXA_raw"] = (
            pd.to_numeric(out["I_F_primaryAssists"], errors="coerce").fillna(0)
            + pd.to_numeric(out["I_F_secondaryAssists"], errors="coerce").fillna(0)
        )
        out["iXA_Source"] = "A1+A2"

    out["iXG_pct"] = out["iXG_raw"].rank(pct=True) * 100.0
    out["iXA_pct"] = out["iXA_raw"].rank(pct=True) * 100.0

    out["Player"] = out["Player"].astype(str)
    out["Pos"] = out["Pos"].astype(str).replace({"nan": "F", "None": "F"}).fillna("F")

    gp = pd.to_numeric(out["games_played"], errors="coerce")
    out["TOI_per_game"] = (pd.to_numeric(out["icetime"], errors="coerce") / gp).replace([np.inf, -np.inf], np.nan)

    # ensure numeric for totals if present
    for c in ["G", "A", "P"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if debug:
        print("\n[DEBUG] normalize_skaters_all: ok")
        print("[DEBUG] actual totals present:", {c: (c in out.columns) for c in ["G", "A", "P"]})
        print("[DEBUG] iXA_Source counts:", out["iXA_Source"].value_counts(dropna=False).to_dict())
        print("[DEBUG] sample:", out[["Player", "Team", "Pos", "iXA_Source"]].head(6).to_dict(orient="records"))

    return out

# ============================
# MoneyPuck normalize: skaters (5v5 proxies)
# ============================
def normalize_skaters_5v5(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.strip()

    col_pid = find_col(out, ["playerid", "playerId"])
    col_name = find_col(out, ["name", "playername", "playerName", "player", "skatername", "skaterName"])
    col_team = find_col(out, ["team", "teamabbrev", "teamAbbrev", "team_abbrev"])
    col_pos  = find_col(out, ["position", "pos", "positioncode", "positionCode"])
    col_sit  = find_col(out, ["situation"])
    col_it   = find_col(out, ["icetime", "timeonice", "toi", "timeOnIce"])

    if not (col_pid and col_team and col_sit and col_it):
        raise KeyError("Skaters CSV missing required columns for 5v5 (playerId/team/situation/icetime).")

    def norm_sit(x: Any) -> str:
        s = str(x).lower()
        return "".join(ch for ch in s if ch.isalnum())

    sit = out[col_sit].apply(norm_sit)
    out = out[sit.isin({"5on5", "5v5", "ev5", "ev", "evenstrength", "even"})].copy()

    out["playerId"] = pd.to_numeric(out[col_pid], errors="coerce")
    out["Team"] = out[col_team].apply(norm_team)
    out["Pos"] = out[col_pos].astype(str) if col_pos else "F"
    out["icetime"] = pd.to_numeric(out[col_it], errors="coerce")
    out["Player"] = out[col_name].astype(str) if col_name else ""

    col_points = find_col(out, ["I_F_points", "points"])
    col_goals  = find_col(out, ["I_F_goals", "goals"])
    col_a1     = find_col(out, ["I_F_primaryAssists", "i_f_primaryassists", "primaryassists", "primaryAssists"])
    col_a2     = find_col(out, ["I_F_secondaryAssists", "i_f_secondaryassists", "secondaryassists", "secondaryAssists"])

    if col_points:
        points = pd.to_numeric(out[col_points], errors="coerce")
    else:
        g  = pd.to_numeric(out[col_goals], errors="coerce") if col_goals else pd.Series(0.0, index=out.index)
        a1 = pd.to_numeric(out[col_a1], errors="coerce") if col_a1 else pd.Series(0.0, index=out.index)
        a2 = pd.to_numeric(out[col_a2], errors="coerce") if col_a2 else pd.Series(0.0, index=out.index)
        points = (g.fillna(0) + a1.fillna(0) + a2.fillna(0)).astype("float64")

    a1s = pd.to_numeric(out[col_a1], errors="coerce") if col_a1 else pd.Series(np.nan, index=out.index)

    col_sa  = find_col(out, ["I_F_shotAssists", "shotAssists", "shotassists"])
    col_icf = find_col(out, ["I_F_iCF", "icf", "iCF"])

    if debug and (col_sa is None or col_icf is None):
        print("\n[DEBUG] skaters.csv 5v5 missing columns:")
        print(f"  col_sa = {col_sa}  col_icf = {col_icf}")
        print("  Columns containing 'assist':", debug_find_like(out, "assist")[:20])
        print("  Columns containing 'cf':", debug_find_like(out, "cf")[:20])

    shot_assists = pd.to_numeric(out[col_sa], errors="coerce") if col_sa else pd.Series(np.nan, index=out.index)
    icf = pd.to_numeric(out[col_icf], errors="coerce") if col_icf else pd.Series(np.nan, index=out.index)

    out["i5v5_points60"] = per60(points, out["icetime"])
    out["i5v5_primaryAssists60"] = per60(a1s, out["icetime"])
    out["i5v5_shotAssists60"] = per60(shot_assists, out["icetime"])
    out["i5v5_iCF60"] = per60(icf, out["icetime"])

    keep = ["playerId", "Team", "i5v5_points60", "i5v5_primaryAssists60", "i5v5_shotAssists60", "i5v5_iCF60"]
    return out[keep].copy()


# ============================
# MoneyPuck normalize: teams (5v5 defense + team offense context)
# ============================
def normalize_teams_5v5(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.strip()

    col_team = find_col(out, ["team", "teamabbrev", "teamAbbrev", "team_abbrev"])
    col_sit  = find_col(out, ["situation"])
    col_toi  = find_col(out, ["icetime", "timeonice", "toi", "timeOnIce", "TOI", "minutes", "Min"])
    col_xgf  = find_col(out, ["xGoalsFor", "xgoalsfor", "xGF", "xgf"])
    col_sf   = find_col(out, [
        "shotsFor", "ShotsFor", "shots_for", "sf", "SF",
        "shotsOnGoalFor", "ShotsOnGoalFor", "shotsongoalfor", "sogfor", "SOGFor",
        "shotAttemptsFor", "ShotAttemptsFor", "corsiFor", "CorsiFor", "cf", "CF",
        "5on5ShotsFor", "5v5ShotsFor", "shotsFor5v5", "shotsFor5on5"
    ])

    if not (col_team and col_sit and col_toi):
        raise KeyError(f"Teams CSV missing required columns (team/situation/TOI). Found team={col_team}, sit={col_sit}, toi={col_toi}")

    def norm_sit(x: Any) -> str:
        s = str(x).lower()
        return "".join(ch for ch in s if ch.isalnum())

    sit = out[col_sit].apply(norm_sit)
    out = out[sit.isin({"5on5", "5v5", "ev5", "evenstrength", "even"})].copy()

    out["Team"] = out[col_team].apply(norm_team)
    out["TOI"] = pd.to_numeric(out[col_toi], errors="coerce")

    col_xga  = find_col(out, ["xGoalsAgainst", "xgoalsagainst", "xGA", "xga"])
    col_hdca = find_col(out, ["HDCA", "hdca", "highDangerShotsAgainst", "highdangerchancesagainst"])
    col_slot = find_col(out, ["slotShotsAgainst", "slotshotsagainst", "SlotShotsAgainst", "slotSA", "slot_sa"])

    if debug and col_slot is None:
        print("\n[DEBUG] teams.csv slot column NOT FOUND. Columns containing 'slot':")
        print(debug_find_like(out, "slot")[:40])

    xga  = pd.to_numeric(out[col_xga], errors="coerce") if col_xga else pd.Series(np.nan, index=out.index)
    hdca = pd.to_numeric(out[col_hdca], errors="coerce") if col_hdca else pd.Series(np.nan, index=out.index)
    slot = pd.to_numeric(out[col_slot], errors="coerce") if col_slot else pd.Series(np.nan, index=out.index)
    xgf  = pd.to_numeric(out[col_xgf], errors="coerce") if col_xgf else pd.Series(np.nan, index=out.index)
    sf   = pd.to_numeric(out[col_sf], errors="coerce") if col_sf else pd.Series(np.nan, index=out.index)

    out["team_5v5_xGF60"] = per60(xgf, out["TOI"])
    out["team_5v5_SF60"]  = per60(sf, out["TOI"])

    out["opp_5v5_xGA60"] = per60(xga, out["TOI"])
    out["opp_5v5_HDCA60"] = per60(hdca, out["TOI"])
    out["opp_5v5_SlotSA60"] = per60(slot, out["TOI"])

    t5 = (
        out[[
            "Team",
            "opp_5v5_xGA60", "opp_5v5_HDCA60", "opp_5v5_SlotSA60",
            "team_5v5_xGF60", "team_5v5_SF60",
        ]]
        .groupby("Team", as_index=False)
        .mean(numeric_only=True)
    )
    return t5


# ============================
# MoneyPuck normalize: goalies
# ============================
def normalize_goalies(df: pd.DataFrame) -> pd.DataFrame:
    col_name = find_col(df, ["name", "goalie", "playername", "playerName"])
    col_team = find_col(df, ["team", "teamabbrev", "teamAbbrev", "team_abbrev"])
    col_sit  = find_col(df, ["situation"])
    col_gp   = find_col(df, ["games_played", "gamesplayed", "gp"])
    col_it   = find_col(df, ["icetime", "timeonice", "toi", "timeOnIce"])
    col_ongoal = find_col(df, ["ongoal", "shotsagainst", "shots_against", "shotsAgainst"])
    col_goals  = find_col(df, ["goals", "ga", "goalsagainst", "goals_against"])

    missing = [k for k, v in {
        "name": col_name, "team": col_team, "situation": col_sit, "gp": col_gp,
        "icetime": col_it, "ongoal": col_ongoal, "goals": col_goals
    }.items() if v is None]
    if missing:
        raise KeyError(f"Goalies CSV missing columns: {missing}")

    out = df[[col_name, col_team, col_sit, col_gp, col_it, col_ongoal, col_goals]].copy()
    out.columns = ["Goalie", "Team", "situation", "GP", "icetime", "SOG_Against", "GA"]

    out["Team"] = out["Team"].apply(norm_team)
    out["situation"] = out["situation"].astype(str).str.lower()
    out = out[out["situation"].isin(["all", "all situations", "all_situations", "all-situations"])].copy()

    out["GP"] = pd.to_numeric(out["GP"], errors="coerce")
    out["icetime"] = pd.to_numeric(out["icetime"], errors="coerce")
    out["SOG_Against"] = pd.to_numeric(out["SOG_Against"], errors="coerce")
    out["GA"] = pd.to_numeric(out["GA"], errors="coerce")

    out["SV"] = (out["SOG_Against"] - out["GA"]) / out["SOG_Against"]
    out.loc[out["SOG_Against"] <= 0, "SV"] = np.nan

    sec_mask = out["icetime"] > 10000
    min_mask = ~sec_mask
    out["GAA"] = np.nan
    out.loc[min_mask & (out["icetime"] > 0), "GAA"] = out.loc[min_mask, "GA"] * 60.0 / out.loc[min_mask, "icetime"]
    out.loc[sec_mask & (out["icetime"] > 0), "GAA"] = out.loc[sec_mask, "GA"] * 3600.0 / out.loc[sec_mask, "icetime"]

    return out

def build_team_goalie_map(goalies_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for team, grp in goalies_df.groupby("Team"):
        g2 = grp.dropna(subset=["GP"]).sort_values("GP", ascending=False)
        if g2.empty:
            continue
        top = g2.iloc[0]
        m[str(team)] = {
            "Goalie": str(top["Goalie"]),
            "GP": safe_float(top["GP"]),
            "SV": safe_float(top["SV"]),
            "GAA": safe_float(top["GAA"]),
        }
    return m

def resolve_goalie_for_team(
    gdf: pd.DataFrame,
    team_goalie_map: dict[str, dict[str, Any]],
    team_abbr: str,
    starter_name: str | None,
) -> dict[str, Any]:
    team_abbr = norm_team(team_abbr)
    starter_name = (starter_name or "").strip()

    fallback = team_goalie_map.get(team_abbr, {}).copy()
    if fallback:
        fallback["Source"] = "moneypuck_team_proxy"
    else:
        fallback = {"Goalie": "", "GP": None, "SV": None, "GAA": None, "Source": "none"}

    if not starter_name:
        return fallback

    sub = gdf[gdf["Team"] == team_abbr].copy()
    if sub.empty:
        fallback["Goalie"] = starter_name
        fallback["Source"] = "dailyfaceoff_name_only"
        return fallback

    want = _norm_name(starter_name)
    sub["__n"] = sub["Goalie"].astype(str).map(_norm_name)

    hit = sub[sub["__n"] == want]
    if hit.empty:
        want_last = want.split(" ")[-1] if want else ""
        if want_last:
            hit = sub[sub["__n"].str.contains(rf"\b{re.escape(want_last)}\b", regex=True, na=False)]

    if hit.empty:
        fallback["Goalie"] = starter_name
        fallback["Source"] = "dailyfaceoff_name_fallback_stats"
        return fallback

    hit_row = hit.sort_values("GP", ascending=False).iloc[0]
    return {
        "Goalie": starter_name,
        "GP": safe_float(hit_row.get("GP")),
        "SV": safe_float(hit_row.get("SV")),
        "GAA": safe_float(hit_row.get("GAA")),
        "Source": "dailyfaceoff_name_and_stats",
    }

def goalie_weak_score(gp: Optional[float], sv: Optional[float], gaa: Optional[float]) -> float:
    if gp is None or gp < MIN_GOALIE_GP:
        return 50.0
    sv_score = 50.0 if sv is None else clamp((0.920 - sv) / (0.920 - 0.880) * 100.0)
    gaa_score = 50.0 if gaa is None else clamp((gaa - 2.20) / (3.80 - 2.20) * 100.0)
    return round(0.55 * sv_score + 0.45 * gaa_score, 1)


# ============================
# NHL logs cache + parsing
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

def pick_sog_from_row(r: dict) -> Optional[int]:
    """
    Robust SOG getter from NHL game-log rows.
    Prefers true SOG fields; only uses generic "shots" when it almost certainly is SOG.
    """
    v = _pick_stat(r, ("shotsOnGoal", "shots_on_goal", "sog"))
    if v is not None:
        return v

    for nk in ("playerGameStats", "stats", "skaterStats", "summary"):
        sub = r.get(nk)
        if isinstance(sub, dict):
            v2 = _pick_stat(sub, ("shotsOnGoal", "shots_on_goal", "sog"))
            if v2 is not None:
                return v2

    shots = _pick_stat(r, ("shots",))
    if shots is None:
        for nk in ("playerGameStats", "stats", "skaterStats", "summary"):
            sub = r.get(nk)
            if isinstance(sub, dict):
                shots = _pick_stat(sub, ("shots",))
                if shots is not None:
                    break
    if shots is None:
        return None

    attempts_keys = {"shotAttempts", "shot_attempts", "corsi", "iCF", "icf", "attempts"}

    def keys_of(x):
        return set(map(str, x.keys())) if isinstance(x, dict) else set()

    all_keys = set(map(str, r.keys()))
    for nk in ("playerGameStats", "stats", "skaterStats", "summary"):
        sub = r.get(nk)
        if isinstance(sub, dict):
            all_keys |= keys_of(sub)

    if all_keys & attempts_keys:
        sa = _pick_stat(r, ("shotAttempts", "shot_attempts", "iCF", "icf", "attempts"))
        if sa is None:
            for nk in ("playerGameStats", "stats", "skaterStats", "summary"):
                sub = r.get(nk)
                if isinstance(sub, dict):
                    sa = _pick_stat(sub, ("shotAttempts", "shot_attempts", "iCF", "icf", "attempts"))
                    if sa is not None:
                        break
        try:
            s_shots = int(shots)
            s_att = int(sa) if sa is not None else None
        except Exception:
            return None

        # Reject classic inflation
        if s_att is not None and (s_att == s_shots or s_att > 15):
            return None

        # Otherwise allow shots if sane
        if 0 <= s_shots <= 15:
            return s_shots
        return None

    # If no attempts keys exist, allow sane shots
    try:
        s = int(shots)
        if 0 <= s <= 15:
            return s
    except Exception:
        return None
    return None

def compute_lastN_features(payload: Dict[str, Any], n10: int = 10, n5: int = 5) -> Dict[str, Any]:
    rows = _extract_game_rows(payload)

    shots: List[int] = []
    goals: List[int] = []
    assists: List[int] = []

    for r in rows:
        sog = pick_sog_from_row(r)
        g = _pick_stat(r, ("goals", "g"))
        a = _pick_stat(r, ("assists", "a"))

        if sog is None and g is None and a is None:
            continue

        shots.append(int(sog) if sog is not None else 0)
        goals.append(int(g) if g is not None else 0)
        assists.append(int(a) if a is not None else 0)

    v10_shots = shots[:n10]
    v10_goals = goals[:n10]
    v10_assists = assists[:n10]
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

    def drought_since(vals: List[int], thresh: int) -> Optional[int]:
        if not vals:
            return None
        for i, v in enumerate(vals):
            if v >= thresh:
                return i
        return len(vals)

    med10 = float(median(v10_shots)) if len(v10_shots) >= 3 else None
    trim10 = trimmed_mean(v10_shots, 1) if len(v10_shots) >= 5 else None
    avg5 = (sum(v5_shots) / len(v5_shots)) if v5_shots else None

    g5_total = sum(v5_goals) if v5_goals else None
    a5_total = sum(v5_assists) if v5_assists else None
    a10_total = sum(v10_assists) if v10_assists else None

    p10_total = (sum(v10_goals) + sum(v10_assists)) if (v10_goals or v10_assists) else None
    g10_total = sum(v10_goals) if v10_goals else None
    s10_total = sum(v10_shots) if v10_shots else None

    p10 = [(g + a) for g, a in zip(v10_goals, v10_assists)]
    drought_p = drought_since(p10, 1)
    drought_a = drought_since(v10_assists, 1)
    drought_g = drought_since(v10_goals, 1)
    drought_sog2 = drought_since(v10_shots, 2)
    drought_sog3 = drought_since(v10_shots, 3)

    return {
        "Median10_SOG": med10,
        "TrimMean10_SOG": trim10,
        "Avg5_SOG": avg5,
        "G5_total": g5_total,
        "A5_total": a5_total,
        "P10_total": p10_total,
        "G10_total": g10_total,
        "S10_total": s10_total,
        "N_games_found": len(shots),
        "A10_total": a10_total,
        "Drought_P": drought_p,
        "Drought_A": drought_a,
        "Drought_G": drought_g,
        "Drought_SOG2": drought_sog2,
        "Drought_SOG3": drought_sog3,
    }


# ============================
# Scoring / matrices / regression / confidence
# ============================
def sog_med10_green(pos: str) -> float:
    return SOG_MED10_GREEN_DEF if is_defense(pos) else SOG_MED10_GREEN_FWD

def sog_med10_yellow(pos: str) -> float:
    return SOG_MED10_YELLOW_DEF if is_defense(pos) else SOG_MED10_YELLOW_FWD

def heat_from_gap(gap10: Optional[float]) -> str:
    if gap10 is None:
        return "COOL"
    if gap10 >= REG_HOT_GAP:
        return "HOT"
    if gap10 >= REG_WARM_GAP:
        return "WARM"
    return "COOL"

def matrix_sog(ixg_pct: float, med10: Optional[float], pos: str,
               shot_intent_pct: Optional[float] = None, toi_pct: Optional[float] = None,
               team_sf_pct: Optional[float] = None, opp_defweak: Optional[float] = None) -> str:
    if ixg_pct < 72:
        return "Red"
    if med10 is None:
        return "Yellow"

    green_line = sog_med10_green(pos)
    yellow_line = sog_med10_yellow(pos)

    sip = 50.0 if shot_intent_pct is None else float(shot_intent_pct)
    toi = 50.0 if toi_pct is None else float(toi_pct)
    tsf = 50.0 if team_sf_pct is None else float(team_sf_pct)
    odw = 50.0 if opp_defweak is None else float(opp_defweak)

    if ixg_pct >= 91 and med10 >= green_line:
        return "Green"

    if ixg_pct >= 90 and med10 >= (green_line - 0.30) and sip >= 83 and toi >= 60 and tsf >= 60:
        return "Green"

    if ixg_pct >= 80 and med10 >= 3.5 and sip >= 95 and odw >= 60:
        return "Green"

    if med10 >= yellow_line:
        support = 0
        support += 1 if ixg_pct >= 82 else 0
        support += 1 if sip >= 68 else 0
        support += 1 if toi >= 60 else 0
        support += 1 if tsf >= 62 else 0
        support += 1 if odw >= 62 else 0
        if med10 >= (green_line - 0.20) and support >= 4:
            return "Green"
        return "Yellow"

    return "Red"

def matrix_goal_v2(ixg_pct: float, med10: Optional[float], pos: str,
                   g5_total: Optional[int], goalie_weak: float) -> str:
    if ixg_pct < 88:
        return "Red"
    if med10 is None:
        return "Yellow"
    need = 3.3 if not is_defense(pos) else 2.8
    if med10 < (need - 0.5):
        return "Red"
    if med10 >= need and (g5_total or 0) >= 2:
        return "Green"
    if med10 >= need and goalie_weak >= 70:
        return "Green"
    return "Yellow"

def matrix_points_v2(ixa_pct: float, v2_stab: Optional[float], reg_heat_p: str = "COOL",
                     toi_pct: Optional[float] = None, team_xgf_pct: Optional[float] = None,
                     opp_defweak: Optional[float] = None) -> str:
    stab = 50.0 if v2_stab is None else float(v2_stab)
    toi = 50.0 if toi_pct is None else float(toi_pct)
    tx = 50.0 if team_xgf_pct is None else float(team_xgf_pct)
    dw = 50.0 if opp_defweak is None else float(opp_defweak)

    if ixa_pct < 78:
        return "Red" if ixa_pct < 70 else "Yellow"

    if ixa_pct >= 90 and stab >= 62:
        return "Green"
    if ixa_pct >= 90 and stab >= 65 and toi >= 55:
        return "Green"
    if ixa_pct >= 94 and stab >= 60 and reg_heat_p == "HOT":
        return "Green"
    if ixa_pct >= 90 and stab >= 63 and tx >= 60:
        return "Green"
    if ixa_pct >= 90 and stab >= 63 and dw >= 60:
        return "Green"

    return "Yellow"

def matrix_assists_v1(ixa_pct: float, v2_stab: Optional[float], reg_heat_a: str = "COOL",
                      toi_pct: Optional[float] = None, team_xgf_pct: Optional[float] = None,
                      opp_defweak: Optional[float] = None, shot_assists60: Optional[float] = None) -> str:
    stab = 50.0 if v2_stab is None else float(v2_stab)
    toi = 50.0 if toi_pct is None else float(toi_pct)
    tx  = 50.0 if team_xgf_pct is None else float(team_xgf_pct)
    dw  = 50.0 if opp_defweak is None else float(opp_defweak)

    sa60 = 0.0 if shot_assists60 is None or (isinstance(shot_assists60, float) and math.isnan(shot_assists60)) else float(shot_assists60)
    sa_pct = clamp(sa60 * 20.0)

    if ixa_pct < 78:
        return "Red" if ixa_pct < 70 else "Yellow"

    if ixa_pct >= 92 and stab >= 58:
        return "Green"
    if ixa_pct >= 90 and stab >= 60 and reg_heat_a == "HOT":
        return "Green"
    if ixa_pct >= 90 and stab >= 63 and tx >= 60:
        return "Green"
    if ixa_pct >= 92 and stab >= 58 and dw >= 60:
        return "Green"
    if ixa_pct >= 90 and stab >= 60 and sa_pct >= 70 and toi >= 55:
        return "Green"

    return "Yellow"

def conf_sog(ixg_pct: float, shot_intent_pct: float, defweak: float, goalieweak: float, toi_pct: float) -> int:
    base = 0.50 * ixg_pct + 0.28 * shot_intent_pct + 0.15 * defweak + 0.12 * goalieweak
    base += USAGE_WEIGHT_SOG * (toi_pct - 50.0)
    return int(round(clamp(base)))

def conf_goal(ixg_pct: float, g5: Optional[int], defweak: float, goalieweak: float, toi_pct: float) -> int:
    g5s = 50.0 if g5 is None else clamp((g5 / 5.0) * 100.0)
    base = 0.60 * ixg_pct + 0.18 * g5s + 0.10 * defweak + 0.12 * goalieweak
    base += 0.05 * (toi_pct - 50.0)
    return int(round(clamp(base)))

def conf_points(ixa_pct: float, p10_gap: Optional[float], stab: float, defweak: float, goalieweak: float, toi_pct: float) -> int:
    reg = 65.0 if p10_gap is None else clamp((p10_gap / 4.0) * 100.0)
    base = 0.52 * ixa_pct + 0.10 * stab + 0.10 * defweak + 0.08 * goalieweak + 0.14 * reg
    base += USAGE_WEIGHT_POINTS * (toi_pct - 50.0)
    return int(round(clamp(base)))

def conf_assists(ixa_pct: float, a10_gap: Optional[float], stab: float, defweak: float,
                 team_xgf_pct: float, toi_pct: float, sa60: Optional[float] = None) -> int:
    reg = 65.0 if a10_gap is None else clamp((a10_gap / 3.0) * 100.0)
    sa = 50.0
    if sa60 is not None and not (isinstance(sa60, float) and math.isnan(sa60)):
        sa = clamp(float(sa60) * 20.0)

    base = (
        0.50 * ixa_pct +
        0.10 * stab +
        0.10 * defweak +
        0.10 * team_xgf_pct +
        0.10 * reg +
        0.10 * sa
    )
    base += 0.06 * (toi_pct - 50.0)
    return int(round(clamp(base)))


# ============================
# v2 stability + opponent defweak (0-100)
# ============================
def add_v2_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["i5v5_points60", "i5v5_shotAssists60", "i5v5_iCF60"]:
        if col not in out.columns:
            out[col] = np.nan

    for col in ["opp_5v5_xGA60", "opp_5v5_HDCA60", "opp_5v5_SlotSA60"]:
        if col not in out.columns:
            out[col] = np.nan

    pts_p = pct_rank(out["i5v5_points60"]).fillna(50)
    sa_p  = pct_rank(out["i5v5_shotAssists60"]).fillna(50)
    icf_p = pct_rank(out["i5v5_iCF60"]).fillna(50)
    out["v2_player_stability"] = (0.50 * pts_p + 0.25 * sa_p + 0.25 * icf_p).round(1)

    xga_p  = pct_rank(out["opp_5v5_xGA60"]).fillna(50)
    hdca_p = pct_rank(out["opp_5v5_HDCA60"]).fillna(50)
    slot_p = pct_rank(out["opp_5v5_SlotSA60"]).fillna(50)
    out["v2_defense_vulnerability"] = (0.45 * xga_p + 0.35 * hdca_p + 0.20 * slot_p).round(1)

    return out


# ============================
# Star prior (TalentMult) + tiers
# ============================
def add_star_prior(sk: pd.DataFrame) -> pd.DataFrame:
    out = sk.copy()
    toi_pct = pct_rank(out["TOI_per_game"]).fillna(50)
    out["TOI_Pct"] = toi_pct.round(1)

    star = (0.45 * out["iXG_pct"].fillna(50) + 0.35 * out["iXA_pct"].fillna(50) + 0.20 * toi_pct).fillna(50)
    out["StarScore"] = star.round(1)

    mult = 1.0 + ((star - 50.0) / 100.0) * TALENT_MULT_STRENGTH
    mult = mult.clip(TALENT_MULT_MIN, TALENT_MULT_MAX)
    out["TalentMult"] = mult.round(3)
    return out

def add_talent_tiers(sk: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    out = sk.copy()

    starscore = pd.to_numeric(out.get("StarScore", 50.0), errors="coerce").fillna(50.0)
    toi_pct   = pd.to_numeric(out.get("TOI_Pct", 50.0), errors="coerce").fillna(50.0)

    ixg = pd.to_numeric(out.get("iXG_pct", 50.0), errors="coerce").fillna(50.0)
    ixa = pd.to_numeric(out.get("iXA_pct", 50.0), errors="coerce").fillna(50.0)
    top_pct = pd.concat([ixg, ixa], axis=1).max(axis=1)

    sip = pd.to_numeric(out.get("ShotIntent_Pct", 50.0), errors="coerce").fillna(50.0)
    
    out["PPG"] = safe_ppg(out).round(3)
    ppg = pd.to_numeric(out["PPG"], errors="coerce")


    names = out.get("Player", "").astype(str)
    elite_seed_hit = names.isin(ELITE_SEED)
    star_seed_hit  = names.isin(STAR_SEED)

    # -------------------------
    # ELITE (hardened): require 2-of-3 proofs OR elite seed OR elite shot-intent lane
    # proofs: StarScore, TOI, TopPct
    # -------------------------
    elite_a = (starscore >= ELITE_STARSCORE)
    elite_b = (toi_pct   >= ELITE_TOI_PCT)
    elite_c = (top_pct   >= ELITE_TOPPCT)

    elite_proofs = elite_a.astype(int) + elite_b.astype(int) + elite_c.astype(int)
    
    elite_d = (ppg >= ELITE_MIN_PPG)
    
    star_e = (ppg >= STAR_MIN_PPG)


    # PPG (best-effort; if missing, it'll be NaN and fail the gate unless seeded)
    ppg = pd.to_numeric(out.get("PPG"), errors="coerce")

    elite_ppg_gate = (ppg >= ELITE_MIN_PPG)

    # keep your shot-intent elite lane, but require strong TOI + PPG too
    elite_shot_lane = (
        (sip >= ELITE_SHOTINTENT_PCT) &
        (top_pct >= ELITE_TOPPCT) &
        (toi_pct >= ELITE_TOI_PCT) &
        elite_ppg_gate
    )

    # ELITE: seeded OR (3-of-3 proofs AND PPG gate) OR (elite shot lane AND PPG gate)
    elite = elite_seed_hit | ((elite_proofs >= 3) & elite_ppg_gate) | elite_shot_lane
    # -------------------------
    # STAR (hardened): require 3-of-4 proofs OR star seed
    # proofs: StarScore, TOI, TopPct, ShotIntent
    # Uses your STAR_* constants but forces combined strength.
    # -------------------------
    # -------------------------
    # STAR (hardened): require 3-of-4 proofs OR star seed
    # proofs: StarScore, TOI, TopPct, ShotIntent
    # Uses your STAR_* constants but forces combined strength.
    # -------------------------
    star_a = (starscore >= STAR_STARSCORE)
    star_b = (toi_pct   >= STAR_TOI_PCT)
    star_c = (top_pct   >= STAR_TOPPCT)
    star_d = (sip        >= STAR_SHOTINTENT_PCT)

    star_proofs = star_a.astype(int) + star_b.astype(int) + star_c.astype(int) + star_d.astype(int)

    star = star_seed_hit | (star_proofs >= 3)

    # Make tiers mutually exclusive (STAR excludes ELITE)
    out["Is_Elite"] = elite
    out["Is_Star"]  = star & ~elite

    out["Talent_Tier"] = "NONE"
    out.loc[out["Is_Star"],  "Talent_Tier"] = "STAR"
    out.loc[out["Is_Elite"], "Talent_Tier"] = "ELITE"

    if debug:
        print("\n[TALENT] Tier counts:", out["Talent_Tier"].value_counts(dropna=False).to_dict())

        # Optional quick debug: see why STAR is large
        try:
            print("[TALENT] STAR proofs breakdown (mean %):",
                  {
                      "StarScore": round(100.0 * star_a.mean(), 1),
                      "TOI":       round(100.0 * star_b.mean(), 1),
                      "TopPct":    round(100.0 * star_c.mean(), 1),
                      "ShotIntent":round(100.0 * star_d.mean(), 1),
                      "3of4":      round(100.0 * (star_proofs >= 3).mean(), 1),
                  })
        except Exception:
            pass

    return out

def get_team_recent_gf(sess: requests.Session, team_abbrev: str, n_games: int = 5, debug: bool = False) ->  tuple[int, float, int]:
    """
    Returns (gf_sum, gf_avg, n_used) over the last n completed games.
    Uses api-web.nhle.com club schedule endpoint.
    """
    team_abbrev = norm_team(team_abbrev)
    url = f"https://api-web.nhle.com/v1/club-schedule-season/{team_abbrev}/now"

    try:
        data = http_get_json(sess, url)
    except Exception as e:
        if debug:
            print(f"[TEAM_GF] fetch failed for {team_abbrev}: {type(e).__name__}: {e}")
        return 0, 0.0, 0

    games = data.get("games", []) or []

    # Keep only completed games
    completed = []
    for g in games:
        state = str(g.get("gameState", "") or "").upper()
        if state in ("OFF", "FINAL", "FINAL_OT", "FINAL_SO"):
            completed.append(g)

    # Sort by date ascending, then take the most recent n
    def _gdate(x):
        return str(x.get("gameDate", "") or x.get("startTimeUTC", "") or "")
    completed.sort(key=_gdate)

    last = completed[-n_games:] if completed else []
    gf = 0

    for g in last:
        home = g.get("homeTeam", {}) or {}
        away = g.get("awayTeam", {}) or {}

        h_abbrev = str(home.get("abbrev", "") or home.get("teamAbbrev", "") or "").upper()
        a_abbrev = str(away.get("abbrev", "") or away.get("teamAbbrev", "") or "").upper()

        h_score = int(home.get("score", 0) or 0)
        a_score = int(away.get("score", 0) or 0)

        if h_abbrev == team_abbrev:
            gf += h_score
        elif a_abbrev == team_abbrev:
            gf += a_score

    n_used = len(last)
    avg = (gf / n_used) if n_used > 0 else 0.0
    return gf, round(avg, 2), n_used



# ============================
# Drought bumps (game regression)
# ============================
def drought_bump(tier: str, market: str, drought: Optional[int]) -> tuple[int, bool]:
    if drought is None:
        return 0, False

    d = int(drought)
    bump = 0
    flag = False

    if market in {"SOG", "POINTS"}:
        if tier == "ELITE":
            if d >= 1: bump = 2
            if d >= 2: bump = 5
            if d >= 3: bump = 8; flag = True
        elif tier == "STAR":
            if d >= 2: bump = 2
            if d >= 3: bump = 5
            if d >= 4: bump = 8; flag = True
        else:
            if d >= 3: bump = 2
            if d >= 4: bump = 5
            if d >= 5: bump = 8; flag = True

    if market in {"GOAL", "ASSISTS"}:
        if tier == "ELITE":
            if d >= 2: bump = 4
            if d >= 3: bump = 7; flag = True
        elif tier == "STAR":
            if d >= 3: bump = 3
            if d >= 4: bump = 6; flag = True
        else:
            if d >= 4: bump = 2
            if d >= 5: bump = 5; flag = True

    return bump, flag



# ============================
# MAIN build
# ============================

def build_tracker(today_local: date, debug: bool = False) -> str:
    ensure_dirs()
    sess = http_session()

    print(f"\nNHL EDGE TOOL — v7.2 — {today_local.isoformat()}\n")

    games = nhl_schedule_today(sess, today_local)
    if not games:
        raise RuntimeError("No games found for today.")

    teams_playing = set()
    game_map: Dict[str, str] = {}
    opp_map: Dict[str, str] = {}
    game_time_utc: Dict[str, str] = {}
    game_time_local: Dict[str, str] = {}

    print("Matchups:")
    for g in games:
        away, home = g["away"], g["home"]
        start_utc = str(g.get("startTimeUTC", "") or "")
        matchup = f"{away}@{home}"
        print(f"  {matchup}")

        teams_playing.add(away)
        teams_playing.add(home)

        game_map[away] = matchup
        game_map[home] = matchup
        opp_map[away] = home
        opp_map[home] = away

        try:
            dt_utc = datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
            dt_local = dt_utc.astimezone(ZoneInfo(LOCAL_TZ))
            start_local = dt_local.strftime("%I:%M %p").lstrip("0")
        except Exception:
            start_local = ""

        game_time_utc[matchup] = start_utc
        game_time_local[matchup] = start_local

    print("")

    # -------------------------
    # TEAM GF L5 HARD GATE (2.70) — affects GOAL/POINTS/ASSISTS
    # -------------------------
    team_gf = {}
    for t in sorted(teams_playing):
        gf, avg, n_used = get_team_recent_gf(sess, t, n_games=TEAM_GF_WINDOW, debug=debug)
        team_gf[t] = {"GF_L5": gf, "GF_Avg_L5": avg, "GF_N": n_used, "GF_Gate": (avg >= TEAM_GF_MIN_AVG)}

    
    if debug:
        bad = {k: v for k, v in team_gf.items() if not v.get("GF_Gate")}
        print(f"[TEAM_GF] gate={TEAM_GF_MIN_AVG:.2f} window={TEAM_GF_WINDOW} | failing teams:", bad)


    # MoneyPuck skaters
    sk_raw = load_moneypuck_best_effort(sess, "skaters")
    sk = normalize_skaters_all(sk_raw, debug=debug)
    sk = sk[sk["Team"].isin(teams_playing)].copy()
    if sk.empty:
        raise RuntimeError("No skaters matched today's teams. Team abbrev mismatch.")

    # Goalies (MoneyPuck)
    g_raw = load_moneypuck_best_effort(sess, "goalies")
    gdf = normalize_goalies(g_raw)
    team_goalie = build_team_goalie_map(gdf)

    # DFO starter map
    dfo_map = fetch_dailyfaceoff_starters(today_local, debug=debug)

    # DFO injuries
    inj_df = fetch_dfo_injuries_for_teams(teams_playing, debug=debug)

    # Schedule columns
    sk["Game"] = sk["Team"].map(game_map).fillna("")
    sk["Opp"] = sk["Team"].map(opp_map).fillna("")
    sk["StartTimeUTC"] = sk["Game"].map(game_time_utc).fillna("")
    sk["StartTimeLocal"] = sk["Game"].map(game_time_local).fillna("")

    # Goalie info
    def _goalie_pack(r: pd.Series) -> pd.Series:
        opp_team = norm_team(str(r.get("Opp", "")).strip())
        starter_name = dfo_map.get(opp_team, {}).get("goalie", "")
        g = resolve_goalie_for_team(gdf, team_goalie, opp_team, starter_name)
        return pd.Series({
            "Opp_Goalie": g.get("Goalie", ""),
            "Opp_GP": g.get("GP", None),
            "Opp_SV": g.get("SV", None),
            "Opp_GAA": g.get("GAA", None),
            "Opp_Goalie_Source": g.get("Source", ""),
            "Opp_Goalie_Status": dfo_map.get(opp_team, {}).get("status", ""),
        })

    # attach to skaters
    sk["Team_GF_L5"] = sk["Team"].map(lambda x: team_gf.get(norm_team(x), {}).get("GF_L5", 0))
    sk["Team_GF_Avg_L5"] = sk["Team"].map(lambda x: team_gf.get(norm_team(x), {}).get("GF_Avg_L5", 0.0))
    sk["Team_GF_Gate"] = sk["Team"].map(lambda x: bool(team_gf.get(norm_team(x), {}).get("GF_Gate", False)))


    goalie_cols = sk.apply(_goalie_pack, axis=1)
    sk = pd.concat([sk, goalie_cols], axis=1)

 
    sk["Goalie_Weak"] = sk.apply(
        lambda r: goalie_weak_score(
            safe_float(r.get("Opp_GP")),
            safe_float(r.get("Opp_SV")),
            safe_float(r.get("Opp_GAA")),
        ),
        axis=1
    )

    # 5v5 player
    try:
        sk5 = normalize_skaters_5v5(sk_raw, debug=debug)
        sk = sk.merge(sk5, on=["playerId", "Team"], how="left")
    except Exception as e:
        print(f"WARNING: Player 5v5 unavailable; continuing neutral. ({type(e).__name__}: {e})")
        for c in ["i5v5_points60", "i5v5_primaryAssists60", "i5v5_shotAssists60", "i5v5_iCF60"]:
            sk[c] = np.nan

    # Assist volume proxy
    sk["Assist_Volume"] = (
        pd.to_numeric(sk.get("i5v5_shotAssists60"), errors="coerce")
          .fillna(0.0) * 12.0
    ).round(2)


    # 5v5 teams
    try:
        teams_raw = load_moneypuck_best_effort(sess, "teams")
        t5 = normalize_teams_5v5(teams_raw, debug=debug)

        t5_opp = t5[["Team", "opp_5v5_xGA60", "opp_5v5_HDCA60", "opp_5v5_SlotSA60"]].copy()
        sk = sk.merge(t5_opp, left_on="Opp", right_on="Team", how="left")
        sk = sk.drop(columns=["Team_y"], errors="ignore")
        sk = sk.rename(columns={"Team_x": "Team"})

        t5_team = t5[["Team", "team_5v5_xGF60", "team_5v5_SF60"]].copy()
        t5_team["team_5v5_SF60_pct"]  = pct_rank(t5_team["team_5v5_SF60"]).fillna(50).round(1)
        t5_team["team_5v5_xGF60_pct"] = pct_rank(t5_team["team_5v5_xGF60"]).fillna(50).round(1)

        sk = sk.merge(t5_team, on="Team", how="left")

    except Exception as e:
        print(f"WARNING: Opponent/team 5v5 unavailable; filling neutral. ({type(e).__name__}: {e})")
        sk["opp_5v5_xGA60"] = np.nan
        sk["opp_5v5_HDCA60"] = np.nan
        sk["opp_5v5_SlotSA60"] = np.nan
        sk["team_5v5_xGF60"] = np.nan
        sk["team_5v5_SF60"] = np.nan
        sk["team_5v5_SF60_pct"] = np.nan
        sk["team_5v5_xGF60_pct"] = np.nan

    for c in ["team_5v5_xGF60", "team_5v5_SF60"]:
        if c not in sk.columns:
            sk[c] = np.nan
        sk[c] = pd.to_numeric(sk[c], errors="coerce").fillna(50.0)
    sk["team_5v5_SF60_pct"] = pct_rank(sk["team_5v5_SF60"]).fillna(50).round(1)
    sk["team_5v5_xGF60_pct"] = pct_rank(sk["team_5v5_xGF60"]).fillna(50).round(1)

    for c in ["opp_5v5_xGA60", "opp_5v5_HDCA60", "opp_5v5_SlotSA60"]:
        if c not in sk.columns:
            sk[c] = np.nan
        sk[c] = pd.to_numeric(sk[c], errors="coerce")
    if sk["opp_5v5_SlotSA60"].isna().mean() >= 0.99:
        if debug:
            print("\n[DEBUG] SlotSA60 essentially missing for all teams. Filling neutral 50.")
        sk["opp_5v5_SlotSA60"] = 50.0
    sk["opp_5v5_xGA60"] = sk["opp_5v5_xGA60"].fillna(50.0)
    sk["opp_5v5_HDCA60"] = sk["opp_5v5_HDCA60"].fillna(50.0)
    sk["opp_5v5_SlotSA60"] = sk["opp_5v5_SlotSA60"].fillna(50.0)

    # v2 scores
    sk = add_v2_scores(sk)
    sk["Opp_DefWeak"] = sk["v2_defense_vulnerability"].fillna(50.0)

    # star prior + toi pct
    sk = add_star_prior(sk)

    # injuries
    sk = apply_injury_dfo(sk, inj_df, debug=debug)

    # Availability rules:
    # - OUT/IR/Scratch removed
    # - GTD stays but gets confidence penalty
    sk["Available"] = ~sk["Injury_Status"].isin(["IR", "Out", "Scratch"])
    sk["Unavailable_Reason"] = np.where(sk["Available"], "", sk["Injury_Status"])

    # candidate pool for logs
    sk["TopPct"] = sk[["iXG_pct", "iXA_pct"]].max(axis=1)
    cand = sk[sk["TopPct"] >= PROCESS_MIN_PCT].copy()
    cand = cand.sort_values("TopPct", ascending=False).head(CAND_CAP)
    cand_ids = sorted({int(x) for x in cand["playerId"].dropna().tolist()})

    print(f"Fetching NHL logs for {len(cand_ids)} players (cap={CAND_CAP}) ...\n")

    cache = load_cache(today_local)

    def fetch_one(pid: int) -> Tuple[int, Optional[Dict[str, Any]]]:
        key = str(pid)
        if key in cache:
            return pid, cache[key]
        data = nhle_player_gamelog_now(sess, pid)
        time.sleep(HTTP_SLEEP_SEC)
        if data is not None:
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
            feats_rows.append({"playerId": pid})
            continue
        feats = compute_lastN_features(payload, 10, 5)
        feats_rows.append({"playerId": pid, **feats})

    feats_df = pd.DataFrame(feats_rows)
    sk = sk.merge(feats_df, on="playerId", how="left")

    # ShotIntent proxy
    med10 = pd.to_numeric(sk.get("Median10_SOG", np.nan), errors="coerce")
    avg5 = pd.to_numeric(sk.get("Avg5_SOG", np.nan), errors="coerce")
    sk["ShotIntent"] = (0.65 * med10 + 0.35 * avg5)
    sk["ShotIntent_Pct"] = pct_rank(sk["ShotIntent"]).fillna(50).round(1)

    # Talent tiers
    sk = add_talent_tiers(sk, debug=debug)

    # -------------------------
    # Context-only: Elite PPG cohort (SAFE — no scoring impact)
    # -------------------------
    # PPG already computed inside add_talent_tiers(): sk["PPG"]
    # Elite mask should use your tier system (NOT Elite_Flag).
    elite_mask = sk.get("Talent_Tier", "NONE").astype(str).str.upper().eq("ELITE")

    elite_ppg_avg = np.nan
    if elite_mask.any():
        elite_ppg_avg = float(pd.to_numeric(sk.loc[elite_mask, "PPG"], errors="coerce").mean())

    sk["Elite_PPG_Avg"] = elite_ppg_avg
    sk["PPG_vs_EliteAvg"] = pd.to_numeric(sk.get("PPG"), errors="coerce") - elite_ppg_avg

    def elite_bucket(ppg: Any) -> str:
        try:
            if ppg is None or (isinstance(ppg, float) and math.isnan(ppg)):
                return ""
            p = float(ppg)
        except Exception:
            return ""
        if p >= 1.10:
            return "Elite 1.10+"
        if p >= 0.95:
            return "Elite 0.95-1.10"
        if p >= 0.80:
            return "Elite 0.80-0.95"
        return "Elite <0.80"

    sk["Elite_PPG_Bucket"] = np.where(
        elite_mask,
        sk["PPG"].apply(elite_bucket),
        ""
    )

    
  

    # Simple UI tag (emoji label)
    sk["Tier_Tag"] = np.where(
        sk.get("Talent_Tier", "NONE").astype(str).str.upper().eq("ELITE"),
        "👑 ELITE",
        np.where(
            sk.get("Talent_Tier", "NONE").astype(str).str.upper().eq("STAR"),
            "⭐ STAR",
            ""
        )
    )

    

    # Regression expectations (TalentMult only here)
    gp = pd.to_numeric(sk["games_played"], errors="coerce")
    exp_p10 = ((pd.to_numeric(sk["iXG_raw"], errors="coerce").fillna(0) + pd.to_numeric(sk["iXA_raw"], errors="coerce").fillna(0)) / gp) * 10.0
    exp_g10 = (pd.to_numeric(sk["iXG_raw"], errors="coerce").fillna(0) / gp) * 10.0
    exp_a10 = (pd.to_numeric(sk["iXA_raw"], errors="coerce").fillna(0) / gp) * 10.0

    tm = pd.to_numeric(sk["TalentMult"], errors="coerce").fillna(1.0)
    exp_p10 = exp_p10 * tm
    exp_g10 = exp_g10 * tm
    exp_a10 = exp_a10 * tm

    sk["Exp_P_10"] = exp_p10.round(2)
    sk["Exp_G_10"] = exp_g10.round(2)
    sk["Exp_A_10"] = exp_a10.round(2)

    sk["L10_P"] = pd.to_numeric(sk.get("P10_total", np.nan), errors="coerce")
    sk["L10_G"] = pd.to_numeric(sk.get("G10_total", np.nan), errors="coerce")
    sk["L10_S"] = pd.to_numeric(sk.get("S10_total", np.nan), errors="coerce")
    sk["L10_A"] = pd.to_numeric(sk.get("A10_total", np.nan), errors="coerce")

    sk["Exp_S_10"] = (pd.to_numeric(sk.get("ShotIntent", np.nan), errors="coerce") * 10.0).round(2)

    sk["Reg_Gap_P10"] = (sk["Exp_P_10"] - sk["L10_P"]).round(2)
    sk["Reg_Gap_G10"] = (sk["Exp_G_10"] - sk["L10_G"]).round(2)
    sk["Reg_Gap_S10"] = (sk["Exp_S_10"] - sk["L10_S"]).round(2)
    sk["Reg_Gap_A10"] = (sk["Exp_A_10"] - sk["L10_A"]).round(2)

    sk["Reg_Heat_A"] = sk["Reg_Gap_A10"].apply(lambda x: heat_from_gap(safe_float(x)))
    sk["Reg_Heat_P"] = sk["Reg_Gap_P10"].apply(lambda x: heat_from_gap(safe_float(x)))
    sk["Reg_Heat_G"] = sk["Reg_Gap_G10"].apply(lambda x: heat_from_gap(safe_float(x)))
    sk["Reg_Heat_S"] = sk["Reg_Gap_S10"].apply(lambda x: heat_from_gap(safe_float(x)))

    # Matrices
    sk["Matrix_SOG"] = sk.apply(
        lambda r: matrix_sog(
            float(r.get("iXG_pct", 50)),
            safe_float(r.get("Median10_SOG")),
            str(r.get("Pos", "F")),
            shot_intent_pct=safe_float(r.get("ShotIntent_Pct")),
            toi_pct=safe_float(r.get("TOI_Pct")),
            team_sf_pct=safe_float(r.get("team_5v5_SF60_pct")),
            opp_defweak=safe_float(r.get("Opp_DefWeak")),
        ),
        axis=1
    )

    sk["Matrix_Assists"] = sk.apply(
        lambda r: matrix_assists_v1(
            ixa_pct=float(r.get("iXA_pct", 50)),
            v2_stab=safe_float(r.get("v2_player_stability")),
            reg_heat_a=str(r.get("Reg_Heat_A", "COOL")),
            toi_pct=safe_float(r.get("TOI_Pct")),
            team_xgf_pct=safe_float(r.get("team_5v5_xGF60_pct")),
            opp_defweak=safe_float(r.get("Opp_DefWeak")),
            shot_assists60=safe_float(r.get("i5v5_shotAssists60")),
        ),
        axis=1
    )

    sk["Matrix_Goal"] = sk.apply(
        lambda r: matrix_goal_v2(
            ixg_pct=float(r.get("iXG_pct", 50)),
            med10=safe_float(r.get("Median10_SOG")),
            pos=str(r.get("Pos", "F")),
            g5_total=safe_int(r.get("G5_total")),
            goalie_weak=float(r.get("Goalie_Weak", 50)),
        ),
        axis=1
    )

    sk["Matrix_Points"] = sk.apply(
        lambda r: matrix_points_v2(
            ixa_pct=float(r.get("iXA_pct", 50)),
            v2_stab=safe_float(r.get("v2_player_stability")),
            reg_heat_p=str(r.get("Reg_Heat_P", "COOL")),
            toi_pct=safe_float(r.get("TOI_Pct")),
            team_xgf_pct=safe_float(r.get("team_5v5_xGF60_pct")),
            opp_defweak=safe_float(r.get("Opp_DefWeak")),
        ),
        axis=1
    )

    # Confidence (base)
    sk["Conf_SOG"] = sk.apply(
        lambda r: conf_sog(
            float(r.get("iXG_pct", 50)),
            float(r.get("ShotIntent_Pct", 50)),
            float(r.get("Opp_DefWeak", 50)),
            float(r.get("Goalie_Weak", 50)),
            float(r.get("TOI_Pct", 50)),
        ),
        axis=1
    )
    sk["Conf_Goal"] = sk.apply(
        lambda r: conf_goal(
            float(r.get("iXG_pct", 50)),
            safe_int(r.get("G5_total")),
            float(r.get("Opp_DefWeak", 50)),
            float(r.get("Goalie_Weak", 50)),
            float(r.get("TOI_Pct", 50)),
        ),
        axis=1
    )
    sk["Conf_Points"] = sk.apply(
        lambda r: conf_points(
            float(r.get("iXA_pct", 50)),
            safe_float(r.get("Reg_Gap_P10")),
            float(r.get("v2_player_stability", 50)),
            float(r.get("Opp_DefWeak", 50)),
            float(r.get("Goalie_Weak", 50)),
            float(r.get("TOI_Pct", 50)),
        ),
        axis=1
    )
    sk["Conf_Assists"] = sk.apply(
        lambda r: conf_assists(
            float(r.get("iXA_pct", 50)),
            safe_float(r.get("Reg_Gap_A10")),
            float(r.get("v2_player_stability", 50)),
            float(r.get("Opp_DefWeak", 50)),
            float(r.get("team_5v5_xGF60_pct", 50)),
            float(r.get("TOI_Pct", 50)),
            safe_float(r.get("i5v5_shotAssists60")),
        ),
        axis=1
    )

    # -------------------------
    # Injury adjustment into confidence (GTD down, ROLE+ up)
    # -------------------------
    inj_adj = (
        pd.to_numeric(sk.get("Injury_DFO_Score"), errors="coerce")
        .fillna(0.0)
        * INJURY_CONF_MULT
    ).clip(-6.0, 2.0)

    for c in ["Conf_SOG", "Conf_Goal", "Conf_Points", "Conf_Assists"]:
        sk[c] = (
            pd.to_numeric(sk[c], errors="coerce")
            .fillna(0)
            .add(inj_adj)
            .clip(0, 100)
            .astype(int)
        )

    # -------------------------
    # Game regression bumps (drought)
    # -------------------------
    sk["Drought_SOG"] = np.where(
        sk["Pos"].astype(str).str.upper().isin(["D", "LD", "RD"]),
        pd.to_numeric(sk.get("Drought_SOG2"), errors="coerce"),
        pd.to_numeric(sk.get("Drought_SOG3"), errors="coerce"),
    )

    def _apply_drought(r: pd.Series) -> pd.Series:
        tier = str(r.get("Talent_Tier", "NONE")).upper()
        if tier not in {"ELITE", "STAR"}:
            tier = "NONE"

        b_s, f_s = drought_bump(tier, "SOG", safe_int(r.get("Drought_SOG")))
        b_p, f_p = drought_bump(tier, "POINTS", safe_int(r.get("Drought_P")))
        b_a, f_a = drought_bump(tier, "ASSISTS", safe_int(r.get("Drought_A")))
        b_g, f_g = drought_bump(tier, "GOAL", safe_int(r.get("Drought_G")))

        return pd.Series({
            "GameReg_Bump_SOG": b_s,
            "Flag_SOG_Drought": f_s,
            "GameReg_Bump_Points": b_p,
            "Flag_Points_Drought": f_p,
            "GameReg_Bump_Assists": b_a,
            "Flag_Assists_Drought": f_a,
            "GameReg_Bump_Goal": b_g,
            "Flag_Goal_Drought": f_g,
        })

    drought_df = sk.apply(_apply_drought, axis=1)
    sk = pd.concat([sk, drought_df], axis=1)

    # -------------------------
    # Best Drought (ALL markets) — longest drought overall
    # -------------------------
    def _best_drought_all_markets(r: pd.Series) -> tuple[str, int]:
        def _to_int(x) -> int:
            v = pd.to_numeric(x, errors="coerce")
            if pd.isna(v):
                return 0
            return int(v)

        d_p = _to_int(r.get("Drought_P"))
        d_a = _to_int(r.get("Drought_A"))
        d_g = _to_int(r.get("Drought_G"))
        d_s = _to_int(r.get("Drought_SOG"))

        opts = [
            ("GOAL", d_g),
            ("ASSISTS", d_a),
            ("POINTS", d_p),
            ("SOG", d_s),
        ]

        # Longest drought wins; tie-breaker order = GOAL > ASSISTS > POINTS > SOG (because of list order)
        opts.sort(key=lambda x: x[1], reverse=True)
        return opts[0][0], opts[0][1]

    best_d = sk.apply(_best_drought_all_markets, axis=1, result_type="expand")
    sk["Best_Drought_Market"] = best_d[0].astype(str)
    sk["Best_Drought"] = pd.to_numeric(best_d[1], errors="coerce").fillna(0).astype(int)
    sk["Best_Drought_Tag"] = np.where(
        sk["Best_Drought"] > 0,
        sk["Best_Drought_Market"] + " " + sk["Best_Drought"].astype(str),
        ""
    )


    # -------------------------
    # Apply drought bumps into confidence
    # -------------------------
    sk["Conf_SOG"] = (
        pd.to_numeric(sk["Conf_SOG"], errors="coerce")
        .fillna(0)
        .add(pd.to_numeric(sk["GameReg_Bump_SOG"], errors="coerce").fillna(0))
        .clip(0, 100)
        .astype(int)
    )

    sk["Conf_Points"] = (
        pd.to_numeric(sk["Conf_Points"], errors="coerce")
        .fillna(0)
        .add(pd.to_numeric(sk["GameReg_Bump_Points"], errors="coerce").fillna(0))
        .clip(0, 100)
        .astype(int)
    )

    sk["Conf_Assists"] = (
        pd.to_numeric(sk["Conf_Assists"], errors="coerce")
        .fillna(0)
        .add(pd.to_numeric(sk["GameReg_Bump_Assists"], errors="coerce").fillna(0))
        .clip(0, 100)
        .astype(int)
    )

    sk["Conf_Goal"] = (
        pd.to_numeric(sk["Conf_Goal"], errors="coerce")
        .fillna(0)
        .add(pd.to_numeric(sk["GameReg_Bump_Goal"], errors="coerce").fillna(0))
        .clip(0, 100)
        .astype(int)
    )

    # -------------------------
    # HARD FAIL: team scoring environment gate
    # If team GF L5 avg < 2.70 => NO GOAL / POINTS / ASSISTS
    # -------------------------
    gf_fail = ~sk.get("Team_GF_Gate", False).astype(bool)

    sk.loc[gf_fail, ["Conf_Goal", "Conf_Points", "Conf_Assists"]] = 0

    sk.loc[gf_fail, "Matrix_Goal"] = "FAIL_GF"
    sk.loc[gf_fail, "Matrix_Points"] = "FAIL_GF"
    sk.loc[gf_fail, "Matrix_Assists"] = "FAIL_GF"

    # -------------------------
    # Best market
    # -------------------------
    def choose_best(r: pd.Series) -> Tuple[str, int]:
        opts = [
            ("ASSISTS", int(r.get("Conf_Assists", 0) or 0)),
            ("POINTS",  int(r.get("Conf_Points", 0) or 0)),
            ("SOG",     int(r.get("Conf_SOG", 0) or 0)),
            ("GOAL",    int(r.get("Conf_Goal", 0) or 0)),
        ]
        opts.sort(key=lambda x: x[1], reverse=True)
        return opts[0][0], opts[0][1]

    best = sk.apply(choose_best, axis=1, result_type="expand")
    sk["Best_Market"] = best[0]
    sk["Best_Conf"] = best[1]

    # Filter unavailable players
    sk = sk[sk["Available"]].reset_index(drop=True)
    # -------------------------
    # Output tracker
    # -------------------------
    tracker = pd.DataFrame({
        "Date": today_local.isoformat(),
        "Game": sk["Game"].fillna(""),
        "StartTimeLocal": sk.get("StartTimeLocal"),
        "StartTimeUTC": sk.get("StartTimeUTC"),

        "Player": sk["Player"].fillna(""),
        "Team": sk["Team"].fillna(""),
        "Pos": sk["Pos"].fillna("F"),
        "Tier": sk.get("Tier_Tag", ""),


        "Injury_Status": sk.get("Injury_Status", "Healthy"),
        "Injury_Type": sk.get("Injury_Type", "Unknown"),
        "Injury_Text": sk.get("Injury_Text", ""),
        "Injury_Return": sk.get("Injury_Return", ""),
        "Team_Out_Count": sk.get("Team_Out_Count", 0),
        "Injury_DFO_Score": sk.get("Injury_DFO_Score", 0.0),
        "Injury_Badge": sk.get("Injury_Badge", ""),

        "Opp": sk["Opp"].fillna(""),
        "Opp_Goalie": sk["Opp_Goalie"].fillna(""),
        "Opp_SV": sk["Opp_SV"],
        "Opp_GAA": sk["Opp_GAA"],
        "Goalie_Weak": sk["Goalie_Weak"],
        "Opp_Goalie_Status": sk.get("Opp_Goalie_Status", ""),
        "Opp_Goalie_Source": sk.get("Opp_Goalie_Source", ""),

        "iXG%": sk["iXG_pct"].round(1),
        "iXA%": sk["iXA_pct"].round(1),
        "iXA_Source": sk.get("iXA_Source", ""),

        "TOI_per_game": sk["TOI_per_game"].round(2),
        "TOI_Pct": sk["TOI_Pct"].round(1),
        "StarScore": sk["StarScore"].round(1),
        "TalentMult": sk["TalentMult"].round(3),
        "Talent_Tier": sk.get("Talent_Tier", "NONE"),
        "Is_Elite": sk.get("Is_Elite", False),
        "Is_Star": sk.get("Is_Star", False),

        "team_5v5_SF60": sk.get("team_5v5_SF60"),
        "team_5v5_xGF60": sk.get("team_5v5_xGF60"),
        "team_5v5_SF60_pct": sk.get("team_5v5_SF60_pct"),
        "team_5v5_xGF60_pct": sk.get("team_5v5_xGF60_pct"),
        # --- Team Recent Scoring (HARD GF GATE) ---
        "Team_GF_L5": sk.get("Team_GF_L5"),
        "Team_GF_Avg_L5": sk.get("Team_GF_Avg_L5"),
        "Team_GF_Gate": sk.get("Team_GF_Gate"),


        "Med10_SOG": sk.get("Median10_SOG"),
        "Trim10_SOG": sk.get("TrimMean10_SOG"),
        "Avg5_SOG": sk.get("Avg5_SOG"),
        "ShotIntent": sk.get("ShotIntent"),
        "ShotIntent_Pct": sk.get("ShotIntent_Pct"),
        "L10_P": sk.get("L10_P"),
        "L10_G": sk.get("L10_G"),
        "L10_S": sk.get("L10_S"),
        "L5_G": sk.get("G5_total"),
        "L5_A": sk.get("A5_total"),
        "L10_A": sk.get("L10_A"),

        "i5v5_points60": sk.get("i5v5_points60"),
        "i5v5_primaryAssists60": sk.get("i5v5_primaryAssists60"),
        "i5v5_shotAssists60": sk.get("i5v5_shotAssists60"),
        "i5v5_iCF60": sk.get("i5v5_iCF60"),
        "Assist_Volume": sk.get("Assist_Volume"),

        "opp_5v5_xGA60": sk.get("opp_5v5_xGA60"),
        "opp_5v5_HDCA60": sk.get("opp_5v5_HDCA60"),
        "opp_5v5_SlotSA60": sk.get("opp_5v5_SlotSA60"),
        "Opp_DefWeak": sk.get("Opp_DefWeak"),

        "v2_player_stability": sk.get("v2_player_stability"),
        "v2_defense_vulnerability": sk.get("v2_defense_vulnerability"),

        "Exp_P_10": sk.get("Exp_P_10"),
        "Exp_G_10": sk.get("Exp_G_10"),
        "Exp_S_10": sk.get("Exp_S_10"),
        "Reg_Gap_P10": sk.get("Reg_Gap_P10"),
        "Reg_Gap_G10": sk.get("Reg_Gap_G10"),
        "Reg_Gap_S10": sk.get("Reg_Gap_S10"),
        "Reg_Heat_P": sk.get("Reg_Heat_P"),
        "Reg_Heat_G": sk.get("Reg_Heat_G"),
        "Reg_Heat_S": sk.get("Reg_Heat_S"),
        "Exp_A_10": sk.get("Exp_A_10"),
        "Reg_Gap_A10": sk.get("Reg_Gap_A10"),
        "Reg_Heat_A": sk.get("Reg_Heat_A"),

        "Matrix_Points": sk.get("Matrix_Points"),
        "Matrix_SOG": sk.get("Matrix_SOG"),
        "Matrix_Goal": sk.get("Matrix_Goal"),
        "Matrix_Assists": sk.get("Matrix_Assists"),

        "Conf_Points": sk.get("Conf_Points"),
        "Conf_SOG": sk.get("Conf_SOG"),
        "Conf_Goal": sk.get("Conf_Goal"),
        "Conf_Assists": sk.get("Conf_Assists"),

        "Best_Market": sk.get("Best_Market"),
        "Best_Conf": sk.get("Best_Conf"),

        # raw droughts (optional but useful)
        "Drought_P": sk.get("Drought_P"),
        "Drought_A": sk.get("Drought_A"),
        "Drought_G": sk.get("Drought_G"),
        "Drought_SOG": sk.get("Drought_SOG"),

        "Line": "",
        "Odds": "",
        "Result": "",
    })

    # -------------------------
    # Best_Drought (visual tag for the best market)
    # -------------------------
    def _fmt_int(v: Any) -> str:
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return ""
            return str(int(float(v)))
        except Exception:
            return ""

    def best_drought(row: pd.Series) -> str:
        tier = str(row.get("Talent_Tier", "")).upper()
        bm = str(row.get("Best_Market", "")).upper()

        tags: list[str] = []

        # --- Always surface POINTS drought for STAR / ELITE ---
        if tier in {"ELITE", "STAR"}:
            dp = _fmt_int(row.get("Drought_P"))
            if dp and int(dp) >= 2:
                tags.append(f"🅿️ PTS D:{dp}")

        # --- Best market drought ---
        if bm == "ASSISTS":
            da = _fmt_int(row.get("Drought_A"))
            if da:
                tags.append(f"🅰️ D:{da}")

        elif bm == "POINTS":
            dp = _fmt_int(row.get("Drought_P"))
            if dp:
                tags.append(f"🅿️ D:{dp}")

        elif bm == "GOAL":
            dg = _fmt_int(row.get("Drought_G"))
            if dg:
                tags.append(f"🥅 D:{dg}")

        elif bm == "SOG":
            ds = _fmt_int(row.get("Drought_SOG"))
            if ds:
                tags.append(f"🎯 D:{ds}")

        # de-dupe, preserve order
        return " | ".join(dict.fromkeys(tags))

    tracker["Best_Drought"] = tracker.apply(best_drought, axis=1)

    


    # Play tags (Points)
    tracker["Conf_Points"] = pd.to_numeric(tracker.get("Conf_Points"), errors="coerce")
    tracker["Play_Tag"] = ""
    tracker["Plays_Points"] = False

    hot_points = (
        (tracker["Matrix_Points"] == "Green") &
        (tracker["Reg_Heat_P"] == "HOT") &
        (tracker["Conf_Points"] >= 70)
    )
    tracker.loc[hot_points, "Play_Tag"] = "🔥"
    tracker.loc[hot_points, "Plays_Points"] = True
    

    # Assists earned rule (single version)
    for c in [
        "iXA%", "Conf_Assists", "v2_player_stability",
        "team_5v5_xGF60_pct", "Assist_Volume",
        "i5v5_primaryAssists60"
    ]:
        if c in tracker.columns:
            tracker[c] = pd.to_numeric(tracker[c], errors="coerce")

    if "Plays_Assists" not in tracker.columns:
        tracker["Plays_Assists"] = False
    tracker["Assist_ProofCount"] = 0
    tracker["Assist_Why"] = ""

    proof_ixA  = (tracker["iXA%"] >= 92)
    proof_v2   = (tracker["v2_player_stability"] >= 65)
    proof_team = (tracker["team_5v5_xGF60_pct"] >= 60)
    proof_vol  = (
        (tracker["Assist_Volume"] >= 6) |
        (tracker["i5v5_primaryAssists60"] >= 0.45)
    )

    proofs = pd.concat([proof_ixA, proof_v2, proof_team, proof_vol], axis=1).fillna(False)
    tracker["Assist_ProofCount"] = proofs.sum(axis=1)

    tier = tracker.get("Talent_Tier", "").astype(str).str.upper()
    is_star = tier.isin(["ELITE", "STAR"])

    earned_gate = (
        (tracker["Assist_ProofCount"] >= 3) |
        (is_star & (tracker["Assist_ProofCount"] >= 2))
    )

    assists_green_earned = (
        (tracker["Matrix_Assists"] == "Green") &
        (tracker["Conf_Assists"] >= 72) &
        earned_gate
    ).fillna(False)

    tracker["Plays_Assists"] = assists_green_earned

    def _assist_why(r):
        reasons = []
        if float(r.get("iXA%", 0) or 0) >= 90: reasons.append("iXA")
        if float(r.get("v2_player_stability", 0) or 0) >= 60: reasons.append("v2")
        if float(r.get("team_5v5_xGF60_pct", 0) or 0) >= 60: reasons.append("xGF")
        if (float(r.get("Assist_Volume", 0) or 0) >= 6 or float(r.get("i5v5_primaryAssists60", 0) or 0) >= 0.45):
            reasons.append("VOL")
        return ",".join(reasons)

    tracker.loc[assists_green_earned, "Assist_Why"] = tracker.loc[assists_green_earned].apply(_assist_why, axis=1)

    mask = assists_green_earned & ~tracker["Play_Tag"].fillna("").str.contains("ASSISTS EARNED", regex=False)

    tracker.loc[mask, "Play_Tag"] = np.where(
    tracker.loc[mask, "Play_Tag"].fillna("").astype(str).str.len() > 0,
    tracker.loc[mask, "Play_Tag"].fillna("").astype(str) + " | 🅰️ ASSISTS EARNED",
    "🅰️ ASSISTS EARNED"
)
    
    # -------------------------
    # SOG earned rule (shot profile playable)
    # Purpose:
    # - Keep CONF tight (>=70)
    # - Promote elite shot profiles even if Matrix_SOG is Yellow
    # - Require alignment of the TOP 4 SOG categories
    # -------------------------

    # Ensure numeric safety
    for c in [
        "Conf_SOG",
        "ShotIntent_Pct",
        "Goalie_Weak",
        "Med10_SOG",
        "Avg5_SOG",
        "Reg_Gap_S10",
        "Drought_SOG"
    ]:
        if c in tracker.columns:
            tracker[c] = pd.to_numeric(tracker[c], errors="coerce")

    if "Plays_SOG" not in tracker.columns:
        tracker["Plays_SOG"] = False

    tracker["SOG_ProofCount"] = 0
    tracker["SOG_Why"] = ""

    # ---- TOP 4 SOG PROOFS ----

    # 1) ShotIntent (elite intent, not noisy volume)
    proof_si = (tracker["ShotIntent_Pct"] >= 95)

    # 2) Regression / timing
    # HOT/WARM OR meaningful positive gap OR drought
    proof_reg = (
        tracker["Reg_Heat_S"].astype(str).str.upper().isin(["HOT", "WARM"]) |
        (tracker["Reg_Gap_S10"] >= 1.5) |
        (tracker["Drought_SOG"] >= 1.5)
    )

    # 3) Volume floor (true shooter)
    proof_vol = (
        (tracker["Med10_SOG"] >= 3.5) |
        (tracker["Avg5_SOG"] >= 3.5)
    )

    # 4) Goalie environment (not a brick wall)
    proof_goalie = (tracker["Goalie_Weak"] >= 70)

    proofs = pd.concat(
        [proof_si, proof_reg, proof_vol, proof_goalie],
        axis=1
    ).fillna(False)

    tracker["SOG_ProofCount"] = proofs.sum(axis=1)

    # ---- HARD CONFIDENCE GATE (DO NOT LOOSEN) ----
    conf_gate = (tracker["Conf_SOG"] >= 77)

    # ---- EARNED SOG PLAY ----
    # Require:
    # - Confidence gate
    # - At least 3 of the 4 SOG proofs
    sog_earned = (
        conf_gate &
        (tracker["SOG_ProofCount"] >= 3)
    )

    tracker.loc[sog_earned, "Plays_SOG"] = True

    # Optional explanation tag
    def _sog_why(r):
        reasons = []
        if r.get("ShotIntent_Pct", 0) >= 95: reasons.append("SI")
        if (
            str(r.get("Reg_Heat_S", "")).upper() in {"HOT", "WARM"} or
            (r.get("Reg_Gap_S10", 0) >= 1.5) or
            (r.get("Drought_SOG", 0) >= 1)
        ): reasons.append("REG")
        if (r.get("Med10_SOG", 0) >= 3.5 or r.get("Avg5_SOG", 0) >= 3.5): reasons.append("VOL")
        if r.get("Goalie_Weak", 0) >= 70: reasons.append("G")
        return ",".join(reasons)

    tracker.loc[sog_earned, "SOG_Why"] = tracker.loc[sog_earned].apply(_sog_why, axis=1)

    # Optional UI tag (does NOT override other tags)
    mask = sog_earned & ~tracker["Play_Tag"].fillna("").str.contains("SOG EARNED", regex=False)

    tracker.loc[mask, "Play_Tag"] = np.where(
        tracker.loc[mask, "Play_Tag"].fillna("").astype(str).str.len() > 0,
        tracker.loc[mask, "Play_Tag"].fillna("").astype(str) + " | 🎯 SOG EARNED",
        "🎯 SOG EARNED"
    )

    

    # Sort
    tracker["_bc"] = pd.to_numeric(tracker["Best_Conf"], errors="coerce").fillna(0)
    tracker["_gw"] = pd.to_numeric(tracker["Goalie_Weak"], errors="coerce").fillna(50)
    tracker["_dw"] = pd.to_numeric(tracker["Opp_DefWeak"], errors="coerce").fillna(50)
    tracker = tracker.sort_values(["_bc", "_gw", "_dw"], ascending=[False, False, False]).drop(columns=["_bc", "_gw", "_dw"])

    # -------------------------
    # FORCE rounding right before write (for Streamlit display consistency)
    # -------------------------
    ROUND_1 = [
        "iXA%", "iXG%", "v2_player_stability",
        "team_5v5_SF60_pct", "team_5v5_xGF60_pct",
        "Opp_DefWeak", "Goalie_Weak",
        "TOI_Pct", "StarScore"
    ]
    ROUND_2 = [
        "Exp_A_10", "Reg_Gap_A10",
        "Exp_P_10", "Reg_Gap_P10",
        "Exp_G_10", "Reg_Gap_G10",
        "Exp_S_10", "Reg_Gap_S10"
    ]

    for c in ROUND_1:
        if c in tracker.columns:
            tracker[c] = pd.to_numeric(tracker[c], errors="coerce").round(1)

    for c in ROUND_2:
        if c in tracker.columns:
            tracker[c] = pd.to_numeric(tracker[c], errors="coerce").round(2)

    
 
    stamp = datetime.now().strftime("%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"tracker_{today_local.isoformat()}_{stamp}.csv")
    tracker.to_csv(out_path, index=False)

    print(f"CSV saved to: {out_path}")
    print(f"Cache saved to: {cache_path_today(today_local)}\n")
    return out_path



# ============================
# Entrypoint
# ============================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="YYYY-MM-DD (default today)", default=None)
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    args = parser.parse_args()

    today_local = date.fromisoformat(args.date) if args.date else date.today()
    build_tracker(today_local, debug=bool(args.debug))

if __name__ == "__main__":
    main()
