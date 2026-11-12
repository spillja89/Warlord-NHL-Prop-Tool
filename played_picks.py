"""Owner pick marks. Separate from model moves and the manual betting ledger.

PostgreSQL is required for durable hosted storage. Public reads select only a
whitelisted, explicitly shared projection, never the private payload.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone

MARKETS = {"Points", "Assists", "SOG", "Goal"}
PUBLIC_FIELDS = ("pick_id", "date", "game_id", "player_id", "player", "team", "game",
                 "market", "line", "side", "odds", "book", "start_utc", "shared_at")


def utc_now():
    return datetime.now(timezone.utc)


def clean(value):
    text = str(value).strip()
    return "" if text.lower() in {"", "none", "nan", "nat", "<na>"} else text


def numeric_id(value):
    text = clean(value)
    try:
        return str(int(float(text))) if text else ""
    except (ValueError, OverflowError):
        return ""


def make_pick(card, row):
    """Use authoritative tracker IDs; never moves, rank, price or card position."""
    market = card.get("market")
    if market not in MARKETS:
        raise ValueError("Unsupported market")
    game_id, player_id = numeric_id(row.get("Game_ID")), numeric_id(row.get("Player_ID"))
    day = clean(row.get("Date"))[:10]
    datetime.strptime(day, "%Y-%m-%d")
    line, odds = float(card["line"]), float(card["odds"])
    if not game_id or not player_id or not all(map(math.isfinite, (line, odds))) or line < 0 or abs(odds) < 100:
        raise ValueError("A valid player, game, line and price are required")
    identity = [day, game_id, player_id, market, format(line, ".8g"), "Over"]
    pick_id = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()[:32]
    return dict(date=day, game_id=game_id, player_id=player_id, market=market, side="Over",
                pick_id=pick_id, line=line, player=clean(card.get("player")), team=clean(card.get("team")),
                game=clean(card.get("game")), odds=odds, book=clean(card.get("book")),
                start_utc=clean(row.get("StartTimeUTC")),
                move_name=clean((card.get('move') or {}).get('name')),
                move_rule=clean((card.get('move') or {}).get('rule')))


def pregame(pick, now=None):
    try:
        start = datetime.fromisoformat(pick["start_utc"].replace("Z", "+00:00"))
        return start.tzinfo is not None and start > (now or utc_now())
    except (KeyError, ValueError):
        return False


def public_projection(pick, shared_at):
    return {k: (shared_at if k == "shared_at" else pick.get(k)) for k in PUBLIC_FIELDS}


class PickStore:
    def __init__(self, database_url):
        from ledger_store import _connect
        self.database_url = database_url
        self.connect = lambda: _connect(self.database_url)

    @staticmethod
    def _ensure(cursor):
        cursor.execute("""CREATE TABLE IF NOT EXISTS warlord_pick_marks (
            pick_id text PRIMARY KEY, played boolean NOT NULL DEFAULT false,
            shared boolean NOT NULL DEFAULT false, private_pick jsonb NOT NULL,
            public_pick jsonb, revision bigint NOT NULL DEFAULT 1,
            updated_at timestamptz NOT NULL DEFAULT now())""")

    def private_rows(self):
        with self.connect() as conn, conn.cursor() as cur:
            self._ensure(cur)
            cur.execute("SELECT pick_id, played, shared, private_pick, revision FROM warlord_pick_marks ORDER BY updated_at DESC")
            return {key: dict(payload, played=played, shared=shared, revision=revision)
                    for key, played, shared, payload, revision in cur.fetchall()}

    def public_rows(self):
        with self.connect() as conn, conn.cursor() as cur:
            self._ensure(cur)
            cur.execute("SELECT public_pick FROM warlord_pick_marks WHERE shared=true AND public_pick IS NOT NULL ORDER BY updated_at DESC")
            # Also whitelist on read in case a future writer changes its payload.
            return [{k: payload.get(k) for k in PUBLIC_FIELDS} for (payload,) in cur.fetchall()]

    def change(self, pick, field, value, expected_revision, now=None):
        """Transactional compare-and-set: stale tabs cannot overwrite new marks."""
        from psycopg.types.json import Jsonb
        if field not in {"played", "shared"} or type(value) is not bool:
            raise ValueError("Invalid pick action")
        now = now or utc_now()
        key = pick["pick_id"]
        with self.connect() as conn, conn.cursor() as cur:
            self._ensure(cur)
            cur.execute("INSERT INTO warlord_pick_marks (pick_id, private_pick) VALUES (%s,%s) ON CONFLICT DO NOTHING", (key, Jsonb(pick)))
            cur.execute("SELECT played,shared,private_pick,public_pick,revision FROM warlord_pick_marks WHERE pick_id=%s FOR UPDATE", (key,))
            played, shared, private, public, revision = cur.fetchone()
            # A new row's initial revision is 1; the UI uses 0 for absent rows.
            if revision != (expected_revision or 1):
                raise ValueError("This pick changed in another tab. Reload and try again.")
            if field == "played":
                if value and not played:
                    private = dict(pick, marked_at=now.isoformat(), timing="pregame" if pregame(pick, now) else "after start / start unknown")
                played = value
                if not played:
                    shared, public = False, None
            else:
                if value and (not played or not pregame(pick, now)):
                    raise ValueError("Only a marked pregame pick can be shared.")
                if value and not shared:
                    public = public_projection(pick, now.isoformat())
                shared = value
                if not shared:
                    public = None
            cur.execute("UPDATE warlord_pick_marks SET played=%s,shared=%s,private_pick=%s,public_pick=%s,revision=revision+1,updated_at=now() WHERE pick_id=%s",
                        (played, shared, Jsonb(private), Jsonb(public) if public else None, key))
        # Return only after the transaction context has committed successfully.
        return dict(private, played=played, shared=shared, revision=revision + 1)


def results_for_picks(picks, graded_dir):
    """Official saved grades only. No match, missing or conflicting credit stays pending."""
    import pandas as pd
    from pathlib import Path
    outcome = {}
    for day in {p["date"] for p in picks}:
        candidates = list(Path(graded_dir).glob(f"tracker_{day}*_GRADED.csv"))
        values = {}
        for path in candidates:
            try:
                frame = pd.read_csv(path, low_memory=False)
            except (OSError, ValueError):
                continue
            for row in frame.to_dict("records"):
                if clean(row.get('Grade_Status')).upper() != 'FINAL':
                    continue
                key = (numeric_id(row.get("Game_ID")), numeric_id(row.get("Player_ID")))
                for market in MARKETS:
                    status = clean(row.get(f"Match_Status_{market}" )).upper()
                    actual = row.get(f"Actual_{market}")
                    if status not in {"MATCHED", "OK", "MATCH"} or pd.isna(actual):
                        continue
                    values.setdefault((*key, market), set()).add(float(actual))
        for pick in picks:
            if pick["date"] != day:
                continue
            actuals = values.get((pick["game_id"], pick["player_id"], pick["market"]), set())
            actual = next(iter(actuals)) if len(actuals) == 1 else None
            outcome[pick["pick_id"]] = ("Pending" if actual is None else "Win" if actual > pick["line"] else "Push" if actual == pick["line"] else "Loss")
    return outcome
