"""Private durable bet storage for Streamlit deployments.

The database URL stays in server-side secrets. The app calls this module only
after owner authentication. No credentials or bet payloads are logged.
"""

from __future__ import annotations

from typing import Any


def _connect(database_url: str):
    import psycopg

    if not database_url or not database_url.startswith(("postgres://", "postgresql://")):
        raise ValueError("WARLORD_LEDGER_DATABASE_URL must be a PostgreSQL URL")
    return psycopg.connect(database_url, sslmode="require", connect_timeout=8)


def append_bet(database_url: str, row: dict[str, Any]) -> None:
    from psycopg.types.json import Jsonb

    bet_id = str(row.get("bet_id") or "").strip()
    if not bet_id:
        raise ValueError("A bet ID is required")
    with _connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS warlord_bets (
                    bet_id text PRIMARY KEY,
                    placed_at timestamptz NOT NULL DEFAULT now(),
                    payload jsonb NOT NULL
                )
            """)
            cursor.execute("INSERT INTO warlord_bets (bet_id, payload) VALUES (%s, %s)",
                           (bet_id, Jsonb(row)))


def recent_bets(database_url: str, limit: int = 10) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 100))
    with _connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS warlord_bets (
                    bet_id text PRIMARY KEY,
                    placed_at timestamptz NOT NULL DEFAULT now(),
                    payload jsonb NOT NULL
                )
            """)
            cursor.execute("SELECT payload FROM warlord_bets ORDER BY placed_at DESC, bet_id DESC LIMIT %s",
                           (safe_limit,))
            return [payload for (payload,) in cursor.fetchall()]
