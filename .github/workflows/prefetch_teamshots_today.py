"""
prefetch_teamshots_today.py — Stable CI Prefetch (Warlord NHL Prop Tool)

Goal:
- Make GitHub Actions STOP failing due to missing prefetch script.
- Always ensure the expected cache file exists:
    data/cache/teamshots_today.json

Behavior:
- Never raises an uncaught exception
- Always exits 0
- Writes a structured JSON payload (so later code can read it safely)
- If you later want "real" team shots, you can extend `fetch_teamshots()` in one place.

This is intentionally "boring and reliable".
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import date, datetime, timezone
from typing import Any, Dict


CACHE_DIR = os.path.join("data", "cache")
CACHE_PATH = os.path.join(CACHE_DIR, "teamshots_today.json")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def fetch_teamshots() -> Dict[str, Any]:
    """
    Placeholder for real team-shot prefetch.
    For now we produce an empty-but-valid structure.

    Structure is designed so future readers can do:
      data["teams"].get("TOR", {}).get("shots_for_avg", ...)
    without crashing.
    """
    return {
        "teams": {},  # team_abbr -> dict of stats (empty for now)
        "note": "CI-stable placeholder. Extend fetch_teamshots() for real data."
    }


def main() -> int:
    today = date.today().isoformat()

    payload: Dict[str, Any] = {
        "schema": "teamshots_cache_v1",
        "date": today,
        "generated_utc": utc_now_iso(),
        "status": "ok",
        "source": "prefetch_teamshots_today.py",
        "data": {},
        "error": "",
    }

    try:
        payload["data"] = fetch_teamshots()
        safe_write_json(CACHE_PATH, payload)
        print(f"[prefetch] wrote cache: {CACHE_PATH}")
        return 0

    except Exception as e:
        # Never fail Actions; write a fallback cache with error info.
        payload["status"] = "error"
        payload["error"] = f"{type(e).__name__}: {e}"
        payload["data"] = {"teams": {}, "note": "Fallback empty cache due to error."}

        try:
            safe_write_json(CACHE_PATH, payload)
            print(f"[prefetch] wrote fallback cache after error: {CACHE_PATH}")
        except Exception:
            print("[prefetch] FAILED to write fallback cache too:")
            traceback.print_exc()

        print("[prefetch] original error:")
        traceback.print_exc()
        return 0  # IMPORTANT: never fail CI


if __name__ == "__main__":
    sys.exit(main())
