"""Link goalie names to NHL IDs without fuzzy matching or importing NHL rates."""
from __future__ import annotations

import unicodedata

VERSION = "nhl-goalie-id-2026-11-13"


def goalie_name_key(name):
    plain = unicodedata.normalize("NFKD", str(name or ""))
    return "".join(ch for ch in plain.casefold()
                   if ch.isalnum() and not unicodedata.combining(ch))


# NHL ID checked against both sources. MoneyPuck currently spells his name
# "Leevi Merilinen". Retain this identity if the official name feed is down.
VERIFIED_NAME_IDS = {goalie_name_key("Leevi Merilainen"): 8482447}


def goalie_ids_from_rows(rows):
    names = {}
    for row in rows:
        try:
            player_id = int(row["playerId"])
            name = goalie_name_key(row["goalieFullName"])
        except (KeyError, TypeError, ValueError):
            continue
        if name and player_id > 0:
            names.setdefault(name, set()).add(player_id)
    # A shared name is not evidence that two IDs identify the same person.
    return {name: next(iter(ids)) for name, ids in names.items() if len(ids) == 1}


def fetch_goalie_name_ids(session, season_start, *, timeout=8):
    """Read official identities only; every modeled rate stays from MoneyPuck."""
    rows = []
    try:
        for start in range(0, 1000, 100):
            response = session.get(
                "https://api.nhle.com/stats/rest/en/goalie/summary",
                params={"isAggregate": "false", "isGame": "false",
                        "start": start, "limit": 100,
                        "cayenneExp": f"seasonId={int(season_start)}{int(season_start)+1} and gameTypeId=2"},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            page = payload["data"]
            if not isinstance(page, list):
                raise ValueError("Invalid official goalie identity response")
            rows.extend(page)
            if not page or start + len(page) >= int(payload.get("total", len(rows))):
                break
    except Exception:
        # Exact team/name matching remains available; no other goalie is substituted.
        pass
    observed_names = {goalie_name_key(row.get("goalieFullName"))
                      for row in rows if isinstance(row, dict)}
    fallback = {name: player_id for name, player_id in VERIFIED_NAME_IDS.items()
                if name not in observed_names}
    return {**fallback, **goalie_ids_from_rows(rows)}
