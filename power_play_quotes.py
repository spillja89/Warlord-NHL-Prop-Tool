"""Read real power play point prices from a generated tracker."""

from __future__ import annotations

import math

import pandas as pd


COLUMNS = ("Game", "Player", "Team", "Opponent", "PPP line", "Over odds",
           "Book", "PP unit", "PP TOI/game", "PPP last 10", "Opp PK xGA/60")


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def priced_ppp_quotes(frame: pd.DataFrame) -> pd.DataFrame:
    """One best observed quote per player and PPP line, including alternates."""
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)
    quotes = {}
    for row in frame.to_dict("records"):
        player = str(row.get("Player") or "").strip()
        team = str(row.get("Team") or "").strip()
        if not player:
            continue
        for suffix in ("", "_1", "_2", "_3", "_4"):
            line = _finite(row.get(f"BDL_PPP_Line{suffix}"))
            odds = _finite(row.get(f"BDL_PPP_Odds{suffix}"))
            if line is None or odds is None or odds == 0:
                continue
            book = str(row.get(f"BDL_PPP_Book{suffix}") or "").strip()
            key = (team.casefold(), player.casefold(), line)
            candidate = {
                "Game": str(row.get("Game") or ""), "Player": player,
                "Team": team, "Opponent": str(row.get("Opp") or ""),
                "PPP line": line, "Over odds": int(odds), "Book": book,
                "PP unit": str(row.get("PP_Unit") or row.get("PP_Role") or ""),
                "PP TOI/game": _finite(row.get("PP_TOI_per_game")),
                "PPP last 10": _finite(row.get("PPP10_total")),
                "Opp PK xGA/60": _finite(row.get("Opp_PK_xGA60")),
            }
            if key not in quotes or odds > quotes[key]["Over odds"]:
                quotes[key] = candidate
    result = pd.DataFrame(quotes.values(), columns=COLUMNS)
    if not result.empty:
        result = result.sort_values(["Game", "Player", "PPP line"],
                                    kind="stable").reset_index(drop=True)
    return result
