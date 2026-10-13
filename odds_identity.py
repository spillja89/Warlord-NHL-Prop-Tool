"""Keep name-only sportsbook quotes off same-name teammates."""

from __future__ import annotations

import pandas as pd

from odds_ev_bdl import _norm_name


MARKETS = ("Goal", "ATG", "Assists", "Points", "SOG", "PPP")


def withhold_ambiguous_player_odds(frame: pd.DataFrame) -> pd.DataFrame:
    """Withhold quotes when team/name identifies multiple NHL player IDs.

    Neither odds feed supplies an NHL player ID for every quote. Assigning a
    quote to both players would make one or both player cards misleading.
    """
    if frame.empty or not {"Team", "Player", "Player_ID"}.issubset(frame.columns):
        return frame
    result = frame.copy()
    names = result["Player"].fillna("").map(_norm_name)
    teams = result["Team"].fillna("").astype(str).str.upper().str.strip()
    ids = result["Player_ID"].fillna("").astype(str).str.strip()
    identities = pd.DataFrame({"team": teams, "name": names, "id": ids}, index=result.index)
    ambiguous = (identities["id"].ne("") &
                 identities.groupby(["team", "name"])["id"].transform("nunique").gt(1))
    result["Odds_Name_Ambiguous"] = ambiguous
    if not ambiguous.any():
        return result
    price_columns = []
    for column in result.columns:
        if column in {"Line", "Odds", "Book"}:
            price_columns.append(column)
            continue
        for market in MARKETS:
            if (column.startswith(f"BDL_{market}_") or
                column in {f"{market}_Line", f"{market}_Odds_Over", f"{market}_Book",
                           f"{market}_Odds_Source", f"{market}_Available_Books",
                           f"Plays_EV_{market}"} or
                column.startswith(f"{market}_EV")):
                price_columns.append(column)
                break
    result.loc[ambiguous, price_columns] = pd.NA
    return result
