"""Join saved pregame Jev labels to a final graded NHL tracker.

Run after games: python grade_jev_shadow.py jev_shadow_DATE.csv tracker_DATE_GRADED.csv
This is descriptive research; it does not alter the board or its moves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ACTUAL_COLUMNS = {
    "Goals": "Actual_Goal",
    "Assists": "Actual_Assists",
    "Points": "Actual_Points",
    "Shots": "Actual_SOG",
    "Power play": "Actual_PPP",
}
LABEL_COLUMNS = ("Opportunity", "Matchup", "Evidence")


def grade_shadow(shadow: pd.DataFrame, graded: pd.DataFrame) -> pd.DataFrame:
    """Use frozen shadow lines and final NHL actuals; leave unresolved cards blank."""
    required = {"Date", "Player_ID", "Market", "Prop line", "Saved odds"}
    missing = required - set(shadow.columns)
    if missing:
        raise ValueError(f"Shadow file is missing: {', '.join(sorted(missing))}")
    required = {"Date", "Player_ID", "Grade_Status", *ACTUAL_COLUMNS.values()}
    missing = required - set(graded.columns)
    if missing:
        raise ValueError(f"Graded tracker is missing: {', '.join(sorted(missing))}")
    left = shadow.copy()
    right = graded[list(required)].copy()
    for frame in (left, right):
        frame["Date"] = frame["Date"].astype(str).str.slice(0, 10)
        frame["Player_ID"] = pd.to_numeric(frame["Player_ID"], errors="coerce").astype("Int64")
    right = right.drop_duplicates(["Date", "Player_ID"])
    output = left.merge(right, on=["Date", "Player_ID"], how="left", validate="many_to_one")
    output["Actual"] = pd.NA
    output["Outcome"] = ""
    for market, column in ACTUAL_COLUMNS.items():
        mask = output["Market"].eq(market) & output["Grade_Status"].eq("FINAL")
        actual = pd.to_numeric(output.loc[mask, column], errors="coerce")
        line = pd.to_numeric(output.loc[mask, "Prop line"], errors="coerce")
        valid = actual.notna() & line.notna()
        indices = actual.index[valid]
        output.loc[indices, "Actual"] = actual.loc[indices].values
        output.loc[indices, "Outcome"] = ["W" if a > l else "L" if a < l else "P"
                                           for a, l in zip(actual.loc[indices], line.loc[indices])]
    return output.drop(columns=list(ACTUAL_COLUMNS.values()))


def summarize_shadow(cards: pd.DataFrame) -> pd.DataFrame:
    """Count settled results by market and Jev label; no probability inference."""
    records = []
    for market, group in cards.groupby("Market", dropna=False):
        for label_column in LABEL_COLUMNS:
            if label_column not in group:
                continue
            for label, rows in group.groupby(label_column, dropna=False):
                wins = int(rows["Outcome"].eq("W").sum())
                losses = int(rows["Outcome"].eq("L").sum())
                pushes = int(rows["Outcome"].eq("P").sum())
                records.append({"Market": market, "Dimension": label_column,
                                "Label": label, "Wins": wins, "Losses": losses,
                                "Pushes": pushes, "Unresolved": len(rows)-wins-losses-pushes,
                                "Settled hit %": round(100*wins/(wins+losses), 1)
                                if wins+losses else None})
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shadow_csv", type=Path)
    parser.add_argument("graded_tracker_csv", type=Path)
    args = parser.parse_args()
    cards = grade_shadow(pd.read_csv(args.shadow_csv), pd.read_csv(args.graded_tracker_csv))
    detail_path = args.shadow_csv.with_name(args.shadow_csv.stem + "_graded.csv")
    summary_path = args.shadow_csv.with_name(args.shadow_csv.stem + "_summary.csv")
    cards.to_csv(detail_path, index=False)
    summarize_shadow(cards).to_csv(summary_path, index=False)
    print(f"Saved {detail_path} and {summary_path}")


if __name__ == "__main__":
    main()
