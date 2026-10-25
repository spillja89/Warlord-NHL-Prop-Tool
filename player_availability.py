"""Shared pregame availability guard for tracker rows and board cards."""

import pandas as pd
import unicodedata


UNAVAILABLE_STATUSES = {"out", "ir", "ltir", "injured reserve", "scratch", "scratched", "inactive"}


def is_unavailable(row: dict) -> bool:
    status = str(row.get("Injury_Status", "")).strip().casefold()
    if status in UNAVAILABLE_STATUSES:
        return True
    available = str(row.get("Available", "")).strip().casefold()
    return available in {"false", "0", "no", "n"}


def unavailable_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(False, index=frame.index)
    status = frame.get("Injury_Status", pd.Series("", index=frame.index))
    mask = status.fillna("").astype(str).str.strip().str.casefold().isin(UNAVAILABLE_STATUSES)
    if "Available" in frame.columns:
        available = frame["Available"].fillna("").astype(str).str.strip().str.casefold()
        mask |= available.isin({"false", "0", "no", "n"})
    return mask


def apply_live_injury_overlay(frame: pd.DataFrame, reports: pd.DataFrame, slate_date) -> tuple[pd.DataFrame, int]:
    """Exclude newly reported absences from a previously saved current-day tracker."""
    if frame.empty or reports is None or reports.empty or not {"Date", "Team", "Player"}.issubset(frame.columns):
        return frame, 0
    out = frame.copy()
    before = unavailable_mask(out)
    for column in ("Injury_Status", "Injury_Badge", "Injury_Text", "Injury_Return"):
        if column in out.columns:
            out[column] = out[column].astype(object)
    current_day = out["Date"].astype(str).str[:10].eq(str(slate_date))
    teams = out["Team"].fillna("").astype(str).str.upper()

    def name_key(value):
        normalized = unicodedata.normalize("NFKD", str(value).casefold())
        return " ".join("".join(c for c in normalized if not unicodedata.combining(c)).split())

    names = out["Player"].map(name_key)
    for report in reports.itertuples(index=False):
        status = str(report.Status)
        if status.casefold() not in UNAVAILABLE_STATUSES:
            continue
        matched = current_day & teams.eq(str(report.Team).upper()) & names.eq(name_key(report.Player))
        if "Lineup_Status" in out.columns:
            matched &= ~out["Lineup_Status"].fillna("").astype(str).eq("Dressed")
        if not matched.any():
            continue
        out.loc[matched, "Injury_Status"] = status
        out.loc[matched, "Available"] = False
        out.loc[matched, "Injury_Badge"] = "🔴 OUT"
        out.loc[matched, "Injury_Text"] = str(report.Injury)
        out.loc[matched, "Injury_Return"] = str(report.Expected_Return)
    newly_out = int((unavailable_mask(out) & ~before).sum())
    return out, newly_out
