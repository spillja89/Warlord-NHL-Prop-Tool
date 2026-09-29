"""Shared pregame availability guard for tracker rows and board cards."""

import pandas as pd


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
