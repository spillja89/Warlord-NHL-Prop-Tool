"""Read-only browsing helpers. Sorting never changes a move or saved pick."""
from collections import OrderedDict
import math
import pandas as pd
from chosen_priority import priority_key

BOARD_ORDERS = ("Top ranked", "Earliest start", "Latest start", "Player A–Z",
                "Highest odds", "Lowest odds")
CHOSEN_ORDERS = ("Top ranked", "Tier · S first", "Earliest start", "Latest start", "Player A–Z",
                 "Highest odds", "Lowest odds")
TIER_ORDER = {"S": 0, "A": 1, "B": 2, "Review": 3, "Unrated": 4}


def start_stamp(utc=None, local=None):
    stamp = pd.to_datetime(utc, utc=True, errors="coerce")
    if pd.notna(stamp):
        return stamp
    stamp = pd.to_datetime(local, errors="coerce")
    if pd.isna(stamp):
        return pd.NaT
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("America/Chicago", ambiguous="NaT", nonexistent="NaT")
    return stamp.tz_convert("UTC") if pd.notna(stamp) else pd.NaT


def slot_key(utc=None, local=None):
    stamp = start_stamp(utc, local)
    return stamp.floor("min").isoformat() if pd.notna(stamp) else "Start TBD"


def slot_options(keys):
    keys = set(keys)
    known = sorted(key for key in keys if key != "Start TBD")
    dates = {pd.Timestamp(key).tz_convert("America/Chicago").date() for key in known}
    labels = {}
    for key in known:
        stamp = pd.Timestamp(key).tz_convert("America/Chicago")
        time = stamp.strftime("%I:%M %p").lstrip("0") + " CT"
        labels[key] = stamp.strftime("%b %d · ") + time if len(dates) > 1 else time
    if "Start TBD" in keys:
        labels["Start TBD"] = "Start TBD"
    return labels


def frame_slots(frame):
    return pd.Series([slot_key(row.get("StartTimeUTC"), row.get("StartTimeLocal"))
                      for row in frame.to_dict("records")], index=frame.index, dtype=str)


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def sort_cards(cards, order, starts_by_game):
    if order == "Top ranked":
        return cards
    def key(card):
        if order == "Player A–Z":
            name = str(card.get("player") or "").strip().casefold()
            return (not bool(name), name)
        if order in ("Highest odds", "Lowest odds"):
            odds = number(card.get("odds"))
            return (odds is None, (-odds if order == "Highest odds" else odds) if odds is not None else 0)
        start = starts_by_game.get(str(card.get("game") or ""))
        return (pd.isna(start), 0 if pd.isna(start) else
                (start.value if order == "Earliest start" else -start.value))
    return sorted(cards, key=key)


def sort_rows(frame, order, starts_by_game=None, odds_field=None):
    if order == "Top ranked" or frame.empty:
        return frame
    if order == "Player A–Z":
        values = frame.get("Player", pd.Series(index=frame.index, dtype=str)).fillna("").astype(str).str.casefold().replace("", pd.NA)
        ascending = True
    elif order in ("Highest odds", "Lowest odds"):
        fields = (odds_field, "Over odds", "Best_Odds", "Odds", "Price", "odds")
        field = next((field for field in fields if field and field in frame), None)
        values = pd.to_numeric(frame[field], errors="coerce") if field else pd.Series(float("nan"), index=frame.index)
        ascending = order == "Lowest odds"
    else:
        values = pd.Series([start_stamp(row.get("StartTimeUTC"), row.get("StartTimeLocal"))
                            for row in frame.to_dict("records")], index=frame.index)
        if "Game" in frame:
            fallback = pd.to_datetime(frame["Game"].astype(str).map(starts_by_game or {}), utc=True, errors="coerce")
            values = pd.to_datetime(values, utc=True, errors="coerce").fillna(fallback)
        ascending = order == "Earliest start"
    # A fresh temporary column cannot overwrite a user's uploaded tracker field.
    temp = "_browse_sort"
    while temp in frame:
        temp += "_"
    return frame.assign(**{temp: values}).sort_values(temp, ascending=ascending, kind="stable", na_position="last").drop(columns=temp)


def prop_label(pick):
    market = {"PPP": "Power Play", "Goal": "Goals"}.get(pick.get("market"), pick.get("market") or "Other")
    return f'{market} · {float(pick["line"]):g}'


def browse_picks(picks, metadata, *, props=(), tiers=(), slots=(), order="Top ranked", group="Prop"):
    """Every selected prop appears once. Tier belongs to that exact prop, not its teammate/other market."""
    selected = [pick for pick in picks
                if (not props or prop_label(pick) in props)
                and (not tiers or metadata[pick["pick_id"]]["tier"] in tiers)
                and (not slots or slot_key(pick.get("start_utc")) in slots)]
    def key(pick):
        meta = metadata[pick["pick_id"]]
        stamp = start_stamp(pick.get("start_utc"))
        start = (pd.isna(stamp), 0 if pd.isna(stamp) else stamp.value)
        if order in ("Top ranked", "Tier · S first"):
            return priority_key(pick,metadata)
        if order == "Player A–Z":
            return (str(pick.get("player") or "").casefold(),)
        if order == "Latest start":
            return (start[0], -start[1])
        if order in ("Highest odds", "Lowest odds"):
            odds = number(pick.get("odds"))
            return (odds is None, 0 if odds is None else -odds if order == "Highest odds" else odds)
        return start
    selected = sorted(selected, key=key)
    grouped = OrderedDict()
    labels = slot_options(slot_key(p.get("start_utc")) for p in selected)
    def group_key(pick):
        if group == "Prop":
            return prop_label(pick)
        if group == "Tier":
            tier = metadata[pick["pick_id"]]["tier"]
            return tier + " tier" if tier in ("S", "A", "B") else tier
        if group == "Start time":
            return labels[slot_key(pick.get("start_utc"))]
        return "All chosen champions"
    if group == "Prop":
        selected.sort(key=prop_label)
    elif group == "Tier":
        selected.sort(key=lambda pick: TIER_ORDER[metadata[pick["pick_id"]]["tier"]])
    elif group == "Start time":
        selected.sort(key=lambda pick: (slot_key(pick.get("start_utc")) == "Start TBD", slot_key(pick.get("start_utc"))))
    for pick in selected:
        grouped.setdefault(group_key(pick), []).append(pick)
    return grouped

