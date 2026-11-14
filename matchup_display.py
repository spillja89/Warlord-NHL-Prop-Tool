"""Presentation-only Central start times; never modify saved matchup identities."""
import pandas as pd


def start_time_text(start_utc=None, start_local=None):
    stamp = pd.to_datetime(start_utc, utc=True, errors="coerce")
    if pd.notna(stamp):
        return stamp.tz_convert("America/Chicago").strftime("%I:%M %p").lstrip("0") + " CT"
    stamp = pd.to_datetime(start_local, errors="coerce")
    if pd.notna(stamp):
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("America/Chicago")
        return stamp.strftime("%I:%M %p").lstrip("0") + " CT"
    return "Start TBD"


def matchup_label(game, start_utc=None, start_local=None):
    game = str(game or "").strip()
    return f"{game} · {start_time_text(start_utc, start_local)}" if game else "—"


def with_matchup_times(frame, reference=None):
    """Copy a display table, resolving dates before falling back to slate times."""
    if "Game" not in frame.columns:
        return frame
    out = frame.copy()
    dated, undated = {}, {}
    if reference is not None and "Game" in reference.columns:
        for row in reference.to_dict("records"):
            game = str(row.get("Game") or "")
            date = str(row.get("Date") or "")[:10]
            time = start_time_text(row.get("StartTimeUTC"), row.get("StartTimeLocal"))
            if time != "Start TBD":
                dated[(date, game)] = time
                undated.setdefault(game, set()).add(time)
    def label(row):
        game = str(row.get("Game") or "")
        time = start_time_text(row.get("StartTimeUTC"), row.get("StartTimeLocal"))
        if time == "Start TBD":
            date = str(row.get("Date") or "")[:10]
            time = dated.get((date, game), "Start TBD")
            if time == "Start TBD" and not date and len(undated.get(game, set())) == 1:
                time = next(iter(undated[game]))
        if time == "Start TBD":
            for field in ("Start (CT)", "Time"):
                existing = row.get(field)
                if existing is not None and pd.notna(existing) and str(existing).strip():
                    time = str(existing).strip()
                    if "CT" not in time:
                        time += " CT"
                    break
        return f"{game} · {time}" if game else "—"
    out["Game"] = [label(row) for row in out.to_dict("records")]
    return out
