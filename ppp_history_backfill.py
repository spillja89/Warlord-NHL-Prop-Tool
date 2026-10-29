"""Fill pregame PPP history for priced players outside the model log pool."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import pandas as pd

from player_form import compact_regular_log


def fill_missing_priced_ppp_history(tracker, fetch_log, before_date, max_workers=8):
    """Fetch only priced PPP players whose regular-season log is missing."""
    odds_columns = [
        column for suffix in ("", "_1", "_2", "_3", "_4")
        if (column := f"BDL_PPP_Odds{suffix}") in tracker.columns
    ]
    if not odds_columns or "Player_ID" not in tracker.columns:
        return tracker, 0

    priced = pd.concat(
        [pd.to_numeric(tracker[column], errors="coerce").notna()
         for column in odds_columns], axis=1
    ).any(axis=1)
    player_ids = pd.to_numeric(tracker["Player_ID"], errors="coerce")
    drought = pd.to_numeric(
        tracker["Drought_PPP"] if "Drought_PPP" in tracker.columns
        else pd.Series(index=tracker.index, dtype=float), errors="coerce"
    )
    form = tracker["Form_Log"] if "Form_Log" in tracker.columns else pd.Series("", index=tracker.index)
    missing = priced & player_ids.notna() & (drought.isna() | form.fillna("").astype(str).eq(""))
    target_ids = sorted(set(player_ids[missing].astype(int)))
    if not target_ids:
        return tracker, 0

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_log, player_id): player_id for player_id in target_ids}
        for future in as_completed(futures):
            player_id = futures[future]
            try:
                log = compact_regular_log(future.result(), before_date)
            except Exception:
                continue
            if not log or not log["games"]:
                continue
            recent = log["games"][:10]
            pp_points = [game.get("pp") for game in recent]
            if any(points is None for points in pp_points):
                drought_games = None
                recent_total = None
            else:
                drought_games = next(
                    (i for i, points in enumerate(pp_points) if points > 0), len(pp_points)
                )
                recent_total = sum(pp_points)
            results[player_id] = (json.dumps(log, separators=(",", ":")), drought_games, recent_total)

    if not results:
        return tracker, 0
    tracker = tracker.copy()
    for column in ("Form_Log", "Drought_PPP", "PPP10_total"):
        if column not in tracker.columns:
            tracker[column] = pd.NA
    for player_id, (form_json, drought_games, recent_total) in results.items():
        rows = missing & player_ids.eq(player_id)
        tracker.loc[rows, "Form_Log"] = form_json
        tracker.loc[rows, "Drought_PPP"] = drought_games
        tracker.loc[rows, "PPP10_total"] = recent_total
    return tracker, len(results)

