"""Read-only comparison of a saved pregame tracker with fresh season sources."""

from __future__ import annotations

import math

import pandas as pd


def _num(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def compare_slate_sources(frame: pd.DataFrame, league_rows: list[dict]) -> list[dict]:
    """Report source timing/coverage differences; never infer a player is out."""
    by_id = {int(row["_id"]): row for row in league_rows if _num(row.get("_id")) is not None}
    findings = []
    for row in frame.to_dict("records"):
        player_id = _num(row.get("Player_ID"))
        if player_id is None:
            continue
        live = by_id.get(int(player_id))
        if live is None:
            continue
        name, team = str(row.get("Player") or ""), str(row.get("Team") or "")
        saved_gp, nhl_gp, mp_gp = (_num(row.get("Model_Player_GP")),
                                   _num(live.get("GP")), _num(live.get("MP GP")))
        if nhl_gp is not None and mp_gp is not None and nhl_gp - mp_gp >= 2:
            findings.append({"Player": name, "Team": team, "Source check": "MoneyPuck games trail NHL",
                             "Saved tracker": f"{saved_gp:g} GP" if saved_gp is not None else "GP unavailable",
                             "Fresh sources": f"NHL {nhl_gp:g} GP · MoneyPuck {mp_gp:g} GP",
                             "Meaning": "Current-season MoneyPuck sample is behind; rates may lag."})
        elif saved_gp is not None and mp_gp is not None and abs(saved_gp - mp_gp) >= 2:
            findings.append({"Player": name, "Team": team, "Source check": "Saved tracker sample differs",
                             "Saved tracker": f"MoneyPuck {saved_gp:g} GP",
                             "Fresh sources": f"MoneyPuck now {mp_gp:g} GP · NHL {nhl_gp:g} GP" if nhl_gp is not None else f"MoneyPuck now {mp_gp:g} GP",
                             "Meaning": "Check capture times; a newer season file may have arrived."})
        saved_pp, fresh_pp = _num(row.get("PP_TOI_per_game")), _num(live.get("PP TOI/GP"))
        if mp_gp is not None and mp_gp >= 3 and saved_pp is not None and fresh_pp is not None and abs(saved_pp - fresh_pp) >= 0.75:
            role = _num(row.get("PP_Role"))
            role_label = {0: "No unit", 1: "PP2", 2: "PP1"}.get(int(role), "Unknown") if role is not None else "Unknown"
            findings.append({"Player": name, "Team": team, "Source check": "PP ice time changed",
                             "Saved tracker": f"{saved_pp:.1f} min/game · estimated {role_label}",
                             "Fresh sources": f"MoneyPuck now {fresh_pp:.1f} min/game ({mp_gp:g} GP)",
                             "Meaning": "Review recent deployment; season ice-time rank is not a confirmed unit."})
    return findings
