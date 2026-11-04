"""Versioned SOG player-profile ranking; not a shot probability.

Approved November 4, 2026 after a Green exact2.5 current-season replay.
The original score remains available to already-graded move rules.
"""
from __future__ import annotations

import math

VERSION = "sog-profile-v2-2026-11-04"


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _clip(value):
    return max(0.0, min(100.0, value))


def sog_drought_bonus(tier, drought):
    """Preserve the existing tier-specific SOG drought schedule."""
    d = _number(drought)
    if d is None or d < 0:
        return None
    tier = str(tier).strip().upper()
    steps = ([(1, 2), (2, 6), (3, 11)] if tier == "ELITE" else
             [(2, 2), (3, 6), (4, 10)] if tier == "STAR" else
             [(3, 2), (4, 5), (5, 8)])
    bonus = 0
    for cutoff, value in steps:
        if d >= cutoff:
            bonus = value
    return bonus


def sog_profile_confidence(toi_pct, shot_share_5v5, opponent_sog_l10,
                           tier, drought):
    """Return an integer profile score, or None for missing required inputs.

    Shot share is a percentage (e.g.15 means15%, not0.15). The fixed
    normalization anchors are35% team shot share and40 opponent SOG/game.
    Round once after the drought bonus. There is no injury-score adjustment.
    """
    toi, share, opponent = map(_number, (toi_pct, shot_share_5v5, opponent_sog_l10))
    bonus = sog_drought_bonus(tier, drought)
    if (any(v is None for v in (toi, share, opponent, bonus))
            or not 0 <= toi <= 100 or not 0 <= share <= 100 or opponent <= 0):
        return None
    base = (.55 * toi + .15 * _clip(share / 35.0 * 100.0)
            + .30 * _clip(opponent / 40.0 * 100.0))
    return int(round(_clip(base + bonus)))


def apply_sog_confidence(frame):
    """Update the displayed score while preserving original move inputs."""
    import pandas as pd

    out = frame.copy()
    if "Conf_SOG_Move" not in out:
        out["Conf_SOG_Move"] = out.get("Conf_SOG", pd.Series(float("nan"), index=out.index))
    values = [sog_profile_confidence(
        row.get("TOI_Pct"), row.get("Player_5v5_SOG_Share"),
        row.get("Opp_SOG_Against_L10"), row.get("Talent_Tier"),
        row.get("Drought_SOG")) for row in out.to_dict("records")]
    out["Conf_SOG"] = pd.Series(values, index=out.index, dtype="Float64")
    out["Conf_SOG_Version"] = VERSION
    return out
