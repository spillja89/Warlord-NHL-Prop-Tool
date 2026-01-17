
"""
NHL EDGE v2.0 — 5v5 Context + Defensive Matchup Upgrade (NA-safe)
=================================================================

Patch notes:
- Makes percentile helper NA-safe: pd.NA / None / strings -> coerced with pd.to_numeric(errors="coerce")
- Missing values become neutral (50th percentile) so you can test wiring with placeholder columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd


# -----------------------------
# 1) EXACT FIELDS (FINALIZED)
# -----------------------------
PLAYER_5V5_FIELDS: List[str] = [
    "i5v5_oiGF60",
    "i5v5_shotAssists60",
    "i5v5_points60",
    "i5v5_iCF60",
]

TEAM_DEF_5V5_FIELDS: List[str] = [
    "i5v5_HDCA60",
    "i5v5_xGA60",
    "i5v5_slotShotsAgainst60",
    "i5v5_DZturnovers60",
]

PLAYER_CORE_FIELDS: List[str] = [
    "ixG_pct",
    "ixA_pct",
]


@dataclass(frozen=True)
class EdgeV2Config:
    defense_window_games: int = 10
    player_stability_window_games: int = 15

    blend_recent_weight: float = 0.65
    blend_season_weight: float = 0.35

    # Player Stability Score weights
    w_oiGF60: float = 0.35
    w_shotAst60: float = 0.30
    w_pts60: float = 0.20
    w_iCF60: float = 0.15

    # Defense Vulnerability Score weights
    w_HDCA60: float = 0.35
    w_xGA60: float = 0.30
    w_slotSA60: float = 0.20
    w_DZto60: float = 0.15

    # Add-ons
    points_add_player: float = 0.25
    points_add_defense: float = 0.25

    assist_add_player: float = 0.10
    assist_add_defense: float = 0.15

    goals_add_defense: float = 0.30
    sog_add_defense: float = 0.10

    # Auto-pivot thresholds
    th_ix_involvement: float = 85.0
    th_player_stability: float = 65.0
    th_def_vuln: float = 60.0


def _pct_rank(series: pd.Series) -> pd.Series:
    """Return 0–100 percentile rank for a series (higher is better)."""
    return series.rank(pct=True, method="average") * 100.0


def _safe_pct(df: pd.DataFrame, col: str, group_col: Optional[str] = None) -> pd.Series:
    """
    Compute percentile rank 0–100 for a column.

    NA-safe:
    - coerces pd.NA/None/strings to numeric (NaN)
    - returns neutral 50 when NaN (lets you test with placeholder columns)
    """
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

    if group_col and group_col in df.columns:
        out = df.groupby(group_col, dropna=False)[col].transform(
            lambda x: _pct_rank(pd.to_numeric(x, errors="coerce"))
        )
        return out.fillna(50.0)

    s = pd.to_numeric(df[col], errors="coerce")
    return _pct_rank(s).fillna(50.0)


def _clamp01(x: pd.Series) -> pd.Series:
    return x.clip(lower=0.0, upper=100.0)


def add_v2_scores(
    df: pd.DataFrame,
    cfg: EdgeV2Config = EdgeV2Config(),
    *,
    position_col: Optional[str] = None,
) -> pd.DataFrame:
    out = df.copy()

    required = PLAYER_5V5_FIELDS + TEAM_DEF_5V5_FIELDS + PLAYER_CORE_FIELDS
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(
            "DataFrame is missing required columns for v2 upgrade:\n"
            + "\n".join(f" - {c}" for c in missing)
        )

    # Player Stability (within position percentiles if provided)
    oiGF_p = _safe_pct(out, "i5v5_oiGF60", position_col)
    sa_p   = _safe_pct(out, "i5v5_shotAssists60", position_col)
    pts_p  = _safe_pct(out, "i5v5_points60", position_col)
    icf_p  = _safe_pct(out, "i5v5_iCF60", position_col)

    out["v2_player_stability"] = _clamp01(
        cfg.w_oiGF60 * oiGF_p +
        cfg.w_shotAst60 * sa_p +
        cfg.w_pts60 * pts_p +
        cfg.w_iCF60 * icf_p
    )

    # Defense Vulnerability (higher = worse defense = better matchup)
    hdca_p = _safe_pct(out, "i5v5_HDCA60", None)
    xga_p  = _safe_pct(out, "i5v5_xGA60", None)
    slot_p = _safe_pct(out, "i5v5_slotShotsAgainst60", None)
    dzto_p = _safe_pct(out, "i5v5_DZturnovers60", None)

    out["v2_defense_vulnerability"] = _clamp01(
        cfg.w_HDCA60 * hdca_p +
        cfg.w_xGA60 * xga_p +
        cfg.w_slotSA60 * slot_p +
        cfg.w_DZto60 * dzto_p
    )

    # Add-on deltas (scaled to ~0-1)
    out["v2_points_delta"] = (
        cfg.points_add_player * out["v2_player_stability"] +
        cfg.points_add_defense * out["v2_defense_vulnerability"]
    ) / 100.0

    out["v2_assists_delta"] = (
        cfg.assist_add_player * out["v2_player_stability"] +
        cfg.assist_add_defense * out["v2_defense_vulnerability"]
    ) / 100.0

    out["v2_goals_delta"] = (cfg.goals_add_defense * out["v2_defense_vulnerability"]) / 100.0
    out["v2_sog_delta"]   = (cfg.sog_add_defense   * out["v2_defense_vulnerability"]) / 100.0

    ixg = pd.to_numeric(out["ixG_pct"], errors="coerce").fillna(0.0)
    ixa = pd.to_numeric(out["ixA_pct"], errors="coerce").fillna(0.0)
    ix_involve = np.maximum(ixg, ixa)

    out["v2_points_auto_flag"] = (
        (ix_involve >= cfg.th_ix_involvement) &
        (out["v2_player_stability"] >= cfg.th_player_stability) &
        (out["v2_defense_vulnerability"] >= cfg.th_def_vuln)
    )

    def _best_market_row(ixg_pct: float, ixa_pct: float, def_vuln: float) -> str:
        if max(ixg_pct, ixa_pct) < 60:
            return "PASS"
        if ixg_pct >= 85 and ixg_pct >= ixa_pct + 8:
            return "GOAL"
        if ixg_pct >= 75 and def_vuln >= 65:
            return "SOG"
        return "POINT"

    out["v2_best_market"] = [
        _best_market_row(float(a), float(b), float(c))
        for a, b, c in zip(ixg.tolist(), ixa.tolist(), out["v2_defense_vulnerability"].tolist())
    ]

    out["v2_pivot"] = np.where(out["v2_points_auto_flag"], "PIVOT: POINT", "")
    return out


def apply_v2_to_existing_edges(
    df: pd.DataFrame,
    *,
    points_edge_col: Optional[str] = None,
    assists_edge_col: Optional[str] = None,
    goals_edge_col: Optional[str] = None,
    sog_edge_col: Optional[str] = None,
    cfg: EdgeV2Config = EdgeV2Config(),
    position_col: Optional[str] = None,
) -> pd.DataFrame:
    out = add_v2_scores(df, cfg, position_col=position_col)

    def _add(col: Optional[str], delta_col: str):
        if col and col in out.columns:
            out[f"{col}_v2"] = pd.to_numeric(out[col], errors="coerce").fillna(0.0) + pd.to_numeric(out[delta_col], errors="coerce").fillna(0.0)

    _add(points_edge_col, "v2_points_delta")
    _add(assists_edge_col, "v2_assists_delta")
    _add(goals_edge_col, "v2_goals_delta")
    _add(sog_edge_col, "v2_sog_delta")

    return out


if __name__ == "__main__":
    # Quick smoke test
    n = 5
    df = pd.DataFrame({
        "Pos": ["C","W","D","C","W"],
        "ixG_pct": [90, 10, 80, 70, 88],
        "ixA_pct": [92, 30, 20, 75, 70],
        "i5v5_oiGF60": [pd.NA]*n,
        "i5v5_shotAssists60": [pd.NA]*n,
        "i5v5_points60": [pd.NA]*n,
        "i5v5_iCF60": [pd.NA]*n,
        "i5v5_HDCA60": [pd.NA]*n,
        "i5v5_xGA60": [pd.NA]*n,
        "i5v5_slotShotsAgainst60": [pd.NA]*n,
        "i5v5_DZturnovers60": [pd.NA]*n,
    })
    out = apply_v2_to_existing_edges(df, position_col="Pos")
    print(out[["v2_player_stability","v2_defense_vulnerability","v2_best_market","v2_pivot"]])
