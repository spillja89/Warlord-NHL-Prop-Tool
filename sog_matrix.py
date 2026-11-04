"""Versioned, additive SOG2.5 Green lane approved November4,2026."""
from __future__ import annotations
import math

from sog_confidence import VERSION as CONFIDENCE_VERSION

VERSION = "sog25-green-ixg60-2026-11-04"
IXG_FLOOR = 60


def _num(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def apply_sog25_green(frame, matrix_classifier):
    """Preserve existing Green/Elite; add only the approved iXG60 lane.

    Reuse the engine's original classifier for every remaining requirement.
    Passing100 for its iXG argument removes only its repeated iXG checks;
    the actual saved iXG must first meet60. No source stat is overwritten.
    Confidence is scored separately and is not a matrix requirement.
    """
    import pandas as pd
    out = frame.copy()
    if "Matrix_SOG_Original" not in out:
        out["Matrix_SOG_Original"] = out.get("Matrix_SOG", pd.Series("",index=out.index))
    if "Matrix_SOG_Version" not in out:
        out["Matrix_SOG_Version"] = ""
    if "Matrix_SOG_Route" not in out:
        out["Matrix_SOG_Route"] = ""
    for index, row in out.iterrows():
        if (_num(row.get("SOG_Line")) != 2.5
                or row.get("Model_Stats_Season") != "2026-2027"
                or row.get("Conf_SOG_Version") != CONFIDENCE_VERSION):
            continue
        out.at[index,"Matrix_SOG_Version"] = VERSION
        original = str(row.get("Matrix_SOG_Original", "")).strip().casefold()
        if original in {"green", "🟢"}:
            out.at[index,"Matrix_SOG"] = "Green"
            out.at[index,"Matrix_SOG_Route"] = "Original Green / Elite"
            continue
        fields = ("iXG%", "Med10_SOG", "ShotIntent_Pct", "TOI_Pct",
                  "team_5v5_SF60_pct", "Opp_DefWeak")
        ixg,median,intent,toi,team,opponent = [_num(row.get(k)) for k in fields]
        valid = (all(v is not None for v in (ixg,median,intent,toi,team,opponent))
                 and median >= 0 and all(0 <= v <= 100 for v in (ixg,intent,toi,team,opponent)))
        if valid and ixg >= IXG_FLOOR and matrix_classifier(
                100.0, median, str(row.get("Pos", "F")),
                shot_intent_pct=intent,toi_pct=toi,team_sf_pct=team,opp_defweak=opponent) == "Green":
            out.at[index,"Matrix_SOG"] = "Green"
            out.at[index,"Matrix_SOG_Route"] = "Shot profile · iXG60"
        else:
            out.at[index,"Matrix_SOG_Route"] = "No Green path"
    return out
