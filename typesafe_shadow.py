"""Owner-only Jev review of pregame prop context. Never changes board rankings."""

from __future__ import annotations

import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

# Explicit allowlists keep settled outcomes and other postgame fields out of requests.
COMMON = (
    "Date", "Game", "Player", "Team", "Opp", "Pos", "Lineup_Status",
    "Injury_Status", "Opp_Goalie_Status", "Model_Player_GP", "Team_GP",
    "TOI_per_game", "TOI_Pct", "StarScore", "ShotIntent_Pct",
)
MARKETS = {
    "Goals": (
        "Goal_Line", "Goal_Odds_Over", "Goal_Book", "Conf_Goal", "Goal_mu",
        "iXG%", "i5v5_xG60", "PP_iXG60", "L5_G", "L10_G",
        "Pregame_G_L5_per_game", "Pregame_xG_L5_per_game", "opp_5v5_xGA60",
        "opp_5v5_HDCA60", "Opp_GAA", "Opp_GP", "Goalie_Context_Status",
    ),
    "Assists": (
        "Assists_Line", "Assists_Odds_Over", "Assists_Book", "Conf_Assists",
        "Assists_mu", "iXA%", "i5v5_primaryAssists60", "Assist_Volume",
        "L5_A", "L10_A", "PP_TOI_per_game", "Team_PP_xGF60",
    ),
    "Points": (
        "Points_Line", "Points_Odds_Over", "Points_Book", "Conf_Points",
        "Points_mu", "L10_P", "L5_G", "L5_A", "i5v5_points60",
        "PP_TOI_per_game", "Team_GF_Avg_L5", "Team_PP_xGF60",
    ),
    "Shots": (
        "SOG_Line", "SOG_Odds_Over", "SOG_Book", "Conf_SOG", "SOG_mu",
        "i5v5_SOG60", "i5v5_iCF60", "Med10_SOG", "Avg5_SOG",
        "Player_5v5_SOG_Share", "PP_SOG60", "Opp_SOG_Against_L10",
    ),
    "Power play": (
        "BDL_PPP_Line", "BDL_PPP_Odds", "BDL_PPP_Book", "PP_Role",
        "PP_Tier", "PP_TOI_per_game", "PP_TOI_Share", "PP_TOI_stability",
        "PP_iXG60", "PP_iXA60", "PP_Points60", "PP_OnIce_xGF60",
        "Team_PP_xGF60", "Opp_PK_xGA60", "Opp_PK_GA60", "PP_Env_Score",
        "PPP10_total",
    ),
}


def _clean(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    try:
        value = value.item()  # numpy scalar
    except AttributeError:
        pass
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    value = str(value).strip()
    return value[:120] if value and value.lower() not in {"nan", "none", "nat"} else None


def pregame_state(row, market: str) -> dict:
    """Only send a bounded set of source fields that existed before the slate."""
    if market not in MARKETS:
        raise ValueError(f"Unknown market: {market}")
    result = {"Market": market}
    for field in COMMON + MARKETS[market]:
        value = _clean(row.get(field))
        if value is not None:
            result[field] = value
    return result


def request_payload(row, market: str) -> dict:
    return {
        "model": MODEL,
        "state": pregame_state(row, market),
        "questions": {
            "opportunity": {
                "type": "choice",
                "instructions": (
                    "Classify the player's pregame opportunity for the selected prop market. "
                    "Use deployment, volume, role and directly relevant player stats. "
                    "A model confidence score or price alone is not evidence of opportunity. "
                    "Do not estimate a hit probability."
                ),
                "criteria": {
                    "strong": "Several direct, current opportunity indicators support this market.",
                    "mixed": "Direct indicators disagree or support is modest.",
                    "weak": "Direct indicators show limited opportunity for this market.",
                    "unknown": "Relevant opportunity data is missing or too sparse.",
                },
            },
            "matchup": {
                "type": "choice",
                "instructions": (
                    "Classify the pregame opponent environment for this specific market. "
                    "Only use opponent and goalie fields actually present; do not invent data."
                ),
                "criteria": {
                    "favorable": "Relevant opponent evidence favors opportunity.",
                    "neutral": "Relevant opponent evidence is mixed or ordinary.",
                    "difficult": "Relevant opponent evidence limits opportunity.",
                    "unknown": "Relevant opponent evidence is absent or too sparse.",
                },
            },
            "evidence": {
                "type": "choice",
                "instructions": (
                    "Rate whether the supplied pregame fields are sufficient for a useful "
                    "qualitative review of this prop. Thin season or goalie samples matter."
                ),
                "criteria": {
                    "adequate": "Multiple directly relevant fields with usable samples are present.",
                    "thin": "Useful fields exist, but samples or coverage are thin.",
                    "insufficient": "Essential opportunity or matchup fields are absent.",
                },
            },
        },
    }


def review(row, market: str, api_key: str, *, timeout: float = 15) -> dict:
    """One explicit owner action makes one billable TypeSafe request."""
    if not api_key:
        raise ValueError("TYPESAFE_API_KEY is not configured")
    payload = request_payload(row, market)
    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"TypeSafe returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError("TypeSafe is unreachable right now") from exc
    answers = result.get("answers")
    if not isinstance(answers, dict) or not all(
        isinstance(answers.get(name), dict) and answers[name].get("type") == "choice"
        for name in ("opportunity", "matchup", "evidence")
    ):
        raise RuntimeError("TypeSafe returned an unexpected answer format")
    return {
        "market": market,
        "state": pregame_state(row, market),
        "answers": answers,
        "model": result.get("model", MODEL),
        "usage": result.get("usage", {}),
    }
