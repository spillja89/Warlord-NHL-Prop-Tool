"""Owner-only Jev review of pregame prop context. Never changes board rankings."""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
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
QUOTE_FIELDS = {
    "Goals": ("Goal_Line", "Goal_Odds_Over"),
    "Assists": ("Assists_Line", "Assists_Odds_Over"),
    "Points": ("Points_Line", "Points_Odds_Over"),
    "Shots": ("SOG_Line", "SOG_Odds_Over"),
    "Power play": ("BDL_PPP_Line", "BDL_PPP_Odds"),
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


def review_source_difference(finding: dict, nhl_checked: str, mp_updated: str,
                             api_key: str, *, timeout: float = 15) -> dict:
    """Classify one displayed source difference; this cannot certify either feed."""
    if not api_key:
        raise ValueError("TYPESAFE_API_KEY is not configured")
    fields = ("Player", "Team", "Source check", "Saved tracker", "Fresh sources", "Meaning")
    state = {key: _clean(finding.get(key)) for key in fields}
    state["NHL checked UTC"] = _clean(nhl_checked)
    state["MoneyPuck file updated"] = _clean(mp_updated)
    payload = {
        "model": MODEL,
        "state": state,
        "questions": {
            "interpretation": {
                "type": "choice",
                "instructions": (
                    "Classify the displayed difference between saved pregame tracker values and fresh "
                    "season source values. Use only the fields and timestamps supplied. "
                    "Do not decide whether a player will hit a prop."
                ),
                "criteria": {
                    "update_timing": "The difference is consistent with feeds updating at different times.",
                    "definition_or_sample": "The values appear to use different samples or definitions.",
                    "needs_source_review": "The values materially disagree and a human should inspect the source records.",
                    "insufficient": "The supplied fields do not establish the cause of the difference.",
                },
            },
        },
    }
    request = Request(
        API_URL, data=json.dumps(payload).encode("utf-8"),
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
    answer = result.get("answers", {}).get("interpretation", {})
    if answer.get("type") != "choice" or answer.get("choice") not in payload["questions"]["interpretation"]["criteria"]:
        raise RuntimeError("TypeSafe returned an unexpected source-review format")
    return {"state": state, "answer": answer, "model": result.get("model", MODEL),
            "usage": result.get("usage", {})}


def review_flagged_cards(candidates: list[dict], api_key: str, *, limit: int = 8) -> list[dict]:
    """Run a bounded, explicit shadow review; one billable request per unique card."""
    if not api_key:
        raise ValueError("TYPESAFE_API_KEY is not configured")
    selected, seen = [], set()
    for candidate in candidates:
        row, market = candidate["row"], candidate["market"]
        key = (str(row.get("Player_ID") or candidate["player"]), market)
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
        if len(selected) >= max(1, min(limit, 8)):
            break

    def one(candidate):
        row, market = candidate["row"], candidate["market"]
        output = {"Date": _clean(row.get("Date")), "Player_ID": _clean(row.get("Player_ID")),
                  "Player": candidate["player"], "Team": candidate["team"], "Market": market,
                  "Flag": "; ".join(candidate.get("issues", [])),
                  "Odds_Checked_UTC": _clean(row.get("Odds_Checked_UTC"))}
        line_field, odds_field = QUOTE_FIELDS[market]
        if market == "Power play":
            quotes = []
            for suffix in ("", "_1", "_2", "_3", "_4"):
                try:
                    line = float(row.get(f"BDL_PPP_Line{suffix}"))
                    odds = float(row.get(f"BDL_PPP_Odds{suffix}"))
                except (TypeError, ValueError):
                    continue
                if math.isclose(line, 0.5) and math.isfinite(odds):
                    quotes.append(odds)
            output["Prop line"] = 0.5 if quotes else None
            output["Saved odds"] = max(quotes) if quotes else None
        else:
            output["Prop line"] = _clean(row.get(line_field))
            output["Saved odds"] = _clean(row.get(odds_field))
        output["Fired moves"] = _clean(row.get("Fired_Moves"))
        try:
            result = review(row, market, api_key)
            for name in ("opportunity", "matchup", "evidence"):
                answer = result["answers"][name]
                output[name.title()] = answer.get("choice")
                output[f"{name.title()} label confidence"] = answer.get("confidence")
            output["Jev model"] = result["model"]
            output["Status"] = "Reviewed"
        except (RuntimeError, ValueError) as exc:
            output["Status"] = f"Error: {exc}"
        return output

    with ThreadPoolExecutor(max_workers=3) as pool:
        return list(pool.map(one, selected))
