import pandas as pd

from points15_research import attach_flags, context, summarize


def _row(day, player, outcome, *, pp=9.2, attempts=15.4, hdca=1.4,
         season="2026-2027", line=1.5):
    return {
        "Date": day, "Game": "A@B", "Player": player,
        "Points_Line": line, "Points_Odds_Over": 150,
        "Matrix_Points": "Green", "Model_Stats_Season": season,
        "PP_OnIce_xGF60": pp, "i5v5_iCF60": attempts,
        "opp_5v5_HDCA60": hdca, "Outcome_Points": outcome,
        "Match_Status_Points": "OK",
    }


def test_research_tags_are_pregame_context_and_missing_is_not_a_failed_signal():
    tagged = attach_flags(pd.DataFrame([
        _row("2026-10-28", "A", "W", hdca=1.1),
        _row("2026-10-28", "B", "L", pp=None, attempts=None),
        _row("2026-10-28", "C", "W", season="2025-2026"),
        _row("2026-10-28", "D", "W", line=0.5),
    ]))
    assert tagged.loc[0, "P15_Research_Dual"]
    assert tagged.loc[0, "P15_Research_Stingy_Defense"]
    assert not tagged.loc[1, "P15_Research_Metrics_Complete"]
    assert not tagged.loc[1, "P15_Research_Dual"]
    assert not tagged.loc[2, "P15_Research_Eligible"]
    assert not tagged.loc[3, "P15_Research_Eligible"]
    assert context(tagged.loc[0])["pp_pressure"]


def test_results_use_matched_fields_and_keep_forward_separate():
    graded = attach_flags(pd.DataFrame([
        _row("2026-10-27", "A", "W"),
        _row("2026-10-27", "B", "L", pp=8, attempts=16),
        _row("2026-10-27", "C", "W", pp=None, attempts=None),
        _row("2026-10-28", "D", "W", pp=9.5, attempts=16),
        _row("2026-10-28", "D", "W", pp=9.5, attempts=16),
    ]))
    discovery = summarize(graded, forward=False).set_index("Group")
    assert discovery.loc["Green baseline · all", "Picks"] == 3
    assert discovery.loc["Green baseline · PP + shots recorded", "Picks"] == 2
    assert discovery.loc["Both research signals", "Wins"] == 1
    forward = summarize(graded, forward=True).set_index("Group")
    assert forward.loc["Green baseline · all", "Picks"] == 1
    assert forward.loc["Both research signals", "Picks"] == 1
