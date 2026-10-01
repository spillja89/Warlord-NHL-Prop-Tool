import unittest
from unittest.mock import patch

import pandas as pd

import grade_daily_tracker as grader


DAY = "2026-09-30"
GAME_ID = 2026020007


def _box():
    return {
        "gameState": "OFF", "startTimeUTC": "2026-10-01T02:00:00Z",
        "awayTeam": {"abbrev": "LAK"}, "homeTeam": {"abbrev": "COL"},
        "playerByGameStats": {
            "awayTeam": {"forwards": [], "defense": []},
            "homeTeam": {"forwards": [
                {"playerId": 1, "name": {"default": "A. Skater"}, "goals": 1, "assists": 0, "points": 1, "sog": 2},
                {"playerId": 2, "name": {"default": "B. Skater"}, "goals": 0, "assists": 1, "points": 1, "sog": 1},
                {"playerId": 3, "name": {"default": "C. Skater"}, "goals": 0, "assists": 0, "points": 0, "sog": 1},
            ], "defense": []},
        },
    }


def _landing():
    return {"gameState": "OFF", "summary": {"scoring": [
        {"goals": [
            {"strength": "pp", "playerId": 1, "assists": [{"playerId": 2}]},
            {"strength": "ev", "playerId": 3, "assists": []},
        ]},
    ]}}


def _frame():
    return pd.DataFrame([
        {"Date": DAY, "Game": "LAK@COL", "Team": "COL", "Player": name,
         "Player_ID": player_id, "BDL_PPP_Line": line}
        for name, player_id, line in [("A. Skater", 1, 0.5), ("B. Skater", 2, 0.5),
                                      ("C. Skater", 3, 0.5)]
    ])


class PowerPlayGradingTests(unittest.TestCase):
    def test_only_power_play_scorer_and_assists_win(self):
        with patch.object(grader, "schedule_for_day", return_value={("LAK", "COL"): GAME_ID}), \
             patch.object(grader, "_json", side_effect=lambda _s, path: _landing() if path.endswith("landing") else _box()):
            graded, summary = grader.grade_tracker(_frame(), DAY, object())
        self.assertEqual(graded["Actual_PPP"].tolist(), [1, 1, 0])
        self.assertEqual(graded["Outcome_PPP"].tolist(), ["W", "W", "L"])
        self.assertEqual(summary["markets"]["PPP"]["W"], 2)

    def test_missing_scoring_summary_does_not_invent_loss(self):
        with patch.object(grader, "schedule_for_day", return_value={("LAK", "COL"): GAME_ID}), \
             patch.object(grader, "_json", side_effect=lambda _s, path: {"gameState": "OFF"} if path.endswith("landing") else _box()):
            graded, _ = grader.grade_tracker(_frame(), DAY, object())
        self.assertTrue(graded["Actual_PPP"].isna().all())
        self.assertEqual(graded["Match_Status_PPP"].tolist(), ["NO_STAT"] * 3)
        self.assertEqual(graded["Outcome_PPP"].tolist(), [""] * 3)


if __name__ == "__main__":
    unittest.main()
