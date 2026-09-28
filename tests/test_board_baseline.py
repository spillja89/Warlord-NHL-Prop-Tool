import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

import nhl_edge
from warlords_night_board import baseline_audit, rank_warlords, render_warlords


class BoardBaselineTests(unittest.TestCase):
    def test_priced_baselines_are_audited_even_without_named_moves(self):
        rows = [
            {"Player": "Goal Scout", "Team": "BOS", "Game": "NYR@BOS",
             "Goal_Line": 0.5, "Goal_Odds_Over": 125, "Matrix_Goal": "Green",
             "Conf_Points": 84},
            {"Player": "Yellow Goal", "Team": "BOS", "Game": "NYR@BOS",
             "Goal_Line": 0.5, "Goal_Odds_Over": 200, "Matrix_Goal": "Yellow",
             "Conf_Points": 90},
            {"Player": "Point Tank", "Team": "CAR", "Game": "FLA@CAR",
             "Points_Line": 0.5, "Points_Odds_Over": -150,
             "Matrix_Points": "Green", "Conf_Points": 80},
            {"Player": "Assist Scout", "Team": "FLA", "Game": "FLA@CAR",
             "Assists_Line": 0.5, "Assists_Odds_Over": 140,
             "Matrix_Assists": "Green", "Conf_Assists": 70},
            {"Player": "Shot Scout", "Team": "FLA", "Game": "FLA@CAR",
             "SOG_Line": 2.5, "SOG_Odds_Over": 110,
             "Matrix_SOG": "Green", "Conf_SOG": 75},
        ]
        summary, by_team, roster = baseline_audit(pd.DataFrame(rows))
        goals = summary.set_index("Prop").loc["Goals"]
        self.assertEqual((goals["Priced"], goals["Green"], goals["Baseline"],
                          goals["Named move"]), (2, 1, 1, 0))
        self.assertEqual(roster.set_index("Player").loc["Goal Scout", "Status"], "Baseline only")
        carry = rank_warlords(pd.DataFrame(rows))["Carry"]
        self.assertEqual(len(carry), 1)
        self.assertTrue(carry[0]["baseline_only"])
        self.assertEqual(carry[0]["move"]["kind"], "BASELINE")
        html = render_warlords({"Carry": carry}, roles=("Carry",))
        self.assertIn("BASELINE ONLY", html)
        self.assertIn("Baseline screen · no upgraded move", html)
        self.assertEqual(summary.set_index("Prop").loc["Points", "Baseline"], 1)
        self.assertEqual(summary.set_index("Prop").loc["Assists", "Baseline"], 1)
        self.assertEqual(summary.set_index("Prop").loc["Shots", "Baseline"], 1)
        self.assertEqual(by_team.set_index(["Prop", "Team"]).loc[("Goals", "BOS"), "Baseline"], 1)

    def test_empty_slate_has_safe_audit(self):
        summary, by_team, roster = baseline_audit(pd.DataFrame())
        self.assertEqual(len(summary), 4)
        self.assertTrue(by_team.empty)
        self.assertTrue(roster.empty)

    def test_goalie_lookup_uses_slate_date_and_never_borrows_other_goalie_stats(self):
        with patch.object(nhl_edge, "http_session", return_value=object()), patch.object(
            nhl_edge, "http_get_text", return_value="<html></html>"
        ) as get_text:
            nhl_edge.fetch_dailyfaceoff_starters(date(2026, 9, 29))
        self.assertEqual(get_text.call_args.args[1],
                         "https://www.dailyfaceoff.com/starting-goalies/2026-09-29")

        goalie_df = pd.DataFrame([{"Team": "BOS", "Goalie": "Existing Goalie",
                                   "GP": 40, "SV": 0.900, "GAA": 3.1}])
        proxy = nhl_edge.build_team_goalie_map(goalie_df)
        resolved = nhl_edge.resolve_goalie_for_team(goalie_df, proxy, "BOS", "New Goalie")
        self.assertEqual(resolved["Goalie"], "New Goalie")
        self.assertIsNone(resolved["GAA"])
        self.assertEqual(resolved["Source"], "dailyfaceoff_name_only")


if __name__ == "__main__":
    unittest.main()
