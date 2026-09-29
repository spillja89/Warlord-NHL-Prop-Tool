import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

import nhl_edge
from player_availability import unavailable_mask
from warlords_night_board import (baseline_audit, featured_warlords,
                                  rank_priced_slate, rank_warlords, render_warlords)


class BoardBaselineTests(unittest.TestCase):
    def test_injury_feed_returns_matches_and_blocks_out_players(self):
        reports = {
            "BOS": [{"Team": "BOS", "Player": "Out Player", "Status": "out",
                     "Injury": "Lower Body", "Expected_Return": ""}],
            "CAR": [{"Team": "CAR", "Player": "Questionable Player", "Status": "DTD",
                     "Injury": "Upper Body", "Expected_Return": ""}],
        }
        with patch.object(nhl_edge, "fetch_dfo_injuries_for_team",
                          side_effect=lambda team, debug=False: reports[team]), patch.object(
                              nhl_edge.time, "sleep"):
            injuries = nhl_edge.fetch_dfo_injuries_for_teams({"BOS", "CAR"})
        self.assertEqual(len(injuries), 2)
        self.assertEqual(injuries.set_index("Player").loc["Out Player", "Player_norm"], "out player")
        skaters = pd.DataFrame([
            {"Player": "Out Player", "Team": "BOS", "Goal_Line": 0.5,
             "Goal_Odds_Over": 120, "Matrix_Goal": "Green", "Conf_Points": 90},
            {"Player": "Questionable Player", "Team": "CAR", "Goal_Line": 0.5,
             "Goal_Odds_Over": 120, "Matrix_Goal": "Green", "Conf_Points": 90},
        ])
        tagged = nhl_edge.apply_injury_dfo(skaters, injuries)
        self.assertEqual(tagged["Injury_Status"].tolist(), ["Out", "GTD"])
        self.assertEqual(unavailable_mask(tagged).tolist(), [True, False])
        board = rank_priced_slate(tagged)["Carry"]
        self.assertEqual([card["player"] for card in board], ["Questionable Player"])

    def test_saved_tracker_available_false_is_excluded_from_board(self):
        rows = pd.DataFrame([
            {"Player": "Out Player", "Team": "BOS", "Available": False,
             "Goal_Line": 0.5, "Goal_Odds_Over": 120,
             "Matrix_Goal": "Green", "Conf_Points": 90},
            {"Player": "Active Player", "Team": "BOS", "Available": True,
             "Goal_Line": 0.5, "Goal_Odds_Over": 120,
             "Matrix_Goal": "Green", "Conf_Points": 90},
        ])
        self.assertEqual(unavailable_mask(rows).tolist(), [True, False])
        self.assertEqual([card["player"] for card in rank_priced_slate(rows)["Carry"]],
                         ["Active Player"])

    def test_board_price_gap_uses_the_displayed_line_and_american_odds(self):
        rows = pd.DataFrame([
            {"Player": "Priced", "Team": "BOS", "Goal_Line": 0.5,
             "Goal_Odds_Over": 130, "Goal_p_model_over": 0.50,
             "Matrix_Goal": "Green", "Conf_Points": 90},
            {"Player": "Mismatched Model", "Team": "CAR", "Goal_Line": 0.5,
             "Goal_Odds_Over": -150, "ATG_Line": 1.5,
             "ATG_p_model_over": 0.40, "Matrix_Goal": "Green", "Conf_Points": 90},
        ])
        cards = {card["player"]: card for card in rank_priced_slate(rows)["Carry"]}
        self.assertAlmostEqual(cards["Priced"]["book_prob"], 100 / 230)
        self.assertAlmostEqual(cards["Priced"]["price_gap"], 0.50 - 100 / 230)
        self.assertIsNone(cards["Mismatched Model"]["model_prob"])
        html = render_warlords({"Carry": [cards["Priced"]]}, roles=("Carry",))
        self.assertIn("Model 50.0%", html)
        self.assertIn("Book break-even 43.5%", html)
        self.assertIn("Gap +6.5 pp", html)

    def test_priced_pool_is_complete_but_cards_require_baseline_and_half_rate_move(self):
        strong = {"name": "Strong Move", "kind": "HEAVY", "rule": "test",
                  "wins": 6, "picks": 10, "later_wins": 3, "later_picks": 5}
        weak = {**strong, "name": "Weak Move", "wins": 4}
        rows = [
            {"Player": "Strong", "Team": "BOS", "Goal_Line": 0.5,
             "Goal_Odds_Over": 120, "Matrix_Goal": "Green", "Conf_Points": 84,
             "Conf_Goal": 88},
            {"Player": "Yellow", "Team": "BOS", "Goal_Line": 0.5,
             "Goal_Odds_Over": 140, "Matrix_Goal": "Yellow", "Conf_Points": 95,
             "Conf_Goal": 93},
            {"Player": "Weak", "Team": "CAR", "Goal_Line": 0.5,
             "Goal_Odds_Over": 130, "Matrix_Goal": "Green", "Conf_Points": 85,
             "Conf_Goal": 85},
            {"Player": "Alternate", "Team": "FLA", "Goal_Line": 1.5,
             "Goal_Odds_Over": 500, "Matrix_Goal": "Green", "Conf_Points": 90,
             "Conf_Goal": 90},
        ]
        def moves(row):
            goal = [strong] if row["Player"] in {"Strong", "Yellow", "Alternate"} else [weak]
            return {"Goal": goal, "Assists": [], "Points": [], "SOG": []}
        with patch("warlords_night_board.fired_moves", side_effect=moves):
            pool = rank_priced_slate(pd.DataFrame(rows))
        self.assertEqual(len(pool["Carry"]), 4)
        self.assertEqual(pool["Carry"][0]["player"], "Yellow")
        featured = featured_warlords(pool)
        self.assertEqual([card["player"] for card in featured["Carry"]], ["Strong"])
        html = render_warlords(featured, roles=("Carry",))
        self.assertIn("60.0%", html)
        self.assertIn("HISTORICAL MOVE", html)
        self.assertIn("CONF 88 Green", html)
        self.assertIn("Full fired move list (1)", html)

    def test_featured_cards_show_and_rank_by_best_qualifying_move(self):
        common = {"kind": "HEAVY", "rule": "test", "later_wins": 3,
                  "later_picks": 5}
        weaker = {**common, "name": "Weaker", "wins": 6, "picks": 10}
        stronger = {**common, "name": "Stronger", "wins": 8, "picks": 10}
        cards = {"Carry": [
            {"player": "High Conf", "baseline_rule": "Green", "confidence": 99,
             "matrix": "Green", "moves": [weaker], "move": weaker},
            {"player": "High Move", "baseline_rule": "Green", "confidence": 84,
             "matrix": "Green", "moves": [weaker, stronger], "move": weaker},
        ]}
        featured = featured_warlords(cards)
        self.assertEqual([card["player"] for card in featured["Carry"]],
                         ["High Move", "High Conf"])
        self.assertEqual(featured["Carry"][0]["move"]["name"], "Stronger")
        display = {**featured["Carry"][0], "team": "BOS", "game": "NYR@BOS",
                   "market": "Goal", "line": 0.5, "odds": 120, "goalie": "",
                   "goalie_status": "Unknown"}
        html = render_warlords({"Carry": [display]}, roles=("Carry",))
        self.assertIn("80.0%", html)
        self.assertIn("CONF 84 Green", html)

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
