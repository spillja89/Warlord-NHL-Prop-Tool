import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from odds_api_nhl import _market_quote, merge_odds_api_props


class OddsAPINHLTests(unittest.TestCase):
    def test_market_quote_only_accepts_priced_over_and_anytime_yes(self):
        self.assertEqual(
            _market_quote("player_points", {"name": "Over", "description": "Leon Draisaitl", "point": 1.5, "price": 145}),
            ("Points", "leon draisaitl", 1.5, 145.0),
        )
        self.assertIsNone(_market_quote("player_points", {"name": "Under", "description": "Leon Draisaitl", "point": 1.5, "price": -165}))
        self.assertIsNone(_market_quote("player_points", {"name": "Over", "description": "Leon Draisaitl", "point": 1.0, "price": 145}))
        self.assertEqual(
            _market_quote("player_goal_scorer_anytime", {"name": "Leon Draisaitl", "price": 125}),
            ("ATG", "leon draisaitl", 0.5, 125.0),
        )
        self.assertEqual(
            _market_quote("player_power_play_points", {"name": "Over", "description": "Cole Caufield", "point": 0.5, "price": 175}),
            ("PPP", "cole caufield", 0.5, 175.0),
        )

    @patch("odds_api_nhl._get")
    def test_uses_matching_game_and_best_book_with_bdl_fallback(self, get):
        get.side_effect = [
            [{"id": "edm-van", "commence_time": "2026-09-30T01:00:00Z", "home_team": "Vancouver Canucks", "away_team": "Edmonton Oilers"}],
            {"bookmakers": [
                {"title": "Fanatics", "markets": [
                    {"key": "player_goal_scorer_anytime", "outcomes": [{"name": "Leon Draisaitl", "price": 125}]},
                    {"key": "player_points", "outcomes": [{"name": "Over", "description": "Leon Draisaitl", "point": 1.5, "price": 145}]},
                ]},
                {"title": "BetRivers", "markets": [
                    {"key": "player_goal_scorer_anytime", "outcomes": [{"name": "Leon Draisaitl", "price": 130}]},
                ]},
            ]},
        ]
        tracker = pd.DataFrame([
            {"Player": "Leon Draisaitl", "Team": "EDM", "Opp": "VAN", "BDL_Goal_Line_1": 0.5, "BDL_Goal_Odds_1": 120, "BDL_Goal_Book_1": "DraftKings", "BDL_Goal_Line": 0.5},
            {"Player": "Leon Draisaitl", "Team": "EDM", "Opp": "CGY"},
        ])
        out = merge_odds_api_props(tracker, date(2026, 9, 29), "test-key")
        self.assertEqual(out.loc[0, "BDL_Goal_Odds"], 130)
        self.assertEqual(out.loc[0, "BDL_Goal_Book"], "BetRivers")
        self.assertEqual(out.loc[0, "BDL_Points_Line"], 1.5)
        self.assertEqual(out.loc[0, "BDL_Points_Book"], "Fanatics")
        self.assertTrue(pd.isna(out.loc[1, "BDL_Goal_Odds"]))


if __name__ == "__main__":
    unittest.main()
