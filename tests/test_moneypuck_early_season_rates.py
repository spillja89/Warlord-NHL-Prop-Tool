import unittest

import pandas as pd

from nhl_edge import normalize_goalies, normalize_skaters_pp, per60


class MoneyPuckEarlySeasonRatesTests(unittest.TestCase):
    def test_per60_uses_seconds_below_and_above_ten_thousand(self):
        stats = pd.Series([2.0, 4.0])
        seconds = pd.Series([3600.0, 14400.0])

        self.assertEqual(per60(stats, seconds).tolist(), [2.0, 1.0])


    def test_goalie_gaa_uses_seconds_early_in_season(self):
        goalies = pd.DataFrame([{
            "name": "Early Goalie", "team": "VAN", "situation": "all",
            "games_played": 2, "icetime": 7200, "ongoal": 60, "goals": 6,
        }])

        normalized = normalize_goalies(goalies)

        self.assertEqual(normalize_goalies(goalies).iloc[0]["GAA"], 3.0)

    def test_pp_on_ice_xg_is_separate_from_credited_assists(self):
        skaters = pd.DataFrame([{
            "playerId": 123, "team": "BOS", "situation": "5on4",
            "games_played": 5, "icetime": 1200,
            "I_F_xGoals": 0.4, "OnIce_F_xGoals": 2.0,
            "I_F_primaryAssists": 1, "I_F_secondaryAssists": 0,
        }])

        row = normalize_skaters_pp(skaters).iloc[0]

        self.assertAlmostEqual(row["PP_OnIce_xGF60"], 6.0)
        self.assertAlmostEqual(row["PP_iXG60"], 1.2)
        self.assertAlmostEqual(row["PP_iXA60"], 3.0)
