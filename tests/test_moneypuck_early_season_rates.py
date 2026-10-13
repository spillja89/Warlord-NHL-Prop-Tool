import unittest

import pandas as pd

from nhl_edge import normalize_goalies, per60


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
