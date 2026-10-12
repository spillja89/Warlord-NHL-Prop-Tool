import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from nhl_edge import load_moneypuck_best_effort


class CurrentSeasonCutoverTests(unittest.TestCase):
    def test_october_12_keeps_existing_ten_game_gate(self):
        current = pd.DataFrame([{"team": "OTT", "games_played": 6}])
        prior = pd.DataFrame([{"team": "OTT", "games_played": 82}])
        with patch("nhl_edge.load_moneypuck_csv", side_effect=[current, prior]) as fetch:
            result = load_moneypuck_best_effort(None, "skaters", {"OTT"}, date(2026, 10, 12))
        self.assertEqual(result.attrs["moneypuck_start_year"], 2025)
        self.assertEqual(fetch.call_count, 2)

    def test_october_13_uses_current_season_even_below_ten_games(self):
        current = pd.DataFrame([{"team": "OTT", "games_played": 6}])
        with patch("nhl_edge.load_moneypuck_csv", return_value=current) as fetch:
            result = load_moneypuck_best_effort(None, "teams", {"OTT"}, date(2026, 10, 13))
        self.assertEqual(result.attrs["moneypuck_start_year"], 2026)
        self.assertEqual(fetch.call_count, 1)

    def test_october_13_never_uses_prior_season_if_current_is_incomplete(self):
        current = pd.DataFrame([{"team": "OTT", "games_played": 6}])
        with patch("nhl_edge.load_moneypuck_csv", return_value=current) as fetch:
            with self.assertRaisesRegex(RuntimeError, "current-season.*missing slate teams"):
                load_moneypuck_best_effort(None, "goalies", {"OTT", "NJD"}, date(2026, 10, 13))
        self.assertEqual(fetch.call_count, 1)


if __name__ == "__main__":
    unittest.main()
