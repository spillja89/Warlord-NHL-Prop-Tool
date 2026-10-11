import unittest
from unittest.mock import patch

import pandas as pd

from nhl_edge import load_moneypuck_best_effort


class MoneyPuckSeasonGateTests(unittest.TestCase):
    @patch("nhl_edge.current_season_start_year", return_value=2026)
    @patch("nhl_edge.load_moneypuck_csv")
    def test_seven_team_games_uses_current_source_across_all_inputs(self, load_csv, _year):
        def feed(_session, url):
            if "/2026/regular/teams.csv" in url:
                return pd.DataFrame({"team": ["CAR", "FLA"], "games_played": [7, 7]})
            if "/2026/regular/" in url:
                return pd.DataFrame({"team": ["CAR", "FLA"], "games_played": [5, 4]})
            raise AssertionError(f"Unexpected feed: {url}")

        load_csv.side_effect = feed
        for kind in ("skaters", "goalies", "teams"):
            with self.subTest(kind=kind):
                result = load_moneypuck_best_effort(object(), kind, {"CAR", "FLA"})
                self.assertEqual(result.attrs["moneypuck_start_year"], 2026)

    @patch("nhl_edge.current_season_start_year", return_value=2026)
    @patch("nhl_edge.load_moneypuck_csv")
    def test_six_team_games_retains_prior_season_for_slate(self, load_csv, _year):
        def feed(_session, url):
            if "/2026/regular/teams.csv" in url:
                return pd.DataFrame({"team": ["CAR", "FLA"], "games_played": [7, 6]})
            return pd.DataFrame({"team": ["CAR", "FLA"], "games_played": [5, 4]})

        load_csv.side_effect = feed
        result = load_moneypuck_best_effort(object(), "skaters", {"CAR", "FLA"})
        self.assertEqual(result.attrs["moneypuck_start_year"], 2025)


if __name__ == "__main__":
    unittest.main()
