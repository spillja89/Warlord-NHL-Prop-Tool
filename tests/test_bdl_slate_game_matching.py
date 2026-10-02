import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from odds_ev_bdl import _game_on_slate_date, merge_bdl_props_altlines


class BDLSlateGameMatchingTests(unittest.TestCase):
    def test_utc_midnight_still_belongs_to_central_time_slate(self):
        self.assertTrue(_game_on_slate_date(
            {"start_time_utc": "2026-10-03T01:00:00.000Z"}, date(2026, 10, 2)
        ))
        self.assertFalse(_game_on_slate_date(
            {"start_time_utc": "2026-10-04T00:00:00.000Z"}, date(2026, 10, 2)
        ))

    @patch("odds_ev_bdl.fetch_bdl_players_map")
    @patch("odds_ev_bdl.fetch_bdl_props_for_game")
    @patch("odds_ev_bdl.fetch_bdl_games_for_date")
    def test_next_games_better_price_cannot_replace_todays_quote(self, games, props, players):
        games.side_effect = [
            [{"id": 3328068, "start_time_utc": "2026-10-03T01:00:00.000Z"}],
            [{"id": 3328078, "start_time_utc": "2026-10-04T00:00:00.000Z"}],
        ]
        props.return_value = [{
            "player_id": 1, "prop_type": "shots_on_goal", "line_value": 2.5,
            "vendor": "fanduel", "market": {"type": "over_under", "over_odds": -113},
        }]
        players.return_value = {1: {"full_name": "Wyatt Johnston", "team": {"abbreviation": "DAL"}}}
        tracker = pd.DataFrame([{"Player": "Wyatt Johnston", "Team": "DAL", "Exp_S_10": 30}])

        result = merge_bdl_props_altlines(tracker, "2026-10-02", api_key="test-key")

        self.assertEqual(result.loc[0, "BDL_SOG_Odds"], -113)
        self.assertEqual(result.loc[0, "BDL_SOG_Line"], 2.5)
        props.assert_called_once()
        self.assertEqual(props.call_args.args[0], 3328068)


if __name__ == "__main__":
    unittest.main()
