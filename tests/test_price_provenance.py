from datetime import date
from unittest.mock import patch
import unittest
import sys
import types

import pandas as pd

# The test stubs the three feed calls; no HTTP package is needed to run it.
try:
    import requests  # noqa: F401
except ImportError:
    sys.modules["requests"] = types.ModuleType("requests")
import odds_ev_bdl
from roster_watchlist import add_roster_watchlist


class PriceProvenanceTests(unittest.TestCase):
    def test_watchlist_records_price_source_and_capture_time(self):
        self._check_watchlist()

    def _check_watchlist(self):
        games = lambda day, api_key: [
        {"id": 1, "home_team": {"tricode": "NYR"}, "away_team": {"tricode": "ANA"}}
        ] if day == "2026-10-19" else []
        props = lambda game_id, api_key: [
        {"player_id": 1, "prop_type": "anytime_goal", "line_value": 1,
         "market": {"type": "milestone", "odds": 500}, "vendor": "Book A"},
        {"player_id": 2, "prop_type": "anytime_goal", "line_value": 1,
         "market": {"type": "milestone", "odds": 900}, "vendor": "Book B"},
        ]
        players = lambda ids, api_key: {
        1: {"full_name": "Existing Player"}, 2: {"full_name": "New Player"}
        }
        frame = pd.DataFrame([{"Player": "Existing Player", "Player_ID": 101,
                           "Team": "NYR", "Goal_Line": float("nan"),
                           "Goal_Odds_Over": float("nan")}])
        rosters = {"NYR": {
        "existingplayer": {"name": "Existing Player", "id": 101, "position": "C"},
        "newplayer": {"name": "New Player", "id": 102, "position": "C"},
        }}
        with patch.object(odds_ev_bdl, "fetch_bdl_games_for_date", games), \
             patch.object(odds_ev_bdl, "fetch_bdl_props_for_game", props), \
             patch.object(odds_ev_bdl, "fetch_bdl_players_map", players):
            result = add_roster_watchlist(frame, None, {"NYR", "ANA"},
                                          {"NYR": "ANA@NYR"}, date(2026, 10, 19),
                                          "test-key", rosters=rosters)
        existing = result.loc[result["Player"].eq("Existing Player")].iloc[0]
        new = result.loc[result["Player"].eq("New Player")].iloc[0]
        self.assertEqual(existing["Goal_Odds_Source"], "BallDontLie")
        self.assertEqual(new["Goal_Odds_Source"], "BallDontLie")
        self.assertTrue(new["Odds_Checked_UTC"])
        self.assertEqual(new["Goal_Odds_Over"], 900)

