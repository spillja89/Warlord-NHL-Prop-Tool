from datetime import date
import unittest
from unittest.mock import patch

import pandas as pd

from roster_watchlist import add_roster_watchlist
from odds_ev_bdl import _norm_name, add_bdl_ev_all
from warlords_night_board import featured_warlords, rank_priced_slate


class _Response:
    def raise_for_status(self):
        pass

    def json(self):
        return {"forwards": [
            {"id": 8477493, "firstName": {"default": "Aleksander"},
             "lastName": {"default": "Barkov"}, "positionCode": "C"},
        ], "defensemen": []}


class _Session:
    def get(self, url, timeout):
        assert url.endswith("/roster/FLA/current")
        return _Response()


class RosterWatchlistTests(unittest.TestCase):
    def test_corrupt_umlaut_matches_moneypuck_spelling(self):
        self.assertEqual(_norm_name("Tim St\ufffdtzle"), _norm_name("Tim Sttzle"))

    def test_missing_bdl_book_does_not_erase_existing_source(self):
        quotes = pd.DataFrame([{"Goal_Line": 0.5, "Goal_Odds_Over": 190,
                                "Goal_Book": None, "BDL_Goal_Book": None,
                                "ATG_Line": 0.5, "ATG_Odds_Over": 190,
                                "ATG_Book": "fanduel", "SOG_Line": 2.5,
                                "SOG_Odds_Over": 104, "SOG_Book": "fanatics",
                                "BDL_SOG_Book": None}])
        result = add_bdl_ev_all(quotes)
        self.assertEqual(result.iloc[0]["Goal_Book"], "fanduel")
        self.assertEqual(result.iloc[0]["SOG_Book"], "fanatics")

    def test_existing_current_season_player_keeps_model_row_when_name_is_corrupt(self):
        modeled = pd.DataFrame([{"Date": "2026-10-13", "Team": "OTT", "Player": "Tim Sttzle",
                                 "Player_ID": 8482116, "Game": "STL@OTT",
                                 "Model_Stats_Season": "2026-2027", "Conf_Points": 88}])
        rosters = {"OTT": {"timstutzle": {"name": "Tim St\ufffdtzle", "id": 8482116,
                                            "position": "C"}}}
        props = [{"player_id": 500, "vendor": "fanduel", "prop_type": "shots_on_goal",
                  "line_value": "2.5", "market": {"type": "over_under", "over_odds": 104}}]
        games = [{"id": 1, "home_team": {"tricode": "OTT"},
                  "away_team": {"tricode": "STL"}}]
        with patch("odds_ev_bdl.fetch_bdl_games_for_date", return_value=games), \
             patch("odds_ev_bdl.fetch_bdl_props_for_game", return_value=props), \
             patch("odds_ev_bdl.fetch_bdl_players_map", return_value={500: {"full_name": "Tim St\ufffdtzle"}}):
            result = add_roster_watchlist(modeled, None, {"OTT", "STL"}, {"OTT": "STL@OTT"},
                                          date(2026, 10, 13), "test-key", rosters=rosters)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["Model_Stats_Season"], "2026-2027")
        self.assertFalse(result.iloc[0]["Roster_Watch"])
        self.assertEqual(result.iloc[0]["SOG_Odds_Over"], 104)

    def test_active_returner_appears_as_odds_only_and_old_player_is_removed(self):
        old = pd.DataFrame([{"Date": "2026-09-29", "Team": "FLA", "Player": "Departed Player",
                             "Player_ID": 1234, "Game": "FLA@CAR"}])
        props = [
            {"player_id": 268, "vendor": "draftkings", "prop_type": "assists", "line_value": "0.5",
             "market": {"type": "over_under", "over_odds": 105}},
            {"player_id": 268, "vendor": "draftkings", "prop_type": "points", "line_value": "1",
             "market": {"type": "milestone", "odds": -160}},
            {"player_id": 268, "vendor": "fanduel", "prop_type": "power_play_points", "line_value": "1",
             "market": {"type": "milestone", "odds": 250}},
        ]
        with patch("odds_ev_bdl.fetch_bdl_games_for_date", return_value=[
            {"id": 1, "home_team": {"tricode": "FLA"}, "away_team": {"tricode": "CAR"}}]), \
             patch("odds_ev_bdl.fetch_bdl_props_for_game", return_value=props), \
             patch("odds_ev_bdl.fetch_bdl_players_map", return_value={268: {"full_name": "Aleksander Barkov"}}):
            result = add_roster_watchlist(old, _Session(), {"FLA", "CAR"}, {"FLA": "FLA@CAR"},
                                          date(2026, 9, 29), "test-key")
        self.assertEqual(result["Player"].tolist(), ["Aleksander Barkov"])
        row = result.iloc[0]
        self.assertEqual(row["Roster_Status"], "Active roster")
        self.assertTrue(row["Roster_Watch"])
        self.assertEqual(row["Opp"], "CAR")
        self.assertEqual(row["Assists_Odds_Over"], 105)
        self.assertEqual(row["Points_Odds_Over"], -160)
        self.assertEqual(row["BDL_PPP_Odds"], 250)
        self.assertEqual(row["Model_Stats_Season"], "Unavailable")
        board = rank_priced_slate(result)
        self.assertEqual(len(board["Support"]), 1)
        self.assertEqual(len(board["Tank"]), 1)
        self.assertFalse(any(featured_warlords(board).values()))

    def test_out_returner_is_not_added(self):
        old = pd.DataFrame([{"Date": "2026-09-29", "Team": "FLA", "Player": "Departed Player",
                             "Player_ID": 1234, "Game": "FLA@CAR"}])
        props = [{"player_id": 268, "vendor": "draftkings", "prop_type": "assists",
                  "line_value": "0.5", "market": {"type": "over_under", "over_odds": 105}}]
        reports = pd.DataFrame([{"Team": "FLA", "Player": "Aleksander Barkov", "Status": "Out"}])
        with patch("odds_ev_bdl.fetch_bdl_games_for_date", return_value=[
            {"id": 1, "home_team": {"tricode": "FLA"}, "away_team": {"tricode": "CAR"}}]), \
             patch("odds_ev_bdl.fetch_bdl_props_for_game", return_value=props), \
             patch("odds_ev_bdl.fetch_bdl_players_map", return_value={268: {"full_name": "Aleksander Barkov"}}):
            result = add_roster_watchlist(old, _Session(), {"FLA", "CAR"}, {"FLA": "FLA@CAR"},
                                          date(2026, 9, 29), "test-key", injury_reports=reports)
        self.assertTrue(result.empty)
