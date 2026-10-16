import unittest

import pandas as pd

from nhl_edge import normalize_goalies, normalize_skaters_pp, normalize_teams_pppk, per60


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

    def test_pp_on_ice_expected_goals_is_saved_separately(self):
        skaters = pd.DataFrame([{
            "playerId": 1, "team": "COL", "situation": "5on4",
            "games_played": 4, "icetime": 1200,
            "OnIce_F_xGoals": 2.0, "I_F_xGoals": 0.4,
            "I_F_primaryAssists": 1, "I_F_secondaryAssists": 0,
        }])

        player = normalize_skaters_pp(skaters).iloc[0]

        self.assertAlmostEqual(player["PP_OnIce_xGF60"], 6.0)
        self.assertAlmostEqual(player["PP_iXG60"], 1.2)
        self.assertAlmostEqual(player["PP_iXA60"], 3.0)

    def test_team_pp_and_pk_expected_and_actual_rates_remain_distinct(self):
        teams = pd.DataFrame([
            {"team": "COL", "situation": "5on4", "icetime": 3600,
             "xGoalsFor": 2.0, "goalsFor": 1.0,
             "xGoalsAgainst": 0.0, "goalsAgainst": 0.0},
            {"team": "COL", "situation": "4on5", "icetime": 3600,
             "xGoalsFor": 0.0, "goalsFor": 0.0,
             "xGoalsAgainst": 3.0, "goalsAgainst": 2.0},
        ])

        team = normalize_teams_pppk(teams).iloc[0]

        self.assertEqual(team["Team_PP_xGF60"], 2.0)
        self.assertEqual(team["Team_PP_GF60"], 1.0)
        self.assertEqual(team["Team_PK_xGA60"], 3.0)
        self.assertEqual(team["Team_PK_GA60"], 2.0)
