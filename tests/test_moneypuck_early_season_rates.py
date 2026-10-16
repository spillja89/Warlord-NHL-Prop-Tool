import unittest

import pandas as pd

from nhl_edge import (compute_lastN_features, matrix_points_v2, normalize_goalies,
                      normalize_skaters_all, normalize_skaters_5v5, normalize_skaters_pp,
                      normalize_teams_pppk, per60, ten_game_pace_gap)
from warlord_moves_2026 import sog_moves


class MoneyPuckEarlySeasonRatesTests(unittest.TestCase):
    def test_historical_volley_rule_keeps_its_proxy_input(self):
        row = {"SOG_Line": 2.5, "Matrix_SOG": "Green", "Conf_SOG": 80,
               "L10_Rate_SOG": 3.0, "Player_SOG_Share_Proxy": 20.0,
               "Player_5v5_SOG_Share": 5.0}
        self.assertIn("Berserker Volley", [m["name"] for m in sog_moves(row)])

    def test_regression_gap_compares_equal_windows(self):
        gap = ten_game_pace_gap(pd.Series([10.0, 10.0]), pd.Series([3.0, 1.0]),
                                pd.Series([5, 2]))
        self.assertEqual(gap.iloc[0], 4.0)
        self.assertTrue(pd.isna(gap.iloc[1]))

    def test_5v5_shot_share_uses_full_team_shot_totals(self):
        skaters = pd.DataFrame([
            {"playerId": 1, "team": "COL", "situation": "5on5", "icetime": 600,
             "I_F_shotsOnGoal": 4, "I_F_shotAttempts": 7, "I_F_points": 1},
            {"playerId": 2, "team": "COL", "situation": "5on5", "icetime": 1200,
             "I_F_shotsOnGoal": 6, "I_F_shotAttempts": 8, "I_F_points": 0},
        ])
        result = normalize_skaters_5v5(skaters).set_index("playerId")
        self.assertEqual(result.loc[1, "Player_5v5_SOG_Share"], 40.0)
        self.assertEqual(result.loc[2, "Player_5v5_SOG_Share"], 60.0)
        self.assertEqual(result.loc[1, "i5v5_iCF60"], 42.0)

    def test_points_matrix_does_not_reference_missing_confidence(self):
        self.assertEqual(matrix_points_v2(96, 55, toi_pct=50), "Yellow")

    def test_recent_windows_report_actual_game_counts(self):
        payload = {"gameLog": [
            {"gameDate": "2026-10-03", "goals": 1, "assists": 0, "shots": 3},
            {"gameDate": "2026-10-02", "goals": 0, "assists": 1, "shots": 2},
            {"gameDate": "2026-10-01", "goals": 0, "assists": 0, "shots": 1},
        ]}
        stats = compute_lastN_features(payload)
        self.assertEqual((stats["L10_Games"], stats["L20_Games"], stats["L40_Games"]), (3, 3, 3))
        self.assertEqual(stats["S10_total"], 6)

    def test_all_skaters_ice_time_is_seconds_even_below_ten_thousand(self):
        skaters = pd.DataFrame([{
            "playerId": 1, "name": "Early Skater", "team": "COL", "situation": "all",
            "games_played": 2, "icetime": 2400,
            "I_F_xGoals": 0.5, "I_F_primaryAssists": 1,
            "I_F_secondaryAssists": 0,
        }])
        player = normalize_skaters_all(skaters).iloc[0]
        self.assertEqual(player["TOI_per_game"], 20.0)
        self.assertEqual(player["iXA_Source"], "A1+A2")
        self.assertFalse(player["iXA_Is_Expected"])

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
            "I_F_primaryAssists": 1, "I_F_secondaryAssists": 0, "I_F_points": 2,
        }])

        player = normalize_skaters_pp(skaters).iloc[0]

        self.assertAlmostEqual(player["PP_OnIce_xGF60"], 6.0)
        self.assertAlmostEqual(player["PP_iXG60"], 1.2)
        self.assertAlmostEqual(player["PP_iXA60"], 3.0)
        self.assertAlmostEqual(player["PP_WeightedAssists60"], 3.0)
        self.assertAlmostEqual(player["PP_ActualPoints60"], 6.0)
        self.assertIn("observed", player["PP_iXA60_Source"])

    def test_pp_assist_proxy_is_unavailable_without_assist_fields(self):
        skaters = pd.DataFrame([{
            "playerId": 1, "team": "COL", "situation": "5on4",
            "games_played": 1, "icetime": 600, "I_F_xGoals": 0.1,
        }])
        player = normalize_skaters_pp(skaters).iloc[0]
        self.assertTrue(pd.isna(player["PP_iXA60"]))
        self.assertTrue(pd.isna(player["PP_ActualPoints60"]))

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
