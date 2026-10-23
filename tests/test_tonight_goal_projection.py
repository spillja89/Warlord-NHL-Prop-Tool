import unittest

import pandas as pd

from tonight_goal_projection import (add_tonight_goal_projection,
                                     add_tonight_market_projections,
                                     add_player_rate_sources)
from warlords_night_board import rank_priced_slate, render_warlords


class TonightGoalProjectionTests(unittest.TestCase):
    def base(self):
        return {
            "Player": "Example", "Model_Player_GP": 10,
            "EV5_TOI_per_game": 15, "i5v5_xG60": 1.2,
            "PP_TOI_per_game": 3, "PP_iXG60": 2.0,
            "Goal_mu": 0.4,
            "Opp_Team_GP": 10,
            "opp_5v5_xGA60": 2.5, "League_5v5_xGA60": 2.5,
            "Opp_PK_xGA60": 7.0, "League_PK_xGA60": 7.0,
            "Opp_Goalie_Status": "Unconfirmed",
            "Opp_Goalie_GA": 30, "Opp_Goalie_xGA": 20,
            "Goal_Line": 0.5, "Goal_Odds_Over": 200,
        }

    def test_components_and_goal_probability_are_separate_from_legacy_mu(self):
        row = self.base()
        result = add_tonight_goal_projection(pd.DataFrame([row])).iloc[0]
        self.assertAlmostEqual(result["Tonight_Goal_EV_xG"], 0.3)
        self.assertAlmostEqual(result["Tonight_Goal_PP_xG"], 0.1)
        self.assertAlmostEqual(result["Tonight_Goal_xG"], 0.4)
        self.assertEqual(result["Goal_mu"], row["Goal_mu"])
        self.assertGreater(result["Tonight_Goal_1plus_pct"], 0)
        self.assertEqual(result["Tonight_Goal_Goalie_factor"], 1)

    def test_weak_defense_pk_and_confirmed_goalie_raise_projection(self):
        neutral = add_tonight_goal_projection(pd.DataFrame([self.base()])).iloc[0]
        row = {**self.base(), "opp_5v5_xGA60": 3.5, "Opp_PK_xGA60": 10.0,
               "Opp_Goalie_Status": "Confirmed"}
        boosted = add_tonight_goal_projection(pd.DataFrame([row])).iloc[0]
        self.assertGreater(boosted["Tonight_Goal_xG"], neutral["Tonight_Goal_xG"])
        self.assertGreater(boosted["Tonight_Goal_1plus_pct"], neutral["Tonight_Goal_1plus_pct"])
        self.assertGreater(boosted["Tonight_Goal_Goalie_factor"], 1)

    def test_missing_player_opportunity_is_unavailable(self):
        row = {**self.base(), "i5v5_xG60": None}
        result = add_tonight_goal_projection(pd.DataFrame([row])).iloc[0]
        self.assertTrue(pd.isna(result["Tonight_Goal_xG"]))
        self.assertIn("Unavailable", result["Tonight_Goal_Status"])

    def test_missing_team_rates_do_not_use_legacy_fifty_sentinel(self):
        row = {**self.base(), "opp_5v5_xGA60": 50.0,
               "League_5v5_xGA60": None, "Opp_PK_xGA60": None}
        result = add_tonight_goal_projection(pd.DataFrame([row])).iloc[0]
        self.assertEqual(result["Tonight_Goal_EV_factor"], 1)
        self.assertEqual(result["Tonight_Goal_PK_factor"], 1)

    def test_goal_card_shows_separate_research_projection(self):
        row = {**self.base(), "Team": "COL", "Game": "PIT@COL",
               "Matrix_Goal": "Green", "Conf_Goal": 90,
               "Conf_Points": 90}
        calculated = add_tonight_goal_projection(pd.DataFrame([row]))
        card = rank_priced_slate(calculated)["Carry"][0]
        self.assertEqual(card["tonight_goal_xg"], calculated.iloc[0]["Tonight_Goal_xG"])
        html = render_warlords({"Carry": [card]}, roles=("Carry",))
        self.assertIn("TONIGHT RESEARCH PROJECTION", html)
        self.assertIn("being graded, not a tested move", html)

    def test_other_markets_keep_existing_means_and_use_their_own_lines(self):
        row = {**self.base(), "Team_GP": 10,
               "Current_Rate_GP": 5, "Current_Rate_Assists": 8,
               "Current_Rate_Points": 10, "Current_Rate_SOG": 15,
               "Current_Rate_PPP": 3, "Prior_Rate_GP": 80,
               "Prior_Rate_Assists": 64, "Prior_Rate_Points": 96,
               "Prior_Rate_SOG": 240, "Prior_Rate_PPP": 24,
               "Assists_mu": .8, "Assists_Line": .5, "Assists_Odds_Over": -120,
               "Points_mu": 1.2, "Points_Line": 1.5, "Points_Odds_Over": 180,
               "SOG_mu": 3.0, "SOG_Line": 2.5, "SOG_Odds_Over": -110,
               "SOG_Opp_Volume_Factor": 1.1,
               "PP_Points60": 4.0, "PP_ActualPoints60": 3.0,
               "Team_PP_xGF60": 8.0, "League_PP_xGF60": 8.0}
        result = add_tonight_market_projections(pd.DataFrame([row])).iloc[0]
        self.assertEqual(result["Assists_mu"], .8)
        self.assertEqual(result["Points_mu"], 1.2)
        self.assertEqual(result["SOG_mu"], 3.0)
        self.assertAlmostEqual(result["Tonight_SOG_mu"], 3.3)
        self.assertLess(result["Tonight_Assists_mu"], 1.2)
        self.assertGreater(result["Tonight_Assists_over_pct"], 0)
        self.assertGreater(result["Tonight_Points_over_pct"], 0)
        self.assertGreater(result["Tonight_SOG_over_pct"], 0)
        self.assertGreater(result["Tonight_PPP_1plus_pct"], 0)

    def test_ppp_requires_usage_sample(self):
        row = {**self.base(), "Model_Player_GP": 2, "PP_Points60": 4,
               "PP_ActualPoints60": 3, "Current_Rate_GP": 2,
               "Current_Rate_PPP": 1, "Prior_Rate_GP": 80, "Prior_Rate_PPP": 30}
        result = add_tonight_market_projections(pd.DataFrame([row])).iloc[0]
        self.assertTrue(pd.isna(result["Tonight_PPP_1plus_pct"]))
        self.assertIn("Unavailable", result["Tonight_PPP_Status"])

    def test_assist_and_shot_cards_show_market_specific_projection(self):
        row = {**self.base(), "Player": "Example", "Team": "COL", "Game": "PIT@COL",
               "Current_Rate_GP": 5, "Current_Rate_Assists": 4,
               "Current_Rate_SOG": 15, "Prior_Rate_GP": 80,
               "Prior_Rate_Assists": 64, "Prior_Rate_SOG": 240,
               "Assists_mu": .8, "Assists_Line": .5, "Assists_Odds_Over": -120,
               "SOG_mu": 3.0, "SOG_Line": 2.5, "SOG_Odds_Over": -110,
               "SOG_Opp_Volume_Factor": 1.1}
        frame = add_tonight_market_projections(pd.DataFrame([row]))
        cards = rank_priced_slate(frame)
        html = render_warlords({"Support": cards["Support"], "Jungle": cards["Jungle"]},
                               roles=("Support", "Jungle"))
        self.assertIn("expected assists", html)
        self.assertIn("expected shots", html)
        self.assertIn("over 2.5", html)

    def test_extreme_early_assist_mean_is_shrunk_by_actual_prior(self):
        row = {**self.base(), "Assists_mu": 3.46,
               "Current_Rate_GP": 5, "Current_Rate_Assists": 10,
               "Prior_Rate_GP": 80, "Prior_Rate_Assists": 88}
        result = add_tonight_market_projections(pd.DataFrame([row])).iloc[0]
        self.assertLess(result["Tonight_Assists_mu"], 1.5)
        self.assertEqual(result["Assists_mu"], 3.46)

    def test_missing_prior_and_few_current_games_withholds_research_rate(self):
        row = {**self.base(), "Assists_mu": 3.46,
               "Current_Rate_GP": 5, "Current_Rate_Assists": 10,
               "Assists_Line": .5}
        result = add_tonight_market_projections(pd.DataFrame([row])).iloc[0]
        self.assertTrue(pd.isna(result["Tonight_Assists_mu"]))

    def test_rate_source_uses_id_and_regular_season_situation(self):
        row = {**self.base(), "Player_ID": 123}
        current = pd.DataFrame([
            {"playerId": 123, "situation": "all", "games_played": 5,
             "I_F_assists": 3, "I_F_points": 5, "I_F_shotsOnGoal": 16},
            {"playerId": 123, "situation": "5on4", "I_F_points": 2},
        ])
        prior = pd.DataFrame([
            {"playerId": 123, "situation": "all", "games_played": 80,
             "I_F_assists": 64, "I_F_points": 96, "I_F_shotsOnGoal": 240},
            {"playerId": 123, "situation": "5on4", "I_F_points": 24},
        ])
        result = add_player_rate_sources(pd.DataFrame([row]), current, prior).iloc[0]
        self.assertEqual(result["Current_Rate_PPP"], 2)
        self.assertEqual(result["Prior_Rate_Assists"], 64)

    def test_rate_source_uses_primary_and_secondary_assists_when_total_is_absent(self):
        row = {**self.base(), "Player_ID": 123, "Assists_Line": .5}
        current = pd.DataFrame([{
            "playerId": 123, "situation": "all", "games_played": 8,
            "I_F_primaryAssists": 4, "I_F_secondaryAssists": 3,
        }])
        prior = pd.DataFrame([{
            "playerId": 123, "situation": "all", "games_played": 80,
            "I_F_primaryAssists": 34, "I_F_secondaryAssists": 30,
        }])
        sourced = add_player_rate_sources(pd.DataFrame([row]), current, prior)
        self.assertEqual(sourced.iloc[0]["Current_Rate_Assists"], 7)
        self.assertEqual(sourced.iloc[0]["Prior_Rate_Assists"], 64)
        projected = add_tonight_market_projections(sourced).iloc[0]
        self.assertGreater(projected["Tonight_Assists_mu"], 0)
        self.assertGreater(projected["Tonight_Assists_over_pct"], 0)


if __name__ == "__main__":
    unittest.main()

