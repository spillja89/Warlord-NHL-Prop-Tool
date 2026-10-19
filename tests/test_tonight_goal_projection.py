import unittest

import pandas as pd

from tonight_goal_projection import add_tonight_goal_projection, add_tonight_market_projections
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
        self.assertGreater(result["Tonight_Assists_over_pct"], 0)
        self.assertGreater(result["Tonight_Points_over_pct"], 0)
        self.assertGreater(result["Tonight_SOG_over_pct"], 0)
        self.assertGreater(result["Tonight_PPP_1plus_pct"], 0)

    def test_ppp_requires_usage_sample(self):
        row = {**self.base(), "Model_Player_GP": 2, "PP_Points60": 4,
               "PP_ActualPoints60": 3}
        result = add_tonight_market_projections(pd.DataFrame([row])).iloc[0]
        self.assertTrue(pd.isna(result["Tonight_PPP_1plus_pct"]))
        self.assertIn("Unavailable", result["Tonight_PPP_Status"])

    def test_assist_and_shot_cards_show_market_specific_projection(self):
        row = {**self.base(), "Player": "Example", "Team": "COL", "Game": "PIT@COL",
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


if __name__ == "__main__":
    unittest.main()

