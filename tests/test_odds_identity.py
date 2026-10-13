import unittest

import pandas as pd

from odds_identity import withhold_ambiguous_player_odds


class OddsIdentityTests(unittest.TestCase):
    def test_same_name_teammates_do_not_inherit_each_others_quotes(self):
        frame = pd.DataFrame([
            {"Team": "VAN", "Player": "Elias Pettersson", "Player_ID": 8480012,
             "Pos": "C", "Points_Line": 0.5, "Points_Odds_Over": -110,
             "BDL_PPP_Line": 0.5, "BDL_PPP_Odds": 290,
             "Points_EV%": 5.0, "Points_Available_Books": '["FanDuel"]',
             "Plays_EV_Points": True},
            {"Team": "VAN", "Player": "Elias Pettersson", "Player_ID": 8483678,
             "Pos": "D", "Points_Line": 0.5, "Points_Odds_Over": -110,
             "BDL_PPP_Line": 0.5, "BDL_PPP_Odds": 290,
             "Points_EV%": 5.0, "Points_Available_Books": '["FanDuel"]',
             "Plays_EV_Points": True},
            {"Team": "VAN", "Player": "Quinn Hughes", "Player_ID": 8480800,
             "Pos": "D", "Points_Line": 0.5, "Points_Odds_Over": -150,
             "BDL_PPP_Line": 0.5, "BDL_PPP_Odds": 140,
             "Points_EV%": 2.0, "Points_Available_Books": '["BetMGM"]',
             "Plays_EV_Points": True},
        ])
        result = withhold_ambiguous_player_odds(frame)
        self.assertEqual(result["Odds_Name_Ambiguous"].tolist(), [True, True, False])
        for column in ("Points_Line", "Points_Odds_Over", "BDL_PPP_Line",
                       "BDL_PPP_Odds", "Points_EV%"):
            self.assertTrue(result.loc[:1, column].isna().all(), column)
            self.assertEqual(result.loc[2, column], frame.loc[2, column])
        self.assertEqual(result.loc[:1, "Points_Available_Books"].tolist(), ["", ""])
        self.assertEqual(result["Plays_EV_Points"].tolist(), [False, False, True])


if __name__ == "__main__":
    unittest.main()
