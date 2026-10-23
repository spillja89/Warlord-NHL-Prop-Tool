import unittest

import pandas as pd

from warlord_moves_2026 import points_moves
from warlords_night_board import featured_warlords, rank_priced_slate, render_warlords


class PointsCardDisplayTests(unittest.TestCase):
    def test_onehalf_card_shows_backbone_and_keeps_other_moves_for_tracking(self):
        row = {
            "Player": "Example", "Team": "COL", "Game": "TBL@COL",
            "Points_Line": 1.5, "Points_Odds_Over": 128,
            "Matrix_Points": "Green", "Conf_Points": 85,
            "Conf_Goal": 87, "Conf_SOG": 84, "Assists_mu": 1.5,
            "Goal_ProofCount": 3, "Goal_Model%": 40,
            "Opp_PK_xGA60": 8,
        }
        self.assertIn("Ragnarok", [move["name"] for move in points_moves(row)])
        card = rank_priced_slate(pd.DataFrame([row]))["Tank"][0]
        self.assertEqual({move["name"] for move in card["moves"]},
                         {"DPS Tank stance", "Backbone"})
        self.assertTrue(card["points_power_tier"])
        featured = featured_warlords({"Tank": [card]})["Tank"]
        self.assertEqual(featured[0]["move"]["name"], "Backbone")
        html = render_warlords({"Tank": featured}, roles=("Tank",))
        self.assertIn("Enhancement: Blade Impale (Power Tier)", html)
        self.assertNotIn("Ragnarok", html)


if __name__ == "__main__":
    unittest.main()
