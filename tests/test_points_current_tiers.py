import unittest

import pandas as pd

from warlords_night_board import featured_warlords, rank_priced_slate, render_warlords


class PointsCurrentTierTests(unittest.TestCase):
    def _card(self, player, assists, pp_minutes, *, line=0.5, conf=78):
        row = {
            "Player": player, "Team": "COL", "Game": "COL@DAL",
            "Points_Line": line, "Points_Odds_Over": -120,
            "Matrix_Points": "Green", "Conf_Points": conf,
            "L10_A": assists, "PP_TOI_per_game": pp_minutes,
        }
        return rank_priced_slate(pd.DataFrame([row]))["Tank"][0]

    def test_special_has_priority_when_both_thresholds_fire(self):
        card = self._card("Special", 8, 3.5)
        self.assertEqual(card["points_current_tier"]["kind"], "SPECIAL")
        shown = featured_warlords({"Tank": [card]})["Tank"]
        self.assertEqual(len(shown), 1)
        html = render_warlords({"Tank": shown}, roles=("Tank",))
        self.assertIn("Points Special", html)
        self.assertIn("27/33", html)
        self.assertIn("Last 10 assists", html)

    def test_heavy_and_base_keep_players_without_old_moves(self):
        heavy = self._card("Heavy", 7, 3.5)
        base = self._card("Base", 7, 3.49)
        self.assertEqual(heavy["points_current_tier"]["kind"], "HEAVY")
        self.assertEqual(base["points_current_tier"]["kind"], "BASE")
        shown = featured_warlords({"Tank": [base, heavy]})["Tank"]
        self.assertEqual([card["player"] for card in shown], ["Heavy", "Base"])
        self.assertEqual([card["move"]["picks"] for card in shown], [52, 76])

    def test_only_green_half_point_conf_78_enters_current_tiers(self):
        self.assertIsNone(self._card("Low", 8, 4, conf=77)["points_current_tier"])
        self.assertIsNone(self._card("OnePointFive", 8, 4, line=1.5)["points_current_tier"])


if __name__ == "__main__":
    unittest.main()
