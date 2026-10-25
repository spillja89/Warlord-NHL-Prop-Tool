import unittest
from unittest.mock import patch

import pandas as pd

import blade_check


class BladeRoleChangeTests(unittest.TestCase):
    def test_same_day_role_change_is_labeled_estimated(self):
        common = {"Date": "2026-10-25", "Game": "STL@WPG", "Player_ID": 123,
                  "Player": "Example Skater", "Team": "STL", "Lineup_Status": "Lineup pending",
                  "Odds_Checked_UTC": "2026-10-25T15:00:00Z"}
        before = pd.DataFrame([{**common, "PP_Role": 1}])
        after = pd.DataFrame([{**common, "PP_Role": 2}])
        with patch.object(blade_check, "_counts", return_value=({"Carry": [], "Support": [],
                                                                    "Tank": [], "Jungle": []},
                                                                   {"Goals": 0, "Assists": 0,
                                                                    "Points": 0, "Shots": 0,
                                                                    "Power play": 0})), \
             patch.object(blade_check, "_pp_half_rows", return_value=[]):
            audit = blade_check.check_slate(after, previous=before,
                                            now_utc=pd.Timestamp("2026-10-25T15:05:00Z"))
        changes = [item for item in audit["findings"] if item["Check"] == "Estimated PP unit changed"]
        self.assertEqual(len(changes), 1)
        self.assertIn("PP2 → PP1", changes[0]["Evidence"])
        self.assertIn("not tonight's confirmed unit", changes[0]["Evidence"])


if __name__ == "__main__":
    unittest.main()
