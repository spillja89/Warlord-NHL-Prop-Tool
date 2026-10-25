import unittest

import pandas as pd

from source_reconcile import compare_slate_sources


class SourceReconcileTests(unittest.TestCase):
    def test_independent_game_count_and_pp_time_are_labeled_as_source_checks(self):
        tracker = pd.DataFrame([{
            "Player_ID": 123, "Player": "Example Skater", "Team": "STL",
            "Model_Player_GP": 3, "PP_TOI_per_game": 2.1, "PP_Role": 2,
        }])
        live = [{"_id": 123, "GP": 7, "MP GP": 4, "PP TOI/GP": 3.2}]
        findings = compare_slate_sources(tracker, live)
        self.assertEqual({item["Source check"] for item in findings},
                         {"MoneyPuck games trail NHL", "PP ice time changed"})
        self.assertTrue(all("Example Skater" == item["Player"] for item in findings))
        self.assertIn("estimated PP1", next(item for item in findings if item["Source check"] == "PP ice time changed")["Saved tracker"])

    def test_small_update_is_not_flagged(self):
        tracker = pd.DataFrame([{"Player_ID": 123, "Player": "Skater", "Team": "STL",
                                 "Model_Player_GP": 5, "PP_TOI_per_game": 3.0}])
        live = [{"_id": 123, "GP": 6, "MP GP": 5, "PP TOI/GP": 3.2}]
        self.assertEqual(compare_slate_sources(tracker, live), [])


if __name__ == "__main__":
    unittest.main()
