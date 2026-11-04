import math
import unittest

import pandas as pd

from sog_confidence import VERSION, apply_sog_confidence, sog_profile_confidence


class SOGConfidenceTests(unittest.TestCase):
    def test_approved_units_and_weights(self):
        # 55%*80 +15%*50 +30%*75 =74; Elite1-game bonus adds2.
        self.assertEqual(sog_profile_confidence(80, 17.5, 30, "ELITE", 1), 76)

    def test_missing_inputs_stay_missing(self):
        for args in [(None, 17.5, 30, "STAR", 0), (80, float("nan"), 30, "STAR", 0),
                     (80, 17.5, None, "STAR", 0), (80, 17.5, 30, "STAR", None)]:
            self.assertIsNone(sog_profile_confidence(*args))

    def test_invalid_units_are_not_neutral(self):
        for args in [(101, 17.5, 30, "STAR", 0), (80, -1, 30, "STAR", 0),
                     (80, 17.5, 0, "STAR", 0), (80, 17.5, 30, "STAR", -1)]:
            self.assertIsNone(sog_profile_confidence(*args))

    def test_bonus_is_applied_once_and_score_is_bounded(self):
        self.assertEqual(sog_profile_confidence(100, 35, 40, "ELITE", 3), 100)
        self.assertEqual(sog_profile_confidence(80, 17.5, 30, "STAR", 2), 76)
        self.assertEqual(sog_profile_confidence(80, 17.5, 30, "NONE", 2), 74)

    def test_no_injury_adjustment_and_original_is_preserved(self):
        frame = pd.DataFrame([dict(Conf_SOG=88, TOI_Pct=80, Player_5v5_SOG_Share=17.5,
                                  Opp_SOG_Against_L10=30, Talent_Tier="ELITE", Drought_SOG=1,
                                  Injury_DFO_Score=score) for score in (-3, 0, 1)])
        out = apply_sog_confidence(frame)
        self.assertEqual(out.Conf_SOG.tolist(), [76, 76, 76])
        self.assertEqual(out.Conf_SOG_Move.tolist(), [88, 88, 88])
        pd.testing.assert_frame_equal(out, apply_sog_confidence(out))
        self.assertNotIn("Conf_SOG_Move", frame)

    def test_watchlist_and_empty_frame(self):
        out = apply_sog_confidence(pd.DataFrame([dict(Player="Returner", Conf_SOG=math.nan)]))
        self.assertTrue(out.Conf_SOG.isna().all())
        self.assertEqual(out.Conf_SOG_Version.iloc[0], VERSION)
        self.assertTrue(apply_sog_confidence(pd.DataFrame()).empty)


if __name__ == "__main__":
    unittest.main()
