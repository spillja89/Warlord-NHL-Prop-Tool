import json
import unittest

import pandas as pd

from opportunity_views import add_opportunity_views
from price_validation import price_records, summarize_prices


class OpportunityAndPriceViewsTests(unittest.TestCase):
    def test_opportunity_views_keep_missing_inputs_unknown(self):
        rows = pd.DataFrame([{
            "EV5_TOI_per_game": 15, "i5v5_SOG60": 8,
            "PP_TOI_per_game": 4, "PP_SOG60": 6,
            "PP_OnIce_xGF60": 9, "PP_iXG60": 1.5,
            "i5v5_xG60": 1.2, "i5v5_shotAssists60": 5,
            "PP_WeightedAssists60": 3,
            "Opp_SOG_Against_L10": 36, "Opp_SOG_Used_L10": 5,
            "PP_Score": 60, "Opp_PK_Weak": 80,
        }, {
            "PP_TOI_per_game": 3, "PP_OnIce_xGF60": 12,
        }])
        result = add_opportunity_views(rows)
        self.assertEqual(result.loc[0, "SOG_Opportunity_Base"], 2.4)
        self.assertEqual(result.loc[0, "SOG_Opportunity_Matchup"], 2.52)
        self.assertEqual(result.loc[0, "PP_OnIce_xG_per_game"], 0.6)
        self.assertEqual(result.loc[0, "PP_Matchup_Equal"], 70)
        self.assertEqual(result.loc[0, "Assist_5v5_Setups_per_game"], 1.25)
        self.assertEqual(result.loc[1, "SOG_Opportunity_Coverage"], "unavailable")
        self.assertTrue(pd.isna(result.loc[1, "SOG_Opportunity_Matchup"]))

    def test_price_check_uses_frozen_price_and_named_move_only(self):
        tags = json.dumps([{"market": "SOG", "name": "Shot Barrage", "kit_version": "v1"}])
        graded = pd.DataFrame([{
            "Date": "2026-10-01", "Game": "A@B", "Player": "Shooter",
            "Grade_Status": "FINAL", "Fired_Moves": tags,
            "SOG_Line": 2.5, "SOG_Odds_Over": -115, "SOG_Book": "Book A",
            "Outcome_SOG": "W", "Odds_Checked_UTC": "2026-10-01T12:00:00Z",
        }, {
            "Date": "2026-10-02", "Game": "A@B", "Player": "Shooter",
            "Grade_Status": "FINAL", "Fired_Moves": tags,
            "SOG_Line": 2.5, "SOG_Odds_Over": 170, "SOG_Book": "Book B",
            "Outcome_SOG": "L", "Odds_Checked_UTC": "2026-10-02T12:00:00Z",
        }, {
            "Date": "2026-10-03", "Grade_Status": "PENDING_GAME",
            "Fired_Moves": tags, "SOG_Line": 2.5,
            "SOG_Odds_Over": 300, "Outcome_SOG": "W",
        }])
        records = price_records(graded)
        self.assertEqual(len(records), 2)
        self.assertEqual(records["Profit_units"].tolist(), [round(100 / 115, 4), -1.0])
        self.assertEqual(records["Captured_UTC"].iloc[0], "2026-10-01T12:00:00Z")
        summary = summarize_prices(records)
        self.assertEqual(summary["Picks"].sum(), 2)
        self.assertTrue(summary["Evidence"].eq("early sample").all())


if __name__ == "__main__":
    unittest.main()
