import unittest

import pandas as pd

from grade_jev_shadow import grade_shadow, summarize_shadow


class JevShadowGradeTests(unittest.TestCase):
    def test_uses_frozen_line_and_final_actual_only(self):
        shadow = pd.DataFrame([
            {"Date": "2026-10-25", "Player_ID": 1, "Market": "Goals", "Prop line": 0.5,
             "Saved odds": 180, "Opportunity": "strong", "Matchup": "weak", "Evidence": "complete"},
            {"Date": "2026-10-25", "Player_ID": 2, "Market": "Power play", "Prop line": 0.5,
             "Saved odds": 220, "Opportunity": "weak", "Matchup": "strong", "Evidence": "thin"},
        ])
        graded = pd.DataFrame([
            {"Date": "2026-10-25", "Player_ID": 1, "Grade_Status": "FINAL",
             "Actual_Goal": 1, "Actual_Assists": 0, "Actual_Points": 1,
             "Actual_SOG": 2, "Actual_PPP": 0},
            {"Date": "2026-10-25", "Player_ID": 2, "Grade_Status": "PENDING_GAME",
             "Actual_Goal": None, "Actual_Assists": None, "Actual_Points": None,
             "Actual_SOG": None, "Actual_PPP": None},
        ])
        cards = grade_shadow(shadow, graded)
        self.assertEqual(cards["Outcome"].tolist(), ["W", ""])
        summary = summarize_shadow(cards)
        goal = summary[(summary["Market"] == "Goals") & (summary["Dimension"] == "Opportunity")]
        self.assertEqual(goal.iloc[0]["Settled hit %"], 100.0)
        pp = summary[(summary["Market"] == "Power play") & (summary["Dimension"] == "Opportunity")]
        self.assertEqual(pp.iloc[0]["Unresolved"], 1)


if __name__ == "__main__":
    unittest.main()
