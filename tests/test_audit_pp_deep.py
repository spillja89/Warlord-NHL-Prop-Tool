import tempfile
import unittest
from pathlib import Path

import pandas as pd

from audit_pp_deep import _prepare


class PowerPlayDeepAuditTests(unittest.TestCase):
    def test_only_final_pregame_priced_props_enter_audit(self):
        base = {
            "Date": "2026-10-15", "Game": "COL@BOS", "Team": "COL",
            "Opp": "BOS", "BDL_PPP_Line": 0.5, "BDL_PPP_Odds": 150,
            "Match_Status_PPP": "OK", "StartTimeUTC": "2026-10-16T00:00:00Z",
        }
        rows = [
            {**base, "Player_ID": 1, "Grade_Status": "FINAL", "Outcome_PPP": "W",
             "Odds_Checked_UTC": "2026-10-15T19:00:00Z"},
            {**base, "Player_ID": 2, "Grade_Status": "FINAL", "Outcome_PPP": "L",
             "Odds_Checked_UTC": "2026-10-16T01:00:00Z"},
            {**base, "Player_ID": 3, "Grade_Status": "NO_PLAYER", "Outcome_PPP": "",
             "Odds_Checked_UTC": "2026-10-15T19:00:00Z"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graded.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            prepared, quality = _prepare([path])

        self.assertEqual(prepared["Player_ID"].tolist(), [1])
        self.assertEqual(quality["eligible_unique_props"], 1)
        self.assertEqual(quality["pregame_ungraded_statuses"], {"NO_PLAYER": 1})


if __name__ == "__main__":
    unittest.main()
