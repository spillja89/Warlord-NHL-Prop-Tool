import unittest

import pandas as pd

from power_play_quotes import priced_ppp_quotes


class PowerPlayQuotesTests(unittest.TestCase):
    def test_main_and_alternate_lines_show_real_prices_once(self):
        tracker = pd.DataFrame([{
            "Game": "NYR@BOS", "Player": "PP Skater", "Team": "BOS", "Opp": "NYR",
            "BDL_PPP_Line": 0.5, "BDL_PPP_Odds": 145,
            "BDL_PPP_Book": "Book A", "BDL_PPP_Line_1": 0.5,
            "BDL_PPP_Odds_1": 150, "BDL_PPP_Book_1": "Book B",
            "BDL_PPP_Line_2": 1.5, "BDL_PPP_Odds_2": 650,
            "BDL_PPP_Book_2": "Book C", "PP_TOI_per_game": 3.4,
        }, {
            "Game": "NYR@BOS", "Player": "Unpriced Skater", "Team": "NYR",
            "BDL_PPP_Line": 0.5, "BDL_PPP_Odds": None,
        }])
        quotes = priced_ppp_quotes(tracker)
        self.assertEqual(len(quotes), 2)
        self.assertEqual(quotes["PPP line"].tolist(), [0.5, 1.5])
        self.assertEqual(quotes["Over odds"].tolist(), [150, 650])
        self.assertEqual(quotes["Book"].tolist(), ["Book B", "Book C"])

    def test_empty_tracker_has_no_fabricated_quotes(self):
        self.assertTrue(priced_ppp_quotes(pd.DataFrame()).empty)


if __name__ == "__main__":
    unittest.main()
