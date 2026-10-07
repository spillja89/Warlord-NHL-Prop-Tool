import unittest

import pandas as pd

from power_play_quotes import priced_ppp_quotes, summarize_pp_usage


class PowerPlayQuotesTests(unittest.TestCase):
    def test_current_pp_usage_uses_only_games_before_the_slate(self):
        rows = [
            {"playerId": 8481617, "gameDate": "2026-09-29", "ppTimeOnIce": 290},
            {"playerId": 8481617, "gameDate": "2026-10-01", "ppTimeOnIce": 242},
            {"playerId": 8481617, "gameDate": "2026-10-03", "ppTimeOnIce": 256},
            {"playerId": 8481617, "gameDate": "2026-10-07", "ppTimeOnIce": 600},
        ]
        self.assertEqual(summarize_pp_usage(rows, "2026-10-07")[8481617], (4.4, 3))
        self.assertEqual(summarize_pp_usage(rows, "2026-10-01")[8481617], (4.8, 1))

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
        half_quotes = priced_ppp_quotes(tracker, line_filter=0.5)
        self.assertEqual(half_quotes["PPP line"].tolist(), [0.5])
        self.assertEqual(half_quotes["Over odds"].tolist(), [150])
        self.assertEqual(half_quotes["Book break-even %"].tolist(), [40.0])
        self.assertEqual(half_quotes["PP unit"].tolist(), ["Unit unknown"])

    def test_history_and_matchup_are_labeled_without_inventing_an_edge(self):
        tracker = pd.DataFrame([{
            "Player": "New Club", "Team": "BOS", "Game": "NYR@BOS",
            "BDL_PPP_Line": 0.5, "BDL_PPP_Odds": -125,
            "BDL_PPP_Book": "Book A", "Team_Changed": True,
            "PP_Role": None, "PP_TOI_per_game": 3.2,
            "Opp_PK_xGA60": 7.4, "Team_PP_xGF60": 8.1,
            "PP_Matchup": 63, "Model_Stats_Season": "2025-2026",
        }, {
            "Player": "Odds Only", "Team": "NYR", "Game": "NYR@BOS",
            "BDL_PPP_Line": 0.5, "BDL_PPP_Odds": 200,
            "BDL_PPP_Book": "Book B", "Roster_Watch": True,
            "Model_Stats_Season": "Unavailable", "PP_Matchup": 50,
        }])
        quotes = priced_ppp_quotes(tracker, line_filter=0.5).set_index("Player")
        self.assertEqual(quotes.loc["New Club", "Book break-even %"], 55.6)
        self.assertEqual(quotes.loc["New Club", "PP unit"], "New team · verify unit")
        self.assertEqual(quotes.loc["New Club", "PP matchup /100"], 63)
        self.assertEqual(quotes.loc["Odds Only", "Context"], "Odds only · history missing")
        self.assertTrue(pd.isna(quotes.loc["Odds Only", "PP matchup /100"]))

    def test_empty_tracker_has_no_fabricated_quotes(self):
        self.assertTrue(priced_ppp_quotes(pd.DataFrame()).empty)


if __name__ == "__main__":
    unittest.main()
