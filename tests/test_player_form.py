import unittest

from player_form import compact_regular_log, summarize_form


class PlayerFormTests(unittest.TestCase):
    def test_regular_season_only_and_line_specific(self):
        playoff = {"seasonId": 20252026, "gameTypeId": 3, "gameLog": [
            {"gameId": 2025030001, "gameDate": "2026-04-20", "goals": 3,
             "assists": 2, "shots": 8}]}
        self.assertIsNone(compact_regular_log(playoff))

        regular = {"seasonId": 20262027, "gameTypeId": 2, "gameLog": [
            {"gameId": 2026020001, "gameDate": "2026-10-02", "goals": 1,
             "assists": 1, "shots": 4, "homeRoadFlag": "H", "toi": "20:30"},
            {"gameId": 2026020002, "gameDate": "2026-10-04", "goals": 0,
             "assists": 1, "shots": 2, "homeRoadFlag": "R", "toi": "18:00"},
            {"gameId": 2026030003, "gameDate": "2026-10-05", "goals": 3,
             "assists": 1, "shots": 6},
        ]}
        compact = compact_regular_log(regular, "2026-10-05")
        self.assertEqual(len(compact["games"]), 2)
        form = summarize_form(compact, "Points", 1.5)
        self.assertEqual(form["l10"], (1, 2))
        self.assertEqual(form["l5"], (1, 2))
        self.assertEqual(form["average"], 1.5)
        self.assertEqual(form["splits"], {"Home": (1, 1), "Away": (0, 1)})
        self.assertEqual(summarize_form(compact, "Assists", 0.5)["l10"], (2, 2))
        self.assertEqual(summarize_form(compact, "SOG", 2.5)["l10"], (1, 2))

    def test_power_play_form_uses_credited_points_and_skips_missing_fields(self):
        regular = {"seasonId": 20262027, "gameTypeId": 2, "gameLog": [
            {"gameId": 2026020001, "gameDate": "2026-10-02", "goals": 1,
             "assists": 0, "shots": 3, "powerPlayPoints": 1},
            {"gameId": 2026020002, "gameDate": "2026-10-04", "goals": 0,
             "assists": 1, "shots": 2, "powerPlayPoints": 0},
            {"gameId": 2026020003, "gameDate": "2026-10-05", "goals": 0,
             "assists": 1, "shots": 4},
        ]}
        compact = compact_regular_log(regular, "2026-10-06")
        form = summarize_form(compact, "PPP", 0.5)
        self.assertEqual(form["l5"], (1, 2))
        self.assertEqual(form["season_rate"], (1, 2))
        self.assertEqual(form["recent"], [(0, False, "2026-10-04"),
                                          (1, True, "2026-10-02")])

    def test_new_season_without_games(self):
        compact = compact_regular_log({"seasonId": 20262027, "gameTypeId": 2, "gameLog": []})
        self.assertEqual(summarize_form(compact, "Goal", 0.5),
                         {"season": "2026-27", "empty": True})


if __name__ == "__main__":
    unittest.main()
