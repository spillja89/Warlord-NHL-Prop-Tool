import unittest

from season_sources import power_play_team_ranks, slate_team_matchup_rows


class SlateTeamMatchupsTests(unittest.TestCase):
    def test_each_side_uses_its_own_pp_and_opponents_pk(self):
        teams = [
            {"Team": "OTT", "GP": 6, "PP xGF/60": 8.1, "PP GF/60": 7.2,
             "PK xGA/60": 5.0, "PK GA/60": 6.0, "GA/GP": 2.5,
             "5v5 xGA/60": 2.1, "SOG against/GP": 25.0},
            {"Team": "NJD", "GP": 5, "PP xGF/60": 7.4, "PP GF/60": 8.0,
             "PK xGA/60": 9.2, "PK GA/60": 10.1, "GA/GP": 3.2,
             "5v5 xGA/60": 2.8, "SOG against/GP": 29.0},
        ]
        rows = slate_team_matchup_rows(teams, ["OTT@NJD", "OTT@NJD"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["Team PP xGF/60"], 8.1)
        self.assertEqual(rows[0]["Opp PK xGA/60"], 9.2)
        self.assertEqual(rows[0]["Opp PK GA/60"], 10.1)
        self.assertEqual(rows[1]["Team PP xGF/60"], 7.4)
        self.assertEqual(rows[1]["Opp PK xGA/60"], 5.0)

    def test_missing_team_stats_remain_blank(self):
        rows = slate_team_matchup_rows([{"Team": "OTT", "GP": 6}], ["OTT@NJD"])
        self.assertIsNone(rows[0]["Opp PK xGA/60"])
        self.assertIsNone(rows[1]["Team PP xGF/60"])

    def test_pp_and_pk_ranks_use_different_team_metrics(self):
        ranks = power_play_team_ranks([
            {"Team": "OTT", "PP xGF/60": 10.0, "PK xGA/60": 4.0},
            {"Team": "NJD", "PP xGF/60": 6.0, "PK xGA/60": 9.0},
            {"Team": "FLA", "PP xGF/60": 6.0, "PK xGA/60": None},
        ])
        self.assertEqual(ranks["OTT"]["PP attack rank"], 1)
        self.assertEqual(ranks["NJD"]["PK matchup rank"], 1)
        self.assertEqual(ranks["NJD"]["PP attack rank"], 2)
        self.assertEqual(ranks["FLA"]["PP attack rank"], 2)
        self.assertNotIn("PK matchup rank", ranks["FLA"])
