import unittest
from unittest.mock import Mock

import pandas as pd

from goalie_identity import fetch_goalie_name_ids, goalie_ids_from_rows
from nhl_edge import normalize_goalies, resolve_goalie_for_team


class GoalieIdentityTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame([dict(
            playerId=8482447, name="Leevi Merilinen", team="VAN", situation="all",
            games_played=14, icetime=45444, ongoal=386, goals=43, xGoals=40.71,
        )])
        self.goalies = normalize_goalies(self.raw)

    def test_verified_id_repairs_provider_spelling(self):
        row = resolve_goalie_for_team(self.goalies, {}, "VAN", "Leevi Merilainen")
        self.assertEqual(row["Goalie_ID"], 8482447)
        self.assertEqual(row["GP"], 14)
        self.assertAlmostEqual(row["SV"], 343 / 386)
        self.assertAlmostEqual(row["GAA"], 43 * 3600 / 45444)
        self.assertEqual(row["xGA"], 40.71)
        self.assertEqual(row["Match_Method"], "nhl_player_id")

    def test_wrong_team_or_explicit_id_cannot_borrow_stats(self):
        self.assertIsNone(resolve_goalie_for_team(
            self.goalies, {}, "STL", "Leevi Merilainen")["GP"])
        self.assertIsNone(resolve_goalie_for_team(
            self.goalies, {}, "VAN", "Leevi Merilainen", starter_id=1)["GP"])

    def test_official_lineup_id_handles_abbreviated_name(self):
        row = resolve_goalie_for_team(
            self.goalies, {}, "VAN", "L. Merilainen", starter_id=8482447)
        self.assertEqual(row["Goalie_ID"], 8482447)

    def test_feed_outage_retains_verified_identity(self):
        session = Mock()
        session.get.side_effect = TimeoutError("unavailable")
        self.assertEqual(fetch_goalie_name_ids(session, 2026)["leevimerilainen"], 8482447)

    def test_duplicate_official_names_are_not_guessed(self):
        self.assertEqual(goalie_ids_from_rows([
            dict(playerId=1, goalieFullName="Same Name"),
            dict(playerId=2, goalieFullName="Same Name"),
        ]), {})


if __name__ == "__main__":
    unittest.main()
