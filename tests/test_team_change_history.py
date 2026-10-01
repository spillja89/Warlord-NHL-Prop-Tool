import unittest
from datetime import date

import pandas as pd

from nhl_edge import align_skater_stats_team
from roster_watchlist import add_roster_watchlist, current_roster_teams_by_id


class TeamChangeHistoryTests(unittest.TestCase):
    def test_prior_club_stats_follow_official_player_id_to_todays_team(self):
        rosters = {
            "FLA": {"tradedplayer": {"name": "Traded Player", "id": 123, "position": "L"}},
            "OTT": {"otherplayer": {"name": "Other Player", "id": 456, "position": "C"}},
        }
        current_teams = current_roster_teams_by_id(rosters)
        prior = pd.DataFrame([
            {"playerId": 123, "Player": "Traded Player", "Team": "OTT", "icetime": 70000, "iXG_raw": 30.0},
            {"playerId": 456, "Player": "Other Player", "Team": "OTT", "icetime": 60000, "iXG_raw": 20.0},
        ])
        aligned = align_skater_stats_team(prior, current_teams, keep_source=True, weight_column="icetime")
        traded = aligned.loc[aligned["playerId"] == 123].iloc[0]
        self.assertEqual((traded["Team"], traded["Model_Stats_Team"], traded["iXG_raw"]),
                         ("FLA", "OTT", 30.0))
        self.assertTrue(traded["Team_Changed"])
        tracker = aligned.rename(columns={"Player": "Player"})
        tracker = add_roster_watchlist(tracker, None, {"FLA", "OTT"}, {}, date(2026, 10, 1), None,
                                       rosters=rosters)
        self.assertEqual(len(tracker), 2)
        self.assertEqual(tracker.loc[tracker["playerId"] == 123, "Roster_Status"].iloc[0], "Active roster")

    def test_duplicate_historical_clubs_do_not_duplicate_current_player(self):
        prior = pd.DataFrame([
            {"playerId": 123, "Team": "OTT", "icetime": 60000},
            {"playerId": 123, "Team": "TOR", "icetime": 25000},
        ])
        aligned = align_skater_stats_team(prior, {123: "FLA"}, keep_source=True, weight_column="icetime")
        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned.iloc[0]["Model_Stats_Team"], "OTT")


if __name__ == "__main__":
    unittest.main()
