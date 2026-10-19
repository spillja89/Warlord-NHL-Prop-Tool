import unittest
from datetime import datetime, timedelta, timezone

from pregame_checks import fetch_official_lineups, inspect_boxscore_lineup, player_lineup_status


def boxscore(away_count=18, home_count=18, starter=True):
    return {
        "awayTeam": {"abbrev": "NYR"}, "homeTeam": {"abbrev": "BOS"},
        "playerByGameStats": {
            "awayTeam": {"forwards": [{"playerId": i} for i in range(1, away_count + 1)],
                         "goalies": [{"playerId": 901, "name": "Away Goalie", "starter": starter}]},
            "homeTeam": {"forwards": [{"playerId": i} for i in range(101, 101 + home_count)],
                         "goalies": [{"playerId": 902, "name": "Home Goalie", "starter": starter}]},
        },
    }


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class Session:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def get(self, url, timeout):
        self.calls += 1
        return Response(self.payload)


class PregameChecksTest(unittest.TestCase):
    def test_full_game_roster_identifies_dressed_and_absent(self):
        lineups = inspect_boxscore_lineup(boxscore())
        self.assertEqual(player_lineup_status(1, "NYR", {"NYR": {"status": "Official lineup", **lineups["NYR"]}}), "Dressed")
        self.assertEqual(player_lineup_status(99, "NYR", {"NYR": {"status": "Official lineup", **lineups["NYR"]}}), "Not dressed")
        self.assertEqual(lineups["BOS"]["starter_name"], "Home Goalie")

    def test_incomplete_lineup_never_implies_scratch(self):
        self.assertEqual(inspect_boxscore_lineup(boxscore(home_count=17)), {})
        self.assertEqual(player_lineup_status(99, "NYR", {}), "Lineup pending")

    def test_distant_games_are_not_polled(self):
        now = datetime(2026, 10, 19, 15, tzinfo=timezone.utc)
        game = {"id": 2026020138, "away": "NYR", "home": "BOS",
                "startTimeUTC": (now + timedelta(hours=4)).isoformat()}
        session = Session(boxscore())
        status, checked = fetch_official_lineups(session, [game], now=now)
        self.assertEqual(session.calls, 0)
        self.assertEqual(status["NYR"]["status"], "Lineup pending")
        self.assertTrue(checked.startswith("2026-10-19"))

    def test_near_start_uses_full_game_roster(self):
        now = datetime(2026, 10, 19, 22, tzinfo=timezone.utc)
        game = {"id": 2026020138, "away": "NYR", "home": "BOS",
                "startTimeUTC": (now + timedelta(minutes=45)).isoformat()}
        session = Session(boxscore())
        status, _ = fetch_official_lineups(session, [game], now=now)
        self.assertEqual(session.calls, 1)
        self.assertEqual(status["NYR"]["status"], "Official lineup")
        self.assertEqual(status["BOS"]["starter_name"], "Home Goalie")


if __name__ == "__main__":
    unittest.main()

