import unittest

from nhl_edge import fetch_team_pp_opportunities


class _Response:
    def __init__(self, rows):
        self.rows = rows

    def raise_for_status(self):
        pass

    def json(self):
        return {"data": self.rows}


class _Session:
    def __init__(self, reports):
        self.reports = reports

    def get(self, url, **kwargs):
        report = url.rsplit("/", 1)[-1]
        return _Response(self.reports[report])


class TeamPPOTests(unittest.TestCase):
    def test_official_opportunities_and_time_are_kept_as_team_context(self):
        session = _Session({
            "team": [{"id": 13, "triCode": "FLA"},
                     {"id": 30, "triCode": "MIN"}],
            "powerplay": [
                {"teamId": 13, "ppOpportunitiesPerGame": 3.4,
                 "ppTimeOnIcePerGame": 300, "gamesPlayed": 10},
                {"teamId": 30, "ppOpportunitiesPerGame": 2.8,
                 "ppTimeOnIcePerGame": 240, "gamesPlayed": 11},
            ],
            "penaltykill": [
                {"teamId": 13, "timesShorthandedPerGame": 2.6,
                 "pkTimeOnIcePerGame": 210, "gamesPlayed": 10},
                {"teamId": 30, "timesShorthandedPerGame": 3.1,
                 "pkTimeOnIcePerGame": 270, "gamesPlayed": 11},
            ],
        })
        rows = fetch_team_pp_opportunities(session, 20262027).set_index("Team")
        self.assertEqual(set(rows.index), {"FLA", "MIN"})
        self.assertAlmostEqual(rows.loc["FLA", "Team_PPO_PG"], 3.4)
        self.assertAlmostEqual(rows.loc["FLA", "Team_PP_Time_PG"], 5.0)
        self.assertAlmostEqual(rows.loc["MIN", "Team_PKO_PG"], 3.1)
        self.assertAlmostEqual(rows.loc["MIN", "Team_PK_Time_PG"], 4.5)

    def test_feed_failure_returns_unavailable_columns(self):
        session = _Session({"team": [], "powerplay": [], "penaltykill": []})
        rows = fetch_team_pp_opportunities(session, 20262027)
        self.assertTrue(rows.empty)
        self.assertIn("Team_PPO_PG", rows.columns)


if __name__ == "__main__":
    unittest.main()
