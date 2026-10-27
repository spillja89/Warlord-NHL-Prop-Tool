import unittest

import pandas as pd

from nhl_edge import fetch_team_edge_zone_time, normalize_skaters_5v5


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class _Session:
    def get(self, url, **kwargs):
        if url.endswith("/team"):
            return _Response({"data": [{"id": 14, "triCode": "TBL"}]})
        return _Response({"zoneTimeDetails": [
            {"strengthCode": "es", "offensiveZonePctg": 0.4194166,
             "defensiveZonePctg": 0.3914216,
             "offensiveZoneRank": 7},
            {"strengthCode": "pp", "offensiveZonePctg": 0.5686933,
             "offensiveZoneRank": 23},
            {"strengthCode": "pk", "defensiveZonePctg": 0.5991088},
        ]})


class OzoneContextTests(unittest.TestCase):
    def test_edge_strengths_remain_separate(self):
        result = fetch_team_edge_zone_time(_Session(), 20262027, ["TBL"])
        row = result.iloc[0]
        self.assertEqual(row["Team_EV_OZ_Time_Pct"], 41.9)
        self.assertEqual(row["Team_EV_DZ_Time_Pct"], 39.1)
        self.assertEqual(row["Team_PP_OZ_Time_Pct"], 56.9)
        self.assertEqual(row["Team_PP_OZ_Rank"], 23)
        self.assertEqual(row["Team_PK_DZ_Time_Pct"], 59.9)

    def test_exact_five_on_five_zone_starts_use_oz_and_dz(self):
        source = pd.DataFrame([{
            "playerId": 7, "name": "Test Skater", "team": "TBL",
            "situation": "5on5", "icetime": 600, "games_played": 2,
            "I_F_oZoneShiftStarts": 12, "I_F_dZoneShiftStarts": 8,
        }])
        row = normalize_skaters_5v5(source).iloc[0]
        self.assertEqual(row["EV5_OZS_Pct"], 60.0)
        self.assertEqual(row["EV5_OZ_Starts"], 12)


if __name__ == "__main__":
    unittest.main()