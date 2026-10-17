import unittest
from datetime import date

from position_sog import current_slate_only, parse_position_sog, position_matchup


def fixture(season="26_27"):
    teams = ["ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL",
             "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD"]
    tables = []
    for code in ("C", "L", "R", "D"):
        rows = "".join(
            f'<tr data-team="{team}"><td class="col-rk" data-sort-value="{rank}">{rank}</td>'
            f'<td class="team-cell">{team}</td><td class="col-gp" data-sort-value="8">8</td>'
            f'<td class="value" data-sort-value="7.5">7.5</td></tr>'
            for rank, team in enumerate(teams, 1))
        tables.append(f'<div class="spz-table"><h3>SOG Against to {code}</h3>'
                      f'<table class="sob-{code}"><tbody>{rows}</tbody></table>'
                      '<div>Last updated: Oct 17, 2026</div></div>')
    return (f'<select data-spz-f="season"><option value="{season}" selected="selected">2026-27</option></select>'
            '<select data-spz-f="span"><option value="YTD" selected="selected">YTD</option></select>'
            + "".join(tables))


class PositionSogTests(unittest.TestCase):
    def test_parse_and_match(self):
        data = parse_position_sog(fixture(), "2026-27")
        self.assertEqual(data["updated"], "Oct 17, 2026")
        match = position_matchup("ANA", "ANA@BOS", "L", data)
        self.assertEqual(match, {"opponent": "BOS", "position": "LW", "rank": 2,
                                 "gp": 8, "shots_per_game": 7.5})
        self.assertIsNone(position_matchup("NYR", "ANA@BOS", "C", data))
        self.assertIsNone(position_matchup("ANA", "ANA@BOS", "G", data))

    def test_reject_wrong_season(self):
        with self.assertRaisesRegex(ValueError, "different NHL season"):
            parse_position_sog(fixture("25_26"), "2026-27")

    def test_only_today(self):
        self.assertTrue(current_slate_only("2026-10-17", date(2026, 10, 17)))
        self.assertFalse(current_slate_only("2026-10-16", date(2026, 10, 17)))


if __name__ == "__main__":
    unittest.main()
