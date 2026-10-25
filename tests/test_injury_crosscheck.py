from datetime import date, datetime, timezone

import pandas as pd

from nhl_edge import apply_injury_dfo, combine_injury_reports, parse_espn_injuries


def _espn_payload(status="Out", season=2027):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "season": {"year": season},
        "injuries": [{
            "displayName": "Montreal Canadiens",
            "injuries": [{
                "status": status,
                "athlete": {"displayName": "Nick Suzuki", "team": {"abbreviation": "MTL"}},
                "details": {"type": "Upper Body", "returnDate": "2026-11-18"},
            }],
        }],
    }


def test_espn_confirmed_out_survives_missing_dfo_report():
    dfo = pd.DataFrame(columns=["Team", "Player", "Status", "Injury", "Expected_Return", "Player_norm"])
    espn = parse_espn_injuries(_espn_payload(), {"MTL"}, date(2026, 10, 25))
    reports = combine_injury_reports(dfo, espn)
    skaters = pd.DataFrame([{"Team": "MTL", "Player": "Nick Suzuki", "TOI_Pct": 80}])
    checked = apply_injury_dfo(skaters, reports)
    assert checked.loc[0, "Injury_Status"] == "Out"
    assert checked.loc[0, "Injury_Type"] == "Upper"
    assert checked.loc[0, "Injury_Return"] == "2026-11-18"
    assert checked.loc[0, "Injury_Badge"] == "🔴 OUT"


def test_confirmed_out_beats_dfo_game_time_decision():
    dfo = pd.DataFrame([{
        "Team": "MTL", "Player": "Nick Suzuki", "Player_norm": "nick suzuki",
        "Status": "GTD", "Injury": "", "Expected_Return": "",
    }])
    espn = parse_espn_injuries(_espn_payload(), {"MTL"}, date(2026, 10, 25))
    reports = combine_injury_reports(dfo, espn)
    assert len(reports) == 1
    assert reports.iloc[0]["Status"] == "Out"


def test_wrong_season_is_ignored():
    reports = parse_espn_injuries(_espn_payload(season=2026), {"MTL"}, date(2026, 10, 25))
    assert reports.empty
