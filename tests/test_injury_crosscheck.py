from datetime import date, datetime, timezone

import pandas as pd

from nhl_edge import apply_injury_dfo, clear_dressed_injury_reports, combine_injury_reports, parse_espn_injuries
from player_availability import apply_live_injury_overlay, unavailable_mask


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


def test_live_check_removes_suzuki_from_saved_slate_only():
    reports = parse_espn_injuries(_espn_payload(), {"MTL"}, date(2026, 10, 25))
    saved = pd.DataFrame([
        {"Date": "2026-10-25", "Team": "MTL", "Player": "Nick Suzuki", "Injury_Status": "Healthy", "Available": True,
         "Injury_Text": float("nan"), "Injury_Return": float("nan")},
        {"Date": "2026-10-24", "Team": "MTL", "Player": "Nick Suzuki", "Injury_Status": "Healthy", "Available": True,
         "Injury_Text": float("nan"), "Injury_Return": float("nan")},
    ])
    checked, newly_excluded = apply_live_injury_overlay(saved, reports, date(2026, 10, 25))
    assert newly_excluded == 1
    assert unavailable_mask(checked).tolist() == [True, False]
    assert checked.loc[0, "Injury_Return"] == "2026-11-18"


def test_official_dressed_lineup_overrides_stale_injury_report():
    reports = parse_espn_injuries(_espn_payload(), {"MTL"}, date(2026, 10, 25))
    dressed = pd.DataFrame([{"Team": "MTL", "Player": "Nick Suzuki", "Lineup_Status": "Dressed"}])
    assert clear_dressed_injury_reports(reports, dressed).empty
    saved = dressed.assign(Date="2026-10-25", Injury_Status="Healthy", Available=True)
    checked, newly_excluded = apply_live_injury_overlay(saved, reports, date(2026, 10, 25))
    assert newly_excluded == 0
    assert not unavailable_mask(checked).any()
