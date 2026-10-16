import pandas as pd

from audit_pp_units_history import add_prior_roles


def test_unit_estimate_uses_only_previous_games():
    rows = []
    for game in range(1, 5):
        for player in range(1, 11):
            # The two groups exchange PP time in game four. Its unit estimate
            # must still reflect what was known before that game.
            top_before = player <= 5
            pp_seconds = (180 if top_before else 30) if game < 4 else (30 if top_before else 180)
            rows.append({"gameDate": f"2025-10-{game + 6:02d}", "gameId": game,
                         "teamAbbrev": "BOS", "playerId": player,
                         "ppTimeOnIce": pp_seconds, "ppPoints": 0})
    result = add_prior_roles(pd.DataFrame(rows))
    fourth = result.loc[result["gameId"].eq(4)].set_index("playerId")
    assert fourth.loc[1, "estimated_unit"] == "PP1"
    assert fourth.loc[10, "estimated_unit"] == "PP2"
    assert result.loc[result["gameId"].lt(4), "estimated_unit"].isna().all()
