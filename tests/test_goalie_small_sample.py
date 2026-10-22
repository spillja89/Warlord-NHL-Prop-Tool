import unittest

from nhl_edge import goalie_weak_score
from warlord_moves_2026 import goals_moves


class GoalieSmallSampleTests(unittest.TestCase):
    def test_one_or_two_games_do_not_make_goalie_matchup_extreme(self):
        for games in (1, 2):
            self.assertLess(goalie_weak_score(games, .980, 1.0), 50.0)
            self.assertGreater(goalie_weak_score(games, .800, 5.0), 50.0)
            self.assertEqual(goalie_weak_score(games, .800, 5.0, confirmed=False), 50.0)
        tiny = goalie_weak_score(2, .800, 5.0, shots_faced=40)
        established = goalie_weak_score(20, .800, 5.0, shots_faced=600)
        self.assertLess(tiny, established)

    def test_goalie_gaa_alone_never_fires_current_goal_attack(self):
        row = {"Goal_Line": .5, "Matrix_Goal": "Green", "Conf_Goal": 85,
               "Conf_Points": 90, "Opp_GAA": 4.2, "Opp_GP": 20}
        self.assertEqual(goals_moves(row), [])
        shooter = {**row, "TOI_per_game": 19, "Conf_SOG": 80}
        self.assertEqual([move["name"] for move in goals_moves(shooter)], ["Base Attack"])
        self.assertEqual(goals_moves({**row, "Opp_GAA": 1.8}), [])
