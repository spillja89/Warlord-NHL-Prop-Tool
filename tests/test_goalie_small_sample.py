import unittest

from nhl_edge import goalie_weak_score
from warlord_moves_2026 import goals_moves


class GoalieSmallSampleTests(unittest.TestCase):
    def test_one_or_two_games_do_not_make_goalie_matchup_extreme(self):
        for games in (1, 2):
            self.assertEqual(goalie_weak_score(games, .980, 1.0), 50.0)
            self.assertEqual(goalie_weak_score(games, .800, 5.0), 50.0)
        self.assertLess(goalie_weak_score(3, .980, 1.0), 50.0)
        self.assertGreater(goalie_weak_score(3, .800, 5.0), 50.0)

    def test_small_sample_does_not_fire_gaa_based_goal_move(self):
        row = {"Goal_Line": .5, "Matrix_Goal": "Green", "Conf_Points": 90,
               "Goal_mu": .5, "Opp_GAA": 4.2, "Opp_GP": 2}
        small_sample = {move["name"] for move in goals_moves(row)}
        established = {move["name"] for move in goals_moves({**row, "Opp_GP": 3})}

        self.assertIn("Base Attack", small_sample)
        self.assertNotIn("Fallen Guardian", small_sample)
        self.assertNotIn("Berserker Aggression", small_sample)
        self.assertIn("Fallen Guardian", established)
        self.assertIn("Berserker Aggression", established)
