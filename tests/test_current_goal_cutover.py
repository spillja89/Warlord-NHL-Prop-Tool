import io
import unittest
from types import SimpleNamespace
from zipfile import ZipFile

import pandas as pd

from goal_moves_current import REPLACES, RULES, goals_moves
from goal_recent_form import recent_goal_form_by_player, recent_goals_from_log
from warlords_night_board import goal_display_cards, rank_priced_slate


class CurrentGoalCutoverTests(unittest.TestCase):
    def test_names_and_records_have_one_cutover_mapping_each(self):
        names = [rule[0] for rule in RULES]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), set(REPLACES))
        old_names = [old for old in REPLACES.values() if old != "NEW"]
        self.assertEqual(len(old_names), len(set(old_names)))

    def test_goal_attack_uses_goal_confidence_and_base_signals(self):
        row = {"Goal_Line": .5, "Matrix_Goal": "Green", "Conf_Goal": 85,
               "Conf_Points": 20, "TOI_per_game": 19, "Conf_SOG": 80,
               "Goal_Odds_Over": 175, "Goal_Book": "Book"}
        self.assertEqual([move["name"] for move in goals_moves(row)], ["Base Attack"])
        self.assertEqual(goals_moves({**row, "Conf_Goal": 84}), [])
        self.assertEqual(goals_moves({**row, "Matrix_Goal": "Yellow"}), [])
        self.assertEqual(goals_moves({**row, "TOI_per_game": 18.9}), [])
        self.assertEqual(goals_moves({**row, "Opp_GAA": 5, "Conf_SOG": 79}), [])

    def test_goal_cards_need_named_attack(self):
        common = {"Goal_Line": .5, "Goal_Odds_Over": 175, "Goal_Book": "Book",
                  "Matrix_Goal": "Green", "Conf_Goal": 85, "Conf_Points": 95}
        frame = pd.DataFrame([
            {**common, "Player": "Shooter", "Team": "MIN", "TOI_per_game": 19, "Conf_SOG": 80},
            {**common, "Player": "Baseline only", "Team": "MIN", "TOI_per_game": 17, "Conf_SOG": 70},
        ])
        priced = rank_priced_slate(frame)["Carry"]
        self.assertEqual(len(priced), 2)
        self.assertEqual([card["player"] for card in goal_display_cards(priced)], ["Shooter"])

    def test_recent_goal_form_excludes_slate_day_and_playoffs(self):
        raw = '{"games":[{"d":"2026-10-22","g":3},{"d":"2026-10-21","g":1},{"d":"2026-10-20","g":0}]}'
        self.assertEqual(recent_goals_from_log(raw, "2026-10-22"), .5)
        csv_data = ("playerId,gameDate,situation,I_F_goals,I_F_xGoals\n"
                    "7,20261020,all,1,0.7\n7,20261021,all,0,0.5\n"
                    "7,20261022,all,2,1.0\n7,20261021,5on5,0,0.1\n")
        data = io.BytesIO()
        with ZipFile(data, "w") as archive:
            archive.writestr("2026.csv", csv_data)
        session = SimpleNamespace(get=lambda *args, **kwargs:
                                  SimpleNamespace(content=data.getvalue(), raise_for_status=lambda: None))
        self.assertEqual(recent_goal_form_by_player(session, 2026, "2026-10-22"),
                         {7: {"G": .5, "xG": .6}})


if __name__ == "__main__":
    unittest.main()
