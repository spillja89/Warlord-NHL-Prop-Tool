import unittest
import pandas as pd

from assist_moves_current import assists_moves
from warlords_night_board import _move_rank, featured_warlords, rank_priced_slate, render_warlords


class AssistMoveTreeTests(unittest.TestCase):
    def base_row(self, **changes):
        row = {
            "Assists_Line": 0.5, "Matrix_Assists": "Green",
            "Conf_Assists": 81, "iXA%": 95,
            "PP_Points60": 3.75, "PP_iXA60": 3.9,
            "PP_TOI_per_game": 3.75, "TOI_per_game": 22,
            "Team_PP_xGF60": 8.3, "Assists_mu": 1.1,
            "Drought_G": 2,
        }
        row.update(changes)
        return row

    def test_all_three_crits_fire_once_and_outrank_parent_specials(self):
        moves = assists_moves(self.base_row())
        crits = [move for move in moves if move["kind"] == "CRIT"]
        self.assertEqual({move["name"] for move in crits}, {
            "Runic Infusion Crit", "Rune Conduit Crit", "Odin's Vigil Crit"})
        self.assertEqual(len(crits), 3)
        self.assertEqual(max(moves, key=_move_rank)["kind"], "CRIT")
        self.assertEqual({(move["wins"], move["picks"]) for move in crits},
                         {(29, 41), (20, 25), (20, 24)})

    def test_pp_environment_cutoff_upgrades_without_changing_player_specials(self):
        below = assists_moves(self.base_row(Team_PP_xGF60=8.29))
        above = assists_moves(self.base_row(Team_PP_xGF60=8.3))
        self.assertFalse(any(move["kind"] == "CRIT" for move in below))
        self.assertEqual(sum(move["kind"] == "SPECIAL" for move in below), 3)
        self.assertEqual(sum(move["kind"] == "CRIT" for move in above), 3)

    def test_entry_and_role_cutoffs_are_exact(self):
        for change in ({"Assists_Line": 1.5}, {"Matrix_Assists": "Yellow"},
                       {"Conf_Assists": 80}, {"iXA%": 94.99}):
            self.assertEqual(assists_moves(self.base_row(**change)), [])
        no_special = assists_moves(self.base_row(
            PP_iXA60=3.89, PP_TOI_per_game=3.74, TOI_per_game=21.99))
        self.assertFalse(any(move["kind"] in {"SPECIAL", "CRIT"} for move in no_special))

    def test_mu_and_drought_are_context_only(self):
        moves = assists_moves(self.base_row(
            PP_Points60=0, PP_iXA60=0, PP_TOI_per_game=0, TOI_per_game=0))
        self.assertEqual({move["kind"] for move in moves}, {"STANCE", "CONTEXT"})
        self.assertEqual(max(moves, key=_move_rank)["kind"], "STANCE")

    def test_featured_support_card_uses_one_primary_crit(self):
        row = self.base_row(Player="Test Playmaker", Team="VGK", Opp="DAL",
                            Game="VGK@DAL", Assists_Odds_Over=155,
                            Assists_Book="BetMGM")
        boards = rank_priced_slate(pd.DataFrame([row]))
        featured = featured_warlords(boards)
        self.assertEqual(len(featured["Support"]), 1)
        card = featured["Support"][0]
        self.assertEqual(card["move"]["kind"], "CRIT")
        self.assertEqual(card["move_count"], len(assists_moves(row)))
        html = render_warlords(featured, roles=("Support",))
        self.assertIn("TEAM PP xGF/60", html)
        self.assertIn("CRIT", html)
        self.assertIn("2026–27 REG MOVE", html)


if __name__ == "__main__":
    unittest.main()
