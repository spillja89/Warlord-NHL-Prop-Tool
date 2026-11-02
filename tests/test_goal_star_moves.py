"""Regression checks for the independent StarScore entry and Goal presentation."""
import ast
import copy
import math
import unittest
from pathlib import Path

import pandas as pd

from goal_moves_current import goals_moves, goal_move_summary
from goal_star_moves import star_goals_moves
from warlords_night_board import goal_display_cards, rank_priced_slate, render_warlords


class GoalStarMovesTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(Player="Goal Fixture", Team="MIN", Game="MIN@CGY", Pos="L",
                        Goal_Line=.5, Goal_Odds_Over=200, Matrix_Goal="Green",
                        Model_Stats_Season="2026-2027", Form_Season="2026-27",
                        Team_Stats_Season="2026-2027", StarScore=90, Drought_G=1,
                        TOI_Pct=90, TOI_per_game=20, **{"iXG%": 95},
                        ShotIntent=3, ShotIntent_Pct=85, Avg5_SOG=3,
                        L10_S=30, L10_G=4, Points_mu=1.4, team_5v5_SF60=25.8,
                        Opp_PK_GA60=8.2, Team_PP_xGF60=8, Pregame_xG_L5_per_game=.1,
                        Roster_Watch=False, Injury_Status="Healthy", Form_Log="[]")

    def test_star_route_has_no_recent_xg_gate(self):
        for value in (None, "", math.nan, .1, .49, .50, .55):
            with self.subTest(value=value):
                moves=star_goals_moves({**self.row,"Pregame_xG_L5_per_game":value})
                self.assertEqual(len(moves),17)
                self.assertEqual(len({m['id'] for m in moves}),17)

    def test_entry_thresholds_and_season_sources(self):
        for field,value in (("StarScore",84.49),("Drought_G",0),("TOI_Pct",80.99),
                            ("Goal_Line",1.5),("Matrix_Goal","Yellow"),
                            ("Model_Stats_Season","2025-2026"),("Form_Season","2025-26"),
                            ("StarScore",None),("Drought_G",math.nan)):
            with self.subTest(field=field,value=value):
                self.assertEqual(star_goals_moves({**self.row,field:value}),[])

    def test_missing_environment_preserves_attacks(self):
        moves=star_goals_moves({**self.row,"Team_Stats_Season":""})
        self.assertEqual(len(moves),8)
        self.assertFalse(any(m['matchup_buff'] for m in moves))
        self.assertEqual(len(star_goals_moves({**self.row,"Opp_PK_GA60":None})),14)

    def test_level_two_uses_current_formula_not_saved_old_score(self):
        row={**self.row,"iXG%":85,"Conf_Goal":100}
        names={m['name'] for m in star_goals_moves(row)}
        self.assertIn('Drought Strike I',names)
        self.assertNotIn('Drought Strike II',names)

    def test_crit_does_not_inherit_special_conditions(self):
        row={**self.row,"L10_S":45,"ShotIntent":4.5,"Points_mu":1.0}
        moves=star_goals_moves(row)
        names={m['name'] for m in moves}
        self.assertIn('Controlled Crit',names)
        self.assertNotIn("Reaper's Return",names)
        self.assertNotIn('Measured Strike',names)
        self.assertNotIn('Arcane Strike',names)
        self.assertFalse(any(m['matchup_buff'] for m in moves))

    def test_player_record_stays_primary_and_all_buffs_remain_visible(self):
        moves=goals_moves(self.row)
        summary=goal_move_summary(moves)
        self.assertEqual(summary['Goal_Attack'],'Precision Crit')
        self.assertEqual(summary['Goal_Attack_Record'],'24/33 · 72.7%')
        self.assertEqual(summary['Goal_Attack_Sample_BE'],36.9)
        self.assertEqual(len(summary['Goal_Buffs'].splitlines()),9)
        self.assertIn('20/24 · 83.3%',summary['Goal_Buffs'])
        cards=goal_display_cards(rank_priced_slate(pd.DataFrame([self.row]))['Carry'])
        self.assertEqual(len(cards),1)
        self.assertFalse(cards[0]['move']['matchup_buff'])
        html=render_warlords({'Carry':cards},limit=1,roles=('Carry',),show_hero=False)
        self.assertIn('wn-unit--crit',html)
        for move in star_goals_moves(self.row):
            # Apostrophes are HTML escaped in player-visible text.
            from html import escape
            self.assertIn(escape(move['name']),html)

    def test_app_sort_probe_never_ranks_buff_as_player_attack(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'app.py').read_text(encoding='utf-8'))
        functions=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef)
                   and n.name in {'_adj_win','_probe_goals_best'}]
        namespace={'_goals_carry_moves':goals_moves}
        exec(compile(ast.Module(body=functions,type_ignores=[]),'app_goal_probe','exec'),namespace)
        result=namespace['_probe_goals_best'](self.row)
        self.assertEqual(result['title'],'Precision Crit')


if __name__=='__main__':
    unittest.main()
