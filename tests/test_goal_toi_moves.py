"""Regression tests for the third Goal entry path and cumulative tier records."""
import unittest
import math
import pandas as pd
from goal_moves_current import goals_moves, goal_move_summary
from goal_toi_moves import toi_goals_moves
from warlords_night_board import rank_priced_slate, goal_display_cards, render_warlords


class GoalToiMovesTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(Player="TOI Fixture",Team="FLA",Game="DET@FLA",Goal_Line=.5,
                      Goal_Odds_Over=180,Matrix_Goal="Green",StarScore=84.5,
                      TOI_per_game=20,L10_S=39,Drought_G=0,Drought_PPP=0,
                      Model_Stats_Season="2026-2027",Form_Season="2026-27",
                      Pregame_xG_L5_per_game=.1,Conf_Goal=1,
                      Injury_Status="Healthy",Roster_Watch=False,Form_Log="[]")

    def test_entry_is_independent_of_goal_drought_confidence_and_recent_xg(self):
        for value in (None,math.nan,.1,.49):
            row={**self.row,'Pregame_xG_L5_per_game':value}
            moves=goals_moves(row)
            self.assertEqual([m['name'] for m in moves],['Star Barrage'])
            self.assertEqual((moves[0]['wins'],moves[0]['picks']),(66,131))

    def test_highest_ppp_level_only_with_cumulative_historical_record(self):
        for drought,name,record in ((1,'Runic Barrage I',(45,79)),
                                    (2,'Runic Barrage II',(35,53)),
                                    (3,'Runic Barrage II',(35,53)),
                                    (7,'Runic Barrage II',(35,53))):
            with self.subTest(drought=drought):
                moves=toi_goals_moves({**self.row,'Drought_PPP':drought})
                self.assertEqual([m['name'] for m in moves],['Star Barrage',name])
                self.assertEqual((moves[-1]['wins'],moves[-1]['picks']),record)
                self.assertNotIn('III',moves[-1]['name'])

    def test_missing_ppp_keeps_volume_attack(self):
        for value in (None,'',math.nan):
            self.assertEqual([m['name'] for m in toi_goals_moves({**self.row,'Drought_PPP':value})],['Star Barrage'])

    def test_required_thresholds_and_seasons(self):
        for field,value in (('StarScore',84.49),('TOI_per_game',19.99),('L10_S',40),
                            ('L10_S',None),('TOI_per_game',math.nan),('Goal_Line',1.5),
                            ('Matrix_Goal','Yellow'),('Model_Stats_Season','2025-2026'),
                            ('Form_Season','2025-26')):
            with self.subTest(field=field,value=value):
                self.assertEqual(toi_goals_moves({**self.row,field:value}),[])

    def test_third_path_reaches_cards_tracker_and_visible_signals(self):
        row={**self.row,'Drought_PPP':2}
        cards=goal_display_cards(rank_priced_slate(pd.DataFrame([row]))['Carry'])
        self.assertEqual(len(cards),1)
        self.assertEqual(cards[0]['move']['name'],'Runic Barrage II')
        summary=goal_move_summary(goals_moves(row))
        self.assertIn('Star Barrage',summary['Goal_All_Attacks'])
        self.assertIn('Runic Barrage II',summary['Goal_All_Attacks'])
        self.assertNotIn('Runic Barrage I ·',summary['Goal_All_Attacks'])
        html=render_warlords({'Carry':cards},limit=1,roles=('Carry',),show_hero=False)
        self.assertIn('Star Barrage',html)
        self.assertIn('Runic Barrage II',html)
        self.assertIn('sample BE 39.6%',html)
        self.assertIn('PPP drought 2',html)


if __name__=='__main__':unittest.main()
