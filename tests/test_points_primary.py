import unittest
from unittest.mock import patch
from datetime import date
import json
import pandas as pd

from points_primary import select_primary, resolve_frame, implied
from odds_api_nhl import merge_odds_api_props


class PrimaryPointsTests(unittest.TestCase):
    def test_book_primary_consensus_overrides_stale_half_point(self):
        quotes = {.5: (-526, 'BetOnline'), 1.5: (120, 'BetMGM'), 2.5: (370, 'FanDuel')}
        mains = [{'line': 1.5, 'odds': 110, 'book': 'FanDuel'},
                 {'line': 1.5, 'odds': 120, 'book': 'BetMGM'},
                 {'line': .5, 'odds': -526, 'book': 'BetOnline'}]
        line, source, books = select_primary(quotes, mains)
        self.assertEqual(line, 1.5)
        self.assertEqual(source, 'Book main-market consensus')
        self.assertEqual(len(books), 2)

    def test_book_votes_not_inflated_by_duplicate_quotes(self):
        quotes = {.5: (-170, 'A'), 1.5: (110, 'B')}
        mains = [{'line': .5, 'odds': -170, 'book': 'A'}] * 20 + [
            {'line': 1.5, 'odds': 110, 'book': 'B'},
            {'line': 1.5, 'odds': 115, 'book': 'C'}]
        self.assertEqual(select_primary(quotes, mains)[0], 1.5)

    def test_older_snapshot_inference_uses_price_not_name_or_projection(self):
        for name, odds in [('Leon Draisaitl', -526), ('Connor McDavid', -900), ('Nathan MacKinnon', -700), ('Unknown Player', -700)]:
            row = dict(Player=name, Model_Stats_Season='2026-2027', Points_Line=.5,
                       Points_Odds_Over=odds, Points_Book='Offshore', Points_mu=2.6,
                       BDL_Points_Line_1=.5, BDL_Points_Odds_1=odds, BDL_Points_Book_1='Offshore',
                       BDL_Points_Line_2=1.5, BDL_Points_Odds_2=110, BDL_Points_Book_2='BetMGM',
                       BDL_Points_Line_3=2.5, BDL_Points_Odds_3=325, BDL_Points_Book_3='Other',
                       Points_p_model_over=.99, Tonight_Points_mu=1.9, Tonight_Points_over_pct=90.)
            out = resolve_frame(pd.DataFrame([row]))
            self.assertEqual(out.iloc[0].Points_Line, 1.5)
            self.assertEqual(out.iloc[0].Points_Odds_Over, 110)
            self.assertIn('Inferred', out.iloc[0].Points_Primary_Source)
            self.assertEqual(out.iloc[0].Points_PrePrimary_Line, .5)
            self.assertEqual(out.iloc[0].Points_mu, 2.6)
            self.assertLess(out.iloc[0].Points_p_model_over, .99)
            self.assertLess(out.iloc[0].Tonight_Points_over_pct, 90)
            self.assertAlmostEqual(out.iloc[0].Points_p_imp_over, implied(110))
            pd.testing.assert_frame_equal(resolve_frame(out), out)

    def test_real_half_point_primary_stays_half_point(self):
        q = {.5: (-115, 'A'), 1.5: (350, 'A')}
        self.assertEqual(select_primary(q)[0], .5)
        self.assertEqual(select_primary({.5: (-900, 'A')})[0], .5)  # no invented 1.5
        self.assertIsNone(select_primary({.5: (float('nan'), 'A')})[0])

    def test_exact_line_best_price_and_missing_probability(self):
        row = dict(Model_Stats_Season='2026-2027', Points_Line=.5, Points_Odds_Over=-900, Points_Book='A',
                   BDL_Points_Line_1=1.5, BDL_Points_Odds_1=100, BDL_Points_Book_1='B',
                   BDL_Points_Line_2=1.5, BDL_Points_Odds_2=120, BDL_Points_Book_2='C',
                   Points_p_model_over=.99, Tonight_Points_over_pct=99.)
        out = resolve_frame(pd.DataFrame([row])).iloc[0]
        self.assertEqual((out.Points_Line, out.Points_Odds_Over, out.Points_Book), (1.5, 120, 'C'))
        self.assertTrue(pd.isna(out.Points_p_model_over))
        self.assertTrue(pd.isna(out.Tonight_Points_over_pct))

    @patch('odds_api_nhl._get')
    def test_api_reselects_primary_with_alternates_and_keeps_other_markets(self, get):
        get.side_effect = [[{'id':'edm-van', 'commence_time':'2026-11-08T03:00:00Z',
                            'home_team':'Edmonton Oilers','away_team':'Vancouver Canucks'}],
            {'bookmakers': [
                {'key': 'betmgm', 'title': 'BetMGM', 'markets': [
                    {'key': 'player_points', 'outcomes': [{'name':'Over','description':'Leon Draisaitl','point':1.5,'price':110}]},
                    {'key': 'player_points_alternate', 'outcomes': [{'name':'Over','description':'Leon Draisaitl','point':.5,'price':-600}]},
                    {'key': 'player_power_play_points', 'outcomes': []}]},
                {'key': 'betrivers', 'title': 'BetRivers', 'markets': [
                    {'key': 'player_points', 'outcomes': [{'name':'Over','description':'Leon Draisaitl','point':1.5,'price':120}]},
                    {'key': 'player_power_play_points', 'outcomes': []}]}]}]
        row = dict(Player='Leon Draisaitl', Team='EDM', Opp='VAN', Model_Stats_Season='2026-2027',
                   BDL_Points_Line=.5, BDL_Points_Line_1=.5, BDL_Points_Odds_1=-526, BDL_Points_Book_1='BetOnline',
                   BDL_Goal_Line=.5, Goal_mu=.6)
        out = merge_odds_api_props(pd.DataFrame([row]), date(2026,11,7), 'test-key').iloc[0]
        self.assertEqual(out.BDL_Points_Line, 1.5)
        self.assertEqual((out.BDL_Points_Odds, out.BDL_Points_Book), (120, 'BetRivers'))
        self.assertEqual(len(json.loads(out.Points_Main_Market_Quotes)), 2)
        self.assertEqual(out.BDL_Goal_Line, .5)
        self.assertEqual(out.Goal_mu, .6)

    def test_old_season_and_nonpoints_values_unchanged(self):
        row = dict(Model_Stats_Season='2025-2026', Points_Line=.5, Points_Odds_Over=-900, Points_Book='A',
                   BDL_Points_Line_1=1.5, BDL_Points_Odds_1=100, BDL_Points_Book_1='B',
                   Goal_mu=.4, Conf_Points=87, Conf_Assists=90)
        frame = pd.DataFrame([row])
        pd.testing.assert_frame_equal(resolve_frame(frame), frame)


if __name__ == '__main__':
    unittest.main()
