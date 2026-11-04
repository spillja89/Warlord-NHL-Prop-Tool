import unittest
import textwrap
from pathlib import Path
import pandas as pd

from sog_confidence import VERSION as CONF_VERSION
from sog_moves_current import (LEGACY_VERSION, SOURCE, sog_moves, pk_duration_seconds,
                               attach_sog_route_context)
from warlord_moves_2026 import sog_moves as live_sog_moves, legacy_sog_moves, frozen_move_tags
from warlords_night_board import rank_priced_slate, featured_warlords, render_warlords


def row(**changes):
    r=dict(Player="Test Jungle",Team="TST",Opp="OPP",Game="TST@OPP",
        Model_Stats_Season="2026-2027",SOG_Line=2.5,SOG_Odds_Over=110,
        SOG_Book="Test book",Matrix_SOG="Green",Conf_SOG=67,
        Conf_SOG_Move=1,Conf_SOG_Version=CONF_VERSION,
        Talent_Tier="ELITE",Drought_SOG=0,
        PPO_Source=SOURCE,Opp_PKO_GP=8,Opp_PK_Time_PG=5.2,Opp_PKO_PG=3,
        TOI_Pct=85,TOI_per_game=21,Player_5v5_SOG_Share=15,Opp_SOG_Against_L10=28)
    r.update(changes)
    return r


class CurrentSOGKitTests(unittest.TestCase):
    def test_base_is_a_featured_attack_without_legacy_confidence(self):
        r=row(Opp_PK_Time_PG=6)
        self.assertEqual([m['name'] for m in live_sog_moves(r)],["Berserker Swipe"])
        cards=featured_warlords(rank_priced_slate(pd.DataFrame([r])))['Jungle']
        self.assertEqual(len(cards),1)
        self.assertEqual((cards[0]['move']['wins'],cards[0]['move']['picks']),(31,52))

    def test_boundary_and_current_source(self):
        for change in [dict(Conf_SOG=66.99),dict(Matrix_SOG="Yellow"),
                       dict(SOG_Line=3.5),dict(Model_Stats_Season="2025-2026"),
                       dict(Conf_SOG=None),dict(Conf_SOG_Version="old")]:
            self.assertEqual(sog_moves(row(**change)),[])

    def test_stronger_and_buff_are_separate(self):
        self.assertEqual([m['name'] for m in sog_moves(row(Conf_SOG=71,Opp_PK_Time_PG=6))],
            ["Berserker Swipe","Berserker’s Rage"])
        self.assertEqual([m['name'] for m in sog_moves(row())],
            ["Berserker Swipe","Paralysis"])

    def test_pk_duration_uses_exact_boundary(self):
        for seconds,expected in [(103.999,True),(104,True),(104.001,False)]:
            moves=sog_moves(row(Opp_PK_Time_PG=seconds*3/60))
            self.assertEqual(any(m.get('matchup_buff') for m in moves),expected)
        for changes in [dict(PPO_Source="unavailable"),dict(Opp_PKO_GP=2),
                        dict(Opp_PK_Time_PG=None),dict(Opp_PKO_PG=0)]:
            self.assertIsNone(pk_duration_seconds(row(**changes)))
            self.assertFalse(any(m.get('matchup_buff') for m in sog_moves(row(**changes))))

    def test_empowered_record_and_single_card(self):
        r=row(Conf_SOG=71)
        moves=sog_moves(r)
        self.assertEqual(len(moves),4)
        self.assertEqual((moves[-1]['wins'],moves[-1]['picks']),(24,31))
        self.assertTrue(moves[-1]['empowered'])
        cards=featured_warlords(rank_priced_slate(pd.DataFrame([r,r])))
        self.assertEqual(len(cards['Jungle']),1)
        self.assertEqual(cards['Jungle'][0]['move']['name'],"Assassin’s Overdrive")
        html=render_warlords(cards,roles=('Jungle',),show_hero=False)
        for text in ["77.4%","24/31","EMPOWERED","PARALYSIS ACTIVE","104.00s",
                     "2026–27 REG MOVE","sample BE 52.0%","31/52","27/37","28/40"]:
            self.assertIn(text,html)
        self.assertNotIn("Mar 15+",html)

    def test_tracker_metadata_matches_primary_card(self):
        for conf,minutes,name in [(67,6,"Berserker Swipe"),(67,5.2,"Berserker Swipe"),
                                 (71,6,"Berserker’s Rage"),(71,5.2,"Assassin’s Overdrive")]:
            frame=attach_sog_route_context(pd.DataFrame([row(Conf_SOG=conf,Opp_PK_Time_PG=minutes)]))
            card=featured_warlords(rank_priced_slate(frame))['Jungle'][0]
            self.assertEqual(frame.SOG_Attack.iloc[0],name)
            self.assertEqual(card['move']['name'],name)
            self.assertAlmostEqual(frame.SOG_Move_Pct.iloc[0],100*card['move']['wins']/card['move']['picks'])
            pd.testing.assert_frame_equal(frame,attach_sog_route_context(frame))

    def test_versioned_frozen_moves_and_legacy_35(self):
        tags=[m for m in frozen_move_tags(row(Conf_SOG=71)) if m['market']=='SOG']
        self.assertEqual(len(tags),4)
        self.assertTrue(all(m['kit_version']==LEGACY_VERSION for m in tags))
        r=row(SOG_Line=3.5,Conf_SOG_Move=90)
        self.assertEqual(live_sog_moves(r),legacy_sog_moves(r))
        self.assertTrue(all(not m.get('sog_current') for m in live_sog_moves(r)))

    def test_engine_finalization_sets_sources_before_freezing(self):
        import json
        source=(Path(__file__).resolve().parents[1]/'nhl_edge.py').read_text(encoding='utf-8')
        start=source.index('    # Publish source provenance before evaluating')
        end=source.index('    tracker = attach_points15_research_flags',start)
        r=row();r.pop('Model_Stats_Season');r['Roster_Watch']=False
        namespace=dict(tracker=pd.DataFrame([r]),sk_stats_year=2026,team_stats_year=2026,
                       goalie_stats_year=2026,MOVE_KIT_VERSION='test',frozen_move_tags=frozen_move_tags,json=json,pd=pd,matrix_sog=lambda *a,**k: 'Red')
        exec(textwrap.dedent(source[start:end]),namespace)
        out=namespace['tracker']
        tags=[m for m in json.loads(out.Fired_Moves.iloc[0]) if m['market']=='SOG']
        self.assertEqual(len(tags),4)
        self.assertEqual(out.SOG_Attack.iloc[0],"Assassin’s Overdrive")


if __name__=='__main__':unittest.main()
