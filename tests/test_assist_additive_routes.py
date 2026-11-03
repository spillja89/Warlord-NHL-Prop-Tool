import unittest
import pandas as pd
from assist_moves_current import assists_moves, existing_assists_moves
from assist_routes_2026 import new_assists_moves, VERSION
from warlords_night_board import rank_priced_slate, featured_warlords, render_warlords


def row(**changes):
    value = dict(Player="Test Support", Team="TST", Opp="OPP", Game="TST@OPP",
                 Assists_Line=.5, Assists_Odds_Over=125, Assists_Book="Test book",
                 Matrix_Assists="Green", Conf_Assists=80, **{"iXA%":94},
                 Model_Stats_Season="2026-2027", Team_Stats_Season="2026-2027",
                 Goal_mu=.33, i5v5_primaryAssists60=1.05, PPP10_total=4,
                 Player_SOG_Share_Proxy=11, Player_5v5_SOG_Share=8,
                 Team_PP_GF60=5.7, opp_5v5_HDCA60=2.2, L10_S=19)
    value.update(changes)
    return value


class AdditiveAssistTests(unittest.TestCase):
    def test_new_entry_does_not_inherit_legacy_gate(self):
        r=row()
        self.assertEqual(existing_assists_moves(r), [])
        self.assertIn("Runic Orchestration Crit", {m['name'] for m in assists_moves(r)})
        for field,number in [('Conf_Assists',79.99),('iXA%',93.99)]:
            self.assertEqual(new_assists_moves(row(**{field:number})), [])

    def test_legacy_rules_and_records_remain_in_union(self):
        r=row(Conf_Assists=90, **{'iXA%':99}, PP_Points60=4,
              PP_iXA60=4, PP_TOI_per_game=4, TOI_per_game=23, Team_PP_xGF60=9,
              Assists_mu=1.2, Drought_G=2)
        old=existing_assists_moves(r)
        expected={"Staff (Green Stance)":(137,238),"Arcane Channel I":(106,172),
                  "Runic Infusion":(70,105),"Rune Conduit":(50,75),"Odin's Vigil":(47,69),
                  "Runic Infusion Crit":(29,41),"Rune Conduit Crit":(20,25),
                  "Odin's Vigil Crit":(20,24),"High Assist μ":(100,163),"Arcane Rebound":(66,100)}
        self.assertEqual({m['name']:(m['wins'],m['picks']) for m in old}, expected)
        combined=assists_moves(r)
        self.assertEqual([m for m in combined if m['kit_version']!=VERSION],old)
        self.assertEqual(len({m['name'] for m in combined}),len(combined))

    def test_environment_requires_current_source_and_real_values(self):
        for changes in [{'Team_Stats_Season':'2025-2026'}, {'Team_PP_GF60':None},
                        {'opp_5v5_HDCA60':float('nan')},{'Team_PP_GF60':5.69},
                        {'opp_5v5_HDCA60':2.21}]:
            moves=new_assists_moves(row(**changes))
            self.assertFalse(any(m['environment_enhanced'] for m in moves))
            self.assertIn("Oracle’s Guidance",{m['name'] for m in moves})

    def test_share_environment_does_not_inherit_actual_share_gate(self):
        names={m['name'] for m in new_assists_moves(row(Player_5v5_SOG_Share=0))}
        self.assertIn('Empowered Storm Conduit',names)
        self.assertNotIn('Storm Conduit II',names)

    def test_crit_requires_both_player_routes(self):
        for changes in [{'PPP10_total':3},{'i5v5_primaryAssists60':1.06}]:
            names={m['name'] for m in new_assists_moves(row(**changes))}
            self.assertNotIn('Runic Orchestration Crit',names)

    def test_no_prior_season_or_wrong_line_new_moves(self):
        for changes in [{'Model_Stats_Season':'2025-2026'},{'Assists_Line':1.5},{'Matrix_Assists':'Yellow'}]:
            self.assertEqual(new_assists_moves(row(**changes)),[])

    def test_one_card_displays_all_routes_and_correct_crit_caption(self):
        r=row()
        board=rank_priced_slate(pd.DataFrame([r,r]))
        self.assertEqual(len(board['Support']),1)
        featured=featured_warlords(board)
        self.assertEqual(featured['Support'][0]['move']['name'],'Runic Orchestration Crit')
        html=render_warlords(featured,roles=('Support',))
        self.assertEqual(html.count('<article class="wn-unit'),1)
        for text in ['Runic Orchestration Crit','Oracle’s Guidance','Runic Resonance I',
                     'Storm Conduit I','Oracle + Runesmith','TEAM PP ACTUAL GF/60 ≥5.7',
                     'OPP 5v5 HDCA/60 ≤2.2','prefers-reduced-motion']:
            self.assertIn(text,html)
        self.assertNotIn('CRIT GATE 8.3',html)
        legacy=row(Conf_Assists=90, **{'iXA%':99},Model_Stats_Season='2025-2026',PP_iXA60=4,Team_PP_xGF60=9)
        legacy_html=render_warlords(featured_warlords(rank_priced_slate(pd.DataFrame([legacy]))),roles=('Support',))
        self.assertIn('CRIT GATE 8.3',legacy_html)


if __name__=='__main__':unittest.main()
