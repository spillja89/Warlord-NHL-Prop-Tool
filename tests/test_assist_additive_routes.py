import unittest
import pandas as pd
from assist_moves_current import assists_moves, existing_assists_moves
from assist_routes_2026 import new_assists_moves, VERSION, assist_display_tier, assist_display_name
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
        expected={"Staff (Green Stance)":(124,215),"Arcane Channel I":(90,148),
                  "Runic Infusion":(65,109),"Rune Conduit":(66,112),"Odin's Vigil":(45,66),
                  "Runic Infusion Crit":(46,73),"Rune Conduit Crit":(32,52),
                  "Odin's Vigil Crit":(31,39),"High Assist μ":(115,195),"Arcane Rebound":(64,96)}
        self.assertEqual({m['name']:(m['wins'],m['picks']) for m in old}, expected)
        combined=assists_moves(r)
        self.assertEqual([m for m in combined if m['kit_version']!=VERSION],old)
        self.assertEqual(len({m['name'] for m in combined}),len(combined))
        vigil=next(m for m in old if m['name']=="Odin's Vigil Crit")
        self.assertEqual(vigil['original_record'],{'wins':20,'picks':24,'players':7})
        self.assertEqual(assist_display_tier(vigil)['label'],'CRIT I')
        for name in ('Runic Infusion Crit','Rune Conduit Crit'):
            self.assertEqual(assist_display_tier(next(m for m in old if m['name']==name))['label'],'PP UPGRADE')

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
        self.assertIn('PP UPGRADE GATE 8.3',legacy_html)
        self.assertIn('PP UPGRADE',legacy_html)

    def test_shared_display_tiers_and_volume_floor(self):
        cases=[(24,30,'CRIT II'),(23,30,'CRIT I'),(22,30,'PP UPGRADE'),
               (24,29,'MONITOR'),(80,100,'CRIT II'),(75,100,'CRIT I'),(74,100,'PP UPGRADE')]
        for wins,picks,label in cases:
            move={'name':'Legacy Crit','kind':'CRIT','wins':wins,'picks':picks}
            before=move.copy()
            self.assertEqual(assist_display_tier(move)['label'],label)
            self.assertEqual(move,before)
        self.assertEqual(assist_display_name({'name':'Legacy Crit','kind':'CRIT','wins':70,'picks':100}), 'Legacy · PP Upgrade')

    def test_new_enhanced_move_displays_crit_and_outranks_old_low_rate_crit(self):
        from warlords_night_board import _move_rank
        old={'name':'Old Crit','kind':'CRIT','market':'Assists','wins':29,'picks':41,'later_wins':0,'later_picks':0}
        new=next(m for m in new_assists_moves(row()) if m['name']=='Runic Resonance II')
        self.assertEqual(assist_display_tier(new)['label'],'CRIT II')
        self.assertGreater(_move_rank(new),_move_rank(old))

    def test_route_portraits_and_fallback(self):
        from warlords_night_board import _character_uri, _support_variant
        gorilla=_character_uri('Support')
        orangutan=_character_uri('Support','Runesmith')
        chimp=_character_uri('Support','Stormcaller')
        self.assertTrue(gorilla.startswith('data:image/webp;base64,'))
        self.assertEqual(len({gorilla,orangutan,chimp}),3)
        self.assertEqual(_character_uri('Support','Unknown'),gorilla)
        self.assertEqual(_support_variant({'name':'Runic Infusion Crit'}),'Runesmith')
        self.assertEqual(_support_variant({'name':'Rune Conduit'}),'Stormcaller')
        self.assertEqual(_support_variant({'name':"Odin's Vigil Crit"}),'Arcanist')
        self.assertEqual(_character_uri('Support','Oracle + Runesmith'),orangutan)

    def test_actual_assist_values_reach_card_without_inventing_missing_data(self):
        r=row(TOI_per_game=24.8,PP_TOI_per_game=4,Team_PP_Time_PG=6,
              PP_Role=2,Assists_mu=1.27,Team_PP_GF60=7.12,opp_5v5_HDCA60=1.94)
        board=featured_warlords(rank_priced_slate(pd.DataFrame([r])))
        html=render_warlords(board,roles=('Support',))
        for value in ('24.8 min/g','4.0 min/g','66.7%','PP1','1.27','7.12','1.94',
                      'MATCHUP · VALUE / CUTOFF','MOVE SAMPLE AVG BOOK BE','ICE TIME &amp; ROLE'):
            self.assertIn(value,html)
        from warlords_night_board import _assist_card_context, _render_assist_facts
        missing=_assist_card_context({})
        facts=_render_assist_facts({'assist_context':missing},None)
        self.assertIn('Unit unknown',facts)
        self.assertIn('—',facts)
        self.assertNotIn('0.0 min/g',facts)
        changed=_assist_card_context(dict(r,Team_Changed=True))
        self.assertEqual(changed['unit'],'New team · verify unit')

    def test_header_clock_modes_and_safe_text(self):
        from warlord_header import render_header
        for state, label in [('pre','STRIKES IN'),('live','SLATE LIVE'),('post','NEXT STRIKE')]:
            html=render_header(state=state,timer='05:52:37',strike_label='Strike at 6:00 PM CT',status='<unsafe>')
            self.assertIn(label,html)
            self.assertIn('05:52:37',html)
            self.assertIn('&lt;unsafe&gt;',html)
            self.assertEqual(html.count('<header '),1)
            self.assertEqual(html.count('<h1 '),1)
            self.assertIn('max-width:640px',html)
            self.assertNotIn('<img',html)
            self.assertFalse(any(line.startswith('    ') for line in html.splitlines()))


if __name__=='__main__':unittest.main()
