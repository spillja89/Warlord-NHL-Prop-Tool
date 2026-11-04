import ast
from pathlib import Path
import typing
import unittest
import pandas as pd

from sog_confidence import VERSION as CONF_VERSION, sog_confidence_color
from sog_matrix import apply_sog25_green, VERSION as MATRIX_VERSION
from sog_moves_current import sog_moves, VERSION, LEGACY_VERSION, SOURCE
from warlord_moves_2026 import frozen_move_tags

ROOT=Path(__file__).resolve().parents[1]
tree=ast.parse((ROOT/'nhl_edge.py').read_text(encoding='utf-8'))
names={'is_defense','sog_med10_green','sog_med10_yellow','matrix_sog'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
nodes += [n for n in tree.body if isinstance(n,ast.Assign) and any(
    isinstance(t,ast.Name) and t.id.startswith('SOG_MED10_') for t in n.targets)]
env={'Optional':typing.Optional}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual_matrix','exec'),env)

def row(**changes):
    data=dict(Model_Stats_Season='2026-2027',SOG_Line=2.5,Conf_SOG_Version=CONF_VERSION,
        Matrix_SOG='Red',Conf_SOG=71,Conf_SOG_Move=1,Pos='D',
        **{'iXG%':60,'Med10_SOG':3,'ShotIntent_Pct':95,'TOI_Pct':95,
           'team_5v5_SF60_pct':50,'Opp_DefWeak':50},PPO_Source=SOURCE,
        Opp_PKO_GP=10,Opp_PKO_PG=3,Opp_PK_Time_PG=5)
    data.update(changes)
    return data

def revised(data):
    return apply_sog25_green(pd.DataFrame([data]),env['matrix_sog']).iloc[0].to_dict()

class SOGGreenRevisionTests(unittest.TestCase):
    def test_ixg_boundary_and_remaining_shot_checks(self):
        self.assertEqual(revised(row())['Matrix_SOG'],'Green')
        self.assertEqual(revised(row(**{'iXG%':59.999}))['Matrix_SOG'],'Red')
        self.assertEqual(revised(row(Med10_SOG=.5))['Matrix_SOG'],'Red')
        self.assertEqual(revised(row(ShotIntent_Pct=50))['Matrix_SOG'],'Red')

    def test_only_current_primary25_and_complete_new_lane_inputs(self):
        for change in [dict(SOG_Line=3.5),dict(SOG_Line=1.5),dict(Model_Stats_Season='2025-2026'),
                       dict(Conf_SOG_Version='old'),dict(Med10_SOG=None),dict(ShotIntent_Pct=None),
                       {'iXG%':float('nan')},dict(TOI_Pct=None),dict(Opp_DefWeak=None)]:
            self.assertEqual(revised(row(**change))['Matrix_SOG'],'Red',change)

    def test_existing_green_and_elite_exception_stay(self):
        out=revised(row(Matrix_SOG='Green',**{'iXG%':10,'Med10_SOG':None}))
        self.assertEqual(out['Matrix_SOG'],'Green')
        self.assertEqual(out['Matrix_SOG_Route'],'Original Green / Elite')

    def test_confidence_is_independent_and_color_does_not_fire(self):
        out=revised(row(Conf_SOG=66))
        self.assertEqual(out['Matrix_SOG'],'Green')
        self.assertEqual(sog_moves(out),[])
        self.assertEqual(sog_confidence_color(67),'green')
        self.assertEqual(sog_moves(row(Conf_SOG=90,Matrix_SOG='Yellow')),[])

    def test_versioned_records_and_frozen_metadata(self):
        old=sog_moves(row(Matrix_SOG='Green'))
        new=sog_moves(revised(row()))
        self.assertEqual([(m['wins'],m['picks']) for m in old],[(31,52),(27,37),(28,40),(24,31)])
        self.assertEqual([(m['wins'],m['picks']) for m in new],[(50,84),(46,69),(42,64),(38,55)])
        self.assertTrue(all(m['kit_version']==LEGACY_VERSION for m in old))
        self.assertTrue(all(m['kit_version']==VERSION for m in new))
        tags=[m for m in frozen_move_tags(revised(row())) if m['market']=='SOG']
        self.assertTrue(all(m['kit_version']==VERSION for m in tags))

    def test_source_values_preserved_and_revision_idempotent(self):
        before=pd.DataFrame([row()]);after=apply_sog25_green(before,env['matrix_sog'])
        pd.testing.assert_frame_equal(after,apply_sog25_green(after,env['matrix_sog']))
        pd.testing.assert_frame_equal(before.drop(columns='Matrix_SOG'),after[before.columns].drop(columns='Matrix_SOG'))
        self.assertEqual(after.Matrix_SOG_Original.iloc[0],'Red')
        self.assertEqual(after.Matrix_SOG_Version.iloc[0],MATRIX_VERSION)

if __name__=='__main__':unittest.main()
