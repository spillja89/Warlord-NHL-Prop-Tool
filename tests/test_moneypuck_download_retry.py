"""Exercise the live engine loader with interrupted and invalid HTTP bodies."""
import ast
from datetime import date
from io import StringIO
from pathlib import Path
import types
import unittest
from unittest.mock import Mock

import pandas as pd
import requests


class MoneyPuckDownloadTests(unittest.TestCase):
    def setUp(self):
        source=(Path(__file__).resolve().parents[1]/'nhl_edge.py').read_text(encoding='utf-8')
        tree=ast.parse(source)
        names={'load_moneypuck_csv','load_moneypuck_best_effort',
               'current_season_start_year','moneypuck_url'}
        functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
        self.fetch=Mock()
        self.sleep=Mock()
        self.ns=dict(requests=requests,pd=pd,StringIO=StringIO,
                     http_get_text=self.fetch,time=types.SimpleNamespace(sleep=self.sleep),
                     date=date,CURRENT_SEASON_CUTOVER=date(2026,10,13))
        exec(compile(ast.Module(body=functions,type_ignores=[]),'nhl_edge.py','exec'),self.ns)
        self.load=self.ns['load_moneypuck_csv']
        self.url='https://moneypuck.com/moneypuck/playerData/seasonSummary/2026/regular/skaters.csv'

    def test_interrupted_body_restarts_complete_download(self):
        self.fetch.side_effect=[requests.exceptions.ChunkedEncodingError('Response ended prematurely'),
                                'playerId,team\n8471,MIN\n']
        result=self.load(object(),self.url)
        self.assertEqual(result.to_dict('records'),[{'playerId':8471,'team':'MIN'}])
        self.assertEqual(self.fetch.call_count,2)
        self.sleep.assert_called_once_with(.5)

    def test_parse_error_retries_and_discards_partial_body(self):
        self.fetch.side_effect=['playerId,team\n"unfinished', 'playerId,team\n8472,TOR\n']
        result=self.load(object(),self.url)
        self.assertEqual(result.iloc[0]['team'],'TOR')
        self.assertEqual(len(result),1)

    def test_three_failed_downloads_fail_and_never_use_prior_season(self):
        self.fetch.side_effect=requests.exceptions.ChunkedEncodingError('cut off')
        with self.assertRaisesRegex(RuntimeError,'current-season skaters.csv'):
            self.ns['load_moneypuck_best_effort'](object(),'skaters',as_of=date(2026,11,3))
        self.assertEqual(self.fetch.call_count,3)
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list],[.5,1.5])
        self.assertTrue(all(call.args[1]==self.url for call in self.fetch.call_args_list))

    def test_empty_and_html_responses_are_not_accepted_as_stats(self):
        self.fetch.side_effect=['', '<html>Temporary unavailable</html>', 'playerId,team\n1,MIN\n']
        self.assertEqual(len(self.load(object(),self.url)),1)
        self.assertEqual(self.fetch.call_count,3)

    def test_http_403_does_not_retry_but_503_does(self):
        for status,count in [(403,1),(503,3)]:
            self.fetch.reset_mock()
            response=requests.Response(); response.status_code=status
            self.fetch.side_effect=requests.exceptions.HTTPError('HTTP failure',response=response)
            with self.assertRaises(requests.exceptions.HTTPError):
                self.load(object(),self.url)
            self.assertEqual(self.fetch.call_count,count)
