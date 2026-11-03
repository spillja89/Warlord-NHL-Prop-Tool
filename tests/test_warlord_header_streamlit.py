"""Exercise the masthead through Streamlit's actual HTML element."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


class HeaderStreamlitTests(unittest.TestCase):
    def test_direct_html_renderer(self):
        app=AppTest.from_string('''
import streamlit as st
from warlord_header import render_header
st.html(render_header(state='pre', timer='05:37:44', strike_label='Strike at 6 PM CT', status='VENGEANCE IS COMING'))
''').run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.get('html')),1)
        body=app.get('html')[0].proto.body
        self.assertIn('<h1 class="wv-title">THE WARLORD’S',body)
        self.assertIn('05:37:44',body)
        self.assertNotIn('<img',body)
        self.assertEqual(len(app.code),0)
        source=(Path(__file__).resolve().parents[1]/'app.py').read_text(encoding='utf-8')
        self.assertIn('st.html(render_header(',source)
        self.assertNotIn('st.markdown(render_header(',source)

