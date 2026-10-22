import json
import unittest
from unittest.mock import patch

import typesafe_shadow


class TypeSafeShadowTests(unittest.TestCase):
    def test_all_markets_exclude_settled_and_other_sensitive_fields(self):
        row = {
            "Date": "2026-10-22", "Player": "Example Player", "Team": "STL",
            "Goal_mu": 0.45, "PP_TOI_per_game": 3.2,
            "Outcome": "W", "Result": "2", "Fired_Moves": "Example move",
            "WARLORD_ADMIN_PASSWORD": "never-send", "TYPESAFE_API_KEY": "never-send",
        }
        for market in typesafe_shadow.MARKETS:
            with self.subTest(market=market):
                state = typesafe_shadow.pregame_state(row, market)
                self.assertEqual(state["Player"], "Example Player")
                self.assertNotIn("Outcome", state)
                self.assertNotIn("Result", state)
                self.assertNotIn("Fired_Moves", state)
                self.assertNotIn("TYPESAFE_API_KEY", state)
                self.assertNotIn("WARLORD_ADMIN_PASSWORD", state)

    def test_single_request_uses_official_typed_choice_format(self):
        row = {"Player": "Example Player", "Goal_mu": 0.45, "Goal_Line": 0.5}
        response = {
            "model": "jev-latest",
            "answers": {name: {"type": "choice", "choice": "unknown", "confidence": 0.7}
                        for name in ("opportunity", "matchup", "evidence")},
            "usage": {"input_tokens": 100, "output_tokens": 15},
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, *_args):
                return json.dumps(response).encode()

        with patch.object(typesafe_shadow, "urlopen", return_value=FakeResponse()) as call:
            result = typesafe_shadow.review(row, "Goals", "test-key")
        request = call.call_args.args[0]
        sent = json.loads(request.data)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(sent["model"], "jev-latest")
        self.assertEqual(sent["state"]["Player"], "Example Player")
        self.assertEqual(set(sent["questions"]), {"opportunity", "matchup", "evidence"})
        self.assertEqual(result["answers"]["opportunity"]["choice"], "unknown")
        self.assertNotIn("test-key", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
