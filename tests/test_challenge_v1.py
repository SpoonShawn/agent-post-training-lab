import unittest
from copy import deepcopy
from collections import Counter

from scripts.prepare_challenge_v1 import build
from scripts.verify_challenge_v1 import verify
from training.pilot_data import oracle
from evaluation.scoring import evaluate_case


class ChallengeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = build()

    def test_frozen_bundle_and_dimension_counts(self):
        verify()
        self.assertEqual(Counter(c["challenge_dimension"] for c in self.cases),
                         dict(expression=40, structure=40, constraints=20, recovery=20))

    def test_all_oracles_and_pending_answers(self):
        for case in self.cases:
            result = oracle(case)["result"]
            score = evaluate_case(case, result)
            self.assertTrue(score["execution_success"], case["id"])
            self.assertIsNone(score["task_success"])

    def test_expression_pairs_preserve_conditions_and_all_clauses(self):
        pairs = {}
        for case in self.cases:
            if case["challenge_dimension"] == "expression":
                pairs.setdefault(case["pair_id"], []).append(case)
        self.assertEqual(len(pairs), 20)
        for first, second in pairs.values():
            for key in ("environment", "required_calls", "max_steps", "public_contract", "success_criteria"):
                self.assertEqual(first[key], second[key])
            for sentence in first["query"].split("。"):
                if sentence:
                    self.assertIn(sentence, second["query"])

    def test_missing_logs_rejected_for_every_case(self):
        for case in self.cases:
            result = oracle(case)["result"]
            result["trajectory"] = [t for t in result["trajectory"] if
                                    t["tool_results"][0]["tool_call"]["name"] != "query_logs"]
            self.assertFalse(evaluate_case(case, result)["execution_success"], case["id"])

    def test_repeated_milestone_cannot_be_skipped(self):
        case = next(c for c in self.cases if c["group_id"] == "structure_length4_home")
        result = oracle(case)["result"]
        changes = [t for t in result["trajectory"] if
                   t["tool_results"][0]["tool_call"]["arguments"].get("action") == "set_graphics"]
        result["trajectory"].remove(changes[2])
        self.assertFalse(evaluate_case(case, result)["execution_success"])

    def test_recovery_without_post_failure_check_rejected(self):
        for case in self.cases:
            if not case["requires_recovery"]:
                continue
            result = oracle(case)["result"]
            i = next(i for i, t in enumerate(result["trajectory"]) if not t["tool_results"][0]["result"]["ok"])
            del result["trajectory"][i + 1]
            self.assertFalse(evaluate_case(case, result)["execution_success"])

    def test_forbidden_actions_rejected_even_if_final_state_restored(self):
        for kind, action in (("read_only", {"name": "execute_action", "arguments": {"action": "back_home"}}),
                             ("no_set_graphics", {"name": "execute_action", "arguments": {"action": "set_graphics", "value": "standard"}})):
            case = next(c for c in self.cases if c["group_id"] == "constraints_" + kind)
            result = oracle(case)["result"]
            extra = deepcopy(result["trajectory"][0])
            extra["parsed_tool_calls"][0]["tool_call"] = action
            extra["tool_results"][0]["tool_call"] = action
            result["trajectory"].insert(1, extra)
            self.assertFalse(evaluate_case(case, result)["execution_success"])
