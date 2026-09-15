from copy import deepcopy
import unittest

from evaluation.protocol_v24 import evaluate_case, validate_case
from evaluation.scoring import evaluate_case as score_v23
from scripts.prepare_control_v1 import build, oracle_result


class Protocol24Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = build()

    def readonly(self):
        return next(c for c in self.cases if c["control_arm"] == "readonly_explicit")

    def test_76_oracles_pass_execution_not_answers(self):
        self.assertEqual(len(self.cases), 76)
        for case in self.cases:
            metrics = evaluate_case(case, oracle_result(case))
            self.assertTrue(metrics["execution_success"], case["id"])
            self.assertIsNone(metrics["task_success"])

    def test_hidden_final_state_cannot_replace_observation(self):
        case = self.readonly()
        result = oracle_result(case)
        result["trajectory"] = [t for t in result["trajectory"]
                                if t["tool_results"][0]["tool_call"]["name"] == "query_logs"]
        legacy = deepcopy(case)
        legacy["protocol_version"] = "2.3"
        self.assertTrue(score_v23(legacy, result)["execution_success"])
        metrics = evaluate_case(case, result)
        self.assertFalse(metrics["execution_success"])
        self.assertIn("final_state_observation_missing_or_mismatched", metrics["observation_violations"])

    def test_verify_and_action_returns_are_equivalent(self):
        case = self.readonly()
        result = oracle_result(case)
        result["trajectory"] = [t for t in result["trajectory"]
                                if t["tool_results"][0]["tool_call"]["name"] != "inspect_ui_state"]
        action = {"name": "verify_state", "arguments": {"expected": {}}}
        result["trajectory"].append({
            "step": 10, "parsed_tool_calls": [{"valid": True, "tool_call": action}],
            "tool_results": [{"tool_call": action, "result": {
                "ok": True, "result": {"success": True, "current_state": result["final_environment_state"]}}}]})
        self.assertTrue(evaluate_case(case, result)["execution_success"])
        moving = next(c for c in self.cases if c["control_arm"] == "original_explicit")
        result = oracle_result(moving)
        # Last successful action returns complete final state; no extra inspect required.
        result["trajectory"].pop()
        self.assertTrue(evaluate_case(moving, result)["execution_success"])

    def test_missing_and_early_logs_fail(self):
        case = next(c for c in self.cases if c["control_arm"] == "original_explicit")
        for early in (False, True):
            result = oracle_result(case)
            logs = [t for t in result["trajectory"] if t["tool_results"][0]["tool_call"]["name"] == "query_logs"]
            result["trajectory"] = [t for t in result["trajectory"] if t not in logs]
            if early:
                result["trajectory"].insert(0, logs[0])
            metrics = evaluate_case(case, result)
            self.assertFalse(metrics["log_timing_success"])
            self.assertFalse(metrics["execution_success"])

    def test_all_log_removals_rejected(self):
        for case in self.cases:
            result = oracle_result(case)
            result["trajectory"] = [t for t in result["trajectory"]
                                    if t["tool_results"][0]["tool_call"]["name"] != "query_logs"]
            self.assertFalse(evaluate_case(case, result)["execution_success"])

    def test_unknown_contract_and_old_protocol_rejected(self):
        for edit in ("version", "field", "missing", "duplicate", "boolean"):
            case = deepcopy(self.readonly())
            if edit == "version":
                case["protocol_version"] = "2.3"
            elif edit == "field":
                case["observation_contract"]["unexpected"] = True
            elif edit == "missing":
                del case["observation_contract"]
            elif edit == "duplicate":
                case["observation_contract"]["final_state_fields"] *= 2
            else:
                case["observation_contract"]["logs_after_last_mutation"] = "true"
            with self.assertRaises(ValueError):
                validate_case(case)

    def test_changed_query_cannot_reuse_result(self):
        case = self.readonly()
        result = oracle_result(case)
        result["query"] = "old input"
        with self.assertRaises(ValueError):
            evaluate_case(case, result)

    def test_failed_last_application_attempt_also_requires_new_logs(self):
        case = next(c for c in self.cases if c["control_arm"] == "original_explicit")
        result = oracle_result(case)
        action = {"name": "navigate_ui", "arguments": {"target_page": "settings"}}
        result["trajectory"].append({
            "step": 99, "parsed_tool_calls": [{"valid": True, "tool_call": action}],
            "tool_results": [{"tool_call": action, "result": {"ok": False, "error": "illegal"}}]})
        self.assertFalse(evaluate_case(case, result)["log_timing_success"])
