import unittest
from copy import deepcopy

from evaluation.protocol_v21 import evaluate_execution, revise_case
from scripts.generate_benchmark_v2 import build_cases
from tests.test_benchmark_evaluator_integration import build_oracle_record


class ProtocolTests(unittest.TestCase):
    def case(self, family):
        return next(case for case in build_cases() if case["scenario_family"] == family)

    def test_navigation_accepts_action_state_evidence(self):
        case = self.case("navigate_graphics")
        result = build_oracle_record(case)["result"]
        result["trajectory"] = [step for step in result["trajectory"] if not any(
            call["tool_call"]["name"] in {"inspect_ui_state", "verify_state"}
            for call in step["parsed_tool_calls"]
        )]
        metrics = evaluate_execution(case, result)
        self.assertTrue(metrics["execution_success"])
        self.assertIsNone(metrics["task_success"])

    def test_missing_requested_logs_still_fails(self):
        case = self.case("reproduce_black_screen")
        result = build_oracle_record(case)["result"]
        result["trajectory"] = [step for step in result["trajectory"] if not any(
            call["tool_call"]["name"] == "query_logs"
            for call in step["parsed_tool_calls"]
        )]
        self.assertFalse(evaluate_execution(case, result)["execution_success"])

    def test_wrong_answer_is_never_auto_approved(self):
        case = self.case("negative_control_ios")
        result = build_oracle_record(case)["result"]
        result["final_answer"] = "已复现黑屏，加载失败。"
        metrics = evaluate_execution(case, result)
        self.assertTrue(metrics["execution_success"])
        self.assertIsNone(metrics["task_success"])
        self.assertEqual(metrics["answer_review"], "pending")

    def test_prompt_repairs_preserve_original_and_require_inference(self):
        originals = build_cases()
        snapshot = deepcopy(originals)
        changed = [revise_case(case) for case in originals if "用事件编号" in case["query"]]
        self.assertTrue(changed)
        self.assertTrue(all(case["requires_new_inference"] for case in changed))
        self.assertTrue(all("INC-" in case["query"] for case in changed))
        self.assertEqual(originals, snapshot)

    def test_search_text_not_counted_as_structured_argument_error(self):
        case = self.case("knowledge_black_screen")
        result = build_oracle_record(case)["result"]
        self.assertIsNone(evaluate_execution(case, result)["structured_argument_accuracy"])

    def test_all_oracles_have_valid_execution(self):
        for case in build_cases():
            with self.subTest(case=case["id"]):
                self.assertTrue(evaluate_execution(case, build_oracle_record(case)["result"])["execution_success"])


if __name__ == "__main__":
    unittest.main()
