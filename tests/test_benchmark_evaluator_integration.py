import unittest

from evaluation.evaluator import aggregate_results, evaluate_case
from scripts.generate_benchmark_v2 import build_cases
from tools.environment_tools import inspect_ui_state, reset_environment
from tools.executor import execute_tool_json


def build_oracle_record(case):
    reset_environment(case["environment"])
    trajectory = []

    for step, call in enumerate(case["required_calls"]):
        outcome = execute_tool_json(call)
        trajectory.append({
            "step": step,
            "model_output": "",
            "parsed_tool_calls": [{
                "valid": True,
                "tool_call": call,
            }],
            "tool_results": [{
                "tool_call": call,
                "result": outcome,
            }],
        })

    answer_spec = case["success_criteria"]["final_answer"]
    answer_parts = list(answer_spec.get("contains_all", []))
    if answer_spec.get("contains_any"):
        answer_parts.append(answer_spec["contains_any"][0])
    final_answer = "；".join(answer_parts) or "任务完成。"
    trajectory.append({
        "step": len(trajectory),
        "model_output": final_answer,
        "parsed_tool_calls": [],
        "tool_results": [],
    })

    result = {
        "query": case["query"],
        "trajectory": trajectory,
        "final_answer": final_answer,
        "final_environment_state": inspect_ui_state(),
        "terminated_reason": "final_answer",
    }
    return {
        "case": case,
        "result": result,
        "metrics": evaluate_case(case, result),
    }


class BenchmarkEvaluatorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = [build_oracle_record(case) for case in build_cases()]

    @classmethod
    def tearDownClass(cls):
        reset_environment()

    def test_all_360_oracles_pass_task_and_call_criteria(self):
        for record in self.records:
            case_id = record["case"]["id"]
            metrics = record["metrics"]
            self.assertTrue(metrics["task_success"], case_id)
            self.assertEqual(metrics["tool_precision"], 1.0, case_id)
            self.assertEqual(metrics["tool_recall"], 1.0, case_id)
            if metrics["argument_accuracy"] is not None:
                self.assertEqual(metrics["argument_accuracy"], 1.0, case_id)
            self.assertEqual(metrics["unexpected_failed_calls"], 0, case_id)

    def test_recovery_oracles_trigger_and_recover_from_planned_fault(self):
        recovery = [
            record
            for record in self.records
            if record["case"]["requires_recovery"]
        ]
        self.assertEqual(len(recovery), 60)
        for record in recovery:
            metrics = record["metrics"]
            self.assertTrue(metrics["recovery_attempted"], record["case"]["id"])
            self.assertTrue(metrics["recovery_success"], record["case"]["id"])
            self.assertEqual(metrics["observed_expected_failed_calls"], 1)
            self.assertGreaterEqual(metrics["retry_calls"], 1)

    def test_perfect_oracles_have_no_redundant_repeats(self):
        for record in self.records:
            self.assertEqual(
                record["metrics"]["repeated_tool_calls"],
                0,
                record["case"]["id"],
            )

    def test_aggregate_reports_full_oracle_success(self):
        summary = aggregate_results(self.records)["overall"]
        self.assertEqual(summary["num_cases"], 360)
        self.assertEqual(summary["task_success_rate"], 1.0)
        self.assertEqual(summary["long_horizon_success_rate"], 1.0)
        self.assertEqual(summary["recovery_success_rate"], 1.0)
        self.assertEqual(summary["tool_precision"], 1.0)
        self.assertEqual(summary["tool_recall"], 1.0)
        self.assertEqual(summary["argument_accuracy"], 1.0)
        self.assertEqual(summary["total_expected_failed_calls"], 60)
        self.assertEqual(summary["total_unexpected_failed_calls"], 0)


if __name__ == "__main__":
    unittest.main()
