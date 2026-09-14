import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from copy import deepcopy

from evaluation.protocol_v21 import revise_case as revise_v21
from evaluation.protocol_v22 import revise_case, process_metrics
from evaluation.scoring import evaluate_case, dataset_protocol, review_queue
from scripts.generate_benchmark_v2 import build_cases
from tests.test_benchmark_evaluator_integration import build_oracle_record
from scripts.audit_baseline_v21 import main as audit_main
from scripts.summarize_baseline import load_records


class ProcessContractTests(unittest.TestCase):
    def row(self):
        case = revise_case(next(c for c in build_cases() if c["scenario_family"] == "recovery_action"))
        return build_oracle_record(case)

    def check_index(self, row):
        evidence = process_metrics(row["case"], row["result"])["process_evidence"][0]
        return evidence["observation_step"]

    def test_all_360_oracles(self):
        for original in build_cases():
            case = revise_case(original)
            row = build_oracle_record(case)
            self.assertTrue(evaluate_case(case, row["result"])["execution_success"], case["id"])
            self.assertEqual(case["query"], revise_v21(original)["query"])

    def test_missing_check_fails_even_with_good_answer_and_final_state(self):
        row = self.row()
        del row["result"]["trajectory"][self.check_index(row)]
        metrics = evaluate_case(row["case"], row["result"], {"verdict": "pass"})
        self.assertTrue(metrics["execution_success_before_process"])
        self.assertFalse(metrics["execution_success"])
        self.assertFalse(metrics["task_success"])
        self.assertFalse(metrics["recovery_success"])
        old = deepcopy(row["case"])
        old["protocol_version"] = "2.1"
        self.assertTrue(evaluate_case(old, row["result"], {"verdict": "pass"})["task_success"])

    def test_pre_failure_or_after_retry_check_does_not_count(self):
        for destination in (0, -1):
            row = self.row()
            check = row["result"]["trajectory"].pop(self.check_index(row))
            row["result"]["trajectory"].insert(destination, check)
            self.assertFalse(process_metrics(row["case"], row["result"])["process_success"])

    def test_failed_empty_or_wrong_state_check_does_not_count(self):
        for result in ({"ok": False}, {"ok": True, "result": {}},
                       {"ok": True, "result": {"version": "2.3.1"}}):
            row = self.row()
            row["result"]["trajectory"][self.check_index(row)]["tool_results"][0]["result"] = result
            self.assertFalse(process_metrics(row["case"], row["result"])["process_success"])

    def test_verify_current_state_is_equivalent_even_when_expectation_false(self):
        row = self.row()
        entry = row["result"]["trajectory"][self.check_index(row)]["tool_results"][0]
        state = entry["result"]["result"]
        entry["tool_call"] = {"name": "verify_state", "arguments": {"expected": {"current_page": "home"}}}
        entry["result"] = {"ok": True, "result": {"success": False, "current_state": state}}
        self.assertTrue(process_metrics(row["case"], row["result"])["process_success"])

    def test_same_turn_check_and_retry_cannot_react_to_feedback(self):
        row = self.row()
        index = self.check_index(row)
        row["result"]["trajectory"][index]["tool_results"].extend(
            row["result"]["trajectory"].pop(index + 1)["tool_results"])
        self.assertFalse(process_metrics(row["case"], row["result"])["process_success"])

    def test_unchecked_first_retry_cannot_be_erased_by_later_check(self):
        row = self.row()
        index = self.check_index(row)
        early = deepcopy(row["result"]["trajectory"][index + 1])
        row["result"]["trajectory"].insert(index, early)
        self.assertFalse(process_metrics(row["case"], row["result"])["process_success"])

    def test_contract_validation_and_mixed_versions(self):
        row = self.row()
        for rules in (None, [], ["unknown"]):
            case = dict(row["case"], process_constraints=rules)
            with self.assertRaises(ValueError):
                process_metrics(case, row["result"])
        with self.assertRaises(ValueError):
            dataset_protocol([row["case"], {"protocol_version": "2.1"}])

    def test_v22_queue_version(self):
        row = self.row()
        row["metrics"] = evaluate_case(row["case"], row["result"])
        self.assertEqual(review_queue([row], "source")[0]["protocol_version"], "2.2")

    def test_absent_failure_and_absent_retry_fail(self):
        row = self.row()
        row["result"]["trajectory"] = []
        self.assertIn("required_injected_failure_not_observed",
                      process_metrics(row["case"], row["result"])["process_violations"])
        row = self.row()
        index = self.check_index(row)
        row["result"]["trajectory"] = row["result"]["trajectory"][:index + 1]
        self.assertIn("retry_missing", process_metrics(row["case"], row["result"])["process_violations"])

    def test_native_loader_recomputes_v22_instead_of_trusting_cached_success(self):
        row = self.row()
        del row["result"]["trajectory"][self.check_index(row)]
        row["metrics"] = {"task_success": True}
        row["evaluator_version"] = "2.2"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertFalse(load_records(path)[0]["metrics"]["task_success"])

    def test_real_retrospective_audit_and_cross_version_overwrite_guard(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            args = ["audit", "--protocol", "2.2", "--output-dir", tmp, "--reviews",
                    str(root / "results/reviews/v21_adjudicated_reviews.jsonl")]
            with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                audit_main()
            summary = json.loads((Path(tmp) / "summary.json").read_text())
            overall = summary["eligible_summary"]["overall"]
            self.assertEqual(overall["num_cases"], 352)
            self.assertEqual(overall["task_pass_count"], 188)
            self.assertEqual(overall["task_unresolved_count"], 0)
            self.assertEqual(summary["by_category"]["recovery"]["execution_success"], 33)
            args[2] = "2.1"
            with patch("sys.argv", args), self.assertRaisesRegex(ValueError, "overwrite"):
                audit_main()


if __name__ == "__main__":
    unittest.main()
