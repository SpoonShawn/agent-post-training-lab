import json
from copy import deepcopy
from pathlib import Path
import unittest
import argparse
import tempfile

from evaluation.protocol_v23 import revise_case
from evaluation.scoring import evaluate_case
from scripts.generate_benchmark_v2 import build_cases as old_cases
from scripts.audit_contracts_v22 import simulate
from scripts.verify_training_bundle import verify
from tests.test_benchmark_evaluator_integration import build_oracle_record
from training.pilot_data import build_cases, oracle
from training.sft import encode_examples
from scripts.run_baseline import build_run_metadata


class ContractV23Tests(unittest.TestCase):
    def test_all_360_legacy_diagnostic_oracles_and_new_prompts(self):
        for original in old_cases():
            case = revise_case(original)
            result = build_oracle_record(case)["result"]
            self.assertTrue(evaluate_case(case, result)["execution_success"], case["id"])
            self.assertNotEqual(case["query"], original["query"])
            self.assertEqual(case["split"], "development_audit")
            if any(r["tool"] == "query_logs" for r in case["success_criteria"]["required_tool_results"]):
                self.assertIn("必须查询本轮操作日志", case["query"])
            result["query"] = original["query"]
            with self.assertRaises(ValueError):
                evaluate_case(case, result)

    def test_order_and_no_navigation_counterexamples_now_fail(self):
        case = revise_case(next(c for c in old_cases() if c["id"] == "reproduce_black_screen_001"))
        calls = [c for c in case["required_calls"] if c["name"] != "search_knowledge"]
        calls += [c for c in case["required_calls"] if c["name"] == "search_knowledge"]
        self.assertFalse(simulate(case, calls)["execution_success"])
        case = revise_case(next(c for c in old_cases() if c["id"] == "navigate_graphics_005"))
        calls = [{"name": "execute_action", "arguments": {"action": "back_home"}},
                 {"name": "navigate_ui", "arguments": {"target_page": "settings"}},
                 {"name": "navigate_ui", "arguments": {"target_page": "graphics"}}] + case["required_calls"]
        self.assertFalse(simulate(case, calls)["execution_success"])

    def test_equivalent_build_observation_now_passes(self):
        case = revise_case(next(c for c in old_cases() if c["id"] == "build_info_001"))
        self.assertTrue(simulate(case, [{"name": "inspect_ui_state", "arguments": {}}])["execution_success"])

    def test_no_redundant_set_and_read_only_attempts_rejected(self):
        for phrase in ("不重复改设置", "不要改变界面"):
            case = revise_case(next(c for c in old_cases() if phrase in c["query"]))
            row = build_oracle_record(case)
            extra = {"name": "execute_action", "arguments": {"action": "set_graphics", "value": "standard"}}
            row["result"]["trajectory"].insert(0, {
                "parsed_tool_calls": [{"valid": True, "tool_call": extra}], "tool_results": []})
            self.assertFalse(evaluate_case(case, row["result"])["execution_success"])


class TrainingPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = build_cases()

    def test_group_isolation_and_counts(self):
        groups = {}
        for split, count in (("train", 320), ("validation", 80), ("confirmation", 80)):
            subset = [c for c in self.cases if c["split"] == split]
            self.assertEqual(len(subset), count)
            groups[split] = {c["group_id"] for c in subset}
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["confirmation"])
        self.assertFalse(groups["validation"] & groups["confirmation"])

    def test_all_independent_oracles_and_milestone_negative(self):
        for case in self.cases:
            self.assertTrue(oracle(case)["execution_validated"])
        case = next(c for c in self.cases if not c["requires_recovery"])
        calls = deepcopy(case["required_calls"])
        indices = [i for i, c in enumerate(calls) if c["arguments"].get("action") == "set_graphics"]
        calls[indices[0]], calls[indices[1]] = calls[indices[1]], calls[indices[0]]
        self.assertFalse(simulate(case, calls)["execution_success"])

    def test_saved_bundle_and_no_legacy_exact_query(self):
        verify()
        legacy = {c["query"] for c in old_cases()}
        self.assertFalse(legacy & {c["query"] for c in self.cases})
        module = Path("training/pilot_data.py").read_text()
        self.assertNotIn("generate_benchmark_v2", module)
        self.assertNotIn("results/baseline", module)

    def test_supervision_masks_all_context_and_keeps_eos(self):
        class Tokenizer:
            eos_token_id = 99
            def apply_chat_template(self, messages, **kwargs):
                return "context"
            def encode(self, text, **kwargs):
                return [1, 2, 3] if text == "context" else [4, 5]
        row = {"id": "example", "messages": [{"role": "user", "content": "question"},
                                           {"role": "assistant", "content": "answer"}]}
        example = encode_examples([row], Tokenizer(), 10)[0]
        self.assertEqual(example["labels"], [-100, -100, -100, 4, 5, 99])
        with self.assertRaisesRegex(ValueError, "Overlength"):
            encode_examples([row], Tokenizer(), 5)

    def test_adapter_content_is_part_of_inference_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            adapter = directory / "adapter"
            adapter.mkdir()
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"version1")
            path = directory / "eval.jsonl"
            case = self.cases[0]
            path.write_text(json.dumps(case) + "\n")
            args = argparse.Namespace(model_path="nonlocal-model", adapter_path=adapter,
                                      eval_path=path, max_new_tokens=512, max_steps=None)
            first = build_run_metadata(args, [case])
            weights.write_bytes(b"version2")
            second = build_run_metadata(args, [case])
            self.assertNotEqual(first["fingerprint"], second["fingerprint"])
