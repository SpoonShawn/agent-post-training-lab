from copy import deepcopy
import unittest
from unittest.mock import Mock

from agent.guarded_runtime import guarded_execute, facts, render_facts, run_guarded
from tools.environment_tools import reset_environment, inspect_ui_state


def entry(name, result, **args):
    return {"tool_call": {"name": name, "arguments": args}, "result": result}


STATE = dict(current_page="home", graphics_preset="standard",
             battle_hud_visible=False, black_screen=False)


class GuardTests(unittest.TestCase):
    def test_blocks_before_dispatch(self):
        executor = Mock()
        for name in ["navigate_ui", "execute_action", "new_tool", None, []]:
            result = guarded_execute({"name": name}, read_only=True, executor=executor)
            self.assertEqual(result["error_type"], "policy_blocked")
        executor.assert_not_called()

    def test_read_tools_and_policy_validation(self):
        executor = Mock(return_value={"ok": True, "result": []})
        self.assertTrue(guarded_execute({"name": "query_logs"}, read_only=True, executor=executor)["ok"])
        with self.assertRaises(ValueError):
            guarded_execute({}, read_only="false")

    def test_no_hidden_state_or_invented_completion(self):
        data = facts([])
        self.assertEqual(len(data["missing_fields"]), 4)
        self.assertIn("未知", render_facts(data))
        self.assertEqual(data["completion_verdict"], "not_assigned")

    def test_errors_verifications_empty_logs(self):
        trajectory = [{"step": 0, "tool_results": [
            entry("execute_action", {"ok": False, "error_type": "policy_blocked"}),
            entry("verify_state", {"ok": True, "result": {"success": False, "current_state": STATE}}),
            entry("query_logs", {"ok": True, "result": []})]}]
        data = facts(trajectory)
        self.assertEqual(data["failure_count"], 2)
        self.assertEqual(data["policy_block_count"], 1)
        self.assertEqual(data["missing_fields"], [])
        self.assertTrue(data["logs_after_last_attempt"])
        self.assertEqual(data["successful_inspect_calls"], 0)

    def test_stale_state_invalidation_and_kb_exclusion(self):
        trajectory = [{"tool_results": [
            entry("inspect_ui_state", {"ok": True, "result": STATE}),
            entry("navigate_ui", {"ok": True, "result": {}}),
            entry("search_knowledge", {"ok": True, "result": STATE})]}]
        self.assertEqual(len(facts(trajectory)["missing_fields"]), 4)
        trajectory[0]["tool_results"][1]["result"] = {"ok": False}
        self.assertEqual(facts(trajectory)["missing_fields"], [])

    def test_batched_calls_cannot_bypass_guard(self):
        replies = iter([
            '<tool_call>{"name":"navigate_ui","arguments":{"target_page":"settings"}}</tool_call>'
            '<tool_call>{"name":"execute_action","arguments":{"action":"back_home"}}</tool_call>',
            '<tool_call>{"name":"inspect_ui_state","arguments":{}}</tool_call>'
            '<tool_call>{"name":"query_logs","arguments":{}}</tool_call>',
            "原始模型声明保留"])
        messages_seen = []
        def generate(messages):
            messages_seen.append(deepcopy(messages))
            return next(replies)
        result = run_guarded(generate, "只读", None, 3)
        self.assertEqual(result["final_answer"], "原始模型声明保留")
        self.assertEqual(facts(result["trajectory"])["policy_block_count"], 2)
        self.assertEqual(facts(result["trajectory"])["successful_mutations"], [])
        self.assertIn("policy_blocked", str(messages_seen[1]))

    def test_successful_action_state_counts_without_inspect(self):
        data = facts([{"tool_results": [entry("execute_action", {"ok": True, "result": STATE})]}])
        self.assertEqual(data["missing_fields"], [])
        self.assertEqual(data["successful_inspect_calls"], 0)

    def test_wrong_types_not_published(self):
        state = dict(STATE, black_screen=0)
        data = facts([{"tool_results": [entry("inspect_ui_state", {"ok": True, "result": state})]}])
        self.assertIn("black_screen", data["missing_fields"])

    def test_all_real_control_fact_counts_and_observations(self):
        from scripts.analyze_control_results import read_source, ROOT
        from scripts.audit_answer_evidence import audit
        for role in ["base", "sft"]:
            for row in read_source(role, ROOT / f"results/baseline/control_v1_{role}.jsonl"):
                actual = facts(row["result"]["trajectory"])
                self.assertEqual(actual["failure_count"], audit(row)["observed_failure_count"])
                expected = row["metrics"]["observation_evidence"]["evidence"]
                self.assertEqual({k: v["value"] for k, v in actual["observed_state"].items()},
                                 {k: v["value"] for k, v in expected.items()})
                self.assertEqual(actual["logs_after_last_attempt"], row["metrics"]["log_timing_success"])

    def test_first_forbidden_calls_in_four_sft_regressions(self):
        from scripts.analyze_control_results import read_source, ROOT
        seen = 0
        for row in read_source("sft", ROOT / "results/baseline/control_v1_sft.jsonl"):
            if row["case"]["control_arm"] != "readonly_checklist" or row["metrics"]["execution_success"]:
                continue
            reset_environment(row["case"]["environment"])
            for turn in row["result"]["trajectory"]:
                blocked = False
                for e in turn.get("tool_results", []):
                    before = inspect_ui_state()
                    value = guarded_execute(e["tool_call"], read_only=True)
                    if value.get("error_type") == "policy_blocked":
                        self.assertEqual(before, inspect_ui_state())
                        seen += 1
                        blocked = True
                        break
                    self.assertEqual(value, e["result"])
                if blocked:
                    break
        self.assertEqual(seen, 4)

    def test_frozen_guard_spec(self):
        from scripts.run_guard_pilot import verify_spec, specification
        verify_spec()
        self.assertEqual(len(specification()["case_ids"]), 16)

    def test_gpu_entrypoint_write_and_resume_with_fake_model(self):
        import itertools
        import json
        from pathlib import Path
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts import run_guard_pilot as runner
        from scripts.verify_control_v1 import verify
        _, cases = verify()
        case = next(c for c in cases if c["control_arm"] == "readonly_explicit")
        replies = itertools.cycle([
            '<tool_call>{"name":"inspect_ui_state","arguments":{}}</tool_call>'
            '<tool_call>{"name":"query_logs","arguments":{}}</tool_call>', "保留原回答"])
        model = Mock()
        model.generate.side_effect = lambda messages: next(replies)
        cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
                               get_device_name=lambda i: "FAKE TEST GPU",
                               get_device_properties=lambda i: SimpleNamespace(total_memory=1))
        fake_torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda="test"))
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(runner, "ROOT", Path(folder)), patch.object(runner, "verify_spec"), \
                 patch.object(runner, "verify", return_value=({}, [case])), \
                 patch.object(runner, "metadata_for", return_value={"test": True}), \
                 patch.object(runner, "digest", return_value="test_digest"), \
                 patch("agent.baseline_runner.BaselineAgent", return_value=model) as constructor, \
                 patch.dict("sys.modules", {"torch": fake_torch}), \
                 patch("sys.argv", ["run_guard_pilot", "--role", "base", "--resume"]):
                runner.main()
                runner.main()
                constructor.assert_called_once()
            rows = (Path(folder) / "results/baseline/guard_pilot_v1_base.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 1)
            row = json.loads(rows[0])
            self.assertEqual(row["result"]["final_answer"], "保留原回答")
            self.assertTrue(row["metrics"]["execution_success"])
            self.assertEqual(row["system_facts"]["successful_inspect_calls"], 1)
