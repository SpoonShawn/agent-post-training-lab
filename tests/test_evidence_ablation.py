from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from training.evidence_ablation import build, shape, STYLES, training_rows, with_style
from training.evidence_pilot import canonical, expected_report, score
from training.sft import encode_examples
from scripts.prepare_evidence_ablation import bundle, DATA, ROOT
from scripts.evidence_stress_v1 import replay_prefix, resume_ids
from scripts.evidence_ablation_gpu import eval_cases, run_metadata, write_once, evaluate


class TokenizerStub:
    eos_token_id = 999999
    def apply_chat_template(self, messages, **kwargs):
        return canonical(messages)
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


class EvidenceAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = build()

    def test_frozen_and_deterministic(self):
        self.assertEqual(build(), self.rows)
        manifest = bundle()
        self.assertEqual(manifest["training"]["expected_steps"], 24)
        self.assertEqual(Counter(r["split"] for r in self.rows), {"train":192,"validation":48,"confirmation":48})

    def test_structural_split_and_pair_coverage(self):
        shapes = {s:{shape(r) for r in self.rows if r["split"] == s} for s in ("train","validation","confirmation")}
        for a,b in (("train","validation"),("train","confirmation"),("validation","confirmation")):
            self.assertFalse(shapes[a] & shapes[b])
        cases = eval_cases()
        self.assertEqual(len(cases),288)
        self.assertEqual(len({c["id"] for c in cases}),288)
        self.assertTrue(all(v == 3 for v in Counter(c["context_id"] for c in cases).values()))
        self.assertFalse((DATA/"fixed_confirmation.jsonl").exists())

    def test_equal_histories_targets_and_layout_balance(self):
        train = [r for r in self.rows if r["split"] == "train"]
        fixed, mixed = (training_rows(train, role) for role in ("fixed", "mixed"))
        self.assertEqual(Counter(r["mixed_style"] for r in train), {s:64 for s in STYLES})
        for a,b in zip(fixed,mixed):
            self.assertEqual(a["group_id"], b["group_id"])
            self.assertEqual(a["messages"][-1], b["messages"][-1])
        for r in self.rows:
            variants = [with_style(r,s) for s in STYLES]
            self.assertEqual(len({v["query"] for v in variants}),3)
            self.assertTrue(all(v["entries"] == r["entries"] and v["target"] == r["target"] for v in variants))

    def test_replay_every_history_using_past_permission(self):
        for row in self.rows:
            case = deepcopy(row)
            case["read_only"] = case["historical_read_only"]
            replay_prefix(case)

    def test_independent_counts_state_and_latest_evidence(self):
        for row in self.rows:
            target, entries = row["target"], row["entries"]
            tool_errors = sum(not e["result"]["ok"] for e in entries)
            checks = [e["result"]["result"] for e in entries if e["tool_call"]["name"] == "verify_state" and e["result"]["ok"]]
            states = [e["result"]["result"] for e in entries if e["tool_call"]["name"] == "inspect_ui_state" and e["result"]["ok"]]
            logs = [e["result"]["result"] for e in entries if e["tool_call"]["name"] == "query_logs" and e["result"]["ok"]]
            self.assertEqual(target["tool_errors"], tool_errors)
            self.assertEqual(target["total_failures"], tool_errors + sum(not c["success"] for c in checks))
            self.assertEqual(target["last_verification_matches"], checks[-1]["matches"])
            self.assertEqual(target["log_count"], len(logs[-1]))
            self.assertEqual(target["state"], {k:states[-1][k] for k in target["state"]})
            decision = "stop_mutations" if row["read_only"] else {
                "clean":"no_retry_needed", "policy":"no_retry_needed", "arguments":"repair_arguments", "transient":"inspect_before_retry"}[row["family"]]
            self.assertEqual(target["decision"], decision)
            self.assertTrue(score(canonical(target), target)["exact_report"])

    def test_minimal_changes_latest_verify_log_and_current_permission(self):
        row = next(r for r in self.rows if r["family"] == "clean")
        original = row["target"]
        # Append only a successful last verification: counts and state remain fixed.
        entries = deepcopy(row["entries"])
        last = deepcopy(next(e for e in reversed(entries) if e["tool_call"]["name"] == "verify_state"))
        last["tool_call"]["arguments"] = {"expected":{"black_screen":False}}
        last["result"]["result"].update(success=True, matches={"black_screen":True})
        entries.append(last)
        changed = expected_report(entries, row["read_only"])
        self.assertEqual([k for k in original if changed[k] != original[k]], ["last_verification_matches"])
        # Repeat identical latest logs: latest list length does not accumulate.
        logs = next(e for e in reversed(row["entries"]) if e["tool_call"]["name"] == "query_logs")
        self.assertEqual(expected_report(row["entries"]+[logs], row["read_only"]), original)
        changed = expected_report(row["entries"], not row["read_only"])
        self.assertEqual([k for k in original if changed[k] != original[k]], ["decision"])

    def test_only_report_tokens_supervised_and_truncation_rejected(self):
        rows = training_rows(self.rows[:6], "mixed")
        examples = encode_examples(rows, TokenizerStub(), 50000)
        for r,e in zip(rows,examples):
            self.assertEqual([m["role"] for m in r["messages"]], ["system","user","assistant"])
            self.assertEqual([v for v in e["labels"] if v != -100],
                             [ord(c) for c in r["messages"][-1]["content"]]+[999999])
        with self.assertRaises(ValueError):
            encode_examples(rows, TokenizerStub(), 16)

    def test_metadata_write_protect_and_resume_rejection(self):
        c = eval_cases()[0]
        meta = run_metadata({"test":True}, "base")
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/"meta.json"
            write_once(p, meta)
            write_once(p, meta)
            with self.assertRaises(ValueError):
                write_once(p, dict(meta, role="mixed"))
            output = Path(folder)/"base.jsonl"
            row = dict(case=c, answer=canonical(c["target"]), metrics=score(canonical(c["target"]), c["target"]), metadata=meta)
            output.write_text(canonical(row)+"\n")
            self.assertEqual(resume_ids(output, meta, [c]), {c["id"]})
            with self.assertRaises(ValueError):
                resume_ids(output, run_metadata({"test":True}, "mixed"), [c])
            output.write_text((canonical(row)+"\n")*2)
            with self.assertRaises(ValueError):
                resume_ids(output, meta, [c])

    def test_completed_eval_skips_loading_model(self):
        cases = eval_cases()[:1]
        meta = {"test":True}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"base.jsonl"
            c = cases[0]
            path.write_text(canonical(dict(case=c, answer=canonical(c["target"]), metrics=score(canonical(c["target"]), c["target"]), metadata=run_metadata(meta,"base")))+"\n")
            with patch("scripts.evidence_ablation_gpu.RESULTS", Path(folder)), patch("scripts.evidence_ablation_gpu.eval_cases", return_value=cases), patch("agent.baseline_runner.BaselineAgent") as model:
                evaluate("missing-model", meta, "base")
                model.assert_not_called()

    def test_evaluation_writes_real_case_and_only_public_prompt(self):
        c = eval_cases()[0]
        meta = {"test":True}
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda _: "FAKE TEST GPU"))
        with tempfile.TemporaryDirectory() as folder:
            with patch("scripts.evidence_ablation_gpu.RESULTS", Path(folder)), patch("scripts.evidence_ablation_gpu.eval_cases", return_value=[c]), patch("agent.baseline_runner.BaselineAgent") as model, patch.dict("sys.modules", {"torch":fake_torch}):
                model.return_value.generate.return_value = canonical(c["target"])
                evaluate("fake-model", meta, "base")
                messages = model.return_value.generate.call_args.args[0]
                self.assertEqual(messages[1], {"role":"user", "content":c["query"]})
                saved = json.loads((Path(folder)/"base.jsonl").read_text())
                self.assertEqual(saved["case"], c)
                self.assertTrue(saved["metrics"]["exact_report"])
                self.assertEqual(resume_ids(Path(folder)/"base.jsonl",run_metadata(meta,"base"),[c]),{c["id"]})

    def test_shell_stops_at_failure_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"scripts").mkdir()
            script = root/"scripts/superpod_evidence_ablation.sh"
            script.write_text((ROOT/"scripts/superpod_evidence_ablation.sh").read_text())
            fake = root/"python"
            fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$TEST_CALLS\"\ncase \"$*\" in *'--stage base'*) exit 9;; esac\n")
            fake.chmod(0o755)
            log = root/"calls"
            result = subprocess.run(["bash",str(script)], capture_output=True,
                        env=dict(os.environ, PATH=str(root)+os.pathsep+os.environ["PATH"], TEST_CALLS=str(log)))
            self.assertEqual(result.returncode,9)
            self.assertIn("--stage base", log.read_text())
            self.assertNotIn("--stage train_fixed", log.read_text())
