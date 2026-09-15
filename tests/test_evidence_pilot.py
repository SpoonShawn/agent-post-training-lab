from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.evidence_pilot import build, trajectory, score, canonical, expected_report
from training.sft import encode_examples


class TokenizerStub:
    eos_token_id = 999
    def apply_chat_template(self, messages, **kwargs):
        return canonical(messages)
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


class EvidencePilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = build()

    def test_group_isolation_and_counts(self):
        self.assertEqual(Counter(c["split"] for c in self.cases),
                         {"train":192, "validation":48, "confirmation":48})
        groups = {}
        for c in self.cases:
            groups.setdefault(c["split"], set()).add(c["group_id"])
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["confirmation"])
        self.assertFalse(groups["validation"] & groups["confirmation"])
        queries = {s: {c["query"] for c in self.cases if c["split"] == s} for s in groups}
        self.assertFalse(queries["train"] & queries["confirmation"])

    def test_independent_counts_from_raw_returns(self):
        for c in self.cases:
            entries = c["entries"]
            errors = sum(e["result"]["ok"] is False for e in entries)
            checks = sum(e["result"]["ok"] is True and e["tool_call"]["name"]=="verify_state"
                         and e["result"]["result"]["success"] is False for e in entries)
            t = c["target"]
            self.assertEqual((t["tool_errors"],t["verification_failures"],t["total_failures"]),
                             (errors,checks,errors+checks))
            self.assertEqual(t["policy_blocks"], sum(
                e["result"].get("error_type")=="policy_blocked" for e in entries))
            self.assertTrue(score(canonical(t), t)["exact_report"])

    def test_decision_error_distinction(self):
        expected = {"clean":"no_retry_needed", "negative_check":"no_retry_needed",
                    "bad_arguments":"repair_arguments", "transient_once":"inspect_before_retry",
                    "transient_twice":"inspect_before_retry", "policy_once":"stop_mutations",
                    "policy_twice":"stop_mutations", "mixed_policy_check":"stop_mutations"}
        for c in self.cases:
            self.assertEqual(c["target"]["decision"], expected[c["pattern"]])

    def test_permissions_override_retry(self):
        c = next(c for c in self.cases if c["pattern"] == "transient_once")
        self.assertEqual(expected_report(c["entries"], True)["decision"], "stop_mutations")

    def test_no_failed_action_supervision(self):
        rows = [trajectory(c) for c in self.cases if c["split"] == "train"]
        self.assertTrue(all([m["role"] for m in r["messages"]] == ["system","user","assistant"] for r in rows))
        examples = encode_examples(rows, TokenizerStub(), 20000)
        self.assertEqual(len(examples), 192)
        for r,e in zip(rows,examples):
            target = [ord(c) for c in r["messages"][-1]["content"]]+[999]
            self.assertEqual([v for v in e["labels"] if v != -100], target)

    def test_no_silent_truncation(self):
        with self.assertRaisesRegex(ValueError, "Overlength"):
            encode_examples([trajectory(self.cases[0])], TokenizerStub(), 32)

    def test_strict_json_negatives(self):
        t = self.cases[0]["target"]
        bad = deepcopy(t)
        bad["total_failures"] += 1
        self.assertFalse(score(canonical(bad),t)["exact_report"])
        bad = deepcopy(t)
        bad["state"]["black_screen"] = 0
        self.assertFalse(score(canonical(bad),t)["exact_report"])
        self.assertFalse(score('{"decision":"a","decision":"b"}',t)["valid_json"])
        self.assertFalse(score("说明："+canonical(t),t)["exact_report"])
        bad = deepcopy(t)
        bad["extra"] = "unsupported"
        self.assertFalse(score(canonical(bad),t)["exact_report"])

    def test_adapter_digest_uses_existing_helper(self):
        from scripts.evidence_pilot_gpu import sha256_adapter
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/"adapter"
            with self.assertRaises(ValueError):
                sha256_adapter(p)
            p.mkdir()
            (p/"adapter_config.json").write_text("{}")
            self.assertEqual(len(sha256_adapter(p)),64)

    def test_launcher_stage_order_and_fail_stop(self):
        import os
        import subprocess
        from scripts.prepare_control_v1 import ROOT
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"scripts").mkdir()
            (root/"scripts"/"superpod_evidence_pilot.sh").write_text(
                (ROOT/"scripts/superpod_evidence_pilot.sh").read_text())
            fake = root/"python"
            fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$TEST_CALLS\"\ncase \"$*\" in *'--stage old_sft'*) exit 9;; esac\n")
            fake.chmod(0o755)
            log = root/"calls"
            env = dict(os.environ, PATH=str(root)+os.pathsep+os.environ["PATH"], TEST_CALLS=str(log))
            result = subprocess.run(["bash",str(root/"scripts/superpod_evidence_pilot.sh")],
                                    env=env, capture_output=True)
            self.assertNotEqual(result.returncode,0)
            text = log.read_text()
            self.assertIn("--stage base",text)
            self.assertIn("--stage old_sft",text)
            self.assertNotIn("--stage train",text)

    def test_dataset_deterministic(self):
        self.assertEqual(build(),self.cases)
