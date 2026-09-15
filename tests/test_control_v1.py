from collections import Counter
from copy import deepcopy
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_control_v1 import ROOT, build, oracle_result, CHECKLIST, PUBLIC_CONTRACT, digest
from scripts.verify_control_v1 import verify
from scripts.run_control_v1 import parse_args, run_cases, validate_resume, metadata_for
from scripts.summarize_control_v1 import summarize, load_records


class ControlStudyTests(unittest.TestCase):
    def test_frozen_counts_and_prompt_isolation(self):
        manifest, cases = verify()
        self.assertEqual(manifest["contexts"], 28)
        self.assertEqual(Counter(c["control_arm"] for c in cases),
                         dict(original_explicit=20, reordered_explicit=20, reordered_checklist=20,
                              readonly_explicit=8, readonly_checklist=8))
        contexts = {}
        for case in cases:
            self.assertIn(PUBLIC_CONTRACT, case["query"])
            contexts.setdefault(case["context_id"], {})[case["control_arm"]] = case
        for variants in contexts.values():
            group = list(variants.values())
            for c in group[1:]:
                for key in ("environment", "required_calls", "max_steps", "success_criteria", "observation_contract"):
                    self.assertEqual(c[key], group[0][key])
            if "reordered_checklist" in variants:
                self.assertEqual(variants["reordered_checklist"]["query"],
                                 variants["reordered_explicit"]["query"] + CHECKLIST)
            else:
                self.assertEqual(variants["readonly_checklist"]["query"],
                                 variants["readonly_explicit"]["query"] + CHECKLIST)

    def test_fake_agent_end_to_end_records_score_and_resume(self):
        cases = build()[:3]
        class FakeAgent:
            def run(self, query, environment, max_steps):
                case = next(c for c in cases if c["query"] == query)
                assert environment == case["environment"]
                assert max_steps == case["max_steps"]
                return oracle_result(case)
        metadata = {"fingerprint": "test"}
        output = io.StringIO()
        run_cases(FakeAgent(), cases, metadata, output)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertTrue(all(r["metrics"]["execution_success"] for r in rows))
        self.assertTrue(all(r["metrics"]["task_success"] is None for r in rows))
        self.assertEqual(summarize(rows)["coverage"], "3/76")
        self.assertFalse(summarize(rows)["complete"])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "results.jsonl"
            path.write_text(output.getvalue())
            self.assertEqual(validate_resume(path, metadata, cases), {c["id"] for c in cases})
            with self.assertRaises(ValueError):
                validate_resume(path, {"fingerprint": "other"}, cases)
            rows[0]["case"]["query"] = "tampered"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaises(ValueError):
                validate_resume(path, metadata, cases)

    def test_model_roles_do_not_silently_mix(self):
        self.assertIsNone(parse_args(["--role", "base"]).adapter_path)
        self.assertIsNotNone(parse_args(["--role", "sft"]).adapter_path)
        with self.assertRaises(SystemExit):
            parse_args(["--role", "base", "--adapter-path", "unexpected"])

    def test_dedicated_reader_checks_partial_result_provenance(self):
        manifest, cases = verify()
        meta = dict(evaluator_version="2.4", role="base",
                    extension_sha256=manifest["extension_sha256"],
                    runtime_sha256=manifest["legacy_runtime_sha256"],
                    benchmark_sha256=manifest["cases_sha256"],
                    model_content_sha256=manifest["model_sha256"],
                    generation=manifest["generation"], selected_case_count=76,
                    study_manifest_sha256=digest(ROOT / "data/control_v1/manifest.json"),
                    training_library_versions=manifest["versions"])
        meta["fingerprint"] = hashlib.sha256(
            json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        row = dict(evaluator_version="2.4", run_metadata=meta, case=cases[0], result=oracle_result(cases[0]))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "partial.jsonl"
            path.write_text(json.dumps(row) + "\n")
            summary = summarize(load_records(path))
            self.assertEqual(summary["coverage"], "1/76")
            self.assertFalse(summary["complete"])
            row["run_metadata"]["fingerprint"] = "invalid"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                load_records(path)

    def test_legacy_reader_refuses_new_protocol_instead_of_misscoring(self):
        from scripts.summarize_baseline import load_records as legacy_load
        case = build()[0]
        row = dict(case=case, result=oracle_result(case), evaluator_version="2.4")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "new_protocol.jsonl"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                legacy_load(path, True)

    def test_preflight_binds_versions_and_weights_without_loading_model(self):
        manifest, cases = verify()
        args = parse_args(["--role", "sft", "--preflight-only"])
        fake_meta = {"model_content_sha256": manifest["model_sha256"],
                     "adapter_sha256": manifest["adapter_sha256"], "fingerprint": "old"}
        with patch("scripts.run_control_v1.build_run_metadata", return_value=fake_meta.copy()), \
             patch("scripts.run_control_v1.package_version", side_effect=lambda k: manifest["versions"][k]):
            meta = metadata_for(args, cases, manifest)
        self.assertEqual(meta["evaluator_version"], "2.4")
        self.assertEqual(meta["role"], "sft")
        self.assertNotEqual(meta["fingerprint"], "old")
        self.assertTrue(all(c["protocol_version"] == "2.4" for c in cases))
        with patch("scripts.run_control_v1.build_run_metadata", return_value=fake_meta.copy()), \
             patch("scripts.run_control_v1.package_version", return_value="wrong"):
            with self.assertRaises(ValueError):
                metadata_for(args, cases, manifest)

    def test_wrong_weights_rejected_before_agent_creation(self):
        manifest, cases = verify()
        with patch("scripts.run_control_v1.build_run_metadata",
                   return_value={"model_content_sha256": "wrong"}):
            with self.assertRaises(ValueError):
                metadata_for(parse_args(["--role", "base"]), cases, manifest)
