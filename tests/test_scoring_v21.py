import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from evaluation.evaluator import evaluate_case as evaluate_v2
from evaluation.protocol_v21 import revise_case
from evaluation.scoring import (
    dataset_protocol, evaluate_case, load_reviews, record_sha256,
    review_queue, summarize_results,
)
from scripts.generate_benchmark_v2 import build_cases
from scripts.audit_baseline_v21 import main as audit_main
from scripts.run_baseline import build_run_metadata, main as run_main, parse_args
from scripts.summarize_baseline import load_records, main as summarize_main
from tests.test_benchmark_evaluator_integration import build_oracle_record


class ScoringTests(unittest.TestCase):
    def row(self, family="navigate_graphics"):
        case = revise_case(next(c for c in build_cases() if c["scenario_family"] == family))
        row = build_oracle_record(case)
        row["metrics"] = evaluate_case(case, row["result"])
        return row

    def test_v2_unchanged(self):
        row = self.row()
        row["case"].pop("protocol_version")
        self.assertEqual(evaluate_case(row["case"], row["result"]), evaluate_v2(row["case"], row["result"]))

    def test_changed_input_cannot_reuse_old_inference(self):
        row = self.row()
        row["result"]["query"] = "different prompt"
        with self.assertRaisesRegex(ValueError, "fresh inference"):
            evaluate_case(row["case"], row["result"])

    def test_unknown_or_mixed_protocol_rejected(self):
        for cases in ([{"protocol_version": "3"}], [{}, {"protocol_version": "2.1"}]):
            with self.assertRaises(ValueError):
                dataset_protocol(cases)

    def test_pending_is_not_averaged_away(self):
        pending = self.row()
        failed = self.row("reproduce_black_screen")
        failed["result"]["trajectory"] = []
        failed["metrics"] = evaluate_case(failed["case"], failed["result"])
        summary = summarize_results([pending, failed])
        self.assertIsNone(summary["overall"]["task_success_rate"])
        self.assertEqual(summary["overall"]["task_unresolved_count"], 1)
        self.assertEqual(summary["overall"]["task_success_upper_bound"], 0.5)
        self.assertIsNone(summary["category_macro"]["task_success_rate"])
        self.assertEqual(summary["by_category"]["long_horizon"]["task_success_rate"], 0)

    def test_reviews_resolve_only_successful_execution(self):
        row = self.row()
        for verdict, expected in (("pass", True), ("fail", False), ("uncertain", None)):
            metrics = evaluate_case(row["case"], row["result"], {"verdict": verdict})
            self.assertIs(metrics["task_success"], expected)
        row["result"]["final_environment_state"]["current_page"] = "home"
        self.assertFalse(evaluate_case(row["case"], row["result"], {"verdict": "pass"})["task_success"])

    def test_recovery_execution_is_not_recovery_task_success(self):
        row = self.row("recovery_navigation")
        self.assertTrue(row["metrics"]["recovery_execution_success"])
        self.assertIsNone(row["metrics"]["recovery_success"])
        self.assertIsNone(summarize_results([row])["overall"]["recovery_success_rate"])

    def test_complete_review_publishes_rate(self):
        row = self.row()
        row["metrics"] = evaluate_case(row["case"], row["result"], {"verdict": "pass"})
        summary = summarize_results([row])
        self.assertEqual(summary["overall"]["task_success_rate"], 1)
        self.assertEqual(summary["category_macro"]["task_success_rate"], 1)

    def test_review_binding_and_provenance(self):
        row = self.row()
        review = review_queue([row], "source-hash")[0]
        self.assertNotIn("metrics", review)
        self.assertIn("tool_evidence", review)
        review.update(verdict="pass", reason="Verified final state and answer", reviewer="test",
                      reviewer_type="protocol_author_ai", reviewed_at="2026-09-10T00:00:00Z")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reviews.jsonl"
            path.write_text(json.dumps(review) + "\n")
            self.assertIn(row["case"]["id"], load_reviews(path, [row], "source-hash"))
            for field, value in (("source_sha256", "stale"), ("record_sha256", "stale"),
                                 ("verdict", "pending"), ("reviewer", ""),
                                 ("protocol_version", "2.0"), ("id", "unknown")):
                bad = dict(review, **{field: value})
                path.write_text(json.dumps(bad) + "\n")
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_reviews(path, [row], "source-hash")
            path.write_text((json.dumps(review) + "\n") * 2)
            with self.assertRaises(ValueError):
                load_reviews(path, [row], "source-hash")
        changed = deepcopy(row)
        changed["result"]["final_answer"] += " changed"
        self.assertNotEqual(record_sha256(row["case"], row["result"]), record_sha256(changed["case"], changed["result"]))

    def test_runner_metadata_and_summarizer_dispatch(self):
        row = self.row()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.jsonl"
            path.write_text(json.dumps(row["case"]) + "\n")
            args = parse_args(["--eval-path", str(path), "--model-path", "remote-model"])
            with patch("scripts.run_baseline.sha256_model_path", return_value=(None, None)):
                metadata = build_run_metadata(args, [row["case"]])
            self.assertEqual(metadata["evaluator_version"], "2.1")
            row.update(evaluator_version="2.1", run_metadata=metadata)
            path.write_text(json.dumps(row) + "\n")
            loaded = load_records(path, recompute=True)
            self.assertIsNone(loaded[0]["metrics"]["task_success"])
            queue = Path(tmp) / "queue.jsonl"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                summarize_main(["--input-path", str(path), "--recompute", "--review-queue", str(queue)])
            self.assertIn("v2.1 Summary", output.getvalue())
            self.assertIn("Task answers unresolved: 1", output.getvalue())
            self.assertEqual(len(queue.read_text().splitlines()), 1)
            row["evaluator_version"] = "2.0"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                load_records(path)

    def test_runner_does_not_overwrite_experiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.jsonl"
            path.write_text("original")
            with self.assertRaises(FileExistsError):
                run_main(["--output-path", str(path)])
            self.assertEqual(path.read_text(), "original")

    def test_real_baseline_audit_and_review_import(self):
        source = Path(__file__).resolve().parents[1] / "results/baseline/qwen3_4b_baseline_v2.jsonl"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.assertEqual(digest, "0f7d39563c3b319ed1bf38e43a1108a2288914834c79f2aa22cd7b4bc603601c")
        with tempfile.TemporaryDirectory() as tmp:
            args = ["audit_baseline_v21.py", "--input", str(source), "--output-dir", tmp]
            with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                audit_main()
            directory = Path(tmp)
            summary = json.loads((directory / "summary.json").read_text())
            overall = summary["eligible_summary"]["overall"]
            self.assertEqual(overall["num_cases"], 352)
            self.assertEqual(overall["task_unresolved_count"], 206)
            self.assertIsNone(overall["task_success_rate"])
            self.assertEqual(len((directory / "rerun_v21.jsonl").read_text().splitlines()), 8)
            queue = [json.loads(line) for line in (directory / "answer_review_queue.jsonl").read_text().splitlines()]
            self.assertEqual(len(queue), 206)
            # Synthetic verdict tests plumbing only; it is never persisted as a real review.
            review = queue[0]
            review.update(verdict="pass", reason="test fixture only", reviewer="unit-test",
                          reviewer_type="synthetic_test", reviewed_at="2026-09-10T00:00:00Z")
            review_path = directory / "test_reviews.jsonl"
            review_path.write_text(json.dumps(review) + "\n")
            with patch("sys.argv", args + ["--reviews", str(review_path)]), contextlib.redirect_stdout(io.StringIO()):
                audit_main()
            summary = json.loads((directory / "summary.json").read_text())
            overall = summary["eligible_summary"]["overall"]
            self.assertEqual(overall["task_unresolved_count"], 205)
            self.assertEqual(overall["task_pass_count"], 1)
            self.assertIsNone(overall["task_success_rate"])
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
