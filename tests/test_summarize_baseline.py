import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_baseline import (
    load_records,
    print_text_summary,
    summarize_coverage,
)


def record(case_id, fingerprint="run-a", evaluator_version="2.0"):
    return {
        "evaluator_version": evaluator_version,
        "run_metadata": {"fingerprint": fingerprint},
        "case": {"id": case_id, "expected_tools": []},
        "result": {
            "trajectory": [],
            "final_answer": "完成",
        },
        "metrics": {"task_success": True},
    }


def write_records(path, records):
    Path(path).write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )


class SummarizerInputValidationTests(unittest.TestCase):
    def test_one_consistent_run_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            write_records(path, [record("a"), record("b")])

            loaded = load_records(path)

            self.assertEqual([item["case"]["id"] for item in loaded], ["a", "b"])

    def test_mixed_run_fingerprints_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            write_records(path, [
                record("a", fingerprint="run-a"),
                record("b", fingerprint="run-b"),
            ])

            with self.assertRaisesRegex(ValueError, "不同运行指纹"):
                load_records(path)

    def test_duplicate_case_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            write_records(path, [record("same"), record("same")])

            with self.assertRaisesRegex(ValueError, "case 重复"):
                load_records(path)

    def test_malformed_empty_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            write_records(path, [{}])

            with self.assertRaisesRegex(ValueError, "case 或 result"):
                load_records(path)

    def test_v1_records_without_metadata_remain_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            legacy = record("legacy")
            legacy.pop("run_metadata")
            legacy.pop("evaluator_version")
            write_records(path, [legacy])

            loaded = load_records(path, recompute=True)

            self.assertEqual(loaded[0]["case"]["id"], "legacy")
            self.assertIn("ordered_tool_match", loaded[0]["metrics"])

    def test_partial_run_coverage_is_reported_and_warned(self):
        rows = [record("a"), record("b")]
        for row in rows:
            row["run_metadata"]["selected_case_count"] = 5

        coverage = summarize_coverage(rows)

        self.assertEqual(coverage["completed_case_count"], 2)
        self.assertEqual(coverage["selected_case_count"], 5)
        self.assertEqual(coverage["coverage_rate"], 0.4)
        self.assertFalse(coverage["is_complete"])

        summary = {
            "overall": {"num_cases": 2},
            "category_macro": {},
            "by_category": {},
            "run_coverage": coverage,
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_text_summary(summary)
        self.assertIn("Run coverage: 2/5", output.getvalue())
        self.assertIn("PARTIAL RESULT", output.getvalue())

    def test_inconsistent_or_excess_coverage_is_rejected(self):
        inconsistent = [record("a"), record("b")]
        inconsistent[0]["run_metadata"]["selected_case_count"] = 2
        inconsistent[1]["run_metadata"]["selected_case_count"] = 3
        with self.assertRaisesRegex(ValueError, "不一致"):
            summarize_coverage(inconsistent)

        excessive = [record("a"), record("b")]
        for row in excessive:
            row["run_metadata"]["selected_case_count"] = 1
        with self.assertRaisesRegex(ValueError, "超过"):
            summarize_coverage(excessive)


if __name__ == "__main__":
    unittest.main()
