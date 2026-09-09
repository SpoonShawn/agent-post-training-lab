import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_baseline import (
    build_run_metadata,
    load_cases,
    load_completed_ids,
)


class RunnerResumeSafetyTests(unittest.TestCase):
    def make_metadata(self, directory, model_path="model-a"):
        eval_path = Path(directory) / "eval.jsonl"
        case = {
            "schema_version": 2,
            "id": "case-1",
            "query": "测试",
        }
        eval_path.write_text(
            json.dumps(case, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        args = argparse.Namespace(
            model_path=model_path,
            eval_path=eval_path,
            max_new_tokens=128,
            max_steps=None,
        )
        return case, build_run_metadata(args, [case])

    def test_matching_fingerprint_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            case, metadata = self.make_metadata(directory)
            output = Path(directory) / "result.jsonl"
            output.write_text(
                json.dumps({
                    "run_metadata": metadata,
                    "case": case,
                    "result": {},
                    "metrics": {},
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            completed = load_completed_ids(
                output,
                expected_fingerprint=metadata["fingerprint"],
            )

            self.assertEqual(completed, {"case-1"})

    def test_changed_model_is_rejected_before_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            case, metadata = self.make_metadata(directory, "model-a")
            _, changed = self.make_metadata(directory, "model-b")
            output = Path(directory) / "result.jsonl"
            output.write_text(
                json.dumps({
                    "run_metadata": metadata,
                    "case": case,
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "指纹不一致"):
                load_completed_ids(
                    output,
                    expected_fingerprint=changed["fingerprint"],
                )

    def test_benchmark_and_generation_config_affect_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            case, original = self.make_metadata(directory)
            eval_path = Path(directory) / "eval.jsonl"
            changed_case = {**case, "query": "已经修改的测试"}
            eval_path.write_text(
                json.dumps(changed_case, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                model_path="model-a",
                eval_path=eval_path,
                max_new_tokens=256,
                max_steps=8,
            )
            changed = build_run_metadata(args, [changed_case])

            self.assertNotEqual(
                original["benchmark_sha256"],
                changed["benchmark_sha256"],
            )
            self.assertNotEqual(
                original["fingerprint"],
                changed["fingerprint"],
            )

    def test_local_model_content_affects_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"
            model_dir.mkdir()
            weights = model_dir / "weights.bin"
            weights.write_bytes(b"first checkpoint")
            case, original = self.make_metadata(directory, str(model_dir))

            weights.write_bytes(b"other checkpoint")
            eval_path = Path(directory) / "eval.jsonl"
            args = argparse.Namespace(
                model_path=str(model_dir),
                eval_path=eval_path,
                max_new_tokens=128,
                max_steps=None,
            )
            changed = build_run_metadata(args, [case])

            self.assertNotEqual(
                original["model_content_sha256"],
                changed["model_content_sha256"],
            )
            self.assertNotEqual(
                original["fingerprint"],
                changed["fingerprint"],
            )

    def test_resume_discards_only_an_incomplete_tail_record(self):
        with tempfile.TemporaryDirectory() as directory:
            case, metadata = self.make_metadata(directory)
            output = Path(directory) / "result.jsonl"
            valid_line = json.dumps({
                "run_metadata": metadata,
                "case": case,
            }, ensure_ascii=False).encode("utf-8") + b"\n"
            output.write_bytes(valid_line + b'{"case": {"id": "cut')

            with contextlib.redirect_stdout(io.StringIO()):
                completed = load_completed_ids(
                    output,
                    expected_fingerprint=metadata["fingerprint"],
                )

            self.assertEqual(completed, {"case-1"})
            self.assertEqual(output.read_bytes(), valid_line)

    def test_duplicate_benchmark_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.jsonl"
            case = {"id": "duplicate", "query": "测试"}
            path.write_text(
                json.dumps(case, ensure_ascii=False) + "\n"
                + json.dumps(case, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "id 重复"):
                load_cases(path)

    def test_legacy_output_without_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.jsonl"
            output.write_text(
                json.dumps({"case": {"id": "legacy"}}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "缺少 run_metadata"):
                load_completed_ids(output, expected_fingerprint="expected")


if __name__ == "__main__":
    unittest.main()
