import json
from pathlib import Path
import tempfile
import unittest

from training.transaction_cache import DiskExamples, build_cache, latest_checkpoint
from scripts.transaction_full_sft import CONFIG, SPLITS, inference_metadata
from scripts.analyze_transaction_smoke import audit


class FullSFTTests(unittest.TestCase):
    def test_uploaded_smoke_replays(self):
        summary, failures = audit()
        self.assertEqual(len(failures), 28)
        for role in ("base", "sft_smoke"):
            self.assertEqual(summary["roles"][role]["cases"], 14)
            self.assertEqual(summary["roles"][role]["task_success"], 0)
            self.assertEqual(summary["roles"][role]["execution_success"], 2)

    def test_disk_cache_exact_roundtrip_and_bounds(self):
        example = dict(input_ids=[1, 2], attention_mask=[1, 1], labels=[-100, 2])
        stats = dict(examples=1, total_input_tokens_including_targets=2, supervised_tokens=1, max_tokens=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.sqlite"
            data = build_cache(path, iter([example]), stats)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0], example)
            with self.assertRaises(IndexError):
                data[1]
            with self.assertRaises(FileExistsError):
                build_cache(path, iter([example]), stats)
            data.connection.close()
            again = DiskExamples(path)
            self.assertEqual(again[0], example)
            again.connection.close()

    def test_bad_token_counts_never_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.sqlite"
            with self.assertRaises(ValueError):
                build_cache(path, iter([]), {})
            self.assertFalse(path.exists())

    def test_resume_requires_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / "checkpoint-500"
            partial.mkdir()
            (partial / "trainer_state.json").write_text(json.dumps(dict(global_step=500)))
            self.assertIsNone(latest_checkpoint(root))
            complete = root / "checkpoint-250"
            complete.mkdir()
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors", "adapter_config.json"):
                (complete / name).touch()
            (complete / "trainer_state.json").write_text(json.dumps(dict(global_step=250)))
            self.assertEqual(latest_checkpoint(root), str(complete))
            (complete / "trainer_state.json").write_text(json.dumps(dict(global_step=249)))
            self.assertIsNone(latest_checkpoint(root))

    def test_frozen_budget_and_test_separation(self):
        self.assertEqual(CONFIG["expected_steps"], (31542 + 7) // 8)
        self.assertEqual(CONFIG["epochs"], 1)
        self.assertEqual(SPLITS, ("dev", "confirmation_id", "confirmation_ood"))


if __name__ == "__main__":
    unittest.main()
