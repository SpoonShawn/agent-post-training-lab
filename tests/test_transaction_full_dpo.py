import math
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_transaction_dpo_gate import audit
from scripts.transaction_full_dpo import order, CONFIG
from training.dpo_checkpoint import seal, latest


class FullDPOTests(unittest.TestCase):
    def test_gate_math_recomputes(self):
        result, reference = audit()
        self.assertEqual(len(reference), 98)
        self.assertAlmostEqual(result["steps"][0]["loss"], math.log(2))
        self.assertEqual(len(result["probes"]), 4)
        self.assertTrue(all(p["rejected_logp_delta"] < 0 for p in result["probes"].values()))
        self.assertEqual(sum(p["chosen_logp_delta"] < 0 for p in result["probes"].values()), 2)

    def test_fixed_train_only_order(self):
        examples = [dict(case_id=str(i), split="preference_train" if i < 80 else "preference_dev") for i in range(98)]
        self.assertEqual(order(examples), order(list(reversed(examples))))
        self.assertEqual(len(order(examples)), 80)
        self.assertTrue(all(p["split"] == "preference_train" for p in order(examples)))
        self.assertEqual(CONFIG["initialization"], "original_full_SFT_not_gate")

    def test_checkpoint_integrity_partial_and_wrong_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"checkpoint-20-partial").mkdir()
            self.assertIsNone(latest(root, "plan"))
            folder = root/"checkpoint-10-complete"
            (folder/"adapter").mkdir(parents=True)
            (folder/"adapter/adapter_model.safetensors").write_bytes(b"test-only-not-real-weights")
            (folder/"optimizer.pt").write_bytes(b"test-only-not-real-state")
            seal(folder, dict(plan_sha256="plan", step=10, steps=[]))
            self.assertEqual(latest(root, "plan")[0], folder)
            with self.assertRaises(ValueError):
                latest(root, "another-plan")
            (folder/"optimizer.pt").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                latest(root, "plan")


if __name__ == "__main__":
    unittest.main()
