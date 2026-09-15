import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_control_results import analyze, read_source, review_summary


class ControlResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparison, cls.reviews, cls.evidence, cls.failures, cls.summary = analyze()

    def test_replay_and_coverage(self):
        self.assertEqual(self.comparison["base"]["replayed_calls"], 805)
        self.assertEqual(self.comparison["sft"]["replayed_calls"], 753)
        self.assertEqual(len(self.evidence), 94)
        self.assertEqual(len(self.failures), 58)

    def test_primary_pairs_keep_regressions(self):
        self.assertEqual(self.comparison["sft"]["pairs"][
            "readonly_explicit -> readonly_checklist"], {"1->0": 4, "1->1": 4})
        self.assertEqual(self.comparison["base"]["pairs"][
            "reordered_explicit -> reordered_checklist"]["1->0"], 1)

    def test_review_denominators_and_uncertainty(self):
        b, s = self.summary["base"]["overall"], self.summary["sft"]["overall"]
        self.assertEqual((b["task_pass"], b["task_fail"], b["task_pending"]), (9, 66, 1))
        self.assertIsNone(b["task_rate"])
        self.assertEqual((s["task_pass"], s["task_fail"], s["task_pending"]), (62, 14, 0))
        self.assertEqual(s["task_rate"], 62/76)
        self.assertEqual(self.summary["sft"]["readonly_explicit"]["task_pass"], 1)

    def test_sources_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "changed.jsonl"
            path.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "Changed source"):
                read_source("base", path)

    def test_review_coverage_rejected(self):
        rows = [dict(case=e["case"], result=e["result"], metrics=e["metrics"])
                for e in self.evidence if e["model"] == "sft"]
        with self.assertRaisesRegex(ValueError, "coverage"):
            review_summary(rows, self.reviews["sft"][:-1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            review_summary(rows, self.reviews["sft"] + self.reviews["sft"][:1])

    def test_persisted_evidence_and_labels_reproduce(self):
        root = Path(__file__).resolve().parents[1]
        for role, expected in self.reviews.items():
            actual = [json.loads(s) for s in (
                root / f"results/reviews/control_v1/{role}_author_reviews.jsonl").read_text().splitlines()]
            self.assertEqual(actual, expected)
            self.assertTrue(all(r["reviewer_type"] == "protocol_author_ai" for r in actual))


if __name__ == "__main__":
    unittest.main()
