import copy
import json
import unittest

from scripts.analyze_evidence_stress import analyze, ARMS, RESULTS, transitions, validate_rows


class EvidenceStressResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary, cls.bad = analyze()
        cls.rows = [json.loads(l) for l in (RESULTS / "base.jsonl").read_text().splitlines()]

    def test_frozen_counts(self):
        for role, counts in (("base", [15, 8, 0, 4]), ("new_sft", [24, 13, 0, 7])):
            self.assertEqual([self.summary[role]["arms"][a]["exact_report"] for a in ARMS], counts)
            self.assertTrue(all(self.summary[role]["arms"][a]["valid_json"] == 24 for a in ARMS))
            self.assertEqual(self.summary[role]["original_answer_byte_reproduction"], 24)
        self.assertEqual(len(self.bad), 121)

    def test_semantic_failures_and_paired_regressions(self):
        self.assertEqual(self.summary["new_sft"]["arms"]["current_policy_flip"]["field_errors"], {"decision": 17})
        self.assertEqual(self.summary["new_sft"]["arms"]["extended_history"]["field_errors"]["total_failures"], 15)
        self.assertEqual(self.summary["base_to_new_paired"]["surface"]["1->0"], 5)
        self.assertEqual(self.summary["new_sft"]["flip_wrong_decisions_equal_original_answer"], 10)

    def test_reject_missing_duplicate_or_changed_cases(self):
        cases = [r["case"] for r in self.rows]
        for rows in (self.rows[:-1], self.rows + self.rows[:1], list(reversed(self.rows))):
            with self.assertRaises(ValueError):
                validate_rows(rows, cases, self.rows[0]["metadata"])

    def test_reject_metadata_score_and_time_drift(self):
        for field in ("metadata", "metrics", "started_at"):
            rows = copy.deepcopy(self.rows)
            if field == "started_at":
                rows[0][field] = "2099-01-01T00:00:00+00:00"
            else:
                rows[0][field] = {}
            with self.assertRaises(ValueError):
                validate_rows(rows, [r["case"] for r in self.rows], self.rows[0]["metadata"])

    def test_reject_unpaired_comparison(self):
        with self.assertRaises(ValueError):
            transitions({"one": self.rows[0]}, {})
