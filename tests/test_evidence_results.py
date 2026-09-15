import unittest
from scripts.analyze_evidence_pilot import analyze


class EvidenceResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary, cls.bad = analyze()

    def test_frozen_metrics_and_training(self):
        s = self.summary
        self.assertEqual(s["base"]["splits"]["confirmation"]["exact_report"],29)
        self.assertEqual(s["base"]["splits"]["validation"]["exact_report"],32)
        for split in ("validation","confirmation"):
            self.assertEqual(s["new_sft"]["splits"][split]["exact_report"],48)
            self.assertEqual(s["old_sft"]["splits"][split]["valid_json"],0)
        self.assertEqual(s["training"]["run"]["trainer_state"]["global_step"],24)
        self.assertEqual(len(self.bad),131)

    def test_format_and_semantic_errors_separate(self):
        s = self.summary["base"]["splits"]
        self.assertEqual(sum(s[k]["only_policy_blocks_empty_list"] for k in s),12)
        self.assertEqual(sum(s[k]["field_errors_among_valid_json"]["decision"] for k in s),5)
        self.assertTrue(all(b["wrong_fields"]==["non_json_report"]
                            for b in self.bad if b["role"]=="old_sft"))

    def test_paired_improvements_not_regressions(self):
        self.assertEqual(self.summary["base_to_new_paired"]["validation"],{"0->1":16,"1->1":32})
        self.assertEqual(self.summary["base_to_new_paired"]["confirmation"],{"0->1":19,"1->1":29})
