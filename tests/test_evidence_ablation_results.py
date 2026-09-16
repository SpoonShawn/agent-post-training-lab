from copy import deepcopy
import unittest
from unittest.mock import patch

from scripts.analyze_evidence_ablation import analyze, read_rows


class EvidenceAblationResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary, cls.bad = analyze()

    def test_frozen_results_and_all_failures_retained(self):
        for role, counts in (("base",[8,7,6]),("fixed",[21,13,22]),("mixed",[24,16,43])):
            split = self.summary[role]["splits"]["confirmation"]
            self.assertEqual([split[s]["exact_report"] for s in ("json","records","table")], counts)
            self.assertTrue(all(split[s]["valid_json"] == 48 for s in split))
        self.assertEqual(len(self.bad), 524)

    def test_paired_regressions_not_hidden(self):
        p = self.summary["fixed_to_mixed_paired"]["confirmation"]
        self.assertEqual(p["records"]["1->0"],7)
        self.assertEqual(sum(v.get("0->1",0) for v in p.values()),34)
        self.assertEqual(self.summary["mixed"]["splits"]["confirmation"]["records"]["field_errors"],
                         {"total_failures":32,"verification_failures":32})

    def test_training_budget_and_token_difference(self):
        s = self.summary
        for role in ("fixed","mixed"):
            self.assertEqual(s["training"][role]["trainer_state"]["global_step"],24)
            self.assertEqual(s["preflight"]["token_statistics"][role]["train"]["supervised_tokens"],15960)
        self.assertNotEqual(s["preflight"]["token_statistics"]["fixed"]["train"]["input_tokens"],
                            s["preflight"]["token_statistics"]["mixed"]["train"]["input_tokens"])

    def test_reject_changed_score_and_incomplete_run(self):
        for mutation in ("score", "missing"):
            def changed(path):
                rows = deepcopy(read_rows(path))
                if path.name == "mixed.jsonl":
                    if mutation == "missing":
                        return rows[:-1]
                    rows[0]["metrics"]["exact_report"] = not rows[0]["metrics"]["exact_report"]
                return rows
            with patch("scripts.analyze_evidence_ablation.read_rows", side_effect=changed):
                with self.assertRaises(ValueError):
                    analyze()

    def test_group_denominators_not_independent_layouts(self):
        for role in ("base","fixed","mixed"):
            groups = self.summary["group_confirmation_totals"][role]
            self.assertEqual(len(groups),8)
            self.assertTrue(all(0 <= n <= 18 for n in groups.values()))
        self.assertEqual(self.summary["group_confirmation_totals"]["fixed"]["transient_r1_t3"],13)
        self.assertEqual(self.summary["group_confirmation_totals"]["mixed"]["transient_r1_t3"],12)
