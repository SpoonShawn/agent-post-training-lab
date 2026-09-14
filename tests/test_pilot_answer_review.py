import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.record_pilot_v1_review import ROOT, check_source, check_sft_claims
from scripts.summarize_baseline import load_records


class PilotAnswerReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_records(ROOT / 'results/baseline/pilot_v1_sft_confirmation.jsonl', True)

    def test_reviewed_sft_claims_have_evidence(self):
        for row in self.rows:
            check_sft_claims(row)

    def test_changed_text_requires_review(self):
        row = deepcopy(self.rows[0])
        row['result']['final_answer'] += '所有版本都已验证有效。'
        with self.assertRaises(ValueError):
            check_sft_claims(row)

    def test_missing_recovery_inspection_rejected(self):
        row = deepcopy(next(r for r in self.rows if r['case']['requires_recovery']))
        trajectory = row['result']['trajectory']
        i = next(i for i, t in enumerate(trajectory)
                 if any(not e['result']['ok'] for e in t.get('tool_results', [])))
        del trajectory[i + 1]
        with self.assertRaises(ValueError):
            check_sft_claims(row)

    def test_source_changes_block_historical_verdicts(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'source'
            path.write_text('changed')
            with self.assertRaises(ValueError):
                check_source(path, '0' * 64)
