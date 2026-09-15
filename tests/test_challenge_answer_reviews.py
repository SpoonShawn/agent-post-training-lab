import hashlib
import unittest

from scripts.record_challenge_v1_reviews import ROOT, decision, SOURCES
from scripts.summarize_baseline import load_records
from scripts.audit_answer_evidence import audit
from evaluation.scoring import load_reviews, evaluate_case


class ChallengeAnswerReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = {m: load_records(ROOT / f'results/baseline/challenge_v1_{m}.jsonl', True)
                    for m in SOURCES}

    def test_bound_review_counts_and_task_results(self):
        for mode, expected in [('base', 6), ('sft', 59)]:
            rows = self.runs[mode]
            path = ROOT / f'results/baseline/challenge_v1_{mode}.jsonl'
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), SOURCES[mode])
            reviews = load_reviews(ROOT / f'results/reviews/challenge_v1/{mode}_author_reviews.jsonl', rows, SOURCES[mode])
            scores = [evaluate_case(r['case'], r['result'], reviews.get(r['case']['id'])) for r in rows]
            self.assertEqual(sum(s['task_success'] is True for s in scores), expected)
            self.assertFalse(any(s['task_success'] is None for s in scores))

    def test_known_frozen_execution_blind_spot_is_exposed_not_hidden(self):
        gaps = [r for r in self.runs['base'] if r['metrics']['execution_success']
                and audit(r)['last_observed_state'] is None]
        self.assertEqual(len(gaps), 8)
        for r in gaps:
            self.assertEqual(decision('base', r)[0], 'fail')

    def test_correct_nonzero_number_does_not_excuse_contradictory_prose(self):
        row = next(r for r in self.runs['sft'] if r['case']['id'] == 'challenge_structure_length1_home_settings')
        self.assertIn('实际工具失败1次', row['result']['final_answer'])
        self.assertIn('本轮未发生工具失败', row['result']['final_answer'])
        self.assertEqual(decision('sft', row)[0], 'fail')

    def test_no_tools_claim_does_not_mean_no_mutation(self):
        row = next(r for r in self.runs['sft'] if r['case']['id'] == 'challenge_constraints_read_only_android_multiplayer_dungeon')
        self.assertEqual(decision('sft', row)[0], 'fail')
