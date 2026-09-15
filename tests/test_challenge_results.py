import unittest
from scripts.summarize_baseline import load_records
from scripts.compare_pilot_v1 import ROOT, replay


class ChallengeResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = {m: load_records(ROOT / f'results/baseline/challenge_v1_{m}.jsonl', True)
                    for m in ('base', 'sft')}

    def test_observed_execution_counts(self):
        for mode, expected in [('base', 18), ('sft', 82)]:
            rows = self.runs[mode]
            self.assertEqual(len(rows), 120)
            self.assertEqual(sum(r['metrics']['execution_success'] for r in rows), expected)
            self.assertEqual(sum(r['metrics']['task_success'] is None for r in rows), expected)

    def test_rewritten_failure_is_not_equivalent_to_wrong_final_state(self):
        rows = [r for r in self.runs['sft'] if r['case'].get('expression_variant') == 'rewrite']
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(r['metrics']['final_state_match'] for r in rows), 19)
        for r in rows:
            self.assertFalse(r['metrics']['execution_success'])
            logs = [e['result']['result'] for t in r['result']['trajectory']
                    for e in t.get('tool_results', []) if e['tool_call']['name'] == 'query_logs']
            self.assertEqual(logs, [[]])

    def test_recorded_calls_replay(self):
        self.assertEqual(sum(replay(r) for rows in self.runs.values() for r in rows), 2366)

    def test_recovery_failures_precede_injected_fault(self):
        failed = [r for r in self.runs['sft'] if r['case']['category'] == 'recovery'
                  and not r['metrics']['execution_success']]
        self.assertEqual(len(failed), 7)
        for r in failed:
            self.assertIn('required_injected_failure_not_observed', r['metrics']['process_violations'])
