import unittest
from copy import deepcopy

from scripts.compare_pilot_v1 import ROOT, compact, replay
from scripts.summarize_baseline import load_records


class PilotComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_records(ROOT / 'results/baseline/pilot_v1_sft_confirmation.jsonl', recompute=True)

    def test_record_replay(self):
        self.assertGreater(replay(self.rows[0]), 0)

    def test_changed_return_rejected(self):
        row = deepcopy(self.rows[0])
        entry = next(entry for step in row['result']['trajectory'] for entry in step.get('tool_results', []))
        entry['result'] = {'tampered': True}
        with self.assertRaises(ValueError):
            replay(row)

    def test_changed_final_state_rejected(self):
        row = deepcopy(self.rows[0])
        row['result']['final_environment_state'] = {'tampered': True}
        with self.assertRaises(ValueError):
            replay(row)

    def test_execution_is_not_answer_approval(self):
        result = compact(self.rows)
        self.assertEqual(result['execution_pass'], 80)
        self.assertEqual(result['answer_pending'], 80)
        self.assertIsNone(result['task_rate'])
