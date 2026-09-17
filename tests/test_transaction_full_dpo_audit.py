import unittest
from scripts.analyze_transaction_full_dpo import audit


class FullDPOAuditTests(unittest.TestCase):
    def test_uploaded_three_model_replay(self):
        result, changes, failures = audit()
        self.assertEqual(result['training']['steps'], 80)
        self.assertEqual(len(failures), 144)
        self.assertEqual(len(changes), 5)
        self.assertEqual(result['splits']['confirmation_ood']['dpo']['task_success']['count'], 256)
        self.assertEqual(result['splits']['confirmation_ood']['paired_vs_sft']['regressed'], 2)
        self.assertEqual(result['splits']['confirmation_ood']['paired_vs_sft']['improved'], 0)


if __name__ == '__main__':
    unittest.main()
