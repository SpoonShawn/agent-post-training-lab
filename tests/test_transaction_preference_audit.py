import unittest
from scripts.analyze_transaction_preference_probe import audit
from scripts.transaction_base_preferences import pairs


class PreferenceAuditTests(unittest.TestCase):
    def test_uploaded_supply_replays_and_abstains(self):
        summary, records = audit()
        self.assertEqual(len(records), 392)
        self.assertEqual(summary["successful_rollouts"], 392)
        self.assertEqual(summary["usable_pairs"], 0)
        self.assertEqual(summary["abstained_tasks"], 98)
        self.assertEqual(summary["distinct_trajectories_per_task"], {"1": 98})

    def test_cross_policy_pairs_require_same_case_and_actual_success(self):
        def row(success, candidate=0):
            return dict(case=dict(id="a", split="train"), candidate=candidate,
                        metrics=dict(task_success=success))
        self.assertEqual(pairs([row(True)], [row(True)])["pairs"], [])
        result = pairs([row(False)], [row(True)])
        self.assertEqual(result["pairs"][0]["chosen"]["source"], "sft_probe")
        self.assertEqual(pairs([row(True)], [row(False)])["pairs"][0]["chosen"]["source"], "base")
        changed = row(False)
        changed["case"]["split"] = "confirmation_id"
        with self.assertRaises(ValueError):
            pairs([changed], [row(True)])


if __name__ == "__main__":
    unittest.main()
