import unittest
from scripts.analyze_transaction_full_sft import audit, answer_errors


class AuditTests(unittest.TestCase):
    def test_complete_paired_replay_and_failures(self):
        summary, failures = audit()
        self.assertEqual(len(failures), 926)
        for split, count, passed in (("dev", 256, 256), ("confirmation_id", 128, 128), ("confirmation_ood", 400, 258)):
            result = summary["splits"][split]
            self.assertEqual(result["base"]["cases"], count)
            self.assertEqual(result["base"]["task_success"]["count"], 0)
            self.assertEqual(result["sft"]["task_success"]["count"], passed)
        ood = summary["splits"]["confirmation_ood"]["sft"]
        self.assertEqual(ood["execution_success"]["count"], 400)
        self.assertEqual(ood["policy_violating_cases"], 16)
        self.assertEqual(ood["policy_violating_calls"], 244)
        self.assertEqual(summary["sft_answer_errors"], {"tool_errors": 126, "missing_or_non_json_report": 16})

    def test_field_error_preserves_types(self):
        record = dict(result=dict(final_answer='{"tool_errors":true}'),
                      metrics=dict(expected_report={"tool_errors":1}, answer_correct=False))
        self.assertEqual(answer_errors(record), ["tool_errors"])
        record["result"]["final_answer"] = None
        self.assertEqual(answer_errors(record), ["missing_or_non_json_report"])


if __name__ == "__main__":
    unittest.main()
