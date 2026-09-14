import unittest

from scripts.generate_benchmark_v2 import build_cases
from evaluation.protocol_v22 import revise_case
from scripts.audit_contracts_v22 import build_report


class ContractAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = [revise_case(case) for case in build_cases()]
        cls.rows, cls.probes, cls.summary = build_report(cls.cases, "test-source")

    def test_inventory_has_all_cases_and_no_model_labels(self):
        self.assertEqual(len(self.rows), 360)
        self.assertEqual(len({row["id"] for row in self.rows}), 360)
        self.assertTrue(all("verdict" not in row for row in self.rows))
        self.assertEqual(self.summary["risk_counts"]["mandatory_logs_not_explicit_in_prompt"], 72)
        self.assertEqual(self.summary["risk_counts"]["knowledge_before_action_order_not_enforced"], 90)

    def test_control_probes_are_executable(self):
        for probe in self.probes:
            self.assertTrue(probe["control"]["execution_success"], probe["probe"])

    def test_counterexamples_expose_existing_gaps(self):
        expected = {
            "remove_unrequested_logs": False,
            "move_knowledge_after_execution": True,
            "leave_and_return_despite_no_extra_navigation": True,
            "equivalent_build_state_tool": False,
        }
        for probe in self.probes:
            self.assertIs(probe["mutation"]["execution_success"], expected[probe["probe"]])
            self.assertEqual(probe["control"]["result"]["final_environment_state"],
                             probe["mutation"]["result"]["final_environment_state"])
            self.assertIsNot(probe["mutation"]["task_success"], True)

    def test_does_not_modify_cases_or_accept_wrong_scope(self):
        self.assertTrue(all(case["protocol_version"] == "2.2" for case in self.cases))
        with self.assertRaises(ValueError):
            build_report(self.cases[:5], "source")


if __name__ == "__main__":
    unittest.main()
