import unittest
from evaluation.observation_audit import audit_observation


def result(*entries):
    return {"final_environment_state": {"current_page": "home"},
            "trajectory": [{"step": i, "tool_results": [
                {"tool_call": {"name": name, "arguments": {}}, "result": value}]}
                for i, (name, value) in enumerate(entries)]}


class ObservationAuditTests(unittest.TestCase):
    expected = {"current_page": "home"}

    def test_hidden_truth_is_not_an_observation(self):
        self.assertFalse(audit_observation(result(), self.expected)["observation_satisfied"])

    def test_equivalent_action_and_verify_results_accepted(self):
        for name, payload in [
            ("execute_action", self.expected),
            ("inspect_ui_state", self.expected),
            ("verify_state", {"success": True, "current_state": self.expected})]:
            self.assertTrue(audit_observation(result((name, {"ok": True, "result": payload})),
                                              self.expected)["observation_satisfied"])

    def test_stale_state_invalidated_by_later_mutation(self):
        r = result(("inspect_ui_state", {"ok": True, "result": self.expected}),
                   ("execute_action", {"ok": True, "result": {}}))
        self.assertFalse(audit_observation(r, self.expected)["observation_satisfied"])

    def test_error_payload_is_not_an_observation(self):
        r = result(("inspect_ui_state", {"ok": False, "result": self.expected}))
        self.assertFalse(audit_observation(r, self.expected)["observation_satisfied"])

    def test_knowledge_text_is_not_current_state(self):
        r = result(("search_knowledge", {"ok": True, "result": self.expected}))
        self.assertFalse(audit_observation(r, self.expected)["observation_satisfied"])

    def test_boolean_is_not_integer(self):
        r = result(("inspect_ui_state", {"ok": True, "result": {"black_screen": 0}}))
        self.assertFalse(audit_observation(r, {"black_screen": False})["observation_satisfied"])
