import unittest
from scripts.audit_answer_evidence import audit


def row(answer, results=()):
    return {"case": {"id": "test", "query": "报告实际失败次数"},
            "result": {"final_answer": answer, "trajectory": [
                {"step": i, "tool_results": [
                    {"tool_call": {"name": "verify_state", "arguments": {}}, "result": r}]}
                for i, r in enumerate(results)]}}


class EvidenceAuditTests(unittest.TestCase):
    def test_actual_error_not_hidden_by_empty_application_log(self):
        result = audit(row("实际发生的失败次数：0", [{"ok": False, "error": "invalid"}]))
        self.assertEqual(result["count_audit_status"], "mismatch_needs_review")

    def test_negative_verification_is_separate_from_transport_error(self):
        result = audit(row("实际工具失败1次；", [{"ok": True, "result": {"success": False}}]))
        self.assertEqual(result["negative_verification_count"], 1)
        self.assertEqual(result["tool_error_count"], 0)

    def test_matching_count_is_never_semantic_approval(self):
        result = audit(row("实际工具失败0次；所有版本都已修复。"))
        self.assertEqual(result["count_audit_status"], "matched_count_only")
        self.assertTrue(result["requires_semantic_review"])
        self.assertEqual(result["semantic_verdict"], "not_assigned")

    def test_unknown_negated_and_range_claims_abstain(self):
        for text in ("没有失败", "并非实际工具失败0次", "实际发生的失败次数：0到2次",
                     "工具说“实际工具失败0次”，但我不同意"):
            self.assertEqual(audit(row(text))["count_audit_status"], "unparsed_needs_review")

    def test_conflicting_counts_abstain(self):
        self.assertEqual(audit(row("实际工具失败0次；实际工具失败1次。"))["count_audit_status"],
                         "conflicting_claims_needs_review")

    def test_markdown_explicit_count(self):
        self.assertEqual(audit(row("- **实际发生的失败次数**：0 次。"))["claimed_failure_counts"], [0])
