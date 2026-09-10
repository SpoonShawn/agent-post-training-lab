"""Post-audit protocol. Preserve v2 scores; separate execution from prose review."""

from copy import deepcopy

from evaluation.evaluator import evaluate_case

PROTOCOL_VERSION = "2.1"


def revise_case(original):
    case = deepcopy(original)
    case["protocol_version"] = PROTOCOL_VERSION
    case["source_case_id"] = original["id"]
    # Every input must identify the incident without relying on hidden labels.
    if "用事件编号" in case["query"]:
        incident = "INC-102" if case["scenario_family"] == "knowledge_hud" else "INC-101"
        case["query"] = case["query"].replace("用事件编号", f"用事件编号 {incident}")
    case["requires_new_inference"] = case["query"] != original["query"]
    return case


def evaluate_execution(case, result):
    """Score observable execution, not completeness/correctness of final prose.

    Logs and knowledge evidence remain mandatory. State returned by actions is
    equivalent to a redundant inspect/verify call. Argument exact matching for
    free-text search remains a legacy diagnostic, excluded from structured args.
    """
    execution_case = deepcopy(case)
    criteria = execution_case["success_criteria"]
    criteria.pop("final_answer", None)
    if case["category"] in {"navigation", "long_horizon", "recovery"}:
        criteria["required_tool_results"] = [
            rule for rule in criteria["required_tool_results"]
            if rule["tool"] not in {"inspect_ui_state", "verify_state"}
        ]
    metrics = evaluate_case(execution_case, result)
    structured = deepcopy(execution_case)
    structured["required_calls"] = [
        call for call in structured["required_calls"]
        if call["name"] != "search_knowledge"
    ]
    argument_score = evaluate_case(structured, result)["argument_accuracy"]
    success = metrics["task_success"]
    if case.get("requires_recovery"):
        success = metrics["recovery_success"]
    return {
        "execution_success": success,
        "structured_argument_accuracy": argument_score,
        "final_state_match": metrics["final_state_match"],
        "required_tool_results_accuracy": metrics["required_tool_results_accuracy"],
        "recovery_success": metrics["recovery_success"],
        # Free-form prose needs independent review; keyword hits are insufficient.
        "answer_review": "pending",
        "task_success": False if not success else None,
    }
