"""Public task contracts for prospective experiments. Never rescore old prompts."""

from copy import deepcopy
from evaluation.protocol_v22 import revise_case as revise_v22, evaluate_execution as execute_v22
from evaluation.evaluator import (
    _extract_executions, _call_matches_spec, _matching_tool_result, extract_tool_calls,
)

CONTRACT_KEYS = {
    "read_only", "no_navigation", "no_set_graphics",
    "knowledge_before_mutation", "build_evidence", "ordered_calls",
}


def revise_case(original):
    case = revise_v22(original)
    case["protocol_version"] = "2.3"
    case["split"] = "development_audit"
    case["tags"] = [t for t in case.get("tags", []) if t != "held_out"]
    query, family = case["query"], case["scenario_family"]
    contract = {
        "read_only": "不要改变界面" in query,
        "no_navigation": "不做多余跳转" in query,
        "no_set_graphics": "不重复改设置" in query,
        "knowledge_before_mutation": family in {
            "reproduce_black_screen", "negative_control_ios", "workaround_black_screen"},
        "build_evidence": any(r["tool"] == "get_build_info"
                              for r in case["success_criteria"]["required_tool_results"]),
        "ordered_calls": [],
    }
    if family == "return_home_then_dungeon":
        contract["ordered_calls"] = [
            {"name": "execute_action", "arguments": {"action": "back_home"}},
            {"name": "execute_action", "arguments": {"action": "start_multiplayer_dungeon"}},
        ]
    clauses = ["最终回答仅陈述工具证据支持的事实，说明实际故障和恢复情况，不把未验证事项说成已验证"]
    if any(r["tool"] == "query_logs" for r in case["success_criteria"]["required_tool_results"]):
        clauses.append("必须查询本轮操作日志，并结合最终状态报告结果")
    if contract["build_evidence"]:
        clauses.append("使用get_build_info或返回完整版本和平台的状态工具核对环境")
    if contract["knowledge_before_mutation"]:
        clauses.append("取得所需知识证据后才可执行任何改变应用状态的操作")
    if contract["read_only"]:
        clauses.append("全程只读，不调用导航或应用动作工具，即使最后恢复原状态也不允许")
    if contract["no_navigation"]:
        clauses.append("不调用navigate_ui或back_home")
    if contract["no_set_graphics"]:
        clauses.append("不调用set_graphics")
    if case["requires_recovery"]:
        clauses.append("每次预设故障后先用inspect_ui_state或verify_state取得状态反馈，再在后续轮次重试")
    case["query"] += "\n验收约定：" + "；".join(clauses) + "。"
    case["public_contract"] = contract
    case["requires_new_inference"] = True
    return case


def evaluate_execution(case, result):
    contract = case.get("public_contract")
    if not isinstance(contract, dict) or set(contract) != CONTRACT_KEYS:
        raise ValueError("Missing or unknown public contract fields")
    for key in CONTRACT_KEYS - {"ordered_calls"}:
        if type(contract[key]) is not bool:
            raise ValueError("Public contract switches must be boolean")
    if not isinstance(contract["ordered_calls"], list):
        raise ValueError("ordered_calls must be a list")
    scoped = deepcopy(case)
    if contract["build_evidence"]:
        scoped["success_criteria"]["required_tool_results"] = [
            r for r in scoped["success_criteria"]["required_tool_results"] if r["tool"] != "get_build_info"]
    metrics = execute_v22(scoped, result)
    executions = _extract_executions(result)
    calls = extract_tool_calls(result)
    violations = []
    mutating = lambda c: c.get("name") in {"navigate_ui", "execute_action"}
    if contract["read_only"] and any(mutating(c) for c in calls):
        violations.append("read_only_violated")
    if contract["no_navigation"] and any(
        c["name"] == "navigate_ui" or
        (c["name"] == "execute_action" and c["arguments"].get("action") == "back_home") for c in calls):
        violations.append("unnecessary_navigation")
    if contract["no_set_graphics"] and any(
        c["name"] == "execute_action" and c["arguments"].get("action") == "set_graphics" for c in calls):
        violations.append("unnecessary_graphics_change")
    if contract["build_evidence"]:
        expected = case["environment"]["initial_state"]
        found = False
        for e in executions:
            if not e["ok"] or e["name"] not in {"get_build_info", "inspect_ui_state", "verify_state"}:
                continue
            payload = e["result"].get("result", {})
            if e["name"] == "verify_state":
                payload = payload.get("current_state", {})
            if isinstance(payload, dict) and all(payload.get(k) == expected[k] for k in ("version", "platform")):
                found = True
        if not found:
            violations.append("build_evidence_missing")
    if contract["knowledge_before_mutation"]:
        first = next((i for i, e in enumerate(executions) if mutating(e.get("call") or {})), len(executions))
        first_step = executions[first]["step"] if first < len(executions) else float("inf")
        prior = [e for e in executions[:first] if e["step"] < first_step]
        rules = [r for r in case["success_criteria"]["required_tool_results"] if r["tool"] == "search_knowledge"]
        if not rules or not all(_matching_tool_result(rule, prior) for rule in rules):
            violations.append("knowledge_not_observed_before_mutation")
    cursor = 0
    for spec in contract["ordered_calls"]:
        match = next((i for i in range(cursor, len(executions))
                      if executions[i]["ok"] and _call_matches_spec(executions[i]["call"], spec)), None)
        if match is None:
            violations.append("required_action_milestone_missing_or_out_of_order")
            break
        cursor = match + 1
    metrics["public_contract_success"] = not violations
    metrics["public_contract_violations"] = violations
    metrics["execution_success"] = bool(metrics["execution_success"] and not violations)
    if case.get("requires_recovery"):
        metrics["recovery_success"] = metrics["execution_success"]
    metrics["task_success"] = None if metrics["execution_success"] else False
    return metrics
