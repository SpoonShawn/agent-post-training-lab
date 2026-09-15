"""Prospective explicit observation and log-timing contracts.

Separate entry point: frozen 2.3 scoring and runtime files are not edited.
"""
from copy import deepcopy
from evaluation.scoring import evaluate_case as score_v23
from evaluation.observation_audit import audit_observation

FIELDS = {"current_page", "graphics_preset", "battle_hud_visible", "black_screen", "version", "platform"}


def validate_case(case):
    if case.get("protocol_version") != "2.4":
        raise ValueError("2.4 requires a newly declared case; no silent old-case upgrade")
    contract = case.get("observation_contract")
    if not isinstance(contract, dict) or set(contract) != {"final_state_fields", "logs_after_last_mutation"}:
        raise ValueError("Missing/unknown observation contract")
    fields = contract["final_state_fields"]
    if (not isinstance(fields, list) or not fields or any(not isinstance(k, str) for k in fields)
            or len(fields) != len(set(fields)) or not set(fields) <= FIELDS):
        raise ValueError("Explicit unique supported state fields required")
    if type(contract["logs_after_last_mutation"]) is not bool:
        raise ValueError("Log timing must be boolean")
    expected = case.get("success_criteria", {}).get("final_state", {})
    if any(k not in expected for k in fields):
        raise ValueError("Observed fields must have explicit final-state expectations")
    for k in fields:
        kind = bool if k in {"battle_hud_visible", "black_screen"} else str
        if type(expected[k]) is not kind:
            raise ValueError("Invalid expected state type")
    if not isinstance(case.get("query"), str) or not case["query"].strip():
        raise ValueError("Missing public query")


def evaluate_case(case, result):
    validate_case(case)
    if result.get("query") != case["query"]:
        raise ValueError("Changed query requires fresh inference")
    legacy = deepcopy(case)
    legacy["protocol_version"] = "2.3"
    metrics = score_v23(legacy, result)
    required = {k: case["success_criteria"]["final_state"][k]
                for k in case["observation_contract"]["final_state_fields"]}
    observation = audit_observation(result, required)
    executions = [(turn.get("step"), entry) for turn in result.get("trajectory", [])
                  for entry in turn.get("tool_results", [])]
    # Query logs after every attempted application action/navigation, including errors.
    last_mutation = max((i for i, (_, e) in enumerate(executions)
                         if e.get("tool_call", {}).get("name") in {"navigate_ui", "execute_action"}),
                        default=-1)
    logs = [dict(execution_index=i, step=step) for i, (step, e) in enumerate(executions)
            if i > last_mutation and e.get("tool_call", {}).get("name") == "query_logs"
            and e.get("result", {}).get("ok") is True
            and isinstance(e["result"].get("result"), list)]
    log_ok = bool(logs) if case["observation_contract"]["logs_after_last_mutation"] else True
    previous = metrics["execution_success"]
    violations = []
    if not observation["observation_satisfied"]:
        violations.append("final_state_observation_missing_or_mismatched")
    if not log_ok:
        violations.append("logs_not_read_after_last_application_attempt")
    metrics.update(
        protocol_version="2.4", execution_success_v23=previous,
        observation_success=observation["observation_satisfied"], observation_evidence=observation,
        log_timing_success=log_ok, final_log_evidence=logs, observation_violations=violations,
        execution_success=bool(previous and not violations), answer_review="pending")
    metrics["task_success"] = None if metrics["execution_success"] else False
    metrics["recovery_execution_success"] = metrics["execution_success"] if case.get("requires_recovery") else None
    metrics["recovery_success"] = metrics["task_success"] if case.get("requires_recovery") else None
    return metrics
