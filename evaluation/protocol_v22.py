"""Explicit recovery ordering contracts; legacy protocols remain immutable."""

from evaluation.protocol_v21 import revise_case as revise_v21, evaluate_execution as execution_v21
from evaluation.evaluator import (
    _extract_executions, _observed_expected_failures, _execution_targets_fault,
)

PROTOCOL_VERSION = "2.2"


def revise_case(original):
    case = revise_v21(original)
    case["protocol_version"] = PROTOCOL_VERSION
    # These two audited benchmark families explicitly request post-fault checks.
    # Do not infer requirements from model answers or loose keyword matching.
    rules = []
    if case.get("requires_recovery"):
        if case.get("scenario_family") not in {"recovery_action", "recovery_navigation"}:
            raise ValueError("Recovery family needs an explicit process contract")
        rules = ["observe_state_before_retry"]
    case["process_constraints"] = rules
    return case


def process_metrics(case, result):
    rules = case.get("process_constraints")
    if rules not in ([], ["observe_state_before_retry"]):
        raise ValueError("Missing or unsupported 2.2 process constraints")
    if case.get("requires_recovery") and not rules:
        raise ValueError("Recovery requires a process contract")
    if not rules:
        return {"process_success": True, "process_violations": [], "process_evidence": []}
    executions = _extract_executions(result)
    faults = _observed_expected_failures(case, executions)
    violations, evidence = [], []
    if not faults:
        violations.append("required_injected_failure_not_observed")
    for index, fault in faults.items():
        failure = executions[index]
        # Inspect must precede the FIRST subsequent attempt, not just some later
        # successful retry. An unobserved retry cannot be repaired retroactively.
        retry_index = next((j for j in range(index + 1, len(executions))
                            if _execution_targets_fault(executions[j], fault)), None)
        observation = None
        fields = ["current_page"]
        if fault.get("target") == "set_graphics":
            fields.append("graphics_preset")
        if retry_index is not None:
            retry = executions[retry_index]
            for candidate in executions[index + 1:retry_index]:
                if not (failure["step"] < candidate["step"] < retry["step"]):
                    continue  # Same model turn cannot condition on new feedback.
                envelope = candidate.get("result")
                if not isinstance(envelope, dict) or envelope.get("ok") is not True:
                    continue
                payload = envelope.get("result")
                if not isinstance(payload, dict):
                    continue
                if candidate["name"] == "verify_state":
                    payload = payload.get("current_state")
                elif candidate["name"] != "inspect_ui_state":
                    continue  # Mutating actions are not read-only state checks.
                if (isinstance(payload, dict)
                        and all(isinstance(payload.get(key), str) and payload[key] for key in fields)):
                    observation = candidate["step"]
                    break
        reason = None
        if retry_index is None:
            reason = "retry_missing"
        elif observation is None:
            reason = "post_failure_state_check_missing_before_first_retry"
        if reason:
            violations.append(reason)
        evidence.append({
            "failure_step": failure["step"], "failure_result_index": failure["index"],
            "fault": fault, "required_state_fields": fields,
            "observation_step": observation,
            "retry_step": executions[retry_index]["step"] if retry_index is not None else None,
            "success": reason is None,
        })
    return {"process_success": not violations, "process_violations": violations,
            "process_evidence": evidence}


def evaluate_execution(case, result):
    metrics = execution_v21(case, result)
    process = process_metrics(case, result)
    metrics["execution_success_before_process"] = metrics["execution_success"]
    metrics.update(process)
    metrics["execution_success"] = bool(metrics["execution_success"] and process["process_success"])
    if case.get("requires_recovery"):
        metrics["recovery_success"] = bool(metrics["recovery_success"] and process["process_success"])
    metrics["task_success"] = None if metrics["execution_success"] else False
    return metrics
