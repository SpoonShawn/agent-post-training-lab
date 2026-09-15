"""Prospective observation check, NOT wired into frozen 2.3 scoring.

Caller supplies the explicit required fields. No natural-language requirement inference.
Only successful model-visible state results count; saved environment truth does not.
"""
STATE_TOOLS = {"inspect_ui_state", "verify_state", "get_build_info", "navigate_ui", "execute_action"}
MUTATIONS = {"navigate_ui", "execute_action"}


def audit_observation(result, required_state):
    if not required_state:
        raise ValueError("Explicit nonempty observation requirement needed")
    evidence = {}
    for turn in result.get("trajectory", []):
        for entry in turn.get("tool_results", []):
            name = entry.get("tool_call", {}).get("name")
            response = entry.get("result", {})
            if response.get("ok") is not True:
                continue
            if name in MUTATIONS:
                evidence.clear()  # Even return-to-same-state requires fresh evidence.
            payload = response.get("result")
            if name == "verify_state" and isinstance(payload, dict):
                payload = payload.get("current_state")
            if name not in STATE_TOOLS or not isinstance(payload, dict):
                continue
            for key, value in payload.items():
                if key in required_state:
                    evidence[key] = dict(value=value, tool=name, step=turn.get("step"))
    missing = [key for key in required_state if key not in evidence]
    mismatched = [key for key, value in required_state.items() if key in evidence
                  and (type(evidence[key]["value"]) is not type(value) or evidence[key]["value"] != value)]
    return dict(observation_satisfied=not missing and not mismatched,
                missing_fields=missing, mismatched_fields=mismatched, evidence=evidence,
                changes_frozen_score=False)
