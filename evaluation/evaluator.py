"""Outcome-aware metrics for BugOps tool-use trajectories.

``evaluate_case(case, result)`` remains compatible with the v1 benchmark.  V2
separates tool selection, ordering, arguments, execution health, and observable
task success so that one exact reference trajectory is not the only way to pass.
The extraction helpers are defensive by design: incomplete/error trajectories
should produce useful metrics rather than make an evaluation run fail.
"""

from collections import Counter
import json
import re


_MISSING = object()


def _json_value(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def _call_from_item(item):
    """Normalize a parser wrapper or direct call to name/arguments."""

    if isinstance(item, str):
        item = _json_value(item)
    if not isinstance(item, dict):
        return None
    call = item.get("tool_call", item.get("call", item))
    if isinstance(call, str):
        call = _json_value(call)
    if not isinstance(call, dict):
        return None
    name = call.get("name", call.get("tool"))
    if not isinstance(name, str) or not name:
        return None
    return {
        "name": name,
        "arguments": _json_value(call.get("arguments", call.get("args", {}))),
    }


def _parser_item_valid(item):
    if isinstance(item, dict) and "valid" in item:
        # Syntax validity, execution validity, and argument correctness are
        # intentionally separate metrics.
        return item.get("valid") is True and _call_from_item(item) is not None
    return _call_from_item(item) is not None


def _trajectory(result):
    if not isinstance(result, dict):
        return []
    value = result.get("trajectory", [])
    return value if isinstance(value, list) else []


def _step_call_items(step):
    if not isinstance(step, dict):
        return []
    for key in ("parsed_tool_calls", "tool_calls", "calls"):
        if key in step:
            value = step.get(key)
            if isinstance(value, list):
                return value
            return [] if value is None else [value]
    for key in ("tool_call", "call"):
        if key in step:
            return [step.get(key)]
    return []


def _step_result_items(step):
    if not isinstance(step, dict):
        return []
    for key in ("tool_results", "results"):
        if key in step:
            value = step.get(key)
            if isinstance(value, list):
                return value
            return [] if value is None else [value]
    for key in ("tool_result", "result"):
        if key in step and ("tool_call" in step or "call" in step):
            return [{
                "tool_call": step.get("tool_call", step.get("call")),
                "result": step.get(key),
            }]
    return []


def _has_tool_call_marker(value):
    return isinstance(value, str) and re.search(
        r"</?tool_call(?=\s|>|$)", value, flags=re.IGNORECASE
    ) is not None


def _extract_attempts(result):
    attempts = []
    for step_index, step in enumerate(_trajectory(result)):
        items = _step_call_items(step)
        if not items and isinstance(step, dict) and _has_tool_call_marker(
            step.get("model_output")
        ):
            # An unclosed/malformed marker expresses call intent even though
            # the parser returns no items.  Count one invalid intent per turn.
            attempts.append({
                "step": step_index,
                "index": 0,
                "valid": False,
                "call": None,
                "malformed_marker": True,
            })
            continue
        if not items:
            # Fallback for compact runners which persist only executed calls.
            for result_item in _step_result_items(step):
                if isinstance(result_item, dict):
                    call = result_item.get("tool_call", result_item.get("call"))
                    if call is not None:
                        items.append(call)
        for call_index, item in enumerate(items):
            attempts.append({
                "step": step_index,
                "index": call_index,
                "valid": _parser_item_valid(item),
                "call": _call_from_item(item),
                "malformed_marker": False,
            })

    if attempts or not isinstance(result, dict):
        return attempts
    items = result.get("parsed_tool_calls", result.get("tool_calls", []))
    if not isinstance(items, list):
        items = [] if items is None else [items]
    for call_index, item in enumerate(items):
        attempts.append({
            "step": 0,
            "index": call_index,
            "valid": _parser_item_valid(item),
            "call": _call_from_item(item),
            "malformed_marker": False,
        })
    if not attempts and _has_tool_call_marker(result.get("model_output")):
        attempts.append({
            "step": 0,
            "index": 0,
            "valid": False,
            "call": None,
            "malformed_marker": True,
        })
    return attempts


def _result_envelope(item):
    if not isinstance(item, dict):
        return item
    # {tool_call, result:{ok,result/error}} is a wrapper.  Bare {ok,result}
    # is already the envelope and must remain intact.
    if "tool_call" in item or "call" in item:
        if "result" in item:
            return item.get("result")
        if "tool_result" in item:
            return item.get("tool_result")
    return item


def _execution_ok(envelope):
    if isinstance(envelope, dict):
        if "ok" in envelope:
            return envelope.get("ok") is True
        if envelope.get("error") is not None or envelope.get("error_type"):
            return False
        if isinstance(envelope.get("success"), bool):
            return envelope.get("success")
        return True
    return envelope is not None


def _extract_executions(result):
    executions = []
    for step_index, step in enumerate(_trajectory(result)):
        local_calls = []
        for item in _step_call_items(step):
            call = _call_from_item(item)
            if _parser_item_valid(item) and call is not None:
                local_calls.append(call)
        for result_index, item in enumerate(_step_result_items(step)):
            call = None
            if isinstance(item, dict):
                call = _call_from_item(item.get("tool_call", item.get("call")))
                if call is None and "name" in item:
                    call = _call_from_item(item)
            if call is None and result_index < len(local_calls):
                call = local_calls[result_index]
            envelope = _result_envelope(item)
            executions.append({
                "step": step_index,
                "index": result_index,
                "call": call,
                "name": call.get("name") if call else None,
                "arguments": call.get("arguments") if call else None,
                "result": envelope,
                "ok": _execution_ok(envelope),
            })

    if executions or not isinstance(result, dict):
        return executions
    items = result.get("tool_results", [])
    if not isinstance(items, list):
        items = [] if items is None else [items]
    calls = [
        item["call"] for item in _extract_attempts(result)
        if item["valid"] and item["call"] is not None
    ]
    for result_index, item in enumerate(items):
        call = None
        if isinstance(item, dict):
            call = _call_from_item(item.get("tool_call", item.get("call")))
        if call is None and result_index < len(calls):
            call = calls[result_index]
        envelope = _result_envelope(item)
        executions.append({
            "step": 0,
            "index": result_index,
            "call": call,
            "name": call.get("name") if call else None,
            "arguments": call.get("arguments") if call else None,
            "result": envelope,
            "ok": _execution_ok(envelope),
        })
    return executions


def extract_tool_calls(result):
    """Return all syntax-valid, named calls in trajectory order."""

    return [
        item["call"] for item in _extract_attempts(result)
        if item["valid"] and item["call"] is not None
    ]


def extract_tools(result):
    """V1-compatible projection of valid calls to tool names."""

    return [call["name"] for call in extract_tool_calls(result)]


def _strict_json_equal(left, right):
    # Python considers True == 1, while JSON booleans and numbers are distinct.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        return set(left) == set(right) and all(
            _strict_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list):
            return False
        return len(left) == len(right) and all(
            _strict_json_equal(a, b) for a, b in zip(left, right)
        )
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    return type(left) is type(right) and left == right


def _json_subset(expected, actual):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            key in actual and _json_subset(value, actual[key])
            for key, value in expected.items()
        )
    # Arrays are ordered JSON values, not mapping-like subsets.
    return _strict_json_equal(expected, actual)


def _spec_name(spec):
    if not isinstance(spec, dict):
        return None
    name = spec.get("name", spec.get("tool"))
    return name if isinstance(name, str) and name else None


def _required_names(case):
    has_v2 = "required_tools" in case or "required_calls" in case
    if not has_v2:
        expected = case.get("expected_tools")
        return list(expected) if isinstance(expected, list) else None

    required_tools = case.get("required_tools", [])
    required_calls = case.get("required_calls", [])
    if not isinstance(required_tools, list):
        required_tools = []
    if not isinstance(required_calls, list):
        required_calls = []
    tool_counts = Counter(
        item if isinstance(item, str) else _spec_name(item)
        for item in required_tools
    )
    call_counts = Counter(_spec_name(item) for item in required_calls)
    tool_counts.pop(None, None)
    call_counts.pop(None, None)
    combined = tool_counts.copy()
    for name, count in call_counts.items():
        combined[name] = max(combined[name], count)

    names = []
    seen = set()
    for source in (required_tools, required_calls):
        for item in source:
            name = item if isinstance(item, str) else _spec_name(item)
            if name and name not in seen:
                names.extend([name] * combined[name])
                seen.add(name)
    return names


def _optional_names(case):
    optional = case.get("optional_tools", [])
    if not isinstance(optional, list):
        return Counter()
    names = []
    for item in optional:
        name = item if isinstance(item, str) else _spec_name(item)
        if name:
            names.append(name)
    return Counter(names)


def _tool_metrics(case, predicted_names):
    required_names = _required_names(case)
    if required_names is None:
        return {
            "tool_true_positives": None,
            "tool_false_positives": None,
            "tool_false_negatives": None,
            "tool_precision": None,
            "tool_recall": None,
            "tool_f1": None,
        }

    required = Counter(required_names)
    predicted = Counter(predicted_names)
    optional = _optional_names(case)
    true_positives = sum(
        min(count, predicted.get(name, 0))
        for name, count in required.items()
    )
    false_negatives = sum(
        max(0, count - predicted.get(name, 0))
        for name, count in required.items()
    )
    false_positives = sum(
        max(0, count - required.get(name, 0) - optional.get(name, 0))
        for name, count in predicted.items()
    )
    p_denominator = true_positives + false_positives
    r_denominator = true_positives + false_negatives
    precision = true_positives / p_denominator if p_denominator else 1.0
    recall = true_positives / r_denominator if r_denominator else 1.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "tool_true_positives": true_positives,
        "tool_false_positives": false_positives,
        "tool_false_negatives": false_negatives,
        "tool_precision": precision,
        "tool_recall": recall,
        "tool_f1": f1,
    }


def _argument_mode(spec):
    mode = spec.get(
        "argument_match",
        spec.get("arguments_match", spec.get("match", "subset")),
    )
    return mode.lower() if isinstance(mode, str) else "subset"


def _call_matches_spec(call, spec):
    if not isinstance(call, dict) or not isinstance(spec, dict):
        return False
    if call.get("name") != _spec_name(spec):
        return False
    expected = _json_value(spec.get("arguments", spec.get("args", {})))
    actual = _json_value(call.get("arguments", {}))
    if _argument_mode(spec) == "exact":
        return _strict_json_equal(expected, actual)
    return _json_subset(expected, actual)


def _maximum_spec_matches(specs, calls):
    edges = [
        [index for index, call in enumerate(calls) if _call_matches_spec(call, spec)]
        for spec in specs
    ]
    call_to_spec = {}

    def assign(spec_index, visited):
        for call_index in edges[spec_index]:
            if call_index in visited:
                continue
            visited.add(call_index)
            if call_index not in call_to_spec or assign(
                call_to_spec[call_index], visited
            ):
                call_to_spec[call_index] = spec_index
                return True
        return False

    return sum(assign(index, set()) for index in range(len(specs)))


def _argument_metrics(case, calls):
    specs = case.get("required_calls")
    if not isinstance(specs, list) or not specs:
        return {
            "argument_matches": None,
            "argument_total": 0,
            "argument_accuracy": None,
        }
    matches = _maximum_spec_matches(specs, calls)
    return {
        "argument_matches": matches,
        "argument_total": len(specs),
        "argument_accuracy": matches / len(specs),
    }


def _canonical_call(call):
    try:
        arguments = json.dumps(
            call.get("arguments", {}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        arguments = repr(call.get("arguments", {}))
    return call.get("name"), arguments


def _call_outcomes(calls, executions):
    """Align calls with ``(ok, model_step, execution_index)`` observations."""

    outcomes = []
    cursor = 0
    for call in calls:
        signature = _canonical_call(call)
        found = (None, None, None)
        for index in range(cursor, len(executions)):
            execution_call = executions[index].get("call")
            if execution_call and _canonical_call(execution_call) == signature:
                found = (
                    executions[index].get("ok"),
                    executions[index].get("step"),
                    index,
                )
                cursor = index + 1
                break
        outcomes.append(found)
    return outcomes


def _repeat_metrics(calls, executions):
    """Separate legitimate failure retries from redundant exact repeats."""

    states = {}
    repeats = 0
    retries = 0
    outcomes = _call_outcomes(calls, executions)
    last_global_failure = -1
    last_global_failure_step = None
    state_mutating_tools = {"navigate_ui", "execute_action"}
    state_sensitive_tools = {
        "inspect_ui_state",
        "navigate_ui",
        "execute_action",
        "query_logs",
        "verify_state",
    }
    for call_index, (call, observation) in enumerate(zip(calls, outcomes)):
        outcome, current_step, current_execution_index = observation
        signature = _canonical_call(call)
        state = states.setdefault(signature, {
            "attempts": 0,
            "succeeded": False,
            "last_outcome": None,
            "last_index": -1,
            "last_step": None,
            "last_execution_index": None,
        })
        if state["attempts"]:
            if state["last_outcome"] is False:
                adaptive_retry = (
                    isinstance(current_step, int)
                    and isinstance(state["last_step"], int)
                    and current_step > state["last_step"]
                )
                if adaptive_retry:
                    retries += 1
                # A same-turn duplicate cannot have read the failure.  The
                # first genuinely later-turn retry is recovery work; continued
                # retries after repeated failures are redundant.
                if not adaptive_retry or state["attempts"] > 1:
                    repeats += 1
            elif (
                call.get("name") in state_sensitive_tools
                and isinstance(state["last_execution_index"], int)
                and isinstance(current_execution_index, int)
                and any(
                    execution.get("ok")
                    and execution.get("name") in state_mutating_tools
                    for execution in executions[
                        state["last_execution_index"] + 1:
                        current_execution_index
                    ]
                )
            ):
                # Identical stateful calls can be necessary again after the
                # app changes.  For example, back_home before and after
                # editing graphics has the same arguments but a new context.
                pass
            elif (
                last_global_failure > state["last_index"]
                and isinstance(current_step, int)
                and isinstance(last_global_failure_step, int)
                and current_step > last_global_failure_step
            ):
                # Re-observing state after an intervening failure is useful
                # recovery work, even when the observation call is identical.
                pass
            else:
                repeats += 1
        state["attempts"] += 1
        state["last_outcome"] = outcome
        state["succeeded"] = state["succeeded"] or outcome is True
        state["last_index"] = call_index
        state["last_step"] = current_step
        state["last_execution_index"] = current_execution_index
        if outcome is False:
            last_global_failure = call_index
            last_global_failure_step = current_step
    denominator = len(calls)
    return {
        "repeated_tool_calls": repeats,
        "repeated_tool_call_rate": repeats / denominator if denominator else 0.0,
        "retry_calls": retries,
        "retry_call_rate": retries / denominator if denominator else 0.0,
    }


def _error_type(execution):
    value = execution.get("result")
    return str(value.get("error_type", "")) if isinstance(value, dict) else ""


def _error_message(execution):
    value = execution.get("result")
    if not isinstance(value, dict):
        return ""
    return str(value.get("error", value.get("message", "")))


def _is_unknown_tool(execution):
    value = _error_type(execution).strip().lower().replace("-", "_")
    return value in {"unknown_tool", "unknowntool"}


def _is_invalid_transition(execution):
    if execution.get("ok"):
        return False
    value = execution.get("result")
    if isinstance(value, dict) and value.get("invalid_transition") is True:
        return True
    error_type = re.sub(r"[^a-z]", "", _error_type(execution).lower())
    if error_type in {
        "invalidtransition",
        "illegaltransition",
        "invalidnavigation",
        "invalidstate",
        "preconditionfailed",
    }:
        return True
    message = _error_message(execution).lower()
    markers = (
        "无效页面跳转",
        "非法跳转",
        "invalid transition",
        "illegal transition",
        "invalid navigation",
        "cannot navigate",
        "not allowed from",
        "只能在",
        "必须先进入",
    )
    return any(marker in message for marker in markers)


def _lcs_length(left, right):
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def _order_metrics(case, predicted_names):
    expected = case.get("expected_tools")
    if not isinstance(expected, list):
        specs = case.get("required_calls")
        if isinstance(specs, list) and specs:
            expected = [name for name in map(_spec_name, specs) if name]
        else:
            expected = _required_names(case)
    if expected is None:
        return None, None
    length = _lcs_length(expected, predicted_names)
    score = length / len(expected) if expected else 1.0
    return length == len(expected), score


def _unwrap_payload(envelope):
    if isinstance(envelope, dict) and "ok" in envelope and "result" in envelope:
        return envelope.get("result")
    return envelope


def _path_parts(path):
    if path in (None, "", "$", "."):
        return []
    if isinstance(path, (list, tuple)):
        return list(path)
    if not isinstance(path, str):
        return [path]
    path = path.strip()
    if path.startswith("$/") or path.startswith("/"):
        if path.startswith("$/"):
            path = path[1:]
        return [
            item.replace("~1", "/").replace("~0", "~")
            for item in path.split("/")[1:] if item != ""
        ]
    if path.startswith("$."):
        path = path[2:]
    elif path.startswith("$"):
        path = path[1:]
    path = re.sub(r"\[([0-9]+)\]", r".\1", path)
    return [item for item in path.split(".") if item]


def _resolve_path(value, path):
    current = value
    for part in _path_parts(path):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list):
            try:
                index = int(part)
            except (TypeError, ValueError):
                return _MISSING
            if not -len(current) <= index < len(current):
                return _MISSING
            current = current[index]
        else:
            return _MISSING
    return current


def _deep_contains(container, expected):
    if isinstance(expected, dict):
        if isinstance(container, dict) and _json_subset(expected, container):
            return True
        values = (
            container.values() if isinstance(container, dict)
            else container if isinstance(container, list) else []
        )
        return any(_deep_contains(item, expected) for item in values)
    if isinstance(expected, list):
        return isinstance(container, list) and all(
            any(_deep_contains(item, expected_item) for item in container)
            for expected_item in expected
        )
    if isinstance(container, str) and isinstance(expected, str):
        return expected in container
    if _strict_json_equal(container, expected):
        return True
    values = (
        container.values() if isinstance(container, dict)
        else container if isinstance(container, list) else []
    )
    return any(_deep_contains(item, expected) for item in values)


def _matching_tool_result(criterion, executions):
    if not isinstance(criterion, dict):
        return False
    tool = criterion.get("tool", criterion.get("name"))
    if not isinstance(tool, str) or not tool:
        return False
    if "equals" not in criterion and "contains" not in criterion:
        return False
    for execution in executions:
        if not execution.get("ok") or execution.get("name") != tool:
            continue
        payload = _unwrap_payload(execution.get("result"))
        observed = _resolve_path(payload, criterion.get("path", "$"))
        if observed is _MISSING:
            continue
        if "equals" in criterion and _strict_json_equal(
            observed, criterion.get("equals")
        ):
            return True
        if "contains" in criterion and _deep_contains(
            observed, criterion.get("contains")
        ):
            return True
    return False


def _state_candidate(result, executions, expected_state):
    for field in ("final_environment_state", "final_state"):
        value = result.get(field) if isinstance(result, dict) else None
        if isinstance(value, dict):
            return value

    state_keys = {
        "version",
        "platform",
        "current_page",
        "graphics_preset",
        "battle_hud_visible",
        "black_screen",
    }
    expected_keys = set(expected_state) if isinstance(expected_state, dict) else set()
    partial_candidate = None
    for execution in reversed(executions):
        if not execution.get("ok"):
            continue
        payload = _unwrap_payload(execution.get("result"))
        if not isinstance(payload, dict):
            continue
        candidates = []
        for key in ("current_state", "final_state", "state"):
            if isinstance(payload.get(key), dict):
                candidates.append(payload[key])
        candidates.append(payload)
        for candidate in candidates:
            if expected_keys and expected_keys.issubset(candidate):
                return candidate
            if state_keys.issubset(candidate):
                return candidate
            if partial_candidate is None and state_keys.intersection(candidate):
                partial_candidate = candidate
    return partial_candidate


def _final_answer_match(spec, final_answer):
    if not isinstance(spec, dict):
        return False
    present = final_answer is not None and str(final_answer).strip() != ""
    if spec.get("required") is True and not present:
        return False
    groups = [
        spec.get("contains_all", []),
        spec.get("contains_any", []),
        spec.get("regex_any", []),
        spec.get("forbidden_any", []),
        spec.get("regex_none", []),
    ]
    if any(not isinstance(group, list) for group in groups):
        return False
    contains_all, contains_any, regex_any, forbidden_any, regex_none = groups
    if (contains_all or contains_any or regex_any) and not present:
        return False
    answer = str(final_answer or "")
    folded_answer = answer.casefold()
    if contains_all and not all(
        str(item).casefold() in folded_answer for item in contains_all
    ):
        return False
    if contains_any and not any(
        str(item).casefold() in folded_answer for item in contains_any
    ):
        return False
    if regex_any:
        matched = False
        for pattern in regex_any:
            try:
                matched = matched or re.search(
                    str(pattern), answer, flags=re.IGNORECASE
                ) is not None
            except re.error:
                pass
        if not matched:
            return False
    if forbidden_any and any(
        str(item).casefold() in folded_answer for item in forbidden_any
    ):
        return False
    for pattern in regex_none:
        try:
            if re.search(str(pattern), answer, flags=re.IGNORECASE):
                return False
        except re.error:
            # An invalid negative regex cannot match; positive regex_any keeps
            # its stricter all-invalid => no-match behavior above.
            continue
    return True


def _task_success_metrics(case, result, executions, tool_attempt_count):
    criteria = case.get("success_criteria")
    empty = {
        "task_success": None,
        "success_criteria_passed": 0,
        "success_criteria_total": 0,
        "final_state_match": None,
        "required_tool_result_matches": None,
        "required_tool_result_total": 0,
        "required_tool_results_accuracy": None,
        "final_answer_match": None,
        "max_tool_calls_match": None,
        "success_criteria_details": None,
    }
    if not isinstance(criteria, dict) or not criteria:
        return empty

    passed = 0
    total = 0
    details = {}
    final_state_match = None
    if "final_state" in criteria:
        total += 1
        expected = criteria.get("final_state")
        observed = _state_candidate(result, executions, expected)
        final_state_match = (
            isinstance(expected, dict)
            and isinstance(observed, dict)
            and _json_subset(expected, observed)
        )
        passed += int(final_state_match)
        details["final_state"] = {
            "matched": final_state_match,
            "observed": observed,
        }

    tool_matches = None
    tool_total = 0
    if "required_tool_results" in criteria:
        specs = criteria.get("required_tool_results")
        matches = (
            [_matching_tool_result(spec, executions) for spec in specs]
            if isinstance(specs, list) else [False]
        )
        tool_matches = sum(matches)
        tool_total = len(matches)
        passed += tool_matches
        total += tool_total
        details["required_tool_results"] = matches

    final_answer_match = None
    if "final_answer" in criteria:
        total += 1
        final_answer_match = _final_answer_match(
            criteria.get("final_answer"), result.get("final_answer")
        )
        passed += int(final_answer_match)
        details["final_answer"] = final_answer_match

    max_tool_calls_match = None
    if "max_tool_calls" in criteria:
        total += 1
        maximum = criteria.get("max_tool_calls")
        max_tool_calls_match = (
            isinstance(maximum, int)
            and not isinstance(maximum, bool)
            and maximum >= 0
            and tool_attempt_count <= maximum
        )
        passed += int(max_tool_calls_match)
        details["max_tool_calls"] = {
            "matched": max_tool_calls_match,
            "maximum": maximum,
            "observed": tool_attempt_count,
        }

    return {
        "task_success": total > 0 and passed == total,
        "success_criteria_passed": passed,
        "success_criteria_total": total,
        "final_state_match": final_state_match,
        "required_tool_result_matches": tool_matches,
        "required_tool_result_total": tool_total,
        "required_tool_results_accuracy": (
            tool_matches / tool_total if tool_total else None
        ),
        "final_answer_match": final_answer_match,
        "max_tool_calls_match": max_tool_calls_match,
        "success_criteria_details": details,
    }


def _recovery_metrics(
    case,
    executions,
    task_success,
    expected_failure_indices,
    has_fault_plan,
):
    if not case.get("requires_recovery"):
        return {"recovery_attempted": None, "recovery_success": None}
    success_after_failure = False
    if has_fault_plan:
        failure_seen = bool(expected_failure_indices)
        for failure_index, fault in expected_failure_indices.items():
            failure_step = executions[failure_index].get("step")
            for execution in executions[failure_index + 1:]:
                if (
                    execution.get("ok")
                    and isinstance(execution.get("step"), int)
                    and isinstance(failure_step, int)
                    and execution.get("step") > failure_step
                    and _execution_targets_fault(execution, fault)
                ):
                    success_after_failure = True
                    break
            if success_after_failure:
                break
    else:
        failure_seen = False
        failure_step = None
        for execution in executions:
            if not execution.get("ok"):
                failure_seen = True
                if failure_step is None:
                    failure_step = execution.get("step")
            elif (
                failure_seen
                and isinstance(execution.get("step"), int)
                and isinstance(failure_step, int)
                and execution.get("step") > failure_step
            ):
                success_after_failure = True
                break
    return {
        "recovery_attempted": failure_seen,
        "recovery_success": bool(success_after_failure and task_success is True),
    }


def _expected_failure_count(case):
    environment = case.get("environment", {})
    if not isinstance(environment, dict):
        return 0
    faults = environment.get("fault_injections", [])
    if not isinstance(faults, list):
        return 0
    total = 0
    for fault in faults:
        if not isinstance(fault, dict):
            continue
        times = fault.get("times", 1)
        if isinstance(times, int) and not isinstance(times, bool) and times > 0:
            total += times
    return total


def _fault_plan(case):
    environment = case.get("environment", {})
    if not isinstance(environment, dict):
        return []
    faults = environment.get("fault_injections", [])
    return faults if isinstance(faults, list) else []


def _execution_targets_fault(execution, fault):
    if not isinstance(fault, dict):
        return False
    operation = fault.get("operation")
    target = fault.get("target")
    call = execution.get("call") or {}
    arguments = call.get("arguments", {})
    if not isinstance(arguments, dict):
        return False
    if operation == "navigate":
        target_matches = (
            call.get("name") == "navigate_ui"
            and arguments.get("target_page") == target
        )
    elif operation == "action":
        target_matches = (
            call.get("name") == "execute_action"
            and arguments.get("action") == target
        )
    else:
        target_matches = call.get("name") == operation
    return target_matches


def _execution_matches_fault(execution, fault):
    if execution.get("ok") or not _execution_targets_fault(execution, fault):
        return False
    error_type = _error_type(execution).strip().lower()
    message = _error_message(execution)
    return error_type == "runtimeerror" or "注入的临时故障" in message


def _observed_expected_failures(case, executions):
    """Return indices of failures attributable to configured fault injection."""

    matched_indices = {}
    for fault in _fault_plan(case):
        if not isinstance(fault, dict):
            continue
        times = fault.get("times", 1)
        if not isinstance(times, int) or isinstance(times, bool) or times < 1:
            continue
        remaining = times
        for index, execution in enumerate(executions):
            if index in matched_indices:
                continue
            if _execution_matches_fault(execution, fault):
                matched_indices[index] = fault
                remaining -= 1
                if remaining == 0:
                    break
    return matched_indices


def evaluate_case(case, result):
    """Evaluate one v1 or v2 benchmark case against an agent result."""

    if not isinstance(case, dict):
        case = {}
    if not isinstance(result, dict):
        result = {}
    attempts = _extract_attempts(result)
    calls = extract_tool_calls(result)
    predicted = [call["name"] for call in calls]
    executions = _extract_executions(result)

    invalid_calls = sum(not item["valid"] for item in attempts)
    malformed_markers = sum(
        bool(item.get("malformed_marker")) for item in attempts
    )
    valid_calls = len(attempts) - invalid_calls
    failed_calls = sum(not item["ok"] for item in executions)
    unknown_tools = sum(_is_unknown_tool(item) for item in executions)
    invalid_transitions = sum(_is_invalid_transition(item) for item in executions)
    repeats = _repeat_metrics(calls, executions)
    execution_count = len(executions)
    expected_failures = _expected_failure_count(case)
    expected_failure_indices = _observed_expected_failures(case, executions)
    observed_expected_failures = len(expected_failure_indices)
    unexpected_failures = max(0, failed_calls - observed_expected_failures)

    execution_success_rate = (
        (execution_count - failed_calls) / execution_count
        if execution_count else 1.0
    )
    adjusted_success_rate = (
        (execution_count - unexpected_failures) / execution_count
        if execution_count else 1.0
    )
    ordered_subsequence_match, tool_order_score = _order_metrics(case, predicted)
    metrics = {
        # Original keys and semantics are preserved for v1 records.
        "predicted_tools": predicted,
        "num_steps": len(_trajectory(result)),
        "invalid_tool_calls": invalid_calls,
        "malformed_tool_call_markers": malformed_markers,
        "failed_tool_calls": failed_calls,
        "unknown_tool_calls": unknown_tools,
        # V2 format/execution health.
        "tool_attempt_count": len(attempts),
        "tool_call_count": len(calls),
        "tool_execution_count": execution_count,
        "valid_call_rate": valid_calls / len(attempts) if attempts else 1.0,
        "execution_success_rate": execution_success_rate,
        "tool_execution_success_rate": execution_success_rate,
        "expected_failed_calls": expected_failures,
        "observed_expected_failed_calls": observed_expected_failures,
        "unexpected_failed_calls": unexpected_failures,
        "adjusted_execution_success_rate": adjusted_success_rate,
        "unknown_tool_rate": unknown_tools / execution_count if execution_count else 0.0,
        "invalid_transitions": invalid_transitions,
        "invalid_transition_count": invalid_transitions,
        "invalid_transition_rate": (
            invalid_transitions / execution_count if execution_count else 0.0
        ),
        "ordered_subsequence_match": ordered_subsequence_match,
        "tool_order_match": ordered_subsequence_match,
        "tool_order_score": tool_order_score,
    }
    metrics.update(repeats)
    metrics["repeated_exact_call_count"] = repeats["repeated_tool_calls"]
    metrics["repeated_exact_call_rate"] = repeats["repeated_tool_call_rate"]
    metrics.update(_tool_metrics(case, predicted))
    metrics.update(_argument_metrics(case, calls))
    metrics.update(_task_success_metrics(case, result, executions, len(attempts)))
    metrics.update(_recovery_metrics(
        case,
        executions,
        metrics["task_success"],
        expected_failure_indices,
        bool(_fault_plan(case)),
    ))

    expected = case.get("expected_tools")
    if expected is not None:
        metrics["ordered_tool_match"] = predicted == expected
        metrics["tool_set_match"] = set(predicted) == set(expected)
    return metrics


_MACRO_METRICS = (
    "tool_precision",
    "tool_recall",
    "tool_f1",
    "argument_accuracy",
    "valid_call_rate",
    "execution_success_rate",
    "adjusted_execution_success_rate",
    "unknown_tool_rate",
    "repeated_tool_call_rate",
    "retry_call_rate",
    "invalid_transition_rate",
    "tool_order_score",
    "required_tool_results_accuracy",
)

_BOOLEAN_RATES = {
    "task_success_rate": "task_success",
    "recovery_success_rate": "recovery_success",
    "ordered_subsequence_match_rate": "ordered_subsequence_match",
    "ordered_tool_match_rate": "ordered_tool_match",
    "tool_set_match_rate": "tool_set_match",
}

_AVERAGE_COUNTS = {
    "average_num_steps": "num_steps",
    "average_tool_attempts": "tool_attempt_count",
    "average_tool_calls": "tool_call_count",
    "average_invalid_tool_calls": "invalid_tool_calls",
    "average_malformed_tool_call_markers": "malformed_tool_call_markers",
    "average_failed_tool_calls": "failed_tool_calls",
    "average_expected_failed_calls": "expected_failed_calls",
    "average_unexpected_failed_calls": "unexpected_failed_calls",
    "average_unknown_tool_calls": "unknown_tool_calls",
    "average_repeated_tool_calls": "repeated_tool_calls",
    "average_retry_calls": "retry_calls",
    "average_invalid_transitions": "invalid_transition_count",
}

_TOTAL_COUNTS = {
    "total_tool_attempts": "tool_attempt_count",
    "total_tool_calls": "tool_call_count",
    "total_invalid_tool_calls": "invalid_tool_calls",
    "total_malformed_tool_call_markers": "malformed_tool_call_markers",
    "total_failed_tool_calls": "failed_tool_calls",
    "total_expected_failed_calls": "expected_failed_calls",
    "total_unexpected_failed_calls": "unexpected_failed_calls",
    "total_unknown_tool_calls": "unknown_tool_calls",
    "total_repeated_tool_calls": "repeated_tool_calls",
    "total_retry_calls": "retry_calls",
    "total_invalid_transitions": "invalid_transition_count",
}


def _numeric_values(rows, metric_name):
    values = []
    for row in rows:
        value = row["metrics"].get(metric_name)
        if isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, (int, float)):
            values.append(value)
    return values


def _mean_or_none(values):
    return sum(values) / len(values) if values else None


def _summary(rows):
    summary = {"num_cases": len(rows)}
    for metric_name in _MACRO_METRICS:
        values = _numeric_values(rows, metric_name)
        summary[metric_name] = _mean_or_none(values)
        summary[metric_name + "_evaluated_cases"] = len(values)
    for output_name, metric_name in _BOOLEAN_RATES.items():
        values = _numeric_values(rows, metric_name)
        summary[output_name] = _mean_or_none(values)
        summary[output_name + "_evaluated_cases"] = len(values)
    for output_name, metric_name in _AVERAGE_COUNTS.items():
        summary[output_name] = _mean_or_none(_numeric_values(rows, metric_name))
    for output_name, metric_name in _TOTAL_COUNTS.items():
        summary[output_name] = sum(_numeric_values(rows, metric_name))

    long_horizon_rows = [
        row for row in rows
        if row["case"].get("category") == "long_horizon"
        or (
            isinstance(row["case"].get("tags"), list)
            and "long_horizon" in row["case"].get("tags")
        )
    ]
    values = _numeric_values(long_horizon_rows, "task_success")
    summary["long_horizon_success_rate"] = _mean_or_none(values)
    summary["long_horizon_success_evaluated_cases"] = len(values)
    return summary


def _record_row(record):
    if not isinstance(record, dict):
        return {"case": {}, "metrics": {}}
    case = record.get("case", {})
    if not isinstance(case, dict):
        case = {}
    if not case:
        case = {
            field: record[field]
            for field in ("category", "difficulty", "scenario_family", "tags")
            if field in record
        }
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        result = record.get("result")
        if isinstance(result, dict):
            metrics = evaluate_case(case, result)
        elif any(name in record for name in _MACRO_METRICS):
            # Also accept a direct case-metrics mapping for small ad-hoc runs.
            metrics = record
        else:
            metrics = {}
    return {"case": case, "metrics": metrics}


def aggregate_results(records):
    """Macro-average records overall and by category/difficulty/family.

    Records may already contain ``metrics`` or contain only ``case`` and
    ``result``.  Non-applicable (``None``) values are excluded independently
    from each macro denominator.
    """

    rows = [_record_row(record) for record in records]
    aggregated = {"overall": _summary(rows)}
    for field in ("category", "difficulty", "scenario_family"):
        buckets = {}
        for row in rows:
            value = row["case"].get(field)
            key = str(value) if value not in (None, "") else "unknown"
            buckets.setdefault(key, []).append(row)
        aggregated["by_" + field] = {
            key: _summary(buckets[key]) for key in sorted(buckets)
        }

    # ``overall`` is a per-case macro (therefore sample-weighted across
    # categories).  This view weights each category equally, which is useful
    # when benchmark category sizes intentionally differ.
    category_summaries = list(aggregated["by_category"].values())
    category_macro = {"num_categories": len(category_summaries)}
    for metric_name in _MACRO_METRICS + tuple(_BOOLEAN_RATES):
        values = [
            summary.get(metric_name) for summary in category_summaries
            if isinstance(summary.get(metric_name), (int, float))
        ]
        category_macro[metric_name] = _mean_or_none(values)
        category_macro[metric_name + "_evaluated_categories"] = len(values)
    aggregated["category_macro"] = category_macro
    return aggregated
