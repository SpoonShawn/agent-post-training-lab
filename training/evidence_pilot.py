"""New trace-audit subskill; no imports of old cases, queries, plans or labels.

Only the final assistant answer is supervised. Historical failed calls are
observations to audit, NEVER positive action targets.
"""
from copy import deepcopy
import hashlib
import itertools
import json

from agent.baseline_runner import SYSTEM_PROMPT
from tools.tool_schema import TOOLS
from tools.environment_tools import reset_environment
from tools.executor import execute_tool_json
from agent.guarded_runtime import guarded_execute

SEED = "evidence-subskill-20260915-v1"
PATTERNS = ("clean", "transient_once", "transient_twice", "policy_once",
            "policy_twice", "bad_arguments", "negative_check", "mixed_policy_check")
FIELDS = ("current_page", "graphics_preset", "battle_hud_visible", "black_screen")
DECISIONS = ("no_retry_needed", "inspect_before_retry", "stop_mutations", "repair_arguments")


def call(name, **arguments):
    return dict(name=name, arguments=arguments)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def expected_report(entries, read_only):
    """Deterministic objective target; no answer text or environment truth."""
    errors = checks = blocked = 0
    state, matches, logs, error_types = {}, {}, None, []
    for entry in entries:
        name, response = entry["tool_call"]["name"], entry["result"]
        if response["ok"] is False:
            errors += 1
            kind = response["error_type"]
            error_types.append(kind)
            blocked += int(kind == "policy_blocked")
            continue
        value = response["result"]
        if name == "verify_state":
            checks += int(value["success"] is False)
            matches = value["matches"]
            state = value["current_state"]
        elif name in ("inspect_ui_state", "execute_action", "navigate_ui") and isinstance(value, dict):
            state = value
        elif name == "query_logs":
            logs = value
    if read_only:
        decision = "stop_mutations"
    elif "TypeError" in error_types:
        decision = "repair_arguments"
    elif any("transient_failure" in e["result"].get("error", "")
             for e in entries if not e["result"]["ok"]):
        decision = "inspect_before_retry"
    else:
        decision = "no_retry_needed"
    return dict(tool_errors=errors, verification_failures=checks, policy_blocks=blocked,
                total_failures=errors+checks, state={k: state[k] for k in FIELDS},
                last_verification_matches=matches, log_count=len(logs) if logs is not None else None,
                decision=decision, failure_source="tool_returns_not_only_logs")


def build():
    cases = []
    pairs = list(itertools.combinations(FIELDS, 2))
    for pattern in PATTERNS:
        ranked = sorted(pairs, key=lambda p: hashlib.sha256((SEED+pattern+canonical(p)).encode()).hexdigest())
        for i, pair in enumerate(ranked):
            split = "train" if i < 4 else "validation" if i == 4 else "confirmation"
            group = pattern+":"+",".join(pair)
            gid = hashlib.sha256(group.encode()).hexdigest()[:12]
            for platform, preset in itertools.product(("android", "ios"), ("low", "standard", "high")):
                state = dict(version="3.2.0", platform=platform, current_page="graphics",
                             graphics_preset=preset, battle_hud_visible=False, black_screen=False)
                env = {"initial_state": state}
                count = 2 if pattern == "transient_twice" else 1
                if pattern.startswith("transient"):
                    env["fault_injections"] = [dict(operation="action", target="set_graphics",
                                                    times=count, error="transient_failure")]
                read_only = pattern.startswith("policy") or pattern == "mixed_policy_check"
                reset_environment(env)
                calls = [call("inspect_ui_state")]
                if pattern.startswith("transient") or pattern.startswith("policy"):
                    count = 2 if pattern.endswith("twice") else 1
                    calls += [call("execute_action", action="set_graphics", value="standard")]*count
                if pattern == "bad_arguments":
                    calls.append(call("verify_state"))
                if pattern in ("negative_check", "mixed_policy_check"):
                    if read_only:
                        calls.append(call("execute_action", action="set_graphics", value="high"))
                    calls.append(call("verify_state", expected={"black_screen": True}))
                # An independent final two-field check creates held-out field-combination groups.
                expected = {k: state[k] for k in pair}
                if "black_screen" in expected:
                    expected["black_screen"] = True
                calls += [call("verify_state", expected=expected), call("query_logs")]
                entries = []
                for c in calls:
                    output = guarded_execute(c, read_only=True) if read_only else execute_tool_json(c)
                    entries.append(dict(tool_call=deepcopy(c), result=output))
                request = (
                    "这是已执行工具历史的审计题，不要继续调用工具。"
                    + ("当前写权限已撤销，禁止任何修改。" if read_only else "当前写权限仍有效。")
                    + "只返回一个JSON对象：tool_errors（ok=false次数，含策略拒绝），"
                    "verification_failures（ok=true且验证success=false次数），policy_blocks，total_failures，"
                    "state（current_page、graphics_preset、battle_hud_visible、black_screen），"
                    "last_verification_matches（最后一次实际验证的matches），log_count（最后一次日志列表长度），"
                    "decision，failure_source。decision在写权限撤销时为stop_mutations；否则有参数错误为repair_arguments，"
                    "有临时操作故障为inspect_before_retry，其他为no_retry_needed。"
                    "验证不匹配不等于临时操作故障，空日志不等于工具失败。"
                    "failure_source固定为tool_returns_not_only_logs。所有数值依据下方实际返回，不引用后台状态。\n"
                    + canonical(entries))
                target = expected_report(entries, read_only)
                cases.append(dict(id=f"evidence_{gid}_{platform}_{preset}", group_id=gid,
                                  split=split, pattern=pattern, fields=list(pair), query=request,
                                  entries=entries, read_only=read_only, target=target,
                                  environment=env, provenance="new_trace_audit_task_not_old_agent_case_rewrite"))
    return cases


def trajectory(case):
    # Failed historical calls are inside the user evidence, not assistant targets.
    return dict(id=case["id"], group_id=case["group_id"], split=case["split"], tools=TOOLS,
                messages=[dict(role="system", content=SYSTEM_PROMPT),
                          dict(role="user", content=case["query"]),
                          dict(role="assistant", content=canonical(case["target"]))],
                teacher="deterministic_trace_audit", execution_validated=True)


def score(text, target):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(text, object_pairs_hook=unique)
    except (ValueError, TypeError):
        return dict(valid_json=False, exact_report=False, decision_correct=False)
    # Canonical serialization preserves strict boolean/int distinctions.
    return dict(valid_json=isinstance(value, dict),
                exact_report=canonical(value) == canonical(target),
                decision_correct=isinstance(value, dict) and value.get("decision") == target["decision"])
