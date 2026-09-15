"""Opt-in application policy and evidence-only reporting, not a model improvement.

Policy comes from a trusted application setting, NOT the model, tool output,
success criteria, or hidden environment. This pilot supports read-only mode only.
"""
from copy import deepcopy
import json

from agent.baseline_runner import SYSTEM_PROMPT
from agent.tool_parser import parse_tool_calls
from tools.executor import execute_tool_json
from tools.environment_tools import reset_environment, inspect_ui_state

READ_TOOLS = frozenset({"get_build_info", "inspect_ui_state", "verify_state",
                        "query_logs", "search_knowledge"})
STATE_TOOLS = frozenset({"get_build_info", "inspect_ui_state", "verify_state",
                         "navigate_ui", "execute_action"})
FIELDS = {"current_page": str, "graphics_preset": str,
          "battle_hud_visible": bool, "black_screen": bool}


def guarded_execute(call, *, read_only, executor=execute_tool_json):
    if type(read_only) is not bool:
        raise ValueError("Explicit trusted boolean read_only required")
    if not isinstance(call, dict):
        return {"ok": False, "error_type": "policy_blocked",
                "error": "Malformed call blocked before dispatch"}
    if read_only and (not isinstance(call.get("name"), str) or call.get("name") not in READ_TOOLS):
        return {"ok": False, "error_type": "policy_blocked",
                "error": "Application read-only policy forbids this tool. Use read-only tools; do not navigate or act."}
    return executor(deepcopy(call))


def facts(trajectory):
    """Only observed returns; no case, reference answer or saved final state input."""
    state, failures, logs, mutations = {}, [], [], []
    checks = blocks = count = 0
    last_attempt = -1
    for turn in trajectory:
        for entry in turn.get("tool_results", []):
            name, response = entry["tool_call"].get("name"), entry["result"]
            idx = count
            count += 1
            ref = dict(execution_index=idx, step=turn.get("step"), tool=name)
            if name in {"navigate_ui", "execute_action"}:
                last_attempt = idx
            if response.get("ok") is False:
                blocked = response.get("error_type") == "policy_blocked"
                blocks += int(blocked)
                failures.append(dict(**ref, kind="policy_blocked" if blocked else "tool_error",
                                     response=deepcopy(response)))
                continue
            if response.get("ok") is not True:
                raise ValueError("Unknown tool-result status; cannot publish exact facts")
            value = response.get("result")
            if name in {"navigate_ui", "execute_action"}:
                state.clear()
                mutations.append(ref)
            if name == "inspect_ui_state":
                checks += 1
            if name == "verify_state" and isinstance(value, dict):
                if value.get("success") is False:
                    failures.append(dict(**ref, kind="verification_failure", response=deepcopy(response)))
                value = value.get("current_state")
            if name in STATE_TOOLS and isinstance(value, dict):
                for field, kind in FIELDS.items():
                    if field in value and type(value[field]) is kind:
                        state[field] = dict(**ref, value=value[field])
                    elif field in value:
                        state.pop(field, None)
            if name == "query_logs" and isinstance(value, list):
                logs.append(dict(**ref, entries=deepcopy(value)))
    return dict(observed_state=state, missing_fields=[f for f in FIELDS if f not in state],
                failure_count=len(failures), failure_evidence=failures, policy_block_count=blocks,
                successful_inspect_calls=checks, successful_mutations=mutations,
                log_evidence=logs, logs_after_last_attempt=any(
                    x["execution_index"] > last_attempt for x in logs),
                completion_verdict="not_assigned", scope="synchronous_atomic_simulator_only")


def render_facts(data):
    labels = {"current_page": "最终页面", "graphics_preset": "画质",
              "battle_hud_visible": "HUD", "black_screen": "黑屏"}
    parts = ["系统工具事实摘要（不是模型原回答，也不自动证明任务完成）："]
    for field, label in labels.items():
        item = data["observed_state"].get(field)
        value = "未知（缺少有效观测）" if item is None else json.dumps(item["value"], ensure_ascii=False)
        parts.append(f"{label}={value}。")
    parts.append(f"实际失败{data['failure_count']}次，其中策略拦截{data['policy_block_count']}次；"
                 f"inspect_ui_state成功调用{data['successful_inspect_calls']}次。")
    parts.append(f"成功导航/应用动作{len(data['successful_mutations'])}次；"
                 f"成功日志查询{len(data['log_evidence'])}次。")
    parts.append("最后一次应用操作尝试后的日志证据：" +
                 ("已取得。" if data["logs_after_last_attempt"] else "未取得。"))
    parts.append("失败统计来自完整工具返回（含验证失败），不是仅从应用日志计数。")
    return "\n".join(parts)


def run_guarded(generate, query, environment, max_steps, *, read_only=True):
    if type(read_only) is not bool or type(max_steps) is not int or max_steps < 1:
        raise ValueError("Invalid policy or step budget")
    reset_environment(environment)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]
    trajectory, answer = [], None
    for step in range(max_steps):
        response = generate(messages)
        parsed = parse_tool_calls(response)
        record = dict(step=step, model_output=response, parsed_tool_calls=parsed, tool_results=[])
        trajectory.append(record)
        calls = [p["tool_call"] for p in parsed if p.get("valid")]
        if not calls:
            answer = response
            break
        messages.append(dict(role="assistant", content=response))
        for call in calls:
            result = guarded_execute(call, read_only=read_only)
            record["tool_results"].append(dict(tool_call=call, result=result))
            messages.append(dict(role="tool", content=json.dumps(result, ensure_ascii=False)))
    return dict(query=query, trajectory=trajectory, final_answer=answer,
                final_environment_state=inspect_ui_state(),
                terminated_reason="final_answer" if answer is not None else "max_steps")
