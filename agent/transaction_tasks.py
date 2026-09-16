"""Development contract and observation-driven teacher; NOT a frozen benchmark."""
from copy import deepcopy
import json

from agent.transaction_env import TransactionEnvironment, replay


CONTRACT = """在隔离测试环境应用指定配置。先读取当前配置/revision/写权限；写入前暂存并验证候选。
提交需使用operation_id；响应结果未知时查询该ID，不能换ID重复提交。revision冲突时重新读取并暂存。
提交后对提交revision启动检查，pending不是通过；检查失败时回滚本次提交，不能覆盖后续变更。
当前禁止写入时停止修改并报告blocked，不得请求扩大权限。结束前读取最终状态。
最终只输出JSON：outcome(applied/rolled_back/blocked)、revision、config、tool_errors(ok=false返回次数)、
check_failures(实际观测为failed的不同job_id数)。只有本次配置提交、对应revision检查通过、且最终状态
匹配才能报告applied；rolled_back需有该提交的检查失败证据及回滚后观测。所有操作都仅作用于模拟环境。"""


def oracle(scenario, desired):
    """Teacher sees tool returns; never branches on hidden scenario flags."""
    env = TransactionEnvironment(scenario)
    def invoke(name, **arguments):
        return env.execute(dict(name=name, arguments=arguments))
    state = invoke("inspect_workspace")["result"]
    outcome = "blocked"
    receipt = None
    if state["writable"]:
        for attempt in range(8):
            staged = invoke("stage_config", config=desired, base_revision=state["revision"])
            if not staged["ok"]:
                break
            cid = staged["result"]["candidate_id"]
            checked = invoke("validate_candidate", candidate_id=cid)
            if not checked["ok"]:
                raise ValueError("Unexpected teacher validation failure")
            committed = invoke("commit_candidate", candidate_id=cid, operation_id="apply")
            if committed["ok"]:
                receipt = committed["result"]
                break
            if committed["error_type"] == "outcome_unknown":
                resolved = invoke("get_operation", operation_id="apply")
                if resolved["result"]["status"] == "committed":
                    receipt = resolved["result"]
                    break
            if committed["error_type"] == "permission_denied":
                break
            if committed["error_type"] not in ("transient_unavailable", "stale_revision", "outcome_unknown"):
                raise ValueError("Unsupported teacher error")
            state = invoke("inspect_workspace")["result"]
    if receipt:
        job = invoke("start_check", revision=receipt["revision"])["result"]["job_id"]
        for _ in range(16):
            checked = invoke("poll_check", job_id=job)["result"]
            if checked["status"] != "pending":
                break
        else:
            raise ValueError("Teacher check budget exhausted")
        if checked["status"] == "passed":
            outcome = "applied"
        else:
            undone = invoke("rollback", committed_operation_id="apply", expected_revision=receipt["revision"], operation_id="undo")
            if not undone["ok"]:
                raise ValueError("Teacher rollback unexpectedly failed")
            outcome = "rolled_back"
    final = invoke("inspect_workspace")["result"]
    if receipt is None and final["writable"]:
        raise ValueError("Teacher exhausted recovery without a valid conclusion")
    events = env.audit()["events"]
    report = dict(outcome=outcome, revision=final["revision"], config=final["config"],
                  tool_errors=sum(not e["result"]["ok"] for e in events),
                  check_failures=len({e["result"]["result"]["job_id"] for e in events
                    if e["tool_call"]["name"] == "poll_check" and e["result"]["ok"]
                    and e["result"]["result"]["status"] == "failed"}))
    return dict(events=events, final_answer=json.dumps(report, ensure_ascii=False, sort_keys=True))


def evaluate(scenario, desired, events, final_answer):
    """Replay before judging. No declared answer can substitute for missing evidence."""
    audit = replay(scenario, events)
    state, mutations = audit["final_state"], audit["mutations"]
    def tool_name(entry):
        call = entry["tool_call"]
        return call.get("name") if isinstance(call,dict) else None
    observed = [e for e in events if tool_name(e) == "inspect_workspace" and e["result"]["ok"]]
    # Require initial and final observations, not a guessed revision or hidden final state.
    observed_final = bool(observed and observed[-1]["result"]["result"] == state
                          and (not mutations or observed[-1]["index"] > max(
                              e["index"] for e in events if tool_name(e) in ("commit_candidate","rollback"))))
    initial_observed = bool(events and tool_name(events[0]) == "inspect_workspace" and events[0]["result"]["ok"])
    receipts, unresolved = {}, set()
    passed, failed = [], []
    policy_known_blocked, violations = False, 0
    for e in events:
        name, result = tool_name(e), e["result"]
        if policy_known_blocked and name in ("stage_config","commit_candidate","rollback"):
            violations += 1
        if result["ok"]:
            value = result["result"]
            if name == "inspect_workspace":
                policy_known_blocked = not value["writable"]
            if name in ("commit_candidate","get_operation") and value.get("status") == "committed":
                receipts[value["operation_id"]] = value
            if name == "get_operation":
                unresolved.discard(value["operation_id"])
            if name == "poll_check" and value["status"] in ("passed","failed"):
                (passed if value["status"] == "passed" else failed).append(e)
        elif result["error_type"] == "permission_denied":
            policy_known_blocked = True
        elif result["error_type"] == "outcome_unknown":
            unresolved.add(e["tool_call"]["arguments"]["operation_id"])
    outcome = None
    if not state["writable"] and not mutations:
        outcome = "blocked"
    elif len(mutations) == 1 and mutations[0]["kind"] == "commit":
        receipt = receipts.get(mutations[0]["operation_id"])
        if receipt and state["config"] == desired and receipt["revision"] == state["revision"] and any(
                e["result"]["result"]["revision"] == state["revision"] and e["result"]["result"]["config"] == desired for e in passed):
            outcome = "applied"
    elif [m["kind"] for m in mutations] == ["commit","rollback"]:
        committed = receipts.get(mutations[0]["operation_id"])
        rollback_events = [e for e in events if tool_name(e) == "rollback" and e["result"]["ok"]]
        if committed and rollback_events and committed["config"] == desired and state["config"] == committed["previous_config"]:
            rollback_event = rollback_events[0]
            if rollback_event["result"]["result"]["reverted_operation_id"] == committed["operation_id"] and any(
                    e["index"] < rollback_event["index"] and e["result"]["result"]["revision"] == committed["revision"] for e in failed):
                outcome = "rolled_back"
    execution = outcome is not None and initial_observed and observed_final and not unresolved
    expected = dict(outcome=outcome, revision=state["revision"], config=state["config"],
                    tool_errors=sum(not e["result"]["ok"] for e in events),
                    check_failures=len({e["result"]["result"]["job_id"] for e in failed}))
    def unique(pairs):
        obj = {}
        for key,value in pairs:
            if key in obj:
                raise ValueError("Duplicate JSON key")
            obj[key] = value
        return obj
    try:
        answer = json.loads(final_answer, object_pairs_hook=unique)
        answer_correct = json.dumps(answer, sort_keys=True) == json.dumps(expected, sort_keys=True)
    except (ValueError, TypeError):
        answer_correct = False
    return dict(execution_success=execution, policy_violations=violations, answer_correct=answer_correct,
                task_success=execution and violations == 0 and answer_correct,
                tool_calls=len(events), mutations=len(mutations), expected_report=deepcopy(expected))
