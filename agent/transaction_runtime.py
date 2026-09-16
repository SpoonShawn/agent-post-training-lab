"""Model-independent, budgeted episode runner for the new development contract."""
from copy import deepcopy
import json
import re

from agent.transaction_env import TransactionEnvironment
from agent.transaction_tasks import CONTRACT


def schema(name, description, properties):
    return dict(name=name, description=description, parameters=dict(type="object", properties=properties,
                required=list(properties), additionalProperties=False))


STRING = {"type":"string", "minLength":1, "maxLength":80}
REVISION = {"type":"integer", "minimum":1}
CONFIG = {"type":"object", "properties":{"quality":{"type":"string","enum":["low","standard","high"]},
          "cache_enabled":{"type":"boolean"}}, "required":["quality","cache_enabled"], "additionalProperties":False}
TOOLS = [
    schema("inspect_workspace", "读取当前config、revision和writable；不返回故障计划。", {}),
    schema("stage_config", "基于当前revision暂存完整候选配置，不应用。权限禁止或revision过期会拒绝。", {"config":CONFIG,"base_revision":REVISION}),
    schema("validate_candidate", "检查候选有效性；必须在提交前成功。过期候选需重新暂存。", {"candidate_id":STRING}),
    schema("commit_candidate", "应用已验证候选。operation_id幂等，同ID不能换请求；outcome_unknown必须查询该ID。stale_revision需重新读取并暂存。", {"candidate_id":STRING,"operation_id":STRING}),
    schema("get_operation", "只读查询operation_id，返回committed/rolled_back回执或not_found。超时不等于未执行。", {"operation_id":STRING}),
    schema("start_check", "对当前revision启动异步检查，返回job_id和pending；不改变配置。", {"revision":REVISION}),
    schema("poll_check", "查询任务，返回pending/passed/failed。结果绑定启动时的revision和配置，不证明其他revision有效。", {"job_id":STRING}),
    schema("rollback", "将指定成功提交恢复为提交前配置。expected_revision必须匹配当前与被回滚提交的revision，不覆盖后续修改；operation_id幂等。", {"committed_operation_id":STRING,"expected_revision":REVISION,"operation_id":STRING}),
]
SYSTEM = CONTRACT + "\n每轮最多输出一个<tool_call>{\"name\":工具名,\"arguments\":参数对象}</tool_call>，或最终报告JSON。"


def strict_json(text):
    def unique(pairs):
        obj = {}
        for k,v in pairs:
            if k in obj:
                raise ValueError("Duplicate JSON key")
            obj[k] = v
        return obj
    return json.loads(text, object_pairs_hook=unique)


def run_episode(generate, query, scenario, *, max_turns=40, max_calls=36):
    if any(type(v) is not int or v < 1 for v in (max_turns,max_calls)):
        raise ValueError("Positive integer budgets required")
    env = TransactionEnvironment(scenario)
    messages = [dict(role="system",content=SYSTEM),dict(role="user",content=query)]
    turns, answer, reason, calls = [], None, "max_turns", 0
    for step in range(max_turns):
        # The backend sees only public prompt/history and public schemas, never scenario/audit/target.
        response = generate(deepcopy(messages), deepcopy(TOOLS))
        if not isinstance(response,str):
            raise TypeError("Generation backend must return text")
        messages.append(dict(role="assistant",content=response))
        turn = dict(step=step,model_output=response)
        turns.append(turn)
        if "<tool_call" not in response and "</tool_call>" not in response:
            answer, reason = response, "final_answer"
            break
        if calls >= max_calls:
            reason = "max_calls"
            break
        match = re.fullmatch(r"\s*<tool_call>\s*(.*?)\s*</tool_call>\s*", response, re.DOTALL)
        try:
            call = strict_json(match.group(1)) if match else None
        except (ValueError,TypeError):
            call = None
        result = env.execute(call)
        calls += 1
        turn.update(tool_call=deepcopy(call),result=deepcopy(result))
        messages.append(dict(role="tool",content=json.dumps(result,ensure_ascii=False,sort_keys=True)))
    return dict(query=query,turns=turns,events=env.audit()["events"],final_answer=answer,
                terminated_reason=reason,tool_calls=calls)
