"""New configuration-transaction tasks, authored from public tool capabilities.

Never imports old benchmark generators, questions, oracle plans or answers.
Old eval is read only by the separate contamination check after generation.
"""

import hashlib
import itertools
import json
from copy import deepcopy

from agent.baseline_runner import SYSTEM_PROMPT
from evaluation.scoring import evaluate_case
from tools.environment_tools import reset_environment, inspect_ui_state
from tools.executor import execute_tool_json
from tools.tool_schema import TOOLS

SPLIT_SEED = "bugops-config-transactions-pilot-20260914"


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def blueprint_groups():
    groups = []
    for sequence, ending, fault in itertools.product(
            itertools.permutations(("low", "standard", "high")), ("home", "battle"), (False, True)):
        key = "/".join(sequence) + ":" + ending + ":" + str(int(fault))
        groups.append((key, sequence, ending, fault))
    groups.sort(key=lambda group: hashlib.sha256((SPLIT_SEED + group[0]).encode()).hexdigest())
    return [(("train" if i < 16 else "validation" if i < 20 else "confirmation"), *group)
            for i, group in enumerate(groups)]


def build_cases():
    rows = []
    for split, group, sequence, ending, fault in blueprint_groups():
        for platform, page, surface in itertools.product(
                ("android", "ios"), ("home", "settings", "graphics", "dungeon_select", "multiplayer_dungeon"), (0, 1)):
            group_id = hashlib.sha256(group.encode()).hexdigest()[:12]
            initial = {"version": "3.1.0", "platform": platform, "current_page": page,
                       "graphics_preset": "standard", "battle_hud_visible": False, "black_screen": False}
            environment = {"initial_state": initial}
            if fault:
                environment["fault_injections"] = [
                    {"operation": "action", "target": "set_graphics", "times": 1, "error": "transient_failure"}]
            actions = [call("inspect_ui_state")]
            if page not in {"home", "settings", "graphics"}:
                actions.append(call("execute_action", action="back_home"))
            if page not in {"settings", "graphics"}:
                actions.append(call("navigate_ui", target_page="settings"))
            if page != "graphics":
                actions.append(call("navigate_ui", target_page="graphics"))
            milestones = []
            for i, preset in enumerate(sequence):
                change = call("execute_action", action="set_graphics", value=preset)
                actions.append(change)
                if i == 0 and fault:
                    actions.extend([call("inspect_ui_state"), deepcopy(change)])
                milestones.append(change)
            back = call("execute_action", action="back_home")
            actions.append(back)
            milestones.append(back)
            if ending == "battle":
                actions += [call("navigate_ui", target_page="dungeon_select"),
                            call("navigate_ui", target_page="multiplayer_dungeon"),
                            call("execute_action", action="start_multiplayer_dungeon")]
                milestones.append(actions[-1])
            actions += [call("query_logs"), call("inspect_ui_state")]
            final = dict(initial, current_page=ending, graphics_preset=sequence[-1],
                         battle_hud_visible=ending == "battle")
            fault_text = ("第一次修改画质会临时失败；失败后必须读取实际状态，再在后续轮次重试。"
                          if fault else "本次未预设临时故障。")
            core = (f"在{platform} 3.1.0模拟实例上做配置事务验收，起始页面是{page}、画质standard。"
                    f"到图形页依次把画质设为{'、'.join(sequence)}，三次成功设置的顺序必须一致；"
                    + ("然后返回首页并停留。" if ending == "home" else "然后返回首页，再进入并启动多人副本。")
                    + fault_text + "查询本次日志并读取最终状态；报告最终页面、画质、HUD和黑屏，以及实际发生的失败次数。")
            query = core if surface == 0 else "请执行以下配置事务，并只依据实际工具返回提交验收摘要：\n" + core
            case = {
                "schema_version": 2, "protocol_version": "2.3", "split": split,
                "id": f"pilot_{group_id}_{platform}_{page}_{surface}",
                "scenario_id": f"pilot_{group_id}_{platform}_{page}", "group_id": group_id,
                "scenario_family": "configuration_transaction", "category": "recovery" if fault else "long_horizon",
                "difficulty": "hard", "query": query, "environment": environment,
                "required_calls": actions, "required_tools": [a["name"] for a in actions],
                "expected_tools": [a["name"] for a in actions], "optional_tools": ["verify_state", "get_build_info"],
                "max_steps": len(actions) + 5, "requires_recovery": fault,
                "process_constraints": ["observe_state_before_retry"] if fault else [],
                "public_contract": {"read_only": False, "no_navigation": False, "no_set_graphics": False,
                                    "knowledge_before_mutation": False, "build_evidence": False,
                                    "ordered_calls": milestones},
                "success_criteria": {
                    "final_state": final,
                    "required_tool_results": [
                        {"tool": "query_logs", "path": "$", "contains": f"set_graphics:{preset}"} for preset in sequence],
                    "final_answer": {"required": True},
                },
                "authoring_provenance": "independent_configuration_transaction_spec_from_public_tools",
            }
            rows.append(case)
    return rows


def oracle(case):
    reset_environment(case["environment"])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": case["query"]}]
    trajectory = []
    failures = 0
    for step, action in enumerate(case["required_calls"]):
        output = execute_tool_json(action)
        failures += int(not output["ok"])
        response = "<tool_call>\n" + json.dumps(action, ensure_ascii=False) + "\n</tool_call>"
        messages += [{"role": "assistant", "content": response},
                     {"role": "tool", "content": json.dumps(output, ensure_ascii=False)}]
        trajectory.append({"step": step, "model_output": response,
                           "parsed_tool_calls": [{"valid": True, "tool_call": action}],
                           "tool_results": [{"tool_call": action, "result": output}]})
    state = inspect_ui_state()
    answer = (f"配置事务完成。最终页面={state['current_page']}，画质={state['graphics_preset']}，"
              f"HUD={str(state['battle_hud_visible']).lower()}，黑屏={str(state['black_screen']).lower()}。"
              f"实际工具失败{failures}次；" + ("失败后检查状态并重试成功。" if failures else "本轮未发生工具失败。")
              + "已查询本轮日志；以上仅描述本次执行，不代表其他版本或场景已验证。")
    messages.append({"role": "assistant", "content": answer})
    result = {"query": case["query"], "trajectory": trajectory, "final_environment_state": state,
              "final_answer": answer, "terminated_reason": "final_answer"}
    metrics = evaluate_case(case, result)
    if not metrics["execution_success"] or failures != int(case["requires_recovery"]):
        raise ValueError(f"Invalid oracle: {case['id']}")
    return {"id": case["id"], "group_id": case["group_id"], "scenario_id": case["scenario_id"],
            "split": case["split"], "messages": messages, "tools": TOOLS,
            "teacher": "deterministic_executable_oracle_not_model_distillation",
            "result": result, "execution_validated": True}
