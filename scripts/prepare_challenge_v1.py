"""Create a post-training diagnostic set. Never exports training trajectories."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.pilot_data import call, oracle
from scripts.run_baseline import runtime_sha256

PAGES = ("home", "settings", "graphics", "dungeon_select", "multiplayer_dungeon")
REPORT = "查询本轮日志，读取最终状态；报告最终页面、画质、HUD、黑屏和实际工具失败次数。只陈述本次工具证据支持的事实。"


def transaction(template, sequence, ending, page):
    case = deepcopy(template)
    initial = dict(version="3.1.0", platform="android", current_page=page,
                   graphics_preset="standard", battle_hud_visible=False, black_screen=False)
    case["environment"] = {"initial_state": initial}
    actions = [call("inspect_ui_state")]
    if page not in ("home", "settings", "graphics"):
        actions.append(call("execute_action", action="back_home"))
    if page not in ("settings", "graphics"):
        actions.append(call("navigate_ui", target_page="settings"))
    if page != "graphics":
        actions.append(call("navigate_ui", target_page="graphics"))
    changes = [call("execute_action", action="set_graphics", value=x) for x in sequence]
    actions += changes + [call("execute_action", action="back_home")]
    milestones = changes + [actions[-1]]
    if ending == "battle":
        actions += [call("navigate_ui", target_page="dungeon_select"),
                    call("navigate_ui", target_page="multiplayer_dungeon"),
                    call("execute_action", action="start_multiplayer_dungeon")]
        milestones.append(actions[-1])
    actions += [call("query_logs"), call("inspect_ui_state")]
    case.update(required_calls=actions, requires_recovery=False, process_constraints=[])
    case["public_contract"] = {k: False for k in case["public_contract"] if k != "ordered_calls"}
    case["public_contract"]["ordered_calls"] = milestones
    case["success_criteria"] = {
        "final_state": dict(initial, current_page=ending, graphics_preset=sequence[-1],
                            battle_hud_visible=ending == "battle"),
        "required_tool_results": [{"tool": "query_logs", "path": "$",
                                   "contains": [f"set_graphics:{x}" for x in sequence]}],
        "final_answer": {"required": True}}
    case["query"] = (f"android 3.1.0，起点{page}，初始画质standard。请按顺序成功设置画质："
                     + " → ".join(sequence) + "。重复出现的档位也必须分别执行，不能跳过。"
                     + ("随后返回首页并停留。" if ending == "home"
                        else "随后先返回首页，再进入并启动多人副本。")
                     + "本次没有预设临时故障。" + REPORT)
    return case


def finish(case, dimension, group, suffix):
    case.update(id=f"challenge_{dimension}_{group}_{suffix}", group_id=f"{dimension}_{group}",
                scenario_id=f"challenge_{dimension}_{group}_{suffix}",
                scenario_family=f"challenge_{dimension}", challenge_dimension=dimension,
                category=dimension, split="post_training_diagnostic",
                authoring_provenance="stage017_post_training_stress_test_not_independent_holdout")
    case["required_tools"] = [c["name"] for c in case["required_calls"]]
    case["expected_tools"] = case["required_tools"][:]
    # Equal model budgets; expression pairs retain the original case budget.
    if dimension != "expression":
        case["max_steps"] = len(case["required_calls"]) + 5
    return case


def build():
    sources = [json.loads(line) for line in (ROOT / "data/pilot_v1/confirmation_cases.jsonl").read_text().splitlines()]
    template = sources[0]
    cases = []
    # 20 original contexts, each original wording + changed wording. Reuse disclosed.
    for source in sources:
        if "_android_" not in source["id"] or not source["id"].endswith("_0"):
            continue
        for variant in ("control", "rewrite"):
            case = deepcopy(source)
            case["source_case_id"] = source["id"]
            case["pair_id"] = source["id"]
            case["expression_variant"] = variant
            if variant == "rewrite":
                # Reorder complete clauses without changing any requested requirement.
                clauses = source["query"].split("。")
                case["query"] = "请按下面验收单执行，叙述顺序不代表操作顺序。\n" + "。\n".join(
                    [clauses[-2], clauses[1], clauses[0], clauses[2]]) + "。"
            cases.append(finish(case, "expression", source["group_id"],
                                source["environment"]["initial_state"]["current_page"] + "_" + variant))
    sequences = [("low",), ("high", "low"), ("low", "high", "low", "standard"),
                 ("standard", "low", "high", "low", "standard")]
    for n, sequence in enumerate(sequences):
        for ending in ("home", "battle"):
            for page in PAGES:
                cases.append(finish(transaction(template, sequence, ending, page),
                                    "structure", f"length{len(sequence)}_{ending}", page))
    for kind in ("read_only", "no_set_graphics"):
        for platform in ("android", "ios"):
            for page in PAGES:
                case = transaction(template, ("low",), "home", page)
                initial = case["environment"]["initial_state"]
                initial["platform"] = platform
                actions = [call("inspect_ui_state")]
                if kind == "no_set_graphics":
                    actions.append(call("execute_action", action="back_home"))
                actions += [call("query_logs"), call("inspect_ui_state")]
                case["required_calls"] = actions
                case["public_contract"][kind] = True
                case["public_contract"]["ordered_calls"] = actions[1:2] if kind == "no_set_graphics" else []
                case["success_criteria"]["final_state"] = dict(initial, current_page=page if kind == "read_only" else "home")
                case["success_criteria"]["required_tool_results"] = [
                    {"tool": "query_logs", "path": "$", "contains": [] if kind == "read_only" else ["back_home"]}]
                case["query"] = (f"{platform} 3.1.0，当前页面{page}、画质standard。"
                                 + ("全程只读：禁止导航和所有应用动作，即使恢复原状态也不允许。保持当前状态。"
                                    if kind == "read_only" else "返回首页并停留，全程禁止set_graphics，不得改变画质。")
                                 + REPORT)
                cases.append(finish(case, "constraints", kind, platform + "_" + page))
    for operation, target in (("navigate", "graphics"), ("navigate", "multiplayer_dungeon"),
                              ("action", "back_home"), ("action", "start_multiplayer_dungeon")):
        for page in ("home", "settings", "dungeon_select", "multiplayer_dungeon", "battle"):
            case = transaction(template, ("low", "high"), "battle", page)
            case["environment"]["fault_injections"] = [dict(operation=operation, target=target,
                                                           times=1, error="transient_failure")]
            name = "navigate_ui" if operation == "navigate" else "execute_action"
            key = "target_page" if operation == "navigate" else "action"
            actions = case["required_calls"]
            i = next(i for i, c in enumerate(actions) if c["name"] == name and c["arguments"].get(key) == target)
            actions[i+1:i+1] = [call("inspect_ui_state"), deepcopy(actions[i])]
            case.update(requires_recovery=True, process_constraints=["observe_state_before_retry"])
            case["query"] = case["query"].replace(
                "本次没有预设临时故障。",
                f"首次合法执行{operation}:{target}将临时失败一次；失败后必须读取实际状态，并在后续轮次重试。")
            cases.append(finish(case, "recovery", operation + "_" + target, page))
    return cases


def main():
    cases = build()
    for case in cases:
        oracle(case)
    destination = ROOT / "data/challenge_v1"
    if destination.exists():
        raise ValueError("Challenge already exists; do not overwrite frozen files")
    destination.mkdir(parents=True)
    path = destination / "cases.jsonl"
    path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
    from collections import Counter
    manifest = dict(version=1, status="frozen_before_first_challenge_inference",
                    scope="post-training diagnostic; expression controls reuse pilot confirmation",
                    cases=len(cases), dimensions=dict(Counter(c["challenge_dimension"] for c in cases)),
                    groups=len({c["group_id"] for c in cases}),
                    cases_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    runtime_sha256=runtime_sha256(),
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    model_sha256="8a3508ed8739b8816bb69bc671370ade2bcd742abcd6be316d9af6036625f68c",
                    adapter_sha256="54d09ba7ca62db64cba27299a1ff586181f92bfd65028336013234856667e85a",
                    generation=dict(do_sample=False, max_new_tokens=512, max_steps_override=None),
                    oracle_execution_pass=len(cases))
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
