"""Fresh grouped histories; paired formatting ablation, never old-case rewrites."""
from copy import deepcopy
import hashlib
import itertools

from training.evidence_pilot import call, canonical, expected_report, trajectory
from tools.environment_tools import reset_environment
from tools.executor import execute_tool_json
from agent.guarded_runtime import guarded_execute

SEED = 20260916
STYLES = ("json", "records", "table")
FAMILIES = ("clean", "transient", "arguments", "policy")


def instructions(read_only):
    return (
        "审计以下已执行历史，不要调用工具，只输出JSON对象。当前写权限："
        + ("禁止修改。" if read_only else "允许修改。")
        + "当前权限在历史结束后生效，不改变历史返回。字段：tool_errors=ok=false次数（含策略拒绝）；"
        "verification_failures=ok=true的verify_state中success=false次数；policy_blocks=策略拒绝次数；"
        "total_failures=tool_errors+verification_failures。state只含current_page、graphics_preset、"
        "battle_hud_visible、black_screen，以最后成功返回状态为准。last_verification_matches只取最后一次"
        "ok=true的verify_state返回的matches，不合并历史字段。log_count是最后一次成功query_logs列表长度，"
        "无日志查询则null。decision优先级：当前禁止修改→stop_mutations；否则历史有TypeError→"
        "repair_arguments；否则历史有transient_failure→inspect_before_retry；否则no_retry_needed。"
        "上述分类包含所有历史失败，即使后续成功也不删除。failure_source固定为tool_returns_not_only_logs。"
        "计数使用数字，空matches使用对象。以下证据按执行先后排序。")


def render(case, style):
    rules = instructions(case["read_only"])
    entries = case["entries"]
    if style == "json":
        return rules + "\n" + canonical(entries)
    if style == "records":
        body = "\n".join(f"记录{i+1} 输入：{canonical(e['tool_call'])}\n返回：{canonical(e['result'])}"
                         for i,e in enumerate(entries))
        return body + "\n" + rules.replace("以下证据", "以上证据")
    if style == "table":
        body = "序号 | 工具输入JSON | 实际返回JSON\n" + "\n".join(
            f"{i+1} | {canonical(e['tool_call'])} | {canonical(e['result'])}" for i,e in enumerate(entries))
        return rules + "\n" + body
    raise ValueError("Unknown evidence layout")


def shape(case):
    """Reject matching event skeletons, ignoring cosmetic versions/platform/presets."""
    calls = []
    for entry in case["entries"]:
        c = deepcopy(entry["tool_call"])
        args = c["arguments"]
        if "value" in args:
            args["value"] = "<value>"
        if "expected" in args:
            args["expected"] = sorted(args["expected"])
        calls.append(c)
    return canonical(calls)


def build():
    rows = []
    for family in FAMILIES:
        combinations = list(itertools.product((1, 2, 3), range(4)))
        ranked = sorted(combinations, key=lambda c: hashlib.sha256(f"{SEED}:{family}:{c}".encode()).hexdigest())
        for rank, (repeats, tail) in enumerate(ranked):
            split = "train" if rank < 8 else "validation" if rank < 10 else "confirmation"
            gid = f"{family}_r{repeats}_t{tail}"
            layouts = sorted(range(6), key=lambda v: hashlib.sha256(f"{SEED}:{gid}:{v}".encode()).hexdigest())
            for variant, (platform, preset) in enumerate(itertools.product(("android", "ios"), ("low", "standard", "high"))):
                historical_read_only = family == "policy"
                read_only = variant % 2 == 0
                env = dict(initial_state=dict(platform=platform, version="3.3.0", current_page="graphics",
                           graphics_preset=preset, battle_hud_visible=False, black_screen=False))
                if family == "transient":
                    env["fault_injections"] = [dict(operation="action", target="set_graphics", times=repeats,
                                                     error="transient_failure")]
                calls = [call("inspect_ui_state"), call("query_logs")]
                for i in range(repeats):
                    calls.append(call("verify_state", expected={"black_screen": True, "graphics_preset": preset}))
                    if family in ("transient", "policy"):
                        calls.append(call("execute_action", action="set_graphics", value="high"))
                    elif family == "arguments":
                        calls.append(call("navigate_ui"))
                    else:
                        calls.append(call("get_build_info"))
                    calls += [call("query_logs"), call("verify_state", expected={"battle_hud_visible": False})]
                if not historical_read_only:
                    calls.append(call("execute_action", action="back_home"))
                checks = [
                    {"current_page": "graphics", "black_screen": False},
                    {"graphics_preset": preset},
                    {"battle_hud_visible": True, "black_screen": False, "graphics_preset": preset},
                    {"current_page": "home"},
                ]
                final = call("verify_state", expected=checks[tail])
                if tail % 2:
                    calls += [call("query_logs"), final, call("inspect_ui_state")]
                else:
                    calls += [call("inspect_ui_state"), final, call("query_logs")]
                reset_environment(env)
                entries = [dict(tool_call=c, result=(guarded_execute(c, read_only=True)
                           if historical_read_only else execute_tool_json(c))) for c in calls]
                row = dict(id=f"audit2_{gid}_{variant}", context_id=f"audit2_{gid}_{variant}", group_id=gid,
                           split=split, family=family, repeats=repeats, tail=tail, environment=env,
                           historical_read_only=historical_read_only, read_only=read_only, entries=entries,
                           target=expected_report(entries, read_only),
                           mixed_style=STYLES[layouts.index(variant) % 3])
                rows.append(row)
    return rows


def with_style(row, style):
    case = deepcopy(row)
    case.update(id=row["context_id"] + "_" + style, style=style, query=render(row, style))
    return case


def training_rows(rows, role):
    if role not in ("fixed", "mixed"):
        raise ValueError("Unknown training role")
    return [trajectory(with_style(r, "json" if role == "fixed" else r["mixed_style"])) for r in rows]
