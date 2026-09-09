#!/usr/bin/env python3
"""Generate and validate the deterministic BugOps held-out benchmark v2."""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.executor import execute_tool  # noqa: E402
from tools.environment_tools import inspect_ui_state, reset_environment  # noqa: E402


SCHEMA_VERSION = 2
SPLIT = "held_out"
VARIANTS_PER_FAMILY = 30
EXPECTED_CASE_COUNT = 360
DEFAULT_OUTPUT = ROOT / "data" / "eval" / "bugops_eval_v2.jsonl"

ALLOWED_TOOLS = {
    "search_knowledge",
    "get_build_info",
    "inspect_ui_state",
    "navigate_ui",
    "execute_action",
    "query_logs",
    "verify_state",
}

FAMILY_ORDER = [
    "no_tool",
    "build_info",
    "ui_state",
    "knowledge_black_screen",
    "knowledge_hud",
    "navigate_graphics",
    "reproduce_black_screen",
    "negative_control_ios",
    "workaround_black_screen",
    "recovery_navigation",
    "recovery_action",
    "return_home_then_dungeon",
]

FAMILY_CATEGORIES = {
    "no_tool": "no_tool",
    "build_info": "single_tool",
    "ui_state": "single_tool",
    "knowledge_black_screen": "knowledge",
    "knowledge_hud": "knowledge",
    "navigate_graphics": "navigation",
    "reproduce_black_screen": "long_horizon",
    "negative_control_ios": "long_horizon",
    "workaround_black_screen": "long_horizon",
    "recovery_navigation": "recovery",
    "recovery_action": "recovery",
    "return_home_then_dungeon": "long_horizon",
}

EXPECTED_CATEGORY_COUNTS = {
    "no_tool": 30,
    "single_tool": 60,
    "knowledge": 60,
    "navigation": 30,
    "long_horizon": 120,
    "recovery": 60,
}

DIFFICULTIES = {
    "no_tool": "easy",
    "build_info": "easy",
    "ui_state": "easy",
    "knowledge_black_screen": "easy",
    "knowledge_hud": "easy",
    "navigate_graphics": "medium",
    "reproduce_black_screen": "hard",
    "negative_control_ios": "hard",
    "workaround_black_screen": "hard",
    "recovery_navigation": "hard",
    "recovery_action": "hard",
    "return_home_then_dungeon": "medium",
}

SEMANTIC_SCENARIOS_PER_FAMILY = 15
SURFACES_PER_SCENARIO = 2

DEFAULT_STATE = {
    "version": "2.3.1",
    "platform": "android",
    "current_page": "home",
    "graphics_preset": "high",
    "battle_hud_visible": False,
    "black_screen": False,
}

BUILD_VARIANTS = [
    ("2.3.1", "android"), ("2.3.1", "ios"),
    ("2.3.0", "android"), ("2.3.0", "ios"),
    ("2.2.8", "android"), ("2.2.8", "ios"),
    ("2.2.7", "android"), ("2.2.7", "ios"),
    ("2.1.9", "android"), ("2.1.9", "ios"),
    ("2.4.0-rc1", "android"), ("2.4.0-rc1", "ios"),
    ("3.0.0-beta", "android"), ("3.0.0-beta", "ios"),
    ("2.0.5", "android"),
]

UI_STATE_VARIANTS = [
    {"current_page": "home", "graphics_preset": "high"},
    {"current_page": "home", "graphics_preset": "standard"},
    {"current_page": "settings", "graphics_preset": "high"},
    {"current_page": "settings", "graphics_preset": "standard"},
    {"current_page": "graphics", "graphics_preset": "high"},
    {"current_page": "graphics", "graphics_preset": "standard"},
    {"current_page": "dungeon_select", "graphics_preset": "high"},
    {"current_page": "dungeon_select", "graphics_preset": "standard"},
    {"current_page": "multiplayer_dungeon", "graphics_preset": "high"},
    {"current_page": "multiplayer_dungeon", "graphics_preset": "standard"},
    {
        "current_page": "battle", "graphics_preset": "high",
        "battle_hud_visible": False, "black_screen": True,
    },
    {
        "current_page": "battle", "graphics_preset": "standard",
        "battle_hud_visible": True, "black_screen": False,
    },
    {
        "platform": "ios", "current_page": "battle", "graphics_preset": "high",
        "battle_hud_visible": True, "black_screen": False,
    },
    {
        "version": "2.3.0", "current_page": "battle", "graphics_preset": "high",
        "battle_hud_visible": True, "black_screen": False,
    },
    {"platform": "ios", "current_page": "home", "graphics_preset": "high"},
]

NO_TOOL_SCENARIOS = [
    ("不要调用工具，用一句话说明清晰复现步骤为何有助于缺陷定位", ["复现", "重现"]),
    ("不要调用工具，解释缺陷报告为何应写明版本和平台", ["版本", "平台"]),
    ("不要调用工具，说明预期结果与实际结果对问题判断的价值", ["预期", "实际"]),
    ("不要调用工具，概括日志为何能成为缺陷分析证据", ["日志", "证据"]),
    ("不要调用工具，说明记录缺陷出现频率为什么重要", ["频率", "概率"]),
    ("不要调用工具，解释最小复现用例如何缩小排查范围", ["最小", "简化"]),
    ("不要调用工具，说明时间戳和操作时序为何有助于定位竞态", ["时间", "时序"]),
    ("不要调用工具，概括截图或录屏在缺陷报告中的作用", ["截图", "录屏"]),
    ("不要调用工具，说明为什么要记录缺陷严重度和用户影响", ["严重", "影响"]),
    ("不要调用工具，解释回归版本范围如何帮助定位改动", ["回归", "版本"]),
    ("不要调用工具，说明设备型号信息对移动端缺陷排查的意义", ["设备", "机型"]),
    ("不要调用工具，解释网络条件为何应写入在线问题的复现前提", ["网络", "连接"]),
    ("不要调用工具，说明测试账号或数据准备为何属于复现条件", ["数据", "账号"]),
    ("不要调用工具，概括临时规避方案在缺陷处置中的价值", ["规避", "临时"]),
    ("不要调用工具，说明明确验收条件如何帮助确认缺陷已修复", ["验收", "验证"]),
]

BLACK_KNOWLEDGE_SCENARIOS = [
    ("Android 2.3.1 多人副本 黑屏 INC-101", "INC-101", "确认黑屏事件编号是否为 INC-101"),
    ("Android 2.3.1 图形上下文恢复失败", "图形上下文恢复失败", "查询黑屏问题的根因"),
    ("Android 2.3.1 黑屏 临时规避方案 标准", "切换为标准", "查询黑屏的临时规避方案"),
    ("2.3.1 android 已知问题 多人副本", "已知问题", "核实版本说明是否将多人副本列为已知问题"),
    ("进入多人副本后出现黑屏", "进入多人副本后出现黑屏", "检索多人副本黑屏的症状记录"),
    ("INC-101 root cause", "INC-101", "用事件编号查询黑屏根因字段"),
    ("INC-101 workaround", "进入副本前将画质预设切换为标准", "用事件编号查询 workaround"),
    ("2.3.1 Android graphics context", "图形上下文恢复失败", "核对 Android 图形上下文相关说明"),
    ("多人副本 加载 图形预设 2.3.0", "引入多人副本加载流程优化", "查询 2.3.0 引入的多人副本加载改动"),
    ("iOS 2.3.1 多人副本 回归", "当前未发现多人副本加载相关已知回归问题", "查询 iOS 2.3.1 的对照说明"),
    ("多人副本 入口 首页 副本选择 开始", "首页 -> 副本选择", "查询多人副本入口路径"),
    ("graphics_settings 首页 设置 图形", "画质设置入口路径", "查询规避操作所需的图形设置路径"),
    ("battle_hud_visible combat_test", "battle_hud_visible=true", "查询战斗场景成功加载的判据"),
    ("部分 Android 设备 图形上下文", "部分 Android 设备", "确认已知问题影响的平台范围"),
    ("多人副本 黑屏 标准画质", "切换为标准", "核对标准画质与黑屏规避的关系"),
]

HUD_KNOWLEDGE_SCENARIOS = [
    ("INC-102 战斗 HUD 缺失", "INC-102", "确认 HUD 缺失事件编号"),
    ("2.2.8 UI 初始化竞态", "UI 初始化竞态", "查询 HUD 缺失的根因"),
    ("战斗 HUD 缺失 重新加载战斗场景", "重新加载战斗场景", "查询 HUD 缺失的 workaround"),
    ("进入战斗后战斗 HUD 缺失", "进入战斗后战斗 HUD 缺失", "检索 HUD 缺失症状"),
    ("2.2.8 HUD", "2.2.8", "核实 HUD 事件影响版本"),
    ("INC-102 root cause UI", "INC-102", "用事件编号检索根因字段"),
    ("INC-102 workaround reload", "INC-102", "用事件编号检索恢复措施"),
    ("battle_hud_visible true combat_test", "battle_hud_visible=true", "查询 HUD 可见时的成功判据"),
    ("战斗场景 成功加载 HUD 可见", "战斗场景已成功加载", "查询战斗加载成功的 UI 证据"),
    ("UI 初始化 竞态 战斗", "UI 初始化竞态", "检索战斗 UI 初始化竞态记录"),
    ("HUD 缺失 platform all", "all", "确认 HUD 事件的平台范围"),
    ("incident_id INC-102", "INC-102", "按 incident_id 查询 HUD 事件"),
    ("重新加载 战斗场景", "重新加载战斗场景", "检索重新加载场景这一临时方案"),
    ("进入副本 battle_hud_visible", "battle_hud_visible=true", "查询副本内 HUD 状态判据"),
    ("战斗 HUD 2.2.8 历史事件", "INC-102", "查询 2.2.8 HUD 历史事件"),
]

REPRO_KNOWLEDGE_FOCI = [
    ("INC-101 Android 2.3.1 多人副本 黑屏", "事件编号"),
    ("Android 2.3.1 图形上下文恢复失败 根因", "根因记录"),
    ("Android 2.3.1 多人副本 黑屏 已知问题", "已知问题证据"),
]

IOS_CONTROL_KNOWLEDGE_FOCI = [
    ("iOS 2.3.1 多人副本 平台说明", "平台说明"),
    ("iOS 2.3.1 多人副本 未发现已知回归", "无回归记录"),
    ("iOS 2.3.1 多人副本 正常加载 对照", "战斗加载对照"),
]

WORKAROUND_KNOWLEDGE_SCENARIOS = [
    ("INC-101 workaround", "按事件编号确认 workaround"),
    ("INC-101 标准画质", "确认事件建议的目标画质"),
    ("INC-101 临时规避", "核对事件的临时处置"),
    ("Android 2.3.1 黑屏 规避方案", "查询黑屏规避方案"),
    ("Android 2.3.1 画质预设 标准", "确认推荐的画质预设"),
    ("多人副本 黑屏 切换为标准", "核对进入副本前的设置变化"),
    ("图形上下文恢复失败 workaround", "从根因检索对应 workaround"),
    ("进入副本前 画质预设 标准", "确认规避操作的执行时机"),
    ("已知问题 临时规避方案 2.3.1", "从版本说明查临时方案"),
    ("部分 Android 设备 黑屏 标准", "核对受影响平台与规避设置"),
    ("incident INC-101 standard graphics", "用英文键词查询标准画质方案"),
    ("黑屏 root cause workaround INC-101", "同时核对根因和 workaround"),
    ("2.3.1 android multiplayer dungeon workaround", "用平台与版本条件检索方案"),
    ("INC-101 graphics preset workaround", "查询事件中的 graphics preset 建议"),
    ("多人副本 黑屏 临时方案 标准画质", "核对多人副本的标准画质临时方案"),
]


def state(**overrides):
    value = deepcopy(DEFAULT_STATE)
    value.update(overrides)
    return value


def tool_call(name, arguments=None):
    return {"name": name, "arguments": deepcopy(arguments or {})}


def result_rule(tool, path, *, equals=None, contains=None):
    rule = {"tool": tool, "path": path}
    if equals is not None:
        rule["equals"] = equals
    elif contains is not None:
        rule["contains"] = contains
    else:
        raise ValueError("result_rule 需要 equals 或 contains。")
    return rule


def success_criteria(
    final_state,
    tool_results=None,
    answer_any=None,
    answer_all=None,
    forbidden_any=None,
    regex_none=None,
    max_tool_calls=None,
):
    criteria = {
        "final_state": deepcopy(final_state),
        "required_tool_results": deepcopy(tool_results or []),
        "final_answer": {
            "required": True,
            "contains_all": list(answer_all or []),
            "contains_any": list(answer_any or []),
            "regex_any": [],
            "forbidden_any": list(forbidden_any or []),
            "regex_none": list(regex_none or []),
        },
    }
    if max_tool_calls is not None:
        criteria["max_tool_calls"] = max_tool_calls
    return criteria


def environment(initial_state, fault_injections=None):
    value = {"initial_state": deepcopy(initial_state)}
    if fault_injections:
        value["fault_injections"] = deepcopy(fault_injections)
    return value


def make_case(
    family,
    scenario_number,
    surface_number,
    query,
    calls,
    initial_state,
    criteria,
    max_steps,
    optional_tools=None,
    fault_injections=None,
    extra_tags=None,
):
    variant_number = (
        (scenario_number - 1) * SURFACES_PER_SCENARIO
        + surface_number
    )
    expected_tools = [call["name"] for call in calls]
    category = FAMILY_CATEGORIES[family]
    difficulty = DIFFICULTIES[family]
    tags = [
        "bugops",
        "tool_use",
        "zh_cn",
        "held_out",
        family,
        category,
        difficulty,
        f"scenario_{scenario_number:02d}",
        f"surface_{surface_number:02d}",
        f"variant_{variant_number:02d}",
    ]
    tags.extend(extra_tags or [])

    return {
        "schema_version": SCHEMA_VERSION,
        "split": SPLIT,
        "id": f"{family}_{variant_number:03d}",
        "scenario_id": f"{family}_scenario_{scenario_number:02d}",
        "surface_variant": f"surface_{surface_number:02d}",
        "scenario_family": family,
        "category": category,
        "difficulty": difficulty,
        "query": query,
        "expected_tools": expected_tools,
        "required_tools": list(expected_tools),
        "required_calls": deepcopy(calls),
        "optional_tools": list(optional_tools or []),
        "environment": environment(initial_state, fault_injections),
        "success_criteria": criteria,
        "requires_recovery": family in {"recovery_navigation", "recovery_action"},
        "tags": tags,
        "max_steps": max_steps,
    }


PAGE_LABELS = {
    "home": "首页",
    "settings": "设置页",
    "graphics": "图形页",
    "dungeon_select": "副本选择页",
    "multiplayer_dungeon": "多人副本页",
    "battle": "战斗页",
}

BUG_ANSWER_ANY = ["复现成功", "成功复现", "已复现", "出现了黑屏"]
BUG_ANSWER_FORBIDDEN = ["未复现", "没有复现", "无法复现", "复现失败"]
BUG_ANSWER_REGEX_NONE = [r"未.{0,4}出现黑屏"]
NORMAL_ANSWER_ANY = ["未复现", "没有复现", "正常加载", "加载正常"]
NORMAL_ANSWER_FORBIDDEN = ["已复现", "成功复现", "复现成功", "发生黑屏", "不正常", "加载失败"]
NORMAL_ANSWER_REGEX_NONE = [r"(?<!未)出现黑屏"]


def render_query(core, surface_number):
    core = core.rstrip("。！？；")
    if surface_number == 1:
        return f"请{core}。"
    return f"作为本轮回归验证，请{core}；完成后简要说明结论。"


def generic_query_core(stems, suffixes, scenario_index):
    return f"{stems[scenario_index % len(stems)]}，{suffixes[scenario_index // len(stems)]}"


def updated_state(initial, **overrides):
    value = deepcopy(initial)
    value.update(overrides)
    return value


def battle_is_black(initial):
    return (
        initial["version"] == "2.3.1"
        and initial["platform"] == "android"
        and initial["graphics_preset"] == "high"
    )


def calls_to_multiplayer(start_page, force_home=False):
    calls = []
    page = start_page
    if force_home and page != "home":
        calls.append(tool_call("execute_action", {"action": "back_home"}))
        page = "home"
    if page == "home":
        calls.extend([
            tool_call("navigate_ui", {"target_page": "dungeon_select"}),
            tool_call("navigate_ui", {"target_page": "multiplayer_dungeon"}),
        ])
    elif page == "dungeon_select":
        calls.append(tool_call("navigate_ui", {"target_page": "multiplayer_dungeon"}))
    elif page == "multiplayer_dungeon":
        pass
    else:
        calls.extend([
            tool_call("execute_action", {"action": "back_home"}),
            tool_call("navigate_ui", {"target_page": "dungeon_select"}),
            tool_call("navigate_ui", {"target_page": "multiplayer_dungeon"}),
        ])
    return calls


def build_family_case(family, scenario_number, surface_number):
    scenario_index = scenario_number - 1
    initial = state()
    calls = []
    optional = []
    faults = None
    extra_tags = []

    if family == "no_tool":
        query_core, answer_any = NO_TOOL_SCENARIOS[scenario_index]
        criteria = success_criteria(
            initial,
            answer_any=answer_any,
            max_tool_calls=0,
        )
        max_steps = 2

    elif family == "build_info":
        version, platform = BUILD_VARIANTS[scenario_index]
        initial = state(version=version, platform=platform)
        calls = [tool_call("get_build_info")]
        optional = ["inspect_ui_state"]
        platform_words = ["iOS", "ios", "苹果"] if platform == "ios" else ["Android", "android", "安卓"]
        query_core = generic_query_core(
            [
                "读取当前测试环境的构建信息",
                "确认这次运行使用的应用构建",
                "获取当前被测实例的 build 信息",
                "核实眼下测试环境",
                "检查当前移动端构建",
            ],
            ["报告版本号和平台", "分别给出版本与操作系统", "不要根据问题描述猜测版本或平台"],
            scenario_index,
        )
        criteria = success_criteria(
            initial,
            [
                result_rule("get_build_info", "version", equals=version),
                result_rule("get_build_info", "platform", equals=platform),
            ],
            answer_all=[version],
            answer_any=platform_words,
        )
        max_steps = 4

    elif family == "ui_state":
        initial = state(**UI_STATE_VARIANTS[scenario_index])
        calls = [tool_call("inspect_ui_state")]
        optional = ["get_build_info"]
        page_words = {
            "home": ["home", "首页"],
            "settings": ["settings", "设置"],
            "graphics": ["graphics", "图形", "画质"],
            "dungeon_select": ["dungeon_select", "副本选择"],
            "multiplayer_dungeon": ["multiplayer_dungeon", "多人副本"],
            "battle": ["battle", "战斗"],
        }[initial["current_page"]]
        query_core = generic_query_core(
            [
                "检查当前应用所处页面",
                "读取此刻的 UI 状态",
                "查看当前页面且不要改变界面",
                "确认应用当前显示状态",
                "检查可用于诊断的界面字段",
            ],
            ["报告页面与画质预设", "同时说明 HUD 和黑屏标志", "返回全部关键状态"],
            scenario_index,
        )
        criteria = success_criteria(
            initial,
            [result_rule("inspect_ui_state", "current_page", equals=initial["current_page"])],
            answer_any=page_words,
        )
        max_steps = 4

    elif family == "knowledge_black_screen":
        search_query, evidence, focus = BLACK_KNOWLEDGE_SCENARIOS[scenario_index]
        calls = [tool_call("search_knowledge", {"query": search_query})]
        optional = ["get_build_info"]
        query_core = f"查询知识库，{focus}"
        criteria = success_criteria(
            initial,
            [result_rule("search_knowledge", "$", contains=evidence)],
            answer_any=["黑屏", "多人副本", "INC-101", "图形", "iOS", "HUD", "标准"],
        )
        max_steps = 4
        extra_tags = ["known_issue"]

    elif family == "knowledge_hud":
        search_query, evidence, focus = HUD_KNOWLEDGE_SCENARIOS[scenario_index]
        calls = [tool_call("search_knowledge", {"query": search_query})]
        optional = ["get_build_info"]
        query_core = f"查询知识库，{focus}"
        criteria = success_criteria(
            initial,
            [result_rule("search_knowledge", "$", contains=evidence)],
            answer_any=["HUD", "hud", "INC-102", "初始化竞态"],
        )
        max_steps = 4
        extra_tags = ["incident_history"]

    elif family == "navigate_graphics":
        navigation_starts = [
            state(current_page="home", graphics_preset="high"),
            state(current_page="settings", graphics_preset="high"),
            state(current_page="graphics", graphics_preset="high"),
            state(current_page="home", graphics_preset="standard"),
            state(current_page="settings", graphics_preset="standard"),
            state(current_page="graphics", graphics_preset="standard"),
            state(platform="ios", current_page="home", graphics_preset="high"),
            state(platform="ios", current_page="settings", graphics_preset="high"),
            state(platform="ios", current_page="graphics", graphics_preset="high"),
            state(version="2.3.0", current_page="home", graphics_preset="high"),
            state(version="2.3.0", current_page="settings", graphics_preset="high"),
            state(version="2.3.0", current_page="graphics", graphics_preset="high"),
            state(current_page="dungeon_select", graphics_preset="high"),
            state(platform="ios", current_page="dungeon_select", graphics_preset="standard"),
            state(current_page="multiplayer_dungeon", graphics_preset="standard"),
        ]
        initial = navigation_starts[scenario_index]
        start_page = initial["current_page"]
        calls = [
            tool_call("inspect_ui_state"),
        ]
        if start_page == "home":
            calls.extend([
                tool_call("navigate_ui", {"target_page": "settings"}),
                tool_call("navigate_ui", {"target_page": "graphics"}),
            ])
            route_text = "按 home→settings→graphics 的合法路径进入图形页"
        elif start_page == "settings":
            calls.append(tool_call("navigate_ui", {"target_page": "graphics"}))
            route_text = "从 settings 直接进入 graphics"
        elif start_page == "graphics":
            route_text = "识别已经位于 graphics，不做多余跳转"
        else:
            calls.extend([
                tool_call("execute_action", {"action": "back_home"}),
                tool_call("navigate_ui", {"target_page": "settings"}),
                tool_call("navigate_ui", {"target_page": "graphics"}),
            ])
            route_text = f"从{PAGE_LABELS[start_page]}先返回首页，再进入 graphics"
        calls.append(tool_call("verify_state", {"expected": {"current_page": "graphics"}}))
        optional = ["search_knowledge"]
        final = updated_state(initial, current_page="graphics")
        query_core = (
            f"在 {initial['version']} {initial['platform']}、{initial['graphics_preset']} 画质下，"
            f"{route_text}并验证最终页面"
        )
        criteria = success_criteria(
            final,
            [
                result_rule("inspect_ui_state", "current_page", equals=start_page),
                result_rule("verify_state", "success", equals=True),
            ],
            answer_any=["graphics", "图形", "画质"],
            forbidden_any=["未进入图形", "导航失败", "无法到达"],
        )
        max_steps = len(calls) + 4

    elif family == "reproduce_black_screen":
        start_pages = ["home", "dungeon_select", "multiplayer_dungeon", "settings", "graphics"]
        start_page = start_pages[scenario_index % len(start_pages)]
        initial = state(current_page=start_page, graphics_preset="high")
        search_query, evidence_focus = REPRO_KNOWLEDGE_FOCI[scenario_index // 5]
        calls = [
            tool_call("get_build_info"),
            tool_call("search_knowledge", {"query": search_query}),
            tool_call("inspect_ui_state"),
        ]
        calls.extend(calls_to_multiplayer(start_page))
        calls.extend([
            tool_call("execute_action", {"action": "start_multiplayer_dungeon"}),
            tool_call("query_logs"),
            tool_call("verify_state", {"expected": {
                "current_page": "battle", "black_screen": True, "battle_hud_visible": False,
            }}),
        ])
        final = updated_state(initial, current_page="battle", black_screen=True, battle_hud_visible=False)
        query_core = (
            f"从{PAGE_LABELS[start_page]}开始，在 Android 2.3.1 高画质下先核对黑屏的"
            f"{evidence_focus}，再进入多人副本复现并用日志和状态验证"
        )
        criteria = success_criteria(
            final,
            [
                result_rule("get_build_info", "platform", equals="android"),
                result_rule("search_knowledge", "$", contains="INC-101"),
                result_rule("query_logs", "$", contains="graphics context restore failure"),
                result_rule("verify_state", "success", equals=True),
            ],
            answer_any=BUG_ANSWER_ANY,
            forbidden_any=BUG_ANSWER_FORBIDDEN,
            regex_none=BUG_ANSWER_REGEX_NONE,
        )
        max_steps = len(calls) + 5
        extra_tags = ["black_screen", "stateful"]

    elif family == "negative_control_ios":
        start_pages = ["home", "dungeon_select", "multiplayer_dungeon", "settings", "graphics"]
        start_page = start_pages[scenario_index % len(start_pages)]
        initial = state(platform="ios", current_page=start_page, graphics_preset="high")
        search_query, evidence_focus = IOS_CONTROL_KNOWLEDGE_FOCI[scenario_index // 5]
        calls = [
            tool_call("get_build_info"),
            tool_call("search_knowledge", {"query": search_query}),
            tool_call("inspect_ui_state"),
        ]
        calls.extend(calls_to_multiplayer(start_page))
        calls.extend([
            tool_call("execute_action", {"action": "start_multiplayer_dungeon"}),
            tool_call("query_logs"),
            tool_call("verify_state", {"expected": {
                "current_page": "battle", "black_screen": False, "battle_hud_visible": True,
            }}),
        ])
        final = updated_state(initial, current_page="battle", black_screen=False, battle_hud_visible=True)
        query_core = (
            f"从{PAGE_LABELS[start_page]}开始，在 iOS 2.3.1 高画质下先核对{evidence_focus}，"
            "再运行多人副本负向对照并验证是否正常加载"
        )
        criteria = success_criteria(
            final,
            [
                result_rule("get_build_info", "platform", equals="ios"),
                result_rule("search_knowledge", "$", contains="当前未发现多人副本加载相关已知回归问题"),
                result_rule("query_logs", "$", contains="battle scene loaded successfully"),
                result_rule("verify_state", "success", equals=True),
            ],
            answer_any=NORMAL_ANSWER_ANY,
            forbidden_any=NORMAL_ANSWER_FORBIDDEN,
            regex_none=NORMAL_ANSWER_REGEX_NONE,
        )
        max_steps = len(calls) + 5
        extra_tags = ["negative_control", "ios"]

    elif family == "workaround_black_screen":
        starts = [
            ("home", "high"), ("settings", "high"), ("graphics", "high"),
            ("home", "standard"), ("settings", "standard"), ("graphics", "standard"),
            ("dungeon_select", "standard"), ("multiplayer_dungeon", "standard"),
            ("battle", "standard"), ("home", "high"), ("settings", "high"),
            ("graphics", "high"), ("dungeon_select", "standard"),
            ("multiplayer_dungeon", "standard"), ("battle", "standard"),
        ]
        start_page, preset = starts[scenario_index]
        initial = state(current_page=start_page, graphics_preset=preset)
        if start_page == "battle":
            initial.update(
                black_screen=preset == "high",
                battle_hud_visible=preset == "standard",
            )
        search_query, focus = WORKAROUND_KNOWLEDGE_SCENARIOS[scenario_index]
        calls = [
            tool_call("search_knowledge", {"query": search_query}),
            tool_call("inspect_ui_state"),
        ]
        if preset == "high":
            if start_page == "home":
                calls.extend([
                    tool_call("navigate_ui", {"target_page": "settings"}),
                    tool_call("navigate_ui", {"target_page": "graphics"}),
                ])
            elif start_page == "settings":
                calls.append(tool_call("navigate_ui", {"target_page": "graphics"}))
            elif start_page != "graphics":
                raise AssertionError("高画质规避语义场景必须从 home/settings/graphics 开始")
            calls.extend([
                tool_call("execute_action", {"action": "set_graphics", "value": "standard"}),
                tool_call("execute_action", {"action": "back_home"}),
            ])
            calls.extend(calls_to_multiplayer("home"))
            setup_text = "先应用标准画质规避方案"
        else:
            calls.extend(calls_to_multiplayer(start_page))
            setup_text = "识别标准画质规避方案已经生效，不重复改设置"
        calls.extend([
            tool_call("execute_action", {"action": "start_multiplayer_dungeon"}),
            tool_call("query_logs"),
            tool_call("verify_state", {"expected": {
                "current_page": "battle", "graphics_preset": "standard",
                "black_screen": False, "battle_hud_visible": True,
            }}),
        ])
        optional = ["get_build_info"]
        final = updated_state(
            initial, current_page="battle", graphics_preset="standard",
            black_screen=False, battle_hud_visible=True,
        )
        query_core = (
            f"当前在{PAGE_LABELS[start_page]}且为 {preset} 画质；先从知识库{focus}，"
            f"{setup_text}，再进入多人副本验证规避效果"
        )
        criteria = success_criteria(
            final,
            [
                result_rule("search_knowledge", "$", contains="切换为标准"),
                result_rule("execute_action", "graphics_preset", equals="standard"),
                result_rule("query_logs", "$", contains="battle scene loaded successfully"),
                result_rule("verify_state", "success", equals=True),
            ],
            answer_any=["规避成功", "正常加载", "加载正常", "HUD 正常"],
            forbidden_any=["规避失败", "仍然黑屏", "依然黑屏", "不正常", "加载失败"],
            regex_none=[r"(?:仍|依然|仍然).{0,3}黑屏"],
        )
        max_steps = len(calls) + 5
        extra_tags = ["workaround", "black_screen"]

    elif family == "recovery_navigation":
        recovery_navigation_configs = [
            (state(current_page="home", graphics_preset="high"), "settings"),
            (state(current_page="home", graphics_preset="standard"), "settings"),
            (state(platform="ios", current_page="home", graphics_preset="high"), "settings"),
            (state(version="2.3.0", current_page="home", graphics_preset="high"), "settings"),
            (state(platform="ios", current_page="home", graphics_preset="standard"), "settings"),
            (state(current_page="home", graphics_preset="high"), "graphics"),
            (state(current_page="home", graphics_preset="standard"), "graphics"),
            (state(platform="ios", current_page="home", graphics_preset="high"), "graphics"),
            (state(version="2.3.0", current_page="home", graphics_preset="high"), "graphics"),
            (state(platform="ios", current_page="home", graphics_preset="standard"), "graphics"),
            (state(current_page="settings", graphics_preset="high"), "graphics"),
            (state(current_page="settings", graphics_preset="standard"), "graphics"),
            (state(platform="ios", current_page="settings", graphics_preset="high"), "graphics"),
            (state(version="2.3.0", current_page="settings", graphics_preset="high"), "graphics"),
            (state(platform="ios", current_page="settings", graphics_preset="standard"), "graphics"),
        ]
        initial, fault_target = recovery_navigation_configs[scenario_index]
        faults = [{
            "operation": "navigate",
            "target": fault_target,
            "times": 1,
            "error": "transient_failure",
        }]
        calls = [tool_call("inspect_ui_state")]
        if initial["current_page"] == "home" and fault_target == "graphics":
            calls.append(tool_call("navigate_ui", {"target_page": "settings"}))
        calls.extend([
            tool_call("navigate_ui", {"target_page": fault_target}),
            tool_call("inspect_ui_state"),
            tool_call("navigate_ui", {"target_page": fault_target}),
        ])
        if fault_target == "settings":
            calls.append(tool_call("navigate_ui", {"target_page": "graphics"}))
        calls.append(tool_call("verify_state", {"expected": {"current_page": "graphics"}}))
        optional = ["query_logs", "search_knowledge"]
        final = updated_state(initial, current_page="graphics")
        target_label = "settings" if fault_target == "settings" else "graphics"
        query_core = (
            f"在 {initial['version']} {initial['platform']}、{initial['graphics_preset']} 画质、"
            f"当前 {initial['current_page']} 状态下进入图形页；首次进入 {target_label} 会临时失败，"
            "请读取反馈、检查状态、重试并验证"
        )
        criteria = success_criteria(
            final,
            [
                result_rule(
                    "inspect_ui_state", "current_page",
                    equals=initial["current_page"],
                ),
                result_rule("verify_state", "success", equals=True),
            ],
            answer_any=["恢复", "重试", "成功", "graphics", "图形"],
            forbidden_any=["恢复失败", "最终未进入", "无法进入"],
        )
        max_steps = len(calls) + 4
        extra_tags = ["fault_injection", "navigation_recovery"]

    elif family == "recovery_action":
        action_configs = [
            ("set", state(current_page="graphics", graphics_preset="high"), "standard"),
            ("set", state(platform="ios", current_page="graphics", graphics_preset="high"), "standard"),
            ("set", state(version="2.3.0", current_page="graphics", graphics_preset="high"), "standard"),
            ("set", state(current_page="graphics", graphics_preset="standard"), "high"),
            ("set", state(platform="ios", current_page="graphics", graphics_preset="standard"), "high"),
            ("set", state(version="2.3.0", current_page="graphics", graphics_preset="standard"), "high"),
            ("start", state(current_page="multiplayer_dungeon", graphics_preset="high"), None),
            ("start", state(current_page="multiplayer_dungeon", graphics_preset="standard"), None),
            ("start", state(platform="ios", current_page="multiplayer_dungeon", graphics_preset="high"), None),
            ("back", state(current_page="settings", graphics_preset="high"), None),
            ("back", state(current_page="graphics", graphics_preset="standard"), None),
            ("back", state(platform="ios", current_page="dungeon_select", graphics_preset="high"), None),
            ("start_home", state(current_page="home", graphics_preset="high"), None),
            ("start_home", state(current_page="home", graphics_preset="standard"), None),
            ("start_home", state(platform="ios", current_page="home", graphics_preset="high"), None),
        ]
        action_kind, initial, goal = action_configs[scenario_index]
        action_name = {
            "set": "set_graphics",
            "start": "start_multiplayer_dungeon",
            "start_home": "start_multiplayer_dungeon",
            "back": "back_home",
        }[action_kind]
        faults = [{"operation": "action", "target": action_name, "times": 1, "error": "transient_failure"}]
        calls = [tool_call("inspect_ui_state")]
        if action_kind == "start_home":
            calls.extend(calls_to_multiplayer("home"))
        action_arguments = {"action": action_name}
        if action_kind == "set":
            action_arguments["value"] = goal
        calls.extend([
            tool_call("execute_action", action_arguments),
            tool_call("inspect_ui_state"),
            tool_call("execute_action", action_arguments),
        ])
        if action_kind in {"start", "start_home"}:
            black = battle_is_black(initial)
            final = updated_state(
                initial, current_page="battle", black_screen=black,
                battle_hud_visible=not black,
            )
            calls.extend([
                tool_call("query_logs"),
                tool_call("verify_state", {"expected": {
                    "current_page": "battle", "black_screen": black,
                    "battle_hud_visible": not black,
                }}),
            ])
            query_core = (
                f"在 {initial['platform']} {initial['version']}、{initial['graphics_preset']} 画质下"
                f"从{PAGE_LABELS[initial['current_page']]}启动多人副本；首次启动会临时失败，"
                "请检查状态后重试，并根据最终画面判断结果"
            )
            if black:
                answer_kwargs = {
                    "answer_any": BUG_ANSWER_ANY,
                    "forbidden_any": BUG_ANSWER_FORBIDDEN,
                    "regex_none": BUG_ANSWER_REGEX_NONE,
                }
                log_text = "graphics context restore failure"
            else:
                answer_kwargs = {
                    "answer_any": NORMAL_ANSWER_ANY,
                    "forbidden_any": NORMAL_ANSWER_FORBIDDEN,
                    "regex_none": NORMAL_ANSWER_REGEX_NONE,
                }
                log_text = "battle scene loaded successfully"
            result_rules = [
                result_rule("query_logs", "$", contains=log_text),
                result_rule("verify_state", "success", equals=True),
            ]
        elif action_kind == "set":
            final = updated_state(initial, graphics_preset=goal)
            calls.append(tool_call("verify_state", {"expected": {
                "current_page": "graphics", "graphics_preset": goal,
            }}))
            query_core = (
                f"在 {initial['platform']} {initial['version']} 图形页把画质从"
                f" {initial['graphics_preset']} 改为 {goal}；首次设置会临时失败，"
                "请检查状态后重试并验证"
            )
            answer_kwargs = {
                "answer_any": [goal, "设置成功", "修改成功"],
                "forbidden_any": ["设置失败", "修改失败", "没有生效"],
            }
            result_rules = [
                result_rule("execute_action", "graphics_preset", equals=goal),
                result_rule("verify_state", "success", equals=True),
            ]
        else:
            final = updated_state(initial, current_page="home")
            calls.append(tool_call("verify_state", {"expected": {"current_page": "home"}}))
            query_core = (
                f"从{PAGE_LABELS[initial['current_page']]}执行返回首页；第一次 back_home 会临时失败，"
                "请检查状态后重试并验证最终页面"
            )
            answer_kwargs = {
                "answer_any": ["home", "首页", "恢复成功", "返回成功"],
                "forbidden_any": ["返回失败", "无法返回", "仍未回到首页"],
            }
            result_rules = [result_rule("verify_state", "success", equals=True)]
        optional = ["query_logs"]
        criteria = success_criteria(
            final,
            result_rules,
            **answer_kwargs,
        )
        max_steps = len(calls) + 4
        extra_tags = ["fault_injection", "action_recovery"]

    elif family == "return_home_then_dungeon":
        start_pages = ["settings", "graphics", "dungeon_select", "multiplayer_dungeon", "battle"]
        modes = [
            {"platform": "android", "graphics_preset": "high"},
            {"platform": "android", "graphics_preset": "standard"},
            {"platform": "ios", "graphics_preset": "high"},
        ]
        start_page = start_pages[scenario_index % len(start_pages)]
        mode = modes[scenario_index // len(start_pages)]
        initial = state(current_page=start_page, **mode)
        if start_page == "battle":
            initial.update(
                black_screen=battle_is_black(initial),
                battle_hud_visible=not battle_is_black(initial),
            )
        black = battle_is_black(initial)
        calls = [
            tool_call("inspect_ui_state"),
            tool_call("get_build_info"),
            tool_call("execute_action", {"action": "back_home"}),
            tool_call("navigate_ui", {"target_page": "dungeon_select"}),
            tool_call("navigate_ui", {"target_page": "multiplayer_dungeon"}),
            tool_call("execute_action", {"action": "start_multiplayer_dungeon"}),
            tool_call("query_logs"),
            tool_call(
                "verify_state",
                {
                    "expected": {
                        "current_page": "battle",
                        "graphics_preset": initial["graphics_preset"],
                        "black_screen": black,
                        "battle_hud_visible": not black,
                    },
                },
            ),
        ]
        optional = ["search_knowledge"]
        final = updated_state(
            initial, current_page="battle", black_screen=black,
            battle_hud_visible=not black,
        )
        query_core = (
            f"当前在{PAGE_LABELS[start_page]}，环境为 {initial['platform']} {initial['version']}、"
            f"{initial['graphics_preset']} 画质；先返回首页，再进入多人副本，"
            "查询日志并判断最终是正常加载还是出现黑屏"
        )
        if black:
            answer_kwargs = {
                "answer_any": BUG_ANSWER_ANY,
                "forbidden_any": BUG_ANSWER_FORBIDDEN,
                "regex_none": BUG_ANSWER_REGEX_NONE,
            }
            expected_log = "graphics context restore failure"
        else:
            answer_kwargs = {
                "answer_any": NORMAL_ANSWER_ANY,
                "forbidden_any": NORMAL_ANSWER_FORBIDDEN + ["规避失败"],
                "regex_none": NORMAL_ANSWER_REGEX_NONE,
            }
            expected_log = "battle scene loaded successfully"
        criteria = success_criteria(
            final,
            [
                result_rule("query_logs", "$", contains="back_home"),
                result_rule("query_logs", "$", contains=expected_log),
                result_rule("verify_state", "success", equals=True),
            ],
            **answer_kwargs,
        )
        max_steps = len(calls) + 4
        extra_tags = ["non_home_start", "stateful"]

    else:
        raise ValueError(f"未知场景族: {family}")

    return make_case(
        family=family,
        scenario_number=scenario_number,
        surface_number=surface_number,
        query=render_query(query_core, surface_number),
        calls=calls,
        initial_state=initial,
        criteria=criteria,
        max_steps=max_steps,
        optional_tools=optional,
        fault_injections=faults,
        extra_tags=extra_tags,
    )


def build_cases():
    cases = []
    if SEMANTIC_SCENARIOS_PER_FAMILY * SURFACES_PER_SCENARIO != VARIANTS_PER_FAMILY:
        raise AssertionError("语义场景数与表述数必须生成每族 30 条。")
    for scenario_number in range(1, SEMANTIC_SCENARIOS_PER_FAMILY + 1):
        for surface_number in range(1, SURFACES_PER_SCENARIO + 1):
            for family in FAMILY_ORDER:
                cases.append(build_family_case(family, scenario_number, surface_number))
    return cases


def validate_call_arguments(case_id, call):
    name = call.get("name")
    arguments = call.get("arguments")
    if name not in ALLOWED_TOOLS:
        raise ValueError(f"{case_id}: 未知工具 {name}")
    if not isinstance(arguments, dict):
        raise ValueError(f"{case_id}: {name}.arguments 必须是 object")

    allowed_keys = {
        "search_knowledge": {"query"},
        "get_build_info": set(),
        "inspect_ui_state": set(),
        "navigate_ui": {"target_page"},
        "execute_action": {"action", "value"},
        "query_logs": set(),
        "verify_state": {"expected"},
    }[name]
    if set(arguments) - allowed_keys:
        raise ValueError(f"{case_id}: {name} 含无效参数 {sorted(set(arguments) - allowed_keys)}")

    if name == "search_knowledge" and not isinstance(arguments.get("query"), str):
        raise ValueError(f"{case_id}: search_knowledge.query 缺失")
    if name == "navigate_ui" and arguments.get("target_page") not in {
        "settings", "graphics", "dungeon_select", "multiplayer_dungeon"
    }:
        raise ValueError(f"{case_id}: navigate_ui.target_page 无效")
    if name == "execute_action":
        action = arguments.get("action")
        if action not in {"set_graphics", "back_home", "start_multiplayer_dungeon"}:
            raise ValueError(f"{case_id}: execute_action.action 无效")
        if action == "set_graphics" and arguments.get("value") not in {"high", "standard"}:
            raise ValueError(f"{case_id}: set_graphics.value 无效")
    if name == "verify_state" and not isinstance(arguments.get("expected"), dict):
        raise ValueError(f"{case_id}: verify_state.expected 缺失")


def case_signature(case):
    payload = {
        "environment": case["environment"],
        "required_calls": case["required_calls"],
        "success_criteria": case["success_criteria"],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def final_answer_matches(answer, spec):
    if answer is None:
        return not spec.get("required", False)
    text = str(answer)
    folded = text.casefold()
    if spec.get("required") and not text.strip():
        return False
    if any(term.casefold() not in folded for term in spec.get("contains_all", [])):
        return False
    contains_any = spec.get("contains_any", [])
    if contains_any and not any(term.casefold() in folded for term in contains_any):
        return False
    regex_any = spec.get("regex_any", [])
    if regex_any and not any(re.search(pattern, text, re.IGNORECASE) for pattern in regex_any):
        return False
    if any(term.casefold() in folded for term in spec.get("forbidden_any", [])):
        return False
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in spec.get("regex_none", [])):
        return False
    return True


def validate_cases(cases):
    if len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError(f"期望 {EXPECTED_CASE_COUNT} 条，实际 {len(cases)} 条。")

    ids = [case.get("id") for case in cases]
    queries = [case.get("query") for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Benchmark ID 不唯一。")
    if len(set(queries)) != len(queries):
        raise ValueError("Benchmark query 不唯一。")

    required_fields = {
        "schema_version", "split", "id", "scenario_family", "category",
        "scenario_id", "surface_variant", "difficulty", "query",
        "expected_tools", "required_tools",
        "required_calls", "optional_tools", "environment", "success_criteria",
        "requires_recovery", "tags", "max_steps",
    }
    family_counts = Counter()
    category_counts = Counter()
    scenario_counts = Counter()
    scenario_signatures = defaultdict(set)
    family_signatures = defaultdict(set)
    signature_counts = Counter()

    for case in cases:
        case_id = case.get("id", "<missing-id>")
        missing = required_fields - set(case)
        if missing:
            raise ValueError(f"{case_id}: 缺少字段 {sorted(missing)}")
        if case["schema_version"] != SCHEMA_VERSION or case["split"] != SPLIT:
            raise ValueError(f"{case_id}: schema_version/split 不正确")
        family = case["scenario_family"]
        if family not in FAMILY_CATEGORIES:
            raise ValueError(f"{case_id}: 未知 scenario_family")
        if case["category"] != FAMILY_CATEGORIES[family]:
            raise ValueError(f"{case_id}: category 与场景族不匹配")
        if case["difficulty"] not in {"easy", "medium", "hard"}:
            raise ValueError(f"{case_id}: difficulty 无效")
        if not isinstance(case["query"], str) or not case["query"].strip():
            raise ValueError(f"{case_id}: query 为空")
        expected_scenario_prefix = f"{family}_scenario_"
        if not case["scenario_id"].startswith(expected_scenario_prefix):
            raise ValueError(f"{case_id}: scenario_id 与场景族不匹配")
        if case["surface_variant"] not in {"surface_01", "surface_02"}:
            raise ValueError(f"{case_id}: surface_variant 无效")

        calls = case["required_calls"]
        if not isinstance(calls, list):
            raise ValueError(f"{case_id}: required_calls 必须是 array")
        for call in calls:
            if not isinstance(call, dict):
                raise ValueError(f"{case_id}: required_calls 元素必须是 object")
            validate_call_arguments(case_id, call)
        call_names = [call["name"] for call in calls]
        if case["expected_tools"] != call_names or case["required_tools"] != call_names:
            raise ValueError(f"{case_id}: expected/required_tools 必须从 required_calls 派生")
        if any(tool not in ALLOWED_TOOLS for tool in case["optional_tools"]):
            raise ValueError(f"{case_id}: optional_tools 含未知工具")

        env = case["environment"]
        if not isinstance(env, dict) or not isinstance(env.get("initial_state"), dict):
            raise ValueError(f"{case_id}: environment.initial_state 缺失")
        if set(env["initial_state"]) != set(DEFAULT_STATE):
            raise ValueError(f"{case_id}: initial_state 必须包含完整状态")
        faults = env.get("fault_injections", [])
        if not isinstance(faults, list):
            raise ValueError(f"{case_id}: fault_injections 必须是 array")
        for fault in faults:
            if set(fault) != {"operation", "target", "times", "error"}:
                raise ValueError(f"{case_id}: fault_injection 字段不完整")
            if fault["operation"] not in {"navigate", "action"}:
                raise ValueError(f"{case_id}: fault operation 无效")
            if fault["times"] != 1 or fault["error"] != "transient_failure":
                raise ValueError(f"{case_id}: v2 恢复故障必须是一次 transient_failure")

        should_recover = family in {"recovery_navigation", "recovery_action"}
        if case["requires_recovery"] is not should_recover:
            raise ValueError(f"{case_id}: requires_recovery 不正确")
        if should_recover != bool(faults):
            raise ValueError(f"{case_id}: 恢复用例与故障注入不一致")

        criteria = case["success_criteria"]
        required_criteria = {
            "final_state",
            "required_tool_results",
            "final_answer",
        }
        optional_criteria = {"max_tool_calls"}
        if (
            not required_criteria.issubset(criteria)
            or set(criteria) - required_criteria - optional_criteria
        ):
            raise ValueError(f"{case_id}: success_criteria 字段不正确")
        if "max_tool_calls" in criteria:
            maximum = criteria["max_tool_calls"]
            if (
                not isinstance(maximum, int)
                or isinstance(maximum, bool)
                or maximum < 0
            ):
                raise ValueError(f"{case_id}: max_tool_calls 必须是非负整数")
        if family == "no_tool" and criteria.get("max_tool_calls") != 0:
            raise ValueError(f"{case_id}: no_tool 必须设置 max_tool_calls=0")
        if not isinstance(criteria["final_state"], dict):
            raise ValueError(f"{case_id}: final_state 必须是 object")
        for rule in criteria["required_tool_results"]:
            if rule.get("tool") not in ALLOWED_TOOLS or not isinstance(rule.get("path"), str):
                raise ValueError(f"{case_id}: required_tool_results 无效")
            operators = {"equals", "contains"} & set(rule)
            if len(operators) != 1:
                raise ValueError(f"{case_id}: 工具结果规则必须有且仅有一个断言")
        answer = criteria["final_answer"]
        required_answer_fields = {"required", "contains_all", "contains_any", "regex_any"}
        optional_answer_fields = {"forbidden_any", "regex_none"}
        if (
            not required_answer_fields.issubset(answer)
            or set(answer) - required_answer_fields - optional_answer_fields
        ):
            raise ValueError(f"{case_id}: final_answer 字段不正确")
        if not isinstance(answer["required"], bool):
            raise ValueError(f"{case_id}: final_answer.required 必须是 bool")
        for key in ("contains_all", "contains_any", "regex_any", "forbidden_any", "regex_none"):
            if key not in answer:
                continue
            if not isinstance(answer[key], list) or not all(isinstance(x, str) for x in answer[key]):
                raise ValueError(f"{case_id}: final_answer.{key} 必须是 string array")
        for pattern in answer.get("regex_any", []) + answer.get("regex_none", []):
            re.compile(pattern)
        if family in {
            "reproduce_black_screen", "negative_control_ios",
            "workaround_black_screen", "recovery_action",
            "return_home_then_dungeon",
        } and not (answer.get("forbidden_any") or answer.get("regex_none")):
            raise ValueError(f"{case_id}: 结论型用例缺少反向答案约束")
        if not isinstance(case["tags"], list) or SPLIT not in case["tags"]:
            raise ValueError(f"{case_id}: tags 缺少 held_out")
        if not isinstance(case["max_steps"], int) or case["max_steps"] <= len(calls):
            raise ValueError(f"{case_id}: max_steps 必须大于参考调用数")

        family_counts[family] += 1
        category_counts[case["category"]] += 1
        signature = case_signature(case)
        scenario_counts[case["scenario_id"]] += 1
        scenario_signatures[case["scenario_id"]].add(signature)
        family_signatures[family].add(signature)
        signature_counts[signature] += 1

    expected_family_counts = {family: VARIANTS_PER_FAMILY for family in FAMILY_ORDER}
    if dict(family_counts) != expected_family_counts:
        raise ValueError(f"场景族计数错误: {dict(family_counts)}")
    if dict(category_counts) != EXPECTED_CATEGORY_COUNTS:
        raise ValueError(f"类别计数错误: {dict(category_counts)}")
    expected_scenario_count = len(FAMILY_ORDER) * SEMANTIC_SCENARIOS_PER_FAMILY
    if len(scenario_counts) != expected_scenario_count:
        raise ValueError(f"期望 {expected_scenario_count} 个 scenario_id，实际 {len(scenario_counts)}")
    if any(count != SURFACES_PER_SCENARIO for count in scenario_counts.values()):
        raise ValueError("每个 scenario_id 必须恰好有两种 surface form")
    if any(len(signatures) != 1 for signatures in scenario_signatures.values()):
        raise ValueError("同一 scenario_id 的两种表述必须共享同一执行签名")
    if len(signature_counts) < expected_scenario_count:
        raise ValueError(f"执行签名不足: {len(signature_counts)} < {expected_scenario_count}")
    if max(signature_counts.values()) > SURFACES_PER_SCENARIO:
        raise ValueError("任一执行签名最多只能对应两行")
    for family in FAMILY_ORDER:
        if len(family_signatures[family]) < SEMANTIC_SCENARIOS_PER_FAMILY:
            raise ValueError(f"{family}: 执行签名少于 15 个")

    block_size = len(FAMILY_ORDER)
    for offset in range(0, len(cases), block_size):
        if [case["scenario_family"] for case in cases[offset:offset + block_size]] != FAMILY_ORDER:
            raise ValueError("输出必须按 scenario/surface 外层、family 内层交错")


def resolve_path(payload, path):
    if path == "$":
        return payload
    current = payload
    for part in path.split("."):
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    return current


def assertion_matches(payload, rule):
    try:
        value = resolve_path(payload, rule["path"])
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    if "equals" in rule:
        return value == rule["equals"]
    haystack = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(rule["contains"]) in haystack


def fault_key_for_call(call):
    if call["name"] == "navigate_ui":
        return "navigate", call["arguments"]["target_page"]
    if call["name"] == "execute_action":
        return "action", call["arguments"]["action"]
    return None


def validate_reference_oracles(cases):
    """Replay every canonical trajectory against the real synthetic tools."""
    try:
        for case in cases:
            case_id = case["id"]
            reset_environment(case["environment"])
            remaining_faults = Counter(
                (fault["operation"], fault["target"])
                for fault in case["environment"].get("fault_injections", [])
                for _ in range(fault["times"])
            )
            successful_payloads = defaultdict(list)

            for call in case["required_calls"]:
                fault_key = fault_key_for_call(call)
                expected_failure = bool(fault_key and remaining_faults[fault_key] > 0)
                outcome = execute_tool(call["name"], deepcopy(call["arguments"]))

                if expected_failure:
                    if outcome.get("ok") or outcome.get("error_type") != "RuntimeError":
                        raise ValueError(f"{case_id}: 注入故障调用没有按预期失败: {call}")
                    remaining_faults[fault_key] -= 1
                    continue
                if not outcome.get("ok"):
                    raise ValueError(f"{case_id}: 参考调用失败: {call} -> {outcome}")
                successful_payloads[call["name"]].append(outcome.get("result"))

            if any(remaining_faults.values()):
                raise ValueError(f"{case_id}: 参考轨迹未消费全部故障注入")

            final_state = inspect_ui_state()
            for key, expected in case["success_criteria"]["final_state"].items():
                if final_state.get(key) != expected:
                    raise ValueError(
                        f"{case_id}: 最终状态 {key}={final_state.get(key)!r}，期望 {expected!r}"
                    )

            for rule in case["success_criteria"]["required_tool_results"]:
                payloads = successful_payloads[rule["tool"]]
                if not any(assertion_matches(payload, rule) for payload in payloads):
                    raise ValueError(f"{case_id}: 工具结果断言不可满足: {rule}")
    finally:
        reset_environment()


def serialize_cases(cases):
    return "".join(
        json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n"
        for case in cases
    )


def write_benchmark(path=DEFAULT_OUTPUT, validate_oracles=True):
    cases = build_cases()
    validate_cases(cases)
    if validate_oracles:
        validate_reference_oracles(cases)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialize_cases(cases), encoding="utf-8")
    return cases


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="校验现有文件与确定性生成结果一致，不写文件。",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cases = build_cases()
    validate_cases(cases)
    validate_reference_oracles(cases)
    content = serialize_cases(cases)

    if args.check:
        if not args.output.exists():
            raise SystemExit(f"Benchmark 文件不存在: {args.output}")
        if args.output.read_text(encoding="utf-8") != content:
            raise SystemExit(f"Benchmark 与生成器不一致: {args.output}")
        print(f"校验通过: {len(cases)} 条确定性 held-out 用例。")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"已生成 {len(cases)} 条用例: {args.output}")


if __name__ == "__main__":
    main()
