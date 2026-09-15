"""Freeze a small post-hoc control study, not an independent new held-out set."""
from copy import deepcopy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.protocol_v24 import evaluate_case, validate_case
from scripts.run_baseline import runtime_sha256
from training.pilot_data import oracle

PUBLIC_CONTRACT = (
    "\n本轮取证约定：必须实际取得最终页面、画质、HUD和黑屏的工具返回；"
    "可使用状态检查、验证或最后一次成功应用操作返回的完整状态，不要求固定工具名。"
    "状态证据不能早于最后一次成功状态变更。"
    "所有导航或应用动作尝试结束后（包含失败尝试），再查询本轮日志，不能只使用开始时的日志。"
    "若全程只读，则读取当前状态和本轮日志即可。实际失败次数包括工具错误及验证返回失败；"
    "最终回答必须完整、一致地报告这些事实。"
)
CHECKLIST = (
    "\n执行前后参考检查清单（不改变任务目标，也不要求输出思维过程）："
    "先确认当前页面与禁令，再安排合法的页面路径；"
    "按用户要求逐次执行设置，重复档位也不能省略；"
    "遇到失败后读取状态再决定下一步；"
    "全部操作结束后取日志及有效终态证据；"
    "统计整个轨迹的失败，不把没有预设故障当作零失败；"
    "最终核对报告中的数字与文字是否自相矛盾。"
)
EXTENSION_FILES = (
    "training/pilot_data.py",
    "evaluation/observation_audit.py", "evaluation/protocol_v24.py",
    "scripts/prepare_control_v1.py", "scripts/verify_control_v1.py",
    "scripts/run_control_v1.py", "scripts/summarize_control_v1.py",
    "scripts/superpod_eval_control_v1.sh",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extension_hash():
    payload = {name: digest(ROOT / name) for name in EXTENSION_FILES}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def build():
    sources = [json.loads(line) for line in (ROOT / "data/challenge_v1/cases.jsonl").read_text().splitlines()]
    originals = {c["pair_id"]: c for c in sources if c.get("expression_variant") == "control"}
    cases = []
    for source in sources:
        if source.get("expression_variant") == "rewrite":
            context_id = source["pair_id"]
            variations = [
                ("original_explicit", originals[context_id]["query"]),
                ("reordered_explicit", source["query"]),
                ("reordered_checklist", source["query"])]
            family = "expression_control"
        elif source["group_id"] == "constraints_read_only" and source["environment"]["initial_state"]["current_page"] != "home":
            context_id = source["id"]
            variations = [("readonly_explicit", source["query"]), ("readonly_checklist", source["query"])]
            family = "observation_control"
        else:
            continue
        for arm, query in variations:
            case = deepcopy(source)
            case.pop("pair_id", None)
            case.pop("expression_variant", None)
            case.update(id=f"control_{context_id}_{arm}", context_id=context_id,
                        control_arm=arm, category=family, scenario_family=family,
                        protocol_version="2.4", split="post_hoc_control_not_heldout",
                        source_case_id=source["id"], query=query + PUBLIC_CONTRACT,
                        authoring_provenance="explicit_reuse_of_challenge_contexts_after_error_analysis")
            if arm.endswith("checklist"):
                case["query"] += CHECKLIST
            case["observation_contract"] = {
                "final_state_fields": ["current_page", "graphics_preset", "battle_hud_visible", "black_screen"],
                "logs_after_last_mutation": True}
            validate_case(case)
            cases.append(case)
    return cases


def oracle_result(case):
    # Reuse deterministic executor, not old review verdicts or model answers.
    legacy = deepcopy(case)
    legacy["protocol_version"] = "2.3"
    result = oracle(legacy)["result"]
    if not evaluate_case(case, result)["execution_success"]:
        raise ValueError(f"2.4 oracle failed: {case['id']}")
    return result


def main():
    cases = build()
    for case in cases:
        oracle_result(case)
    extension, files = extension_hash()
    folder = ROOT / "data/control_v1"
    if folder.exists():
        raise ValueError("Already frozen; do not overwrite")
    folder.mkdir(parents=True)
    path = folder / "cases.jsonl"
    path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
    previous = json.loads((ROOT / "data/challenge_v1/manifest.json").read_text())
    manifest = dict(
        protocol_version="2.4", study="control_v1", status="frozen_before_first_control_inference",
        scope="post-hoc controls; reused contexts, no independent held-out claim",
        cases=len(cases), contexts=len({c["context_id"] for c in cases}),
        arms=dict(Counter(c["control_arm"] for c in cases)),
        cases_sha256=digest(path), source_cases_sha256=digest(ROOT / "data/challenge_v1/cases.jsonl"),
        legacy_runtime_sha256=runtime_sha256(), extension_sha256=extension, extension_files=files,
        model_sha256=previous["model_sha256"], adapter_sha256=previous["adapter_sha256"],
        generation=dict(do_sample=False, max_new_tokens=512, max_steps_override=None),
        versions=dict(torch="2.9.1", transformers="5.16.1", peft="0.20.0", accelerate="1.14.0"),
        oracle_execution_pass=len(cases))
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
