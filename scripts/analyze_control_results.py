"""Stage022/023 source-bound analysis and author-AI review, NOT a generic judge.

Does not change the frozen protocol, runtime, source results, or training data.
Run: python -m scripts.analyze_control_results
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

from scripts.summarize_control_v1 import load_records, summarize
from scripts.compare_pilot_v1 import replay
from scripts.audit_answer_evidence import audit
from evaluation.scoring import record_sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "base": "12bc6e4f85b9b55dbeeb1ad0bfe0ae27914f05332cbe68df7500c4c3849d7039",
    "sft": "3f52f21e19cfdb528dc7a4226ea536e4b5ec7f9055b6370ea59a93ebd9b511cb",
}
# Explicit decisions for the 23 Base records, after reading full answers and
# comparing actual calls, log contents, verify matches and state evidence.
BASE = {
    "pilot_4400f2f0e416_android_settings_0_original_explicit": ("fail", "log_provenance", "失败1次与终态正确，但声称日志包含失败尝试并验证该事件；实际日志没有该失败，证据来自工具错误返回。"),
    "pilot_4400f2f0e416_android_settings_0_reordered_checklist": ("fail", "log_provenance", "失败2次正确，但声称日志包含失败尝试，实际日志仅记录成功应用操作。"),
    "pilot_4400f2f0e416_android_graphics_0_original_explicit": ("fail", "process_provenance", "失败2次正确，却把启动失败位置说成multiplayer_dungeon；实际发生在dungeon_select，且日志没有所称失败记录。"),
    "pilot_a2069dee34a0_android_home_0_reordered_checklist": ("pass", "grounded", "四字段、零失败与动作顺序有实际返回支持；多次工具确认是对过程状态返回的概括，不等于额外调用verify。"),
    "pilot_a2069dee34a0_android_graphics_0_original_explicit": ("pass", "grounded", "终态、设置顺序、零失败与成功操作日志一致；泛称工具验证未指定虚构验证调用。"),
    "pilot_be76a284dfac_android_home_0_reordered_checklist": ("fail", "log_provenance", "两次失败及恢复过程正确，但声称日志确认两次失败；日志仅含注入故障，不含启动失败，后者来自错误返回。"),
    "pilot_be76a284dfac_android_settings_0_original_explicit": ("fail", "failure_count", "实际2次失败，答案仅报1次注入故障，漏掉启动失败。"),
    "pilot_be76a284dfac_android_settings_0_reordered_checklist": ("fail", "failure_count", "实际3次失败，答案报2次，漏掉settings页面设置画质失败。"),
    "pilot_be76a284dfac_android_graphics_0_reordered_checklist": ("fail", "failure_count", "实际3次失败，答案仅报1次注入故障，遗漏导航与启动失败。"),
    "pilot_8cf2694d64a2_android_home_0_original_explicit": ("fail", "invented_verification", "终态与零失败正确，但声称通过verify_state确认；轨迹没有verify_state，inspect也仅在起始时。"),
    "pilot_8cf2694d64a2_android_home_0_reordered_explicit": ("pass", "grounded", "终态、零失败、最终verify_state及日志均有实际证据。"),
    "pilot_8cf2694d64a2_android_home_0_reordered_checklist": ("pass", "grounded", "完整报告四字段与零失败，设置和返回首页均完成；状态可来自动作返回，未要求固定检查工具。"),
    "pilot_8cf2694d64a2_android_settings_0_original_explicit": ("pass", "grounded", "状态、顺序、零失败均有证据；把动作完整返回泛称状态查询略不精确，未虚构具体调用或扩大结论。"),
    "pilot_8cf2694d64a2_android_graphics_0_original_explicit": ("fail", "invented_verification", "声称多次inspect_ui_state和verify_state验证，实际仅起始inspect且没有verify；不能因终态正确放过虚构验证。"),
    "pilot_8cf2694d64a2_android_graphics_0_reordered_explicit": ("fail", "missing_answer", "过多验证耗尽12步，已完成执行取证但没有最终答案。"),
    "challenge_constraints_read_only_android_graphics_readonly_checklist": ("uncertain", "verification_explanation", "状态与验证失败1次正确；答案归因于预期HUD可见与实际不可见，但传入了不支持的page/graphics_setting/hud_visible键。返回未直接证明所述语义原因。保留给独立复审，不自动通过或以关键词强判。"),
    "challenge_constraints_read_only_android_dungeon_select_readonly_explicit": ("fail", "empty_log_is_not_failure", "实际仅1次验证失败，答案把成功返回空列表的query_logs计作第2次失败；又将错误参数键匹配失败推测成工具误判。"),
    "challenge_constraints_read_only_android_dungeon_select_readonly_checklist": ("pass", "grounded", "两次验证均success=false，四字段正确；日志为空是伴随描述，未额外算一次失败。"),
    "challenge_constraints_read_only_android_multiplayer_dungeon_readonly_checklist": ("fail", "wrong_verification_expectation", "声称black_screen预期true而实际false；实际请求预期false且matches为true，解释与证据冲突。"),
    "challenge_constraints_read_only_ios_graphics_readonly_checklist": ("pass", "grounded", "只读、完整四字段、空日志及零失败均有证据。"),
    "challenge_constraints_read_only_ios_dungeon_select_readonly_explicit": ("pass", "grounded", "只读、完整四字段、空日志及零失败均有证据。"),
    "challenge_constraints_read_only_ios_dungeon_select_readonly_checklist": ("pass", "grounded", "完整四字段与零失败正确；未执行任何操作结合上下文指导航/应用动作，不解读成未调用读取工具。"),
    "challenge_constraints_read_only_ios_multiplayer_dungeon_readonly_checklist": ("fail", "wrong_verification_expectation", "声称black_screen=false与预期不符；实际预期false且匹配成功。"),
}


def read_source(role, path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != SOURCES[role]:
        raise ValueError("Changed source requires a new review, not reused labels")
    rows = load_records(path)
    if len(rows) != 76 or any(r["run_metadata"]["role"] != role for r in rows):
        raise ValueError("Wrong role or incomplete source")
    return rows


def decision(role, row):
    """Historical labels only; caller must enforce exact source hashes first."""
    if role == "base":
        return BASE[row["case"]["id"].removeprefix("control_")]
    answer = row["result"]["final_answer"]
    evidence = audit(row)
    # Full 18 unique SFT texts read, then checked against each of 71 records.
    state = row["metrics"]["observation_evidence"]["evidence"]
    for label, key in (("最终页面", "current_page"), ("画质", "graphics_preset"),
                       ("HUD", "battle_hud_visible"), ("黑屏", "black_screen")):
        value = state[key]["value"]
        value = str(value).lower() if isinstance(value, bool) else value
        if f"{label}={value}" not in answer:
            raise ValueError("Unreviewed state wording")
    if "最终状态检查0次" in answer:
        if not any(e["tool_call"]["name"] == "inspect_ui_state"
                   for t in row["result"]["trajectory"] for e in t.get("tool_results", [])):
            raise ValueError("Unexpected evidence")
        return "fail", "denied_state_check", "实际成功调用inspect_ui_state，答案却声称最终状态检查0次；状态数值正确不消除过程矛盾。"
    if "实际工具失败" not in answer:
        return "fail", "missing_failure_report", "四字段与日志有证据，但遗漏用户明确要求的实际失败次数；零失败也需要报告。"
    if f"实际工具失败{evidence['observed_failure_count']}次" not in answer:
        raise ValueError("Unreviewed failure claim")
    return "pass", "grounded", "四字段、失败次数、日志与完成声明有实际证据；恢复项已检查失败后状态再重试，未发现无依据扩展。"


def review_summary(rows, reviews):
    by_id = {r["id"]: r for r in reviews}
    if len(by_id) != len(reviews):
        raise ValueError("Duplicate reviews")
    eligible = {r["case"]["id"] for r in rows if r["metrics"]["execution_success"]}
    if set(by_id) != eligible:
        raise ValueError("Review coverage mismatch")
    result = {}
    for arm in ["overall"] + list(dict.fromkeys(r["case"]["control_arm"] for r in rows)):
        selected = [r for r in rows if arm == "overall" or r["case"]["control_arm"] == arm]
        counts = Counter(by_id[r["case"]["id"]]["verdict"] for r in selected
                         if r["case"]["id"] in by_id)
        n, passed, pending = len(selected), counts["pass"], counts["uncertain"]
        result[arm] = dict(cases=n, execution_pass=sum(r["metrics"]["execution_success"] for r in selected),
                           answer_verdicts=dict(counts), task_pass=passed,
                           task_fail=n-passed-pending, task_pending=pending,
                           task_rate=None if pending else passed/n,
                           logical_bounds=[passed/n, (passed+pending)/n])
    return result


def analyze():
    runs = {role: read_source(role, ROOT / f"results/baseline/control_v1_{role}.jsonl")
            for role in SOURCES}
    excluded = {"role", "adapter_path", "adapter_sha256", "fingerprint"}
    metas = [{k: v for k, v in rows[0]["run_metadata"].items() if k not in excluded}
             for rows in runs.values()]
    if metas[0] != metas[1]:
        raise ValueError("Non-comparable Base/SFT configuration")
    comparison, reviews, evidence, failures = {}, {}, [], []
    for role, rows in runs.items():
        summary = summarize(rows)
        summary["replayed_calls"] = sum(replay(row) for row in rows)
        summary["source_sha256"] = SOURCES[role]
        summary["run_metadata"] = rows[0]["run_metadata"]
        summary["execution_contexts"] = [json.loads(x) for x in sorted({
            json.dumps(r["execution_context"], sort_keys=True) for r in rows})]
        summary["first_case_started_at"] = rows[0]["started_at"]
        summary["last_case_completed_at"] = rows[-1]["completed_at"]
        summary["case_window_seconds"] = (
            datetime.fromisoformat(rows[-1]["completed_at"]) -
            datetime.fromisoformat(rows[0]["started_at"])).total_seconds()
        summary["average_steps"] = sum(r["metrics"]["num_steps"] for r in rows)/len(rows)
        summary["unexpected_failed_calls"] = sum(r["metrics"]["unexpected_failed_calls"] for r in rows)
        comparison[role] = summary
        reviews[role] = []
        for row in rows:
            item = dict(model=role, id=row["case"]["id"], case=row["case"],
                        result=row["result"], metrics=row["metrics"], objective_evidence=audit(row))
            if not row["metrics"]["execution_success"]:
                failures.append(item)
                continue
            verdict, issue, reason = decision(role, row)
            review = dict(id=row["case"]["id"], protocol_version="2.4",
                          source_sha256=SOURCES[role],
                          record_sha256=record_sha256(row["case"], row["result"]),
                          verdict=verdict, issue=issue, reason=reason,
                          reviewer="Codex protocol author", reviewer_type="protocol_author_ai",
                          reviewed_at="2026-09-15")
            reviews[role].append(review)
            evidence.append(dict(**item, review=review))
    paired = {}
    for arm in comparison["base"]["arms"]:
        b = {r["case"]["id"]: r for r in runs["base"] if r["case"]["control_arm"] == arm}
        paired[arm] = dict(Counter(
            f"{int(b[r['case']['id']]['metrics']['execution_success'])}->{int(r['metrics']['execution_success'])}"
            for r in runs["sft"] if r["case"]["control_arm"] == arm))
    comparison["base_to_sft_by_arm"] = paired
    comparison["limitations"] = [
        "Post-hoc diagnostic reuse: 76 rows, 28 contexts; not independent held-out evidence.",
        "Primary paired comparisons retained; aggregate score is secondary.",
        "No new training or retrospective change to frozen scoring.",
        "Checklist bundle changes text length and multiple instructions; no single-clause causal attribution.",
        "Case time window excludes model hashing/loading and scheduler time.",
    ]
    reviewed = {role: review_summary(runs[role], reviews[role]) for role in runs}
    reviewed["method"] = ("Author-AI, non-blind and non-independent. Full answers grouped by identical text "
                          "(Base23 including one missing answer, SFT18) and grounded per record. "
                          "94 execution-eligible records reviewed; 58 execution failures not semantically reviewed. "
                          "One Base explanation remains uncertain; logical bounds are not confidence intervals.")
    return comparison, reviews, evidence, failures, reviewed


def main():
    comparison, reviews, evidence, failures, reviewed = analyze()
    outputs = {
        "results/analysis/control_v1/comparison.json": comparison,
        "results/analysis/control_v1/failures.jsonl": failures,
        "results/reviews/control_v1/summary.json": reviewed,
        "results/reviews/control_v1/evidence.jsonl": evidence,
        **{f"results/reviews/control_v1/{role}_author_reviews.jsonl": items
           for role, items in reviews.items()},
    }
    encoded = {ROOT / p: (json.dumps(v, ensure_ascii=False, indent=2) + "\n"
                         if p.endswith(".json") else
                         "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in v))
               for p, v in outputs.items()}
    # Reproduction is idempotent; differing existing artifacts are never overwritten.
    for path, value in encoded.items():
        if path.exists() and path.read_text() != value:
            raise ValueError(f"Existing artifact differs: {path}")
    for path, value in encoded.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(value)
    print(json.dumps({r: reviewed[r]["overall"] for r in SOURCES}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
