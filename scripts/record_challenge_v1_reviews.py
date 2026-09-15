"""Record stage019 author-AI review of EXACT sources; not a generic judge.

Full final texts were reviewed grouped by identical text, then grounded per row.
Unknown source bytes require new review. Never imports these decisions into training.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_baseline import load_records
from scripts.compare_pilot_v1 import replay
from scripts.audit_answer_evidence import audit
from evaluation.scoring import record_sha256, load_reviews, evaluate_case, summarize_results
from evaluation.observation_audit import audit_observation

SOURCES = {
    "base": "162b0d8bf02d62add0358fc64c52ab513d1ebc93f4cf547e180caaf1bbb59a4a",
    "sft": "c56627d0bdf178ab30cfd2292edfd249ab586df494d6dbcb9c56cbbabb75fa31",
}


def decision(mode, row):
    """Historical decisions, with mechanical checks against reviewed evidence."""
    a = audit(row)
    answer = row["result"]["final_answer"]
    failures = a["observed_failure_count"]
    observed = a["last_observed_state"]
    if mode == "base":
        if row["case"]["category"] in ("expression", "structure"):
            return "fail", f"最终状态有证据，但实际工具失败{failures}次；答案漏报或否认失败，未忠实报告过程。"
        if observed is None:
            return "fail", ("仅调用query_logs得到空列表，没有取得HUD/黑屏状态。未完成明确要求的读取及报告；"
                            "对未知状态保留不确定性本身不是错误，但未利用可用状态工具完成任务。"
                            "若声称未黑屏也没有实际观测支持。")
        return "pass", "状态字段、失败0次与实际工具返回一致；遵守只读/禁止设置约束，未扩大验证范围。"
    # These phrases occur in the already-reviewed SFT outputs, not rules for new text.
    if failures and "本轮未发生工具失败" in answer:
        return "fail", f"实际工具失败{failures}次，答案同时报告非零或零失败并声称本轮未发生工具失败，过程陈述矛盾。"
    if "本轮未执行工具" in answer:
        return "fail", "实际调用了query_logs及inspect_ui_state，答案却声称本轮未执行工具，与事实及前文矛盾。"
    if not observed or observed["state"] != row["result"]["final_environment_state"]:
        raise ValueError("Unexpected missing state evidence")
    state = observed["state"]
    for label, field in (("最终页面", "current_page"), ("画质", "graphics_preset"),
                         ("HUD", "battle_hud_visible"), ("黑屏", "black_screen")):
        value = str(state[field]).lower() if isinstance(state[field], bool) else state[field]
        if f"{label}={value}" not in answer:
            raise ValueError("Unexpected state wording/claim")
    if f"实际工具失败{failures}次" not in answer:
        raise ValueError("Unexpected failure count")
    return "pass", (f"最终状态四字段与模型取得的状态返回一致；实际失败{failures}次与完整轨迹一致；"
                    "日志查询和完成声明有执行证据，未发现额外无依据结论。")


def main():
    for mode, digest in SOURCES.items():
        path = ROOT / f"results/baseline/challenge_v1_{mode}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Changed source: new review required")
    folder = ROOT / "results/reviews/challenge_v1"
    folder.mkdir(parents=True, exist_ok=True)
    summaries, evidence = {}, []
    for mode, digest in SOURCES.items():
        rows = load_records(ROOT / f"results/baseline/challenge_v1_{mode}.jsonl", True)
        reviews = []
        for row in rows:
            if not row["metrics"]["execution_success"]:
                continue
            replay(row)
            verdict, reason = decision(mode, row)
            item = dict(id=row["case"]["id"], protocol_version="2.3", source_sha256=digest,
                        record_sha256=record_sha256(row["case"], row["result"]), verdict=verdict,
                        reason=reason, reviewer="Codex protocol author", reviewer_type="protocol_author_ai",
                        reviewed_at="2026-09-15")
            reviews.append(item)
            evidence.append(dict(model=mode, **item, query=row["case"]["query"],
                                 result=row["result"], objective_evidence=audit(row),
                                 observation_audit=audit_observation(row["result"], {
                                     k: row["case"]["success_criteria"]["final_state"][k]
                                     for k in ("current_page", "graphics_preset", "battle_hud_visible", "black_screen")})))
        path = folder / f"{mode}_author_reviews.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reviews))
        loaded = load_reviews(path, rows, digest)
        for row in rows:
            row["metrics"] = evaluate_case(row["case"], row["result"], loaded.get(row["case"]["id"]))
        summaries[mode] = summarize_results(rows)
        from collections import Counter
        print(mode, Counter(r["verdict"] for r in reviews),
              "task", summaries[mode]["overall"]["task_success_rate"])
        print({k: v["task_success_rate"] for k,v in summaries[mode]["by_category"].items()})
    summaries["review_method"] = (
        "100 eligible answers; full identical texts read in groups and evidence checked per record. "
        "Protocol-author AI, not blind/independent/human review. 140 execution failures not semantically reviewed. "
        "Frozen execution rule retained despite eight missing-observation false positives.")
    (folder / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n")
    (folder / "evidence.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in evidence))


if __name__ == "__main__":
    main()
