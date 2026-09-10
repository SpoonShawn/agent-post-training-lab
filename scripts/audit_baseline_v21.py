"""Create a versioned benchmark and retrospective audit without GPU inference."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.protocol_v21 import revise_case
from evaluation.scoring import evaluate_case, load_reviews, record_sha256, review_queue, summarize_results
from scripts.summarize_baseline import load_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/baseline/qwen3_4b_baseline_v2.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/audit/v21")
    parser.add_argument("--reviews", type=Path, help="有来源校验的 2.1 答案复核 JSONL")
    args = parser.parse_args()
    records = load_records(args.input)
    source_sha = hashlib.sha256(args.input.read_bytes()).hexdigest()
    originals = [json.loads(line) for line in (ROOT / "data/eval/bugops_eval_v2.jsonl").read_text().splitlines()]
    revised = {case["id"]: revise_case(case) for case in originals}
    reviews = load_reviews(args.reviews, records, source_sha) if args.reviews else {}
    if any(revised[case_id]["requires_new_inference"] for case_id in reviews):
        raise ValueError("Changed prompts cannot use retrospective answer reviews")
    audit = []
    scored = []
    review_sources = []
    counts = {}
    for record in records:
        original = record["case"]
        case = revised[original["id"]]
        if revise_case(original) != case:
            raise ValueError(f"Case differs from frozen benchmark: {case['id']}")
        eligible = not case["requires_new_inference"]
        metrics = evaluate_case(case, record["result"], reviews.get(case["id"])) if eligible else None
        if eligible:
            scored.append({"case": case, "result": record["result"], "metrics": metrics})
            review_sources.append({"case": original, "result": record["result"], "metrics": metrics})
        audit.append({
            "id": case["id"], "scenario_id": case["scenario_id"],
            "category": case["category"], "protocol_version": "2.1",
            "retrospective_only": True, "eligible_for_rescore": eligible,
            "query": case["query"], "source_sha256": source_sha,
            "record_sha256": record_sha256(original, record["result"]),
            "legacy_task_success": record["metrics"]["task_success"],
            "metrics": metrics,
            "answer": record["result"].get("final_answer"),
        })
        group = counts.setdefault(case["category"], Counter())
        group["cases"] += 1
        group["requires_new_inference"] += int(not eligible)
        if eligible:
            group["eligible"] += 1
            group["execution_success"] += int(metrics["execution_success"])
            group["answer_review_pending"] += int(metrics["task_success"] is None)
    summary = {
        "protocol_version": "2.1", "status": "retrospective_diagnostic_not_model_improvement",
        "source_sha256": source_sha,
        "source_case_count": len(records), "benchmark_case_count": len(revised),
        "source_coverage": len(records) / len(revised),
        "eligible_summary": summarize_results(scored),
        "by_category": counts,
        "note": "Retrospective only. Execution excludes prose correctness. Final task rate requires completed answer review with declared provenance, not necessarily independent. Changed prompts excluded; fresh inference required. Eligible denominator differs from the full benchmark.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def write_jsonl(name, rows):
        (args.output_dir / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    write_jsonl("benchmark_v21.jsonl", revised.values())
    write_jsonl("rerun_v21.jsonl", [case for case in revised.values() if case["requires_new_inference"]])
    write_jsonl("audit.jsonl", audit)
    write_jsonl("answer_review_queue.jsonl", review_queue(review_sources, source_sha))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    display = {key: value for key, value in summary.items() if key != "eligible_summary"}
    display["eligible_task_status"] = {
        key: summary["eligible_summary"]["overall"].get(key)
        for key in ("num_cases", "task_execution_success_rate", "task_success_rate",
                    "task_unresolved_count", "task_success_lower_bound", "task_success_upper_bound")
    }
    print(json.dumps(display, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
