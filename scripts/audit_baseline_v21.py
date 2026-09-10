"""Create a versioned benchmark and retrospective audit without GPU inference."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.protocol_v21 import evaluate_execution, revise_case
from scripts.summarize_baseline import load_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/baseline/qwen3_4b_baseline_v2.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/audit/v21")
    args = parser.parse_args()
    records = load_records(args.input)
    originals = [json.loads(line) for line in (ROOT / "data/eval/bugops_eval_v2.jsonl").read_text().splitlines()]
    revised = {case["id"]: revise_case(case) for case in originals}
    audit = []
    counts = {}
    for record in records:
        original = record["case"]
        case = revised[original["id"]]
        if revise_case(original) != case:
            raise ValueError(f"Case differs from frozen benchmark: {case['id']}")
        eligible = not case["requires_new_inference"]
        metrics = evaluate_execution(case, record["result"]) if eligible else None
        audit.append({
            "id": case["id"], "scenario_id": case["scenario_id"],
            "category": case["category"], "protocol_version": "2.1",
            "retrospective_only": True, "eligible_for_rescore": eligible,
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
            group["answer_review_pending"] += int(metrics["execution_success"])
    summary = {
        "protocol_version": "2.1", "status": "retrospective_diagnostic_not_model_improvement",
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "by_category": counts,
        "note": "Execution success excludes prose correctness. No revised task success rate until independent answer review. Changed prompts excluded; fresh inference required.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def write_jsonl(name, rows):
        (args.output_dir / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    write_jsonl("benchmark_v21.jsonl", revised.values())
    write_jsonl("audit.jsonl", audit)
    write_jsonl("answer_review_queue.jsonl", [row for row in audit if row["metrics"] and row["metrics"]["execution_success"]])
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
