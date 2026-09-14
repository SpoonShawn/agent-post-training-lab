"""Read-only-source paired comparison with provenance checks and tool replay."""
import hashlib
import json
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_baseline import load_records
from scripts.run_baseline import runtime_sha256
from scripts.verify_training_bundle import verify
from evaluation.scoring import summarize_results
from tools.environment_tools import reset_environment, inspect_ui_state
from tools.executor import execute_tool_json


def replay(row):
    reset_environment(row["case"]["environment"])
    count = 0
    for step in row["result"]["trajectory"]:
        for entry in step.get("tool_results", []):
            actual = execute_tool_json(entry["tool_call"])
            if actual != entry["result"]:
                raise ValueError(f"Tool replay differs: {row['case']['id']}")
            count += 1
    if inspect_ui_state() != row["result"]["final_environment_state"]:
        raise ValueError("Final state replay differs")
    return count


def compact(rows):
    summary = summarize_results(rows)
    metrics = summary["overall"]
    return {
        "cases": len(rows), "execution_pass": sum(r["metrics"]["execution_success"] for r in rows),
        "execution_rate": metrics["task_execution_success_rate"],
        "task_rate": metrics["task_success_rate"], "answer_pending": metrics["task_unresolved_count"],
        "steps": metrics["average_num_steps"], "unexpected_failed_calls": metrics["total_unexpected_failed_calls"],
        "invalid_transitions": metrics["total_invalid_transitions"], "tool_f1": metrics["tool_f1"],
        "by_category": {k: {"cases": v["num_cases"], "execution_rate": v["task_execution_success_rate"]}
                        for k, v in summary["by_category"].items()},
    }


def main():
    verify()
    report, pairs = {}, []
    for split in ("validation", "confirmation"):
        path = ROOT / f"data/pilot_v1/{split}_cases.jsonl"
        expected = {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}
        runs, metadata, sources = {}, {}, {}
        for mode in ("base", "sft"):
            source = ROOT / f"results/baseline/pilot_v1_{mode}_{split}.jsonl"
            rows = load_records(source, recompute=True)
            if len(rows) != len(expected) or {r["case"]["id"] for r in rows} != set(expected):
                raise ValueError("Missing/extra cases")
            sources[mode] = hashlib.sha256(source.read_bytes()).hexdigest()
            meta = rows[0]["run_metadata"]
            for row in rows:
                if row["case"] != expected[row["case"]["id"]] or row["run_metadata"] != meta:
                    raise ValueError("Case or run metadata drift")
                if meta["benchmark_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError("Benchmark digest mismatch")
                if meta["runtime_sha256"] != runtime_sha256():
                    raise ValueError("Runtime digest mismatch")
            replayed = sum(replay(row) for row in rows)
            runs[mode], metadata[mode] = rows, meta
            report[f"{split}_{mode}"] = dict(compact(rows), replayed_calls=replayed)
        excluded = {"adapter_path", "adapter_sha256", "fingerprint"}
        comparable = lambda m: {k: v for k, v in m.items() if k not in excluded}
        if comparable(metadata["base"]) != comparable(metadata["sft"]):
            raise ValueError("Base/SFT are not comparable")
        if metadata["base"].get("adapter_sha256") or not metadata["sft"].get("adapter_sha256"):
            raise ValueError("Adapter identity missing or applied to Base")
        baseline = {r["case"]["id"]: r for r in runs["base"]}
        transitions = Counter()
        groups = {}
        for after in runs["sft"]:
            case_id = after["case"]["id"]
            before = baseline[case_id]
            b, s = before["metrics"]["execution_success"], after["metrics"]["execution_success"]
            transition = f"{int(b)}->{int(s)}"
            transitions[transition] += 1
            group = groups.setdefault(after["case"]["group_id"], {"cases": 0, "base_pass": 0, "sft_pass": 0})
            group["cases"] += 1
            group["base_pass"] += int(b)
            group["sft_pass"] += int(s)
            pairs.append({"split": split, "id": case_id, "transition": transition,
                          "base": {k: before["metrics"].get(k) for k in
                                   ("final_state_match", "required_tool_results_accuracy", "process_violations", "public_contract_violations")},
                          "base_answer": before["result"].get("final_answer"),
                          "sft_answer": after["result"].get("final_answer")})
        report[split] = {"transitions": dict(transitions), "groups": groups, "source_sha256": sources,
                         "model_sha256": metadata["base"]["model_content_sha256"],
                         "adapter_sha256": metadata["sft"]["adapter_sha256"],
                         "runtime_sha256": metadata["base"]["runtime_sha256"],
                         "generation": metadata["base"]["generation"]}
    report["limitations"] = [
        "Execution only; no new answer verdicts assigned",
        "Each split has only 4 composition groups; 80 rows are not independent samples",
        "Inference adapter hash is available, but training run.json/trainer_state/weights have not been submitted",
        "Tool replay validates recorded environment behavior, not independent proof of model generation or lack of leakage",
    ]
    destination = ROOT / "results/analysis/pilot_v1"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (destination / "paired_cases.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pairs))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
