"""Paired replay audit of the frozen first-seed study; no score modifications."""
from collections import Counter, defaultdict
from datetime import datetime
import json
import math

from scripts.transaction_full_sft import RESULTS, PLAN, CONFIG, SPLITS, protocol, inference_metadata
from scripts.transaction_gpu_smoke import rows, validate_records, write_once
from scripts.prepare_transaction_v1 import DATA
from scripts.prepare_control_v1 import ROOT, digest
from training.transaction_data import canonical


def aggregate(records):
    count = len(records)
    result = dict(cases=count)
    for key in ("task_success", "execution_success", "answer_correct"):
        successes = sum(r["metrics"][key] for r in records)
        result[key] = dict(count=successes, denominator=count, rate=successes / count)
    result.update(policy_violating_cases=sum(r["metrics"]["policy_violations"] > 0 for r in records),
        policy_violating_calls=sum(r["metrics"]["policy_violations"] for r in records),
        mutations=sum(r["metrics"]["mutations"] for r in records),
        mean_tool_calls=sum(r["metrics"]["tool_calls"] for r in records) / count,
        termination=dict(Counter(r["result"]["terminated_reason"] for r in records)),
        generation_seconds=sum(u["seconds"] for r in records for u in r["generation_usage"]))
    return result


def answer_errors(record):
    try:
        answer = json.loads(record["result"]["final_answer"])
    except (ValueError, TypeError):
        return ["missing_or_non_json_report"]
    expected = record["metrics"]["expected_report"]
    if not isinstance(answer, dict):
        return ["non_object_report"]
    wrong = [k for k in expected if k not in answer or canonical(answer[k]) != canonical(expected[k])]
    if set(answer) != set(expected):
        wrong.append("report_keys")
    if not wrong and not record["metrics"]["answer_correct"]:
        wrong.append("strict_json_failure")
    return wrong


def audit():
    plan = protocol()
    run = json.loads((RESULTS / "training_run.json").read_text())
    if (run["status"] != "complete" or run["plan_sha256"] != digest(PLAN)
            or canonical(run["config"]) != canonical(CONFIG)
            or run["trainer_state"]["global_step"] != CONFIG["expected_steps"]
            or run["trainer_state"]["epoch"] != 1.0):
        raise ValueError("Training protocol/status/steps drift")
    for metrics, key in ((run["training_metrics"], "train_loss"), (run["evaluation_metrics"], "eval_loss")):
        if not math.isfinite(metrics[key]):
            raise ValueError("Nonfinite loss")
    summary = dict(scope="frozen_first_seed_full_agent_comparison", plan_sha256=digest(PLAN),
        sources={p.name: digest(p) for p in sorted(RESULTS.glob("*.json*"))}, splits={},
        training=dict(steps=run["trainer_state"]["global_step"], metrics=run["training_metrics"],
            evaluation=run["evaluation_metrics"], attempts=run["attempts"], gpu=run["gpu"],
            peak_allocated_gib=run["peak_cuda_allocated_bytes"] / 2**30,
            peak_reserved_gib=run["peak_cuda_reserved_bytes"] / 2**30),
        limitations=["single seed, synthetic narrow-domain tasks, correlated configuration variants",
            "confirmation now inspected; do not use these trajectories or rewrites as training preferences",
            "execution success alone excludes policy and final-answer correctness",
            "uploaded adapter digest cross-linked, no local weights or GPU rerun",
            "timestamps include disconnections/queue gaps, not continuous GPU usage"])
    failures = []
    for split in SPLITS:
        cases = list(rows(DATA / f"{split}_cases.jsonl"))
        paired = {}
        for role in ("base", "sft"):
            path = RESULTS / f"{role}_{split}.jsonl"
            if role == "base" and digest(path) != run["baseline_sha256"][split]:
                raise ValueError("Baseline differs from training input")
            metadata = inference_metadata(plan, role, split, run["adapter_sha256"] if role == "sft" else None)
            records = validate_records(path, metadata, cases)
            if len(records) != len(cases):
                raise ValueError("Incomplete evaluation coverage")
            for line, r in enumerate(records, 1):
                start, end = (datetime.fromisoformat(r[k]) for k in ("started_at", "completed_at"))
                if end < start:
                    raise ValueError("Negative episode duration")
                boundary = datetime.fromisoformat(run["attempts"][0 if role == "base" else -1][
                    "started_at" if role == "base" else "completed_at"])
                if role == "base" and end > boundary or role == "sft" and start < boundary:
                    raise ValueError("Training/evaluation chronology drift")
                if r["metrics"]["tool_calls"] > plan["generation"]["max_calls"]:
                    raise ValueError("Call budget exceeded")
                if not r["metrics"]["task_success"]:
                    failures.append(dict(role=role, split=split, case_id=r["case"]["id"],
                        source=str(path.relative_to(ROOT)), line=line, metrics=r["metrics"],
                        answer_errors=answer_errors(r), final_answer=r["result"]["final_answer"],
                        terminated_reason=r["result"]["terminated_reason"]))
            paired[role] = records
        result = {role: aggregate(records) for role, records in paired.items()}
        result["paired_task"] = dict(improved=0, regressed=0, both_pass=0, both_fail=0)
        for a, b in zip(paired["base"], paired["sft"]):
            x, y = a["metrics"]["task_success"], b["metrics"]["task_success"]
            key = "both_pass" if x and y else "both_fail" if not x and not y else "improved" if y else "regressed"
            result["paired_task"][key] += 1
        result["by_group"] = {}
        for group in sorted({c["group_id"] for c in cases}):
            result["by_group"][group] = {role: aggregate([r for r in records if r["case"]["group_id"] == group])
                                        for role, records in paired.items()}
        result["by_category"] = {category: {role: aggregate([r for r in records if r["case"]["category"] == category])
            for role, records in paired.items()} for category in sorted({c["category"] for c in cases})}
        summary["splits"][split] = result
    summary["failure_records"] = len(failures)
    summary["sft_answer_errors"] = dict(Counter(k for r in failures if r["role"] == "sft" for k in r["answer_errors"]))
    return summary, failures


def main():
    summary, failures = audit()
    folder = ROOT / "results/analysis/transaction_v1_full_sft"
    write_once(folder / "summary.json", summary)
    write_once(folder / "failure_index.json", failures)
    print(json.dumps({"splits": {s: {k: v for k, v in r.items() if k not in ("by_group", "by_category")}
        for s, r in summary["splits"].items()}, "failure_records": len(failures),
        "sft_answer_errors": summary["sft_answer_errors"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
