"""Versioned scoring and evidence-bound answer reviews; v2 remains unchanged."""

import hashlib
import json
from pathlib import Path

from evaluation.evaluator import aggregate_results, evaluate_case as evaluate_v2
from evaluation.protocol_v21 import evaluate_execution


def protocol_version(case):
    version = case.get("protocol_version", "2.0")
    if version not in {"2.0", "2.1"}:
        raise ValueError(f"Unsupported protocol_version: {version}")
    return version


def dataset_protocol(cases):
    versions = {protocol_version(case) for case in cases}
    if len(versions) > 1:
        raise ValueError("Mixed benchmark protocols are not allowed")
    return next(iter(versions), "2.0")


def record_sha256(case, result):
    payload = json.dumps({"case": case, "result": result}, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_reviews(path, records, source_sha256):
    """Reject stale, duplicate or untraceable reviews before applying any verdict."""
    expected = {row["case"]["id"]: record_sha256(row["case"], row["result"])
                for row in records}
    reviews = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        review = json.loads(line)
        if not isinstance(review, dict):
            raise ValueError("Review must be an object")
        case_id = review.get("id")
        if case_id not in expected or case_id in reviews:
            raise ValueError(f"Unknown or duplicate review ID: {case_id}")
        if (review.get("source_sha256") != source_sha256 or
                review.get("record_sha256") != expected[case_id]):
            raise ValueError(f"Stale review evidence: {case_id}")
        if review.get("protocol_version") != "2.1":
            raise ValueError(f"Wrong review protocol: {case_id}")
        if review.get("verdict") not in {"pass", "fail", "uncertain"}:
            raise ValueError(f"Invalid review verdict: {case_id}")
        for key in ("reason", "reviewer", "reviewer_type", "reviewed_at"):
            if not isinstance(review.get(key), str) or not review[key].strip():
                raise ValueError(f"Missing review {key}: {case_id}")
        reviews[case_id] = review
    return reviews


def review_queue(records, source_sha256):
    """Export evidence without model names or legacy scores (not proof of blinding)."""
    queue = []
    for row in records:
        if row["metrics"]["task_success"] is not None:
            continue
        case, result = row["case"], row["result"]
        queue.append({
            "id": case["id"], "protocol_version": "2.1",
            "source_sha256": source_sha256,
            "record_sha256": record_sha256(case, result),
            "query": case["query"], "answer": result.get("final_answer"),
            "tool_evidence": [{"step": step.get("step"), "tool_results": step.get("tool_results", [])}
                              for step in result.get("trajectory", [])],
            "final_environment_state": result.get("final_environment_state"),
            "verdict": "pending", "reason": "", "reviewer": "",
            "reviewer_type": "", "reviewed_at": "",
        })
    return queue


def evaluate_case(case, result, review=None):
    metrics = evaluate_v2(case, result)
    if protocol_version(case) == "2.0":
        if review is not None:
            raise ValueError("Answer reviews require protocol 2.1")
        return metrics
    if result.get("query") != case.get("query"):
        raise ValueError(f"Result query differs from benchmark; fresh inference required: {case['id']}")
    metrics["legacy_task_success"] = metrics["task_success"]
    execution = evaluate_execution(case, result)
    metrics.update(execution)
    metrics["recovery_execution_success"] = execution["recovery_success"]
    verdict = "pending" if review is None else review["verdict"]
    metrics["answer_review"] = verdict
    if not execution["execution_success"] or verdict == "fail":
        metrics["task_success"] = False
    elif verdict == "pass":
        metrics["task_success"] = True
    else:
        metrics["task_success"] = None
    # Recovery task success includes the answer, unlike recovery execution.
    metrics["recovery_success"] = (
        metrics["task_success"] if case.get("requires_recovery") else None
    )
    metrics["final_answer_match"] = {"pass": True, "fail": False}.get(verdict)
    if review is not None:
        metrics["answer_review_provenance"] = review
    return metrics


def _task_summary(rows):
    count = len(rows)
    passed = sum(row["metrics"]["task_success"] is True for row in rows)
    pending = sum(row["metrics"]["task_success"] is None for row in rows)
    execution = sum(row["metrics"]["execution_success"] is True for row in rows)
    uncertain = sum(row["metrics"]["answer_review"] == "uncertain" for row in rows)
    args = [row["metrics"]["structured_argument_accuracy"] for row in rows
            if row["metrics"]["structured_argument_accuracy"] is not None]
    return {
        "num_cases": count, "task_pass_count": passed,
        "task_fail_count": count - passed - pending,
        "task_unresolved_count": pending, "answer_uncertain_count": uncertain,
        "task_success_rate": passed / count if count and not pending else None,
        # These are logical bounds, NOT confidence intervals.
        "task_success_lower_bound": passed / count if count else None,
        "task_success_upper_bound": (passed + pending) / count if count else None,
        "task_execution_success_rate": execution / count if count else None,
        "structured_argument_accuracy": sum(args) / len(args) if args else None,
    }


def summarize_results(records):
    version = dataset_protocol([row["case"] for row in records])
    summary = aggregate_results(records)
    summary["protocol_version"] = version
    if version == "2.0":
        return summary
    groups = [(summary["overall"], records)]
    for field in ("category", "difficulty", "scenario_family"):
        for key, group_summary in summary["by_" + field].items():
            groups.append((group_summary, [row for row in records
                                          if str(row["case"].get(field) or "unknown") == key]))
    for group_summary, rows in groups:
        group_summary.update(_task_summary(rows))
        long_rows = [row for row in rows if row["case"].get("category") == "long_horizon"]
        recovery_rows = [row for row in rows if row["case"].get("requires_recovery")]
        group_summary["long_horizon_success_rate"] = _task_summary(long_rows)["task_success_rate"]
        group_summary["recovery_success_rate"] = _task_summary(recovery_rows)["task_success_rate"]
    # Never silently average only categories whose answer reviews are complete.
    for metric in ("task_success_rate", "long_horizon_success_rate", "recovery_success_rate"):
        summary["category_macro"][metric] = None
    if all(row["metrics"]["task_success"] is not None for row in records) and records:
        rates = [group["task_success_rate"] for group in summary["by_category"].values()]
        summary["category_macro"]["task_success_rate"] = sum(rates) / len(rates)
    summary["note"] = "Execution is not task success. Pending/uncertain answers prevent a final task rate; bounds are not confidence intervals. Tool matching and argument_accuracy remain legacy diagnostics."
    return summary
