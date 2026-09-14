#!/usr/bin/env python3
"""Summarize BugOps baseline JSONL using its declared evaluation protocol."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.scoring import (  # noqa: E402
    dataset_protocol, evaluate_case, load_reviews, protocol_version, review_queue, summarize_results,
)


DEFAULT_INPUT = (
    ROOT
    / "results"
    / "baseline"
    / "qwen3_4b_baseline_v2.jsonl"
)

DISPLAY_METRICS = [
    ("task_success_rate", "Task success"),
    ("long_horizon_success_rate", "Long-horizon success"),
    ("recovery_success_rate", "Recovery success"),
    ("tool_precision", "Tool precision"),
    ("tool_recall", "Tool recall"),
    ("ordered_tool_match_rate", "Exact ordered-tool match"),
    ("tool_set_match_rate", "Tool-set match"),
    ("argument_accuracy", "Argument accuracy"),
    ("valid_call_rate", "Valid call rate"),
    ("execution_success_rate", "Raw execution success"),
    ("adjusted_execution_success_rate", "Adjusted execution success"),
    ("unknown_tool_rate", "Unknown tool rate"),
    ("repeated_tool_call_rate", "Redundant repeat rate"),
    ("average_num_steps", "Average Agent steps"),
]


def load_records(path, recompute=False):
    records = []
    seen_ids = set()
    run_fingerprints = set()
    evaluator_versions = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number} 不是合法 JSON: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON object。")
            case = record.get("case")
            result = record.get("result")
            if not isinstance(case, dict) or not isinstance(result, dict):
                raise ValueError(
                    f"{path}:{line_number} 缺少合法的 case 或 result。"
                )
            case_id = case.get("id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(
                    f"{path}:{line_number} 的 case.id 必须是非空字符串。"
                )
            if case_id in seen_ids:
                raise ValueError(f"{path}:{line_number} case 重复: {case_id}")
            seen_ids.add(case_id)

            metadata = record.get("run_metadata")
            if metadata is None:
                run_fingerprints.add("<legacy>")
            elif not isinstance(metadata, dict) or not isinstance(
                metadata.get("fingerprint"), str
            ):
                raise ValueError(
                    f"{path}:{line_number} 的 run_metadata 无效。"
                )
            else:
                run_fingerprints.add(metadata["fingerprint"])

            evaluator_version = record.get("evaluator_version", "<legacy>")
            evaluator_versions.add(str(evaluator_version))
            if evaluator_version != "<legacy>" and str(evaluator_version) != protocol_version(case):
                raise ValueError(f"{path}:{line_number} Evaluator 与 case 协议不一致。")

            if recompute or protocol_version(case) in {"2.1", "2.2"} or not record.get("metrics"):
                record["metrics"] = evaluate_case(case, result)
            elif not isinstance(record.get("metrics"), dict):
                raise ValueError(
                    f"{path}:{line_number} 的 metrics 必须是 object。"
                )
            records.append(record)

    if len(run_fingerprints) > 1:
        raise ValueError(f"{path} 混合了不同运行指纹，拒绝汇总。")
    if len(evaluator_versions) > 1:
        raise ValueError(f"{path} 混合了不同 Evaluator 版本，拒绝汇总。")
    dataset_protocol([record["case"] for record in records])
    return records


def summarize_coverage(records):
    """Describe whether a resumable run contains its full selected case set."""

    completed = len(records)
    selected_counts = set()
    for record in records:
        metadata = record.get("run_metadata")
        if not isinstance(metadata, dict):
            continue
        selected = metadata.get("selected_case_count")
        if (
            isinstance(selected, int)
            and not isinstance(selected, bool)
            and selected >= 0
        ):
            selected_counts.add(selected)

    if len(selected_counts) > 1:
        raise ValueError("结果记录包含不一致的 selected_case_count。")
    selected = next(iter(selected_counts), None)
    if selected is not None and completed > selected:
        raise ValueError(
            f"结果记录数 {completed} 超过选定 case 数 {selected}。"
        )

    return {
        "completed_case_count": completed,
        "selected_case_count": selected,
        "coverage_rate": (
            completed / selected
            if selected
            else 1.0 if selected == 0 and completed == 0 else None
        ),
        "is_complete": completed == selected if selected is not None else None,
    }


def _format_metric(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_text_summary(summary):
    overall = summary.get("overall", {})
    category_macro = summary.get("category_macro", {})
    coverage = summary.get("run_coverage", {})
    print(f"=== BugOps Evaluator v{summary.get('protocol_version', '2')} Summary ===")
    print(f"Cases: {overall.get('num_cases', 0)}")
    if coverage.get("selected_case_count") is not None:
        completed = coverage.get("completed_case_count", 0)
        selected = coverage["selected_case_count"]
        rate = coverage.get("coverage_rate")
        print(
            f"Run coverage: {completed}/{selected} "
            f"({_format_metric(rate)})"
        )
        if coverage.get("is_complete") is False:
            print("WARNING: PARTIAL RESULT — 本次选定 case 尚未全部完成。")
    for key, label in DISPLAY_METRICS:
        print(f"{label}: {_format_metric(overall.get(key))}")
    if summary.get("protocol_version") in {"2.1", "2.2"}:
        print(f"Task execution success (not task success): {_format_metric(overall.get('task_execution_success_rate'))}")
        print(f"Structured argument accuracy: {_format_metric(overall.get('structured_argument_accuracy'))}")
        print(f"Task answers unresolved: {overall.get('task_unresolved_count')}")
        print(f"Task success bounds (not CI): [{_format_metric(overall.get('task_success_lower_bound'))}, {_format_metric(overall.get('task_success_upper_bound'))}]")
        print(summary["note"])
    if category_macro:
        print(
            "Category-macro task success: "
            f"{_format_metric(category_macro.get('task_success_rate'))}"
        )

    categories = summary.get("by_category", {})
    if categories:
        print()
        print("By category:")
        print(
            f"{'category':<18} {'cases':>5} "
            f"{'task':>7} {'tool_f1':>7} {'args':>7} {'steps':>7}"
        )
        for category, values in categories.items():
            print(
                f"{category:<18} "
                f"{values.get('num_cases', 0):>5} "
                f"{_format_metric(values.get('task_success_rate')):>7} "
                f"{_format_metric(values.get('tool_f1')):>7} "
                f"{_format_metric(values.get('argument_accuracy')):>7} "
                f"{_format_metric(values.get('average_num_steps')):>7}"
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, help="2.1/2.2 答案复核 JSONL；校验来源和逐条证据后重算。")
    parser.add_argument("--review-queue", type=Path, help="导出 2.1/2.2 待审答案、实际工具证据及复核模板。")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT,
        help="baseline 结果 JSONL；默认使用 v2 输出。",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出完整机器可读 JSON 汇总。",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="按 case 协议重算；2.1/2.2 始终重算，答案复核通过 --reviews 显式加载。",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    records = load_records(args.input_path, recompute=args.recompute or bool(args.reviews))
    if args.reviews:
        if dataset_protocol([row["case"] for row in records]) not in {"2.1", "2.2"}:
            raise ValueError("--reviews 仅支持 2.1/2.2；旧轨迹请使用 audit_baseline_v21.py。")
        reviews = load_reviews(args.reviews, records, hashlib.sha256(args.input_path.read_bytes()).hexdigest())
        for row in records:
            row["metrics"] = evaluate_case(row["case"], row["result"], reviews.get(row["case"]["id"]))
    summary = summarize_results(records)
    if args.review_queue:
        if summary["protocol_version"] not in {"2.1", "2.2"}:
            raise ValueError("Review queue requires protocol 2.1 or 2.2")
        if args.review_queue.resolve() in {args.input_path.resolve(), args.reviews.resolve() if args.reviews else None}:
            raise ValueError("Review queue must not overwrite source or reviews")
        args.review_queue.parent.mkdir(parents=True, exist_ok=True)
        rows = review_queue(records, hashlib.sha256(args.input_path.read_bytes()).hexdigest())
        args.review_queue.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary["run_coverage"] = summarize_coverage(records)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_text_summary(summary)


if __name__ == "__main__":
    main()
