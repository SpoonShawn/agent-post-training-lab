"""CPU-only contract inventory and executable counterexamples, not model scores."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.scoring import evaluate_case, dataset_protocol
from tools.environment_tools import reset_environment, inspect_ui_state
from tools.executor import execute_tool_json

INPUT = ROOT / "results/audit/v22/benchmark_v22.jsonl"


def inventory(cases):
    """Reviewed template-specific risks; not a general natural-language judge."""
    rows = []
    for case in cases:
        family, query = case["scenario_family"], case["query"]
        risks = []
        rules = case["success_criteria"]["required_tool_results"]
        if any(rule["tool"] == "query_logs" for rule in rules) and "日志" not in query:
            risks.append("mandatory_logs_not_explicit_in_prompt")
        if family in {"reproduce_black_screen", "negative_control_ios", "workaround_black_screen"}:
            risks.append("knowledge_before_action_order_not_enforced")
        if "不做多余跳转" in query or "不重复改设置" in query:
            risks.append("explicit_no_redundant_mutation_not_enforced")
        if "不要改变界面" in query:
            risks.append("explicit_read_only_not_enforced_throughout_trace")
        if family == "build_info":
            risks.append("build_tool_identity_required_despite_equivalent_state_tool")
        rows.append({
            "id": case["id"], "scenario_id": case["scenario_id"],
            "family": family, "query": query, "risk_flags": risks,
            "required_tool_results": rules,
            "process_constraints": case.get("process_constraints"),
            "reviewer_type": "protocol_author_ai_template_audit",
            "scope": "candidate_contract_gap_not_model_failure",
        })
    return rows


def simulate(case, calls):
    reset_environment(case["environment"])
    trace = []
    for step, call in enumerate(calls):
        trace.append({
            "step": step,
            "parsed_tool_calls": [{"valid": True, "tool_call": call}],
            "tool_results": [{"tool_call": call, "result": execute_tool_json(call)}],
        })
    result = {
        "query": case["query"], "trajectory": trace,
        "final_environment_state": inspect_ui_state(),
        "final_answer": None,
    }
    metrics = evaluate_case(case, result)
    return {"result": result, "execution_success": metrics["execution_success"],
            "task_success": metrics["task_success"]}


def probes(cases):
    """Mutate real executable oracle plans. No model or semantic labels involved."""
    def pick(family, predicate=lambda c: True):
        return next(c for c in cases if c["scenario_family"] == family and predicate(c))

    specs = []
    case = pick("negative_control_ios")
    specs.append(("remove_unrequested_logs", case,
                  [c for c in case["required_calls"] if c["name"] != "query_logs"],
                  "题面要求验证正常加载，但未明确要求日志；仍到达正确终态却被执行规则拒绝。"))
    case = pick("reproduce_black_screen")
    calls = [c for c in case["required_calls"] if c["name"] != "search_knowledge"]
    calls += [c for c in case["required_calls"] if c["name"] == "search_knowledge"]
    specs.append(("move_knowledge_after_execution", case, calls,
                  "题面明确先核对知识再复现；检索后置仍通过执行评分。"))
    case = pick("navigate_graphics", lambda c: "不做多余跳转" in c["query"])
    extra = [{"name": "execute_action", "arguments": {"action": "back_home"}},
             {"name": "navigate_ui", "arguments": {"target_page": "settings"}},
             {"name": "navigate_ui", "arguments": {"target_page": "graphics"}}]
    specs.append(("leave_and_return_despite_no_extra_navigation", case,
                  extra + case["required_calls"],
                  "已在图形页且题目禁止多余跳转；离开再返回仍通过执行评分。"))
    case = pick("build_info")
    specs.append(("equivalent_build_state_tool", case,
                  [{"name": "inspect_ui_state", "arguments": {}}],
                  "inspect返回版本和平台，但评分绑定get_build_info而拒绝等价证据。"))
    results = []
    for name, case, calls, explanation in specs:
        control = simulate(case, case["required_calls"])
        mutation = simulate(case, calls)
        results.append({"probe": name, "case_id": case["id"], "query": case["query"],
                        "explanation": explanation, "control": control, "mutation": mutation,
                        "kind": "synthetic_execution_probe_not_model_inference"})
    reset_environment()
    return results


def build_report(cases, source_sha):
    if len(cases) != 360 or dataset_protocol(cases) != "2.2":
        raise ValueError("This reviewed inventory is scoped to the 360-case v2.2 benchmark")
    rows = inventory(cases)
    counts = Counter(flag for row in rows for flag in row["risk_flags"])
    # Exact equality of environment + reference execution, not semantic independence.
    signatures = Counter(json.dumps(
        {"environment": c["environment"], "required_calls": c["required_calls"]},
        sort_keys=True, ensure_ascii=False) for c in cases)
    summary = {
        "status": "contract_inventory_complete_protocol_not_frozen",
        "source_sha256": source_sha, "num_cases": len(cases),
        "scenario_ids": len({c["scenario_id"] for c in cases}),
        "exact_environment_reference_signatures": len(signatures),
        "flagged_cases": sum(bool(r["risk_flags"]) for r in rows),
        "risk_counts": dict(counts),
        "knowledge_records": {name: len(json.loads((ROOT / "data/knowledge" / name).read_text()))
                              for name in ("incident_history.json", "build_notes.json", "ui_guide.json")},
        "note": "Overlapping template risks, not counts of model failures. No scoring changes, no new model inference, no automatic semantic adjudication. Existing held_out metadata is historical; this is a development/audit set.",
    }
    return rows, probes(cases), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/audit/contracts_v22")
    args = parser.parse_args()
    cases = [json.loads(line) for line in INPUT.read_text().splitlines()]
    rows, probe_rows, summary = build_report(cases, hashlib.sha256(INPUT.read_bytes()).hexdigest())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("inventory.jsonl", rows), ("probes.jsonl", probe_rows)):
        (args.output_dir / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
