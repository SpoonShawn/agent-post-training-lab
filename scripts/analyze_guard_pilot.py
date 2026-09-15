"""Source-bound guard study analysis. No frozen runtime changes or new inference."""
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json

from scripts.analyze_control_results import ROOT, read_source
from scripts.run_guard_pilot import verify_spec, MANIFEST
from scripts.prepare_control_v1 import digest
from agent.guarded_runtime import guarded_execute, facts, render_facts
from tools.environment_tools import reset_environment, inspect_ui_state
from evaluation.protocol_v24 import evaluate_case
from evaluation.scoring import record_sha256

SOURCES = {
    "base": "26c322e31f1544d03e7ae4db1972edc97846baaf714482b634aa8efbee460870",
    "sft": "25b1ea60c68846ad0487c5f2f8330d7d3e80aea58077d7e83ecf748fafc17aab",
}


def load_guard(role):
    verify_spec()
    old = read_source(role, ROOT / f"results/baseline/control_v1_{role}.jsonl")
    old = [r for r in old if r["case"]["control_arm"].startswith("readonly_")]
    path = ROOT / f"results/baseline/guard_pilot_v1_{role}.jsonl"
    if digest(path) != SOURCES[role]:
        raise ValueError("Changed source requires new audit")
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    if [r["case"] for r in rows] != [r["case"] for r in old]:
        raise ValueError("Incomplete, changed or reordered cases")
    expected = deepcopy(old[0]["run_metadata"])
    ids = [r["case"]["id"] for r in rows]
    expected.update(selected_case_count=16, selected_case_ids_sha256=hashlib.sha256(
        json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        study="guard_pilot_v1", guard_manifest_sha256=digest(MANIFEST),
        policy={"read_only": True}, reporting="post_run_sidecar_not_model_answer")
    expected.pop("fingerprint")
    expected["fingerprint"] = hashlib.sha256(json.dumps(
        expected, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for row in rows:
        if row["run_metadata"] != expected or row["evaluator_version"] != "2.4":
            raise ValueError("Metadata drift")
        if evaluate_case(row["case"], row["result"]) != row["metrics"]:
            raise ValueError("Metrics mismatch")
        computed = facts(row["result"]["trajectory"])
        if computed != row["system_facts"] or render_facts(computed) != row["system_report"]:
            raise ValueError("System evidence/report mismatch")
        reset_environment(row["case"]["environment"])
        for turn in row["result"]["trajectory"]:
            for entry in turn.get("tool_results", []):
                if guarded_execute(entry["tool_call"], read_only=True) != entry["result"]:
                    raise ValueError("Guarded replay mismatch")
        if inspect_ui_state() != row["result"]["final_environment_state"]:
            raise ValueError("Final state replay mismatch")
    return rows, old


def analyze():
    summary, evidence, reviews = {}, [], []
    for role in SOURCES:
        rows, old = load_guard(role)
        old_reviews = {r["id"]: r for r in map(json.loads, (
            ROOT / f"results/reviews/control_v1/{role}_author_reviews.jsonl").read_text().splitlines())}
        detail = dict(source_sha256=SOURCES[role], run_metadata=rows[0]["run_metadata"],
                      execution_context=rows[0]["execution_context"],
                      first_started=rows[0]["started_at"], last_completed=rows[-1]["completed_at"])
        detail["case_window_seconds"] = (datetime.fromisoformat(detail["last_completed"]) -
                                         datetime.fromisoformat(detail["first_started"])).total_seconds()
        detail["replayed_calls"] = sum(len(t["tool_results"]) for r in rows for t in r["result"]["trajectory"])
        detail["identical_results_to_control"] = sum(r["result"] == b["result"] for r,b in zip(rows,old))
        detail["arms"] = {}
        for arm in ("readonly_explicit", "readonly_checklist"):
            selected = [r for r in rows if r["case"]["control_arm"] == arm]
            before = [r for r in old if r["case"]["control_arm"] == arm]
            detail["arms"][arm] = dict(
                cases=len(selected),
                strict_execution_before=sum(r["metrics"]["execution_success"] for r in before),
                strict_execution_after=sum(r["metrics"]["execution_success"] for r in selected),
                successful_mutations_before=sum(len(facts(r["result"]["trajectory"])["successful_mutations"]) for r in before),
                successful_mutations_after=sum(len(r["system_facts"]["successful_mutations"]) for r in selected),
                blocked_calls=sum(r["system_facts"]["policy_block_count"] for r in selected),
                cases_with_blocks=sum(r["system_facts"]["policy_block_count"] > 0 for r in selected),
                state_and_final_logs=sum(not r["system_facts"]["missing_fields"] and
                                         r["system_facts"]["logs_after_last_attempt"] for r in selected))
        verdicts = Counter()
        for row, prior in zip(rows, old):
            if row["metrics"]["execution_success"]:
                if row["result"] != prior["result"]:
                    raise ValueError("Changed eligible answer requires new semantic review")
                review = deepcopy(old_reviews[row["case"]["id"]])
                review.update(source_sha256=SOURCES[role],
                              record_sha256=record_sha256(row["case"], row["result"]),
                              reviewed_at="2026-09-15",
                              method="Exact case/result equality verified; inherited stage023 author-AI decision",
                              prior_source_sha256=old_reviews[row["case"]["id"]]["source_sha256"])
                reviews.append(dict(model=role, **review))
                verdicts[review["verdict"]] += 1
            if row["system_facts"]["policy_block_count"]:
                evidence.append(dict(model=role, case=row["case"], result=row["result"],
                                     system_facts=row["system_facts"], system_report=row["system_report"]))
        detail["semantic_verdicts_on_strict_execution_passes"] = dict(verdicts)
        detail["task_pass"] = verdicts["pass"]
        detail["task_pending"] = verdicts["uncertain"]
        detail["task_fail"] = 16-verdicts["pass"]-verdicts["uncertain"]
        detail["task_rate"] = None if verdicts["uncertain"] else verdicts["pass"]/16
        summary[role] = detail
    summary["limitations"] = [
        "System safety effect, not model improvement. Strict attempt-based evaluator unchanged.",
        "32 cases from 8 reused contexts; no independent held-out or statistical significance claim.",
        "20 eligible case/results identical to prior control: inherited author-AI reviews, not independent human review.",
        "System facts verified against tool evidence, not automatically approved as full task answers.",
        "Three blocked SFT cases exhausted budget; one obtained logs but falsely reported zero failures.",
    ]
    return summary, evidence, reviews


def main():
    summary, evidence, reviews = analyze()
    folder = ROOT / "results/analysis/guard_pilot_v1"
    outputs = {"summary.json": json.dumps(summary, ensure_ascii=False, indent=2)+"\n",
               "blocked_cases.jsonl": "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in evidence),
               "inherited_author_reviews.jsonl": "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in reviews)}
    for name, value in outputs.items():
        path = folder/name
        if path.exists() and path.read_text() != value:
            raise ValueError("Refuse changed artifact overwrite")
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        if not (folder/name).exists():
            (folder/name).write_text(value)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
