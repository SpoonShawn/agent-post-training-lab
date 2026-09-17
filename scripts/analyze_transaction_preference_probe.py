"""Audit candidate supply, including the legitimate zero-pair outcome."""
from collections import Counter, defaultdict
from datetime import datetime
import json

from scripts.transaction_preference_probe import PLAN, RESULTS, plan, pair_indices, SAMPLING
from scripts.transaction_gpu_smoke import validate_records, write_once
from scripts.prepare_control_v1 import ROOT, digest
from training.transaction_data import canonical


def audit():
    spec, schedule = plan()
    path = RESULTS / "rollouts.jsonl"
    metadata = dict(plan_sha256=digest(PLAN), adapter_sha256=spec["adapter_sha256"], sampling=SAMPLING)
    records = validate_records(path, metadata, [s["case"] for s in schedule])
    if len(records) != len(schedule):
        raise ValueError("Incomplete candidate coverage")
    groups = defaultdict(list)
    for record, slot in zip(records, schedule):
        if canonical({k: record[k] for k in ("candidate", "seed")}) != canonical({k: slot[k] for k in ("candidate", "seed")}):
            raise ValueError("Candidate order or seed drift")
        if datetime.fromisoformat(record["completed_at"]) < datetime.fromisoformat(record["started_at"]):
            raise ValueError("Invalid episode timestamps")
        groups[record["case"]["id"]].append(record)
    computed = dict(**pair_indices(records), rollouts_sha256=digest(path))
    if canonical(json.loads((RESULTS / "pair_yield.json").read_text())) != canonical(computed):
        raise ValueError("Pair labels or source digest changed")
    summary = dict(scope="train_only_candidate_supply_not_generalization", records=len(records),
        tasks=len(groups), groups=len({r["case"]["group_id"] for r in records}),
        successful_rollouts=sum(r["metrics"]["task_success"] for r in records),
        policy_violating_calls=sum(r["metrics"]["policy_violations"] for r in records),
        usable_pairs=len(computed["pairs"]), abstained_tasks=len(computed["abstained_task_ids"]),
        distinct_trajectories_per_task=dict(Counter(str(len({canonical(r["result"]) for r in values})) for values in groups.values())),
        all_success_tasks=sum(all(r["metrics"]["task_success"] for r in values) for values in groups.values()),
        all_failed_tasks=sum(not any(r["metrics"]["task_success"] for r in values) for values in groups.values()),
        generation_seconds=sum(u["seconds"] for r in records for u in r["generation_usage"]),
        first_started_at=records[0]["started_at"], last_completed_at=records[-1]["completed_at"],
        sources={"plan": digest(PLAN), "rollouts": digest(path), "pair_yield": digest(RESULTS / "pair_yield.json")},
        interpretation="No preference training is justified by tied candidates; saturation is observed only on these sampled train tasks")
    return summary, records


def main():
    summary, _ = audit()
    write_once(ROOT / "results/analysis/transaction_preference_probe_v1/summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
