"""CPU audit of uploaded engineering evidence; never relabel smoke as a benchmark."""
from collections import Counter
from datetime import datetime
import json
import math

from scripts.transaction_gpu_smoke import RESULTS, rows, run_meta, validate_records, write_once
from scripts.prepare_transaction_v1 import DATA, bundle
from scripts.prepare_control_v1 import ROOT, digest
from training.transaction_data import canonical


def audit():
    manifest = bundle()
    meta = json.loads((RESULTS / "preflight.json").read_text())
    run = json.loads((RESULTS / "training_run.json").read_text())
    for key, expected in (("manifest_sha256", digest(DATA / "manifest.json")),
                          ("model_sha256", manifest["model_sha256"]),
                          ("versions", manifest["versions"]), ("generation", manifest["generation"])):
        if canonical(meta[key]) != canonical(expected):
            raise ValueError(f"Preflight drift: {key}")
    if (run["status"] != "complete" or canonical(run["metadata"]) != canonical(meta)
            or run["baseline_sha256"] != digest(RESULTS / "base.jsonl")
            or run["trainer_state"]["global_step"] != 2 or run["config"]["max_steps"] != 2):
        raise ValueError("Training provenance or steps mismatch")
    for split, size in (("train", 16), ("dev", 8)):
        trajectories = {r["id"]: r for r in rows(DATA / f"{split}_trajectories.jsonl")}
        refs = meta["smoke_sample_refs"][split]
        if len(refs) != size or len({(r["case_id"], r["assistant_index"]) for r in refs}) != size:
            raise ValueError("Smoke references incomplete/duplicate")
        for ref in refs:
            n = sum(m["role"] == "assistant" for m in trajectories[ref["case_id"]]["messages"])
            if type(ref["assistant_index"]) is not int or not 0 <= ref["assistant_index"] < n:
                raise ValueError("Invalid assistant reference")
        stats = meta["statistics"][split]
        if stats["examples"] != manifest["counts"][split]["assistant_examples"]:
            raise ValueError("Example count mismatch")
        if not 0 < stats["supervised_tokens"] <= stats["total_input_tokens_including_targets"]:
            raise ValueError("Invalid token totals")
    for loss in (run["training_metrics"]["train_loss"], run["evaluation_metrics"]["eval_loss"]):
        if not math.isfinite(loss):
            raise ValueError("Nonfinite loss")
    cases = list(rows(DATA / "smoke_cases.jsonl"))
    summary = dict(scope="engineering_only_not_model_improvement", sources={
        p.name: digest(p) for p in sorted(RESULTS.glob("*.json*"))}, roles={},
        statistics=meta["statistics"], training_metrics=run["training_metrics"],
        peak_allocated_gib=run["peak_cuda_allocated_bytes"] / 2**30,
        peak_reserved_gib=run["peak_cuda_reserved_bytes"] / 2**30,
        limitations=["token counts and GPU memory are uploaded measurements, not locally rerun",
                     "adapter weights and private token cache not uploaded; their hashes are cross-linked only"])
    failures = []
    for role in ("base", "sft_smoke"):
        records = validate_records(RESULTS / f"{role}.jsonl", run_meta(
            meta, role, run["adapter_sha256"] if role == "sft_smoke" else None), cases)
        if len(records) != 14:
            raise ValueError("Incomplete smoke coverage")
        boundary = run["started_at"] if role == "base" else run["completed_at"]
        for r in records:
            if datetime.fromisoformat(r["completed_at"]) < datetime.fromisoformat(r["started_at"]):
                raise ValueError("Negative episode duration")
            if role == "base" and r["completed_at"] > boundary or role == "sft_smoke" and r["started_at"] < boundary:
                raise ValueError("Training/inference chronology mismatch")
            if not r["metrics"]["task_success"]:
                failures.append(dict(role=role, **r))
        summary["roles"][role] = dict(cases=len(records), **{
            k: sum(r["metrics"][k] for r in records)
            for k in ("execution_success", "answer_correct", "task_success", "policy_violations", "tool_calls")},
            termination=dict(Counter(r["result"]["terminated_reason"] for r in records)),
            tool_errors=dict(Counter(e["result"]["error_type"] for r in records
                                    for e in r["result"]["events"] if not e["result"]["ok"])),
            generation_seconds=sum(u["seconds"] for r in records for u in r["generation_usage"]))
    return summary, failures


def main():
    summary, failures = audit()
    folder = ROOT / "results/analysis/transaction_v1_smoke"
    write_once(folder / "summary.json", summary)
    write_once(folder / "bad_cases.json", failures)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
