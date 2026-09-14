"""Prepare reproducible grouped pilot data and prospective legacy diagnostic set."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.pilot_data import build_cases, oracle, SPLIT_SEED
from evaluation.protocol_v23 import revise_case
from scripts.run_baseline import RUNTIME_FILES


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if path.exists() and path.read_text() != content:
        raise ValueError(f"Frozen output differs; create a new dataset version: {path}")
    path.write_text(content, encoding="utf-8")


def main():
    cases = build_cases()
    # Old data is consulted ONLY after independent construction, for rejection.
    legacy = [json.loads(line) for line in (ROOT / "data/eval/bugops_eval_v2.jsonl").read_text().splitlines()]
    old_queries = {c["query"] for c in legacy}
    semantic_key = lambda c: json.dumps(
        {"environment": c["environment"], "calls": c["required_calls"]}, ensure_ascii=False, sort_keys=True)
    old_keys = {semantic_key(c) for c in legacy}
    if any(c["query"] in old_queries or semantic_key(c) in old_keys for c in cases):
        raise ValueError("Exact legacy contamination")
    if any("INC-101" in c["query"] or "INC-102" in c["query"] for c in cases):
        raise ValueError("Legacy knowledge event in pilot")
    files, counts, groups = {}, {}, {}
    for split in ("train", "validation", "confirmation"):
        subset = [c for c in cases if c["split"] == split]
        groups[split] = sorted({c["group_id"] for c in subset})
        path = ROOT / f"data/pilot_v1/{split}_cases.jsonl"
        write_rows(path, subset)
        files[str(path.relative_to(ROOT))] = digest(path)
        records = [oracle(case) for case in subset]
        counts[split] = {"trajectories": len(records), "groups": len(groups[split]),
                         "assistant_turns": sum(sum(m["role"] == "assistant" for m in r["messages"]) for r in records)}
        if split != "confirmation":
            path = ROOT / f"data/pilot_v1/{split}_trajectories.jsonl"
            write_rows(path, records)
            files[str(path.relative_to(ROOT))] = digest(path)
    if any(set(groups[a]) & set(groups[b]) for a, b in
           (("train", "validation"), ("train", "confirmation"), ("validation", "confirmation"))):
        raise ValueError("Group contamination")
    path = ROOT / "data/eval/bugops_development_v23.jsonl"
    write_rows(path, [revise_case(c) for c in legacy])
    files[str(path.relative_to(ROOT))] = digest(path)
    code_paths = list(RUNTIME_FILES) + [
        "training/pilot_data.py", "training/sft.py", "scripts/train_sft.py",
        "scripts/prepare_training_pilot.py", "scripts/verify_training_bundle.py",
        "scripts/superpod_train_pilot.sh", "requirements-sft.txt"]
    code = {p: digest(ROOT / p) for p in code_paths}
    manifest = {"version": "pilot_v1", "protocol_version": "2.3", "split_seed": SPLIT_SEED,
                "counts": counts, "groups": groups, "files": files, "code_sha256": code,
                "legacy_exact_query_overlap": 0, "legacy_exact_execution_overlap": 0,
                "scope": "configuration-transaction pilot; no knowledge training; compositional group holdout, not independent external benchmark",
                "review_status": "all executable oracles checked; template-authored answers, not independent human review"}
    path = ROOT / "data/pilot_v1/manifest.json"
    content = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() != content:
        raise ValueError("Frozen manifest differs; bump dataset version")
    path.write_text(content, encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
