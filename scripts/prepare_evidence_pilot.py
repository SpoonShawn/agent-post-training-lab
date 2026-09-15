"""Generate and verify a new grouped trace-audit dataset. Never overwrite a freeze."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.prepare_control_v1 import ROOT, digest
from training.evidence_pilot import build, trajectory, canonical, SEED

FOLDER = ROOT / "data/evidence_pilot_v1"
CODE = ["training/evidence_pilot.py", "scripts/prepare_evidence_pilot.py",
        "scripts/evidence_pilot_gpu.py", "scripts/superpod_evidence_pilot.sh",
        "training/sft.py", "agent/guarded_runtime.py"]


def contents():
    cases = build()
    # Old data consulted ONLY for rejection after construction.
    queries = set()
    for path in (ROOT / "data").rglob("*.jsonl"):
        if FOLDER in path.parents:
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if "query" in row:
                queries.add(row["query"])
    if any(c["query"] in queries for c in cases):
        raise ValueError("Exact old query overlap")
    files, counts, groups = {}, {}, {}
    for split in ("train", "validation", "confirmation"):
        rows = [c for c in cases if c["split"] == split]
        groups[split] = sorted({c["group_id"] for c in rows})
        counts[split] = dict(cases=len(rows), groups=len(groups[split]),
                             patterns=dict(Counter(c["pattern"] for c in rows)))
        files[f"{split}_cases.jsonl"] = "".join(canonical(c)+"\n" for c in rows)
        if split != "confirmation":
            files[f"{split}_trajectories.jsonl"] = "".join(canonical(trajectory(c))+"\n" for c in rows)
    if any(set(groups[a]) & set(groups[b]) for a,b in (
            ("train","validation"), ("train","confirmation"), ("validation","confirmation"))):
        raise ValueError("Group overlap")
    return files, counts, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    files, counts, groups = contents()
    import hashlib
    manifest = dict(version="evidence_pilot_v1", split_seed=SEED, counts=counts, groups=groups,
                    files={k: hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()},
                    code={p: digest(ROOT/p) for p in CODE},
                    control_manifest_sha256=digest(ROOT/"data/control_v1/manifest.json"),
                    scope="trace-audit subskill; 48 pattern/field-combination groups, not full Agent task success",
                    contamination="Exact old query overlap zero; newly executed trace-audit tasks, shared tool primitives; not proof of semantic independence",
                    training="fresh LoRA from original Base; not continuing old adapter; 1 epoch seed20260915",
                    confirmation_use="freeze before any new model inference; no hyperparameter selection on confirmation")
    files["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2)+"\n"
    for name, value in files.items():
        path = FOLDER/name
        if path.exists() and path.read_text() != value:
            raise ValueError(f"Frozen content drift: {name}")
        if args.verify and not path.exists():
            raise ValueError(f"Missing file: {name}")
    if not args.verify:
        FOLDER.mkdir(parents=True, exist_ok=True)
        for name, value in files.items():
            if not (FOLDER/name).exists():
                (FOLDER/name).write_text(value)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
