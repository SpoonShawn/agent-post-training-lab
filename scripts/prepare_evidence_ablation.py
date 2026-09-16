"""Freeze a paired data-layout study before inference; fail closed on drift."""
import argparse
from collections import Counter
import hashlib
import json

from scripts.prepare_control_v1 import ROOT, digest
from training.evidence_ablation import build, shape, training_rows, with_style, STYLES, SEED
from training.evidence_pilot import canonical

DATA = ROOT / "data/evidence_ablation_v1"
CODE = ["training/evidence_ablation.py", "scripts/prepare_evidence_ablation.py",
        "scripts/evidence_ablation_gpu.py", "scripts/superpod_evidence_ablation.sh",
        "training/evidence_pilot.py", "training/sft.py", "agent/baseline_runner.py",
        "agent/guarded_runtime.py", "agent/environment.py", "tools/environment_tools.py",
        "tools/executor.py", "tools/tool_schema.py"]
OLD = [f"data/evidence_pilot_v1/{s}_cases.jsonl" for s in ("train", "validation", "confirmation")] + [
    "benchmarks/evidence_stress_v1/cases.jsonl"]


def contents():
    rows = build()
    old = [json.loads(l) for p in OLD for l in (ROOT/p).read_text().splitlines()]
    old_shapes = {shape(r) for r in old}
    if any(shape(r) in old_shapes for r in rows):
        raise ValueError("Old event skeleton reused")
    groups, shapes, files, counts = {}, {}, {}, {}
    for split in ("train", "validation", "confirmation"):
        selected = [r for r in rows if r["split"] == split]
        groups[split] = sorted({r["group_id"] for r in selected})
        shapes[split] = {shape(r) for r in selected}
        counts[split] = dict(contexts=len(selected), groups=len(groups[split]),
                             families=dict(Counter(r["family"] for r in selected)))
        files[f"{split}_contexts.jsonl"] = "".join(canonical(r)+"\n" for r in selected)
        if split != "confirmation":
            for role in ("fixed", "mixed"):
                files[f"{role}_{split}.jsonl"] = "".join(canonical(r)+"\n" for r in training_rows(selected, role))
        if split != "train":
            files[f"{split}_cases.jsonl"] = "".join(canonical(with_style(r, s))+"\n" for r in selected for s in STYLES)
    for a,b in (("train", "validation"), ("train", "confirmation"), ("validation", "confirmation")):
        if set(groups[a]) & set(groups[b]) or shapes[a] & shapes[b]:
            raise ValueError("Group or event skeleton crosses split")
    return files, counts, groups


def bundle(verify=True):
    from scripts.evidence_stress_v1 import verify_bundle
    verify_bundle()
    files, counts, groups = contents()
    control = json.loads((ROOT / "data/control_v1/manifest.json").read_text())
    manifest = dict(study="evidence_ablation_v1", seed=SEED, counts=counts, groups=groups,
        code={p:digest(ROOT/p) for p in CODE}, rejection_sources={p:digest(ROOT/p) for p in OLD},
        files={p:hashlib.sha256(s.encode()).hexdigest() for p,s in files.items()},
        model_sha256=control["model_sha256"], versions=control["versions"],
        training=dict(epochs=1, examples_per_arm=192, expected_steps=24, r=16, alpha=32,
                      dropout=.05, lr=1e-4, batch=1, accumulation=8, seed=SEED,
                      initialization="fresh_base", selection="last_epoch_no_confirmation_selection"),
        generation=dict(max_length=4096, max_new_tokens=512, do_sample=False),
        comparison="fixed vs mixed evidence layouts; identical histories and supervised answers; not equal input tokens or FLOPs",
        primary="confirmation per-layout paired exact_report; 48 contexts/8 groups, 3 layouts; not 144 independent tasks",
        scope="new event skeleton groups, shared simulator/primitives; not proof of real-world or semantic independence")
    files["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2)+"\n"
    for name,value in files.items():
        path = DATA/name
        if path.exists() and path.read_text() != value:
            raise ValueError(f"Frozen content drift: {name}")
        if verify and not path.exists():
            raise ValueError(f"Missing frozen file: {name}")
    if not verify:
        DATA.mkdir(parents=True, exist_ok=True)
        for name,value in files.items():
            if not (DATA/name).exists():
                (DATA/name).write_text(value)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true")
    args = p.parse_args()
    print(json.dumps(bundle(verify=not args.freeze)["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
