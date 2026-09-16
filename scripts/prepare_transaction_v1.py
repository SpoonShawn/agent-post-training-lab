"""Freeze structural splits before model inference; verify by deterministic rebuild."""
import argparse
from collections import Counter
import hashlib
import json

from scripts.prepare_control_v1 import ROOT,digest
from training.transaction_data import build_cases,audit_cases,teacher_trajectory,smoke_cases,canonical,SEED,SPLITS

DATA = ROOT/"data/transaction_v1"
CODE = ["agent/transaction_env.py","agent/transaction_tasks.py","agent/transaction_runtime.py",
        "agent/transaction_model.py","training/transaction_data.py","training/transaction_sft.py",
        "scripts/prepare_transaction_v1.py","scripts/transaction_gpu_smoke.py","scripts/superpod_transaction_smoke.sh",
        "training/sft.py","scripts/run_baseline.py","scripts/evidence_pilot_gpu.py"]


def payloads():
    cases = build_cases()
    stats = audit_cases(cases)
    files = {}
    for split in SPLITS:
        rows = [c for c in cases if c["split"] == split]
        files[f"{split}_cases.jsonl"] = "".join(canonical(c)+"\n" for c in rows)
        if split in ("train","dev"):
            files[f"{split}_trajectories.jsonl"] = "".join(canonical(teacher_trajectory(c)[0])+"\n" for c in rows)
    files["smoke_cases.jsonl"] = "".join(canonical(c)+"\n" for c in smoke_cases(cases))
    # Same public requests can occur under different private fault profiles; disclose rather than hide.
    counts = Counter(c["query"] for c in cases)
    stats["public_query_reuse"] = dict(unique=len(counts),cases=len(cases),
                                      interpretation="same public goal under different hidden environments; group isolation is structural, not prompt-text isolation")
    return files,stats


def bundle(freeze=False):
    files,stats = payloads()
    control = json.loads((ROOT/"data/control_v1/manifest.json").read_text())
    manifest = dict(study="transaction_v1",seed=SEED,counts=stats,code={p:digest(ROOT/p) for p in CODE},
                    files={n:hashlib.sha256(s.encode()).hexdigest() for n,s in files.items()},
                    model_sha256=control["model_sha256"],versions=control["versions"],
                    generation=dict(max_length=8192,max_new_tokens=512,max_turns=40,max_calls=36,do_sample=False),
                    partition="grouped fault/outcome/delay traces; timeout+concurrency and commit-time revocation reserved OOD",
                    confirmation="128 structural-holdout ID + 400 OOD; report separately; no confirmation examples in SFT or smoke",
                    gpu_gate="preflight, Base engineering smoke14, two-step LoRA on longest train prefixes, smoke-adapter14; not formal SFT",
                    planned_training="full 2058 trajectories; optimizer/epoch/throughput budget frozen after real-token GPU gate, before full model comparison",
                    label_scope="structured execution+evidence+report; oracle checks are not model results")
    files["manifest.json"] = json.dumps(manifest,ensure_ascii=False,indent=2)+"\n"
    for name,content in files.items():
        path = DATA/name
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Frozen data/code drift: {name}")
        if not freeze and not path.exists():
            raise ValueError(f"Missing frozen file: {name}")
    if freeze:
        DATA.mkdir(parents=True,exist_ok=True)
        for name,content in files.items():
            if not (DATA/name).exists():
                (DATA/name).write_text(content)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze",action="store_true")
    args=parser.parse_args()
    print(json.dumps(bundle(args.freeze)["counts"],ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
