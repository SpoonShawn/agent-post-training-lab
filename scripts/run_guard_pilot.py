"""32-run diagnostic: same frozen 16 read-only prompts per model; guard only.

No checklist edits, automatic retries, extra steps or hidden task-answer access.
Facts are post-run sidecars and are never sent to the model.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts.prepare_control_v1 import ROOT, digest
from scripts.verify_control_v1 import verify
from scripts.run_control_v1 import metadata_for, validate_resume
from agent.guarded_runtime import run_guarded, facts, render_facts
from evaluation.protocol_v24 import evaluate_case

FILES = ["agent/guarded_runtime.py", "scripts/run_guard_pilot.py",
         "scripts/superpod_guard_pilot.sh"]
MANIFEST = ROOT / "data/guard_pilot/manifest.json"


def specification():
    manifest, cases = verify()
    selected = [c for c in cases if c["control_arm"].startswith("readonly_")]
    return dict(study="guard_pilot_v1", status="diagnostic_not_heldout", policy={"read_only": True},
                case_ids=[c["id"] for c in selected], control_manifest_sha256=digest(
                    ROOT / "data/control_v1/manifest.json"),
                source_code_sha256={p: digest(ROOT / p) for p in FILES},
                models={k: manifest[k] for k in ("model_sha256", "adapter_sha256")},
                comparison="Existing control raw vs new guarded, same prompts and budgets",
                task_scoring="Frozen 2.4 counts forbidden attempted calls even if blocked",
                reporting="Evidence-only sidecar, not model output; never auto-assign semantic success")


def verify_spec():
    if not MANIFEST.exists() or json.loads(MANIFEST.read_text()) != specification():
        raise ValueError("Guard study missing/changed; do not mix frozen runs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--role", choices=["base", "sft"])
    parser.add_argument("--model-path", default="/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        with MANIFEST.open("x") as handle:
            json.dump(specification(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return
    verify_spec()
    if args.verify_only:
        print("Guard study verified: 16 read-only cases per model; no model loaded.")
        return
    if args.role is None:
        parser.error("--role required for inference")
    if args.role == "base" and args.adapter_path:
        parser.error("Base must not load adapter")
    if args.role == "sft" and args.adapter_path is None:
        args.adapter_path = ROOT / "checkpoints/pilot_v1_lora/adapter"
    manifest, all_cases = verify()
    cases = [c for c in all_cases if c["control_arm"].startswith("readonly_")]
    path = ROOT / f"results/baseline/guard_pilot_v1_{args.role}.jsonl"
    if path.exists() and not args.resume:
        raise FileExistsError("Use --resume; never overwrite results")
    print("Verifying weights and adapter contents; hashing may take time...", flush=True)
    # Base function fingerprints the exact selection as well as the whole source.
    meta = metadata_for(args, cases, manifest)
    meta.update(study="guard_pilot_v1", guard_manifest_sha256=digest(MANIFEST),
                policy={"read_only": True}, reporting="post_run_sidecar_not_model_answer")
    meta.pop("fingerprint", None)
    meta["fingerprint"] = hashlib.sha256(json.dumps(
        meta, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    complete = validate_resume(path, meta, cases) if args.resume else set()
    pending = [c for c in cases if c["id"] not in complete]
    if not pending:
        print("Already complete; no model loaded.")
        return
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Request a bf16 GPU first; no login-node inference")
    from agent.baseline_runner import BaselineAgent
    agent = BaselineAgent(args.model_path, max_steps=16, max_new_tokens=512, adapter_path=args.adapter_path)
    context = dict(gpu_name=torch.cuda.get_device_name(0), cuda_runtime=torch.version.cuda,
                   gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if args.resume else "x") as handle:
        for index, case in enumerate(pending, 1):
            started = datetime.now(timezone.utc).isoformat()
            result = run_guarded(agent.generate, case["query"], case["environment"],
                                 case["max_steps"], read_only=True)
            evidence = facts(result["trajectory"])
            row = dict(evaluator_version="2.4", case=case, result=result,
                       metrics=evaluate_case(case, result), run_metadata=meta,
                       started_at=started, completed_at=datetime.now(timezone.utc).isoformat(),
                       execution_context=context, system_facts=evidence,
                       system_report=render_facts(evidence))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(pending)}] {case['id']} blocks={evidence['policy_block_count']} "
                  f"successful_mutations={len(evidence['successful_mutations'])} "
                  f"strict_execution={row['metrics']['execution_success']}", flush=True)


if __name__ == "__main__":
    main()
