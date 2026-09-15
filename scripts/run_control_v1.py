"""Dedicated 2.4 runner. Legacy runner/scoring remain byte-for-byte unchanged."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.prepare_control_v1 import ROOT, digest
from scripts.verify_control_v1 import verify
from scripts.run_baseline import build_run_metadata, load_completed_ids, package_version
from evaluation.protocol_v24 import evaluate_case


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("base", "sft"), required=True)
    parser.add_argument("--model-path", default="/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Hash local model/adapter and check versions, without loading model")
    args = parser.parse_args(argv)
    if args.role == "base" and args.adapter_path is not None:
        parser.error("Base must not load an adapter")
    if args.role == "sft" and args.adapter_path is None:
        args.adapter_path = ROOT / "checkpoints/pilot_v1_lora/adapter"
    if args.output_path is None:
        args.output_path = ROOT / f"results/baseline/control_v1_{args.role}.jsonl"
    return args


def metadata_for(args, cases, manifest):
    options = SimpleNamespace(
        model_path=args.model_path, adapter_path=args.adapter_path, model_revision=None,
        eval_path=ROOT / "data/control_v1/cases.jsonl", max_steps=None, max_new_tokens=512)
    compatible = deepcopy(cases)
    for c in compatible:
        c["protocol_version"] = "2.3"
    meta = build_run_metadata(options, compatible)
    if meta["model_content_sha256"] != manifest["model_sha256"]:
        raise ValueError("Base weights differ from frozen study")
    expected_adapter = manifest["adapter_sha256"] if args.role == "sft" else None
    if meta.get("adapter_sha256") != expected_adapter:
        raise ValueError("Adapter differs from frozen study")
    actual = {k: package_version(k) for k in manifest["versions"]}
    if actual != manifest["versions"]:
        raise ValueError(f"Runtime packages differ from planned versions: {actual}")
    meta.update(evaluator_version="2.4", study="control_v1", role=args.role,
                extension_sha256=manifest["extension_sha256"],
                study_manifest_sha256=digest(ROOT / "data/control_v1/manifest.json"),
                training_library_versions=actual)
    meta.pop("fingerprint", None)
    meta["fingerprint"] = hashlib.sha256(
        json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return meta


def validate_resume(path, metadata, cases):
    expected = {c["id"]: c for c in cases}
    # Existing tested helper repairs only a truncated last record and prints notice.
    ids = load_completed_ids(path, metadata["fingerprint"])
    if not ids <= set(expected):
        raise ValueError("Unknown resumed cases")
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get("evaluator_version") != "2.4" or row.get("run_metadata") != metadata
                    or row["case"] != expected[row["case"]["id"]]
                    or row["result"]["query"] != row["case"]["query"]):
                raise ValueError("Resume case/query/metadata mismatch")
    return ids


def run_cases(agent, cases, metadata, handle, execution_context=None):
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
        started = datetime.now(timezone.utc).isoformat()
        result = agent.run(case["query"], environment=case["environment"], max_steps=case["max_steps"])
        metrics = evaluate_case(case, result)
        record = dict(evaluator_version="2.4", run_metadata=metadata, case=case, result=result, metrics=metrics,
                      started_at=started, completed_at=datetime.now(timezone.utc).isoformat(),
                      execution_context=execution_context)
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        print("Execution:", metrics["execution_success"], "answer: pending/not yet reviewed", flush=True)


def main(argv=None):
    args = parse_args(argv)
    manifest, cases = verify()
    if args.output_path.exists() and not args.resume and not args.preflight_only:
        raise FileExistsError("Output exists; use --resume or a new path")
    print("Verifying base/adapter contents; hashing may take time...", flush=True)
    metadata = metadata_for(args, cases, manifest)
    if args.preflight_only:
        print("Local model/adapter/version preflight passed; model not loaded.")
        return
    completed = validate_resume(args.output_path, metadata, cases) if args.resume else set()
    pending = [c for c in cases if c["id"] not in completed]
    if not pending:
        print("Already complete; no model loaded.")
        return
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Apply for a bf16-capable GPU; do not infer on a login node")
    from agent.baseline_runner import BaselineAgent
    agent = BaselineAgent(args.model_path, max_steps=16, max_new_tokens=512, adapter_path=args.adapter_path)
    context = dict(gpu_name=torch.cuda.get_device_name(0), cuda_runtime=torch.version.cuda,
                   gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("a" if args.resume else "x") as handle:
        run_cases(agent, pending, metadata, handle, execution_context=context)
    print("Results saved:", args.output_path, flush=True)


if __name__ == "__main__":
    main()
