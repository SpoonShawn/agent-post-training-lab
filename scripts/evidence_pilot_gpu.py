"""Isolated structured-report pilot. No changes to old training or Agent scores."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys

from scripts.prepare_control_v1 import ROOT, digest
from scripts.run_baseline import sha256_model_path
from training.evidence_pilot import canonical, score
from training.sft import load_trajectories, encode_examples, AssistantCollator
from agent.baseline_runner import SYSTEM_PROMPT
from tools.tool_schema import TOOLS

DATA = ROOT / "data/evidence_pilot_v1"
OUTPUT = ROOT / "checkpoints/evidence_pilot_v1"
RESULTS = ROOT / "results/evidence_pilot_v1"


def sha256_adapter(path):
    value, _ = sha256_model_path(path)
    if value is None:
        raise ValueError("Missing adapter directory")
    return value


def preflight(model_path):
    subprocess.run([sys.executable, "-m", "scripts.prepare_evidence_pilot", "--verify"],
                   cwd=ROOT, check=True)
    from scripts.verify_control_v1 import verify
    control, _ = verify()
    versions = {p: importlib.metadata.version(p) for p in control["versions"]}
    if versions != control["versions"]:
        raise ValueError("Package versions differ from planned study")
    model_hash, _ = sha256_model_path(model_path)
    if model_hash != control["model_sha256"]:
        raise ValueError("Base weights changed")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train = encode_examples(load_trajectories(DATA/"train_trajectories.jsonl", "train"), tokenizer, 4096)
    val = encode_examples(load_trajectories(DATA/"validation_trajectories.jsonl", "validation"), tokenizer, 4096)
    prompt_lengths = []
    for split in ("validation", "confirmation"):
        for line in (DATA/f"{split}_cases.jsonl").read_text().splitlines():
            case = json.loads(line)
            prompt = tokenizer.apply_chat_template(
                [dict(role="system", content=SYSTEM_PROMPT), dict(role="user", content=case["query"])],
                tools=TOOLS, add_generation_prompt=True, tokenize=False)
            prompt_lengths.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
    if max(prompt_lengths)+512 > 4096:
        raise ValueError("Evaluation prompt plus generation budget exceeds frozen context")
    meta = dict(manifest_sha256=digest(DATA/"manifest.json"), model_sha256=model_hash,
                versions=versions, seed=20260915, max_length=4096, max_new_tokens=512,
                train_examples=len(train), validation_examples=len(val),
                max_train_validation_tokens=max(len(e["input_ids"]) for e in train+val),
                supervised_tokens=sum(sum(x != -100 for x in e["labels"]) for e in train),
                max_eval_prompt_tokens=max(prompt_lengths),
                chat_template_sha256=hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest())
    return tokenizer, train, val, meta, control


def evaluate(model_path, role, meta, control):
    adapter = None
    if role == "old_sft":
        adapter = ROOT/"checkpoints/pilot_v1_lora/adapter"
        if sha256_adapter(adapter) != control["adapter_sha256"]:
            raise ValueError("Old adapter changed")
    elif role == "new_sft":
        adapter = OUTPUT/"adapter"
        run = json.loads((OUTPUT/"run.json").read_text())
        if run["status"] != "complete" or run["metadata"] != meta:
            raise ValueError("New training provenance mismatch")
        if sha256_adapter(adapter) != run["adapter_sha256"]:
            raise ValueError("New adapter changed after save")
    metadata = dict(meta, role=role, adapter_sha256=sha256_adapter(adapter) if adapter else None)
    metadata["fingerprint"] = hashlib.sha256(canonical(metadata).encode()).hexdigest()
    cases = [json.loads(l) for split in ("validation","confirmation")
             for l in (DATA/f"{split}_cases.jsonl").read_text().splitlines()]
    path = RESULTS/f"{role}.jsonl"
    complete = set()
    if path.exists():
        expected = {c["id"]: c for c in cases}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["metadata"] != metadata or row["case"] != expected.get(row["case"]["id"]):
                raise ValueError("Resume source/metadata drift")
            if row["case"]["id"] in complete:
                raise ValueError("Duplicate result")
            complete.add(row["case"]["id"])
    pending = [c for c in cases if c["id"] not in complete]
    if not pending:
        return
    from agent.baseline_runner import BaselineAgent
    agent = BaselineAgent(model_path, max_new_tokens=512, adapter_path=adapter)
    import torch
    RESULTS.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for index, case in enumerate(pending, 1):
            started = datetime.now(timezone.utc).isoformat()
            answer = agent.generate([dict(role="system", content=SYSTEM_PROMPT),
                                     dict(role="user", content=case["query"])])
            row = dict(case=case, answer=answer, metrics=score(answer, case["target"]),
                       metadata=metadata, started_at=started,
                       completed_at=datetime.now(timezone.utc).isoformat(),
                       gpu=torch.cuda.get_device_name(0))
            handle.write(canonical(row)+"\n")
            handle.flush()
            print(f"{role} [{index}/{len(pending)}] exact_report={row['metrics']['exact_report']}", flush=True)


def train_model(model_path, tokenizer, train, val, meta, smoke):
    import torch
    from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, set_seed
    from peft import get_peft_model, LoraConfig
    folder = ROOT/"checkpoints/evidence_pilot_v1_smoke" if smoke else OUTPUT
    if folder.exists() and any(folder.iterdir()):
        record = json.loads((folder/"run.json").read_text())
        if record.get("status") == "complete" and record["metadata"] == meta:
            print("Training already completed; preserving checkpoint.")
            return
        raise ValueError("Existing incomplete/different training; preserve it and request diagnosis")
    if not smoke:
        # Require complete matching Base AND old-adapter evidence before training.
        for role in ("base", "old_sft"):
            rows = [json.loads(l) for l in (RESULTS/f"{role}.jsonl").read_text().splitlines()]
            expected = [json.loads(l) for s in ("validation","confirmation")
                        for l in (DATA/f"{s}_cases.jsonl").read_text().splitlines()]
            if ({r["case"]["id"] for r in rows} != {c["id"] for c in expected}
                    or len(rows) != len(expected) or any(
                        any(r["metadata"].get(k) != v for k,v in meta.items()) for r in rows)):
                raise ValueError("Run complete frozen pre-training comparisons first")
            by_id = {c["id"]: c for c in expected}
            control = json.loads((ROOT/"data/control_v1/manifest.json").read_text())
            for row in rows:
                metadata = row["metadata"]
                payload = {k:v for k,v in metadata.items() if k != "fingerprint"}
                if (row["case"] != by_id[row["case"]["id"]] or metadata.get("role") != role
                        or metadata.get("adapter_sha256") != (None if role == "base" else control["adapter_sha256"])
                        or metadata.get("fingerprint") != hashlib.sha256(canonical(payload).encode()).hexdigest()
                        or row["metrics"] != score(row["answer"],row["case"]["target"])):
                    raise ValueError("Pre-training evidence content/role/score mismatch")
    folder.mkdir(parents=True, exist_ok=True)
    record = dict(status="started", metadata=meta, started_at=datetime.now(timezone.utc).isoformat(),
                  gpu=torch.cuda.get_device_name(0),
                  training_config=dict(epochs=1, lr=1e-4, r=16, alpha=32, dropout=.05,
                                       batch=1, accumulation=8, packing=False,
                                       selection="last_not_best_on_confirmation", initialization="fresh_base_lora"))
    if not smoke:
        record["baseline_sources"] = {r: digest(RESULTS/f"{r}.jsonl") for r in ("base","old_sft")}
    def save():
        value = json.dumps(record, ensure_ascii=False, indent=2)+"\n"
        (folder/"run.json").write_text(value)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS/("smoke_run.json" if smoke else "training_run.json")).write_text(value)
    save()
    try:
        set_seed(20260915)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=False)
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32,
                                                lora_dropout=.05, target_modules="all-linear", bias="none"))
        record["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        options = TrainingArguments(output_dir=str(folder), num_train_epochs=1,
            max_steps=2 if smoke else -1, learning_rate=1e-4,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=8, bf16=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}, optim="adamw_torch",
            lr_scheduler_type="cosine", warmup_steps=2, eval_strategy="no",
            save_strategy="no", logging_steps=1, report_to="none",
            remove_unused_columns=False, label_names=["labels"], seed=20260915, data_seed=20260915)
        trainer = Trainer(model=model, args=options, train_dataset=train,
                          eval_dataset=val[:4] if smoke else val,
                          data_collator=AssistantCollator(tokenizer.pad_token_id))
        result = trainer.train()
        evaluation = trainer.evaluate()
        if not math.isfinite(result.training_loss) or not math.isfinite(evaluation["eval_loss"]):
            raise ValueError("Non-finite loss")
        model.save_pretrained(folder/"adapter")
        tokenizer.save_pretrained(folder/"adapter")
        trainer.save_state()
        record["trainer_state"] = json.loads((folder/"trainer_state.json").read_text())
        record.update(status="complete", smoke=smoke, training_metrics=result.metrics,
                      evaluation_metrics=evaluation, adapter_sha256=sha256_adapter(folder/"adapter"),
                      completed_at=datetime.now(timezone.utc).isoformat())
        save()
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        save()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight","base","old_sft","smoke","train","new_sft"), required=True)
    parser.add_argument("--model-path", default="/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()
    if args.stage != "preflight":
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Allocate a bf16 GPU before inference/training")
    print("Checking frozen bundle, local weights, packages and tokenization...", flush=True)
    tokenizer, train, val, meta, control = preflight(args.model_path)
    if args.stage == "preflight":
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS/"preflight.json"
        value = json.dumps(meta, ensure_ascii=False, indent=2)+"\n"
        if path.exists() and path.read_text() != value:
            raise ValueError("Preflight changed; do not overwrite")
        path.write_text(value)
        print(value)
    elif args.stage in ("smoke","train"):
        train_model(args.model_path, tokenizer, train, val, meta, args.stage == "smoke")
    else:
        evaluate(args.model_path, args.stage, meta, control)


if __name__ == "__main__":
    main()
