"""Frozen first full SFT arm. CPU freeze; GPU baseline -> fresh SFT -> same evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math

from scripts.prepare_control_v1 import ROOT, digest
from scripts.prepare_transaction_v1 import DATA, bundle
from scripts.transaction_gpu_smoke import (RESULTS as SMOKE, checked_tokenizer, rows,
    write_once, validate_records, sha256_adapter, AssistantCollator)
from training.transaction_data import canonical, SEED
from training.transaction_sft import encode_trajectory
from training.transaction_cache import DiskExamples, build_cache, latest_checkpoint
from agent.transaction_runtime import run_episode
from agent.transaction_tasks import evaluate

RESULTS = ROOT / "results/transaction_v1_full_sft"
OUTPUT = ROOT / "checkpoints/transaction_v1_full_sft"
PLAN = ROOT / "data/transaction_v1_full_sft_plan.json"
SPLITS = ("dev", "confirmation_id", "confirmation_ood")
CODE = ("scripts/transaction_full_sft.py", "training/transaction_cache.py",
        "scripts/superpod_transaction_full_sft.sh", "scripts/analyze_transaction_smoke.py")
CONFIG = dict(epochs=1, expected_steps=3943, batch=1, accumulation=8, lr=1e-4,
              r=16, alpha=32, dropout=.05, target_modules="all-linear", seed=SEED,
              optimizer="adamw_torch", scheduler="constant", warmup_steps=0,
              save_steps=250, max_length=8192, bf16=True, gradient_checkpointing=True,
              checkpoint_selection="last_only_no_confirmation_selection")


def protocol(freeze=False):
    manifest = bundle()
    from scripts.analyze_transaction_smoke import audit
    audit()
    value = dict(study="transaction_v1_full_sft_seed20260917", config=CONFIG,
        dataset_sha256=digest(DATA / "manifest.json"), preflight_sha256=digest(SMOKE / "preflight.json"),
        code={name: digest(ROOT / name) for name in CODE}, generation=manifest["generation"],
        splits={s: manifest["counts"][s] for s in SPLITS},
        comparison="Base versus fresh full SFT; engineering adapter never resumed",
        scope="first seed; DPO/GRPO/multiseed/backend transfer still pending",
        recovery_ablation="separate future arm; freeze matched budget and train-only intervention before its GPU run; no causal recovery claim from this arm")
    if freeze:
        write_once(PLAN, value)
    elif not PLAN.exists() or canonical(json.loads(PLAN.read_text())) != canonical(value):
        raise ValueError("Formal protocol/data/code drift; preserve existing study")
    return value


def inference_metadata(plan, role, split, adapter_hash):
    value = dict(plan_sha256=digest(PLAN), role=role, split=split, adapter_sha256=adapter_hash,
                 generation=plan["generation"])
    value["fingerprint"] = hashlib.sha256(canonical(value).encode()).hexdigest()
    return value


def infer(model_path, plan, role):
    adapter = None
    adapter_hash = None
    if role == "sft":
        run = json.loads((RESULTS / "training_run.json").read_text())
        adapter = OUTPUT / "adapter"
        adapter_hash = sha256_adapter(adapter)
        if run["status"] != "complete" or run["plan_sha256"] != digest(PLAN) or run["adapter_sha256"] != adapter_hash:
            raise ValueError("Incomplete/different formal adapter")
    import torch
    from agent.transaction_model import TransactionAgent
    agent = None
    for split in SPLITS:
        cases = list(rows(DATA / f"{split}_cases.jsonl"))
        metadata = inference_metadata(plan, role, split, adapter_hash)
        path = RESULTS / f"{role}_{split}.jsonl"
        completed = validate_records(path, metadata, cases)
        if len(completed) == len(cases):
            continue
        if agent is None:
            agent = TransactionAgent(model_path, adapter_path=adapter)
        with path.open("a") as handle:
            for index, case in enumerate(cases[len(completed):], len(completed) + 1):
                started = datetime.now(timezone.utc).isoformat()
                agent.usage = []
                print(f"{role}/{split} {index}/{len(cases)} starting", flush=True)
                result = run_episode(agent.generate, case["query"], case["environment"], max_turns=40, max_calls=36)
                metrics = evaluate(case["environment"], case["desired"], result["events"], result["final_answer"])
                record = dict(case=case, result=result, metrics=metrics, metadata=metadata,
                    generation_usage=agent.usage, gpu=torch.cuda.get_device_name(0), started_at=started,
                    completed_at=datetime.now(timezone.utc).isoformat())
                handle.write(canonical(record) + "\n")
                handle.flush()
                print(f"calls={metrics['tool_calls']} reason={result['terminated_reason']} (scores sealed until paired audit)", flush=True)


def train(model_path, tokenizer, plan):
    baseline = {}
    for split in SPLITS:
        cases = list(rows(DATA / f"{split}_cases.jsonl"))
        path = RESULTS / f"base_{split}.jsonl"
        if len(validate_records(path, inference_metadata(plan, "base", split, None), cases)) != len(cases):
            raise ValueError("Complete full baseline before training")
        baseline[split] = digest(path)
    record_path = RESULTS / "training_run.json"
    previous = json.loads(record_path.read_text()) if record_path.exists() else None
    if previous and (previous["plan_sha256"] != digest(PLAN) or previous["baseline_sha256"] != baseline):
        raise ValueError("Training inputs changed")
    if previous and previous["status"] == "complete":
        if previous["adapter_sha256"] != sha256_adapter(OUTPUT / "adapter"):
            raise ValueError("Completed adapter changed")
        print("Completed training preserved", flush=True)
        return
    checkpoint = latest_checkpoint(OUTPUT)
    if previous and checkpoint is None:
        raise ValueError("Interrupted before first complete checkpoint; preserve files and request diagnosis")
    if not previous and (checkpoint or (OUTPUT / "adapter").exists()):
        raise ValueError("Untracked training artifacts; preserve and diagnose")
    # Keep partial checkpoint files rather than letting Trainer overwrite them on recovery.
    preserved = []
    if checkpoint:
        step = int(checkpoint.rsplit("-", 1)[-1])
        for path in OUTPUT.glob("checkpoint-*"):
            suffix = path.name.removeprefix("checkpoint-")
            if suffix.isdigit() and int(suffix) > step:
                destination = path.with_name("interrupted-" + path.name + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
                path.rename(destination)
                preserved.append(str(destination))
    meta = json.loads((SMOKE / "preflight.json").read_text())
    datasets = {}
    for split in ("train", "dev"):
        path = OUTPUT / f"{split}.sqlite"
        stamp = OUTPUT / f"{split}.cache.json"
        if path.exists():
            expected = dict(plan_sha256=digest(PLAN), sha256=digest(path))
            if not stamp.exists() or json.loads(stamp.read_text()) != expected:
                raise ValueError("Cache drift or unfinished publication; preserve and diagnose")
            datasets[split] = DiskExamples(path)
        else:
            def examples():
                for i, row in enumerate(rows(DATA / f"{split}_trajectories.jsonl"), 1):
                    yield from encode_trajectory(row, tokenizer)
                    if i % 200 == 0:
                        print(f"Preparing {split}: {i} trajectories", flush=True)
            datasets[split] = build_cache(path, examples(), meta["statistics"][split])
            write_once(stamp, dict(plan_sha256=digest(PLAN), sha256=digest(path)))
    import torch
    from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, set_seed
    from peft import LoraConfig, get_peft_model
    record = dict(status="started", plan_sha256=digest(PLAN), baseline_sha256=baseline, config=CONFIG,
        gpu=torch.cuda.get_device_name(0), attempts=(previous or {}).get("attempts", []) + [
            dict(started_at=datetime.now(timezone.utc).isoformat(), resume_checkpoint=checkpoint,
                 preserved_partial_checkpoints=preserved)])
    def save():
        temporary = record_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(record_path)
    save()
    try:
        set_seed(SEED)
        torch.cuda.reset_peak_memory_stats()
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=False)
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32,
            lora_dropout=.05, target_modules="all-linear", bias="none"))
        options = TrainingArguments(output_dir=str(OUTPUT), num_train_epochs=1, learning_rate=1e-4,
            per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
            bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant":False},
            optim="adamw_torch", lr_scheduler_type="constant", warmup_steps=0,
            eval_strategy="no", save_strategy="steps", save_steps=250, logging_steps=25,
            report_to="none", remove_unused_columns=False, label_names=["labels"], seed=SEED, data_seed=SEED,
            dataloader_num_workers=0)
        trainer = Trainer(model=model, args=options, train_dataset=datasets["train"],
            eval_dataset=datasets["dev"], data_collator=AssistantCollator(tokenizer.pad_token_id))
        trained = trainer.train(resume_from_checkpoint=checkpoint)
        validation = trainer.evaluate()
        if trainer.state.global_step != CONFIG["expected_steps"] or not math.isfinite(validation["eval_loss"]) or not math.isfinite(trained.training_loss):
            raise ValueError("Unexpected optimizer count or nonfinite loss")
        model.save_pretrained(OUTPUT / "adapter")
        tokenizer.save_pretrained(OUTPUT / "adapter")
        trainer.save_state()
        record.update(status="complete", adapter_sha256=sha256_adapter(OUTPUT / "adapter"),
            training_metrics=trained.metrics, evaluation_metrics=validation,
            trainer_state=json.loads((OUTPUT / "trainer_state.json").read_text()),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        record["attempts"][-1].update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
        save()
    except BaseException as exc:
        record.update(status="interrupted_or_failed")
        record["attempts"][-1].update(status="failed", error_type=type(exc).__name__, error=str(exc))
        save()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--stage", choices=("base", "train", "sft"))
    parser.add_argument("--model-path", default="/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()
    plan = protocol(args.freeze)
    if args.freeze:
        print("Formal SFT plan frozen; no GPU used")
        return
    if not args.stage:
        parser.error("--stage required for GPU execution")
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.device_count() != 1:
        raise RuntimeError("Allocate exactly one bf16 GPU first")
    tokenizer, _ = checked_tokenizer(args.model_path, bundle())
    meta = json.loads((SMOKE / "preflight.json").read_text())
    if hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest() != meta["chat_template_sha256"]:
        raise ValueError("Tokenizer template changed")
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "train":
        train(args.model_path, tokenizer, plan)
    else:
        infer(args.model_path, plan, args.stage)


if __name__ == "__main__":
    main()
