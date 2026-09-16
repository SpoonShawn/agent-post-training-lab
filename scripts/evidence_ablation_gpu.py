"""One fixed-budget paired LoRA experiment; preserve failures and old adapters."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math

from scripts.prepare_control_v1 import ROOT, digest
from scripts.prepare_evidence_ablation import DATA, bundle
from scripts.run_baseline import sha256_model_path
from scripts.evidence_pilot_gpu import sha256_adapter
from scripts.evidence_stress_v1 import resume_ids, run_rows
from training.evidence_pilot import canonical
from training.evidence_ablation import SEED
from training.sft import encode_examples, load_trajectories, AssistantCollator
from agent.baseline_runner import SYSTEM_PROMPT
from tools.tool_schema import TOOLS

RESULTS = ROOT / "results/evidence_ablation_v1"
CHECKPOINTS = ROOT / "checkpoints/evidence_ablation_v1"


def read_rows(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def eval_cases():
    return [r for s in ("validation", "confirmation") for r in read_rows(DATA/f"{s}_cases.jsonl")]


def write_once(path, value):
    content = json.dumps(value, ensure_ascii=False, indent=2)+"\n"
    if path.exists() and path.read_text() != content:
        raise ValueError(f"Refuse to replace metadata: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content)


def preflight(model_path):
    manifest = bundle()
    versions = {p:importlib.metadata.version(p) for p in manifest["versions"]}
    if versions != manifest["versions"]:
        raise ValueError("Package version drift; do not install upgrades for this run")
    model_hash, _ = sha256_model_path(model_path)
    if model_hash != manifest["model_sha256"]:
        raise ValueError("Base weights drift")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = {role:{s:encode_examples(load_trajectories(DATA/f"{role}_{s}.jsonl", s), tokenizer, 4096)
                     for s in ("train", "validation")} for role in ("fixed", "mixed")}
    stats = {}
    for role, splits in encoded.items():
        stats[role] = {s:dict(examples=len(rows), input_tokens=sum(len(r["input_ids"]) for r in rows),
                            supervised_tokens=sum(sum(t != -100 for t in r["labels"]) for r in rows),
                            max_tokens=max(len(r["input_ids"]) for r in rows)) for s,rows in splits.items()}
    # Exact answer-token sequences, not merely equal sample counts.
    for split in ("train", "validation"):
        a,b = (encoded[r][split] for r in ("fixed", "mixed"))
        if [[t for t in e["labels"] if t != -100] for e in a] != [[t for t in e["labels"] if t != -100] for e in b]:
            raise ValueError("Supervised targets differ across arms")
    lengths = []
    for c in eval_cases():
        prompt = tokenizer.apply_chat_template([dict(role="system", content=SYSTEM_PROMPT),
                    dict(role="user", content=c["query"])], tools=TOOLS, add_generation_prompt=True, tokenize=False)
        lengths.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
    if max(lengths)+512 > 4096:
        raise ValueError("Evaluation exceeds context; no silent truncation")
    metadata = dict(manifest_sha256=digest(DATA/"manifest.json"), model_sha256=model_hash,
                    versions=versions, training=manifest["training"], generation=manifest["generation"],
                    token_statistics=stats, max_eval_prompt_tokens=max(lengths),
                    chat_template_sha256=hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest())
    write_once(RESULTS/"preflight.json", metadata)
    return tokenizer, encoded, metadata


def run_metadata(meta, role, adapter_hash=None):
    result = dict(meta, role=role, adapter_sha256=adapter_hash)
    result["fingerprint"] = hashlib.sha256(canonical(result).encode()).hexdigest()
    return result


def completed_training(role, meta):
    folder = CHECKPOINTS/role
    record = json.loads((folder/"run.json").read_text())
    if (record.get("status") != "complete" or record["metadata"] != meta or record["role"] != role
            or record["adapter_sha256"] != sha256_adapter(folder/"adapter")
            or record != json.loads((RESULTS/f"{role}_training.json").read_text())):
        raise ValueError("Training provenance or adapter mismatch")
    return record


def train(model_path, tokenizer, encoded, meta, role):
    import torch
    from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, set_seed
    from peft import LoraConfig, get_peft_model
    folder = CHECKPOINTS/role
    if folder.exists() and any(folder.iterdir()):
        completed_training(role, meta)
        print(f"{role}: completed training preserved", flush=True)
        return
    cases = eval_cases()
    path = RESULTS/"base.jsonl"
    if len(resume_ids(path, run_metadata(meta, "base"), cases)) != len(cases):
        raise ValueError("Complete Base comparison required before training")
    smoke = role == "smoke"
    arm = "mixed" if smoke else role
    record = dict(status="started", role=role, metadata=meta, smoke=smoke,
                  baseline_sha256=digest(path), started_at=datetime.now(timezone.utc).isoformat(),
                  gpu=torch.cuda.get_device_name(0))
    folder.mkdir(parents=True, exist_ok=True)
    def save():
        value = json.dumps(record, ensure_ascii=False, indent=2)+"\n"
        (folder/"run.json").write_text(value)
        (RESULTS/f"{role}_training.json").write_text(value)
    save()
    try:
        set_seed(SEED)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=False)
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32,
                    lora_dropout=.05, target_modules="all-linear", bias="none"))
        record["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        options = TrainingArguments(output_dir=str(folder), num_train_epochs=1, max_steps=2 if smoke else -1,
            learning_rate=1e-4, per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=8, bf16=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant":False}, optim="adamw_torch",
            lr_scheduler_type="cosine", warmup_steps=2, eval_strategy="no", save_strategy="no",
            logging_steps=1, report_to="none", remove_unused_columns=False, label_names=["labels"],
            seed=SEED, data_seed=SEED)
        trainer = Trainer(model=model, args=options, train_dataset=encoded[arm]["train"],
                          eval_dataset=encoded[arm]["validation"][:4] if smoke else encoded[arm]["validation"],
                          data_collator=AssistantCollator(tokenizer.pad_token_id))
        outcome = trainer.train()
        evaluation = trainer.evaluate()
        if not math.isfinite(outcome.training_loss) or not math.isfinite(evaluation["eval_loss"]):
            raise ValueError("Non-finite loss")
        if trainer.state.global_step != (2 if smoke else 24):
            raise ValueError("Unexpected optimizer step budget")
        model.save_pretrained(folder/"adapter")
        tokenizer.save_pretrained(folder/"adapter")
        trainer.save_state()
        record.update(status="complete", training_metrics=outcome.metrics, evaluation_metrics=evaluation,
                      trainer_state=json.loads((folder/"trainer_state.json").read_text()),
                      adapter_sha256=sha256_adapter(folder/"adapter"), completed_at=datetime.now(timezone.utc).isoformat())
        save()
    except Exception as exc:
        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        save()
        raise


def evaluate(model_path, meta, role):
    adapter = None
    if role != "base":
        completed_training(role, meta)
        adapter = CHECKPOINTS/role/"adapter"
    metadata = run_metadata(meta, role, sha256_adapter(adapter) if adapter else None)
    cases = eval_cases()
    path = RESULTS/f"{role}.jsonl"
    seen = resume_ids(path, metadata, cases)
    pending = [c for c in cases if c["id"] not in seen]
    if not pending:
        print(f"{role}: all {len(cases)} results preserved; no model loaded", flush=True)
        return
    from agent.baseline_runner import BaselineAgent
    import torch
    agent = BaselineAgent(model_path, max_new_tokens=512, adapter_path=adapter)
    with path.open("a") as handle:
        # Existing common writer expects an arm field for progress only.
        for index, case in enumerate(pending, 1):
            started = datetime.now(timezone.utc).isoformat()
            answer = agent.generate([dict(role="system", content=SYSTEM_PROMPT), dict(role="user", content=case["query"])])
            from training.evidence_pilot import score
            handle.write(canonical(dict(case=case, answer=answer, metrics=score(answer, case["target"]),
                metadata=metadata, started_at=started, completed_at=datetime.now(timezone.utc).isoformat(),
                gpu=torch.cuda.get_device_name(0)))+"\n")
            handle.flush()
            print(f"{role} [{index}/{len(pending)}] {case['split']} {case['style']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "base", "smoke", "train_fixed", "train_mixed", "fixed", "mixed"), required=True)
    parser.add_argument("--model-path", default="/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("先申请支持bf16的GPU，再启动本脚本")
    print("Checking frozen data, packages, weights and actual token budgets...", flush=True)
    tokenizer, encoded, meta = preflight(args.model_path)
    if args.stage == "preflight":
        print(json.dumps(meta, ensure_ascii=False, indent=2))
    elif args.stage == "smoke" or args.stage.startswith("train_"):
        train(args.model_path, tokenizer, encoded, meta, args.stage.removeprefix("train_"))
    else:
        evaluate(args.model_path, meta, args.stage)


if __name__ == "__main__":
    main()
