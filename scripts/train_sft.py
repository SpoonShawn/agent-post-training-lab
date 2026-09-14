"""Single-GPU LoRA pilot with fail-closed provenance and CPU tiny-model test."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_training_bundle import verify
from scripts.run_baseline import sha256_model_path
from scripts.run_baseline import runtime_sha256
from scripts.summarize_baseline import load_records
from training.sft import load_trajectories, encode_examples, AssistantCollator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--tiny-cpu-smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--baseline-path", type=Path, default=ROOT / "results/baseline/pilot_v1_base_confirmation.jsonl")
    args = parser.parse_args(argv)
    manifest = verify()
    previous = None
    if args.resume_from_checkpoint:
        checkpoint = args.resume_from_checkpoint.resolve()
        if not checkpoint.is_relative_to(args.output_dir.resolve()) or not (checkpoint / "trainer_state.json").is_file():
            raise ValueError("Resume checkpoint must be inside this output directory")
        previous = json.loads((args.output_dir / "run.json").read_text())
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output is not empty; choose a new run directory. Do not overwrite experiments.")
    if args.max_steps == 0 or args.max_steps < -1 or args.max_length < 32:
        raise ValueError("Invalid step or context budget")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model
    if not args.tiny_cpu_smoke and not args.preflight_only and not torch.cuda.is_available():
        raise RuntimeError("Training requires an allocated CUDA GPU; do not run on login node")
    if not args.tiny_cpu_smoke and not Path(args.model_path).is_dir():
        raise ValueError("Production model must be an existing local directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"status": "started", "started_at": datetime.now(timezone.utc).isoformat(),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "manifest_sha256": hashlib.sha256((ROOT / "data/pilot_v1/manifest.json").read_bytes()).hexdigest(),
                "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "accelerate")},
                "seed": 20260914, "pilot_scope": manifest["scope"]}
    if previous:
        previous_args = {k: v for k, v in previous["arguments"].items() if k != "resume_from_checkpoint"}
        current_args = {k: v for k, v in metadata["arguments"].items() if k != "resume_from_checkpoint"}
        if (previous_args != current_args or previous["manifest_sha256"] != metadata["manifest_sha256"]
                or previous["versions"] != metadata["versions"]):
            raise ValueError("Resume configuration/data/runtime differs")
    attempt_name = datetime.now(timezone.utc).strftime("attempt_%Y%m%dT%H%M%S%f.json")
    def record():
        (args.output_dir / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        (args.output_dir / attempt_name).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    record()
    try:
        set_seed(20260914)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_rows = load_trajectories(ROOT / "data/pilot_v1/train_trajectories.jsonl", "train")
        val_rows = load_trajectories(ROOT / "data/pilot_v1/validation_trajectories.jsonl", "validation")
        train = encode_examples(train_rows, tokenizer, args.max_length)
        validation = encode_examples(val_rows, tokenizer, args.max_length)
        metadata["tokenization"] = {
            "train_examples": len(train), "validation_examples": len(validation),
            "max_tokens": max(len(e["input_ids"]) for e in train + validation),
            "train_supervised_tokens": sum(sum(t != -100 for t in e["labels"]) for e in train),
            "truncated_examples": 0,
        }
        metadata["chat_template_sha256"] = hashlib.sha256(
            tokenizer.get_chat_template().encode()).hexdigest()
        record()
        if args.preflight_only:
            metadata["status"] = "tokenization_preflight_complete_no_training"
            record()
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
            return
        if args.tiny_cpu_smoke:
            from transformers import Qwen3Config, Qwen3ForCausalLM
            torch.set_num_threads(2)
            model = Qwen3ForCausalLM(Qwen3Config(
                vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, max_position_embeddings=args.max_length, tie_word_embeddings=True))
            train, validation = train[:4], validation[:2]
            metadata["model_kind"] = "random_tiny_qwen3_not_4b_baseline"
        else:
            metadata["model_sha256"], metadata["resolved_model_path"] = sha256_model_path(args.model_path)
            if previous and previous.get("model_sha256") != metadata["model_sha256"]:
                raise ValueError("Resume model weights changed")
            baseline = load_records(args.baseline_path)
            expected_cases = {c["id"]: c for c in
                              (json.loads(line) for line in (ROOT / "data/pilot_v1/confirmation_cases.jsonl").read_text().splitlines())}
            if {r["case"]["id"] for r in baseline} != set(expected_cases):
                raise ValueError("Run the complete frozen confirmation baseline before training")
            for row in baseline:
                run = row.get("run_metadata", {})
                if (row["case"] != expected_cases[row["case"]["id"]]
                        or run.get("model_content_sha256") != metadata["model_sha256"]
                        or run.get("runtime_sha256") != runtime_sha256()
                        or run.get("adapter_sha256") is not None
                        or run.get("generation", {}).get("max_new_tokens") != 512
                        or run.get("generation", {}).get("max_steps_override") is not None
                        or any(run.get("runtime_versions", {}).get(p) != metadata["versions"][p]
                               for p in ("torch", "transformers"))):
                    raise ValueError("Baseline evidence does not match this untrained model/runtime")
            metadata["baseline_sha256"] = hashlib.sha256(args.baseline_path.read_bytes()).hexdigest()
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("Pilot requires bf16-capable GPU")
            metadata["gpu"] = torch.cuda.get_device_name(0)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, dtype=torch.bfloat16, trust_remote_code=False)
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules="all-linear", bias="none"))
        metadata["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if not metadata["trainable_parameters"]:
            raise ValueError("No trainable adapter parameters")
        options = TrainingArguments(
            output_dir=str(args.output_dir), num_train_epochs=1,
            max_steps=2 if args.tiny_cpu_smoke else args.max_steps,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=1 if args.tiny_cpu_smoke else 8,
            learning_rate=1e-4, warmup_steps=0 if args.tiny_cpu_smoke else 10,
            lr_scheduler_type="cosine", optim="adamw_torch",
            bf16=not args.tiny_cpu_smoke, use_cpu=args.tiny_cpu_smoke,
            gradient_checkpointing=not args.tiny_cpu_smoke,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="steps", eval_steps=1 if args.tiny_cpu_smoke else 50,
            save_strategy="steps", save_steps=1 if args.tiny_cpu_smoke else 50,
            save_total_limit=2, logging_steps=1 if args.tiny_cpu_smoke else 10,
            report_to="none", remove_unused_columns=False, label_names=["labels"],
            seed=20260914, data_seed=20260914)
        trainer = Trainer(model=model, args=options, train_dataset=train, eval_dataset=validation,
                          data_collator=AssistantCollator(tokenizer.pad_token_id))
        metadata["training_config"] = {
            "epochs": 1, "max_steps": options.max_steps, "learning_rate": 1e-4,
            "batch_size": 1, "gradient_accumulation": options.gradient_accumulation_steps,
            "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
            "target_modules": "all-linear", "checkpoint_selection": "last_not_best_on_confirmation",
            "assistant_only_loss": True, "packing": False}
        record()
        output = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
        evaluation = trainer.evaluate()
        if not math.isfinite(output.training_loss) or not math.isfinite(evaluation["eval_loss"]):
            raise RuntimeError("Nonfinite training/evaluation loss")
        adapter = args.output_dir / "adapter"
        model.save_pretrained(adapter)
        tokenizer.save_pretrained(adapter)
        trainer.save_state()
        metadata.update(status="tiny_cpu_smoke_complete" if args.tiny_cpu_smoke else "training_complete",
                        training_metrics=output.metrics, evaluation_metrics=evaluation,
                        adapter_path=str(adapter), completed_at=datetime.now(timezone.utc).isoformat())
        record()
    except Exception as error:
        metadata.update(status="failed", error_type=type(error).__name__, error=str(error))
        record()
        raise


if __name__ == "__main__":
    main()
