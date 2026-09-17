"""Fixed 80-pair one-epoch DPO, from original SFT, then frozen 784-case evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import time

from scripts.analyze_transaction_dpo_gate import audit
from scripts.prepare_transaction_dpo import DATA as PAIRS, freeze
from scripts.prepare_transaction_v1 import DATA, bundle
from scripts.prepare_control_v1 import ROOT, digest
from scripts.transaction_full_sft import OUTPUT as SFT, SPLITS
from scripts.transaction_gpu_smoke import checked_tokenizer, write_once, sha256_adapter, rows, validate_records
from training.transaction_data import canonical
from training.transaction_dpo import encode_pair, trajectory_logp, backward_pair, dpo_terms, torch_self_test
from training.dpo_checkpoint import latest, seal
from agent.transaction_runtime import run_episode
from agent.transaction_tasks import evaluate

PLAN = ROOT / "data/transaction_full_dpo_v1.json"
RESULTS = ROOT / "results/transaction_full_dpo_v1"
OUTPUT = ROOT / "checkpoints/transaction_full_dpo_v1"
GATE = ROOT / "results/transaction_dpo_gate_v1"
CONFIG = dict(beta=.1, learning_rate=5e-6, weight_decay=0., max_grad_norm=1.,
    steps=80, epochs=1, pair_batch_size=1, seed=20260917, dropout=0., checkpoint_every=10,
    selection="fixed_last_no_confirmation_selection", initialization="original_full_SFT_not_gate",
    objective="sum_assistant_token_logp_DPO_no_length_normalization")


def order(pairs):
    return sorted((p for p in pairs if p["split"] == "preference_train"),
        key=lambda p: hashlib.sha256(("full-dpo-20260917:"+p["case_id"]).encode()).hexdigest())


def protocol(create=False):
    gate, reference = audit()
    pairs, _ = freeze()
    train = order(pairs)
    if len(train) != 80:
        raise ValueError("Expected exactly 80 training pairs")
    source = json.loads((ROOT / "results/transaction_v1_full_sft/training_run.json").read_text())
    value = dict(config=CONFIG, pairs_sha256=digest(PAIRS/"manifest.json"),
        reference_sha256=digest(GATE/"reference.json"), gate_run_sha256=digest(GATE/"run.json"),
        source_adapter_sha256=source["adapter_sha256"], generation=bundle()["generation"],
        training_order=[p["case_id"] for p in train],
        code={p: digest(ROOT/p) for p in ("scripts/transaction_full_dpo.py", "training/dpo_checkpoint.py",
            "scripts/analyze_transaction_dpo_gate.py", "scripts/superpod_transaction_full_dpo.sh")},
        scope="small_cross_policy_single_seed_DPO_not_large_scale_alignment",
        evaluation="same 256 dev + 128 ID + 400 OOD; repeated fixed benchmark, no tuning on confirmation")
    if create:
        write_once(PLAN, value)
    elif not PLAN.exists() or canonical(json.loads(PLAN.read_text())) != canonical(value):
        raise ValueError("Formal DPO protocol drift")
    return pairs, reference, value


def now():
    return datetime.now(timezone.utc).isoformat()


def train(model_path, tokenizer, pairs, reference, spec):
    path = RESULTS / "training_run.json"
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous and previous["plan_sha256"] != digest(PLAN):
        raise ValueError("Training record protocol changed")
    if previous and previous["status"] == "complete":
        if previous["adapter_sha256"] != sha256_adapter(OUTPUT/"adapter"):
            raise ValueError("Completed adapter changed")
        print("Completed full DPO preserved", flush=True)
        return
    resume = latest(OUTPUT, digest(PLAN))
    if previous and resume is None:
        raise ValueError("Interrupted before first checkpoint; preserve and diagnose")
    if not previous and (resume or (OUTPUT/"adapter").exists()):
        raise ValueError("Untracked DPO weights; preserve")
    run = dict(status="started", plan_sha256=digest(PLAN), config=CONFIG,
        reference_sha256=digest(GATE/"reference.json"), source_adapter_sha256=spec["source_adapter_sha256"],
        steps=resume[1]["steps"] if resume else [], attempts=(previous or {}).get("attempts", []) + [
            dict(started_at=now(), resume_checkpoint=str(resume[0]) if resume else None, updates=[])])
    def save():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(run, ensure_ascii=False, indent=2)+"\n")
        temporary.replace(path)
    save()
    try:
        import torch
        from transformers import AutoModelForCausalLM, set_seed
        from peft import PeftModel
        set_seed(CONFIG["seed"])
        run["numeric_checks"] = torch_self_test()
        run["gpu"] = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
        base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, trust_remote_code=False)
        source = resume[0]/"adapter" if resume else SFT/"adapter"
        model = PeftModel.from_pretrained(base, source, is_trainable=True).to("cuda")
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        model.enable_input_require_grads()
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
        params = [p for p in model.parameters() if p.requires_grad]
        if not params or any(p.requires_grad and "lora_" not in name for name, p in model.named_parameters()):
            raise ValueError("Only LoRA may train")
        run["trainable_parameters"] = sum(p.numel() for p in params)
        optimizer = torch.optim.AdamW(params, lr=CONFIG["learning_rate"], weight_decay=0.)
        if resume:
            saved = torch.load(resume[0]/"optimizer.pt", map_location="cpu", weights_only=True)
            optimizer.load_state_dict(saved["optimizer"])
            torch.set_rng_state(saved["cpu_rng"])
            torch.cuda.set_rng_state(saved["cuda_rng"])
        train_pairs = order(pairs)
        completed = resume[1]["step"] if resume else 0
        if [s["case_id"] for s in run["steps"]] != [p["case_id"] for p in train_pairs[:completed]]:
            raise ValueError("Checkpoint order differs")
        for index, pair in enumerate(train_pairs[completed:], completed+1):
            started = time.perf_counter()
            encoded = encode_pair(pair, tokenizer)
            ref = reference[pair["case_id"]]
            for side in encoded:
                count = sum(sum(t != -100 for t in e["labels"]) for e in encoded[side])
                if count != ref["tokens"][side]:
                    raise ValueError("Reference/target tokenization drift")
            optimizer.zero_grad(set_to_none=True)
            scores = backward_pair(model, encoded, ref, CONFIG["beta"])
            if index == 1 and abs(scores["reference_adjusted_margin"]) > .01:
                raise ValueError("Fresh policy differs from original SFT reference")
            norm = float(torch.nn.utils.clip_grad_norm_(params, CONFIG["max_grad_norm"], error_if_nonfinite=True))
            if not math.isfinite(scores["loss"]):
                raise ValueError("Nonfinite DPO loss")
            # Zero signal later is a valid measured saturation, not a fabricated update.
            optimizer.step()
            record = dict(step=index, case_id=pair["case_id"], grad_norm_before_clip=norm,
                          seconds=time.perf_counter()-started, **scores)
            run["steps"].append(record)
            run["attempts"][-1]["updates"].append(record)
            save()
            print(f"Formal DPO {index}/80 loss={scores['loss']:.6f} grad={norm:.6f}", flush=True)
            if index % 10 == 0:
                folder = OUTPUT / (f"checkpoint-{index:04d}-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
                model.save_pretrained(folder/"adapter")
                torch.save(dict(optimizer=optimizer.state_dict(), cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state()), folder/"optimizer.pt")
                seal(folder, dict(plan_sha256=digest(PLAN), step=index, steps=run["steps"]))
        optimizer.zero_grad(set_to_none=True)
        dev = {}
        for pair in pairs:
            if pair["split"] != "preference_dev":
                continue
            encoded = encode_pair(pair, tokenizer)
            values = {side: trajectory_logp(model, encoded[side]) for side in encoded}
            ref = reference[pair["case_id"]]
            loss, _, margin = dpo_terms(values["chosen"], values["rejected"], ref["chosen"], ref["rejected"])
            if not all(math.isfinite(v) for v in (*values.values(), loss, margin)):
                raise ValueError("Nonfinite preference-dev scores")
            dev[pair["case_id"]] = dict(**values, loss=loss, reference_adjusted_margin=margin)
        if digest(GATE/"reference.json") != spec["reference_sha256"] or sha256_adapter(SFT/"adapter") != spec["source_adapter_sha256"]:
            raise ValueError("Fixed reference or source SFT changed")
        model.save_pretrained(OUTPUT/"adapter")
        tokenizer.save_pretrained(OUTPUT/"adapter")
        run.update(status="complete", preference_dev=dev, adapter_sha256=sha256_adapter(OUTPUT/"adapter"),
            peak_cuda_allocated_bytes_this_attempt=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes_this_attempt=torch.cuda.max_memory_reserved(), completed_at=now())
        run["attempts"][-1].update(status="complete", completed_at=now())
        save()
    except BaseException as exc:
        run["status"] = "interrupted_or_failed"
        run["attempts"][-1].update(status="failed", error_type=type(exc).__name__, error=str(exc), completed_at=now())
        save()
        raise


def infer(model_path, spec):
    run = json.loads((RESULTS/"training_run.json").read_text())
    adapter = OUTPUT/"adapter"
    if run["status"] != "complete" or run["plan_sha256"] != digest(PLAN) or sha256_adapter(adapter) != run["adapter_sha256"]:
        raise ValueError("Formal DPO not complete or adapter changed")
    from agent.transaction_model import TransactionAgent
    import torch
    agent = None
    for split in SPLITS:
        cases = list(rows(DATA/f"{split}_cases.jsonl"))
        metadata = dict(plan_sha256=digest(PLAN), role="dpo", split=split,
            adapter_sha256=run["adapter_sha256"], generation=spec["generation"])
        path = RESULTS/f"dpo_{split}.jsonl"
        saved = validate_records(path, metadata, cases)
        if len(saved) == len(cases):
            continue
        if agent is None:
            agent = TransactionAgent(model_path, adapter_path=adapter)
        with path.open("a") as handle:
            for index, case in enumerate(cases[len(saved):], len(saved)+1):
                print(f"dpo/{split} {index}/{len(cases)} starting", flush=True)
                started = now()
                agent.usage = []
                result = run_episode(agent.generate, case["query"], case["environment"])
                metrics = evaluate(case["environment"], case["desired"], result["events"], result["final_answer"])
                row = dict(case=case, result=result, metrics=metrics, metadata=metadata,
                    generation_usage=agent.usage, started_at=started, completed_at=now(), gpu=torch.cuda.get_device_name(0))
                handle.write(canonical(row)+"\n")
                handle.flush()
                print(f"calls={metrics['tool_calls']} reason={result['terminated_reason']} (scores sealed until paired audit)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--stage", choices=("train", "infer"))
    parser.add_argument("--model-path", default="/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()
    pairs, reference, spec = protocol(args.freeze)
    if args.freeze:
        print("Formal 80-step DPO and 784-case evaluation frozen")
        return
    if not args.stage:
        parser.error("--stage required")
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (RESULTS/".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.device_count() != 1:
            raise RuntimeError("Allocate one bf16 GPU")
        tokenizer, _ = checked_tokenizer(args.model_path, bundle())
        meta = json.loads((ROOT/"results/transaction_v1_smoke/preflight.json").read_text())
        if hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest() != meta["chat_template_sha256"]:
            raise ValueError("Tokenizer changed")
        if sha256_adapter(SFT/"adapter") != spec["source_adapter_sha256"]:
            raise ValueError("Original SFT changed")
        if args.stage == "train":
            train(args.model_path, tokenizer, pairs, reference, spec)
        else:
            infer(args.model_path, spec)


if __name__ == "__main__":
    main()
