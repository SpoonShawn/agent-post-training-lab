"""Real two-step DPO engineering gate, with immutable cached SFT reference scores."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math

from scripts.prepare_control_v1 import ROOT, digest
from scripts.prepare_transaction_v1 import bundle
from scripts.prepare_transaction_dpo import DATA, freeze
from scripts.transaction_gpu_smoke import checked_tokenizer, write_once, sha256_adapter
from scripts.transaction_full_sft import OUTPUT as SFT, RESULTS as FULL
from training.transaction_data import canonical
from training.transaction_dpo import encode_pair, trajectory_logp, backward_pair, torch_self_test

PLAN = ROOT / "data/transaction_dpo_gate_v1.json"
RESULTS = ROOT / "results/transaction_dpo_gate_v1"
OUTPUT = ROOT / "checkpoints/transaction_dpo_gate_v1"
CONFIG = dict(beta=.1, learning_rate=5e-6, weight_decay=0., max_grad_norm=1., steps=2,
              seed=20260917, dropout=0., objective="sum_assistant_token_logp_DPO_no_length_normalization",
              reference="initial_SFT_cached_before_updates", scope="engineering_gate_never_resume_as_formal_DPO")


def protocol(create=False):
    records, _ = freeze()
    train = sorted((r for r in records if r["split"] == "preference_train"),
                   key=lambda r: hashlib.sha256(("dpo-gate:"+r["case_id"]).encode()).hexdigest())
    selected, groups = [], set()
    for r in train:
        if r["group_id"] not in groups:
            selected.append(r["case_id"])
            groups.add(r["group_id"])
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise ValueError("Four distinct train groups required")
    run = json.loads((FULL / "training_run.json").read_text())
    value = dict(config=CONFIG, pair_manifest_sha256=digest(DATA / "manifest.json"),
        source_training_sha256=digest(FULL / "training_run.json"), adapter_sha256=run["adapter_sha256"],
        probe_case_ids=selected, update_case_ids=selected[:2],
        code={p: digest(ROOT / p) for p in ("scripts/transaction_dpo_gate.py", "training/transaction_dpo.py")})
    if create:
        write_once(PLAN, value)
    elif not PLAN.exists() or canonical(json.loads(PLAN.read_text())) != canonical(value):
        raise ValueError("DPO gate drift")
    return records, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--model-path", default="/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()
    records, spec = protocol(args.freeze)
    if args.freeze:
        print("DPO two-step gate frozen; no GPU execution")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (RESULTS / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = RESULTS / "run.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if (previous["status"] == "complete" and previous["plan_sha256"] == digest(PLAN)
                    and previous["adapter_sha256"] == sha256_adapter(OUTPUT / "adapter")
                    and previous["reference_sha256"] == digest(RESULTS / "reference.json")):
                print("Completed DPO gate preserved")
                return
            raise ValueError("Preserve interrupted/different DPO gate; diagnose before restarting")
        if OUTPUT.exists() and any(OUTPUT.iterdir()):
            raise ValueError("Untracked gate artifacts; preserve")
        import torch
        from transformers import AutoModelForCausalLM, set_seed
        from peft import PeftModel
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.device_count() != 1:
            raise RuntimeError("Allocate one bf16 GPU first")
        tokenizer, _ = checked_tokenizer(args.model_path, bundle())
        old_meta = json.loads((ROOT / "results/transaction_v1_smoke/preflight.json").read_text())
        if hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest() != old_meta["chat_template_sha256"]:
            raise ValueError("Tokenizer drift")
        if sha256_adapter(SFT / "adapter") != spec["adapter_sha256"]:
            raise ValueError("Initial SFT changed")
        run = dict(status="started", plan_sha256=digest(PLAN), config=CONFIG,
                   started_at=datetime.now(timezone.utc).isoformat(), gpu=torch.cuda.get_device_name(0), steps=[])
        def save():
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(run, ensure_ascii=False, indent=2)+"\n")
            temporary.replace(path)
        save()
        try:
            set_seed(CONFIG["seed"])
            run["numeric_checks"] = torch_self_test()
            torch.cuda.reset_peak_memory_stats()
            base = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16, trust_remote_code=False)
            model = PeftModel.from_pretrained(base, SFT / "adapter", is_trainable=True).to("cuda")
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            model.enable_input_require_grads()
            model.train()
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.
            params = [p for p in model.parameters() if p.requires_grad]
            if not params or any(p.requires_grad and "lora_" not in name for name, p in model.named_parameters()):
                raise ValueError("Only LoRA parameters may be updated")
            run["trainable_parameters"] = sum(p.numel() for p in params)
            # Reference is the exact starting SFT, evaluated once BEFORE any optimizer exists.
            reference = {}
            for index, pair in enumerate(records, 1):
                encoded = encode_pair(pair, tokenizer)
                values = {side: trajectory_logp(model, encoded[side]) for side in ("chosen", "rejected")}
                if not all(math.isfinite(v) for v in values.values()):
                    raise ValueError("Nonfinite reference score")
                reference[pair["case_id"]] = dict(**values, split=pair["split"], tokens={
                    side: sum(sum(t != -100 for t in e["labels"]) for e in encoded[side]) for side in encoded})
                print(f"Fixed SFT reference {index}/{len(records)}", flush=True)
            write_once(RESULTS / "reference.json", dict(plan_sha256=digest(PLAN), scores=reference))
            reference_digest = digest(RESULTS / "reference.json")
            by_id = {r["case_id"]: r for r in records}
            optimizer = torch.optim.AdamW(params, lr=CONFIG["learning_rate"], weight_decay=0.)
            for index, cid in enumerate(spec["update_case_ids"], 1):
                pair = by_id[cid]
                if pair["split"] != "preference_train":
                    raise ValueError("Preference-dev must not update weights")
                encoded = encode_pair(pair, tokenizer)
                optimizer.zero_grad(set_to_none=True)
                scores = backward_pair(model, encoded, reference[cid], CONFIG["beta"])
                if index == 1 and (abs(scores["reference_adjusted_margin"]) > .01 or abs(scores["loss"]-math.log(2)) > .01):
                    raise ValueError("Initial policy/reference mismatch")
                norm = float(torch.nn.utils.clip_grad_norm_(params, CONFIG["max_grad_norm"], error_if_nonfinite=True))
                if not math.isfinite(scores["loss"]) or norm <= 0:
                    raise ValueError("No finite nonzero DPO gradient")
                optimizer.step()
                run["steps"].append(dict(step=index, case_id=cid, grad_norm_before_clip=norm, **scores))
                save()
                print(f"DPO gate update {index}/2 loss={scores['loss']:.6f} grad_norm={norm:.6f}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            run["probe_after"] = {}
            for cid in spec["probe_case_ids"]:
                encoded = encode_pair(by_id[cid], tokenizer)
                run["probe_after"][cid] = {side: trajectory_logp(model, encoded[side]) for side in encoded}
                if not all(math.isfinite(v) for v in run["probe_after"][cid].values()):
                    raise ValueError("Nonfinite post-update probe")
            if reference_digest != digest(RESULTS / "reference.json"):
                raise ValueError("Reference cache changed during training")
            model.save_pretrained(OUTPUT / "adapter")
            tokenizer.save_pretrained(OUTPUT / "adapter")
            if sha256_adapter(SFT / "adapter") != spec["adapter_sha256"]:
                raise ValueError("Source SFT artifact changed")
            run.update(status="complete", adapter_sha256=sha256_adapter(OUTPUT / "adapter"),
                reference_sha256=reference_digest, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), completed_at=datetime.now(timezone.utc).isoformat(),
                note="Real two-step DPO only; no task-success or generalization claim, never resume as formal DPO")
            save()
            print("DPO engineering gate complete. Upload run.json and reference.json.", flush=True)
        except BaseException as exc:
            run.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            save()
            raise


if __name__ == "__main__":
    main()
