"""Real Base rollouts on the same 98 TRAIN tasks; offline pair feasibility only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json

from scripts.analyze_transaction_preference_probe import audit
from scripts.transaction_preference_probe import slots
from scripts.transaction_gpu_smoke import checked_tokenizer, validate_records, write_once
from scripts.prepare_control_v1 import ROOT, digest
from scripts.prepare_transaction_v1 import bundle
from training.transaction_data import canonical
from agent.transaction_model import TransactionAgent
from agent.transaction_runtime import run_episode
from agent.transaction_tasks import evaluate

PLAN = ROOT / "data/transaction_base_preferences_v1.json"
RESULTS = ROOT / "results/transaction_base_preferences_v1"


def pairs(base, sft):
    """Fixed SFT candidate 0 only; strict success beats failure, all ties abstain."""
    first = {r["case"]["id"]: (i, r) for i, r in enumerate(sft) if r["candidate"] == 0}
    selected, abstain = [], []
    for index, r in enumerate(base):
        source_index, other = first[r["case"]["id"]]
        if canonical(r["case"]) != canonical(other["case"]):
            raise ValueError("Different environments cannot form a pair")
        b, s = r["metrics"]["task_success"], other["metrics"]["task_success"]
        if b == s:
            abstain.append(r["case"]["id"])
        else:
            selected.append(dict(case_id=r["case"]["id"],
                chosen=dict(source="base" if b else "sft_probe", index=index if b else source_index),
                rejected=dict(source="sft_probe" if b else "base", index=source_index if b else index)))
    return dict(pairs=selected, abstained_task_ids=abstain, tasks=len(base),
        scope="cross_policy_offline_candidate_pairs_not_DPO_training",
        caveat="Base negatives may already have negligible SFT likelihood; measure gradient signal before full DPO")


def protocol(freeze=False):
    summary, sft = audit()
    cases = [s["case"] for s in slots() if s["candidate"] == 0]
    value = dict(scope="train_only_cross_policy_preference_supply", cases=98,
        cases_sha256=hashlib.sha256(canonical(cases).encode()).hexdigest(),
        source_probe=summary["sources"], generation=bundle()["generation"],
        model_sha256=bundle()["model_sha256"],
        code={name: digest(ROOT / name) for name in (
            "scripts/transaction_base_preferences.py", "scripts/analyze_transaction_preference_probe.py")},
        pair_rule="same case; Base greedy versus previously sampled SFT candidate 0; strict success beats failure; ties abstain",
        use="engineering feasibility; no claim of hard negatives, no confirmation-derived training, no optimizer")
    if freeze:
        write_once(PLAN, value)
    elif not PLAN.exists() or canonical(json.loads(PLAN.read_text())) != canonical(value):
        raise ValueError("Cross-policy protocol drift")
    return value, cases, sft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--model-path", default="/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()
    spec, cases, sft = protocol(args.freeze)
    if args.freeze:
        print("98 train-only Base episodes frozen; no GPU used")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (RESULTS / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = dict(plan_sha256=digest(PLAN), model_sha256=spec["model_sha256"], adapter_sha256=None)
        path = RESULTS / "base.jsonl"
        records = validate_records(path, metadata, cases)
        if len(records) < len(cases):
            import torch
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.device_count() != 1:
                raise RuntimeError("Allocate one bf16 GPU")
            tokenizer, _ = checked_tokenizer(args.model_path, bundle())
            preflight = json.loads((ROOT / "results/transaction_v1_smoke/preflight.json").read_text())
            if hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest() != preflight["chat_template_sha256"]:
                raise ValueError("Tokenizer drift")
            agent = TransactionAgent(args.model_path)
            with path.open("a") as handle:
                for index, case in enumerate(cases[len(records):], len(records)+1):
                    print(f"Train Base candidate {index}/98 starting", flush=True)
                    start = datetime.now(timezone.utc).isoformat()
                    agent.usage = []
                    result = run_episode(agent.generate, case["query"], case["environment"])
                    record = dict(case=case, result=result, metrics=evaluate(case["environment"], case["desired"],
                        result["events"], result["final_answer"]), metadata=metadata, generation_usage=agent.usage,
                        started_at=start, completed_at=datetime.now(timezone.utc).isoformat(), gpu=torch.cuda.get_device_name(0))
                    handle.write(canonical(record)+"\n")
                    handle.flush()
                    records.append(record)
        write_once(RESULTS / "pair_yield.json", dict(**pairs(records, sft), base_sha256=digest(path),
            sft_probe_sha256=spec["source_probe"]["rollouts"]))
        print("Base preference collection complete; no DPO training yet", flush=True)


if __name__ == "__main__":
    main()
