"""Train-only stochastic rollout feasibility probe, NOT DPO training or evaluation."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import time

from agent.transaction_model import TransactionAgent
from agent.transaction_runtime import ContextBudgetExceeded, run_episode
from agent.transaction_tasks import evaluate
from scripts.prepare_control_v1 import ROOT, digest
from scripts.prepare_transaction_v1 import DATA, bundle
from scripts.transaction_full_sft import OUTPUT as SFT, RESULTS as FULL, protocol
from scripts.transaction_gpu_smoke import checked_tokenizer, rows, sha256_adapter, write_once, validate_records
from training.transaction_data import canonical

PLAN = ROOT / "data/transaction_preference_probe_v1.json"
RESULTS = ROOT / "results/transaction_preference_probe_v1"
SAMPLING = dict(do_sample=True, temperature=.8, top_p=.95, top_k=50, num_beams=1, repetition_penalty=1.0)


def slots():
    groups = defaultdict(list)
    for case in rows(DATA / "train_cases.jsonl"):
        if case["split"] != "train":
            raise ValueError("Only training cases may be sampled")
        groups[case["group_id"]].append(case)
    selected = []
    for group in sorted(groups):
        ordered = sorted(groups[group], key=lambda c: hashlib.sha256(("probe-v1:" + c["id"]).encode()).hexdigest())
        for case in ordered[:2]:
            for candidate in range(4):
                seed = int(hashlib.sha256(f"probe-v1:{case['id']}:{candidate}".encode()).hexdigest()[:8], 16)
                selected.append(dict(case=case, candidate=candidate, seed=seed))
    return selected


def plan(freeze=False):
    protocol()
    run = json.loads((FULL / "training_run.json").read_text())
    if run["status"] != "complete":
        raise ValueError("Completed SFT required")
    schedule = slots()
    value = dict(scope="train_only_preference_yield_probe_not_DPO", sampling=SAMPLING,
        training_run_sha256=digest(FULL / "training_run.json"), adapter_sha256=run["adapter_sha256"],
        code_sha256=digest(ROOT / "scripts/transaction_preference_probe.py"),
        schedule_sha256=hashlib.sha256(canonical(schedule).encode()).hexdigest(),
        tasks=98, groups=49, candidates_per_task=4, episodes=392,
        generation_budget=bundle()["generation"],
        pair_rule="one pair per task: first strict-success versus first non-success; all ties abstain; never prefer shorter failure",
        boundaries="train cases only; no confirmation trajectories/answers/rewrites; no weights updated",
        caveat="SFT may be saturated on train, yielding zero pairs; preserve that outcome, do not fabricate negatives")
    if len(schedule) != 392:
        raise ValueError("Unexpected training group coverage")
    if freeze:
        write_once(PLAN, value)
    elif not PLAN.exists() or canonical(json.loads(PLAN.read_text())) != canonical(value):
        raise ValueError("Probe source/config drift")
    return value, schedule


class SamplingAgent(TransactionAgent):
    def generate(self, messages, tools):
        start = time.perf_counter()
        inputs = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt")
        prompt_tokens = inputs["input_ids"].shape[-1]
        if prompt_tokens + self.max_new_tokens > self.max_length:
            raise ContextBudgetExceeded("No evidence truncation allowed")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, **SAMPLING)
        tokens = output[0][prompt_tokens:]
        text = self.tokenizer.decode(tokens, skip_special_tokens=False)
        if self.tokenizer.eos_token:
            while text.rstrip().endswith(self.tokenizer.eos_token):
                text = text.rstrip()[:-len(self.tokenizer.eos_token)]
        self.usage.append(dict(prompt_tokens=prompt_tokens, generated_tokens=len(tokens), seconds=time.perf_counter()-start))
        return text


def pair_indices(records):
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record["case"]["id"]].append((index, record))
    pairs, abstain = [], []
    for case_id, candidates in grouped.items():
        good = [i for i, r in candidates if r["metrics"]["task_success"]]
        bad = [i for i, r in candidates if not r["metrics"]["task_success"]]
        if good and bad:
            pairs.append(dict(case_id=case_id, chosen_index=good[0], rejected_index=bad[0]))
        else:
            abstain.append(case_id)
    return dict(pairs=pairs, abstained_task_ids=abstain, tasks=len(grouped),
                note="candidate pairs only; full replay/masking/training review still required")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--model-path", default="/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()
    spec, schedule = plan(args.freeze)
    if args.freeze:
        print("Probe frozen: 98 training cases x 4 rollouts; no GPU used")
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    # Independent of SSH; reject another probe writing the same output concurrently.
    import fcntl
    with (RESULTS / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = dict(plan_sha256=digest(PLAN), adapter_sha256=spec["adapter_sha256"], sampling=SAMPLING)
        path = RESULTS / "rollouts.jsonl"
        saved = validate_records(path, metadata, [s["case"] for s in schedule])
        for r, slot in zip(saved, schedule):
            if r["candidate"] != slot["candidate"] or r["seed"] != slot["seed"]:
                raise ValueError("Candidate order or seed changed")
        if len(saved) < len(schedule):
            import torch
            from transformers import set_seed
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.device_count() != 1:
                raise RuntimeError("Allocate one bf16 GPU first")
            tokenizer, _ = checked_tokenizer(args.model_path, bundle())
            preflight = json.loads((ROOT / "results/transaction_v1_smoke/preflight.json").read_text())
            if hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest() != preflight["chat_template_sha256"]:
                raise ValueError("Tokenizer drift")
            if sha256_adapter(SFT / "adapter") != spec["adapter_sha256"]:
                raise ValueError("SFT adapter changed")
            agent = SamplingAgent(args.model_path, adapter_path=SFT / "adapter")
            with path.open("a") as handle:
                for index, slot in enumerate(schedule[len(saved):], len(saved)+1):
                    set_seed(slot["seed"])
                    agent.usage = []
                    case = slot["case"]
                    print(f"Preference probe {index}/392 {case['id']} candidate={slot['candidate']}", flush=True)
                    start = datetime.now(timezone.utc).isoformat()
                    result = run_episode(agent.generate, case["query"], case["environment"])
                    record = dict(**slot, result=result, metrics=evaluate(case["environment"], case["desired"],
                        result["events"], result["final_answer"]), metadata=metadata, generation_usage=agent.usage,
                        started_at=start, completed_at=datetime.now(timezone.utc).isoformat(), gpu=torch.cuda.get_device_name(0))
                    handle.write(canonical(record)+"\n")
                    handle.flush()
                    saved.append(record)
        write_once(RESULTS / "pair_yield.json", dict(**pair_indices(saved), rollouts_sha256=digest(path)))
        print("Preference probe complete; no DPO training performed. Upload results/transaction_preference_probe_v1/", flush=True)


if __name__ == "__main__":
    main()
