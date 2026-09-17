"""32 train-only SFT rollouts with actual behavior probabilities; no RL updates yet."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import time

from agent.transaction_model import TransactionAgent
from agent.transaction_runtime import ContextBudgetExceeded, run_episode
from training.transaction_grpo import strict_reward, advantages
from training.transaction_data import canonical
from scripts.transaction_preference_probe import slots
from scripts.transaction_full_sft import OUTPUT as SFT, RESULTS as SFT_RESULTS, protocol as sft_protocol
from scripts.prepare_transaction_v1 import bundle
from scripts.prepare_control_v1 import ROOT, digest
from scripts.transaction_gpu_smoke import checked_tokenizer, sha256_adapter, validate_records, write_once

PLAN = ROOT/"data/transaction_grpo_rollout_probe_v1.json"
RESULTS = ROOT/"results/transaction_grpo_rollout_probe_v1"
SAMPLING = dict(do_sample=True, temperature=1.5, top_p=1., top_k=0, num_beams=1,
    repetition_penalty=1., max_new_tokens=512)


def schedule():
    candidates = [s["case"] for s in slots() if s["candidate"]==0]
    selected = []
    for category, count in (("apply",3),("rollback",4),("permission",1)):
        seen = set()
        ranked = sorted((c for c in candidates if c["category"]==category),
            key=lambda c:hashlib.sha256(("grpo-probe-v1:"+c["id"]).encode()).hexdigest())
        for case in ranked:
            if case["group_id"] in seen:
                continue
            if case["split"] != "train":
                raise ValueError("GRPO probe must only use original train cases")
            seen.add(case["group_id"])
            selected.append(case)
            if len(seen)==count:
                break
    if len(selected)!=8 or len({c["group_id"] for c in selected})!=8:
        raise ValueError("Eight distinct train groups required")
    return [dict(case=c, candidate=i, seed=int(hashlib.sha256(f"grpo-probe-v1:{c['id']}:{i}".encode()).hexdigest()[:8],16))
            for c in selected for i in range(4)]


def protocol(create=False):
    sft_protocol()
    run = json.loads((SFT_RESULTS/"training_run.json").read_text())
    tasks = schedule()
    spec = dict(scope="GRPO_behavior_and_reward_variance_probe_no_updates", adapter_sha256=run["adapter_sha256"],
        source_sft_run_sha256=digest(SFT_RESULTS/"training_run.json"), sampling=SAMPLING,
        groups=8, group_size=4, episodes=32, max_calls=36, max_turns=40, max_length=8192,
        schedule_sha256=hashlib.sha256(canonical(tasks).encode()).hexdigest(),
        reward="replayed_strict_task_success_binary", zero_variance="zero_advantages_no_fabricated_signal",
        policy="softmax(raw_logits/1.5), full support; future ratios must use this same policy, not temperature-1 logits",
        reason="prior train sampling at temperature0.8/top_p0.95 gave 392 identical successes; test broader exploration without changing tasks or reward",
        code={p:digest(ROOT/p) for p in ("training/transaction_grpo.py","scripts/transaction_grpo_rollout_probe.py")})
    if create:
        write_once(PLAN,spec)
    elif not PLAN.exists() or canonical(json.loads(PLAN.read_text()))!=canonical(spec):
        raise ValueError("GRPO rollout protocol drift")
    return spec,tasks


class BehaviorAgent(TransactionAgent):
    def configure(self):
        from transformers import GenerationConfig
        # Fresh configuration avoids inheriting top-p, penalties or suppression from model defaults.
        self.behavior_config = GenerationConfig(**SAMPLING,bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
        self.behavior=[]

    def generate(self,messages,tools):
        started=time.perf_counter()
        inputs=self.tokenizer.apply_chat_template(messages,tools=tools,add_generation_prompt=True,
            tokenize=True,return_dict=True,return_tensors="pt")
        prompt=inputs["input_ids"][0].tolist()
        if len(prompt)+512>8192:
            raise ContextBudgetExceeded("No evidence truncation allowed")
        inputs={k:v.to(self.model.device) for k,v in inputs.items()}
        with self.torch.no_grad():
            output=self.model.generate(**inputs,generation_config=self.behavior_config,
                return_dict_in_generate=True,output_scores=True)
        generated=output.sequences[0,len(prompt):].tolist()
        if not generated or len(generated)!=len(output.scores):
            raise ValueError("Behavior token/score alignment mismatch")
        logps,entropies=[],[]
        for token,score in zip(generated,output.scores):
            lp=self.torch.log_softmax(score[0].float(),dim=-1)
            logps.append(float(lp[token]))
            entropy=-self.torch.where(self.torch.isfinite(lp),lp.exp()*lp,self.torch.zeros_like(lp)).sum()
            entropies.append(float(entropy))
        if any(not math.isfinite(v) for v in logps+entropies):
            raise ValueError("Nonfinite sampled behavior probability")
        text=self.tokenizer.decode(generated,skip_special_tokens=False)
        if self.tokenizer.eos_token:
            while text.rstrip().endswith(self.tokenizer.eos_token):
                text=text.rstrip()[:-len(self.tokenizer.eos_token)]
        self.behavior.append(dict(prompt_ids=prompt,generated_ids=generated,old_logprobs=logps,
            token_entropies=entropies,text=text,terminated_by_eos=generated[-1]==self.tokenizer.eos_token_id))
        self.usage.append(dict(prompt_tokens=len(prompt),generated_tokens=len(generated),seconds=time.perf_counter()-started))
        return text


def validate_behavior(record):
    actions=record["behavior"]
    if len(actions)!=len(record["result"]["turns"]):
        raise ValueError("Missing behavior turns")
    for action,turn in zip(actions,record["result"]["turns"]):
        n=len(action["generated_ids"])
        if not 0<n<=512 or n!=len(action["old_logprobs"]) or n!=len(action["token_entropies"]):
            raise ValueError("Behavior token alignment changed")
        if action["text"]!=turn["model_output"] or len(action["prompt_ids"])+512>8192:
            raise ValueError("Behavior text/context mismatch")
        if any(not math.isfinite(p) or p>1e-6 for p in action["old_logprobs"]):
            raise ValueError("Invalid behavior log probability")
        if any(not math.isfinite(h) or h<0 for h in action["token_entropies"]):
            raise ValueError("Invalid entropy")
        if any(type(t) is not int or t<0 for t in action["prompt_ids"]+action["generated_ids"]):
            raise ValueError("Invalid token IDs")


def summarize(records):
    groups=defaultdict(list)
    for record in records:
        reward,_=strict_reward(record["case"],record["result"])
        if type(record["reward"]) is not float or record["reward"]!=reward:
            raise ValueError("Reward differs from executed evidence")
        groups[record["case"]["id"]].append(reward)
    if any(len(r)!=4 for r in groups.values()):
        raise ValueError("Incomplete rollout group")
    values={cid:advantages(r) for cid,r in groups.items()}
    return dict(scope="no_optimizer_steps_no_GRPO_training_claim", groups=values,
        mixed_reward_groups=sum(not v["zero_variance"] for v in values.values()),
        zero_variance_groups=sum(v["zero_variance"] for v in values.values()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--freeze",action="store_true")
    parser.add_argument("--model-path",default="/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507")
    args=parser.parse_args()
    spec,tasks=protocol(args.freeze)
    if args.freeze:
        print("GRPO rollout probe frozen: 8 train tasks x4, no optimizer")
        return
    RESULTS.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (RESULTS/".run.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        meta=dict(plan_sha256=digest(PLAN),adapter_sha256=spec["adapter_sha256"],sampling=SAMPLING,policy_revision=0)
        path=RESULTS/"rollouts.jsonl"
        saved=validate_records(path,meta,[t["case"] for t in tasks])
        for record,task in zip(saved,tasks):
            if canonical({k:record[k] for k in ("candidate","seed")})!=canonical({k:task[k] for k in ("candidate","seed")}):
                raise ValueError("Rollout seed/order drift")
            validate_behavior(record)
        if len(saved)<len(tasks):
            import torch
            from transformers import set_seed
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or torch.cuda.device_count()!=1:
                raise RuntimeError("Allocate one bf16 GPU")
            tokenizer,_=checked_tokenizer(args.model_path,bundle())
            preflight=json.loads((ROOT/"results/transaction_v1_smoke/preflight.json").read_text())
            if hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()!=preflight["chat_template_sha256"]:
                raise ValueError("Tokenizer changed")
            if sha256_adapter(SFT/"adapter")!=spec["adapter_sha256"]:
                raise ValueError("Original SFT changed; do not use DPO adapter")
            agent=BehaviorAgent(args.model_path,adapter_path=SFT/"adapter")
            agent.configure()
            config=agent.behavior_config.to_dict()
            if any(canonical(r["generation_config"])!=canonical(config) for r in saved):
                raise ValueError("Generation defaults changed on resume")
            with path.open("a") as handle:
                for index,task in enumerate(tasks[len(saved):],len(saved)+1):
                    print(f"GRPO rollout probe {index}/32 starting",flush=True)
                    set_seed(task["seed"])
                    agent.usage=[]
                    agent.behavior=[]
                    start=datetime.now(timezone.utc).isoformat()
                    case=task["case"]
                    result=run_episode(agent.generate,case["query"],case["environment"])
                    reward,metrics=strict_reward(case,result)
                    record=dict(**task,result=result,metrics=metrics,reward=reward,behavior=agent.behavior,
                        generation_usage=agent.usage,generation_config=config,metadata=meta,
                        started_at=start,completed_at=datetime.now(timezone.utc).isoformat(),gpu=torch.cuda.get_device_name(0))
                    validate_behavior(record)
                    handle.write(canonical(record)+"\n")
                    handle.flush()
                    saved.append(record)
                    print(f"reward={reward} calls={metrics['tool_calls']} reason={result['terminated_reason']}",flush=True)
        summary=summarize(saved)
        write_once(RESULTS/"summary.json",dict(**summary,rollouts_sha256=digest(path)))
        print("GRPO rollout probe complete. No policy updates performed.",flush=True)


if __name__=="__main__":
    main()
