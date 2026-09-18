"""Real small GRPO pilot. GPU only; probe and confirmation data are excluded."""
import argparse, copy, hashlib, json
from collections import defaultdict
from datetime import datetime, timezone
from scripts.prepare_control_v1 import ROOT, digest
from scripts.prepare_transaction_v1 import DATA, bundle
from scripts.transaction_gpu_smoke import rows, checked_tokenizer, sha256_adapter, write_once
from scripts.transaction_grpo_rollout_probe import strict_reward
from training.transaction_data import canonical
from training.transaction_grpo import advantages, clipped_objective, aggregate_update_stats
from agent.transaction_runtime import run_episode
from agent.transaction_model import TransactionAgent

BASE=ROOT/"data/transaction_grpo_pilot_v1.json"; PLAN=ROOT/"data/transaction_grpo_pilot_v3_protocol.json"
OUT=ROOT/"results/transaction_grpo_pilot_v1"; CKPT=ROOT/"checkpoints/transaction_grpo_pilot_v1"; SFT=ROOT/"checkpoints/transaction_v1_full_sft/adapter"

def protocol(freeze=False):
    base=json.loads(BASE.read_text()); train=list(rows(DATA/"train_cases.jsonl")); selected=[]
    quotas={"apply":12,"rollback":11,"permission":1}
    for category in ("apply","rollback","permission"):
        seen=set()
        for c in sorted((x for x in train if x["category"]==category),key=lambda x:hashlib.sha256(("grpo-pilot:"+x["id"]).encode()).hexdigest()):
            if c["group_id"] in seen: continue
            seen.add(c["group_id"]); selected.append(c)
            if len(seen)==quotas[category]: break
    if len(selected)!=24 or len({x["group_id"] for x in selected})!=24: raise ValueError("selection drift")
    sft_run=json.loads((ROOT/"results/transaction_v1_full_sft/training_run.json").read_text())
    value=dict(base,selected_case_ids=[x["id"] for x in selected],selected_group_ids=[x["group_id"] for x in selected],dataset_sha256=digest(DATA/"manifest.json"),source_sft_sha256=sft_run["adapter_sha256"],code_sha256=digest(ROOT/"scripts/transaction_grpo_pilot.py"))
    if freeze: write_once(PLAN,value)
    elif not PLAN.exists() or canonical(json.loads(PLAN.read_text()))!=canonical(value): raise ValueError("protocol drift")
    return value,selected

def sequence_logp(model,ids,prompt_len):
    import torch
    # Disable KV cache during teacher-forced scoring. Keeping generation caches
    # across 32x4 turns otherwise grows memory until the next full-vocab forward
    # cannot allocate its logits tensor.
    output=model(input_ids=ids,use_cache=False); logits=output.logits[0]; target=ids[0,prompt_len:]; dist=logits[prompt_len-1:-1].float().log_softmax(-1)
    lp=dist.gather(-1,target[:,None]).squeeze(-1); entropy=-(dist.exp()*dist).sum(-1)
    del output, logits, dist
    return lp,entropy

class RolloutAgent(TransactionAgent):
    def configure(self,temp):
        from transformers import GenerationConfig
        self.config=GenerationConfig(do_sample=True,temperature=temp,top_p=1.,top_k=0,num_beams=1,repetition_penalty=1.,max_new_tokens=512,bos_token_id=self.tokenizer.bos_token_id,eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id); self.records=[]
    def generate(self,messages,tools):
        import torch
        inp=self.tokenizer.apply_chat_template(messages,tools=tools,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt"); prompt=inp["input_ids"][0].tolist()
        with torch.no_grad(): out=self.model.generate(**{k:v.to(self.model.device) for k,v in inp.items()},generation_config=self.config,return_dict_in_generate=True,output_scores=True)
        ids=out.sequences[0,len(prompt):].tolist()
        if not ids or len(ids)!=len(out.scores): raise ValueError("generation alignment")
        old=[float(s[0].float().log_softmax(-1)[t]) for s,t in zip(out.scores,ids)]; text=self.tokenizer.decode(ids,skip_special_tokens=False)
        if self.tokenizer.eos_token:
            while text.rstrip().endswith(self.tokenizer.eos_token): text=text.rstrip()[:-len(self.tokenizer.eos_token)]
        self.records.append(dict(prompt_ids=prompt,generated_ids=ids,old_logprobs=old,text=text)); return text

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--freeze',action='store_true'); ap.add_argument('--model-path',default='/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507'); a=ap.parse_args(); spec,cases=protocol(a.freeze)
    if a.freeze: print('GRPO pilot protocol frozen'); return
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('one bf16 GPU required')
    OUT.mkdir(parents=True,exist_ok=True); CKPT.mkdir(parents=True,exist_ok=True); checked_tokenizer(a.model_path,bundle())
    policy=RolloutAgent(a.model_path,adapter_path=SFT)
    # PeftModel.from_pretrained is inference-frozen by default. The policy arm
    # must explicitly re-enable only LoRA weights; the reference remains frozen.
    for name,param in policy.model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
    trainable=[p for p in policy.model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("GRPO policy has no trainable LoRA parameters")
    policy.model.config.use_cache=False
    if hasattr(policy.model, "gradient_checkpointing_enable"):
        policy.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    reference=RolloutAgent(a.model_path,adapter_path=SFT); reference.model.eval()
    reference.model.config.use_cache=False
    for param in reference.model.parameters():
        param.requires_grad_(False)
    opt=torch.optim.AdamW(trainable,lr=spec['lr'])
    for temp in spec['temperatures']:
        policy.configure(temp)
        for update in range(1,spec['updates_per_temperature']+1):
            trajectories=[]
            for case in cases[(update-1)*8:update*8]:
                group=[]
                for sample in range(4):
                    torch.manual_seed(spec['seed']+update*1000+sample); torch.cuda.manual_seed_all(spec['seed']+update*1000+sample); policy.records=[]
                    result=run_episode(policy.generate,case['query'],case['environment'],max_turns=40,max_calls=36); reward,metrics=strict_reward(case,result); group.append(dict(case=case,result=result,reward=reward,metrics=metrics,behavior=copy.deepcopy(policy.records)))
                trajectories.extend(group)
            grouped=defaultdict(list)
            for x in trajectories: grouped[x['case']['id']].append(x)
            amap={cid:advantages([x['reward'] for x in xs])['advantages'] for cid,xs in grouped.items()}; opt.zero_grad(set_to_none=True); ratios=[]; kls=[]; lengths=[]; rewards=[]; advs=[]
            entropies=[]
            for cid,xs in grouped.items():
                for i,x in enumerate(xs):
                    rewards.append(x['reward']); advs.append(amap[cid][i])
                    for turn in x['behavior']:
                        ids=torch.tensor([turn['prompt_ids']+turn['generated_ids']],device=policy.model.device); new,entropy=sequence_logp(policy.model,ids,len(turn['prompt_ids']))
                        with torch.no_grad(): ref,_=sequence_logp(reference.model,ids,len(turn['prompt_ids']))
                        old=torch.tensor(turn['old_logprobs'],device=policy.model.device,dtype=new.dtype); term,ratio,kl=clipped_objective(new,old,ref,amap[cid][i],spec['clip_range'],spec['beta']); (-term.mean()/len(xs)).backward(); ratios.extend(ratio.tolist()); kls.extend(kl.tolist()); entropies.extend(entropy.tolist()); lengths.append(len(turn['generated_ids']))
                        del ids,new,ref,old,term,ratio,kl,entropy
                        torch.cuda.empty_cache()
            torch.nn.utils.clip_grad_norm_([p for p in policy.model.parameters() if p.requires_grad],1.); opt.step()
            stats=aggregate_update_stats(rewards,advs,ratios,kls,entropies,lengths,[1.0]*len(rewards),rewards); stats.update(temperature=temp,update=update,completed_at=datetime.now(timezone.utc).isoformat())
            folder=CKPT/f"checkpoint-t{str(temp).replace('.','')}-{update}"; folder.mkdir(parents=True,exist_ok=True)
            policy.model.save_pretrained(folder/"adapter"); torch.save({'optimizer':opt.state_dict(),'temperature':temp,'update':update},folder/"optimizer.pt")
            print(json.dumps(stats),flush=True)
    write_once(OUT/'protocol_snapshot.json',spec)

if __name__=='__main__': main()
