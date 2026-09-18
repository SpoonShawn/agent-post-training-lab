"""Corrected GRPO pilot: memory gate -> independent arms -> sealed checkpoints."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import time

from agent.transaction_runtime import ContextBudgetExceeded, run_episode
from training.transaction_data import canonical
from training.transaction_grpo import advantages, strict_reward
from training.grpo_update_v2 import token_chunks, scores, objective, mean_std, tool_stats
from scripts.prepare_control_v1 import ROOT, digest
from scripts.prepare_transaction_v1 import DATA, bundle
from scripts.transaction_gpu_smoke import rows, checked_tokenizer, write_once, sha256_adapter, validate_records
from training.dpo_checkpoint import seal, latest

PLAN=ROOT/'data/transaction_grpo_pilot_corrected_v2.json'
OUT=ROOT/'results/transaction_grpo_pilot_corrected_v2'
CHECKPOINTS=ROOT/'checkpoints/transaction_grpo_pilot_corrected_v2'
SFT=ROOT/'checkpoints/transaction_v1_full_sft/adapter'
CONFIG=dict(temperatures=[1.2,1.5,1.8], groups_per_update=12, group_size=4,
    updates=2, episodes_per_arm=96, total_episodes=288, lr=1e-5, beta=.01, clip=.2,
    seed=20260918, max_length=8192, max_new_tokens=512, max_calls=36, max_turns=40,
    logits_chunk=32, dropout=0, attention='sdpa', gradient_checkpointing=True,
    reward='unchanged_replayed_strict_task_success_binary',
    selection='fixed_last_per_arm_no_confirmation_selection')


def selected_cases():
    pool=list(rows(DATA/'train_cases.jsonl')); selected=[]
    for category,quota in (('apply',12),('rollback',11),('permission',1)):
        seen=set()
        for c in sorted((c for c in pool if c['category']==category),
                        key=lambda c:hashlib.sha256(('grpo-pilot:'+c['id']).encode()).hexdigest()):
            if c['group_id'] in seen: continue
            seen.add(c['group_id']); selected.append(c)
            if len(seen)==quota: break
    if len(selected)!=24 or len({c['group_id'] for c in selected})!=24:
        raise ValueError('24 distinct train groups required')
    eval_groups={c['group_id'] for s in ('confirmation_id','confirmation_ood')
                 for c in rows(DATA/f'{s}_cases.jsonl')}
    if eval_groups & {c['group_id'] for c in selected} or any(c['split']!='train' for c in selected):
        raise ValueError('Train/evaluation leakage')
    # Distribute contract categories across the two batches, including permission in first.
    permission=[c for c in selected if c['category']=='permission']
    other=[c for c in selected if c['category']!='permission']
    return permission+other[::2]+other[1::2]


def protocol(freeze=False):
    cases=selected_cases(); manifest=bundle()
    sft_run=json.loads((ROOT/'results/transaction_v1_full_sft/training_run.json').read_text())
    code=('scripts/transaction_grpo_pilot_v2.py','training/grpo_update_v2.py',
          'training/transaction_grpo.py','training/dpo_checkpoint.py')
    spec=dict(config=CONFIG, case_ids=[c['id'] for c in cases],
        groups=[c['group_id'] for c in cases], source_adapter_sha256=sft_run['adapter_sha256'],
        source_run_sha256=digest(ROOT/'results/transaction_v1_full_sft/training_run.json'),
        code={p:digest(ROOT/p) for p in code}, reward_code_sha256=digest(ROOT/'agent/transaction_tasks.py'),
        dataset_sha256=digest(DATA/'manifest.json'), generation=manifest['generation'],
        eval_splits=['confirmation_id','confirmation_ood'], probe_excluded=True,
        correction='v1 invalid pilot preserved; independent SFT arms, consistent tempered distributions, actual tool statistics',
        loss='mean_over_episodes(sum_generated_token_objective / total_episode_generated_tokens)')
    if freeze: write_once(PLAN,spec)
    elif not PLAN.exists() or canonical(json.loads(PLAN.read_text()))!=canonical(spec):
        raise ValueError('Corrected protocol drift; do not refreeze after any completed update')
    return spec,cases


def load_policy(model_path, adapter):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    base=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,
        device_map={'':'cuda:0'},attn_implementation='sdpa',trust_remote_code=False)
    model=PeftModel.from_pretrained(base,adapter,is_trainable=True)
    model.load_adapter(SFT,adapter_name='reference',is_trainable=False)
    model.set_adapter('default')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    model.config.use_cache=False
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout): module.p=0.
        if hasattr(module,'attention_dropout'): module.attention_dropout=0.
    model.train()
    params=[p for name,p in model.named_parameters() if p.requires_grad]
    if not params or any('lora_' not in name for name,p in model.named_parameters() if p.requires_grad):
        raise ValueError('Only policy LoRA parameters must train')
    return model,params


class Sampler:
    def __init__(self,model,tokenizer,temperature):
        from transformers import GenerationConfig
        self.model,self.tokenizer=model,tokenizer
        self.config=GenerationConfig(do_sample=True,temperature=temperature,top_p=1.,top_k=0,
            repetition_penalty=1.,num_beams=1,max_new_tokens=CONFIG['max_new_tokens'],
            use_cache=True,return_dict_in_generate=True,output_scores=True,
            bos_token_id=tokenizer.bos_token_id,eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        self.behavior=[]

    def generate(self,messages,tools):
        import torch
        inputs=self.tokenizer.apply_chat_template(messages,tools=tools,add_generation_prompt=True,
            tokenize=True,return_dict=True,return_tensors='pt')
        prompt=inputs['input_ids'][0].tolist()
        if len(prompt)+CONFIG['max_new_tokens']>CONFIG['max_length']:
            raise ContextBudgetExceeded('Public evidence context exceeds budget')
        self.model.eval()
        started=time.perf_counter()
        with torch.no_grad():
            out=self.model.generate(**{k:v.to(self.model.device) for k,v in inputs.items()},generation_config=self.config)
        generated=out.sequences[0,len(prompt):].tolist()
        if not generated or len(generated)!=len(out.scores): raise ValueError('Generation/token misalignment')
        logps=[float(score[0].float().log_softmax(-1)[token]) for score,token in zip(out.scores,generated)]
        text=self.tokenizer.decode(generated,skip_special_tokens=False)
        if self.tokenizer.eos_token:
            while text.rstrip().endswith(self.tokenizer.eos_token): text=text.rstrip()[:-len(self.tokenizer.eos_token)]
        self.behavior.append(dict(prompt_ids=prompt,generated_ids=generated,behavior_logprobs=logps,
            text=text,seconds=time.perf_counter()-started,eos=generated[-1]==self.tokenizer.eos_token_id))
        return text


def update(model,params,optimizer,records,temperature):
    import torch
    groups={cid:[r for r in records if r['case']['id']==cid] for cid in dict.fromkeys(r['case']['id'] for r in records)}
    distributions={cid:advantages([r['reward'] for r in group]) for cid,group in groups.items()}
    values=[]; ratios=[]; kls=[]; entropies=[]; deviations=[]
    optimizer.zero_grad(set_to_none=True)
    before=[p.detach().clone() for p in params]
    for cid,group in groups.items():
        if len(group)!=CONFIG['group_size']: raise ValueError('Incomplete group')
        for index,record in enumerate(group):
            tokens=sum(len(turn['generated_ids']) for turn in record['behavior'])
            if not tokens: continue
            for turn in record['behavior']:
                for start,stop in token_chunks(turn['prompt_ids'],turn['generated_ids'],CONFIG['logits_chunk']):
                    # Both teacher-forced old and frozen reference use the sampling temperature.
                    # No optimizer step is allowed until every episode has been scored/backpropagated.
                    model.set_adapter('default'); model.eval()
                    with torch.no_grad(): old,_=scores(model,turn['prompt_ids'],turn['generated_ids'],start,stop,temperature)
                    behavior=torch.tensor(turn['behavior_logprobs'][start:stop],device=old.device)
                    difference=float((old-behavior).abs().max())
                    deviations.append(difference)
                    if difference>.25: raise ValueError(f'Behavior/teacher probability mismatch: {difference}')
                    model.set_adapter('reference'); model.eval()
                    with torch.no_grad(): reference,_=scores(model,turn['prompt_ids'],turn['generated_ids'],start,stop,temperature)
                    model.set_adapter('default'); model.train()
                    new,entropy=scores(model,turn['prompt_ids'],turn['generated_ids'],start,stop,temperature)
                    term,ratio,kl=objective(new,old,reference,distributions[cid]['advantages'][index],CONFIG['clip'],CONFIG['beta'])
                    if not torch.isfinite(term).all(): raise ValueError('Nonfinite GRPO loss')
                    (-term.sum()/(len(records)*tokens)).backward()
                    values.extend(term.detach().tolist()); ratios.extend(ratio.detach().tolist())
                    kls.extend(kl.detach().tolist()); entropies.extend(entropy.detach().tolist())
                    del new,entropy,term,ratio,kl,old,reference,behavior
    norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
    optimizer.step()
    delta=math.sqrt(sum(float((p.detach().float()-b.float()).square().sum()) for p,b in zip(params,before)))
    return dict(reward=mean_std([r['reward'] for r in records]),
        advantage=mean_std([a for d in distributions.values() for a in d['advantages']]),
        groups=distributions, mixed_groups=sum(not d['zero_variance'] for d in distributions.values()),
        objective=mean_std(values),ratio=mean_std(ratios),policy_kl=mean_std(kls),entropy=mean_std(entropies),
        max_behavior_logprob_difference=max(deviations,default=None),grad_norm_before_clip=norm,
        policy_parameter_delta_l2=delta,policy_changed=delta>0,tool_validity=tool_stats(records),
        episode_tokens=mean_std([sum(len(t['generated_ids']) for t in r['behavior']) for r in records]),
        distinct_results_by_group={cid:len({canonical(r['result']) for r in group}) for cid,group in groups.items()},
        max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),max_cuda_reserved_bytes=torch.cuda.max_memory_reserved())


def gate(model,params,tokenizer):
    import torch
    model.set_adapter('default'); model.train()
    token=tokenizer.encode('test',add_special_tokens=False)[0]
    torch.cuda.reset_peak_memory_stats()
    # 8192 total tokens, 32 action positions: maximal permitted context stress.
    prompt=[token]*(CONFIG['max_length']-32); generated=[token]*32
    lp,entropy=scores(model,prompt,generated,0,32,1.5)
    (-lp.mean()).backward()
    finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
    grad=math.sqrt(sum(float(p.grad.float().square().sum()) for p in params if p.grad is not None))
    if not finite or grad<=0: raise ValueError('Memory gate has no finite policy gradient')
    model.zero_grad(set_to_none=True)
    del lp,entropy
    # Do not update any parameter in the gate.
    return dict(scope='synthetic_memory_and_gradient_gate_no_update',context_tokens=8192,
        lm_head_positions=32,gradient_norm=grad,max_allocated_bytes=torch.cuda.max_memory_allocated(),
        max_reserved_bytes=torch.cuda.max_memory_reserved(),gpu=torch.cuda.get_device_name(0))


def train_arm(model_path,tokenizer,temp,cases):
    import torch
    from transformers import set_seed
    name=f't{int(round(temp*10))}'
    results=OUT/name; checkpoints=CHECKPOINTS/name
    results.mkdir(parents=True,exist_ok=True); checkpoints.mkdir(parents=True,exist_ok=True)
    previous=latest(checkpoints,digest(PLAN))
    adapter=previous[0]/'adapter' if previous else SFT
    model,params=load_policy(model_path,adapter)
    optimizer=torch.optim.AdamW(params,lr=CONFIG['lr'],weight_decay=0)
    if previous:
        state=previous[1]
        if state['temperature']!=temp: raise ValueError('Wrong temperature checkpoint')
        optimizer.load_state_dict(torch.load(previous[0]/'optimizer.pt',weights_only=True)['optimizer'])
    start_step=previous[1]['step'] if previous else 0
    for step in range(start_step+1,CONFIG['updates']+1):
        records=[]; sampler=Sampler(model,tokenizer,temp)
        batch=cases[(step-1)*12:step*12]
        path=results/f'rollouts-update-{step}.jsonl'
        metadata=dict(plan_sha256=digest(PLAN),temperature=temp,update=step)
        expected=[c for c in batch for _ in range(CONFIG['group_size'])]
        saved=validate_records(path,metadata,expected)
        records.extend(saved)
        with path.open('a') as handle:
            for idx,case in enumerate(expected[len(saved):],len(saved)):
                seed=CONFIG['seed']+step*10000+idx  # identical seeds/tasks across independent arms
                set_seed(seed); sampler.behavior=[]
                print(f'{name} update {step}: rollout {idx+1}/{len(expected)}',flush=True)
                result=run_episode(sampler.generate,case['query'],case['environment'],max_turns=40,max_calls=36)
                reward,metrics=strict_reward(case,result)
                record=dict(case=case,result=result,metrics=metrics,reward=reward,behavior=sampler.behavior,
                    seed=seed,metadata=metadata,completed_at=datetime.now(timezone.utc).isoformat())
                handle.write(canonical(record)+'\n'); handle.flush(); records.append(record)
        torch.cuda.reset_peak_memory_stats()
        stats=update(model,params,optimizer,records,temp)
        stats.update(temperature=temp,step=step,episodes=len(records),rollouts_sha256=digest(path))
        folder=checkpoints/f'checkpoint-{step}'
        if folder.exists():
            # Preserve incomplete writes; never overwrite an unsealed checkpoint.
            folder.rename(checkpoints/f'interrupted-checkpoint-{step}-{time.time_ns()}')
        folder.mkdir(); model.set_adapter('default')
        model.save_pretrained(folder/'adapter',selected_adapters=['default'])
        torch.save(dict(optimizer=optimizer.state_dict()),folder/'optimizer.pt')
        seal(folder,dict(plan_sha256=digest(PLAN),step=step,temperature=temp,stats=stats))
        write_once(results/f'update-{step}.json',stats)
        print(canonical(stats),flush=True)
    del optimizer,model,params
    torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['freeze','gate','train'],required=True)
    parser.add_argument('--model-path',default='/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507')
    args=parser.parse_args(); spec,cases=protocol(args.stage=='freeze')
    if args.stage=='freeze': print('Corrected pilot frozen: 3 independent SFT arms, 288 fresh train rollouts'); return
    import torch
    if torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported(): raise RuntimeError('Allocate one bf16 GPU')
    tokenizer,_=checked_tokenizer(args.model_path,bundle())
    if sha256_adapter(SFT)!=spec['source_adapter_sha256']: raise ValueError('Starting SFT adapter drift')
    OUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (OUT/'.run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=='gate':
            model,params=load_policy(args.model_path,SFT)
            result=gate(model,params,tokenizer)
            result['plan_sha256']=digest(PLAN)
            write_once(OUT/'memory_gate.json',result)
            print(canonical(result),flush=True)
        else:
            memory=json.loads((OUT/'memory_gate.json').read_text())
            if memory['plan_sha256']!=digest(PLAN): raise ValueError('Run memory gate with this frozen plan first')
            for temperature in CONFIG['temperatures']: train_arm(args.model_path,tokenizer,temperature,cases)
            write_once(OUT/'completed.json',dict(plan_sha256=digest(PLAN),temperatures=CONFIG['temperatures']))


if __name__=='__main__': main()
