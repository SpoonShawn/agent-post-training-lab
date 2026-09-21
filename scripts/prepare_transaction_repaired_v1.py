"""Freeze a repaired benchmark with shared fault mechanisms and disjoint groups."""
import argparse, hashlib, itertools, json
from copy import deepcopy
from scripts.prepare_control_v1 import ROOT, digest
from training.transaction_data import canonical
from agent.transaction_tasks import oracle, evaluate
from agent.transaction_runtime import SYSTEM, TOOLS

DATA=ROOT/'data/transaction_repaired_v1'; SEED=20260921
AREAS=('state_revision','retry_idempotency','async_check','permission_recovery','evidence_report')

def gid(split,index): return hashlib.sha256(f'repaired-v1:{split}:{index}'.encode()).hexdigest()[:16]

def trajectory(case):
    result=oracle(case['environment'],case['desired']); metrics=evaluate(case['environment'],case['desired'],result['events'],result['final_answer'])
    if not metrics['task_success']: raise ValueError('oracle failed')
    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':case['query']}]
    for e in result['events']:
        messages += [{'role':'assistant','content':'<tool_call>'+canonical(e['tool_call'])+'</tool_call>'},{'role':'tool','content':json.dumps(e['result'],ensure_ascii=False,sort_keys=True)}]
    messages.append({'role':'assistant','content':result['final_answer']})
    return dict(id=case['id'],group_id=case['group_id'],split=case['split'],messages=messages,tools=TOOLS,teacher='repaired_oracle',execution_validated=True)

def build():
    configs=[{'quality':q,'cache_enabled':c} for q,c in itertools.product(('low','standard','high'),(False,True))]
    variants=[(a,b,r) for a,b,r in itertools.product(configs,configs,(11,23,37,53,71,89,101,127)) if a!=b]
    cases=[]; used_exact=set()
    # Shared mechanism in all splits; group and parameter combinations remain disjoint.
    split_groups={'train':range(0,12),'id':range(12,16),'ood':range(16,20)}
    for split,indices in split_groups.items():
        for index in indices:
            group=gid(split,index); area=AREAS[index%len(AREAS)]
            ranked=sorted(variants,key=lambda v:hashlib.sha256(f'{SEED}:{group}:{canonical(v)}'.encode()).hexdigest())
            chosen=[]
            for candidate in ranked:
                key=canonical([candidate[0],candidate[1],candidate[2]])
                if key in used_exact: continue
                used_exact.add(key); chosen.append(candidate)
                if len(chosen)==8: break
            if len(chosen)!=8: raise ValueError('not enough disjoint variants')
            for j,(initial,desired,revision) in enumerate(chosen):
                env=dict(initial_config=deepcopy(initial),revision=revision,transient_commits=index%3,
                    timeout_after_commit=bool(index%2),check_passes=bool(index%3),check_delay=(index+j)%4,
                    revoke_before_commit=True)
                cases.append(dict(id=f'rv1_{group}_{j:02d}',group_id=group,split=split,category='permission_recovery',
                    knowledge_area=area,environment=env,desired=deepcopy(desired),
                    query=f'请完成配置事务{canonical(desired)}。权限可能在事务中变化；严格依据工具反馈停止、恢复并报告全部证据。',
                    max_turns=40,max_calls=36,protocol='transaction_repaired_v1'))
    return cases

def freeze():
    cases=build(); groups={s:{c['group_id'] for c in cases if c['split']==s} for s in ('train','id','ood')}
    if any(groups[a]&groups[b] for a,b in itertools.combinations(groups,2)): raise ValueError('group leakage')
    exact={s:{canonical([c['environment'],c['desired']]) for c in cases if c['split']==s} for s in groups}
    if any(exact[a]&exact[b] for a,b in itertools.combinations(exact,2)): raise ValueError('scenario leakage')
    files={'train_cases.jsonl':'','id_cases.jsonl':'','ood_cases.jsonl':'','train_trajectories.jsonl':''}
    for split in ('train','id','ood'):
        rows=[c for c in cases if c['split']==split]; files[f'{split}_cases.jsonl']=''.join(canonical(c)+'\n' for c in rows)
        if split=='train': files['train_trajectories.jsonl']=''.join(canonical(trajectory(c))+'\n' for c in rows)
    manifest=dict(study='transaction_repaired_v1',seed=SEED,mechanism='revoke_before_commit shared across train/id/ood',
        knowledge_areas=list(AREAS),counts={s:sum(c['split']==s for c in cases) for s in ('train','id','ood')},
        groups={s:len(groups[s]) for s in groups},files={n:hashlib.sha256(v.encode()).hexdigest() for n,v in files.items()},
        source_code_sha256=digest(ROOT/'scripts/prepare_transaction_repaired_v1.py'),
        no_confirmation_training=True,probe_excluded=True)
    files['manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'; DATA.mkdir(parents=True,exist_ok=True)
    for name,content in files.items():
        path=DATA/name
        if path.exists() and path.read_text()!=content: raise ValueError(f'frozen drift: {name}')
        if not path.exists(): path.write_text(content)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':
    argparse.ArgumentParser().parse_args(); freeze()
