"""Paired audit of frozen SFT versus all GRPO pilot evaluation arms."""
import json
from pathlib import Path
from scripts.prepare_control_v1 import ROOT, digest
from scripts.transaction_gpu_smoke import rows
from training.transaction_data import canonical

SFT=ROOT/'results/transaction_v1_full_sft'
EVAL=ROOT/'results/transaction_grpo_eval_v2'
OUT=ROOT/'results/analysis/transaction_grpo_pilot_eval_v2.json'

def aggregate(records):
    metrics=[r['metrics'] for r in records]
    def rate(k): return sum(bool(m[k]) for m in metrics)/len(metrics)
    return dict(cases=len(records),task_success=dict(count=sum(bool(m['task_success']) for m in metrics),denominator=len(metrics),rate=rate('task_success')),
        execution_success=dict(count=sum(bool(m['execution_success']) for m in metrics),denominator=len(metrics),rate=rate('execution_success')),
        answer_correct=dict(count=sum(bool(m['answer_correct']) for m in metrics),denominator=len(metrics),rate=rate('answer_correct')),
        policy_violating_calls=sum(m['policy_violations'] for m in metrics),mean_tool_calls=sum(m['tool_calls'] for m in metrics)/len(metrics),
        mean_steps=sum(len(r['result']['turns']) for r in records)/len(records))

def main():
    result=dict(scope='paired_SFT_vs_GRPO_confirmation_audit',splits={},sources={})
    for split in ('confirmation_id','confirmation_ood'):
        base=list(rows(SFT/f'sft_{split}.jsonl')); result['sources']['sft_'+split]=digest(SFT/f'sft_{split}.jsonl')
        arms=[]
        for path in sorted(EVAL.glob(f't*/checkpoint-*/{split}.jsonl')):
            arm='/'.join(path.parts[-3:-1]); current=list(rows(path));
            if len(current)!=len(base) or [r['case']['id'] for r in current]!=[r['case']['id'] for r in base]: raise ValueError(f'coverage/order mismatch: {path}')
            improved=[]; regressed=[]; changed=[]
            for old,new in zip(base,current):
                x,y=old['metrics']['task_success'],new['metrics']['task_success']
                if not x and y: improved.append(new['case']['id'])
                if x and not y: regressed.append(new['case']['id'])
                if canonical(old['result'])!=canonical(new['result']): changed.append(new['case']['id'])
            key=f'{arm}/{split}'; result['sources'][key]=digest(path)
            arms.append(dict(arm=arm,aggregate=aggregate(current),paired=dict(improved=improved,regressed=regressed,changed_results=len(changed)),source=str(path.relative_to(ROOT))))
        result['splits'][split]=dict(sft=aggregate(base),arms=arms)
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
