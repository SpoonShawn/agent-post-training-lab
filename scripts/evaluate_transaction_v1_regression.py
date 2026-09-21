"""Evaluate an adapter on frozen transaction_v1 confirmation splits."""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from scripts.prepare_control_v1 import ROOT,digest
from scripts.prepare_transaction_v1 import bundle
from scripts.transaction_gpu_smoke import rows,checked_tokenizer,sha256_adapter
from agent.transaction_model import TransactionAgent
from agent.transaction_runtime import run_episode
from agent.transaction_tasks import evaluate
from training.transaction_data import canonical

DATA=ROOT/'data'/'transaction_v1'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-path',default='/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507')
    ap.add_argument('--adapter-path',required=True)
    ap.add_argument('--split',choices=['confirmation_id','confirmation_ood'],required=True)
    ap.add_argument('--output-path',required=True)
    a=ap.parse_args()
    import torch
    checked_tokenizer(a.model_path,bundle())
    adapter=Path(a.adapter_path)
    cases=list(rows(DATA/f'{a.split}_cases.jsonl'))
    agent=TransactionAgent(a.model_path,adapter_path=adapter)
    meta=dict(scope='transaction_v1_regression',split=a.split,
              dataset_sha256=digest(DATA/'manifest.json'),
              adapter_sha256=sha256_adapter(adapter))
    out=Path(a.output_path); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as h:
        for i,c in enumerate(cases,1):
            r=run_episode(agent.generate,c['query'],c['environment'],max_turns=40,max_calls=36)
            m=evaluate(c['environment'],c['desired'],r['events'],r['final_answer'])
            h.write(canonical(dict(case=c,result=r,metrics=m,metadata=meta,
                                   gpu=torch.cuda.get_device_name(0),
                                   completed_at=datetime.now(timezone.utc).isoformat()))+'\n')
            h.flush()
            print(f'{a.split} {i}/{len(cases)} task={int(m["task_success"])}',flush=True)

if __name__=='__main__': main()
