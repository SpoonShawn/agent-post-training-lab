"""Evaluate one sealed GRPO adapter on an independent confirmation split."""
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from scripts.prepare_control_v1 import ROOT, digest
from scripts.prepare_transaction_v1 import DATA, bundle
from scripts.transaction_gpu_smoke import rows, checked_tokenizer, sha256_adapter, validate_records
from agent.transaction_model import TransactionAgent
from agent.transaction_runtime import run_episode
from agent.transaction_tasks import evaluate
from training.transaction_data import canonical

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model-path',default='/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507'); ap.add_argument('--adapter-path',required=True); ap.add_argument('--split',choices=['confirmation_id','confirmation_ood'],required=True); ap.add_argument('--output-path',required=True); args=ap.parse_args()
    import torch
    manifest=bundle(); checked_tokenizer(args.model_path,manifest)
    adapter=Path(args.adapter_path); expected=adapter/'adapter_model.safetensors'
    if not expected.exists(): raise ValueError('Adapter checkpoint is incomplete')
    cases=list(rows(DATA/f'{args.split}_cases.jsonl'))
    metadata=dict(scope='independent_grpo_pilot_eval',split=args.split,adapter_sha256=sha256_adapter(adapter),dataset_sha256=digest(DATA/'manifest.json'),generation=manifest['generation'])
    output=Path(args.output_path); output.parent.mkdir(parents=True,exist_ok=True)
    saved=validate_records(output,metadata,cases)
    if len(saved)==len(cases):
        print(f'{args.split}: verified existing {len(saved)}/{len(cases)} records',flush=True)
        return
    agent=TransactionAgent(args.model_path,adapter_path=adapter)
    with output.open('a') as handle:
        for i,case in enumerate(cases[len(saved):],len(saved)+1):
            started=datetime.now(timezone.utc).isoformat(); agent.usage=[]
            result=run_episode(agent.generate,case['query'],case['environment'],max_turns=40,max_calls=36)
            metrics=evaluate(case['environment'],case['desired'],result['events'],result['final_answer'])
            record=dict(case=case,result=result,metrics=metrics,generation_usage=agent.usage,metadata=metadata,started_at=started,completed_at=datetime.now(timezone.utc).isoformat(),gpu=torch.cuda.get_device_name(0))
            handle.write(canonical(record)+'\n'); handle.flush()
            print(f'{args.split} {i}/{len(cases)} task={int(metrics["task_success"])} calls={metrics["tool_calls"]}',flush=True)
    task=sum(1 for r in rows(output) if r['metrics']['task_success'])
    print(json.dumps(dict(split=args.split,task_success=task,denominator=len(cases),rate=task/len(cases),output=str(output)),ensure_ascii=False),flush=True)

if __name__=='__main__': main()
