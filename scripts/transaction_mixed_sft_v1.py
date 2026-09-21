"""Mixed original-replay + repaired SFT, initialized from original full SFT."""
import argparse,json
from pathlib import Path
from scripts.prepare_control_v1 import ROOT,digest
from scripts.prepare_transaction_v1 import bundle
from scripts.transaction_gpu_smoke import checked_tokenizer,sha256_adapter
from training.transaction_sft import encode_trajectory
from training.sft import AssistantCollator

DATA=ROOT/'data'/'transaction_mixed_sft_v1'; SFT=ROOT/'checkpoints'/'transaction_v1_full_sft'/'adapter'; PLAN=ROOT/'data'/'transaction_mixed_sft_v1.json'; OUT=ROOT/'checkpoints'/'transaction_mixed_sft_v1'
CONFIG=dict(max_steps=160,lr=3e-5,batch=1,accumulation=4,r=16,alpha=32,dropout=0.,max_length=8192,seed=20260921)

def protocol(freeze=False):
    manifest=json.loads((DATA/'manifest.json').read_text()); source=json.loads((ROOT/'results'/'transaction_v1_full_sft'/'training_run.json').read_text())
    spec=dict(config=CONFIG,dataset_sha256=digest(DATA/'manifest.json'),source_sft_sha256=source['adapter_sha256'],source_code_sha256=digest(Path(__file__)),manifest=manifest)
    from scripts.transaction_gpu_smoke import write_once
    if freeze: write_once(PLAN,spec)
    elif not PLAN.exists() or json.loads(PLAN.read_text())!=spec: raise ValueError('mixed SFT protocol drift')
    return spec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--freeze',action='store_true'); ap.add_argument('--model-path',default='/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507'); a=ap.parse_args(); spec=protocol(a.freeze)
    if a.freeze: print('Mixed SFT protocol frozen'); return
    import torch
    from transformers import AutoModelForCausalLM,Trainer,TrainingArguments,set_seed
    from peft import PeftModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('bf16 GPU required')
    tokenizer,_=checked_tokenizer(a.model_path,bundle()); rows=[json.loads(x) for x in open(DATA/'train_trajectories.jsonl')]
    examples=[e for row in rows for e in encode_trajectory(row,tokenizer,CONFIG['max_length'])]
    base=AutoModelForCausalLM.from_pretrained(a.model_path,dtype=torch.bfloat16,device_map='auto',trust_remote_code=False); base.config.use_cache=False
    model=PeftModel.from_pretrained(base,SFT,is_trainable=True); model.train()
    args=TrainingArguments(output_dir=str(OUT),max_steps=CONFIG['max_steps'],learning_rate=CONFIG['lr'],per_device_train_batch_size=1,gradient_accumulation_steps=CONFIG['accumulation'],bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},optim='adamw_torch',lr_scheduler_type='constant',warmup_steps=0,logging_steps=8,save_strategy='no',report_to='none',remove_unused_columns=False,seed=CONFIG['seed'],data_seed=CONFIG['seed'])
    set_seed(CONFIG['seed']); trainer=Trainer(model=model,args=args,train_dataset=examples,data_collator=AssistantCollator(tokenizer.pad_token_id)); trained=trainer.train(); OUT.mkdir(parents=True,exist_ok=True); model.save_pretrained(OUT/'adapter',selected_adapters=['default']); tokenizer.save_pretrained(OUT/'adapter'); (OUT/'training_run.json').write_text(json.dumps(dict(status='complete',plan_sha256=digest(PLAN),examples=len(examples),metrics=trained.metrics,adapter_sha256=sha256_adapter(OUT/'adapter')),ensure_ascii=False,indent=2)+'\n'); print(json.dumps(trained.metrics,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
