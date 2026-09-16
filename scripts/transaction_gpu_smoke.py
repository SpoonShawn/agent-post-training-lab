"""GPU readiness gate only: real tokenization, 14 engineering episodes, two train steps."""
import argparse
from datetime import datetime,timezone
import hashlib
import heapq
import importlib.metadata
import json
import math

from scripts.prepare_control_v1 import ROOT,digest
from scripts.prepare_transaction_v1 import DATA,bundle
from scripts.run_baseline import sha256_model_path
from scripts.evidence_pilot_gpu import sha256_adapter
from training.transaction_data import canonical,teacher_trajectory,SEED
from training.transaction_sft import encode_trajectory
from training.sft import AssistantCollator
from agent.transaction_runtime import SYSTEM,TOOLS,run_episode
from agent.transaction_tasks import evaluate

RESULTS=ROOT/"results/transaction_v1_smoke"
OUTPUT=ROOT/"checkpoints/transaction_v1"


def rows(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def write_once(path,value):
    content=canonical(value)+"\n"
    if path.exists() and path.read_text()!=content:
        raise ValueError(f"Preserve changed artifact: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        path.write_text(content)


def checked_tokenizer(model_path,manifest):
    versions={p:importlib.metadata.version(p) for p in manifest["versions"]}
    if versions != manifest["versions"]:
        raise ValueError("Dependency drift; do not upgrade the old environment")
    value,_=sha256_model_path(model_path)
    if value!=manifest["model_sha256"]:
        raise ValueError("Base model content changed")
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    return tokenizer,versions


def preflight(model_path):
    manifest=bundle()
    tokenizer,versions=checked_tokenizer(model_path,manifest)
    statistics,samples={},{}
    for split,keep in (("train",16),("dev",8)):
        heap=[]
        count=input_tokens=supervised=maximum=0
        for index,row in enumerate(rows(DATA/f"{split}_trajectories.jsonl"),1):
            for turn,example in enumerate(encode_trajectory(row,tokenizer,8192)):
                count+=1
                length=len(example["input_ids"])
                input_tokens+=length
                supervised+=sum(t!=-100 for t in example["labels"])
                maximum=max(maximum,length)
                item=(length,count,dict(case_id=row["id"],assistant_index=turn,example=example))
                if len(heap)<keep:
                    heapq.heappush(heap,item)
                elif item[:2]>heap[0][:2]:
                    heapq.heapreplace(heap,item)
            if index%200==0:
                print(f"Tokenization {split}: {index} trajectories, {count} assistant examples",flush=True)
        if count!=manifest["counts"][split]["assistant_examples"]:
            raise ValueError("Supervised example count mismatch")
        statistics[split]=dict(examples=count,total_input_tokens_including_targets=input_tokens,
                               supervised_tokens=supervised,max_tokens=maximum)
        samples[split]=[item[2] for item in sorted(heap,reverse=True)]
    max_prompt=0
    # This only tokenizes oracle public prefixes, never trains on confirmation trajectories.
    for split in ("dev","confirmation_id","confirmation_ood"):
        for index,case in enumerate(rows(DATA/f"{split}_cases.jsonl"),1):
            trajectory,_,_=teacher_trajectory(case)
            for i,message in enumerate(trajectory["messages"]):
                if message["role"]!="assistant":
                    continue
                prompt=tokenizer.apply_chat_template(trajectory["messages"][:i],tools=TOOLS,
                                                    add_generation_prompt=True,tokenize=False)
                length=len(tokenizer.encode(prompt,add_special_tokens=False))
                if length+512>8192:
                    raise ValueError(f"Oracle evaluation prefix exceeds budget: {case['id']}")
                max_prompt=max(max_prompt,length)
            if index%100==0:
                print(f"Context audit {split}: {index} cases",flush=True)
    write_once(OUTPUT/"preflight_examples.json",samples)
    metadata=dict(manifest_sha256=digest(DATA/"manifest.json"),model_sha256=manifest["model_sha256"],
                  versions=versions,seed=SEED,generation=manifest["generation"],statistics=statistics,
                  max_oracle_eval_prompt_tokens=max_prompt,smoke_examples_sha256=digest(OUTPUT/"preflight_examples.json"),
                  smoke_sample_refs={s:[dict(case_id=x["case_id"],assistant_index=x["assistant_index"],
                                           tokens=len(x["example"]["input_ids"])) for x in values] for s,values in samples.items()},
                  chat_template_sha256=hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest(),
                  scope="engineering_smoke_not_formal_training_or_confirmation",
                  selection="16 longest train assistant-prefix examples; 8 longest dev examples for loss only")
    write_once(RESULTS/"preflight.json",metadata)
    print(json.dumps(metadata,ensure_ascii=False,indent=2),flush=True)


def load_preflight(model_path):
    manifest=bundle()
    tokenizer,versions=checked_tokenizer(model_path,manifest)
    meta=json.loads((RESULTS/"preflight.json").read_text())
    if (meta["manifest_sha256"]!=digest(DATA/"manifest.json") or meta["model_sha256"]!=manifest["model_sha256"]
            or meta["versions"]!=versions or meta["generation"]!=manifest["generation"]
            or meta["chat_template_sha256"]!=hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()
            or meta["smoke_examples_sha256"]!=digest(OUTPUT/"preflight_examples.json")):
        raise ValueError("Preflight provenance drift")
    samples=json.loads((OUTPUT/"preflight_examples.json").read_text())
    if len(samples["train"])!=16 or len(samples["dev"])!=8:
        raise ValueError("Incomplete smoke examples")
    return tokenizer,samples,meta


def run_meta(meta,role,adapter_hash=None):
    result=dict(meta,role=role,adapter_sha256=adapter_hash)
    result["fingerprint"]=hashlib.sha256(canonical(result).encode()).hexdigest()
    return result


def validate_records(path,metadata,cases):
    saved=list(rows(path)) if path.exists() else []
    if canonical([r["case"] for r in saved])!=canonical(cases[:len(saved)]) or len(saved)>len(cases):
        raise ValueError("Incomplete-order, duplicate or changed smoke cases")
    for row in saved:
        case,result=row["case"],row["result"]
        if canonical(row["metadata"])!=canonical(metadata) or result["query"]!=case["query"]:
            raise ValueError("Smoke metadata/query drift")
        metrics=evaluate(case["environment"],case["desired"],result["events"],result["final_answer"])
        if canonical(row["metrics"])!=canonical(metrics) or type(result["tool_calls"]) is not int or result["tool_calls"]!=len(result["events"]):
            raise ValueError("Smoke replay/score/count drift")
        tool_turns=[t for t in result["turns"] if "result" in t]
        if ([dict(index=i,tool_call=t["tool_call"],result=t["result"]) for i,t in enumerate(tool_turns)]!=result["events"]):
            raise ValueError("Model turns and execution events disagree")
        if result["final_answer"] is not None and result["final_answer"]!=result["turns"][-1]["model_output"]:
            raise ValueError("Final answer differs from model output")
    return saved


def infer(model_path,meta,role):
    adapter=None
    if role=="sft_smoke":
        record=json.loads((RESULTS/"training_run.json").read_text())
        adapter=OUTPUT/"smoke"/"adapter"
        if record["status"]!="complete" or record["metadata"]!=meta or sha256_adapter(adapter)!=record["adapter_sha256"]:
            raise ValueError("Smoke adapter provenance mismatch")
    metadata=run_meta(meta,role,sha256_adapter(adapter) if adapter else None)
    cases=list(rows(DATA/"smoke_cases.jsonl"))
    path=RESULTS/f"{role}.jsonl"
    saved=validate_records(path,metadata,cases)
    pending=cases[len(saved):]
    if not pending:
        print("Complete engineering results preserved; model not loaded",flush=True)
        return
    import torch
    from agent.transaction_model import TransactionAgent
    agent=TransactionAgent(model_path,adapter_path=adapter)
    with path.open("a") as handle:
        for index,case in enumerate(pending,len(saved)+1):
            start=datetime.now(timezone.utc).isoformat()
            agent.usage=[]
            print(f"{role} [{index}/{len(cases)}] {case['id']} starting",flush=True)
            def generate(messages,tools):
                if len(agent.usage)%5==0:
                    print(f"  generating turn {len(agent.usage)+1}...",flush=True)
                return agent.generate(messages,tools)
            result=run_episode(generate,case["query"],case["environment"],max_turns=40,max_calls=36)
            metrics=evaluate(case["environment"],case["desired"],result["events"],result["final_answer"])
            row=dict(case=case,result=result,metrics=metrics,metadata=metadata,generation_usage=agent.usage,
                     started_at=start,completed_at=datetime.now(timezone.utc).isoformat(),gpu=torch.cuda.get_device_name(0))
            handle.write(canonical(row)+"\n")
            handle.flush()
            print(f"{role} [{index}/{len(cases)}] calls={metrics['tool_calls']} reason={result['terminated_reason']} success={metrics['task_success']}",flush=True)


def train_smoke(model_path,tokenizer,samples,meta):
    folder=OUTPUT/"smoke"
    if folder.exists() and any(folder.iterdir()):
        record=json.loads((folder/"run.json").read_text())
        if (record["status"]=="complete" and record["metadata"]==meta
                and record["adapter_sha256"]==sha256_adapter(folder/"adapter")
                and record==json.loads((RESULTS/"training_run.json").read_text())):
            print("Completed two-step smoke preserved",flush=True)
            return
        raise ValueError("Preserve incomplete/different training; request diagnosis, do not delete")
    cases=list(rows(DATA/"smoke_cases.jsonl"))
    if len(validate_records(RESULTS/"base.jsonl",run_meta(meta,"base"),cases))!=14:
        raise ValueError("Complete 14-case engineering Base run first")
    import torch
    from transformers import AutoModelForCausalLM,TrainingArguments,Trainer,set_seed
    from peft import LoraConfig,get_peft_model
    folder.mkdir(parents=True,exist_ok=True)
    record=dict(status="started",metadata=meta,started_at=datetime.now(timezone.utc).isoformat(),
                gpu=torch.cuda.get_device_name(0),baseline_sha256=digest(RESULTS/"base.jsonl"),
                config=dict(max_steps=2,batch=1,accumulation=8,lr=1e-4,r=16,alpha=32,dropout=.05,
                            target_modules="all-linear",seed=SEED,scope="smoke_only_never_resume_as_full_SFT"))
    def save():
        payload=json.dumps(record,ensure_ascii=False,indent=2)+"\n"
        (folder/"run.json").write_text(payload)
        (RESULTS/"training_run.json").write_text(payload)
    save()
    try:
        set_seed(SEED)
        torch.cuda.reset_peak_memory_stats()
        model=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,trust_remote_code=False)
        model.config.use_cache=False
        model=get_peft_model(model,LoraConfig(task_type="CAUSAL_LM",r=16,lora_alpha=32,lora_dropout=.05,
                                             target_modules="all-linear",bias="none"))
        record["trainable_parameters"]=sum(p.numel() for p in model.parameters() if p.requires_grad)
        options=TrainingArguments(output_dir=str(folder),max_steps=2,learning_rate=1e-4,
            per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,
            bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},
            optim="adamw_torch",lr_scheduler_type="constant",warmup_steps=0,eval_strategy="no",save_strategy="no",
            logging_steps=1,report_to="none",remove_unused_columns=False,label_names=["labels"],seed=SEED,data_seed=SEED)
        trainer=Trainer(model=model,args=options,train_dataset=[s["example"] for s in samples["train"]],
                        eval_dataset=[s["example"] for s in samples["dev"]],data_collator=AssistantCollator(tokenizer.pad_token_id))
        trained=trainer.train()
        validation=trainer.evaluate()
        if trainer.state.global_step!=2 or not math.isfinite(trained.training_loss) or not math.isfinite(validation["eval_loss"]):
            raise ValueError("Invalid smoke steps or loss")
        model.save_pretrained(folder/"adapter")
        tokenizer.save_pretrained(folder/"adapter")
        trainer.save_state()
        record.update(status="complete",training_metrics=trained.metrics,evaluation_metrics=validation,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                      trainer_state=json.loads((folder/"trainer_state.json").read_text()),
                      adapter_sha256=sha256_adapter(folder/"adapter"),completed_at=datetime.now(timezone.utc).isoformat())
        save()
    except Exception as exc:
        record.update(status="failed",error_type=type(exc).__name__,error=str(exc))
        save()
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("preflight","base","train_smoke","sft_smoke"),required=True)
    parser.add_argument("--model-path",default="/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507")
    args=parser.parse_args()
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("请先申请bf16 GPU，不要在登录节点启动")
    print("Verifying frozen bundle, weights, package versions and tokenizer...",flush=True)
    if args.stage=="preflight":
        preflight(args.model_path)
    else:
        tokenizer,samples,meta=load_preflight(args.model_path)
        if args.stage=="train_smoke":
            train_smoke(args.model_path,tokenizer,samples,meta)
        else:
            infer(args.model_path,meta,args.stage)


if __name__ == "__main__":
    main()
