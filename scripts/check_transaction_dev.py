"""Replay development fixtures, including composed faults. No model is loaded."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

from agent.transaction_tasks import oracle, evaluate
from agent.transaction_runtime import run_episode

ROOT = Path(__file__).resolve().parents[1]


def check():
    desired = {"quality":"low","cache_enabled":False}
    scenarios = [dict(transient_commits=t,timeout_after_commit=u,
                      concurrent_config={"quality":"standard","cache_enabled":True} if c else None,
                      check_passes=p,check_delay=d)
                 for t,u,c,p,d in itertools.product((0,1,2),(False,True),(False,True),(False,True),(0,2))]
    scenarios += [{"writable":False},{"revoke_before_commit":True}]
    records = []
    for index,scenario in enumerate(scenarios):
        teacher = oracle(scenario,desired)
        answers = iter(["<tool_call>"+json.dumps(e["tool_call"],ensure_ascii=False)+"</tool_call>" for e in teacher["events"]]
                       + [teacher["final_answer"]])
        episode = run_episode(lambda *_:next(answers), "目标配置："+json.dumps(desired), scenario)
        if episode["events"] != teacher["events"]:
            raise ValueError("Scripted runtime differs from teacher replay")
        metrics = evaluate(scenario,desired,episode["events"],episode["final_answer"])
        if not metrics["task_success"]:
            raise ValueError(f"Development oracle failed at {index}")
        records.append(dict(id=f"dev_{index:03d}",scenario=scenario,desired=desired,episode=episode,metrics=metrics))
    files = ["agent/transaction_env.py","agent/transaction_tasks.py","agent/transaction_runtime.py","scripts/check_transaction_dev.py"]
    return dict(status="development_oracle_only_not_model_performance",cases=len(records),
                outcomes=dict(Counter(r["metrics"]["expected_report"]["outcome"] for r in records)),
                max_tool_calls=max(r["metrics"]["tool_calls"] for r in records),
                code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},records=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    result = check()
    if args.output:
        payload = json.dumps(result,ensure_ascii=False,indent=2)+"\n"
        if args.output.exists() and args.output.read_text() != payload:
            raise ValueError("Preserve earlier development audit; choose a new output")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if not args.output.exists():
            args.output.write_text(payload)
    print(json.dumps({k:v for k,v in result.items() if k != "records"},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
