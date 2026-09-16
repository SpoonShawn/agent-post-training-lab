"""Grouped full-episode data: split causal fault profiles before cosmetic variants."""
from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json

from agent.transaction_tasks import oracle, evaluate
from agent.transaction_runtime import SYSTEM, TOOLS

SEED = 20260917
SPLITS = ("train", "dev", "confirmation_id", "confirmation_ood")


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"))


def digest_value(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def trace_shape(events):
    return canonical([(e["tool_call"]["name"],
                       e["result"]["error_type"] if not e["result"]["ok"] else e["result"]["result"].get("status", "ok"))
                      for e in events])


def profiles():
    rows = []
    for t,u,c,p,d in itertools.product((0,1,2),(False,True),(False,True),(False,True),(0,1,2,3)):
        row = dict(transient_commits=t,timeout_after_commit=u,concurrent=c,check_passes=p,check_delay=d)
        row["group_id"] = digest_value(row)[:16]
        if u and c:
            row["split"] = "confirmation_ood"
        rows.append(row)
    for passed in (False,True):
        ranked = sorted([r for r in rows if r["check_passes"] == passed and "split" not in r],
                        key=lambda r:digest_value([SEED,r["group_id"]]))
        for i,row in enumerate(ranked):
            row["split"] = "train" if i < 24 else "dev" if i < 32 else "confirmation_id"
    for key,split in (("initial_readonly","train"),("revoke_before_commit","confirmation_ood")):
        rows.append(dict(permission_case=key,group_id=digest_value(key)[:16],split=split))
    return sorted(rows,key=lambda r:(SPLITS.index(r["split"]),r["group_id"]))


def build_cases():
    configs = [dict(quality=q,cache_enabled=c) for q,c in itertools.product(("low","standard","high"),(False,True))]
    variants = [(a,b,r) for a,b,r in itertools.product(configs,configs,(7,19)) if a != b]
    queries = ("请把测试环境配置更新为{target}，完成检查；失败按约定回滚或安全停止，最后报告实际证据。",
               "目标配置是{target}。请实际执行完整事务，并依据工具反馈处理故障，按约定给出最终报告。",
               "执行配置验收：{target}。不要只给操作建议，按工具返回完成提交、检查或必要恢复。",
               "请在当前测试环境落实{target}，遵守事务与权限约定，用最终观测说明结果。")
    rows = []
    for profile in profiles():
        ranked = sorted(variants,key=lambda v:digest_value([SEED,profile["group_id"],v]))
        for index,(initial,desired,revision) in enumerate(ranked[:42 if profile["split"] == "train" else 16]):
            env = dict(initial_config=deepcopy(initial),revision=revision)
            if "permission_case" in profile:
                env.update({"writable":False} if profile["permission_case"] == "initial_readonly" else {"revoke_before_commit":True})
                category = "permission"
            else:
                env.update({k:profile[k] for k in ("transient_commits","timeout_after_commit","check_passes","check_delay")})
                if profile["concurrent"]:
                    env["concurrent_config"] = deepcopy(next(c for c in configs if c not in (initial,desired)))
                category = "apply" if profile["check_passes"] else "rollback"
            rows.append(dict(id=f"txn1_{profile['group_id']}_{index:02d}",group_id=profile["group_id"],
                        split=profile["split"],category=category,profile=deepcopy(profile),environment=env,
                        desired=deepcopy(desired),query=queries[index%4].format(target=canonical(desired)),
                        max_turns=40,max_calls=36,protocol="transaction_v1"))
    return rows


def teacher_trajectory(case):
    result = oracle(case["environment"],case["desired"])
    metrics = evaluate(case["environment"],case["desired"],result["events"],result["final_answer"])
    if not metrics["task_success"] or len(result["events"]) >= case["max_turns"] or len(result["events"]) > case["max_calls"]:
        raise ValueError("Teacher failed public contract or exceeded budget")
    messages = [dict(role="system",content=SYSTEM),dict(role="user",content=case["query"])]
    for entry in result["events"]:
        messages += [dict(role="assistant",content="<tool_call>"+canonical(entry["tool_call"])+"</tool_call>"),
                     dict(role="tool",content=json.dumps(entry["result"],ensure_ascii=False,sort_keys=True))]
    messages.append(dict(role="assistant",content=result["final_answer"]))
    return dict(id=case["id"],group_id=case["group_id"],split=case["split"],messages=messages,
                tools=TOOLS,teacher="observation_driven_rule_policy_not_model",execution_validated=True), result, metrics


def audit_cases(cases):
    groups, shapes, exact, summary = {}, {}, {}, {}
    for split in SPLITS:
        selected = [c for c in cases if c["split"] == split]
        groups[split] = {c["group_id"] for c in selected}
        shapes[split], exact[split] = set(), set()
        steps, turns, outcomes = [], 0, Counter()
        by_group = {}
        for case in selected:
            trajectory,result,metrics = teacher_trajectory(case)
            shape = trace_shape(result["events"])
            if case["group_id"] in by_group and by_group[case["group_id"]] != shape:
                raise ValueError("Cosmetic variants changed structural trace")
            by_group[case["group_id"]] = shape
            shapes[split].add(shape)
            key = canonical([case["environment"],case["desired"]])
            if key in exact[split]:
                raise ValueError("Duplicate scenario/goal")
            exact[split].add(key)
            steps.append(metrics["tool_calls"])
            turns += sum(m["role"] == "assistant" for m in trajectory["messages"])
            outcomes[metrics["expected_report"]["outcome"]] += 1
        summary[split] = dict(cases=len(selected),groups=len(groups[split]),trace_shapes=len(shapes[split]),
                             assistant_examples=turns,min_calls=min(steps),max_calls=max(steps),outcomes=dict(outcomes))
    for i,a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            if groups[a]&groups[b] or shapes[a]&shapes[b] or exact[a]&exact[b]:
                raise ValueError(f"Cross-split overlap: {a}/{b}")
    return summary


def smoke_cases(cases):
    # Twelve dev groups, plus two train-side readonly engineering probes; no confirmation reads.
    selected = []
    for category in ("apply","rollback"):
        groups = sorted({c["group_id"] for c in cases if c["split"] == "dev" and c["category"] == category})[:6]
        selected += [next(c for c in cases if c["split"] == "dev" and c["group_id"] == gid) for gid in groups]
    selected += [c for c in cases if c["split"] == "train" and c["category"] == "permission"][:2]
    return selected
