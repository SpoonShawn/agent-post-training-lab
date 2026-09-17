"""Frozen three-model replay comparison, including DPO regressions."""
from datetime import datetime
import json
import math

from scripts.transaction_full_dpo import protocol, PLAN, RESULTS, CONFIG, GATE
from scripts.transaction_full_sft import PLAN as SFT_PLAN, RESULTS as SFT_RESULTS, inference_metadata, SPLITS
from scripts.transaction_gpu_smoke import validate_records, rows, write_once
from scripts.prepare_transaction_v1 import DATA
from scripts.prepare_control_v1 import ROOT, digest
from scripts.analyze_transaction_full_sft import aggregate, answer_errors
from training.transaction_data import canonical
from training.transaction_dpo import dpo_terms


def audit():
    pairs, reference, spec = protocol()
    run = json.loads((RESULTS/"training_run.json").read_text())
    if (run["status"] != "complete" or run["plan_sha256"] != digest(PLAN)
            or canonical(run["config"]) != canonical(CONFIG)
            or run["reference_sha256"] != digest(GATE/"reference.json")
            or run["source_adapter_sha256"] != spec["source_adapter_sha256"]):
        raise ValueError("DPO training provenance mismatch")
    if ([s["case_id"] for s in run["steps"]] != spec["training_order"]
            or [s["step"] for s in run["steps"]] != list(range(1,81))):
        raise ValueError("Training order/step mismatch")
    for step in run["steps"]:
        ref = reference[step["case_id"]]
        if ref["split"] != "preference_train":
            raise ValueError("Dev pair entered updates")
        expected = dpo_terms(step["chosen_logp"], step["rejected_logp"], ref["chosen"], ref["rejected"])
        for key, value in zip(("loss", "coefficient", "reference_adjusted_margin"), expected):
            if not math.isclose(step[key], value, rel_tol=1e-8, abs_tol=1e-12):
                raise ValueError("DPO training math changed")
        if not math.isfinite(step["grad_norm_before_clip"]) or step["grad_norm_before_clip"] < 0:
            raise ValueError("Invalid gradient")
    dev_ids = {p["case_id"] for p in pairs if p["split"] == "preference_dev"}
    if set(run["preference_dev"]) != dev_ids:
        raise ValueError("Preference-dev coverage mismatch")
    for cid, score in run["preference_dev"].items():
        ref = reference[cid]
        loss, _, margin = dpo_terms(score["chosen"], score["rejected"], ref["chosen"], ref["rejected"])
        if not math.isclose(loss,score["loss"],abs_tol=1e-12) or not math.isclose(margin,score["reference_adjusted_margin"],abs_tol=1e-9):
            raise ValueError("Preference-dev math mismatch")
    old_plan = json.loads(SFT_PLAN.read_text())
    old_run = json.loads((SFT_RESULTS/"training_run.json").read_text())
    result = dict(scope="single_seed_fixed_benchmark_DPO_comparison", sources={
        p.name:digest(p) for p in sorted(RESULTS.glob("*.json*"))}, splits={},
        training=dict(steps=80, mean_preference_dev_loss=sum(s["loss"] for s in run["preference_dev"].values())/18,
            step_seconds=sum(s["seconds"] for s in run["steps"]), first_step=run["steps"][0], last_step=run["steps"][-1],
            attempt_windows=[{k:v for k,v in a.items() if k!="updates"} for a in run["attempts"]],
            peak_allocated_gib_last_attempt=run["peak_cuda_allocated_bytes_this_attempt"]/2**30,
            peak_reserved_gib_last_attempt=run["peak_cuda_reserved_bytes_this_attempt"]/2**30),
        limitations="repeated synthetic benchmark, single seed, 80 cross-policy pairs with source and length confounding; no causal or significance claim")
    changes, failures = [], []
    for split in SPLITS:
        cases = list(rows(DATA/f"{split}_cases.jsonl"))
        arms = {}
        for role in ("base", "sft", "dpo"):
            if role == "dpo":
                path = RESULTS/f"dpo_{split}.jsonl"
                meta = dict(plan_sha256=digest(PLAN), role="dpo", split=split,
                    adapter_sha256=run["adapter_sha256"], generation=spec["generation"])
            else:
                path = SFT_RESULTS/f"{role}_{split}.jsonl"
                meta = inference_metadata(old_plan, role, split, old_run["adapter_sha256"] if role=="sft" else None)
            records = validate_records(path, meta, cases)
            if len(records) != len(cases):
                raise ValueError("Incomplete comparison arm")
            if role == "dpo":
                for line, r in enumerate(records,1):
                    start,end = (datetime.fromisoformat(r[k]) for k in ("started_at","completed_at"))
                    if end < start or start < datetime.fromisoformat(run["completed_at"]):
                        raise ValueError("Inference chronology mismatch")
                    if not r["metrics"]["task_success"]:
                        failures.append(dict(split=split, case_id=r["case"]["id"], source=str(path.relative_to(ROOT)),
                            line=line, metrics=r["metrics"], answer_errors=answer_errors(r), final_answer=r["result"]["final_answer"]))
            arms[role] = records
        comparison = {r:aggregate(values) for r,values in arms.items()}
        comparison["paired_vs_sft"] = dict(improved=0,regressed=0,both_pass=0,both_fail=0,identical_results=0)
        for old, new in zip(arms["sft"], arms["dpo"]):
            x,y = old["metrics"]["task_success"],new["metrics"]["task_success"]
            key = "both_pass" if x and y else "both_fail" if not x and not y else "improved" if y else "regressed"
            comparison["paired_vs_sft"][key] += 1
            identical = canonical(old["result"]) == canonical(new["result"])
            comparison["paired_vs_sft"]["identical_results"] += identical
            if not identical:
                changes.append(dict(split=split, case_id=new["case"]["id"], task_change=key,
                    sft_metrics=old["metrics"], dpo_metrics=new["metrics"],
                    sft_answer=old["result"]["final_answer"], dpo_answer=new["result"]["final_answer"]))
        comparison["by_group"] = {g:{role:aggregate([r for r in values if r["case"]["group_id"]==g])
            for role,values in arms.items()} for g in sorted({c["group_id"] for c in cases})}
        result["splits"][split] = comparison
    result["dpo_failed_cases"] = len(failures)
    return result, changes, failures


if __name__ == "__main__":
    summary, changes, failures = audit()
    folder = ROOT/"results/analysis/transaction_full_dpo_v1"
    for name,value in (("summary",summary),("changes",changes),("failure_index",failures)):
        write_once(folder/f"{name}.json",value)
    print(json.dumps({s:r["paired_vs_sft"] for s,r in summary["splits"].items()},indent=2))
