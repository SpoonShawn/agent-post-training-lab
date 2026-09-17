"""Audit uploaded DPO math/provenance without claiming local GPU reproduction."""
from datetime import datetime
import json
import math

from scripts.transaction_dpo_gate import protocol, PLAN, RESULTS, CONFIG
from scripts.transaction_gpu_smoke import write_once
from scripts.prepare_control_v1 import ROOT, digest
from training.transaction_data import canonical
from training.transaction_dpo import dpo_terms


def audit():
    pairs, plan = protocol()
    run = json.loads((RESULTS / "run.json").read_text())
    reference = json.loads((RESULTS / "reference.json").read_text())
    if (run["status"] != "complete" or run["plan_sha256"] != digest(PLAN)
            or canonical(run["config"]) != canonical(CONFIG)
            or reference["plan_sha256"] != digest(PLAN)
            or run["reference_sha256"] != digest(RESULTS / "reference.json")):
        raise ValueError("DPO gate provenance mismatch")
    if canonical(run["numeric_checks"]) != canonical(dict(direct_vs_two_pass_gradient=True, non_action_logit_mask=True)):
        raise ValueError("Numeric GPU checks did not pass")
    scores = reference["scores"]
    if set(scores) != {p["case_id"] for p in pairs}:
        raise ValueError("Reference coverage mismatch")
    for pair in pairs:
        value = scores[pair["case_id"]]
        if value["split"] != pair["split"]:
            raise ValueError("Reference split changed")
        for side in ("chosen", "rejected"):
            if not math.isfinite(value[side]) or value[side] > 1e-5:
                raise ValueError("Invalid reference log probability")
            if type(value["tokens"][side]) is not int or value["tokens"][side] <= 0:
                raise ValueError("Invalid token count")
    if [s["case_id"] for s in run["steps"]] != plan["update_case_ids"] or [s["step"] for s in run["steps"]] != [1, 2]:
        raise ValueError("Wrong DPO update cases/steps")
    for step in run["steps"]:
        ref = scores[step["case_id"]]
        expected = dpo_terms(step["chosen_logp"], step["rejected_logp"], ref["chosen"], ref["rejected"], CONFIG["beta"])
        for key, value in zip(("loss", "coefficient", "reference_adjusted_margin"), expected):
            if not math.isclose(step[key], value, abs_tol=1e-9):
                raise ValueError("DPO loss/margin/coefficient mismatch")
        if not math.isfinite(step["grad_norm_before_clip"]) or step["grad_norm_before_clip"] <= 0:
            raise ValueError("Invalid gradient norm")
    if abs(run["steps"][0]["loss"] - math.log(2)) > .01:
        raise ValueError("Initial reference mismatch")
    if set(run["probe_after"]) != set(plan["probe_case_ids"]):
        raise ValueError("Post-update probe coverage mismatch")
    probes = {}
    for cid, after in run["probe_after"].items():
        if not all(math.isfinite(after[s]) for s in ("chosen", "rejected")):
            raise ValueError("Nonfinite post-update log probability")
        before = scores[cid]
        loss, _, margin = dpo_terms(after["chosen"], after["rejected"], before["chosen"], before["rejected"])
        probes[cid] = dict(chosen_logp_delta=after["chosen"]-before["chosen"],
            rejected_logp_delta=after["rejected"]-before["rejected"], loss_after=loss, margin_after=margin)
    elapsed = (datetime.fromisoformat(run["completed_at"])-datetime.fromisoformat(run["started_at"])).total_seconds()
    if elapsed <= 0:
        raise ValueError("Invalid gate chronology")
    result = dict(status="audited_engineering_gate_not_task_improvement", reference_pairs=len(scores),
        steps=run["steps"], probes=probes, elapsed_seconds=elapsed,
        peak_allocated_gib=run["peak_cuda_allocated_bytes"]/2**30,
        peak_reserved_gib=run["peak_cuda_reserved_bytes"]/2**30,
        sources={name: digest(RESULTS/name) for name in ("run.json", "reference.json")},
        limits="GPU gradient checks and scores are uploaded measurements; weights not uploaded or locally recomputed; no task inference")
    return result, scores


if __name__ == "__main__":
    summary, _ = audit()
    write_once(ROOT / "results/analysis/transaction_dpo_gate_v1/summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
