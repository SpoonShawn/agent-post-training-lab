"""Audit real cross-policy pairs and freeze grouped preference splits."""
import argparse
from collections import Counter
import hashlib
import json

from scripts.transaction_base_preferences import protocol, pairs, RESULTS as SOURCE, PLAN as SOURCE_PLAN
from scripts.transaction_gpu_smoke import validate_records, write_once
from scripts.prepare_control_v1 import ROOT, digest
from training.transaction_data import canonical

DATA = ROOT / "data/transaction_dpo_v1"


def build():
    spec, cases, sft = protocol()
    path = SOURCE / "base.jsonl"
    metadata = dict(plan_sha256=digest(SOURCE_PLAN), model_sha256=spec["model_sha256"], adapter_sha256=None)
    base = validate_records(path, metadata, cases)
    if len(base) != len(cases):
        raise ValueError("Incomplete Base coverage")
    pairing = dict(**pairs(base, sft), base_sha256=digest(path), sft_probe_sha256=spec["source_probe"]["rollouts"])
    if canonical(pairing) != canonical(json.loads((SOURCE / "pair_yield.json").read_text())):
        raise ValueError("Cross-policy pairing/source drift")
    groups = sorted({c["group_id"] for c in cases}, key=lambda g: hashlib.sha256(("dpo-v1:"+g).encode()).hexdigest())
    dev_groups = set(groups[:9])
    sources = dict(base=base, sft_probe=sft)
    records = []
    for pair in pairing["pairs"]:
        chosen = sources[pair["chosen"]["source"]][pair["chosen"]["index"]]
        rejected = sources[pair["rejected"]["source"]][pair["rejected"]["index"]]
        if not chosen["metrics"]["task_success"] or rejected["metrics"]["task_success"]:
            raise ValueError("Preference direction mismatch")
        for r in (chosen, rejected):
            if r["result"]["terminated_reason"] != "final_answer" or any(u["generated_tokens"] >= 512 for u in r["generation_usage"]):
                raise ValueError("Truncated trajectory needs explicit EOS handling; do not silently encode")
        records.append(dict(case_id=pair["case_id"], group_id=chosen["case"]["group_id"],
            split="preference_dev" if chosen["case"]["group_id"] in dev_groups else "preference_train",
            case=chosen["case"], chosen=chosen["result"], rejected=rejected["result"], provenance=pair))
    summary = dict(scope="cross_policy_train_only_preferences", pairs=len(records),
        base_cases=len(base), base_success=sum(r["metrics"]["task_success"] for r in base),
        base_execution_success=sum(r["metrics"]["execution_success"] for r in base),
        abstained_tasks=len(pairing["abstained_task_ids"]), splits=dict(Counter(r["split"] for r in records)),
        source_plan_sha256=digest(SOURCE_PLAN), source_base_sha256=digest(path),
        source_sft_sha256=pairing["sft_probe_sha256"], source_pairing_sha256=digest(SOURCE / "pair_yield.json"),
        base_generation_seconds=sum(u["seconds"] for r in base for u in r["generation_usage"]),
        limitations="Preference dev groups were seen during SFT; held out only from DPO updates, not independent SFT confirmation")
    return records, summary


def freeze(allow_create=False):
    records, summary = build()
    payloads = {"pairs.json": records, "manifest.json": dict(summary,
        code_sha256=digest(ROOT / "scripts/prepare_transaction_dpo.py"),
        pairs_sha256=hashlib.sha256((canonical(records)+"\n").encode()).hexdigest())}
    for name, value in payloads.items():
        path = DATA / name
        if allow_create:
            write_once(path, value)
        elif not path.exists() or path.read_text() != canonical(value)+"\n":
            raise ValueError("Preference dataset drift")
    return records, payloads["manifest.json"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    print(json.dumps(freeze(parser.parse_args().freeze)[1], ensure_ascii=False, indent=2))
