"""Audit the preregistered layout comparison without changing frozen scoring."""
from collections import Counter
from datetime import datetime
import json
import math

from scripts.prepare_control_v1 import ROOT, digest
from scripts.prepare_evidence_ablation import DATA, bundle
from scripts.evidence_ablation_gpu import eval_cases, run_metadata, read_rows
from scripts.analyze_evidence_stress import validate_rows, transitions
from training.evidence_pilot import canonical
from training.evidence_ablation import STYLES

RESULTS = ROOT / "results/evidence_ablation_v1"


def analyze():
    manifest = bundle()
    meta = json.loads((RESULTS/"preflight.json").read_text())
    expected = dict(manifest_sha256=digest(DATA/"manifest.json"), model_sha256=manifest["model_sha256"],
                    versions=manifest["versions"], training=manifest["training"], generation=manifest["generation"])
    if any(meta.get(k) != v for k,v in expected.items()):
        raise ValueError("Preflight differs from frozen plan")
    prior_meta = json.loads((ROOT/"results/evidence_pilot_v1/preflight.json").read_text())
    if meta["chat_template_sha256"] != prior_meta["chat_template_sha256"]:
        raise ValueError("Chat template drift")
    if not 0 < meta["max_eval_prompt_tokens"] <= 4096-512:
        raise ValueError("Invalid recorded token budget")
    for split, count in (("train",192),("validation",48)):
        for role in ("fixed","mixed"):
            stat = meta["token_statistics"][role][split]
            if stat["examples"] != count or not 0 < stat["max_tokens"] <= 4096 or not 0 < stat["supervised_tokens"] < stat["input_tokens"]:
                raise ValueError("Invalid training token statistics")
        if meta["token_statistics"]["fixed"][split]["supervised_tokens"] != meta["token_statistics"]["mixed"][split]["supervised_tokens"]:
            raise ValueError("Supervised token counts differ")
    cases = eval_cases()
    records = {}
    for role in ("smoke","fixed","mixed"):
        record = json.loads((RESULTS/f"{role}_training.json").read_text())
        if (record["status"] != "complete" or record["metadata"] != meta or record["role"] != role
                or record["smoke"] is not (role == "smoke")
                or record["trainer_state"]["global_step"] != (2 if role == "smoke" else 24)
                or record["baseline_sha256"] != digest(RESULTS/"base.jsonl")
                or not all(math.isfinite(record[k][v]) for k,v in (("training_metrics","train_loss"),("evaluation_metrics","eval_loss")))
                or (role != "smoke" and record["training_metrics"]["epoch"] != 1)):
            raise ValueError("Training provenance or budget mismatch")
        if datetime.fromisoformat(record["started_at"]) >= datetime.fromisoformat(record["completed_at"]):
            raise ValueError("Invalid training chronology")
        records[role] = record
    if len({r["adapter_sha256"] for r in records.values()}) != 3:
        raise ValueError("Adapters unexpectedly identical")
    runs, bad = {}, []
    summary = dict(study="evidence_ablation_v1", preflight=meta, training=records,
                   source_sha256={p.name:digest(p) for p in sorted(RESULTS.iterdir()) if p.suffix in (".json", ".jsonl")})
    for role in ("base","fixed","mixed"):
        rows = read_rows(RESULTS/f"{role}.jsonl")
        validate_rows(rows, cases, run_metadata(meta,role,None if role == "base" else records[role]["adapter_sha256"]))
        runs[role] = rows
        result = dict(splits={}, first_started=rows[0]["started_at"], last_completed=rows[-1]["completed_at"],
                      gpu=sorted({r["gpu"] for r in rows}))
        for split in ("validation","confirmation"):
            result["splits"][split] = {}
            for style in STYLES:
                selected = [r for r in rows if r["case"]["split"] == split and r["case"]["style"] == style]
                fields, groups = Counter(), {}
                for row in selected:
                    c = row["case"]
                    group = groups.setdefault(c["group_id"], dict(cases=0, exact_report=0))
                    group["cases"] += 1
                    group["exact_report"] += int(row["metrics"]["exact_report"])
                    if row["metrics"]["exact_report"]:
                        continue
                    answer = json.loads(row["answer"]) if row["metrics"]["valid_json"] else None
                    target = c["target"]
                    wrong = ([k for k in sorted(set(target)|set(answer)) if k not in target or k not in answer
                              or canonical(target[k]) != canonical(answer[k])] if isinstance(answer,dict) else ["non_object_report"])
                    fields.update(wrong)
                    bad.append(dict(role=role, split=split, style=style, case=c, answer=row["answer"], wrong_fields=wrong))
                result["splits"][split][style] = dict(cases=len(selected),
                    **{k:sum(r["metrics"][k] for r in selected) for k in ("valid_json","exact_report","decision_correct")},
                    field_errors=dict(fields), groups=groups)
        summary[role] = result
    base_end = datetime.fromisoformat(runs["base"][-1]["completed_at"])
    previous = base_end
    for role in ("smoke","fixed","mixed"):
        if datetime.fromisoformat(records[role]["started_at"]) < previous:
            raise ValueError("Training started before prerequisite completed")
        previous = datetime.fromisoformat(records[role]["completed_at"])
    for role in ("fixed","mixed"):
        if datetime.fromisoformat(runs[role][0]["started_at"]) < previous:
            raise ValueError("Evaluation precedes both training runs")
    summary["fixed_to_mixed_paired"] = {}
    summary["group_confirmation_totals"] = {}
    for split in ("validation","confirmation"):
        summary["fixed_to_mixed_paired"][split] = {}
        for style in STYLES:
            selected = [{r["case"]["context_id"]:r for r in runs[role]
                         if r["case"]["split"] == split and r["case"]["style"] == style} for role in ("fixed","mixed")]
            summary["fixed_to_mixed_paired"][split][style] = transitions(*selected)
    for role in runs:
        groups = Counter()
        for r in runs[role]:
            if r["case"]["split"] == "confirmation":
                groups[r["case"]["group_id"]] += int(r["metrics"]["exact_report"])
        summary["group_confirmation_totals"][role] = dict(groups)
    summary["limitations"] = [
        "Single seed and synthetic tool-history reporting, not complete Agent execution or real-device testing.",
        "48 confirmation contexts in 8 groups; three correlated layouts, not 144 independent contexts.",
        "Equal examples, supervised answers and steps; input tokens and computation differ.",
        "Frozen objective scoring, no independent human review; local weights/tokenizer unavailable for byte/token revalidation.",
        "This compares two new arms; old versus new studies changed prompts and histories, so no single-factor attribution across studies."]
    return summary, bad


def main():
    summary, bad = analyze()
    folder = ROOT/"results/analysis/evidence_ablation_v1"
    outputs = {"summary.json":json.dumps(summary,ensure_ascii=False,indent=2)+"\n",
               "bad_cases.jsonl":"".join(canonical(r)+"\n" for r in bad)}
    for name,value in outputs.items():
        if (folder/name).exists() and (folder/name).read_text() != value:
            raise ValueError("Refuse changed analysis overwrite")
    folder.mkdir(parents=True,exist_ok=True)
    for name,value in outputs.items():
        if not (folder/name).exists():
            (folder/name).write_text(value)
    print(json.dumps({"paired":summary["fixed_to_mixed_paired"], "groups":summary["group_confirmation_totals"], "bad_cases":len(bad)},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
