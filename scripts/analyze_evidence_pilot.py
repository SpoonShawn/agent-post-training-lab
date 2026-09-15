"""Validate frozen evidence pilot and compare report fields, not full Agent success."""
from collections import Counter
from datetime import datetime
import hashlib
import json

from scripts.prepare_evidence_pilot import contents, CODE, FOLDER
from scripts.prepare_control_v1 import ROOT, digest
from scripts.verify_control_v1 import verify
from training.evidence_pilot import canonical, score

RESULTS = ROOT/"results/evidence_pilot_v1"


def analyze():
    control, _ = verify()
    manifest = json.loads((FOLDER/"manifest.json").read_text())
    files, counts, groups = contents()
    if counts != manifest["counts"] or groups != manifest["groups"]:
        raise ValueError("Dataset grouping drift")
    for name, value in files.items():
        if (FOLDER/name).read_text() != value or digest(FOLDER/name) != manifest["files"][name]:
            raise ValueError("Frozen dataset drift")
    if manifest["code"] != {p:digest(ROOT/p) for p in CODE}:
        raise ValueError("Frozen implementation drift")
    train = json.loads((RESULTS/"training_run.json").read_text())
    smoke = json.loads((RESULTS/"smoke_run.json").read_text())
    meta = json.loads((RESULTS/"preflight.json").read_text())
    if (train["status"] != "complete" or train["smoke"] is not False or train["metadata"] != meta
            or smoke["status"] != "complete" or smoke["smoke"] is not True or smoke["metadata"] != meta
            or meta["manifest_sha256"] != digest(FOLDER/"manifest.json")
            or meta["model_sha256"] != control["model_sha256"] or meta["versions"] != control["versions"]
            or train["trainer_state"]["global_step"] != 24 or train["training_metrics"]["epoch"] != 1
            or smoke["trainer_state"]["global_step"] != 2):
        raise ValueError("Training/preflight provenance mismatch")
    cases = [json.loads(l) for s in ("validation","confirmation")
             for l in (FOLDER/f"{s}_cases.jsonl").read_text().splitlines()]
    runs, output, bad = {}, {}, []
    for role in ("base","old_sft","new_sft"):
        path = RESULTS/f"{role}.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        if [r["case"] for r in rows] != cases:
            raise ValueError("Incomplete, duplicate or changed case selection")
        expected = dict(meta, role=role, adapter_sha256=(
            None if role == "base" else control["adapter_sha256"] if role == "old_sft" else train["adapter_sha256"]))
        expected["fingerprint"] = hashlib.sha256(canonical(expected).encode()).hexdigest()
        for row in rows:
            if row["metadata"] != expected or row["metrics"] != score(row["answer"],row["case"]["target"]):
                raise ValueError("Run metadata or recorded metrics mismatch")
        if role != "new_sft" and train["baseline_sources"][role] != digest(path):
            raise ValueError("Training was not based on these baseline files")
        if role != "new_sft" and rows[-1]["completed_at"] > train["started_at"]:
            raise ValueError("Baseline completed after training start")
        if role == "new_sft" and rows[0]["started_at"] < train["completed_at"]:
            raise ValueError("New adapter evaluated before training completion")
        runs[role] = rows
        output[role] = dict(source_sha256=digest(path), metadata=expected, splits={},
                            first_started=rows[0]["started_at"], last_completed=rows[-1]["completed_at"])
        for split in ("validation","confirmation"):
            selected = [r for r in rows if r["case"]["split"] == split]
            field_errors, by_pattern = Counter(), {}
            format_only = 0
            for row in selected:
                pattern = row["case"]["pattern"]
                part = by_pattern.setdefault(pattern, {"cases":0, "exact_report":0})
                part["cases"] += 1
                part["exact_report"] += int(row["metrics"]["exact_report"])
                if not row["metrics"]["exact_report"]:
                    if row["metrics"]["valid_json"]:
                        answer = json.loads(row["answer"])
                        wrong = [k for k,v in row["case"]["target"].items()
                                 if canonical(answer.get(k)) != canonical(v)]
                        field_errors.update(wrong)
                        format_only += int(wrong == ["policy_blocks"] and answer.get("policy_blocks")==[])
                    else:
                        wrong = ["non_json_report"]  # Do not pretend all semantic fields were judged wrong.
                    bad.append(dict(role=role, split=split, case=row["case"], answer=row["answer"],
                                    metrics=row["metrics"], wrong_fields=wrong))
            output[role]["splits"][split] = dict(
                cases=48, groups=len({r["case"]["group_id"] for r in selected}),
                **{k:sum(r["metrics"][k] for r in selected)
                   for k in ("valid_json","exact_report","decision_correct")},
                field_errors_among_valid_json=dict(field_errors),
                only_policy_blocks_empty_list=format_only, by_pattern=by_pattern)
    pairs = {}
    for split in ("validation","confirmation"):
        pairs[split] = dict(Counter(
            f"{int(b['metrics']['exact_report'])}->{int(n['metrics']['exact_report'])}"
            for b,n in zip(runs["base"],runs["new_sft"]) if b["case"]["split"]==split))
    output["base_to_new_paired"] = pairs
    output["training"] = dict(source_sha256=digest(RESULTS/"training_run.json"),
                              run=train, preflight=meta,
                              wall_seconds=(datetime.fromisoformat(train["completed_at"]) -
                                            datetime.fromisoformat(train["started_at"])).total_seconds())
    output["limitations"] = [
        "Structured trace-audit only; not full Agent execution or autonomous recovery.",
        "One template, 8 held-out pattern/field groups per split; shared primitives and explicit decision rules.",
        "New LoRA initialized from Base, not continued old LoRA. Old model's 96 non-JSON outputs indicate task-format mismatch.",
        "No significance claim; no independent human review. Deterministic target checks are field-level validation.",
        "Adapter save hash matches inference metadata; local analysis has no GPU weight files for independent byte rehash.",
    ]
    return output, bad


def main():
    summary, bad = analyze()
    folder = ROOT/"results/analysis/evidence_pilot_v1"
    outputs = {"summary.json":json.dumps(summary,ensure_ascii=False,indent=2)+"\n",
               "bad_cases.jsonl":"".join(canonical(r)+"\n" for r in bad)}
    for name,value in outputs.items():
        p = folder/name
        if p.exists() and p.read_text()!=value:
            raise ValueError("Refuse to overwrite changed analysis")
    folder.mkdir(parents=True,exist_ok=True)
    for name,value in outputs.items():
        if not (folder/name).exists():
            (folder/name).write_text(value)
    print(json.dumps({r:summary[r]["splits"] for r in ("base","old_sft","new_sft")},
                     ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
