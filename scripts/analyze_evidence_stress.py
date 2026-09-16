"""Read-only scoring audit of the frozen, post-training diagnostic study."""
from collections import Counter
from datetime import datetime
import hashlib
import json

from scripts.evidence_stress_v1 import DATA, verify_bundle
from scripts.prepare_control_v1 import ROOT, digest
from training.evidence_pilot import canonical, score

ARMS = ("original", "surface", "extended_history", "current_policy_flip")
RESULTS = ROOT / "results/evidence_stress_v1"


def validate_rows(rows, cases, expected):
    if [r["case"] for r in rows] != cases:
        raise ValueError("Incomplete, duplicated, reordered or changed cases")
    previous = None
    for row in rows:
        if row["metadata"] != expected or row["metrics"] != score(row["answer"], row["case"]["target"]):
            raise ValueError("Metadata or score drift")
        start, end = (datetime.fromisoformat(row[k]) for k in ("started_at", "completed_at"))
        if start.tzinfo is None or end.tzinfo is None or start > end or (previous and start < previous):
            raise ValueError("Invalid run chronology")
        previous = end


def transitions(left, right):
    if set(left) != set(right):
        raise ValueError("Unpaired contexts")
    return dict(Counter(f"{int(left[k]['metrics']['exact_report'])}->{int(right[k]['metrics']['exact_report'])}"
                        for k in left))


def analyze():
    from scripts.analyze_evidence_pilot import analyze as verify_prior
    verify_prior()
    cases, manifest = verify_bundle()
    parent = ROOT / "results/evidence_pilot_v1"
    preflight = json.loads((parent / "preflight.json").read_text())
    train = json.loads((parent / "training_run.json").read_text())
    summary = {"study": "post_training_diagnostic_not_heldout",
               "manifest_sha256": digest(DATA / "manifest.json"), "contexts": 24}
    runs, bad = {}, []
    for role in ("base", "new_sft"):
        path = RESULTS / f"{role}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        length = rows[0]["metadata"]["max_stress_prompt_tokens"]
        if type(length) is not int or not 0 < length <= 4096 - 512:
            raise ValueError("Invalid recorded context budget")
        expected = dict(preflight, study="evidence_stress_v1",
                        stress_manifest_sha256=digest(DATA / "manifest.json"),
                        max_stress_prompt_tokens=length, role=role,
                        adapter_sha256=manifest["adapter_sha256"] if role == "new_sft" else None)
        expected["fingerprint"] = hashlib.sha256(canonical(expected).encode()).hexdigest()
        validate_rows(rows, cases, expected)
        if datetime.fromisoformat(rows[0]["started_at"]) < datetime.fromisoformat(train["completed_at"]):
            raise ValueError("Pressure run predates training completion")
        prior = {r["case"]["id"]: r for r in map(json.loads, (parent / f"{role}.jsonl").read_text().splitlines())}
        runs[role] = {arm: {r["case"]["context_id"]: r for r in rows if r["case"]["arm"] == arm} for arm in ARMS}
        result = dict(source_sha256=digest(path), metadata=expected,
                      gpu=sorted({r["gpu"] for r in rows}), arms={},
                      first_started=rows[0]["started_at"], last_completed=rows[-1]["completed_at"],
                      case_window_seconds=(datetime.fromisoformat(rows[-1]["completed_at"]) -
                                           datetime.fromisoformat(rows[0]["started_at"])).total_seconds())
        for arm, selected in runs[role].items():
            errors = Counter()
            for row in selected.values():
                if row["metrics"]["exact_report"]:
                    continue
                answer = json.loads(row["answer"]) if row["metrics"]["valid_json"] else None
                target = row["case"]["target"]
                wrong = ([k for k in sorted(set(target) | set(answer))
                          if k not in target or k not in answer or canonical(target[k]) != canonical(answer[k])]
                         if isinstance(answer, dict) else ["non_object_report"])
                errors.update(wrong)
                bad.append(dict(role=role, arm=arm, case=row["case"], answer=row["answer"],
                                metrics=row["metrics"], wrong_fields=wrong))
            result["arms"][arm] = dict(cases=len(selected),
                **{k:sum(r["metrics"][k] for r in selected.values())
                   for k in ("valid_json", "exact_report", "decision_correct")},
                field_errors=dict(errors))
        result["original_to_arm_paired"] = {a: transitions(runs[role]["original"], runs[role][a]) for a in ARMS[1:]}
        result["original_answer_byte_reproduction"] = sum(r["answer"] == prior[k]["answer"] for k,r in runs[role]["original"].items())
        flips = runs[role]["current_policy_flip"]
        result["flip_wrong_decisions_equal_original_answer"] = sum(
            json.loads(r["answer"]).get("decision") == json.loads(runs[role]["original"][k]["answer"]).get("decision")
            for k,r in flips.items() if r["metrics"]["valid_json"] and not r["metrics"]["decision_correct"])
        summary[role] = result
    if summary["base"]["metadata"]["max_stress_prompt_tokens"] != summary["new_sft"]["metadata"]["max_stress_prompt_tokens"]:
        raise ValueError("Token length differs across models")
    summary["base_to_new_paired"] = {a:transitions(runs["base"][a],runs["new_sft"][a]) for a in ARMS}
    summary["limitations"] = [
        "24 reused contexts, 8 groups, correlated variants; not 96 independent held-out tasks.",
        "Structured report only, no model tool execution; exact_report is not Agent task success.",
        "Surface arm changes both layout and instruction position; history arm changes several factors.",
        "One model/adapter/seed; weak transfer does not alone establish classical overfitting.",
        "Recorded metadata verified; local machine cannot independently rehash absent GPU weights or retokenize."]
    return summary, bad


def main():
    summary, bad = analyze()
    folder = ROOT / "results/analysis/evidence_stress_v1"
    outputs = {"summary.json":json.dumps(summary, ensure_ascii=False, indent=2)+"\n",
               "bad_cases.jsonl":"".join(canonical(r)+"\n" for r in bad)}
    for name, value in outputs.items():
        if (folder/name).exists() and (folder/name).read_text() != value:
            raise ValueError("Refuse to overwrite changed analysis")
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        if not (folder/name).exists():
            (folder/name).write_text(value)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
