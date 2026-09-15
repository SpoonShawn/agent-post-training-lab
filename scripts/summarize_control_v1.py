"""2.4-specific execution and paired-arm summary; no automatic answer approvals."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

from scripts.prepare_control_v1 import ROOT, digest
from scripts.verify_control_v1 import verify
from evaluation.protocol_v24 import evaluate_case


def load_records(path):
    manifest, cases = verify()
    expected = {c["id"]: c for c in cases}
    rows, seen, metas = [], set(), set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case, meta = row["case"], row["run_metadata"]
        if case["id"] in seen or case != expected.get(case["id"]):
            raise ValueError("Duplicate/changed/unknown case")
        if row.get("evaluator_version") != "2.4" or meta.get("evaluator_version") != "2.4":
            raise ValueError("Wrong scoring protocol")
        if (meta.get("extension_sha256") != manifest["extension_sha256"]
                or meta.get("runtime_sha256") != manifest["legacy_runtime_sha256"]
                or meta.get("benchmark_sha256") != manifest["cases_sha256"]
                or meta.get("model_content_sha256") != manifest["model_sha256"]):
            raise ValueError("Changed source provenance")
        if meta.get("role") not in {"base", "sft"}:
            raise ValueError("Missing model role")
        if meta.get("adapter_sha256") != (manifest["adapter_sha256"] if meta["role"] == "sft" else None):
            raise ValueError("Wrong adapter")
        if any(meta.get("generation", {}).get(k) != v for k,v in manifest["generation"].items()):
            raise ValueError("Changed generation settings")
        payload = {k:v for k,v in meta.items() if k != "fingerprint"}
        computed = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
        if (meta.get("fingerprint") != computed or meta.get("selected_case_count") != len(cases)
                or meta.get("study_manifest_sha256") != digest(ROOT / "data/control_v1/manifest.json")
                or meta.get("training_library_versions") != manifest["versions"]):
            raise ValueError("Invalid fingerprint, selection, manifest or package provenance")
        seen.add(case["id"])
        metas.add(json.dumps(meta, sort_keys=True))
        row["metrics"] = evaluate_case(case, row["result"])
        rows.append(row)
    if len(metas) > 1:
        raise ValueError("Mixed runs")
    return rows


def summarize(rows):
    arms = {}
    for row in rows:
        arm = arms.setdefault(row["case"]["control_arm"], {"cases": 0, "execution_pass": 0})
        arm["cases"] += 1
        arm["execution_pass"] += int(row["metrics"]["execution_success"])
    pairs = {}
    for left, right in (
        ("original_explicit", "reordered_explicit"),
        ("reordered_explicit", "reordered_checklist"),
        ("readonly_explicit", "readonly_checklist")):
        by_context = {}
        for row in rows:
            by_context.setdefault(row["case"]["context_id"], {})[row["case"]["control_arm"]] = row
        counts = Counter()
        for variants in by_context.values():
            if left in variants and right in variants:
                counts[f"{int(variants[left]['metrics']['execution_success'])}->{int(variants[right]['metrics']['execution_success'])}"] += 1
        pairs[left + " -> " + right] = dict(counts)
    return dict(coverage=f"{len(rows)}/76", complete=len(rows) == 76, arms=arms, pairs=pairs,
                task_success_rate=None, note="Execution only; successful answers require separate evidence-bound semantic review.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(load_records(args.input_path)), ensure_ascii=False, indent=2))
