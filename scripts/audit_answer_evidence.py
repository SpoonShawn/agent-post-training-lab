"""Read-only-source evidence audit; never assigns semantic or task verdicts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_baseline import load_records
from evaluation.scoring import record_sha256


def audit(row):
    calls = [dict(step=t.get("step"), **e)
             for t in row["result"].get("trajectory", [])
             for e in t.get("tool_results", [])]
    errors = [e for e in calls if e["result"].get("ok") is False]
    negative_checks = [e for e in calls if e["result"].get("ok") is True
                       and isinstance(e["result"].get("result"), dict)
                       and e["result"]["result"].get("success") is False]
    observations = []
    for e in calls:
        if e["result"].get("ok") is not True:
            continue
        value = e["result"].get("result")
        if isinstance(value, dict):
            value = value.get("current_state", value)
            if "current_page" in value:
                observations.append(dict(step=e["step"], tool=e["tool_call"]["name"], state=value))
    # Deliberately narrow: only standalone, explicit numeric clauses.
    # Prose/negation/quotations/ranges are NOT assumed understood.
    text = (row["result"].get("final_answer") or "").replace("**", "")
    claims = []
    for clause in re.split(r"[。\n；]", text):
        clause = clause.strip().lstrip("- ").strip()
        match = re.fullmatch(
            r"(?:实际发生的失败次数\s*[:：]\s*(\d+)\s*(?:次)?|实际工具失败\s*(\d+)\s*次)",
            clause)
        if match:
            claims.append(int(next(v for v in match.groups() if v is not None)))
    observed = len(errors) + len(negative_checks)
    if not claims:
        count_status = "unparsed_needs_review"
    elif len(set(claims)) > 1:
        count_status = "conflicting_claims_needs_review"
    elif claims[0] != observed:
        count_status = "mismatch_needs_review"
    else:
        count_status = "matched_count_only"
    return dict(
        id=row["case"]["id"], record_sha256=record_sha256(row["case"], row["result"]),
        query=row["case"]["query"], answer=row["result"].get("final_answer"),
        tool_error_count=len(errors), negative_verification_count=len(negative_checks),
        observed_failure_count=observed, failure_evidence=errors + negative_checks,
        last_observed_state=observations[-1] if observations else None,
        recorded_final_state=row["result"].get("final_environment_state"),
        log_evidence=[e for e in calls if e["tool_call"]["name"] == "query_logs"],
        claimed_failure_counts=claims, count_audit_status=count_status,
        semantic_verdict="not_assigned", requires_semantic_review=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    if args.output_path.exists():
        raise ValueError("Refuse to overwrite existing output")
    rows = load_records(args.input_path, recompute=True)
    digest = hashlib.sha256(args.input_path.read_bytes()).hexdigest()
    results = [dict(source_sha256=digest, **audit(row)) for row in rows]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("x") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    from collections import Counter
    print(json.dumps(dict(records=len(results), statuses=dict(
        Counter(r["count_audit_status"] for r in results))), ensure_ascii=False))


if __name__ == "__main__":
    main()
