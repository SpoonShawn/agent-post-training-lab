"""Frozen-source paired execution analysis; no semantic verdicts or rescoring changes."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_challenge_v1 import verify
from scripts.summarize_baseline import load_records
from scripts.compare_pilot_v1 import replay, compact


def main():
    verify()
    manifest = json.loads((ROOT / "data/challenge_v1/manifest.json").read_text())
    cases = [json.loads(line) for line in (ROOT / "data/challenge_v1/cases.jsonl").read_text().splitlines()]
    expected = {c["id"]: c for c in cases}
    runs, metas, report, details = {}, {}, {}, []
    for mode in ("base", "sft"):
        path = ROOT / f"results/baseline/challenge_v1_{mode}.jsonl"
        rows = load_records(path, True)
        if len(rows) != len(expected) or {r["case"]["id"] for r in rows} != set(expected):
            raise ValueError("Incomplete/extra cases")
        meta = rows[0]["run_metadata"]
        if any(r["case"] != expected[r["case"]["id"]] or r["run_metadata"] != meta for r in rows):
            raise ValueError("Case or metadata drift")
        for field, key in (("model_content_sha256", "model_sha256"),
                           ("benchmark_sha256", "cases_sha256"), ("runtime_sha256", "runtime_sha256")):
            if meta[field] != manifest[key]:
                raise ValueError(f"Changed {field}")
        if any(meta["generation"].get(k) != v for k, v in manifest["generation"].items()):
            raise ValueError("Changed generation")
        if meta.get("adapter_sha256") != (manifest["adapter_sha256"] if mode == "sft" else None):
            raise ValueError("Wrong adapter")
        replayed = sum(replay(row) for row in rows)
        run = dict(compact(rows), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   run_metadata=meta, replayed_calls=replayed, groups={})
        for group in dict.fromkeys(r["case"]["group_id"] for r in rows):
            members = [r for r in rows if r["case"]["group_id"] == group]
            run["groups"][group] = dict(cases=len(members), passed=sum(r["metrics"]["execution_success"] for r in members))
        expression = {}
        for row in rows:
            if row["case"]["category"] == "expression":
                expression.setdefault(row["case"]["pair_id"], {})[row["case"]["expression_variant"]] = row
        run["expression_pairs"] = dict(Counter(
            f"{int(p['control']['metrics']['execution_success'])}->{int(p['rewrite']['metrics']['execution_success'])}"
            for p in expression.values()))
        rewritten = [p["rewrite"] for p in expression.values()]
        run["rewrite_evidence"] = dict(
            final_state_pass=sum(r["metrics"]["final_state_match"] for r in rewritten),
            process_pass=sum(r["metrics"]["process_success"] for r in rewritten),
            contract_pass=sum(r["metrics"]["public_contract_success"] for r in rewritten),
            logs_requirements_pass=sum(r["metrics"]["required_tool_results_accuracy"] == 1 for r in rewritten))
        for row in rows:
            if not row["metrics"]["execution_success"]:
                details.append(dict(model=mode, id=row["case"]["id"], query=row["case"]["query"],
                    final_answer=row["result"]["final_answer"],
                    final_state=row["result"]["final_environment_state"],
                    final_state_match=row["metrics"]["final_state_match"],
                    process_violations=row["metrics"]["process_violations"],
                    contract_violations=row["metrics"]["public_contract_violations"],
                    log_accuracy=row["metrics"]["required_tool_results_accuracy"],
                    trajectory=row["result"]["trajectory"]))
        report[mode], runs[mode], metas[mode] = run, {r["case"]["id"]: r for r in rows}, meta
    ignored = {"adapter_path", "adapter_sha256", "fingerprint"}
    if ({k:v for k,v in metas["base"].items() if k not in ignored}
            != {k:v for k,v in metas["sft"].items() if k not in ignored}):
        raise ValueError("Base/SFT metadata differ beyond adapter")
    report["paired_transitions"] = dict(Counter(
        f"{int(runs['base'][i]['metrics']['execution_success'])}->{int(runs['sft'][i]['metrics']['execution_success'])}"
        for i in expected))
    report["regression_ids"] = [i for i in expected if runs["base"][i]["metrics"]["execution_success"]
                                and not runs["sft"][i]["metrics"]["execution_success"]]
    report["limitations"] = [
        "Execution only; all successful answers await semantic review",
        "Post-training diagnostic, not independently sampled business tasks; expression controls reuse pilot",
        "18 composition groups; no IID significance or general overfitting diagnosis",
        "Wording-order perturbation conflates instruction-order sensitivity and logging-evidence timing",
        "Replay validates simulator consistency, not independent generation authenticity"]
    folder = ROOT / "results/analysis/challenge_v1"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (folder / "failures.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in details))
    print(json.dumps({m: {k:v for k,v in report[m].items() if k not in ("run_metadata", "groups")}
                      for m in ("base", "sft")}, ensure_ascii=False, indent=2))
    print("paired", report["paired_transitions"], "regressions", report["regression_ids"])


if __name__ == "__main__":
    main()
