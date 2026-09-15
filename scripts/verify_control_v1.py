"""CPU-only frozen control-study check (does not load model weights)."""
import json
from scripts.prepare_control_v1 import ROOT, build, digest, extension_hash
from scripts.run_baseline import runtime_sha256
from evaluation.protocol_v24 import validate_case


def verify():
    folder = ROOT / "data/control_v1"
    manifest = json.loads((folder / "manifest.json").read_text())
    rows = [json.loads(line) for line in (folder / "cases.jsonl").read_text().splitlines()]
    checks = (
        digest(folder / "cases.jsonl") == manifest["cases_sha256"],
        digest(ROOT / "data/challenge_v1/cases.jsonl") == manifest["source_cases_sha256"],
        runtime_sha256() == manifest["legacy_runtime_sha256"],
        extension_hash()[0] == manifest["extension_sha256"],
        rows == build(), len(rows) == len({r["id"] for r in rows}) == manifest["cases"],
    )
    if not all(checks):
        raise ValueError("Control-study inputs/code changed; do not mix runs")
    for row in rows:
        validate_case(row)
    return manifest, rows


if __name__ == "__main__":
    # Direct-script invocation needs repository root on sys.path before imports;
    # use python -m scripts.verify_control_v1 from the repository.
    manifest, rows = verify()
    print(f"Control study verified: {len(rows)} cases; no model loaded.")
