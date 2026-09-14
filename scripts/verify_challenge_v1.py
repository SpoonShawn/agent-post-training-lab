"""Verify frozen challenge, runtime, and optionally actual model/adapter bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_baseline import runtime_sha256, sha256_model_path
from scripts.prepare_challenge_v1 import build


def verify(model=None, adapter=None):
    folder = ROOT / "data/challenge_v1"
    manifest = json.loads((folder / "manifest.json").read_text())
    path = folder / "cases.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["cases_sha256"], "Changed cases"
    assert runtime_sha256() == manifest["runtime_sha256"], "Changed runtime"
    assert hashlib.sha256((ROOT / "scripts/prepare_challenge_v1.py").read_bytes()).hexdigest() == manifest["generator_sha256"], "Changed generator"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == build(), "Generated cases drift"
    assert len(rows) == len({r["id"] for r in rows}) == manifest["cases"], "Coverage/duplicate IDs"
    for value, key in ((model, "model_sha256"), (adapter, "adapter_sha256")):
        if value:
            print(f"Verifying {key} content (may take time)...", flush=True)
            assert sha256_model_path(str(value))[0] == manifest[key], f"Wrong {key}"
    print("Challenge verified: 120 cases; frozen protocol/model checks passed for supplied paths.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    args = parser.parse_args()
    verify(args.model_path, args.adapter_path)
