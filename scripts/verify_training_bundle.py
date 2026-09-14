"""Fail closed on drift before loading model weights."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify(root=ROOT):
    manifest = json.loads((root / "data/pilot_v1/manifest.json").read_text())
    for path, expected in {**manifest["files"], **manifest["code_sha256"]}.items():
        actual = hashlib.sha256((root / path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen bundle drift: {path}")
    groups = manifest["groups"]
    for a, b in (("train", "validation"), ("train", "confirmation"), ("validation", "confirmation")):
        if set(groups[a]) & set(groups[b]):
            raise ValueError("Group overlap")
    for split in ("train", "validation", "confirmation"):
        cases = [json.loads(line) for line in (root / f"data/pilot_v1/{split}_cases.jsonl").read_text().splitlines()]
        if len(cases) != manifest["counts"][split]["trajectories"]:
            raise ValueError("Case count drift")
        if {c["group_id"] for c in cases} != set(groups[split]) or any(c["split"] != split for c in cases):
            raise ValueError("Split identity mismatch")
    return manifest


if __name__ == "__main__":
    value = verify()
    print(json.dumps({"status": "verified", "counts": value["counts"]}, ensure_ascii=False, indent=2))
