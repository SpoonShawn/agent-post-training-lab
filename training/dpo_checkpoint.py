"""Commit-marker checkpoints: incomplete writes never become resume candidates."""
import json
from scripts.prepare_control_v1 import digest
from scripts.transaction_gpu_smoke import sha256_adapter
from training.transaction_data import canonical


def seal(folder, state):
    (folder / "state.json").write_text(canonical(state)+"\n")
    value = dict(state_sha256=digest(folder/"state.json"), optimizer_sha256=digest(folder/"optimizer.pt"),
                 adapter_sha256=sha256_adapter(folder/"adapter"))
    (folder / "seal.json").write_text(canonical(value)+"\n")


def latest(folder, plan_hash):
    candidates = []
    for marker in folder.glob("checkpoint-*/seal.json"):
        path = marker.parent
        value = json.loads(marker.read_text())
        expected = dict(state_sha256=digest(path/"state.json"), optimizer_sha256=digest(path/"optimizer.pt"),
                        adapter_sha256=sha256_adapter(path/"adapter"))
        if value != expected:
            raise ValueError("Checkpoint integrity failed; preserve for diagnosis")
        state = json.loads((path / "state.json").read_text())
        if state["plan_sha256"] != plan_hash or type(state["step"]) is not int or not 0 < state["step"] <= 80:
            raise ValueError("Checkpoint belongs to a different protocol")
        candidates.append((state["step"], path.name, path, state))
    if not candidates:
        return None
    _, _, path, state = max(candidates)
    return path, state
