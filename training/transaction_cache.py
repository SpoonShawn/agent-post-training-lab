"""Disk-backed map dataset: bounded Python memory, no new dependency or truncation."""
import json
import sqlite3
from pathlib import Path


class DiskExamples:
    def __init__(self, path):
        self.path = Path(path)
        self.connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.size = self.connection.execute("SELECT COUNT(*) FROM examples").fetchone()[0]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if not 0 <= index < self.size:
            raise IndexError(index)
        return json.loads(self.connection.execute("SELECT value FROM examples WHERE id=?", (int(index),)).fetchone()[0])


def build_cache(path, examples, expected):
    """Rebuildable cache, atomically published; no overwrite of completed cache."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(".building")
    # A prior interrupted transaction has no scientific output; preserve it and use a new name.
    import tempfile
    descriptor, name = tempfile.mkstemp(prefix=temporary.name, dir=path.parent)
    import os
    os.close(descriptor)
    connection = sqlite3.connect(name)
    try:
        connection.execute("CREATE TABLE examples (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        count = total = supervised = maximum = 0
        for example in examples:
            length = len(example["input_ids"])
            connection.execute("INSERT INTO examples VALUES (?,?)", (count, json.dumps(example, separators=(",", ":"))))
            count += 1
            total += length
            maximum = max(maximum, length)
            supervised += sum(x != -100 for x in example["labels"])
        statistics = dict(examples=count, total_input_tokens_including_targets=total,
                          supervised_tokens=supervised, max_tokens=maximum)
        if statistics != expected:
            raise ValueError("Real tokenization differs from audited GPU preflight")
        connection.commit()
    finally:
        connection.close()
    os.rename(name, path)
    return DiskExamples(path)


def latest_checkpoint(folder):
    """Only complete Trainer checkpoints; partial directories remain untouched."""
    candidates = []
    for path in Path(folder).glob("checkpoint-*"):
        if not path.name.removeprefix("checkpoint-").isdigit():
            continue
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                    "adapter_model.safetensors", "adapter_config.json")
        if all((path / name).is_file() for name in required):
            state = json.loads((path / "trainer_state.json").read_text())
            step = int(path.name.split("-")[-1])
            if state["global_step"] == step:
                candidates.append((step, path))
    return str(max(candidates)[1]) if candidates else None
