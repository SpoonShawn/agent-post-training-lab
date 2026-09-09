import json
from pathlib import Path


PATH = Path(
    "results/baseline/qwen3_4b_baseline.jsonl"
)

records = []

with open(
    PATH,
    "r",
    encoding="utf-8",
) as f:
    for line in f:
        records.append(
            json.loads(line)
        )

normal_cases = [
    r
    for r in records
    if "ordered_tool_match"
    in r["metrics"]
]

ordered_acc = (
    sum(
        r["metrics"]["ordered_tool_match"]
        for r in normal_cases
    )
    / len(normal_cases)
    if normal_cases
    else 0
)

set_acc = (
    sum(
        r["metrics"]["tool_set_match"]
        for r in normal_cases
    )
    / len(normal_cases)
    if normal_cases
    else 0
)

invalid = sum(
    r["metrics"]["invalid_tool_calls"]
    for r in records
)

failed = sum(
    r["metrics"]["failed_tool_calls"]
    for r in records
)

unknown = sum(
    r["metrics"]["unknown_tool_calls"]
    for r in records
)

avg_steps = (
    sum(
        r["metrics"]["num_steps"]
        for r in records
    )
    / len(records)
)

print(
    "=== Qwen3-4B Zero-shot Baseline ==="
)

print(
    "Cases:",
    len(records),
)

print(
    f"Ordered Tool Accuracy: "
    f"{ordered_acc:.3f}"
)

print(
    f"Tool Set Accuracy: "
    f"{set_acc:.3f}"
)

print(
    "Invalid Tool Calls:",
    invalid,
)

print(
    "Failed Tool Executions:",
    failed,
)

print(
    "Unknown Tool Calls:",
    unknown,
)

print(
    f"Average Agent Steps: "
    f"{avg_steps:.2f}"
)
