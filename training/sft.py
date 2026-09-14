"""Match runtime chat serialization; supervise only the next assistant turn."""

import json
from copy import deepcopy
from pathlib import Path
from tools.tool_schema import TOOLS
from agent.baseline_runner import SYSTEM_PROMPT


def load_trajectories(path, expected_split):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or any(row["split"] != expected_split for row in rows):
        raise ValueError("Wrong or empty training split")
    for row in rows:
        if row["messages"][0] != {"role": "system", "content": SYSTEM_PROMPT} or row["tools"] != TOOLS:
            raise ValueError("Training/runtime prompt or tools mismatch")
    return rows


def encode_examples(rows, tokenizer, max_length):
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    examples = []
    for row in rows:
        messages = row["messages"]
        for i, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            prefix = tokenizer.apply_chat_template(
                messages[:i], tools=TOOLS, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer.encode(prefix, add_special_tokens=False)
            target = tokenizer.encode(message["content"], add_special_tokens=False) + [tokenizer.eos_token_id]
            if len(input_ids) + len(target) > max_length:
                raise ValueError(f"Overlength {row['id']} turn {i}; no silent truncation")
            examples.append({"input_ids": input_ids + target,
                             "attention_mask": [1] * (len(input_ids) + len(target)),
                             "labels": [-100] * len(input_ids) + target})
    if not examples:
        raise ValueError("No assistant supervision")
    return examples


class AssistantCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        import torch
        length = max(len(e["input_ids"]) for e in examples)
        output = {}
        for field, fill in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100)):
            output[field] = torch.tensor(
                [e[field] + [fill] * (length - len(e[field])) for e in examples], dtype=torch.long)
        return output
