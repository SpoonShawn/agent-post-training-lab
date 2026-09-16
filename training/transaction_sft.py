"""Next-assistant-turn supervision for the NEW schema, not the frozen pilot encoder."""
from agent.transaction_runtime import SYSTEM, TOOLS


def encode_trajectory(row, tokenizer, max_length=8192):
    if row["split"] not in ("train","dev"):
        raise ValueError("Confirmation trajectories must never enter SFT encoding")
    if row["tools"] != TOOLS or row["messages"][0] != dict(role="system",content=SYSTEM):
        raise ValueError("New task schema/system mismatch")
    if tokenizer.eos_token_id is None:
        raise ValueError("EOS required")
    for index,message in enumerate(row["messages"]):
        if message["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(row["messages"][:index],tools=TOOLS,
                                               add_generation_prompt=True,tokenize=False)
        prompt = tokenizer.encode(prefix,add_special_tokens=False)
        target = tokenizer.encode(message["content"],add_special_tokens=False)+[tokenizer.eos_token_id]
        if len(prompt)+len(target)>max_length:
            raise ValueError(f"Overlength {row['id']} assistant turn {index}; no truncation")
        yield dict(input_ids=prompt+target,attention_mask=[1]*(len(prompt)+len(target)),
                   labels=[-100]*len(prompt)+target)
