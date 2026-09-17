"""Trajectory DPO: sum conditional assistant log-probabilities, mask tool observations."""
import json
import math

from agent.transaction_runtime import SYSTEM, TOOLS
from training.transaction_sft import encode_trajectory


def trajectory_row(case_id, result):
    messages = [dict(role="system", content=SYSTEM), dict(role="user", content=result["query"])]
    for turn in result["turns"]:
        messages.append(dict(role="assistant", content=turn["model_output"]))
        if "result" in turn:
            messages.append(dict(role="tool", content=json.dumps(turn["result"], ensure_ascii=False, sort_keys=True)))
    if result["terminated_reason"] != "final_answer" or not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Only complete final-answer trajectories supported by this version")
    return dict(id=case_id, split="train", tools=TOOLS, messages=messages)


def encode_pair(pair, tokenizer):
    # preference_dev is for scoring only; callers select preference_train for updates.
    return {side: list(encode_trajectory(trajectory_row(pair["case_id"], pair[side]), tokenizer))
            for side in ("chosen", "rejected")}


def dpo_terms(chosen, rejected, ref_chosen, ref_rejected, beta=.1):
    z = beta * ((chosen-rejected)-(ref_chosen-ref_rejected))
    loss = max(-z, 0) + math.log1p(math.exp(-abs(z)))
    sigmoid_negative = math.exp(-z)/(1+math.exp(-z)) if z >= 0 else 1/(1+math.exp(z))
    return loss, -beta*sigmoid_negative, z


def turn_logp(model, example):
    import torch
    ids = torch.tensor([example["input_ids"]], device=model.device)
    labels = torch.tensor(example["labels"][1:], device=model.device)
    keep = labels != -100
    if not keep.any():
        raise ValueError("No assistant targets")
    # Gather supervised positions BEFORE float32 softmax; avoid context*vocabulary FP32 copies.
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, :-1]
    selected = logits[keep].float()
    target = labels[keep]
    return torch.log_softmax(selected, dim=-1).gather(-1, target[:, None]).sum()


def trajectory_logp(model, examples):
    import torch
    with torch.no_grad():
        return sum(float(turn_logp(model, e)) for e in examples)


def backward_pair(model, encoded, reference, beta=.1):
    """Exact chain rule with two deterministic passes; one graph at a time.

    No optimizer update between passes; all dropout must be disabled. No token averaging.
    Reference values are precomputed under unchanged initial SFT and never differentiated.
    """
    chosen = trajectory_logp(model, encoded["chosen"])
    rejected = trajectory_logp(model, encoded["rejected"])
    loss, coefficient, margin = dpo_terms(chosen, rejected, reference["chosen"], reference["rejected"], beta)
    for side, sign in (("chosen", 1), ("rejected", -1)):
        for example in encoded[side]:
            (sign * coefficient * turn_logp(model, example)).backward()
    return dict(loss=loss, coefficient=coefficient, reference_adjusted_margin=margin,
                chosen_logp=chosen, rejected_logp=rejected)


def torch_self_test():
    """Float64 direct autograd versus exact two-pass chain rule; tool mask has no gradient."""
    import torch
    weights = torch.tensor([.3, -.2, .7, .1], dtype=torch.float64, requires_grad=True)
    c, r = weights[:2].sum(), weights[2:].sum()
    reference = (-1.2, -2.3)
    direct = -torch.nn.functional.logsigmoid(.1*((c-r)-(reference[0]-reference[1])))
    direct.backward()
    expected = weights.grad.clone()
    loss, coefficient, _ = dpo_terms(float(c.detach()), float(r.detach()), *reference)
    weights.grad = None
    for index, sign in enumerate((1, 1, -1, -1)):
        (coefficient*sign*weights[index]).backward()
    if not torch.allclose(weights.grad, expected, atol=1e-12) or abs(loss-float(direct.detach())) > 1e-12:
        raise ValueError("DPO chain-rule gradient mismatch")
    logits = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([-100, 2, -100, 3])
    keep = labels != -100
    logp = torch.log_softmax(logits[keep], -1).gather(-1, labels[keep, None]).sum()
    logp.backward()
    if torch.count_nonzero(logits.grad[~keep]) != 0:
        raise ValueError("Tool/prompt position leaked into action logp")
    return dict(direct_vs_two_pass_gradient=True, non_action_logit_mask=True)
