"""Generated-token-only, temperature-consistent and chunked GRPO scoring."""
import math


def token_chunks(prompt_ids, generated_ids, chunk_size=32, max_length=8192):
    if not prompt_ids or not generated_ids or chunk_size < 1:
        raise ValueError('Nonempty public prompt and completion required')
    if len(prompt_ids) + len(generated_ids) > max_length:
        raise ValueError('Context budget exceeded; evidence must not be truncated')
    # Logit at p-1 predicts first action token p. Tool/history tokens are never labels.
    return [(start, min(start+chunk_size, len(generated_ids)))
            for start in range(0, len(generated_ids), chunk_size)]


def scores(model, prompt_ids, generated_ids, start, stop, temperature):
    import torch
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid temperature')
    device=next(model.parameters()).device
    # No input past the last target needed. Only action-predicting positions hit lm_head.
    ids=torch.tensor([prompt_ids+generated_ids[:stop]], device=device)
    positions=torch.arange(len(prompt_ids)-1+start, len(prompt_ids)-1+stop, device=device)
    output=model(input_ids=ids, use_cache=False, logits_to_keep=positions)
    distribution=(output.logits[0].float()/temperature).log_softmax(-1)
    targets=torch.tensor(generated_ids[start:stop], device=device)
    logp=distribution.gather(-1, targets[:,None]).squeeze(-1)
    entropy=-(distribution.exp()*distribution).sum(-1)
    return logp, entropy


def objective(new, old, reference, advantage, clip=.2, beta=.01):
    import torch
    if new.shape != old.shape or new.shape != reference.shape or new.numel()==0:
        raise ValueError('Policy probability alignment changed')
    if not all(torch.isfinite(v).all() for v in (new, old, reference)):
        raise ValueError('Nonfinite policy probability')
    ratio=(new-old.detach()).exp()
    delta=reference.detach()-new
    kl=delta.exp()-delta-1
    value=torch.minimum(ratio*advantage, ratio.clamp(1-clip,1+clip)*advantage)-beta*kl
    return value, ratio, kl


def mean_std(values):
    if not values:
        return dict(count=0, mean=None, std=None)
    mean=sum(values)/len(values)
    return dict(count=len(values), mean=mean,
                std=math.sqrt(sum((x-mean)**2 for x in values)/len(values)))


def tool_stats(records):
    # Backend schema errors measure validity; transient/permission errors do not.
    events=[e for r in records for e in r['result']['events']]
    errors=[e['result'].get('error_type') for e in events]
    invalid=sum(v in ('unknown_tool','invalid_arguments') for v in errors)
    return dict(calls=len(events), invalid_calls=invalid,
        invalid_call_rate=invalid/len(events) if events else None,
        hallucinated_calls=sum(v=='unknown_tool' for v in errors))
