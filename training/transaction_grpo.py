"""Outcome reward, group advantages, and auditable GRPO update primitives."""
import math
from agent.transaction_tasks import evaluate


def strict_reward(case, result):
    # Recompute from executable evidence, never trust the model's reported success.
    metrics = evaluate(case["environment"], case["desired"], result["events"], result["final_answer"])
    return float(metrics["task_success"]), metrics


def advantages(rewards, epsilon=1e-8):
    if len(rewards) < 2 or any(type(r) not in (float, int) or not math.isfinite(r) or r not in (0,1) for r in rewards):
        raise ValueError("At least two finite binary outcome rewards required")
    mean = sum(rewards)/len(rewards)
    std = math.sqrt(sum((r-mean)**2 for r in rewards)/len(rewards))
    return dict(rewards=rewards, mean=mean, population_std=std, zero_variance=std==0,
                advantages=[0.]*len(rewards) if std==0 else [(r-mean)/(std+epsilon) for r in rewards])


def clipped_objective(new_logp, old_logp, reference_logp, advantage, clip_range=.2, beta=.01):
    """Return scalar per-token GRPO objective terms.

    Inputs are one-dimensional tensors containing only generated assistant tokens.
    The KL estimator is exp(ref-current) - (ref-current) - 1, matching the
    published GRPO estimator. No tool-observation tokens may be passed here.
    """
    import torch
    if not (new_logp.shape == old_logp.shape == reference_logp.shape):
        raise ValueError("Policy log-probability shapes differ")
    if new_logp.numel() == 0 or not torch.isfinite(new_logp).all():
        raise ValueError("Invalid policy log probabilities")
    ratio = torch.exp(new_logp - old_logp.detach())
    adv = torch.as_tensor(advantage, dtype=new_logp.dtype, device=new_logp.device)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
    kl = torch.exp(reference_logp.detach() - new_logp) - (reference_logp.detach() - new_logp) - 1.0
    return torch.minimum(unclipped, clipped) - beta * kl, ratio.detach(), kl.detach()


def aggregate_update_stats(rewards, advantages_, ratios, kls, entropies, lengths, valid_calls, successes):
    """JSON-safe diagnostics; denominators are explicit for the experiment log."""
    import torch
    def mean(values):
        return float(torch.as_tensor(values, dtype=torch.float64).mean()) if values else 0.0
    def std(values):
        return float(torch.as_tensor(values, dtype=torch.float64).std(unbiased=False)) if values else 0.0
    return dict(reward_mean=mean(rewards), reward_std=std(rewards), advantage_mean=mean(advantages_),
                advantage_std=std(advantages_), policy_ratio_mean=mean(ratios), policy_kl_mean=mean(kls),
                entropy_mean=mean(entropies), completion_tokens_mean=mean(lengths),
                valid_call_rate=mean(valid_calls), task_success_rate=mean(successes),
                denominators=dict(episodes=len(rewards), generated_tokens=len(ratios), groups=None))
