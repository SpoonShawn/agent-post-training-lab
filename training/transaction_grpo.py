"""Outcome reward and group advantage primitives; no optimizer or fake RL updates."""
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
