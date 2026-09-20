#!/usr/bin/env bash
set -euo pipefail
cd /home/zshaoaj/agent-post-training-lab
MODEL_PATH="${MODEL_PATH:-/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507}"
for arm in t12 t15 t18; do
  for step in 1 2; do
    adapter="checkpoints/transaction_grpo_pilot_corrected_v2/${arm}/checkpoint-${step}/adapter"
    for split in confirmation_id confirmation_ood; do
      output="results/transaction_grpo_eval_v2/${arm}/checkpoint-${step}/${split}.jsonl"
      if [[ -s "$output" ]]; then
        echo "preserving existing $output"
        continue
      fi
      python -u -m scripts.evaluate_grpo_pilot_v2 --model-path "$MODEL_PATH" --adapter-path "$adapter" --split "$split" --output-path "$output"
    done
  done
done
echo "GRPO independent ID/OOD evaluation complete"
