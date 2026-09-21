#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zshaoaj/agent-post-training-lab
cd "$ROOT"
mkdir -p results/transaction_v1_regression
ADAPTER="$ROOT/checkpoints/transaction_repaired_sft_v1/adapter"
MODEL="/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507"
for split in confirmation_id confirmation_ood; do
  out="$ROOT/results/transaction_v1_regression/repaired_targeted_${split}.jsonl"
  if [[ -s "$out" ]]; then
    echo "SKIP existing $out"
    continue
  fi
  python -u scripts/evaluate_transaction_v1_regression.py \
    --model-path "$MODEL" --adapter-path "$ADAPTER" \
    --split "$split" --output-path "$out"
done
