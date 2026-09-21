#!/usr/bin/env bash
set -euo pipefail
cd /home/zshaoaj/agent-post-training-lab
mkdir -p results/transaction_repaired_v1
for split in id ood; do
  python -u -m scripts.evaluate_repaired_v1 \
    --adapter-path checkpoints/transaction_v1_full_sft/adapter \
    --split "$split" \
    --output-path "results/transaction_repaired_v1/original_sft_${split}.jsonl"
done
