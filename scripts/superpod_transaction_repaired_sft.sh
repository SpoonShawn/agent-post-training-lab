#!/usr/bin/env bash
set -euo pipefail
cd /home/zshaoaj/agent-post-training-lab
python -u -m scripts.transaction_repaired_sft
mkdir -p results/transaction_repaired_v1
python -u -m scripts.evaluate_repaired_v1 --adapter-path checkpoints/transaction_repaired_sft_v1/adapter --split id --output-path results/transaction_repaired_v1/repaired_sft_id.jsonl
python -u -m scripts.evaluate_repaired_v1 --adapter-path checkpoints/transaction_repaired_sft_v1/adapter --split ood --output-path results/transaction_repaired_v1/repaired_sft_ood.jsonl
