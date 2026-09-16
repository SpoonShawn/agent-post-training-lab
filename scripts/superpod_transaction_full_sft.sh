#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/transaction_v1_full_sft
for stage in base train sft; do
    echo "Full transaction SFT: $stage"
    python -u -m scripts.transaction_full_sft --stage "$stage" 2>&1 | tee -a "logs/transaction_v1_full_sft/$stage.log"
done
echo "Full SFT first-seed run complete. Upload results/transaction_v1_full_sft/; DPO and GRPO are still pending."
