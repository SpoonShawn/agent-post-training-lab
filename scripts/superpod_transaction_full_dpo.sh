#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -u -m scripts.transaction_full_dpo --stage train
python -u -m scripts.transaction_full_dpo --stage infer
echo "Full DPO first-seed training and evaluation complete. Upload results/transaction_full_dpo_v1/."
