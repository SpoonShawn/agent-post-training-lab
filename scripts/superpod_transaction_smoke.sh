#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m scripts.prepare_transaction_v1
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "先申请GPU"; print(torch.cuda.get_device_name(0))'
mkdir -p logs/transaction_v1_smoke
for stage in preflight base train_smoke sft_smoke; do
    echo "Transaction engineering gate: $stage"
    python -u -m scripts.transaction_gpu_smoke --stage "$stage" 2>&1 | tee -a "logs/transaction_v1_smoke/$stage.log"
done
echo "Complete: engineering gate only, no full SFT/DPO/GRPO yet. Upload results/transaction_v1_smoke/."
