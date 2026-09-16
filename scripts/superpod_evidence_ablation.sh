#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m scripts.prepare_evidence_ablation
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "先申请GPU"; print(torch.cuda.get_device_name(0))'
mkdir -p logs/evidence_ablation_v1
for stage in preflight base smoke train_fixed train_mixed fixed mixed; do
    echo "Evidence ablation stage: $stage"
    python -u -m scripts.evidence_ablation_gpu --stage "$stage" 2>&1 | tee -a "logs/evidence_ablation_v1/$stage.log"
done
echo "Complete: preserve checkpoints locally; upload results/evidence_ablation_v1/."
