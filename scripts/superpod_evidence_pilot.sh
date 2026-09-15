#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m scripts.prepare_evidence_pilot --verify
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "先申请GPU"; print(torch.cuda.get_device_name(0))'
mkdir -p logs/evidence_pilot_v1
for stage in preflight base old_sft smoke train new_sft; do
    python -u -m scripts.evidence_pilot_gpu --stage "$stage" 2>&1 | tee -a "logs/evidence_pilot_v1/$stage.log"
done
