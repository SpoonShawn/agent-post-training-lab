#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m scripts.evidence_stress_v1 --verify
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "先申请GPU"; print(torch.cuda.get_device_name(0))'
mkdir -p logs/evidence_stress_v1
python -u -m scripts.evidence_stress_v1 --role base 2>&1 | tee -a logs/evidence_stress_v1/base.log
python -u -m scripts.evidence_stress_v1 --role new_sft 2>&1 | tee -a logs/evidence_stress_v1/new_sft.log
