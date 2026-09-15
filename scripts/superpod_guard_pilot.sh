#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m scripts.run_guard_pilot --verify-only
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "先申请GPU"; print(torch.cuda.get_device_name(0))'
mkdir -p logs/guard_pilot_v1
python -u -m scripts.run_guard_pilot --role base --resume 2>&1 | tee -a logs/guard_pilot_v1/base.log
python -u -m scripts.run_guard_pilot --role sft --resume 2>&1 | tee -a logs/guard_pilot_v1/sft.log
