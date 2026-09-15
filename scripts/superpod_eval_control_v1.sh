#!/usr/bin/env bash
# Run only after allocating a GPU. No training or checkpoint selection.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[1/4] Verify frozen 2.4 control study"
python -m scripts.verify_control_v1
echo "[2/4] Check CUDA"
python -u -c 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "Apply for bf16 GPU first"; print(torch.cuda.get_device_name(0), flush=True)'
mkdir -p logs/control_v1
echo "[3/4] Base: 76 control cases, no training"
python -u -m scripts.run_control_v1 --role base --resume 2>&1 | tee -a logs/control_v1/base.log
echo "[4/4] Existing LoRA: same 76 cases, no training"
python -u -m scripts.run_control_v1 --role sft --resume 2>&1 | tee -a logs/control_v1/sft.log
python -m scripts.summarize_control_v1 --input-path results/baseline/control_v1_base.jsonl
python -m scripts.summarize_control_v1 --input-path results/baseline/control_v1_sft.jsonl
echo "Both runs complete. Exit GPU allocation; submit the two result files."
