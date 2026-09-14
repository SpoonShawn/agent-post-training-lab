#!/usr/bin/env bash
# Run inside an allocated GPU shell. Resolve the repo relative to this script.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[1/7] Verify frozen training bundle"
python scripts/verify_training_bundle.py
echo "[2/7] Import PyTorch and check allocated CUDA GPU"
python -u -c 'print("Importing torch...", flush=True); import torch; print("Checking CUDA...", flush=True); assert torch.cuda.is_available(), "Apply for a GPU first"; assert torch.cuda.is_bf16_supported(), "bf16 GPU required"; print(torch.cuda.get_device_name(0), flush=True)'
mkdir -p logs/pilot_v1
echo "[3/7] Base confirmation (80 cases)"
python -u scripts/run_baseline.py --eval-path data/pilot_v1/confirmation_cases.jsonl --output-path results/baseline/pilot_v1_base_confirmation.jsonl --resume 2>&1 | tee -a logs/pilot_v1/base_confirmation.log
echo "[4/7] Base validation (80 cases)"
python -u scripts/run_baseline.py --eval-path data/pilot_v1/validation_cases.jsonl --output-path results/baseline/pilot_v1_base_validation.jsonl --resume 2>&1 | tee -a logs/pilot_v1/base_validation.log
echo "[5/7] Two-step GPU LoRA smoke"
python -u scripts/train_sft.py --max-steps 2 --output-dir checkpoints/pilot_v1_gpu_smoke 2>&1 | tee -a logs/pilot_v1/gpu_smoke.log
echo "[6/7] Full pilot LoRA training"
python -u scripts/train_sft.py --output-dir checkpoints/pilot_v1_lora 2>&1 | tee -a logs/pilot_v1/train.log
echo "[7/7] SFT validation and confirmation"
python -u scripts/run_baseline.py --adapter-path checkpoints/pilot_v1_lora/adapter --eval-path data/pilot_v1/validation_cases.jsonl --output-path results/baseline/pilot_v1_sft_validation.jsonl --resume 2>&1 | tee -a logs/pilot_v1/sft_validation.log
python -u scripts/run_baseline.py --adapter-path checkpoints/pilot_v1_lora/adapter --eval-path data/pilot_v1/confirmation_cases.jsonl --output-path results/baseline/pilot_v1_sft_confirmation.jsonl --resume 2>&1 | tee -a logs/pilot_v1/sft_confirmation.log
