#!/usr/bin/env bash
# Run INSIDE the allocated GPU shell after activating the experiment environment.
set -euo pipefail
cd /home/zshaoaj/agent-post-training-lab
python scripts/verify_training_bundle.py
python -c 'import torch; assert torch.cuda.is_available(), "Apply for a GPU first"; assert torch.cuda.is_bf16_supported(), "bf16 GPU required"; print(torch.cuda.get_device_name(0))'
mkdir -p logs/pilot_v1
# Keep a baseline before any fine-tuning. --resume is guarded by runtime/model hashes.
python -u scripts/run_baseline.py +  --eval-path data/pilot_v1/confirmation_cases.jsonl +  --output-path results/baseline/pilot_v1_base_confirmation.jsonl +  --resume 2>&1 | tee -a logs/pilot_v1/base_confirmation.log
python -u scripts/run_baseline.py +  --eval-path data/pilot_v1/validation_cases.jsonl +  --output-path results/baseline/pilot_v1_base_validation.jsonl +  --resume 2>&1 | tee -a logs/pilot_v1/base_validation.log
# Separate smoke adapter: never use it as the full experiment checkpoint.
python -u scripts/train_sft.py --max-steps 2 +  --output-dir checkpoints/pilot_v1_gpu_smoke 2>&1 | tee logs/pilot_v1/gpu_smoke.log
python -u scripts/train_sft.py +  --output-dir checkpoints/pilot_v1_lora 2>&1 | tee logs/pilot_v1/train.log
python -u scripts/run_baseline.py +  --adapter-path checkpoints/pilot_v1_lora/adapter +  --eval-path data/pilot_v1/validation_cases.jsonl +  --output-path results/baseline/pilot_v1_sft_validation.jsonl +  --resume 2>&1 | tee -a logs/pilot_v1/sft_validation.log
python -u scripts/run_baseline.py +  --adapter-path checkpoints/pilot_v1_lora/adapter +  --eval-path data/pilot_v1/confirmation_cases.jsonl +  --output-path results/baseline/pilot_v1_sft_confirmation.jsonl +  --resume 2>&1 | tee -a logs/pilot_v1/sft_confirmation.log
