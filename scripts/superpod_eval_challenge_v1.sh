#!/usr/bin/env bash
# Inference only: run inside a GPU allocation. Never trains or selects a checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[1/4] Verify frozen data, runtime, base weights and existing adapter"
python -u scripts/verify_challenge_v1.py --model-path /home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507 --adapter-path checkpoints/pilot_v1_lora/adapter
echo "[2/4] Check CUDA"
python -u -c 'import torch; assert torch.cuda.is_available(), "Apply for GPU first"; print(torch.cuda.get_device_name(0), flush=True)'
mkdir -p logs/challenge_v1
echo "[3/4] Base inference: 120 diagnostic cases"
python -u scripts/run_baseline.py --model-path /home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507 --eval-path data/challenge_v1/cases.jsonl --max-new-tokens 512 --output-path results/baseline/challenge_v1_base.jsonl --resume 2>&1 | tee -a logs/challenge_v1/base.log
echo "[4/4] Existing LoRA inference: same 120 cases"
python -u scripts/run_baseline.py --model-path /home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507 --adapter-path checkpoints/pilot_v1_lora/adapter --eval-path data/challenge_v1/cases.jsonl --max-new-tokens 512 --output-path results/baseline/challenge_v1_sft.jsonl --resume 2>&1 | tee -a logs/challenge_v1/sft.log
echo "Inference complete. Exit the GPU allocation, then submit the two JSONL results."
