#!/usr/bin/env bash
set -euo pipefail
cd /home/zshaoaj/agent-post-training-lab
python -u -m scripts.transaction_mixed_sft_v1 --model-path /home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507
