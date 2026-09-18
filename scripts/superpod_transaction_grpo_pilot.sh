#!/usr/bin/env bash
set -euo pipefail
cd /home/zshaoaj/agent-post-training-lab
exec "$(command -v python)" -u -m scripts.transaction_grpo_pilot "$@"
