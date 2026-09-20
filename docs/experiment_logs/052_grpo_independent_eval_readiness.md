# 052 — GRPO pilot独立ID/OOD评测准备

训练侧pilot结果已上传；本阶段只准备独立评测，尚未运行。固定三个temperature arm的step1/step2 adapter，共6个checkpoint；每个checkpoint分别运行confirmation_id和confirmation_ood。训练rollout、probe和评测输出分目录保存。

ID为128题，OOD为400题，两者不进入GRPO训练。SFT baseline使用此前冻结的结果文件，不覆盖原文件。评测脚本重新执行环境重放和严格task_success，保留每题完整轨迹；不会训练或挑选checkpoint。

输出路径：`results/transaction_grpo_eval_v2/{t12,t15,t18}/checkpoint-{1,2}/confirmation_{id,ood}.jsonl`。每个文件必须分别有128或400行。后续汇总task success、已有tool metrics、invalid/hallucinated/repeated calls、平均步骤、token和latency，并逐题比较退化。

## SuperPOD操作

```bash
cd /home/zshaoaj/agent-post-training-lab
git pull --ff-only
conda activate agent-post-training
mkdir -p logs/transaction_grpo_eval_v2
sbatch --account=mscaiesuperpod --partition=normal --gres=gpu:1 \
  --time=08:00:00 --job-name=txn-grpo-eval \
  --chdir=/home/zshaoaj/agent-post-training-lab \
  --output=logs/transaction_grpo_eval_v2/eval-%j.log --export=ALL \
  --wrap="exec bash scripts/superpod_evaluate_grpo_pilot_v2.sh"
```

完成后检查12个JSONL行数并提交评测结果；不要提交模型checkpoint。
