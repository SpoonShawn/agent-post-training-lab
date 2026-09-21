# 056 — repaired SFT首轮结果：32/32 ID与OOD通过

## 状态与证据

提交`3c6a4ac`上传了`repaired_sft_id.jsonl`和`repaired_sft_ood.jsonl`，各32行，覆盖完整。两份结果均来自targeted repaired SFT adapter；训练run元数据和模型权重未上传。当前尚未有同一repaired benchmark上的原SFT baseline，因此暂不计算提升百分点。

## 实测结果

| split | cases | task success | execution | answer | policy violations | mean tool calls |
|---|---:|---:|---:|---:|---:|---:|
| repaired ID | 32 | 32/32 | 32/32 | 32/32 | 0 | 5.0 |
| repaired OOD | 32 | 32/32 | 32/32 | 32/32 | 0 | 5.0 |

两个split的最终outcome均为blocked，覆盖state/revision、retry/idempotency、async check、permission/recovery和evidence/report标签中的实际子集。结果显示targeted adapter在修复后的机制上能够完成任务，但不能单独证明它优于原SFT。

## 仍需补齐的对照

使用同一repaired cases、同一generation参数和同一evaluator，运行原完整SFT adapter的32条ID与32条OOD。只有配对后才能报告改进/退化；不能拿旧benchmark的258/400直接作为本轮baseline。

## SuperPOD操作

```bash
cd /home/zshaoaj/agent-post-training-lab
git pull --ff-only
conda activate agent-post-training
mkdir -p logs/transaction_repaired_v1
sbatch --account=mscaiesuperpod --partition=normal --gres=gpu:1 \
  --time=02:00:00 --job-name=txn-repaired-base \
  --chdir=/home/zshaoaj/agent-post-training-lab \
  --output=logs/transaction_repaired_v1/base-%j.log --export=ALL \
  --wrap="exec bash scripts/superpod_evaluate_repaired_baseline.sh"
```
