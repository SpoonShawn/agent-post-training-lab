# 055 — repaired benchmark与targeted SFT准备

## 设计目的

旧benchmark把`revoke_before_commit`完全留给OOD：train为0、OOD为16。它不能单独回答后训练泛化问题。本阶段新增`transaction_repaired_v1`，共享权限撤销机制但保持group和具体场景组合独立，旧数据与旧分数全部保留。

## 数据规模与覆盖

新数据160题：train96题/12组、ID32题/4组、OOD32题/4组，每组8个配置变体。所有split都包含`revoke_before_commit=true`，具体配置、revision、重试数、检查延迟和知识标签组合不重复。标签覆盖state/revision、retry/idempotency、async check、permission/recovery、evidence/report五个知识面。train只生成oracle轨迹；ID/OOD不进入训练。

## 训练计划

从原完整SFT adapter初始化，targeted SFT最多128步，学习率5e-5、batch1、累积4、LoRA r16、dropout0、bf16、gradient checkpointing。目标是先验证数据修复能否让模型学习共享故障机制；这是新协议，不能和旧SFT/DPO/GRPO成绩混成一条曲线。

## 成功标准

repaired OOD相对repaired SFT baseline提升至少10个百分点；权限撤销违规不增加；evidence/report准确率提高；普通事务能力不退化。若targeted SFT仍无提升，保留为数据覆盖修复失败，不继续盲目追加GRPO。

## 本地验证

`python3 -m scripts.prepare_transaction_repaired_v1`已生成冻结数据；`python3 -m unittest tests.test_transaction_repaired_v1 -q`通过；编译检查通过。尚未加载GPU模型，结果不能预填。

## SuperPOD运行

```bash
cd /home/zshaoaj/agent-post-training-lab
git pull --ff-only
conda activate agent-post-training
mkdir -p logs/transaction_repaired_v1
sbatch --account=mscaiesuperpod --partition=normal --gres=gpu:1 \
  --time=04:00:00 --job-name=txn-repaired-sft \
  --chdir=/home/zshaoaj/agent-post-training-lab \
  --output=logs/transaction_repaired_v1/sft-%j.log --export=ALL \
  --wrap="exec bash scripts/superpod_transaction_repaired_sft.sh"
```

完成后上传`results/transaction_repaired_v1/*.jsonl`和`checkpoints/transaction_repaired_sft_v1/training_run.json`，不要上传模型权重。
