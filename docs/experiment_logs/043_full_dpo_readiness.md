# 043 — 正式小规模DPO与闭环评测冻结

## 状态、目标和来源

2026-09-17，本地入口完成，正式GPU运行尚未开始。阶段042确认两步工程可用后冻结本计划，不依据确认集变化选择参数。新增scripts/transaction_full_dpo.py、training/dpo_checkpoint.py、启动脚本；计划data/transaction_full_dpo_v1.json钉住代码、偏好数据、初始SFT、固定参考、训练顺序和评测预算。目标是测量真实DPO对闭环任务的影响，包括无收益和退化。

## 配置与数据

原完整SFT（不是gate adapter）初始化，原80对preference_train、40组，固定哈希顺序一次epoch；每对一次优化，共80步。18对/9组preference_dev训练后仅计算loss/logp/margin，不选checkpoint；它们在SFT见过，不能称独立未见测试。

beta0.1、lr5e-6、AdamW weight_decay0、max_grad_norm1、dropout0、bf16基础模型、原LoRA33030144参数，seed20260917；sum assistant token logp，不做长度平均、不添加SFT混合损失。每回合user/system/tool内容仅作条件，目标mask沿已冻结编码器。两遍链式梯度算法沿gate，不改公式。固定最后80步模型，没有early stopping或确认集选优。

参考使用阶段041实际在更新前计算的98对初始SFT分数；使用其数值缓存，不使用gate模型权重。源文件hash、初始SFT权重hash、真实分词token数和首步初始margin做一致性检查。后续零梯度/饱和如实记录，不为凑80个“有效更新”改beta或重抽样；实际optimizer步数与梯度是否为零分别记录。

## 比较与指标

训练结束计算18对偏好开发分数，再用最终adapter完整交互：开发256、确认ID128、OOD400。重用原Base/SFT已保存结果，不再为它们推理。工具契约、分区、greedy/8192/512/40回合/36调用与此前一致。确认是既定基准上的后续模型比较，并非新的独立盲测；不可利用其结果回灌训练或选择参数。

关注严格任务成功、权限违规、报告错误、预算耗尽、成本和成对退化；训练loss不能代替这些指标。仍是单seed、80对跨策略小规模实验，不是大规模偏好对齐。原确认已查看的事实保持披露；若后续针对其失败改数据或奖励，必须另冻新确认集。

## 中断保护和资源

每10步保存adapter、优化器、CPU/CUDA RNG、轨迹顺序和step记录，最后写完整性seal；仅完整且hash匹配的checkpoint可恢复。未完成目录保留，不删除；损坏seal/文件拒绝继续。恢复后记录实际尝试和重复计算的步骤，不能把最后一段耗时当总成本。首次完整checkpoint前中断需诊断，不自动抹除记录。训练后评测按完整已写入条目续跑，半行JSON仍需诊断。

新结果results/transaction_full_dpo_v1，checkpoint在checkpoints/transaction_full_dpo_v1，与SFT/gate分开。使用独占锁防并发写。原SFT和参考缓存不覆盖。

规划申请6小时，预计训练与评测合计约3–6小时但不保证；主要时间可能是784题交互（此前SFT生成计时约3小时）。gate的4分钟包含参考打分，不能乘40直接当80步训练ETA。若DPO产生更长路径可能超时，依检查点和逐题结果续跑。保留约10GB磁盘用于本阶段checkpoint较稳妥，不自动清理旧实验。

## 验证与当前结果

CPU新增gate数学核验、固定训练顺序/开发排除、checkpoint完整/部分/错协议/篡改测试；全套288项unittest通过（64.6秒），Shell语法与git diff --check通过，正式协议已冻结。真实PyTorch优化器/RNG恢复仍待上机，不能以文件测试代替GPU恢复。正式训练/784题成绩目前不存在。

## SuperPOD操作

登录节点执行一次，不先srun：

```bash
cd /home/zshaoaj/agent-post-training-lab
git pull --ff-only
conda activate agent-post-training
mkdir -p logs/transaction_full_dpo_v1
df -h .
sbatch --account=mscaiesuperpod --partition=normal --gres=gpu:1 \
  --time=06:00:00 --job-name=txn-full-dpo \
  --chdir=/home/zshaoaj/agent-post-training-lab \
  --output=logs/transaction_full_dpo_v1/full-%j.log --export=ALL \
  --wrap="bash scripts/superpod_transaction_full_dpo.sh"
```

阶段先出现Formal DPO 1/80…80/80，再出现dpo/dev、dpo/confirmation_id、dpo/confirmation_ood。看到Full DPO first-seed training and evaluation complete并确认作业COMPLETED/0:0，再上传：

```bash
cd /home/zshaoaj/agent-post-training-lab
git add results/transaction_full_dpo_v1/training_run.json results/transaction_full_dpo_v1/dpo_*.jsonl
git commit -m "Upload full transaction DPO first-seed results"
git push origin main
```

若中断，先确认旧作业已结束，再提交同命令；不要并发重复任务，不删除checkpoint。拿到结果后审计三模型对照，继续在线GRPO准备与后续路线，不以本轮DPO完成冒充全项目结束。
