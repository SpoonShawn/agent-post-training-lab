# 阶段 059：混合 replay + repaired SFT 设计冻结

日期：2026-09-21  
状态：数据与协议已冻结，等待 SuperPOD 训练

阶段058显示 repaired targeted SFT 能提升 repaired/OOD，但旧 ID 有4条回归。因此下一轮不继续单独扩大 repaired 数据，而是从原始 full-SFT adapter 继续训练一个混合语料 adapter。

## 冻结配置

- 原始训练 replay：49 个旧 train groups，每组按 id 排序取前4条，共196条
- repaired train：12 groups，96条，全部保留
- 总轨迹：292条；不含任何 confirmation、probe 或 validation 数据
- 比例：196 条 original replay + 96 条 repaired train
- 初始化：`checkpoints/transaction_v1_full_sft/adapter`
- LoRA：r=16，alpha=32，dropout=0
- learning rate：3e-5
- max steps：160，batch=1，gradient accumulation=4，bf16，gradient checkpointing
- 随机种子：20260921

## 设计理由

原始 replay 保持普通 `apply`/`rollback` 的旧能力；repaired train 保持权限撤销、异步检查、幂等重试和证据报告的新能力。只从原始 full-SFT 初始化，避免把上一轮 targeted adapter 的偏差继续放大。

## 验收门槛

训练完成后固定评测：

- repaired ID/OOD：各32/32，不能低于阶段057
- 旧 v1 confirmation ID：不低于128/128
- 旧 v1 confirmation OOD：不低于265/400
- policy violations：保持0

若 repaired 仍保持满分且旧 ID 恢复，则该轮作为最终成功候选；若任一门槛失败，保留结果并分析，不修改 evaluator 迎合结果。

## 已完成的本地验证

`data/transaction_mixed_sft_v1/` 已生成并冻结，行数核对为292（196 replay、96 repaired）；训练脚本通过 Python 编译检查。提交：`0953703 Prepare mixed replay and repaired SFT`。

## 运行约束补录

SuperPOD 账户最多允许一个并行任务。因此本阶段必须按“训练 → repaired ID → repaired OOD → 旧 v1 ID → 旧 v1 OOD”串行提交和等待，不能同时提交四个评测作业。此前给出的并行评测示例不适用于该账户，后续命令已改为逐个运行。
