# 阶段 057：Repaired benchmark 配对基线与定向 SFT 对比

日期：2026-09-21  
状态：完成（本地配对分析，无新增 GPU 训练）

## 实验目标

判断阶段 055 的 repaired targeted SFT 是否真的改善了 Tool-Use 行为，而不是因为 repaired 测试集本身容易。使用同一份 frozen `transaction_repaired_v1` ID/OOD benchmark，分别评测原始 full-SFT adapter 和 repaired targeted-SFT adapter。

## 固定边界与数据

- protocol：`transaction_repaired_v1`
- ID：32 条，4 groups；OOD：32 条，4 groups
- 训练集：96 条，未进入 ID/OOD
- 五个 knowledge area：`state_revision`、`retry_idempotency`、`async_check`、`permission_recovery`、`evidence_report`
- 两个 adapter 使用同一 dataset SHA：`face8010251aff303acb9a711345004b783840e819d713dd2d94141e9fdbb922`
- 没有修改 evaluator、benchmark 或 reward，也没有把 probe 数据混入评测。

## 结果

| Adapter | Split | Task success | Answer correct | Execution success | Policy violations | Mean tool calls |
|---|---:|---:|---:|---:|---:|---:|
| Original full SFT | ID | 0/32 (0%) | 0/32 (0%) | 32/32 (100%) | 497 | 36.0 |
| Original full SFT | OOD | 0/32 (0%) | 0/32 (0%) | 32/32 (100%) | 494 | 36.0 |
| Repaired targeted SFT | ID | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) | 0 | 5.0 |
| Repaired targeted SFT | OOD | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) | 0 | 5.0 |

相对原始 SFT，targeted SFT 在 ID 和 OOD 均提升 100 个百分点，平均调用数减少 31 次，策略违规从约 15.5 次/条降为 0。两种 adapter 的执行器成功率均为 100%，因此差异来自模型的工具决策、停止条件和证据报告行为，而不是环境执行器故障。

## 具体 bad case

原始 SFT 在权限被撤销后已收到明确的 `permission_denied`，但没有停止，重复调用 `inspect_workspace` 和 `stage_config`，直到达到 36-call 上限，最终没有给出正确报告。

Repaired targeted SFT 在相同场景下执行 5 次调用：inspect → stage → validate → commit（权限错误）→ inspect，然后报告 `blocked`、最终配置、revision 和 tool error 数量。它没有声称事务成功，也没有继续修改。

## 原因分析

前一版训练集覆盖了普通成功事务，但没有充分覆盖“权限在事务中撤销后必须停止并报告”的组合条件；旧 full-SFT 因此学到了重复尝试的局部模式。repaired train 保持同一协议和 reward 边界，显式覆盖五类知识面及故障条件，targeted SFT 学到了失败后的终止和证据汇报模式。

## 结论边界

这是严格配对、可复现的正结果，但不能外推为所有旧 benchmark 或所有真实 GUI 场景都提升：

1. repaired benchmark 只有 64 条确认评测，规模仍小；
2. targeted SFT 与原始 full SFT 的训练数据不同，不能声称仅靠优化器或 evaluator 改善；
3. 需要保留原始 v1 ID/OOD 结果作为回归检查，确认没有旧能力退化；
4. 后续应在更大 held-out repaired challenge set 上复测，并报告各 knowledge area 分项结果。

## 下一步

先进行无 GPU 的本地汇总与回归检查，然后只申请一次短 GPU 任务评测原始 v1 confirmation ID/OOD，检查 repaired targeted SFT 是否破坏旧能力。若旧集无明显回归，再扩充 repaired challenge set；不立即增加训练轮数。
