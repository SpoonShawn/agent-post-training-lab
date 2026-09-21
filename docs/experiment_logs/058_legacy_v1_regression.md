# 阶段 058：Repaired Targeted SFT 的旧 v1 回归检查

日期：2026-09-21  
状态：完成

## 实验目标

检查阶段 057 的 repaired targeted SFT 在旧 `transaction_v1` confirmation 集上的回归，判断 repaired 训练是否只改善新故障模式，或同时破坏已有事务能力。

## 配置与数据

- 模型：`Qwen3-4B-Instruct-2507`
- adapter：`checkpoints/transaction_repaired_sft_v1/adapter`
- 对照：`results/transaction_v1_full_sft/sft_confirmation_*.jsonl`
- ID confirmation：128 条
- OOD confirmation：400 条
- 旧 v1 benchmark、evaluator 和 reward 均未修改

## 结果

| Split | 原始 Full SFT | Repaired Targeted SFT | 净变化 | 新增成功 | 回归 |
|---|---:|---:|---:|---:|---:|
| ID | 128/128 (100%) | 124/128 (96.88%) | -4 条 | 0 | 4 |
| OOD | 258/400 (64.50%) | 265/400 (66.25%) | +7 条 | 16 | 9 |
| 合计 | 386/528 (73.11%) | 389/528 (73.67%) | +3 条 | 16 | 13 |

其他观察：ID execution success 为 124/128，OOD execution success 为 393/400；两部分 policy violation 均为 0。OOD 的 16 条新增成功来自旧模型失败、targeted SFT 成功的样本，9 条旧成功样本回归。ID 的4条回归全部属于 `apply` 类事务。

## 结论

方向是正确的，但当前训练不是无回归的全面提升：

1. 阶段057的 repaired ID/OOD 从 0/32 提升到 32/32，证明新增故障/证据数据能教会模型正确停止和报告；
2. 旧 OOD 从258/400提升到265/400，新增成功16条，高于回归9条，说明新行为对部分旧分布确有迁移；
3. 旧 ID 从128/128降到124/128，存在可见回归，说明训练集仍偏向 repaired 故障模式，普通 apply 事务保持不足；
4. 总体 528 条只净增3条（73.11%→73.67%），因此不能宣称全面大幅提升。

## 原因分析

targeted SFT 只使用 repaired train，训练信号集中在权限变化、失败后停止和证据报告。旧 ID 的 apply 事务是原始 full-SFT 已完全掌握的能力，targeted 更新可能改变了部分成功提交路径；OOD 的净提升说明新行为并非纯粹过拟合，但训练分布覆盖仍不平衡。

## 下一步方案

下一轮采用混合训练：保留 repaired train 全量，同时加入经过指纹固定的原始成功事务 replay，给普通 `apply`、`rollback` 和权限恢复各设置明确比例。先在本地构造不泄漏的混合训练清单并做 manifest/test，再申请一次短 SFT 训练；验收同时要求 repaired benchmark 维持 100%、旧 ID 不低于 128/128、旧 OOD 不低于265/400。
