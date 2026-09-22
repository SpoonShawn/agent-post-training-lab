# 阶段 060：混合 replay + repaired SFT 最终结果

日期：2026-09-22  
状态：完成，达到预设验收门槛

## 实际配置

- 初始化：原始 `transaction_v1` full-SFT adapter
- 训练集：196 条 original train replay + 96 条 repaired train，共292条
- LoRA：r=16，alpha=32，dropout=0
- learning rate：3e-5；max steps：160；batch=1；gradient accumulation=4；bf16
- confirmation、probe、validation 均未进入训练
- SuperPOD 按单任务串行完成训练和四份评测

## 最终结果

| 评测集 | 原始 Full SFT | Repaired Targeted SFT | Mixed SFT | 相对原始 |
|---|---:|---:|---:|---:|
| Repaired ID | 0/32 | 32/32 | 32/32 | +32 |
| Repaired OOD | 0/32 | 32/32 | 32/32 | +32 |
| 旧 v1 ID | 128/128 | 124/128 | 128/128 | 0 |
| 旧 v1 OOD | 258/400 | 265/400 | 272/400 | +14 |
| 旧 v1 合计 | 386/528 | 389/528 | 400/528 | +14 |

Mixed SFT 的旧 v1 总成功率为400/528=75.76%，原始 full-SFT 为386/528=73.11%，绝对提升2.65个百分点。旧 ID 恢复到128/128；旧 OOD提升到272/400，超过预设的265/400门槛。repaired ID/OOD维持32/32。四份结果的 policy violations 均为0，repaired 两集平均5次工具调用，旧 v1 ID/OOD平均15.75/17.48次。

逐条配对比较显示，旧 ID 无新增成功也无回归；旧 OOD 有16条从失败变成功、2条从成功变失败，净增14条。相比上一轮 targeted SFT，混合 replay 消除了4条旧 ID 回归，并把旧 OOD再提高7条。

## 结论

这轮满足预注册的联合验收条件，可以作为本项目的主结果：针对性 repaired 数据修复了权限撤销后的停止和证据报告，original replay 保留了普通 apply/rollback 能力，混合训练在旧 OOD 上获得净提升且没有旧 ID 回归。

结论范围仍限于本模拟 Tool-Use benchmark；不能外推为真实 GUI 或所有 Agent 任务的普遍提升。GRPO pilot 仍应作为失败/诊断实验保留，不能改写为成功原因。

## 剩余工作

不再申请 GPU。只需完成本地汇总、保存四份 JSONL 与 manifest/adapter 指纹，并准备面试材料中的方法、数字、bad case 和局限性说明。模型权重不提交 GitHub，需在 SuperPOD 的 checkpoint 路径或独立制品存储中保留。
