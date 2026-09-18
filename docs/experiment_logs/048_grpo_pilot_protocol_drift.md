# 048 — GRPO pilot 第二次启动失败：修复后协议指纹未重新冻结

## 状态与证据

作业595203在6秒内失败，Slurm ExitCode `1:0`，MaxRSS 3976K。模型尚未加载。traceback为`ValueError: protocol drift`，位置是pilot runner的冻结协议比较。

## 原因

阶段047修复了LoRA参数冻结问题，导致`transaction_grpo_pilot.py`代码SHA变化；旧的`transaction_grpo_pilot_v1_protocol.json`仍记录修复前SHA。runner拒绝用不同代码继续同一个冻结协议，这是正确的完整性保护，但缺少协议版本升级步骤。

## 解决方案

将正式入口切换到新协议文件`data/transaction_grpo_pilot_v2_protocol.json`，保留v1协议和595203失败记录。训练数据、reward、temperature、group size、初始化SFT和评测边界均不变；变化仅是对已修复runner重新建立可追溯指纹。

## 结果与下一步

本次没有rollout、policy update或checkpoint。重新冻结v2后再提交单个Slurm作业。后续任何runner代码变更都必须先增加协议版本或在无结果时重新冻结，不能直接覆盖旧协议。
