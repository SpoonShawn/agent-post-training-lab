# 047 — GRPO pilot 启动失败：LoRA 参数被冻结

## 状态与证据

作业595095于2026-09-18运行54秒后失败，Slurm状态`FAILED`、ExitCode `1:0`，batch MaxRSS 609832K。日志显示两个模型权重均加载完成，随后在创建AdamW时失败：`ValueError: optimizer got an empty parameter list`。没有生成rollout、update、checkpoint或训练成绩。

## 原因

`PeftModel.from_pretrained`默认以推理模式加载adapter，将LoRA参数设置为`requires_grad=False`。pilot runner直接筛选可训练参数，结果为空。该问题是训练接线错误，不是显存不足、reward设计问题或模型退化。

## 修复

policy模型显式重新开启名字包含`lora_`的参数；reference模型全部冻结；优化器创建前加入空参数检查。原始失败日志保留，未伪造任何训练结果。

## 验证与下一步

本地环境没有torch，无法执行真实LoRA参数检查；新增代码为确定性启动修复。提交修复后重新提交单个Slurm作业。成功标准仍是6个update记录、6个adapter checkpoint和optimizer状态；若再次失败，保留新作业日志并继续按traceback修复。
