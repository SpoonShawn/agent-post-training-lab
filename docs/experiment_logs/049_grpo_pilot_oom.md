# 049 — GRPO pilot OOM：前两次更新完成后显存耗尽

## 状态与证据

作业595207已完成temperature 1.2的两个update，随后在下一档temperature的第一次teacher-forced policy forward失败。日志报`torch.OutOfMemoryError`，GPU总79.18GiB、当时仅1.83GiB可用，单次尝试再分配4.09GiB；已分配72.68GiB、reserved3.93GiB。此前两个update的真实日志已写入`results/transaction_grpo_pilot_v1/updates.jsonl`，但整个pilot未完成。

## 已完成的部分

temperature 1.2 update 1：32 episodes、reward 1.0、std 0，优势全0，说明该batch仍饱和。update 2：32 episodes、reward 0.9375、std 0.2421、advantage std 0.5、policy KL 5.53e-6；出现2条失败，存在有效训练信号。两项都没有ID/OOD评测，不能作为最终GRPO效果。

## 原因分析

模型加载和LoRA训练参数已经通过；显存问题发生在teacher-forced scoring的完整词表logits与反向图叠加。原实现还默认保留KV cache，并在每个token评分后才释放中间对象，跨episode累积后触发峰值。不是reward或数据泄漏问题。

## 修复

teacher-forced forward显式`use_cache=False`，policy启用gradient checkpointing，逐turn删除logits/ratio/KL图并清空CUDA缓存；保留已上传日志，不覆盖历史统计。代码变化后升级到v3协议重新冻结。下一次仍需从未完成temperature开始，不能把两个update拼成完整pilot。

## 局限与下一步

本地无torch，无法模拟H800峰值。需要重新提交一个从头开始的pilot或先实现基于完整checkpoint的可靠resume；当前runner尚未安全恢复已完成update。任何重跑生成的结果必须写入新运行目录，避免重复行。
