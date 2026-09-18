# 050 — GRPO runner redesign after repeated OOM

## 状态与证据

作业日志显示595207仍在完整词表teacher-forced forward申请2.22GiB时OOM；前两个temperature 1.2 update的中间结果保留，但不能组成完整pilot。复查发现v1 runner的显存修复不足，并存在实验接线风险。

## 发现的问题

1. v1仍对整段序列计算lm_head logits，无法保证峰值；`use_cache=False`只解决缓存，不解决完整词表输出。
2. policy未显式进入训练模式，gradient checkpointing的实际收益不可靠。
3. old behavior logprob在temperature处理后生成，但更新侧没有独立检查teacher-forced logprob的一致性。
4. 三个temperature连续使用同一个policy，temperature效果与训练顺序混杂。
5. tool validity统计在原runner中用固定值，不能作为真实指标。

## 修复设计

新增`training/grpo_update_v2.py`和`scripts/transaction_grpo_pilot_v2.py`：使用Qwen3 `logits_to_keep`只计算目标生成token，32 token分块；policy/reference共享一个基座并挂两个独立adapter；每个temperature从同一SFT adapter独立启动；policy显式train、reference冻结；更新前校验behavior与teacher-forced logprob差异；记录实际工具错误类型；先执行8192上下文的memory/gradient gate，再进入训练。

新pilot仍使用固定严格reward、24个train结构组、ID/OOD独立评测边界，probe和旧失败结果不回灌。v1、v2、v3及595207/595203/595095原始日志均保留。

## 验证与下一步

本地完成Python编译、24个train组/类别/边界检查。CPU没有PyTorch，无法运行GPU memory gate。SuperPOD上先执行freeze，再执行gate；gate通过后才训练三档独立arm。gate失败时不进入训练。
