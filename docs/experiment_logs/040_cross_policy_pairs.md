# 040 — 跨策略候选核验与98对偏好冻结

## 状态与来源

2026-09-17，收到用户推送ecfc9e6，运行代码2e3051c。实际Base推理98条完成，原始结果在results/transaction_base_preferences_v1；此前392条SFT候选保留。新增scripts/prepare_transaction_dpo.py严格核验两侧重放/指纹/来源/配对方向，数据冻结在data/transaction_dpo_v1，manifest存完整源SHA。

## 目标、配置、数据

同98个既有train任务、49组各2题，Base greedy与之前SFT candidate0对照；同初态、同工具、同预算。没有使用确认任务或改写。固定SFT candidate0而非选择四次最高分。基础权重与生成协议沿前轮指纹。这里只收集离线偏好，不更新任何模型。

## 指标与结果

Base98/98记录核验，完整任务0/98、执行2/98、最终报告0/98；SFT候选98/98完整成功。因此98/98任务均有一成功一失败，配对98，弃权0。Base生成计时521.14秒（约8分41秒），不含排队、加载或审计，不等于完整作业耗时。全部Base正常生成final_answer，生成长度未触及512，当前版本可以按完整EOS编码；中断轨迹需单独处理，不能静默添加终止标签。

## Bad case与限制

所有98条rejected均保留完整模型输出和工具返回，没有人为改写错误答案。Base两条执行条件正确而报告失败也属于严格任务失败；chosen胜出依据冻结task_success，不是依据“SFT一定比Base好”。具体路径见pairs.json里的case_id/provenance及chosen/rejected。

本轮全部chosen来自SFT、rejected来自Base，存在来源与标签完全相关的局限，不能称为同策略困难负例。按result.turns计数，98/98对的chosen都比rejected更长；这是实际测得的长度混杂，不能忽略。后续记录监督token数和原始logp，不能将长度归一化悄悄混进标准序列DPO。98对很小，只足以做方法可行性探索，不足以承诺稳定OOD收益。

## 分区与原因

按group_id固定哈希，9组18对为preference_dev，其余40组80对为preference_train。结构组不跨偏好分区，两题同组不拆开。但二者原本都是SFT训练任务，因此preference_dev只对DPO更新留出，**不是SFT未见过的确认集**。原确认ID/OOD完全不进入这些偏好对。

## 验证与下一步

新增98对真实来源/方向重放、40/9结构组互斥、分区只来自原train测试。原来0对的同策略probe不覆盖、不改分，新增跨策略来源独立记录。下一阶段实现assistant-only多轮DPO及GPU两步数值/梯度检查；不是DPO已完成，更不是GRPO。完整测试数见阶段041同批记录。
