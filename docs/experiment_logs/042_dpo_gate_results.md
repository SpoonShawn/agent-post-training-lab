# 042 — 两步真实DPO检查实测

## 状态与来源

2026-09-17，用户上传2327f0c，运行代码d314873。原始run.json/reference.json位于results/transaction_dpo_gate_v1，新增审计scripts/analyze_transaction_dpo_gate.py与results/analysis/transaction_dpo_gate_v1/summary.json。此阶段真实GPU进行了两次DPO更新，不是正式全量DPO，也没有任务闭环评测。

## 目标、配置与数据

沿阶段041冻结配置，原完整SFT初始化，LoRA33030144可训练参数；H800、beta0.1、AdamW lr5e-6、clip1、dropout0。98对初始SFT参考分数在更新前缓存，80训练/18偏好开发全部覆盖；仅固定两条preference_train各更新一步，另两条train探针仅计算更新后概率。确认集未参与。数据来源、代码、配置、case顺序与参考hash均通过审计。

## 结果与指标

首步loss0.69314718056=log2，参考校正margin0，梯度范数36.4877；第二步在另一训练pair上的loss0.353592、margin0.857611、梯度范数29.7337。不是同一道题的两点学习曲线。两步梯度均有限且非零，上传torch float64直接autograd/链式梯度一致性与非动作logit mask检查均true。

四条固定探针对照初始同一pair：loss从各自log2降至0.1350、0.1393、0.1722、0.1996，margin约1.51–1.93。rejected logp分别下降19.3436、19.0095、16.7146、15.1016；chosen logp变化约-0.0000244、+0.00000131、+0.00000560、-0.00000322。**改进主要是压低错误轨迹概率，不是证明正确行为或OOD成功率提高**。两条chosen概率略降也保留。

GPU窗口07:00:15–07:04:12UTC，236.67秒，包含加载/参考打分/两步更新/保存，不是两步训练耗时本身。峰值allocated9.8138GiB、reserved25.5527GiB；不是GPU总占用。98对参考完整，2步更新目标一致，4条探针完整，loss/coefficient/margin可从上传logp在CPU重算一致。

## 问题、原因及解决

初始SFT对chosen非常确信，继续提升其概率空间小；当前测量支持偏好差距主要来自压低rejected，不能据此推断任务改善。小数据、来源与标签相关、98/98成功轨迹更长等偏差不消失。保留严格任务评估、冻结最终模型选择，正式DPO后再判断是否有收益或退化。

低rejected概率不会自动消除初始DPO梯度，实际非零梯度验证了阶段041的公式说明。固定参考记录不变；本地只核验元数据/数学一致性，未获取adapter本体、未本地重算模型概率或autograd。上传数值测试仅覆盖玩具图，真实4B两遍梯度没有与保存全部轨迹计算图的版本逐元素对比，不扩大验证结论。

## 下一步

工程检查通过，按阶段043从原SFT重新开始80对/80步正式小规模DPO，明确不继续gate adapter。完成后同预算784题评测；DPO是否有效尚未知，GRPO/多seed/独立后端未完成。GPU gate产物和失败历史永久保留，评分不修改。
