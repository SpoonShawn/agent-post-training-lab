# 041 — 多轮DPO计算实现与真实梯度检查准备

## 状态与目标

2026-09-17，本地实现/测试完成，GPU未运行；本机无torch及4B权重，不能声称完成真实反向传播。新增training/transaction_dpo.py、scripts/transaction_dpo_gate.py和冻结计划data/transaction_dpo_gate_v1.json。目标是验证多轮条件概率、mask、固定参考及实际梯度，避免“代码叫DPO，实际上训练目标不对”。

## 目标函数与工具边界

每条轨迹的logp为各assistant回合目标token（含EOS）的条件logp之和；每个回合保留该轨迹自己的历史工具返回作为输入，user/system/tool不作为预测标签。chosen/rejected不得交换工具观测。模型没有生成环境隐藏字段，DPO也不能监督它们。

loss = -log sigmoid(beta × [(logp_policy_chosen - logp_policy_rejected) - (logp_reference_chosen - logp_reference_rejected)])。

使用原始序列和，不做token平均。方法依据[DPO论文](https://arxiv.org/abs/2305.18290)，此处多轮工具条件概率接线是本项目实现，不宣称原论文已验证本环境。

参考分数在任何更新前，用原始完整SFT对全部98对计算并保存为不可变数值；后续不随policy变化，不需要同时常驻第二份4B。policy从同SFT adapter副本开始，只有LoRA参数可训练，不覆盖原SFT。PEFT可训练加载依据[官方接口](https://huggingface.co/docs/peft/package_reference/peft_model)。

为避免保留整条多轮计算图导致显存累积：先no_grad求当前chosen/rejected总logp及loss对差值的系数，再逐回合反向传播该系数×logp，最后只更新一次。只要参数在两遍之间不变、dropout关闭，这就是同一DPO目标的链式法则，不是逐回合SFT。GPU脚本先做float64直接autograd与两遍计算的梯度一致性测试、非动作logit位置mask测试，再启动真实模型。

## 配置与样本

固定beta0.1、AdamW lr5e-6、weight_decay0、clip1、dropout0、bf16基础模型、原LoRA配置、seed20260917。参考评分98对（80train/18dev）；按case hash选4个不同preference_train组做前后诊断，只在其中前2对各更新一次，共2步。18对dev只计算参考分数，不更新参数；原确认集不参与。每回合最大8192，拒绝截断。

这是**真实两步DPO工程检查**，不是正式完整DPO训练；gate adapter单独保存，不能继续当正式DPO起点。正式训练后续仍从原SFT重新开始，预算待本轮显存/梯度核验后固定。

## 指标、检查与结果边界

记录每对固定参考chosen/rejected logp、监督tokens、偏好分区；两步loss、参考校正margin、裁剪前gradient norm、4对更新后logp、显存、耗时及模型哈希。首步policy=reference时margin应接近0、loss接近log2；要求梯度有限且非零，不要求两步后所有指标都提高。reference文件和原SFT adapter前后hash不变。异常保留failed记录，不自动删除或静默重启。

本地以假tokenizer测试多轮历史/目标mask、数值有限差分、初始log2和极端logp稳定性，不能替代真实GPU测试。暂时没有DPO loss/显存实测，没有新的任务成功率，更不能说OOD提升。

提交前全套285项unittest通过（48.5秒），新增3项覆盖真实98对来源/分组、数值公式与完整多轮mask；数据/工程配置冻结和git diff --check通过。torch autograd自检代码尚未在本地执行，会在GPU入口先运行；不将CPU有限差分通过当成真实4B反向传播通过。

## 问题与勘误

前阶段提示“Base负例可能很容易、在SFT下概率低”是数据局限，但**低负例概率本身不能推出DPO梯度接近零**。policy与reference相同时，即使rejected logp=-5000，参考校正margin仍为0、loss=log2，对logp差的导数仍为-beta/2。这一数值反例已加入测试；真实参数梯度与后续饱和须实测。参考校正后margin大才可能出现sigmoid饱和，不能混淆。

正负轨迹长度偏差、小数据、来源标签相关及训练域饱和仍然存在，不用改测试标准解决。代码使用固定参考分数而不是误将更新后policy重复当reference；梯度检查点配合关闭dropout减少显存需求，真实峰值仍待GPU。

## 下一步与操作

需单GPU后台作业，申请1小时是上限，暂无实测耗时保证。本次主要是对已有轨迹打分和两次反向传播，不再跑784题交互评测：

```bash
cd /home/zshaoaj/agent-post-training-lab
git pull --ff-only
conda activate agent-post-training
mkdir -p logs/transaction_dpo_gate_v1
sbatch --account=mscaiesuperpod --partition=normal --gres=gpu:1 \
  --time=01:00:00 --job-name=txn-dpo-gate \
  --chdir=/home/zshaoaj/agent-post-training-lab \
  --output=logs/transaction_dpo_gate_v1/gate-%j.log --export=ALL \
  --wrap="exec \"$(command -v python)\" -u -m scripts.transaction_dpo_gate"
```

最终显示DPO engineering gate complete后上传：

```bash
cd /home/zshaoaj/agent-post-training-lab
git add results/transaction_dpo_gate_v1/run.json results/transaction_dpo_gate_v1/reference.json
git commit -m "Upload two-step transaction DPO gate"
git push origin main
```

若失败则上传存在的run.json并提供错误；不要删除原SFT、gate结果或冻结计划，不自行跳过检查。完成后核验数值/梯度，再推进正式DPO；完整路线的GRPO、多seed和独立后端尚未完成。
