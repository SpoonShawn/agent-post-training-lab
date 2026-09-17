# 039 — 同任务跨策略候选准备

## 状态、目标、配置

2026-09-17，本地准备完成，GPU未运行，没有新的模型成绩。目标是在不动测试集、评分和SFT模型的前提下，检查真实Base失败能否为后续离线DPO提供候选对。运行scripts/transaction_base_preferences.py；计划data/transaction_base_preferences_v1.json冻结代码、来源、98个case和生成参数哈希。

## 数据与配对规则

沿阶段037完全相同的98个train case，49组各2题，Base每题greedy执行一次，工具环境/预算与前轮一致。不生成新任务、不增加OOD撤权或计数机制、不选择确认失败。SFT使用已保存的candidate0，明确不挑四候选中的“最好”者。

同case且完整初态一致才能配对；严格任务成功者优于失败者，不因模型名称预设胜者；两者同成同败均弃权，每题最多1对。因此最多98对，实际未知。报告覆盖率、保留/弃权率、错误类型、来源模型、概率/梯度后续门槛；不能把98对当成足够的大规模DPO数据。

## 问题与方案

旧probe392/392全部同质成功、0对，原始文件保持不变。新来源是跨策略离线候选，不是on-policy、不是hard-negative挖掘，不是GRPO。风险包括Base负例太容易、SFT对其概率接近零、正负长度不同；真实DPO前需验证目标函数和梯度信号，不保证提升。

读写采用独占锁、源指纹和完整轨迹重放校验，可从已完成条目续跑；不自动删坏行。旧完整SFT和评价代码不修改。此轮只加载原始Base，不需要更新LoRA、更不重新训练SFT。

## 验证和结果

本地配对测试覆盖成功胜失败、反向胜者、平局弃权和不同环境拒绝。原始392轨迹已重放；新98轨迹尚未生成。真实Base权重加载与耗时等待GPU，不做本地虚构测量。

提交前282项unittest全部通过（33.4秒），包含阶段038的完整候选重放及配对一致性核验。新协议本地冻结、git diff --check通过；旧数据/评分/冻结运行代码无修改。

## 下一步：登录节点提交一次

预计约15–40分钟，仅规划，申请1小时上限；排队另算。如超时可确认旧任务结束后重提，不改变协议。

```bash
cd /home/zshaoaj/agent-post-training-lab
git pull --ff-only
conda activate agent-post-training
mkdir -p logs/transaction_base_preferences_v1
sbatch --account=mscaiesuperpod --partition=normal --gres=gpu:1 \
  --time=01:00:00 --job-name=txn-base-prefs \
  --chdir=/home/zshaoaj/agent-post-training-lab \
  --output=logs/transaction_base_preferences_v1/base-%j.log --export=ALL \
  --wrap="exec \"$(command -v python)\" -u -m scripts.transaction_base_preferences"
```

结束后日志出现Base preference collection complete，base.jsonl应98行并生成pair_yield.json。即使0对也原样上传：

```bash
cd /home/zshaoaj/agent-post-training-lab
git add results/transaction_base_preferences_v1/base.jsonl results/transaction_base_preferences_v1/pair_yield.json
git commit -m "Upload train-only Base preference candidates"
git push origin main
```

随后审计实际候选对，准备多轮assistant-only DPO实现及小规模数值/梯度检查。此阶段不把“采集到了偏好”冒充“DPO已完成”，也不跳过既定多seed、消融和独立后端路线。
