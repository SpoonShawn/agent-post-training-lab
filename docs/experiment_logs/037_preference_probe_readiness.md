# 037 — 训练侧偏好采样可行性检查

## 状态与目标

2026-09-17，本地准备完成，实际GPU采样未运行。目标是在训练侧测量真实SFT候选轨迹能否形成可靠偏好对，而不是立即宣布进入DPO训练；没有模型新成绩。前置模型为阶段036固定最后SFT，绝不加载确认轨迹作为训练样本。

## 配置、数据和边界

从原train的49组各按固定hash选2个case，共98个任务，每题4个独立初态rollout，合计392。候选种子按case_id/candidate稳定生成，逐episode重置；模型只见公开请求/工具历史，看不到故障计划。新脚本scripts/transaction_preference_probe.py与计划data/transaction_preference_probe_v1.json冻结来源、采样表hash和adapter哈希。

temperature0.8/top_p0.95/top_k50、do_sample=true、num_beams1、repetition_penalty1；上下文8192、输出512、40回合/36调用，其余模型默认参数继承固定权重包。依据[Transformers生成文档](https://huggingface.co/docs/transformers/main_classes/text_generation)。与正式greedy评测用途不同，不能把采样率和确认成功率直接比较。没有优化器、没有参数更新，不是DPO或GRPO。

## 指标与预先固定判定

报告392条覆盖率、98任务覆盖率、实际候选多样性、严格成功率、每任务有无成功与失败两类候选。只在同case四候选中选首个严格通过为chosen、首个未通过为rejected，每题最多一对；双方均成功/均失败全部弃权，不以更短失败胜出。记录候选来源hash，工具观测后续只能作条件，不能在DPO loss中当预测标签；实际DPO encoder/参考模型训练尚待实现与测试。

## 问题、风险和解决

SFT在模板域可能高度饱和，随机采样也可能全对，造成0可用偏好对。这是可接受的可行性负结果；不得为了训练“看起来在进行”伪造负例或偷用确认失败。此轮只测供给再决定训练方案。采样计划不专挑本轮OOD失败、不增加其改写。后续若因此调整确认涉及的机制，必须另设新确认。

长任务改sbatch；脚本用独占文件锁避免两个probe并发写同文件，按冻结顺序和每题seed续跑，完整记录重放核验。GPU随机采样仍不保证跨设备bitwise一致。仅剩截断JSON行等异常需人工保留与诊断。

## 验证、结果和下一步

CPU测试：训练侧分区/49组98题392slot、稳定种子、成功/失败方向、平局弃权及禁止跨case配对。正式采样、真实偏好供给、DPO/GRPO尚未运行。需要用户执行下述后台GPU任务；预计约2–4小时仅为预算，申请4小时，可续跑，排队另算。拿到rollouts后完整审计及偏好质量评估再冻结DPO训练，不保证一定有足够对。

提交前全套280项unittest通过（约24秒），含阶段036的完整1568轨迹重放；git diff --check通过。计划已本地冻结。计划generation_budget保留原评测来源信息（包含greedy字段），本次生成显式以独立sampling字段覆盖，实际do_sample=true；不修改原评测计划。

## SuperPOD指令

登录节点执行一次（不要先srun）：

```bash
cd /home/zshaoaj/agent-post-training-lab
git pull --ff-only
conda activate agent-post-training
mkdir -p logs/transaction_preference_probe_v1
sbatch --account=mscaiesuperpod --partition=normal --gres=gpu:1 \
  --time=04:00:00 --job-name=txn-pref-probe \
  --chdir=/home/zshaoaj/agent-post-training-lab \
  --output=logs/transaction_preference_probe_v1/probe-%j.log --export=ALL \
  --wrap="exec \"$(command -v python)\" -u -m scripts.transaction_preference_probe"
```

收到job编号后不要重复提交，可断开SSH。完成时日志出现Preference probe complete，392条rollouts和pair_yield.json产生。上传：

```bash
cd /home/zshaoaj/agent-post-training-lab
git add results/transaction_preference_probe_v1/rollouts.jsonl results/transaction_preference_probe_v1/pair_yield.json
git commit -m "Upload train-only transaction preference probe"
git push origin main
```

若超时，只在确认旧作业结束后重新提交同命令。不要删原始输出、冻结文件或adapter。若报错，保留错误再诊断，不自行降低预算或调采样参数。
