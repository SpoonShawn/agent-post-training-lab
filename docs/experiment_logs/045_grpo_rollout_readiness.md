# 045 — GRPO训练侧奖励方差与行为概率采样准备

## 状态与证据

2026-09-17，本地实现及CPU验证；GPU尚未运行，无新模型成绩，没有优化器更新。前置证据为阶段038的392/392同题同质成功，以及阶段044的正式DPO负结果。代码随本日志提交，冻结文件`data/transaction_grpo_rollout_probe_v1.json`记录代码/SFT来源/任务排程指纹。

## 实验目标

回答：原SFT在更广采样下，同一训练任务的4条真实轨迹是否出现不同完整任务奖励？同时记录实际采样token和行为概率，为后续在线更新提供正确分母。不能用离线DPO或仅采样冒充GRPO。

## 配置与数据集

从原完整SFT adapter初始化，不使用DPO。按固定哈希从原训练侧98候选任务选8个不同结构组：apply3、rollback4、permission1，每题4个固定seed，共32 episodes。无开发/确认题、无确认失败改写。冻结排程含全部seed，不声称这8题能代表全部训练分布。

采样temperature1.5、top_p1、top_k0、num_beams1、repetition_penalty1；每轮最多512新token，8192上下文，40轮、36工具调用。与原T0.8/top_p0.95探针不同，不能把差异当模型提升。提高温度是针对已观测的train同质问题，数值1.5为待验证的探索选择，并非已调优结论。

新建GenerationConfig避免继承隐含采样惩罚；记录完整配置、输入prompt IDs、实际generated IDs（包含真实EOS，不补写EOS）、逐token采样logprob与entropy、工具证据、种子和来源。每个episode新环境；超长上下文不截掉证据，失败及预算截断保留。

## 指标

奖励仅使用原evaluator重放所得严格task_success，取0/1；声明成功不能替代工具证据，执行终态正确但违规仍为0。每组4条计算均值、总体标准差与`(r-mean)/(std+1e-8)`；同分组优势全部为0，不伪造负例。报告32条覆盖、8组中混合奖励组/零方差组，而非宣称任务泛化改善。

方法来源：[DeepSeekMath原始论文GRPO部分](https://arxiv.org/html/2402.03300v3)。组内结果奖励比较必须有差异才能产生此处的奖励优势；零优势不等于任何含KL目标都必然零梯度。本阶段没有实现clip/KL策略更新。

## 结果与Bad case

模型结果待运行，不能预填。已有风险案例：阶段038同题4条全正确，组内std=0；若直接做此处奖励优势更新则没有奖励驱动信号。真实新失败将保存在`results/transaction_grpo_rollout_probe_v1/rollouts.jsonl`，不得剔除后只报告混合组。

## 遇到的问题、原因及解决方案

1. 奖励饱和：原SFT训练采样392次全对。先用32次有界探索验证，不直接扩大训练量、不改评分奖励。
2. 行为概率错配：采样使用温度1.5，后续若拿温度1的概率作importance ratio会换了分布。当前从generate处理后的scores计算实际行为logprob；未来new/old/ref概率须遵循明确一致的温度定义。本阶段full support，不使用top-p/top-k截断。
3. 多轮mask：工具返回是下一轮观测，不是模型动作。记录每轮公开prompt与真实生成token，后续只对生成token更新；不对工具文本求策略loss。
4. SSH断线：使用Slurm后台运行，完整记录逐题flush、排程和元数据校验，重跑会续接已完成行；损坏记录会报错，不静默丢弃。

## 验证与尚未验证边界

冻结命令`python3 -m scripts.transaction_grpo_rollout_probe --freeze`通过；全仓294项测试通过（132.348秒），包括本轮新增6项测试。`git diff --check`通过。上述是CPU/重放验证，不是GPU生成验证。

CPU测试覆盖固定train排程、全对/全错零优势、混合奖励、非法奖励、伪造成功无证据、已知只读仍请求写入不给奖励、token/logprob对齐与非有限值拒绝。本地没有torch/模型权重，实际generate scores和teacher-forced前向概率的一致性尚未GPU验证；下一阶段正式更新前必须检查。当前不是完整GRPO实现或训练成绩。

## 下一步：SuperPOD操作

在登录节点运行；1小时是作业上限，不含排队也不是精确完成时间。不要重复提交并发作业；本脚本有排他锁。

```bash
cd /home/zshaoaj/agent-post-training-lab
git pull --ff-only
conda activate agent-post-training
mkdir -p logs/transaction_grpo_rollout_probe_v1
sbatch --account=mscaiesuperpod --partition=normal --gres=gpu:1 \
  --time=01:00:00 --job-name=txn-grpo-probe \
  --chdir=/home/zshaoaj/agent-post-training-lab \
  --output=logs/transaction_grpo_rollout_probe_v1/probe-%j.log --export=ALL \
  --wrap="exec \"$(command -v python)\" -u -m scripts.transaction_grpo_rollout_probe"
```

用返回的作业号检查`sacct -j 作业号 --format=JobID,State,ExitCode,Elapsed`及对应日志；完成标志为32行rollouts、summary存在、作业COMPLETED且ExitCode0:0。

完成后上传（不上传模型、锁文件）：

```bash
cd /home/zshaoaj/agent-post-training-lab
wc -l results/transaction_grpo_rollout_probe_v1/rollouts.jsonl
git add results/transaction_grpo_rollout_probe_v1/rollouts.jsonl results/transaction_grpo_rollout_probe_v1/summary.json
git commit -m "Record train-only GRPO rollout variance probe"
git push
```

上传后审计方差与行为证据，再实现/验证在线GRPO更新。若仍无奖励差异，记录零信号负结果并制定训练侧探索方案，不把无效更新称为训练完成。完整路线的多seed/对照/独立后端仍保留。

## 追加记录／勘误

无。
