# 最后一轮小规模对照：固定呈现 vs 混合呈现

## 目的与停止条件

保持工具历史、监督答案、样本顺序、192条样本、1epoch/24步、学习率和随机种子相同，只比较证据呈现多样性。fixed全部使用JSON历史；mixed为JSON、逐条记录、表格三种布局，各64条。两个adapter均从同一Base新建，不继续训练之前adapter。

两组都使用新生成的较长历史，包含多次检查、重复日志、历史权限与当前权限不一致；因此它们之间能检验布局多样性，不能将它们与旧LoRA的差异归因于某一项修改。新题面也更明确说明最后证据与计数规则，Base会在同一题面重新评估。

训练192上下文/32组；验证48/8组；确认48/8组。每个评估上下文三布局，各split144题，两split288题；Base、fixed、mixed共864次报告推理。不是864个独立任务。训练集两组使用同一192上下文，不是384个独立样本。

主指标：确认集按布局分别统计exact_report（每布局48），比较fixed→mixed配对改善与退化；辅指标是valid_json、decision_correct和各字段错误。按8个底层结构组查看结果，不能把同上下文三布局视为独立样本做显著性声明。不使用确认成绩选checkpoint或反复调参，保留最后epoch。验证损失只作诊断。两组输入token/耗时可能不同，GPU预检记录精确值，不称等FLOPs。

不管提升、持平还是下降，完成这轮预注册比较后先收尾：审计结果、保留失败、整理最终报告与面试说明。不会以必须涨分为理由无限增加轮次。DPO/GRPO、真实设备部署、多seed大规模训练不在本次收尾范围。

## 登录节点：先进入实验目录，再申请GPU

```bash
cd /home/zshaoaj/agent-post-training-lab
git pull --ff-only
conda activate agent-post-training
python -m scripts.prepare_evidence_ablation
srun --account=mscaiesuperpod --partition=normal --gres=gpu:1 --time=02:00:00 --pty bash
```

最后一行沿用此前账号/分区；如调度器拒绝，请保留报错，不在登录节点直接启动模型。2小时是申请上限，不是运行保证；根据前次每题耗时，预计本轮约1–2小时，不含排队，长历史或环境问题可能超出。

## GPU节点：一条入口运行完整对照

```bash
cd /home/zshaoaj/agent-post-training-lab
conda activate agent-post-training
nvidia-smi
bash scripts/superpod_evidence_ablation.sh
```

顺序：实际分词/权重预检 → Base 288题 → 两步smoke → fixed训练 → mixed训练 → fixed 288题 → mixed 288题。GPU预检要求原依赖版本、模型内容一致，训练和评估不得截断；若超长或不匹配，停止并保留错误。不要现场升级依赖。

每个阶段的进度在终端及logs/evidence_ablation_v1/保留。失败即停止，不悄悄跳过。推理支持严格指纹下按完整行续跑；已完成训练会验证checkpoint并跳过。中断训练、损坏末行、不同指纹会停止，不能删除产物直接重跑掩盖失败。请上传现有结果并反馈报错后再诊断。

## 成功完成后上传

看到Complete后再退出GPU；不要在训练过程中exit。

```bash
exit
cd /home/zshaoaj/agent-post-training-lab
git add results/evidence_ablation_v1/
git commit -m "Add paired evidence layout ablation results"
git push origin main
```

结果目录包含preflight、smoke/fixed/mixed训练记录和base/fixed/mixed三份288行JSONL。checkpoints/evidence_ablation_v1/需保留在SuperPOD，不上传大权重。旧模型、旧数据和旧分数均不覆盖。上传后由本地完成来源审计、配对统计和实验收尾。

## 预计完成时间（2026-09-16）

- 当前本地准备已完成，下一步由用户申请GPU；GPU约1–2小时只是规划估计。
- 上传完整产物后，预计0.5–1个工作日完成核验、失败分类和比较；再0.5–1个工作日完善总报告、复现说明及简历/面试可用的事实清单。
- 总体目标为2–3个工作日，若9月16日能运行并上传，争取9月18–21日收尾。排队、配额、环境错误和用户操作延迟顺延，不承诺后台自动完成。
- “完成”指证据闭环、实验可复现、收益及限制清楚；不是保证取得高分，也不是完成最初设想的全部后训练算法。
