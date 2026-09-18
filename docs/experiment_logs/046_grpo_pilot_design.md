# 046 — GRPO Pilot：From Reward Saturation to Real Policy Update

## 状态与证据

设计与代码已完成，本地冻结通过；尚未执行GPU更新，因此没有把设计写成训练结果。正式pilot使用原始SFT adapter，不使用DPO adapter，也不读取32条probe作为训练输入。

## 实验目标

在固定严格任务奖励、固定train/ID/OOD边界的前提下，比较temperature 1.2、1.5、1.8的探索质量，并执行真实GRPO更新。要分别回答训练信号是否存在、策略是否改变、ID/OOD是否改善，而不是只报告训练reward。

## 数据边界与配置

训练subset从原`train`按稳定哈希选择24个不同结构组（apply12、rollback11、permission1），每次8组、每组4条completion，2次更新/temperature，共3×2×8×4=192条训练episodes。实际数据只有1个permission结构组，已在协议中保留，不伪造类别平衡。confirmation_id(128)与confirmation_ood(400)只用于更新后独立评测；probe的32条轨迹完全排除。

每个temperature执行2次update，group size=4，temperature分别1.2/1.5/1.8，top_p=1、top_k=0、max_new_tokens=512、上下文8192。GRPO clip=0.2、KL beta=0.01、学习率1e-5、梯度裁剪1；reference为SFT初始化的冻结副本；每次update保存checkpoint。reward固定为重放得到的binary task_success，不能信任模型自报成功。

## 已实现内容

`training/transaction_grpo.py`新增clipped objective、GRPO KL估计和显式分母统计；`scripts/transaction_grpo_pilot.py`实现真实采样、组内优势、assistant completion token-only policy update、reference KL、entropy、checkpoint和update日志。`data/transaction_grpo_pilot_v1_protocol.json`冻结选择与来源指纹。

## 指标与输出

每次update记录reward mean/std、advantage mean/std、policy ratio、KL、entropy、completion token数、tool validity和task success。输出：`results/transaction_grpo_pilot_v1/updates.jsonl`、`protocol_snapshot.json`；checkpoint为`checkpoints/transaction_grpo_pilot_v1/checkpoint-t*-*.pt`。独立评测应另存SFT/GRPO各split结果，不能覆盖SFT冻结文件。

## 设计问题与限制

24组是小规模pilot，不支持显著性结论；每组4条优势估计仍然高方差。permission只有一个训练结构组，不能宣称类别均衡。temperature数值是预先声明的探索对照，不是调参后挑最好结果。probe证明了非零variance，但不能证明update有效。当前本地无torch/权重，GPU前向、梯度和reference KL尚未验证。

## 下一步

SuperPOD作业入口（正式更新，不是probe）：

```bash
cd /home/zshaoaj/agent-post-training-lab
git pull --ff-only
conda activate agent-post-training
mkdir -p logs/transaction_grpo_pilot_v1
sbatch --account=mscaiesuperpod --partition=normal --gres=gpu:1 --time=08:00:00 \
  --job-name=txn-grpo-pilot --chdir=/home/zshaoaj/agent-post-training-lab \
  --output=logs/transaction_grpo_pilot_v1/pilot-%j.log --export=ALL \
  --wrap="exec \"$(command -v python)\" -u -m scripts.transaction_grpo_pilot"
```

预计需要比32条probe更久；以Slurm状态和日志为准，不要前台运行。完成标志是6个temperature/update记录、每个update有adapter checkpoint和optimizer状态；若中断，保留目录并先诊断，不删除重跑。

先在SuperPOD运行冻结pilot，随后分别用SFT和每个GRPO checkpoint在ID/OOD评测，汇总long-horizon、recovery、tool validity、hallucinated/repeated calls及latency。若reward上升而OOD下降，记录为reward hacking/过拟合；若所有temperature均无提升，保留负结果，不改reward或benchmark。

## 追加记录

本阶段没有训练结果，避免把probe或协议冻结误写成GRPO提升。
