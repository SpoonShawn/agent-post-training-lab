# 051 — GRPO pilot上传结果：temperature探索对比

## 状态与证据

结果已上传提交`495b357`。每个temperature arm有2个update，每个update 12组×4=48条rollout；三档共288条。memory gate为H800、8192上下文、32个lm_head位置、peak allocated约12.36GiB。`completed.json`只代表6个checkpoint已有，之前的1分钟Slurm作业是读取已完成checkpoint后退出，不是重新训练这288条。

## 结果

| arm | update | reward mean±std | advantage std | KL mean | parameter delta L2 |
|---|---:|---:|---:|---:|---:|
| 1.2 | 1 | 0.9792±0.1428 | 0.2887 | 0 | 0.0542 |
| 1.2 | 2 | 1.0000±0 | 0 | -1.4e-9 | 0.0358 |
| 1.5 | 1 | 0.6667±0.4714 | 0.8660 | 0 | 0.0556 |
| 1.5 | 2 | 0.6250±0.4841 | 0.9574 | 4.80e-5 | 0.0406 |
| 1.8 | 1 | 0.0208±0.1428 | 0.2887 | 0 | 0.0531 |
| 1.8 | 2 | 0±0 | 0 | 7.98e-5 | 0.0348 |

所有arm均报告`policy_changed=true`。t=1.2很快重新饱和；t=1.5有最大有效方差但第二步下降；t=1.8近乎全失败，表现为噪声过大。KL接近0或很小，不能单独证明策略安全，且第一步的reference与policy相同属于预期。

## 解释边界

这是训练侧pilot统计，不是泛化成绩。reward是在每次update采样后、更新前记录的；第二步反映第一步更新后的rollout。尚未完成confirmation_id/OOD评测，不能选择temperature或声称GRPO提升。t=1.5只是当前探索/成功率折中候选，不是最优结论。

## 下一步

用新`evaluate_grpo_pilot_v2.py`分别评估三个arm的step1/step2 adapter以及SFT baseline，在confirmation_id和confirmation_ood各保存独立JSONL，再逐题比较task success、tool validity、重复调用和调用步数。评测完成前不修改reward、不选择checkpoint。
