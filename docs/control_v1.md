# Control v1：先写清楚要求，再测试检查清单

## 给初学者的解释

不是再训练一次，也不是再堆一批测试题。这是对照实验：保持模型、场景和目标不变，只改变题目的表达或是否提供操作清单，看看错误能否被一个简单提示解决。

本轮新协议2.4要求实际取得终态和操作结束后的日志；旧成绩不重算。所有对照组都使用同一新协议，不能把新分数与旧分数直接相减当作模型提升。

## 题目与比较（首次运行前冻结）

76题来自28个已接触场景，全部明确为事后诊断，不是独立held-out，也不进入训练。

| 对照条件 | 数量 | 作用 |
|---|---:|---|
| original_explicit | 20 | 原叙述顺序，加明确最终取证约定 |
| reordered_explicit | 20 | 重排叙述，加同一取证约定 |
| reordered_checklist | 20 | 与上组完全相同，再附通用检查清单 |
| readonly_explicit | 8 | 先前缺状态证据的只读场景，加明确观测要求 |
| readonly_checklist | 8 | 同上，再附同一检查清单 |

每个模型都运行这76题：Base76、现有LoRA76，共152次任务推理。20个表达场景来自4个组合族；8个只读场景属于同一任务类型，不能视为28个独立真实业务场景。

优先看三组配对差值与失败转移：

1. 原顺序明确版→重排明确版：写清时机后，还对叙述顺序敏感吗？
2. 重排明确版→重排清单版：同任务加入检查清单是否有帮助？
3. 只读明确版→只读清单版：是否改善状态取证及报告？

两模型在同一条件下也作配对比较。不要把所有76题的重复条件混成一个总分作为主要结论。检查清单有多个提醒，本实验只检验整份清单，不能断言某一句单独起作用。提示变长会增加输入成本，不声称与原提示完全同成本。

若表现变好，说明清晰要求或提示可缓解这些诊断场景的错误，不自动代表广泛泛化；若不变或退化，同样保留结果。此次基于已知失败设计，没有未知测试集的独立性。未安排多seed训练或独立盲审。

## 固定配置

Base与LoRA均使用此前Qwen3-4B-Instruct-2507基础内容哈希和最后adapter。greedy、max_new_tokens512、相同上下文对应相同case步骤预算；没有训练，没有挑checkpoint。模型/代码/数据指纹固定在data/control_v1/manifest.json。

torch2.9.1、transformers5.16.1、peft0.20.0、accelerate1.14.0，沿用上轮环境，不重新升级包。新runner记录逐题开始/结束时间和GPU信息，便于后续核查，不用训练耗时代替推理耗时。

## 资源决策点

最终本地验证：176项测试通过，三套冻结包校验通过；CPU标准路径不是模型实测成绩。

本地构造、oracle、评分负例、模拟runner和Shell参数测试已完成。真实4B新输入尚未推理，需要用户决定是否申请1张GPU。沿用2小时申请上限，但本轮实际耗时尚未知，不保证在2小时内全部完成。共有152次任务推理，少于上一轮240次，但单题耗时可变。

在用户决定上机之前，不自动申请资源或创建后台任务。以下命令仅供决定运行后使用。

### 登录节点：同步、CPU验证、申请GPU

```bash
cd /home/zshaoaj/agent-post-training-lab
git pull --ff-only
conda activate agent-post-training
python -m scripts.verify_control_v1
srun --account=mscaiesuperpod --partition=normal --gres=gpu:1 --time=02:00:00 --pty bash
```

### 已获GPU的shell：启动

```bash
cd /home/zshaoaj/agent-post-training-lab
conda activate agent-post-training
nvidia-smi
bash scripts/superpod_eval_control_v1.sh
```

入口校验固定权重和adapter（哈希可能等待），先Base76再SFT76。只读哈希预检可用python -m scripts.run_control_v1 --role sft --preflight-only，不会加载模型；正常推理不能在登录节点运行。不要用旧summarize_baseline汇总2.4文件。

中断后重新申请GPU，运行相同脚本续跑。--resume检查配置、完整case及query；沿用旧安全续跑实现，若末行是写入中断的残片会提示并丢弃该不完整末行，已完成行不变；非末尾损坏/配置错配会停止。结果不覆盖旧实验。

### 两份都完成后：退出GPU并提交

```bash
exit
cd /home/zshaoaj/agent-post-training-lab
git add results/baseline/control_v1_base.jsonl results/baseline/control_v1_sft.jsonl
git commit -m "Add protocol 2.4 timing and checklist control results"
git push origin main
```

若报错先保留错误及logs/control_v1/日志，不修改评分、题目或模型目录。提交后再本地检查完整覆盖、指纹、配对、失败证据和答案，不因为清单有效就把当前已知场景当作训练后的独立确认结果。
