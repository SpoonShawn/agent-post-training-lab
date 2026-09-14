# Challenge v1：固定模型的训练后压力测试

已构造并在首次推理前冻结，尚无模型成绩。不是新的训练包；不要运行训练脚本。

## 数据与边界

120题、18个组合组，data/challenge_v1/cases.jsonl与manifest.json记录指纹。

| 维度 | 题数 | 组数 | 检验内容 |
|---|---:|---:|---|
| expression | 40 | 4 | 20个既有确认场景，各原文/重新排序表达一对 |
| structure | 40 | 8 | 1/2/4/5次画质操作×home/battle终点×5起点，含重复档位 |
| constraints | 20 | 2 | 全程只读、禁止改画质但返回首页；两平台×5起点 |
| recovery | 20 | 4 | graphics导航、多人副本导航、back_home、启动副本首次合法操作故障×5起点 |

expression明确复用旧确认场景，是成对压力测试，不是独立held-out。其余由公开工具原语构造，也不声称与旧开发任务语义完全不重叠。用于训练后诊断，不回灌训练；本数据不应被包装为18个独立真实业务场景。

结构与恢复题主要为android，只有约束题包含两平台，因此不能据此比较跨平台泛化。故障组不选graphics起点，以确保“导航到graphics”确实发生；包含battle起点，起始HUD设为false是合成状态，不代表真实应用典型状态。

主指标：每维度、每组执行成功率；次指标：来源绑定的语义复核任务率。expression另报20对原文/改写的转移，不以整体120题分数替代成对变化。数字事实审计只作辅助。模型均greedy、max_new_tokens512、按case同预算，保留5步余量；表达对照沿用原case预算。固定基础模型和最后LoRA的内容哈希；不改checkpoint、不根据结果调参。

评分仍为2.3，日志内容采用包含条件、顺序由成功动作里程碑逐项检查；重复设置不因日志存在同一字符串就算执行两遍。只读题允许空日志，但必须成功调用日志工具。人工/AI语义审查仍需明确署名。

## 操作指令

登录节点，仅更新和CPU检查；沿用此前账户/分区，不重装依赖：

```bash
cd /home/zshaoaj/agent-post-training-lab
git pull --ff-only
conda activate agent-post-training
python scripts/verify_challenge_v1.py
srun --account=mscaiesuperpod --partition=normal --gres=gpu:1 --time=02:00:00 --pty bash
```

获得GPU shell后：

```bash
cd /home/zshaoaj/agent-post-training-lab
conda activate agent-post-training
nvidia-smi
bash scripts/superpod_eval_challenge_v1.sh
```

依次验证模型/adapter内容、检查CUDA、Base120题推理、LoRA120题推理；没有任何训练。哈希计算可能需等待，脚本有提示。2小时为申请时限，不保证全程耗时；超时/中断可重新申请GPU并重跑本脚本，--resume按指纹续跑。不要修改冻结文件或删除已完成结果。

两份都提示保存完成后，退出GPU shell，在登录节点提交：

```bash
exit
cd /home/zshaoaj/agent-post-training-lab
git add results/baseline/challenge_v1_base.jsonl results/baseline/challenge_v1_sft.jsonl
git commit -m "Add fixed Base and LoRA challenge v1 results"
git push origin main
```

若报错，保留完整错误和logs/challenge_v1/日志；不要自行改评分或重训。提交结果后本地核对完整覆盖、权重/代码一致性、按组结果及表达配对，再分析bad case。尚未运行独立盲审或多seed重复训练。
