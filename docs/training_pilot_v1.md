# Pilot v1：已具备上机训练条件

准备完成不等于4B模型已训练。已完成CPU小模型训练/保存/重载检查；真实4B显存、耗时和GPU兼容性仍由下方上机smoke检验。旧360题不进入训练。

## 固定实验问题与范围

本轮只研究：同一Qwen3-4B-Instruct-2507经少量可执行配置事务轨迹LoRA SFT后，是否更能遵守多步动作顺序并在临时失败后检查状态再恢复。

不是整个BugOps业务后训练，不训练知识检索/真实根因分析，不宣称生产可用。数据由程序规则教师执行公开工具生成，不是强模型蒸馏，也不含人工推理链。

| 数据 | 轨迹/任务 | 组合组数 | 用途 |
|---|---:|---:|---|
| train | 320 | 16 | 3956个assistant下一轮监督样本 |
| validation | 80 | 4 | 944个assistant样本，验证loss及运行诊断 |
| confirmation | 80 | 4 | 训练前后固定运行；不进入训练或选择checkpoint |
| 旧开发题公开约定版 | 360 | 仍为旧场景 | 可选回归诊断，不是独立确认集 |

训练目标token总量97356；实际Qwen tokenizer最长2254 tokens，4096上限下没有截断。计数不是独立样本数。按画质三步排列（6种）×结束页面（2种）×预设故障开关（2种）形成24个组合，固定哈希排序先分16/4/4组，再展开平台、起始页面和两种措辞。共享工具原语但完整组合组不重叠，属于组合泛化而非工具/任务族完全未见。

训练分布为120无故障+200恢复，验证和确认各60无故障+20恢复；分布不均衡如实报告，不将总体率差异等同单一能力变化。确认仅4个组合组，统计能力很弱，不按80个独立样本作显著性结论。

任务是新写的“依次改变三个画质档位，然后回首页或进入战斗并查日志”配置事务，使用模拟版本3.1.0；不是将旧问题换个版本号。生成器不导入旧benchmark或其答案。构造完成后才读取旧题做精确问题和环境/参考调用重叠拒绝；不能把精确去重当作语义泄漏的完整证明。无INC-101/102训练样本。旧开发集及其改写继续隔离。

## 协议2.3：本轮固定

新增public_contract，并把日志、环境核对、先检索后变更、禁止多余操作、失败后观察再重试写入可见题目。版本证据允许get_build_info或状态工具，动作里程碑按实际成功顺序检查。旧2.0/2.1/2.2不追改。

旧360题全部追加了公开验收约定，因此旧答案不能直接重评分；路径data/eval/bugops_development_v23.jsonl已标为development_audit。本轮不再通过追改旧分数证明收益。原阶段010四组反例现在对应明确约束及测试。

训练及推理都使用相同系统提示、工具定义和Qwen chat template。输入context及工具返回的labels均为-100，只监督下一条assistant工具调用/最终回答，保留EOS；不packing、不静默截断。runner去除回复末尾EOS后再作为历史消息交回chat template，防止重复结束标记；因此必须用当前runner重新跑Base，不与旧运行混比。

代码和数据哈希固定在data/pilot_v1/manifest.json。训练入口自动验证；变化则拒绝训练，后续实验应新建版本而非手改manifest绕过。数据准备脚本重跑仅接受完全一致产物。

## 预先确定的比较

- 主指标：confirmation的execution_success（终态、必要日志、顺序和恢复约束全部满足），同时按有/无故障分组报告。
- 次指标：答案语义复核后的task_success、调用失败/重复、步骤数。答案未审则task_success未决，不拿执行率冒充任务率。
- Base与SFT必须同模型基础权重、协议、题目、工具、greedy配置及步数预算。训练入口强制检查80条完整未训练Base confirmation运行来源，防止遗漏baseline。
- 用最后一个训练checkpoint，不根据confirmation选择学习率、轮数或checkpoint。出现OOM/兼容问题可修工程配置，但必须另记；若看确认成绩后改训练，则该组转开发，另建确认组。
- 旧360题及知识/无工具能力可作后续回归诊断；本轮不据此调整标签提分。旧首轮语义一致性问题未被新训练准备“自动解决”。

## LoRA 配置

1个支持bf16的CUDA GPU；r=16、alpha=32、dropout=0.05、all-linear，学习率1e-4，1 epoch，batch=1，梯度累积8，warmup=10 steps，cosine，gradient checkpointing，max_length=4096，seed=20260914，最多保留2个训练checkpoint。不量化，不上传权重，不启用外部实验追踪。

实现依据[PEFT LoRA说明](https://huggingface.co/docs/peft/main/package_reference/lora)及[Transformers Trainer说明](https://huggingface.co/docs/transformers/main/main_classes/trainer)。仅说明API依据，不把当前文档代替版本测试。

## 上机操作：从实验目录开始

登录节点（不在此运行模型推理/训练）：

```bash
cd /home/zshaoaj/agent-post-training-lab
git pull --ff-only
conda activate agent-post-training
python -m pip install -r requirements-sft.txt
python scripts/verify_training_bundle.py
python -m unittest discover -s tests -q
srun --account=mscaiesuperpod --partition=normal --gres=gpu:1 --time=02:00:00 --pty bash
```

依赖文件仅固定transformers=5.16.1、peft=0.20.0、accelerate=1.14.0，不替换已有CUDA torch。总耗时未经4B实测，不能保证2小时全部完成；到时按下面的断点策略处理。

进入GPU节点后：

```bash
cd /home/zshaoaj/agent-post-training-lab
conda activate agent-post-training
nvidia-smi
bash scripts/superpod_train_pilot.sh
```

脚本依次执行：80条Base确认运行 → 80条Base验证运行 → 2步4B训练smoke → 全部320轨迹的1 epoch LoRA → 验证集与确认集SFT运行。任何一步失败立即停止。不要跳过GPU申请，也不要将本地tiny adapter用于4B评测。

产物：checkpoints/pilot_v1_lora/adapter、训练run.json/attempt_*.json和checkpoint；results/baseline/pilot_v1_*四个运行；logs/pilot_v1日志。结束后exit释放交互资源。大权重不会提交Git。

## 中断与恢复

Base/SFT推理可以执行脚本中对应的run_baseline命令并保留--resume，它核对模型/adapter/代码指纹。不要在成功smoke或训练目录已存在时盲目重跑整段脚本，训练会拒绝覆盖。

若完整训练中断，先查找实际存在的checkpoint编号（不要猜编号），重新申请GPU后：

```bash
cd /home/zshaoaj/agent-post-training-lab
conda activate agent-post-training
ls -d checkpoints/pilot_v1_lora/checkpoint-*
# 将下方 N 替换为实际 checkpoint 编号：
python -u scripts/train_sft.py \
  --output-dir checkpoints/pilot_v1_lora \
  --resume-from-checkpoint checkpoints/pilot_v1_lora/checkpoint-N
```

恢复校验配置、manifest、依赖版本及基础权重不变；保留每次attempt元数据。没有checkpoint时不能恢复，应另起输出目录并记录失败，不删除旧记录。

## 本地验证边界

120项标准库测试，包括480条新oracle、360条公开约定版oracle、反例、组隔离、loss mask、超长拒绝和adapter指纹。真实Qwen tokenizer全量编码；随机微型Qwen3在CPU训练2步、验证、保存LoRA、重载后前向与权重有限性检查。微型模型loss不代表4B效果。

最终本地依赖：transformers5.16.1、peft0.20.0、accelerate1.14.0；macOS torch2.14.0与SuperPOD原torch2.9.1不同。4B CUDA/bf16路径未在本地运行，GPU smoke是必要条件，而不是已经验证的事实。
