# Agent Post-Training Lab

面向中文 BugOps 场景的 Tool-Use Agent 后训练实验仓库。项目用完全合成的应用状态、知识库和工具，评估模型在工具选择、参数生成、长链路规划、状态跟踪和失败恢复上的能力，并为后续 LoRA SFT、DPO / GRPO 对比提供稳定基线。

持续更新：[实验总报告](docs/experiment_report.md) · [阶段日志](docs/experiment_logs/README.md)。每个阶段保留目标、配置、数据、指标、结果、bad case、问题、原因、解决方案和下一步，包括失败实验与设计修订。

## 最终主结果（2026-09-22）

混合 replay + repaired SFT 已完成最终评测。模型从原始 Full SFT 初始化，使用196条原始事务 replay与96条repaired事务混合训练。repaired ID/OOD均为32/32；旧 v1 ID为128/128；旧 v1 OOD从原始SFT的258/400提升到272/400；旧 v1合计从386/528（73.11%）提升到400/528（75.76%），四份评测的 policy violations 均为0。完整过程、失败的DPO/GRPO尝试和回归分析见[实验总报告](docs/experiment_report.md)及[阶段060](docs/experiment_logs/060_mixed_sft_final_results.md)。

仓库只发布代码、合成数据、评测结果和实验记录，不包含基础模型或LoRA权重。训练权重保存在SuperPOD的 `checkpoints/transaction_mixed_sft_v1/adapter`，如需复现需先准备相同基础模型和依赖环境。

## 当前入口：完整Agent实验重启（2026-09-16）

阶段044/045：正式DPO首seed已完成；OOD任务258/400→256/400，0改善、2退化，极低偏好loss未带来闭环收益。[负结果及审计](docs/experiment_logs/044_full_dpo_results.md)。下一步原SFT训练侧32次GRPO奖励方差采样，需[后台GPU操作](docs/experiment_logs/045_grpo_rollout_readiness.md)；不是GRPO训练完成。以下为历史摘要。

阶段042/043：两步真实DPO数值/梯度已核验，未做任务成绩推断；现已冻结正式80对/80步DPO与784题闭环评测，从原SFT重新开始、不续gate模型。[工程实测](docs/experiment_logs/042_dpo_gate_results.md)、[正式GPU操作](docs/experiment_logs/043_full_dpo_readiness.md)。下面为历史阶段摘要。

阶段040/041：已核验98对真实跨策略偏好，冻结80训练/18偏好开发；多轮DPO概率/mask/固定参考接线已实现，需要[两步GPU工程检查](docs/experiment_logs/041_dpo_engineering_gate.md)。正式DPO尚未运行，不承诺小规模简单负例会提高OOD；[数据结果](docs/experiment_logs/040_cross_policy_pairs.md)。下面为历史阶段摘要。

阶段038/039（2026-09-17）：训练侧392次候选全成功且同题输出相同，0可用偏好对，保留这一负结果。下一步仅采集同98个训练任务的Base真实轨迹做跨策略候选；[结果与原因](docs/experiment_logs/038_preference_probe_zero_pairs.md)、[短GPU操作](docs/experiment_logs/039_cross_policy_preferences_ready.md)。DPO尚未训练，不回灌确认失败。

阶段036/037（2026-09-17）：完整SFT实际完成并重放1568条结果；SFT开发256/256、确认ID128/128、OOD258/400。142条SFT失败保留，含16条权限撤销后重复写请求，不能以执行终态正确宣称全面成功。[结果与中断复盘](docs/experiment_logs/036_full_sft_results.md)。下一步[训练侧偏好供给采样](docs/experiment_logs/037_preference_probe_readiness.md)，需后台GPU，不是DPO训练；完整路线继续。

阶段035已核验新事务GPU工程结果：链路正常，Base/两步adapter任务均0/14，失败保留且评分不变。现在需要[完整SFT首轮SuperPOD运行](docs/transaction_full_sft_v1.md)：2058训练轨迹、31542预测例子、1epoch/3943步；Base/SFT各评估开发256、确认ID128、确认OOD400。正式运行尚未开始，支持检查点恢复；[详细日志](docs/experiment_logs/035_transaction_smoke_and_full_sft.md)。DPO/GRPO和后续路线仍待推进。

用户已确认继续完整SFT／DPO／GRPO及独立工具后端迁移路线；阶段001–032保留为pilot，不代表整个项目完成。见[完整计划与验收门槛](docs/full_experiment_plan.md)、[阶段033](docs/experiment_logs/033_full_agent_restart.md)。当前新增隔离事务环境与50条开发oracle重放，无新模型成绩，尚不需要SuperPOD。

本地检查：`python3 -m scripts.prepare_transaction_v1`；开发fixture检查：`python3 -m scripts.check_transaction_dev`。二者均不加载模型。请用阶段034的新GPU入口，不应运行下方历史GPU入口代替新实验；下段033说明保留为历史状态。

## 历史入口：LoRA pilot v1

当前资源决策：[Control v1指南](docs/control_v1.md)。新2.4与76条对照题已准备，拟固定Base/SFT各推理76次；不训练，尚待用户决定申请GPU。旧结果/协议保持不变。

2026-09-15：[阶段019复核](docs/experiment_logs/019_challenge_answer_review.md)给出作者AI口径任务Base5%、SFT49.17%；执行率仍15%/68.33%。发现答案矛盾与8条只读观测漏检，阶段020提供独立审计原型，不静默改旧分。[学习复盘](docs/learning_review.md)汇总实验原理及简历可讲边界。

最新实测：[阶段018挑战结果](docs/experiment_logs/018_challenge_v1_results.md)，Base执行15%、SFT68.33%；表达重排暴露日志取证时机问题，部分起点存在导航循环。100条执行通过答案待审，当前无需GPU或重训。旧pilot100%不等于广泛泛化。

当前操作：[Challenge v1推理指南](docs/challenge_v1.md)。120题/18组已冻结，145项测试通过；申请GPU仅推理固定Base/SFT，不再训练。构造失败与修正见阶段017。

最新：[阶段015答案复核](docs/experiment_logs/015_pilot_answer_review.md)完成作者AI首轮审查，验证/确认各任务成功Base0/80、SFT80/80，非独立人工评审。Base各3条执行通过答案均错报零失败；执行率不变。下一步准备新泛化挑战，不能从同模板100%推断复杂业务能力。

最新：[训练记录核验](docs/experiment_logs/014_training_metadata_audit.md)已补齐，H800完成1 epoch/495步，训练阶段约27分6秒；当前待办仅答案复核等证据审查，无需重跑训练。下文各阶段状态为历史记录。

2026-09-14更新：[真实Base/SFT比较](docs/experiment_logs/013_pilot_base_sft_comparison.md)已完成，验证、确认各80条执行通过均从3提升至80。答案未全量复核，不等于最终任务成功100%；仅4个组合组/split。当前无需重跑训练，待补齐训练元数据与语义复核。下段保留准备阶段说明。

[按这份指南开始训练](docs/training_pilot_v1.md)：公开契约2.3、320条独立配置事务训练轨迹、80验证、80确认，按完整组合隔离。120项测试及CPU随机小模型训练/保存/重载已通过；真实4B GPU检查仍需在SuperPOD执行。旧360条不进训练，本轮不训练知识检索。先进入实验目录并申请GPU，再运行scripts/superpod_train_pilot.sh；它先留存Base，再smoke、训练与评测，不能在登录节点运行。

## Benchmark v2

最新（2026-09-14）：[Evaluator v2.2原理与操作](docs/protocol_v22.md)已补齐恢复题“失败后检查状态再重试”的过程要求。旧352条回顾评分188通过（53.41%），107项测试通过。仅修订评估，无新模型训练；旧2.0/2.1结果和下方历史说明保留。详见[阶段009](docs/experiment_logs/009_recovery_process_v22.md)。

2026-09-10 审计发现 v2 存在指定工具过严、自然语言规则误判和部分输入缺信息的问题。原始成绩保留；新增 [2.1 审计协议](docs/protocol_v21.md) 与 `python scripts/audit_baseline_v21.py`，分离执行指标、待人工复核的回答和需要重新推理的输入。v2 已用于协议开发，后续确认性实验需要独立测试集。

**当前 v2.1 已接入正式 runner 和 summarizer**，需显式指定 v21 数据路径；旧默认路径仍保留兼容。请优先按 [v2.1 操作指南（含 GPU 申请与补跑）](docs/protocol_v21.md) 执行。详细的错误案例、定位过程、修复取舍和未完成边界见 [实验问题复盘](docs/experiment_lessons_v21.md)。2026-09-14：[206条首轮作者AI自审](docs/experiment_logs/007_answer_review_v21.md)已完成，191通过、10失败、5存疑；8条补跑另行自审通过。仍不能把执行率当作最终任务成绩。

`data/eval/bugops_eval_v2.jsonl` 是一套 **360 条中文评测集（原按 held-out 设计，现为开发／审计集）**：12 个任务族 × 每族 15 个语义场景 × 每个场景 2 种表述。成对表述共享环境与评测目标，只改变用户请求的说法；case 覆盖知识检索、单工具、页面导航、长链路复现与规避、异常恢复和无需工具回答。

数据按场景族交错排列，不会先连续放置 30 条同类 case。因此 `--limit 5` 的 smoke run 同时覆盖 no-tool 和多种工具任务，而不是只测数据文件开头的一个任务族。

v1 仅保留为早期探索记录；下文 v2 命令保留用于历史复现，新一轮审计实验使用上方 v2.1 指南。

| | v1 | v2 |
|---|---|---|
| 规模 | 7 条探索样例 | 12 families × 15 scenarios × 2 surfaces = 360 条 |
| 环境 | 固定初始状态 | 每条 case 可配置隐藏初始状态 |
| 恢复测试 | 依赖偶发或无效操作 | 确定性故障注入 |
| 工具期望 | 工具名序列 | 工具名 + 关键参数 |
| 核心判定 | 与单一轨迹完全匹配 | 配置的结果、状态与回答条件是否全部达成 |
| 诊断 | exact / set match、失败次数 | 工具、参数、执行、效率、恢复，以及类别/难度/场景族分组 |

Evaluator v2 不再把“与人工轨迹逐步完全相同”当作任务成功。`task_success` 由最终环境状态等任务目标判定；轨迹匹配只作为诊断信号。字段和指标定义见 [docs/evaluator_v2.md](docs/evaluator_v2.md)。

> **Held-out 边界：严禁将 `data/eval/bugops_eval_v2.jsonl` 的问题、答案、目标状态、工具轨迹或改写版本用于 SFT、偏好数据、提示词示例或数据生成种子。** 训练集必须独立构造；否则 v2 分数不再能代表泛化能力。迭代时可以统计失败类型，但不要把具体评测 case 回灌训练。

## 在 SuperPOD 上运行

以下命令均在仓库根目录执行。

```bash
git pull
```

若当前 Python 环境尚未安装依赖，可选执行：

```bash
python -m pip install -r requirements-lock.txt
```

模型默认路径为 `/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507`。可通过参数或环境变量替换：

```bash
python scripts/run_baseline.py \
  --model-path /path/to/model \
  --limit 5 \
  --output-path results/baseline/qwen3_4b_smoke_v2.jsonl
```

等价的环境变量方式：

```bash
export BUGOPS_MODEL_PATH=/path/to/model
```

若 `--model-path` 使用 Hugging Face 远程模型标识，建议从第一次运行起固定 commit revision，尤其是需要断点续跑时：

```bash
python scripts/run_baseline.py \
  --model-path organization/model \
  --model-revision COMMIT_SHA \
  --limit 5 \
  --output-path results/baseline/remote_model_smoke_v2.jsonl
```

远程模型没有本地内容哈希时，未设置 `--model-revision` 的结果不能安全续跑；后续 `--resume` 也必须重复使用相同的模型标识、revision 和输出路径。

先运行 5 条 smoke test，确认模型加载、工具调用和结果写入正常：

```bash
python scripts/run_baseline.py \
  --limit 5 \
  --output-path results/baseline/qwen3_4b_smoke_v2.jsonl
```

这 5 条只用于检查链路，不应用来估计完整 benchmark 分数。

运行完整 benchmark；结果会逐条追加，任务中断后可用同一命令续跑，已完成 case 会被跳过：

```bash
python scripts/run_baseline.py --resume
```

### 安全续跑

runner 会在加载模型前计算运行指纹，覆盖 benchmark 与选中 case、Evaluator、Agent / 工具运行代码、合成知识库、本地模型内容或远程 revision、Python / PyTorch / Transformers 版本，以及解码和 step 配置。

`--resume` 只接受与现有输出 **完全相同** 的指纹；更换模型、修改 benchmark / 代码 / 知识库、改变 `--limit`、`--max-steps` 或 `--max-new-tokens` 都会拒绝混跑。此时应使用新的 `--output-path` 开始新实验。smoke 和完整评测使用不同输出文件，正是为了保持这条边界。

若任务中断时只留下一个未写完的末尾 JSON，runner 会安全丢弃该尾行后续跑；文件中间损坏、重复 case 或旧版无指纹结果不会被自动放行。

汇总默认 v2 结果：

```bash
python scripts/summarize_baseline.py
```

如需汇总 smoke 输出、用当前 Evaluator 重算旧记录或获得完整 JSON：

```bash
python scripts/summarize_baseline.py \
  --input-path results/baseline/qwen3_4b_smoke_v2.jsonl \
  --recompute \
  --json
```

summarizer 会显示已完成数 / 本次选定 case 数，并对中断产生的 partial result 给出醒目警告；同时拒绝重复 case、不同运行指纹或不同 Evaluator 版本的混合文件。`--recompute` 只重算指标，不会绕过这些检查。

运行本地单元测试：

```bash
python -m unittest discover -s tests -v
```

校验提交的数据可由确定性生成器完整复现：

```bash
python scripts/generate_benchmark_v2.py --check
```

## 目录

```text
agent/          Agent 循环与合成应用环境
tools/          工具 schema、执行器与知识检索
data/eval/      held-out benchmark（仅评测）
evaluation/     case 级 Evaluator v2 与汇总逻辑
scripts/        运行、续跑和汇总入口
tests/          无需加载大模型的单元测试
results/        本地实验输出
docs/           评测协议与实验记录
```

仓库中的产品、缺陷、日志和知识均为合成内容，不包含真实业务代码或内部数据。
