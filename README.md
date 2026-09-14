# Agent Post-Training Lab

面向中文 BugOps 场景的 Tool-Use Agent 后训练实验仓库。项目用完全合成的应用状态、知识库和工具，评估模型在工具选择、参数生成、长链路规划、状态跟踪和失败恢复上的能力，并为后续 LoRA SFT、DPO / GRPO 对比提供稳定基线。

持续更新：[实验总报告](docs/experiment_report.md) · [阶段日志](docs/experiment_logs/README.md)。每个阶段保留目标、配置、数据、指标、结果、bad case、问题、原因、解决方案和下一步，包括失败实验与设计修订。

## 当前入口：LoRA pilot v1

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
