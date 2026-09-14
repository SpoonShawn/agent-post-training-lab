# BugOps 2.1：评估审计修订

日期：2026-09-10。状态：看过 v2 模型输出后的回顾性协议修订，尚非独立确认性测试。

进度更新（2026-09-14，不修改协议）：8 条补跑及作者 AI 自审已完成；原 206 条首轮自审完成，191 pass、10 fail、5 uncertain。详见 [阶段007](experiment_logs/007_answer_review_v21.md)。独立评审和完整任务率仍未定案。

## 原因与边界

v2 原始代码、360 条输入、模型输出与原始 39.4% 任务成功率保留。2.1 的数值变化属于测量口径变化，不是模型提升。不能依据该变化宣称 SFT 有效。

原有 oracle 检查只证明参考轨迹可行，未证明等价轨迹会通过，也未充分检查自然语言否定、同义表达与题目可回答性。2.1 增加等价路径、缺证据、错误答案和输入修复隔离测试。

## 规则

1. navigation/long_horizon/recovery 的目标状态由 harness 最终状态验证，取消强制 `inspect_ui_state` 和 `verify_state` 的工具身份要求。动作结果已经暴露完整状态，额外重复检查不应成为唯一通过路径。
2. 知识检索、日志和其他工具证据保留；不能仅因为状态正确就认定完成所有要求。原有工具 precision/recall 仍只是参考轨迹诊断。
3. recovery 还必须实际触发预设故障并在之后的模型轮次恢复。没有触发故障的成功状态不能计恢复成功。
4. `execution_success` 不检查自然语言结论。回答统一进入人工复核；执行失败的 `task_success=false`，执行通过但未复核的 `task_success=null`。不再把脆弱的关键词规则当作最终答案真值。
5. `structured_argument_accuracy` 只统计非检索工具；搜索字符串精确匹配保留在旧指标中，检索有效性通过返回证据衡量。
6. 明示“用事件编号”却未给编号的 8 条问题补入 INC-101 / INC-102。新输入需要重新推理；旧输出排除 2.1 重评分。其余题目仍需独立可回答性审查，不能声称全部无歧义。

## 回答复核规范

审阅者读取用户问题、实际工具返回和最终答案，不参考模型名称、原分数或训练阶段。若无法安排独立审阅，必须标注作者自审。

- pass：回答覆盖用户所问的关键事实，结论与实际轨迹一致，无重要编造。
- fail：错误结论、重要编造、未回答核心问题或无最终回答。
- uncertain：事实或语言存在歧义，需要第二次复核；不可直接算通过。
- no-tool：必须实际回答解释题；无工具调用本身不等于答题成功。
- 用户要求日志/证据时，必须存在真实返回，不能用回答中声称“日志显示”代替。
- 同义表达、合理否定不扣分；不得对不同模型使用不同标准。

复核记录包含 case ID、来源文件 SHA-256、逐条 case/result SHA-256、判定、证据理由、审阅者类型及时间。队列位于 `answer_review_queue.jsonl`，包含问题、最终答案、实际工具返回和最终状态，不含模型名称和旧分数；这不能保证审阅者没有看过原结果，不应自动宣称双盲或独立评审。

现已支持复核导入。复制队列到独立文件（不要编辑自动生成的队列），逐条填写 `verdict`（pass/fail/uncertain）、`reason`、`reviewer`、`reviewer_type`、`reviewed_at`。`reviewer_type` 应如实写 `independent_human`、`protocol_author_human` 或 `protocol_author_ai`；工具只验证可追溯字段，不替你证明审阅者身份或独立性。保留所有哈希和证据。只提交已复核行，`pending` 模板不允许作为判定导入；`uncertain` 仍视为未解决。

导入会拒绝重复 ID、未知 ID、来源或轨迹变化、错误协议和缺失署名。即使答案判为 pass，执行失败也不能成为任务成功。执行成功但未审／存疑的任务维持 `null`。整体、类别、难度、场景族的任务率不会剔除这些 `null` 后冒充完整分数；另列逻辑上下界（不是统计置信区间）。

## 运行

在登录节点执行，无需 GPU：

```bash
cd /home/zshaoaj/agent-post-training-lab
git pull --ff-only
conda activate agent-post-training
python scripts/audit_baseline_v21.py
```

输出至 `results/audit/v21/`：`benchmark_v21.jsonl`（360 条修订输入与协议标记）、`rerun_v21.jsonl`（8 条修改输入）、`audit.jsonl`（逐条诊断）、`answer_review_queue.jsonl`（206 条待审及证据）、`summary.json`。这些文件可以确定性重建，脚本保留原始结果并校验 case 与 v2 快照一致。206 是当前未导入任何复核时的队列长度。

导入旧轨迹的答案复核，无需 GPU：

```bash
python scripts/audit_baseline_v21.py \
  --reviews results/reviews/v21_author_reviews.jsonl \
  --output-dir results/audit/v21_reviewed
```

该复核文件现已由阶段007实际逐条自审创建，记录为协议作者AI而非独立人工。生成目录与复核文件分离，以免重建时覆盖署名判定。导入后5条存疑仍维持未决；不带 --reviews 的默认命令仍会生成未审快照，不代表已保存判定失效。

## 正式推理与汇总入口

`run_baseline.py` / `summarize_baseline.py` 现在按 `case.protocol_version` 自动分流。无标记走原 v2；`2.1` 走新执行判定加显式答案复核。混合协议、未知协议、结果 query 与新输入不一致均会拒绝。`requires_new_inference` 标记输入相对 v2 有变化，并不意味着新轨迹永远不可评分；新轨迹 query 一致即可评估。

为兼容旧命令，默认 benchmark 仍为 v2；新实验必须显式指定下面的 v21 路径。推理输出会记录正确的协议版本，运行指纹包含新增评分代码。已有文件不允许无意覆盖，续跑要求指纹一致；不能把新代码下的运行追加到旧 v2 结果。**不要修改旧文件的版本或指纹来绕过检查。**

### SuperPOD 补跑 8 条

先在登录节点进入实验目录并更新、生成输入（无需 GPU）：

```bash
cd /home/zshaoaj/agent-post-training-lab
git pull --ff-only
conda activate agent-post-training
python scripts/audit_baseline_v21.py
```

再申请 GPU。以下沿用本实验此前使用的账户和分区；如果集群配置变化，按管理员提供的账户/分区替换，不能在登录节点直接推理。

```bash
srun --account=mscaiesuperpod --partition=normal --gres=gpu:1 --time=02:00:00 --pty bash
```

等到资源分配成功、进入计算节点后执行：

```bash
cd /home/zshaoaj/agent-post-training-lab
conda activate agent-post-training
nvidia-smi
python -u scripts/run_baseline.py \
  --model-path /home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507 \
  --eval-path results/audit/v21/rerun_v21.jsonl \
  --output-path results/baseline/qwen3_4b_rerun_v21.jsonl \
  --resume
exit
```

`exit` 释放交互式 GPU 作业。上述是补跑 8 条，不是 360 条正式测试。若要完整重跑，将 eval 路径改为 `results/audit/v21/benchmark_v21.jsonl`，输出改为新的 `results/baseline/qwen3_4b_baseline_v21.jsonl`。即便完整重跑，这套已观察过的数据仍是开发／审计集，不会重新变成独立确认集。

登录节点汇总补跑并导出待审队列：

```bash
cd /home/zshaoaj/agent-post-training-lab
conda activate agent-post-training
python scripts/summarize_baseline.py \
  --input-path results/baseline/qwen3_4b_rerun_v21.jsonl \
  --review-queue results/reviews/rerun_v21_queue.jsonl
```

完成复核并另存为 `results/reviews/rerun_v21_reviews.jsonl` 后：

```bash
python scripts/summarize_baseline.py \
  --input-path results/baseline/qwen3_4b_rerun_v21.jsonl \
  --reviews results/reviews/rerun_v21_reviews.jsonl
```

v21 汇总始终重算执行，复核必须显式加载；它不会信任旧缓存中的通过判定。新运行队列的哈希绑定新文件，不能与旧审计的复核互用。**不要直接拼接 352 条旧记录与 8 条新记录**：它们的运行指纹和输入来源不同；目前分别报告，若需要统一成绩，优先完整重跑 360 条并复核，或另行制定、验证带来源清单的合并协议。

## 冻结与后续实验

当前 v2 已参与协议开发，标注为 development/audit benchmark；不得继续把它作为未经观察的最终测试证据。180 个 scenario ID 也不意味着统计独立，成对改写必须同组。

在 SFT 前：完成独立题目审查和回答规范校准；构造独立确认集，按场景/模板分组隔离训练、开发和测试；冻结数据 SHA、协议版本、工具环境和推理配置；Base/SFT 在同一版本下对比。置信区间应按场景分组估计，不把两种表述当作独立抽样。

冻结后发现问题仍可修，但需要公开版本、理由和影响范围，所有比较模型统一重算或重跑；不选择对某模型最有利的评分版本。不能通过不断加关键词使已看到的答案过关。
