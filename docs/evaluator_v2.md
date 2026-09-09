# Evaluator v2 评测协议

## 设计目标

Tool-Use Agent 往往存在多条合理路径。Evaluator v2 将“是否完成任务”与“是否复现参考轨迹”分开：

- `task_success` 是核心端到端指标，判断 case 配置的可观测状态、工具结果、最终回答和调用约束是否全部满足；
- 工具顺序、集合和参数匹配用于定位规划问题，不单独决定任务成功；
- 格式错误、未知工具、执行失败、重复调用和步数用于解释失败成本；
- recovery case 额外判断模型在确定性故障后是否真正恢复并完成目标。

因此，一个模型可以采用不同于参考轨迹的有效步骤并获得 `task_success=true`；也可能工具序列看似匹配，却因参数错误或最终状态不符而失败。

## Benchmark case schema

v2 使用 JSONL，每行一个独立 case。顶层字段如下：

| 字段 | 类型 | 含义 |
|---|---|---|
| `schema_version` | integer | 数据 schema 版本；当前为 `2` |
| `split` | string | 数据用途；v2 固定为 `held_out` |
| `id` | string | 全局唯一 case 标识 |
| `scenario_id` | string | 语义场景标识；同一值对应两种表述 |
| `surface_variant` | string | 表述编号：`surface_01` 或 `surface_02` |
| `category` | string | 任务类别，用于 macro 分组汇总 |
| `scenario_family` | string | 同一任务语义下的场景族 |
| `difficulty` | string | 难度分层 |
| `query` | string | 只向模型展示的中文请求 |
| `environment` | object | 隐藏环境配置，不写入模型消息 |
| `required_calls` | array | 参考所需工具调用；每项含 `name` 与 `arguments` |
| `required_tools` | array | 完成任务需要覆盖的工具名列表；重复项代表所需调用次数 |
| `expected_tools` | array | 兼容 v1 的参考工具名序列 |
| `optional_tools` | array | 合理但非必需的额外工具名；每个列表项豁免一次额外调用 |
| `success_criteria` | object | 最终任务成功条件 |
| `requires_recovery` | boolean | 是否为必须触发故障并恢复的 case |
| `max_steps` | integer | 当前 case 允许的最大 Agent 步数 |
| `tags` | array | 便于筛选和分析的静态标签 |

`required_calls` 使用工具名和参数共同描述参考行为；`argument_match` 可选值为 `subset`（默认，只检查列出的关键参数）或 `exact`（参数对象必须完全相同）：

```json
{
  "name": "navigate_ui",
  "arguments": {"target_page": "settings"},
  "argument_match": "subset"
}
```

`environment.initial_state` 可覆盖版本、平台、页面、画质、HUD 和黑屏状态。它是评测 harness 的隐藏条件，模型只能通过工具观察。`environment.fault_injections` 用于 recovery case，见下文。

v2 共 12 个 `scenario_family`，每族 15 个不同执行语义的 `scenario_id`；每个语义场景各有 `surface_01` 和 `surface_02` 两条表述，共 360 条。同一 `scenario_id` 的两行共享工具要求、环境、成功条件和 step 上限，仅 `query`、行级 `id` 与表述 tag 不同。

JSONL 采用 scenario → surface → family 的交错顺序，每连续 12 行各取一个任务族。这样文件前 5 行就包含 no-tool、构建信息、UI 状态和知识检索任务，`--limit 5` 可作为跨任务 smoke test，而不会只抽到一种类别。

`success_criteria` 支持四类可组合条件：

- `final_state`：对 `result.final_environment_state` 中列出的字段做等值检查；兼容旧结果时可回退到轨迹里最近的状态型工具结果；
- `required_tool_results`：按 `tool` 选择成功执行结果，用 `path` 定位字段，再用 `equals` 或 `contains` 验证；执行器的 `{ok, result}` 信封会先被拆包，路径根是 `result`，`$` 表示整个业务 payload；
- `final_answer`：可组合正向与负向文本约束，字段语义见下表。
- `max_tool_calls`：限制模型表达出的工具调用尝试总数，包括解析失败或标签残缺的调用意图；用于 no-tool case 时设为 `0`。

| `final_answer` 字段 | 通过条件 |
|---|---|
| `required` | 为 `true` 时必须存在非空最终回答 |
| `contains_all` | 列出的文本必须全部出现 |
| `contains_any` | 至少出现一个列出的文本 |
| `regex_any` | 至少匹配一个正则表达式 |
| `forbidden_any` | 列出的文本均不得出现 |
| `regex_none` | 所有正则表达式均不得匹配 |

正向和负向条件同时生效。例如复现类 case 可以要求回答包含“复现”，并用 `forbidden_any` / `regex_none` 排除“未复现”一类相反结论。
文本包含与正则匹配均忽略英文大小写，避免 `iOS` / `IOS` 这类纯大小写差异改变评分。

无需工具的 case 以空的 `required_tools` / `required_calls` 表示工具期望，同时要求最终回答并设置 `max_tool_calls=0`。因此模型若调用工具，不仅工具选择指标会受影响，`task_success` 也会失败。具体组合以各行数据为准。

## Case 级指标

### 任务与轨迹

- `task_success`：`success_criteria` 中配置的最终状态、工具结果、最终回答和调用上限全部满足。这是最重要的 case 级结果。
- `max_tool_calls_match`：工具调用尝试数（含格式错误或残缺调用意图）是否未超过 `success_criteria.max_tool_calls`；未配置时为 `null`。
- `ordered_tool_match`：预测工具名序列是否与参考序列完全一致。
- `tool_set_match`：预测与参考工具名集合是否一致，不考虑顺序和重复。
- `ordered_subsequence_match`：完整参考序列是否以正确相对顺序出现在预测轨迹中，允许穿插额外步骤。
- `tool_order_score`：预测序列与参考序列的最长公共子序列长度除以参考长度，用于区分完全错序和部分完成。

这些轨迹指标均为诊断项。合理的额外观察、日志查询或状态确认可能使 exact match 为假，但不应自动令 `task_success` 失败。

### 工具与参数

- `tool_precision` / `tool_recall` / `tool_f1`：按带重复次数的工具名计数，分别描述预测工具的有效性、所需工具覆盖率及其调和平均；`optional_tools` 不进入 recall 分母，每个可选项只豁免一次同名额外调用，更多同名调用仍计为 false positive。
- `argument_accuracy`：将预测调用与 `required_calls` 做最大一对一匹配后，参数按 `subset` 或 `exact` 规则匹配的比例。
- `valid_call_rate`：成功解析为合法调用的比例。
- `invalid_tool_calls`：无法解析为合法工具调用的数量。
- `unknown_tool_calls`：调用未注册工具的数量。
- `unknown_tool_rate`：未知工具执行数占 `tool_execution_count` 的比例。

无调用时，`valid_call_rate`、`execution_success_rate` 和 `adjusted_execution_success_rate` 记为 1，未知、重复、重试和非法状态跳转率记为 0；没有 `required_calls` 时 `argument_accuracy` 为 `null`。汇总会排除不适用的 `null`，并输出对应的 `*_evaluated_cases`，不要自行用总 case 数反推。

### 执行与效率

- `failed_tool_calls`：按工具真实返回统计的原始失败数，包括预期注入的临时故障。
- `expected_failed_calls`：case 中配置的预期故障次数；`observed_expected_failed_calls` 表示与注入操作、目标及错误类型匹配的实际故障数，`unexpected_failed_calls` 表示其余失败。
- `execution_success_rate`：按工具真实返回统计的原始执行成功率；失败包括预期注入的临时故障。
- `adjusted_execution_success_rate`：计算时忽略已匹配的注入故障，只保留普通执行错误的影响。
- `retry_calls`：同工具同参数的上一次执行失败后，在严格更晚的模型 step 再次调用的次数；同一轮批量发出的重复调用无法读取失败反馈，不算自适应 retry。
- `repeated_tool_calls`：在应用状态没有进展时重复已成功调用、同一模型 step 内的失败后重复，或连续多次失败后的额外重复；第一次跨 step retry、故障后跨 step 重新观察，以及中间已有成功导航/动作改变状态后的同名同参调用，不计为冗余。
- `invalid_transitions`：因当前页面或状态不满足前置条件而失败的导航/动作。
- 对应的 `retry_call_rate`、`repeated_tool_call_rate` 和 `invalid_transition_rate` 用于跨 case 比较。
- `tool_attempt_count` / `tool_call_count` / `tool_execution_count`：分别统计全部调用意图、其中语法有效且带名称的调用，以及实际拿到执行结果的调用。
- `num_steps`：Agent 生成轮数，而非纯工具调用数。
- 是否耗尽步数：查看运行结果的 `terminated_reason` 和 case 的 `max_steps`。

故障注入产生的第一次失败是 recovery 测试的一部分，不应仅凭原始失败次数大于 0 判定任务失败。报告时应同时保留 raw expected / unexpected failure 和调整后执行率，以免隐藏真实错误。

## Recovery 与故障注入

recovery case 在 `environment.fault_injections` 中声明可复现的临时故障：

```json
{
  "operation": "navigate",
  "target": "settings",
  "times": 1,
  "error": "transient_failure"
}
```

- `operation` 当前支持 `navigate` 和 `action`；
- `target` 对应页面或动作；
- `times` 是该匹配操作在成功前被强制失败的次数；
- 失败不改变应用状态，并写入合成日志；计数耗尽后，同一合法操作可正常执行。

v2 recovery case 的 `required_calls` 会同时列出预期失败的首次尝试和后续重试，因此工具 recall 与顺序指标不会把必要恢复步骤当作额外调用。

配置只传给环境重置逻辑，不出现在 system / user 消息中。模型必须从工具错误中识别失败、基于当前状态调整或重试，并最终满足 `success_criteria`。`recovery_success` 同时要求：匹配的预设故障确实被触发、严格更晚的模型 step 出现成功工具执行、最终任务成功；从未触发故障或在同一 step 预先批量调用不能算恢复成功。

## 汇总与分组

`aggregate_results(records)` 接收包含 `case` 与 `metrics` 的结果记录，返回 `overall`、`by_category`、`by_difficulty`、`by_scenario_family` 和 `category_macro`。三种 `by_*` 分组使用相同的 summary schema，包含参与计算的 case 数、平均值与总计。

`overall` 对 case 级 rate 做算术平均，因此类别样本量不同会影响总体值；计数项同时提供 total。`category_macro` 先得到各类别的核心 rate，再对可用类别等权平均。`by_category`、`by_difficulty` 和 `by_scenario_family` 则用于定位具体薄弱组。

跨类别对比优先使用 macro：先在每个类别内计算指标，再对类别等权平均，避免样本较多的类别掩盖 recovery 或长链路任务。报告结果时至少同时给出：

- 总 case 数与各类别 case 数；
- overall task success；
- category macro task success；
- long-horizon 与 recovery success；
- 工具 precision / recall、参数准确率；
- 合法调用率、执行成功率、未知工具与重复调用；
- 平均 Agent steps，以及按类别的分组结果。

不要只报告 `ordered_tool_match` 或 `tool_set_match`，也不要把二者称为端到端准确率。

## 运行指纹与结果隔离

每行 baseline 结果包含 `evaluator_version`、`run_metadata`、`case`、`result` 和 `metrics`。`run_metadata.fingerprint` 由整套运行元数据计算，覆盖：

- benchmark 文件 SHA-256、schema 版本、选中 case 数及 ID 摘要；
- Evaluator 版本，以及 Agent、环境、解析器、评估器、runner、工具和合成知识库共同构成的 runtime SHA-256；
- 本地模型解析路径与内容 SHA-256，或远程模型的固定 `model_revision`；
- Python、PyTorch、Transformers 版本；
- `do_sample`、`max_new_tokens`、step override 和默认 step 上限等生成配置。

`--resume` 会逐行校验同一输出文件：case ID 不能重复，每条记录都必须带有完全一致的运行指纹。模型、benchmark、代码、知识库、依赖版本、解码配置、`--limit` 选集等任一部分变化，都会拒绝续跑，防止结果静默混用。若文件最后一条是断电造成的、不完整且没有换行的 JSON，runner 只截掉该尾行后继续；中间损坏、完整但非法的行或旧版无指纹结果仍会报错。

汇总脚本也会拒绝同一文件中的重复 case、不同运行指纹或不同 Evaluator 版本。它会将当前记录数与 `selected_case_count` 对比，输出 run coverage，并对尚未跑完的 partial result 明确告警。`--recompute` 只用当前 Evaluator 重算指标，不会解除混跑检查；旧版无元数据记录可单独汇总，但不能与带指纹的新记录混在一起。

## 局限与解释边界

- 数据和环境均为合成，状态空间、工具集合和故障模式小于真实客户端与线上系统；分数不能直接外推为生产可靠性。
- `surface_01` / `surface_02` 是受控改写，不等价于开放域中文表达覆盖；同一 `scenario_id` 的两行共享执行语义，不能当作两个独立语义场景解释。
- 最终状态判定能容纳多条有效轨迹，但无法评价所有自然语言结论的事实完整性与表达质量。
- 参数匹配针对当前 schema 的关键字段，不代表复杂嵌套参数、模糊约束或跨工具数据流能力。
- 确定性故障注入适合可复现实验，但未覆盖延迟、并发、部分成功、服务抖动等真实分布。
- v2 是 held-out 集。若其内容、轨迹或派生改写进入训练，后续结果必须标注数据泄漏，且不得与未污染的 v2 分数直接比较。
