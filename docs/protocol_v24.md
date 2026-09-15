# 协议2.4：明确的最终状态观测与日志时机

## 为什么要新增，而不是改旧分

阶段019发现只查空日志的只读任务也能通过2.3执行评分。原因是后台终态正确，但模型没有读取用户要求的HUD/黑屏。2.4补上“取得证据”条件，不把后台真值当模型知识。

2.0–2.3原文件不改，旧runner与指纹保持原样。新入口evaluation/protocol_v24.py继承2.3的状态、顺序、禁令、恢复要求，再增加观测/日志条件。新题必须显式声明2.4并公开相应要求；旧题不能静默升级。原18/120、82/120和作者复核6/120、59/120均保留，不用新协议替换旧分。

## 新契约

每题必须含observation_contract，只有两个字段：

```json
{
  "final_state_fields": ["current_page", "graphics_preset", "battle_hud_visible", "black_screen"],
  "logs_after_last_mutation": true
}
```

final_state_fields必须是非空、不重复、已知类型的字段，success_criteria.final_state同时给出期望值。字段要求从契约读取，不从模型答案/题面关键词猜测。生成器将同样要求写到题面，测试检查所有新题都含公开约定。

- **模型可见状态**：成功返回的inspect、verify.current_state、navigate/execute状态均可；get_build_info可提供其实际含有的构建字段。接受等价证据，不要求唯一工具。
- **观测新鲜度**：成功的导航/动作使此前状态证据失效；本次动作返回本身可提供新状态。不要求多余的末尾inspect。
- **最终日志**：当logs_after_last_mutation=true，须在最后一次导航/应用动作尝试之后成功query_logs，包含失败尝试；只读没有动作时，成功查询日志即可，允许空列表。
- **后台记录**：final_environment_state仍用于检查环境目标，但绝不作为模型取得观测的证据。

新日志时机按实际顺序判断，同一模型轮里顺序执行的动作→日志可满足时机；故障后观察再重试依然沿用2.2跨模型轮要求，以保证重试能依据新的返回。

## 输出指标

保留execution_success_v23作为内部诊断快照，新增observation_success、log_timing_success、observation_violations及带工具/步骤的证据。最终execution_success要求旧执行条件、新观测、新日志条件全部通过。

task_success仍是执行失败时false，执行通过时null等待答案审查。数字和状态一致不自动批准自然语言结论。当前新汇总只发布执行和配对统计，不导入旧review，也不声称已做独立答案评审。

## 工程边界

2.4使用独立run_control_v1与summarize_control_v1；旧summarize_baseline遇到2.4会明确拒绝，防止错误地按旧规则算新题。新运行指纹包含旧runtime SHA、新扩展SHA、题集与manifest SHA、模型/adapter内容、依赖和生成配置。旧运行字节未变，所以旧冻结验证继续通过。

当前环境同步、失败原子化且无外部并发。真实应用的部分成功、延迟状态、跨进程修改需要另一套证据新鲜度模型，不能直接套用。2.4不是通用自然语言评判器，也不保证所有未来题面都与契约自动一致。

## 验证反例

只查日志/只有隐藏终态失败；过早查询日志失败；最后一次失败动作之后未重新取日志失败；动作自带终态或verify等价通过；未知/缺失契约、字段重复/类型错误、query错配拒绝；76条oracle全部执行通过但答案仍待审。
