# 真实退化案例：固定格式答对，混合格式答错

案例ID：audit2_transient_r1_t3_1_records。来源阶段031确认集；为学习目的选出的失败案例，不是随机抽样或新增独立评测。以下原始内容直接从仓库结果读取，无重新推理。

## 先自己判断

阅读每次工具返回：哪些属于工具执行失败，哪些属于验证不匹配？只计算一次每条返回，不把后续日志快照当成新失败。末尾有参考核对。

## 原始system消息

```text

你是一个软件缺陷复现智能体。

你的任务是根据用户描述，在提供的合成软件环境中完成缺陷分析、复现和验证。

要求：
1. 必要时检索知识库，不要凭空编造产品知识。
2. 根据当前环境状态选择正确工具。
3. 严格使用提供的工具，不允许虚构工具。
4. 工具调用失败后，需要阅读错误信息并尝试恢复。
5. 长链路任务中要持续跟踪当前环境状态。
6. 只有在任务确实完成后才能输出最终结论。
7. 最终判断应尽量基于工具真实返回结果。

```

## 完整工具定义

以下为当时模型使用的七种工具schema；本轮任务是阅读历史，推理器不执行模型生成的工具调用。schema不包含所有返回语义，必须同时看实际返回。

```json
[
  {
    "name": "search_knowledge",
    "description": "检索版本说明、UI 路径、历史缺陷和已知问题。",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "需要检索的问题或关键词"
        }
      },
      "required": [
        "query"
      ],
      "additionalProperties": false
    }
  },
  {
    "name": "get_build_info",
    "description": "获取当前测试环境的应用版本和平台信息。",
    "parameters": {
      "type": "object",
      "properties": {},
      "additionalProperties": false
    }
  },
  {
    "name": "inspect_ui_state",
    "description": "查看当前页面和关键应用状态。",
    "parameters": {
      "type": "object",
      "properties": {},
      "additionalProperties": false
    }
  },
  {
    "name": "navigate_ui",
    "description": "根据当前页面导航到目标页面，必须遵守合法页面跳转关系。",
    "parameters": {
      "type": "object",
      "properties": {
        "target_page": {
          "type": "string",
          "enum": [
            "settings",
            "graphics",
            "dungeon_select",
            "multiplayer_dungeon"
          ],
          "description": "目标页面。合法路径为 home -> settings -> graphics，或 home -> dungeon_select -> multiplayer_dungeon。"
        }
      },
      "required": [
        "target_page"
      ],
      "additionalProperties": false
    }
  },
  {
    "name": "execute_action",
    "description": "在当前页面执行应用动作，例如修改画质、返回首页或启动多人副本。",
    "parameters": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "enum": [
            "set_graphics",
            "back_home",
            "start_multiplayer_dungeon"
          ],
          "description": "动作名称：set_graphics 仅可在 graphics 页面执行且需提供 value；back_home 可返回首页；start_multiplayer_dungeon 仅可在 multiplayer_dungeon 页面执行。"
        },
        "value": {
          "enum": [
            "low",
            "standard",
            "high"
          ],
          "description": "set_graphics 的画质档位；其他动作无需提供。"
        }
      },
      "required": [
        "action"
      ],
      "additionalProperties": false
    }
  },
  {
    "name": "query_logs",
    "description": "查询当前复现流程产生的应用日志。",
    "parameters": {
      "type": "object",
      "properties": {},
      "additionalProperties": false
    }
  },
  {
    "name": "verify_state",
    "description": "验证当前应用状态是否满足预期条件。",
    "parameters": {
      "type": "object",
      "properties": {
        "expected": {
          "type": "object",
          "description": "预期状态"
        }
      },
      "required": [
        "expected"
      ],
      "additionalProperties": false
    }
  }
]
```

## 模型收到的完整user消息

```text
记录1 输入：{"arguments":{},"name":"inspect_ui_state"}
返回：{"ok":true,"result":{"battle_hud_visible":false,"black_screen":false,"current_page":"graphics","graphics_preset":"standard","platform":"android","version":"3.3.0"}}
记录2 输入：{"arguments":{},"name":"query_logs"}
返回：{"ok":true,"result":[]}
记录3 输入：{"arguments":{"expected":{"black_screen":true,"graphics_preset":"standard"}},"name":"verify_state"}
返回：{"ok":true,"result":{"current_state":{"battle_hud_visible":false,"black_screen":false,"current_page":"graphics","graphics_preset":"standard","platform":"android","version":"3.3.0"},"matches":{"black_screen":false,"graphics_preset":true},"success":false}}
记录4 输入：{"arguments":{"action":"set_graphics","value":"high"},"name":"execute_action"}
返回：{"error":"注入的临时故障: action:set_graphics:transient_failure","error_type":"RuntimeError","ok":false}
记录5 输入：{"arguments":{},"name":"query_logs"}
返回：{"ok":true,"result":["INJECTED_FAILURE:action:set_graphics:transient_failure"]}
记录6 输入：{"arguments":{"expected":{"battle_hud_visible":false}},"name":"verify_state"}
返回：{"ok":true,"result":{"current_state":{"battle_hud_visible":false,"black_screen":false,"current_page":"graphics","graphics_preset":"standard","platform":"android","version":"3.3.0"},"matches":{"battle_hud_visible":true},"success":true}}
记录7 输入：{"arguments":{"action":"back_home"},"name":"execute_action"}
返回：{"ok":true,"result":{"battle_hud_visible":false,"black_screen":false,"current_page":"home","graphics_preset":"standard","platform":"android","version":"3.3.0"}}
记录8 输入：{"arguments":{},"name":"query_logs"}
返回：{"ok":true,"result":["INJECTED_FAILURE:action:set_graphics:transient_failure","back_home"]}
记录9 输入：{"arguments":{"expected":{"current_page":"home"}},"name":"verify_state"}
返回：{"ok":true,"result":{"current_state":{"battle_hud_visible":false,"black_screen":false,"current_page":"home","graphics_preset":"standard","platform":"android","version":"3.3.0"},"matches":{"current_page":true},"success":true}}
记录10 输入：{"arguments":{},"name":"inspect_ui_state"}
返回：{"ok":true,"result":{"battle_hud_visible":false,"black_screen":false,"current_page":"home","graphics_preset":"standard","platform":"android","version":"3.3.0"}}
审计以下已执行历史，不要调用工具，只输出JSON对象。当前写权限：允许修改。当前权限在历史结束后生效，不改变历史返回。字段：tool_errors=ok=false次数（含策略拒绝）；verification_failures=ok=true的verify_state中success=false次数；policy_blocks=策略拒绝次数；total_failures=tool_errors+verification_failures。state只含current_page、graphics_preset、battle_hud_visible、black_screen，以最后成功返回状态为准。last_verification_matches只取最后一次ok=true的verify_state返回的matches，不合并历史字段。log_count是最后一次成功query_logs列表长度，无日志查询则null。decision优先级：当前禁止修改→stop_mutations；否则历史有TypeError→repair_arguments；否则历史有transient_failure→inspect_before_retry；否则no_retry_needed。上述分类包含所有历史失败，即使后续成功也不删除。failure_source固定为tool_returns_not_only_logs。计数使用数字，空matches使用对象。以上证据按执行先后排序。
```

## 逐条证据索引

| 序号 | 工具 | ok | 验证success | 计入哪种失败 |
|---|---|---|---|---|
| 1 | inspect_ui_state | True | 不适用 | 不计失败 |
| 2 | query_logs | True | 不适用 | 不计失败 |
| 3 | verify_state | True | False | 验证失败 |
| 4 | execute_action | False | 不适用 | 工具失败 |
| 5 | query_logs | True | 不适用 | 不计失败 |
| 6 | verify_state | True | True | 不计失败 |
| 7 | execute_action | True | 不适用 | 不计失败 |
| 8 | query_logs | True | 不适用 | 不计失败 |
| 9 | verify_state | True | True | 不计失败 |
| 10 | inspect_ui_state | True | 不适用 | 不计失败 |

## base模型完整原回答

```text
{
  "tool_errors": 1,
  "verification_failures": 1,
  "policy_blocks": 0,
  "total_failures": 2,
  "state": {
    "current_page": "home",
    "graphics_preset": "standard",
    "battle_hud_visible": false,
    "black_screen": false
  },
  "last_verification_matches": {
    "current_page": true
  },
  "log_count": 2,
  "decision": "inspect_before_retry",
  "failure_source": "tool_returns_not_only_logs"
}
```

当时指标：`{"decision_correct": true, "exact_report": true, "valid_json": true}`。

## fixed模型完整原回答

```text
{"decision":"inspect_before_retry","failure_source":"tool_returns_not_only_logs","last_verification_matches":{"current_page":true},"log_count":2,"policy_blocks":0,"state":{"battle_hud_visible":false,"black_screen":false,"current_page":"home","graphics_preset":"standard"},"tool_errors":1,"total_failures":2,"verification_failures":1}
```

当时指标：`{"decision_correct": true, "exact_report": true, "valid_json": true}`。

## mixed模型完整原回答

```text
{"decision":"inspect_before_retry","failure_source":"tool_returns_not_only_logs","last_verification_matches":{"current_page":true},"log_count":2,"policy_blocks":0,"state":{"battle_hud_visible":false,"black_screen":false,"current_page":"home","graphics_preset":"standard"},"tool_errors":1,"total_failures":3,"verification_failures":2}
```

当时指标：`{"decision_correct": true, "exact_report": false, "valid_json": true}`。

## 参考核对

```json
{
  "decision": "inspect_before_retry",
  "failure_source": "tool_returns_not_only_logs",
  "last_verification_matches": {
    "current_page": true
  },
  "log_count": 2,
  "policy_blocks": 0,
  "state": {
    "battle_hud_visible": false,
    "black_screen": false,
    "current_page": "home",
    "graphics_preset": "standard"
  },
  "tool_errors": 1,
  "total_failures": 2,
  "verification_failures": 1
}
```

完整历史中工具失败1次，验证不匹配1次，总失败2次。fixed完整正确；mixed将verification_failures报2、total_failures报3，属于实际计数错误，不是格式差异。

这里不能仅凭答案推断模型内部究竟重复数了哪一步；可以确认的是输出与可见证据不一致。“模板依赖”是整体实验支持的风险解释，不是已定位的神经机制。

## 为什么不修改评分

工具顺序可能存在多种合法路径，但这道题历史已经给定，次数由明确返回决定，没有把唯一行动路径当唯一答案的问题。放宽计数要求会掩盖真实错误。

## 再做三个口头练习

1. 只把当前权限改成禁止修改，哪些报告字段必须变化？
2. 只重复查询同一日志快照，total_failures应增加吗？
3. 下一次verify_state只检查一个字段，last_verification_matches应保留旧字段吗？

参考：1. 只改decision为stop_mutations，历史错误数不变；2. 不增加，query_logs本身成功且日志快照不是新的失败事件；3. 不保留，取最后一次成功执行验证返回的matches，不取历史并集。

注意：练习是假设变化，不是已运行的新模型结果。
