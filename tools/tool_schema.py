TOOLS = [
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
            "required": ["query"],
            "additionalProperties": False,
        }
    },
    {
        "name": "get_build_info",
        "description": "获取当前测试环境的应用版本和平台信息。",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
    },
    {
        "name": "inspect_ui_state",
        "description": "查看当前页面和关键应用状态。",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
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
                        "multiplayer_dungeon",
                    ],
                    "description": (
                        "目标页面。合法路径为 home -> settings -> graphics，"
                        "或 home -> dungeon_select -> multiplayer_dungeon。"
                    ),
                }
            },
            "required": ["target_page"],
            "additionalProperties": False,
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
                        "start_multiplayer_dungeon",
                    ],
                    "description": (
                        "动作名称：set_graphics 仅可在 graphics 页面执行且需"
                        "提供 value；back_home 可返回首页；"
                        "start_multiplayer_dungeon 仅可在 multiplayer_dungeon "
                        "页面执行。"
                    ),
                },
                "value": {
                    "enum": ["low", "standard", "high"],
                    "description": "set_graphics 的画质档位；其他动作无需提供。",
                }
            },
            "required": ["action"],
            "additionalProperties": False,
        }
    },
    {
        "name": "query_logs",
        "description": "查询当前复现流程产生的应用日志。",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
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
            "required": ["expected"],
            "additionalProperties": False,
        }
    }
]
