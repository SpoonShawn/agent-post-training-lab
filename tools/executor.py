import json

from tools.knowledge_tool import search_knowledge
from tools.environment_tools import (
    get_build_info,
    inspect_ui_state,
    navigate_ui,
    execute_action,
    query_logs,
    verify_state,
)

TOOL_FUNCTIONS = {
    "search_knowledge": search_knowledge,
    "get_build_info": get_build_info,
    "inspect_ui_state": inspect_ui_state,
    "navigate_ui": navigate_ui,
    "execute_action": execute_action,
    "query_logs": query_logs,
    "verify_state": verify_state,
}


def execute_tool(name, arguments):
    if name not in TOOL_FUNCTIONS:
        return {
            "ok": False,
            "error_type": "unknown_tool",
            "error": f"不存在工具: {name}",
        }

    try:
        result = TOOL_FUNCTIONS[name](**arguments)
        return {
            "ok": True,
            "result": result,
        }
    except Exception as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error": str(e),
        }


def execute_tool_json(tool_call):
    name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})

    if not isinstance(arguments, dict):
        try:
            arguments = json.loads(arguments)
        except Exception:
            return {
                "ok": False,
                "error_type": "invalid_arguments",
                "error": "arguments 无法解析为 JSON object",
            }

    return execute_tool(name, arguments)
