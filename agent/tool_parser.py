import json
import re


PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)


def parse_tool_calls(text):
    matches = PATTERN.findall(text)
    calls = []

    for raw in matches:
        try:
            obj = json.loads(raw)
            calls.append({
                "valid": True,
                "raw": raw,
                "tool_call": obj,
            })
        except Exception as e:
            calls.append({
                "valid": False,
                "raw": raw,
                "error": str(e),
            })

    return calls
