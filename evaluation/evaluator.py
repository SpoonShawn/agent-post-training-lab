def extract_tools(result):
    tools = []

    for step in result["trajectory"]:
        for item in step["parsed_tool_calls"]:
            if item.get("valid"):
                name = item["tool_call"].get("name")
                if name:
                    tools.append(name)

    return tools


def evaluate_case(case, result):
    predicted = extract_tools(result)
    expected = case.get("expected_tools")

    invalid_calls = 0
    failed_calls = 0
    unknown_tools = 0

    for step in result["trajectory"]:
        for item in step["parsed_tool_calls"]:
            if not item.get("valid"):
                invalid_calls += 1

        for item in step["tool_results"]:
            tool_result = item["result"]

            if not tool_result.get("ok"):
                failed_calls += 1

            if tool_result.get("error_type") == "unknown_tool":
                unknown_tools += 1

    metrics = {
        "predicted_tools": predicted,
        "num_steps": len(result["trajectory"]),
        "invalid_tool_calls": invalid_calls,
        "failed_tool_calls": failed_calls,
        "unknown_tool_calls": unknown_tools,
    }

    if expected is not None:
        metrics["ordered_tool_match"] = predicted == expected
        metrics["tool_set_match"] = set(predicted) == set(expected)

    return metrics
