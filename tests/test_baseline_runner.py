import unittest

from agent.baseline_runner import BaselineAgent


def tool_call(name, arguments):
    import json

    payload = json.dumps(
        {"name": name, "arguments": arguments},
        ensure_ascii=False,
    )
    return f"<tool_call>{payload}</tool_call>"


class ScriptedAgent(BaselineAgent):
    def __init__(self, outputs, max_steps=12):
        self.outputs = iter(outputs)
        self.max_steps = max_steps
        self.max_new_tokens = 1

    def generate(self, messages):
        return next(self.outputs)


class BaselineRunnerCompatibilityTests(unittest.TestCase):
    def test_original_run_query_signature_still_works(self):
        agent = ScriptedAgent([
            tool_call("get_build_info", {}),
            "当前为 Android 2.3.1。",
        ])

        result = agent.run("当前是什么构建？")

        tool_result = result["trajectory"][0]["tool_results"][0]["result"]
        self.assertEqual(
            tool_result["result"],
            {"version": "2.3.1", "platform": "android"},
        )
        self.assertEqual(result["terminated_reason"], "final_answer")
        self.assertEqual(
            result["final_environment_state"]["current_page"],
            "home",
        )

    def test_case_environment_is_hidden_but_applied(self):
        agent = ScriptedAgent([
            tool_call("get_build_info", {}),
            "当前为 iOS 2.3.1。",
        ])

        result = agent.run(
            "当前是什么构建？",
            environment={
                "initial_state": {
                    "version": "2.3.1",
                    "platform": "ios",
                },
            },
        )

        tool_result = result["trajectory"][0]["tool_results"][0]["result"]
        self.assertEqual(tool_result["result"]["platform"], "ios")

    def test_per_case_step_limit_reports_exhaustion(self):
        agent = ScriptedAgent(
            [tool_call("inspect_ui_state", {}) for _ in range(3)],
            max_steps=12,
        )

        result = agent.run("持续检查", max_steps=2)

        self.assertEqual(len(result["trajectory"]), 2)
        self.assertIsNone(result["final_answer"])
        self.assertEqual(result["terminated_reason"], "max_steps")


if __name__ == "__main__":
    unittest.main()
