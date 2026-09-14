import json

from tools.tool_schema import TOOLS
from tools.executor import execute_tool_json
from tools.environment_tools import inspect_ui_state, reset_environment
from agent.tool_parser import parse_tool_calls


SYSTEM_PROMPT = """
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
"""


class BaselineAgent:
    def __init__(
        self,
        model_path,
        max_steps=12,
        max_new_tokens=512,
        revision=None,
        adapter_path=None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self._torch = torch

        revision_kwargs = (
            {"revision": revision}
            if revision is not None
            else {}
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            **revision_kwargs,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            **revision_kwargs,
        )

        self.model.eval()
        if adapter_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()

    def generate(self, messages):
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.model.device)
            for k, v in inputs.items()
        }

        with self._torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated = output[0][
            inputs["input_ids"].shape[-1]:
        ]

        decoded = self.tokenizer.decode(
            generated,
            skip_special_tokens=False,
        )
        # Chat templates add turn terminators themselves. Avoid duplicate EOS
        # in subsequent prompts and keep SFT/runtime serialization consistent.
        eos = self.tokenizer.eos_token
        if eos:
            while decoded.rstrip().endswith(eos):
                decoded = decoded.rstrip()[:-len(eos)]
        return decoded

    def run(self, query, environment=None, max_steps=None):
        """Run one case while keeping the original ``run(query)`` API valid.

        The optional environment configuration is benchmark metadata and is
        never included in the model messages.  It controls only the synthetic
        app reset and deterministic fault injection used by recovery cases.
        """

        reset_environment(environment)

        effective_max_steps = (
            self.max_steps
            if max_steps is None
            else max_steps
        )
        if effective_max_steps < 1:
            raise ValueError("max_steps 必须大于 0。")

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        trajectory = []
        final_answer = None

        for step in range(effective_max_steps):
            response = self.generate(messages)

            parsed = parse_tool_calls(response)

            record = {
                "step": step,
                "model_output": response,
                "parsed_tool_calls": parsed,
                "tool_results": [],
            }

            trajectory.append(record)

            valid_calls = [
                x["tool_call"]
                for x in parsed
                if x.get("valid")
            ]

            if not valid_calls:
                final_answer = response
                break

            messages.append({
                "role": "assistant",
                "content": response,
            })

            for call in valid_calls:
                result = execute_tool_json(call)

                record["tool_results"].append({
                    "tool_call": call,
                    "result": result,
                })

                messages.append({
                    "role": "tool",
                    "content": json.dumps(
                        result,
                        ensure_ascii=False,
                    ),
                })

        return {
            "query": query,
            "trajectory": trajectory,
            "final_answer": final_answer,
            "final_environment_state": inspect_ui_state(),
            "terminated_reason": (
                "final_answer"
                if final_answer is not None
                else "max_steps"
            ),
        }
