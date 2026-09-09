import json
from pathlib import Path

from agent.baseline_runner import BaselineAgent
from evaluation.evaluator import evaluate_case


MODEL_PATH = "/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507"

EVAL_PATH = Path(
    "data/eval/bugops_eval_v1.jsonl"
)

OUTPUT_PATH = Path(
    "results/baseline/qwen3_4b_baseline.jsonl"
)


def load_cases():
    cases = []

    with open(
        EVAL_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            cases.append(
                json.loads(line)
            )

    return cases


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    agent = BaselineAgent(
        MODEL_PATH,
        max_steps=12,
    )

    cases = load_cases()

    records = []

    for idx, case in enumerate(cases):
        print("=" * 70)
        print(
            f"[{idx + 1}/{len(cases)}] "
            f"{case['id']}"
        )
        print(case["query"])

        result = agent.run(
            case["query"]
        )

        metrics = evaluate_case(
            case,
            result,
        )

        print(
            "预测工具:",
            metrics["predicted_tools"],
        )

        print(
            "期望工具:",
            case.get("expected_tools"),
        )

        print(
            "指标:",
            metrics,
        )

        records.append({
            "case": case,
            "result": result,
            "metrics": metrics,
        })

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print(
        "结果已保存:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
