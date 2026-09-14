import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.scoring import dataset_protocol, evaluate_case  # noqa: E402


DEFAULT_MODEL_PATH = os.environ.get(
    "BUGOPS_MODEL_PATH",
    "/home/zshaoaj/hf_models/Qwen3-4B-Instruct-2507",
)
EVAL_PATH = ROOT / "data" / "eval" / "bugops_eval_v2.jsonl"
OUTPUT_PATH = (
    ROOT
    / "results"
    / "baseline"
    / "qwen3_4b_baseline_v2.jsonl"
)
RUN_METADATA_VERSION = 1
DEFAULT_MAX_STEPS = 16
DEFAULT_MAX_NEW_TOKENS = 512
RUNTIME_FILES = (
    "agent/baseline_runner.py",
    "agent/environment.py",
    "agent/tool_parser.py",
    "data/knowledge/build_notes.json",
    "data/knowledge/incident_history.json",
    "data/knowledge/ui_guide.json",
    "evaluation/evaluator.py",
    "evaluation/protocol_v21.py",
    "evaluation/protocol_v22.py",
    "evaluation/scoring.py",
    "scripts/run_baseline.py",
    "tools/environment_tools.py",
    "tools/executor.py",
    "tools/knowledge_tool.py",
    "tools/tool_schema.py",
)


def load_cases(path=EVAL_PATH, limit=None):
    cases = []
    seen_ids = set()

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number} 不是合法 JSON: {exc}"
                ) from exc

            if not isinstance(case, dict):
                raise ValueError(
                    f"{path}:{line_number} 必须是 JSON object。"
                )
            if "id" not in case or "query" not in case:
                raise ValueError(
                    f"{path}:{line_number} 缺少 id 或 query。"
                )
            if not isinstance(case["id"], str) or not case["id"]:
                raise ValueError(
                    f"{path}:{line_number} 的 id 必须是非空字符串。"
                )
            if case["id"] in seen_ids:
                raise ValueError(
                    f"{path}:{line_number} 的 id 重复: {case['id']}"
                )
            if not isinstance(case["query"], str) or not case["query"].strip():
                raise ValueError(
                    f"{path}:{line_number} 的 query 必须是非空字符串。"
                )
            seen_ids.add(case["id"])
            cases.append(case)

            if limit is not None and len(cases) >= limit:
                break

    return cases


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_model_path(model_path):
    """Hash local model contents so resume cannot mix changed checkpoints."""

    path = Path(model_path).expanduser()
    if not path.exists():
        return None, None

    resolved = path.resolve()
    if resolved.is_file():
        return sha256_file(resolved), str(resolved)
    if not resolved.is_dir():
        return None, str(resolved)

    digest = hashlib.sha256()
    files = sorted(
        item
        for item in resolved.rglob("*")
        if item.is_file()
    )
    for item in files:
        relative = item.relative_to(resolved).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest(), str(resolved)


def runtime_sha256():
    digest = hashlib.sha256()
    for relative_path in RUNTIME_FILES:
        path = ROOT / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def package_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_run_metadata(args, cases):
    model_content_hash, resolved_model_path = sha256_model_path(
        args.model_path
    )
    selected_ids = [case["id"] for case in cases]
    selection_digest = hashlib.sha256(
        json.dumps(
            selected_ids,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    schema_versions = sorted({
        str(case.get("schema_version", "v1"))
        for case in cases
    })
    metadata = {
        "metadata_version": RUN_METADATA_VERSION,
        "model_path": str(args.model_path),
        "model_revision": getattr(args, "model_revision", None),
        "model_resolved_path": resolved_model_path,
        "model_content_sha256": model_content_hash,
        "benchmark_sha256": sha256_file(args.eval_path),
        "benchmark_schema_versions": schema_versions,
        "selected_case_count": len(cases),
        "selected_case_ids_sha256": selection_digest,
        "evaluator_version": dataset_protocol(cases),
        "runtime_sha256": runtime_sha256(),
        "runtime_versions": {
            "python": platform.python_version(),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
        },
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "max_steps_override": args.max_steps,
            "default_max_steps": DEFAULT_MAX_STEPS,
        },
    }
    fingerprint_payload = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    metadata["fingerprint"] = hashlib.sha256(
        fingerprint_payload
    ).hexdigest()
    return metadata


def load_completed_ids(path, expected_fingerprint=None):
    completed = set()
    path = Path(path)
    if not path.exists():
        return completed

    raw_lines = path.read_bytes().splitlines(keepends=True)
    valid_byte_count = 0
    repaired_tail = False
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            valid_byte_count += len(raw_line)
            continue
        try:
            line = raw_line.decode("utf-8")
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            is_truncated_tail = (
                line_number == len(raw_lines)
                and not raw_line.endswith((b"\n", b"\r"))
            )
            if is_truncated_tail:
                with path.open("r+b") as handle:
                    handle.truncate(valid_byte_count)
                repaired_tail = True
                print(
                    f"已丢弃 {path} 中未写完的最后一条记录，继续续跑。"
                )
                break
            raise ValueError(
                f"无法从 {path}:{line_number} 恢复: {exc}"
            ) from exc

        if not isinstance(record, dict):
            raise ValueError(
                f"无法从 {path}:{line_number} 恢复: 记录必须是 object。"
            )
        try:
            case = record["case"]
            if not isinstance(case, dict):
                raise TypeError("case 必须是 object")
            case_id = case["id"]
            if not isinstance(case_id, str) or not case_id:
                raise TypeError("case.id 必须是非空字符串")
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"无法从 {path}:{line_number} 恢复: {exc}"
            ) from exc

        if case_id in completed:
            raise ValueError(
                f"无法从 {path} 恢复: case {case_id} 重复。"
            )
        if expected_fingerprint is not None:
            metadata = record.get("run_metadata")
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"无法安全续跑 {path}: 旧结果缺少 run_metadata。"
                )
            observed = metadata.get("fingerprint")
            if observed is None:
                raise ValueError(
                    f"无法安全续跑 {path}: 旧结果缺少运行指纹。"
                )
            if observed != expected_fingerprint:
                raise ValueError(
                    f"无法安全续跑 {path}: 运行配置指纹不一致。"
                )
        completed.add(case_id)
        valid_byte_count += len(raw_line)

    # A valid final JSON object without a newline must be separated from the
    # next append, otherwise the resumed record would corrupt both objects.
    if (
        raw_lines
        and not repaired_tail
        and valid_byte_count == path.stat().st_size
    ):
        last_line = raw_lines[-1]
        if last_line.strip() and not last_line.endswith((b"\n", b"\r")):
            with path.open("ab") as handle:
                handle.write(b"\n")

    return completed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="运行 Qwen3 BugOps held-out benchmark。",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="本地 Hugging Face 模型目录；也可设置 BUGOPS_MODEL_PATH。",
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=EVAL_PATH,
        help="JSONL benchmark 路径。",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="远程模型的固定 commit revision；本地目录通常无需设置。",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help="逐条写入的评测结果 JSONL 路径。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="覆盖所有 case 的最大 Agent 步数。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="每轮生成的最大 token 数。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只运行前 N 条，适合先做 smoke test。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="保留已有输出并跳过已完成的 case。",
    )
    args = parser.parse_args(argv)

    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps 必须大于 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须大于 0")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens 必须大于 0")

    return args


def main(argv=None):
    args = parse_args(argv)
    if args.output_path.exists() and not args.resume:
        raise FileExistsError("输出已存在；请指定新路径或用 --resume，禁止覆盖历史实验。")
    cases = load_cases(args.eval_path, args.limit)
    dataset_protocol(cases)
    print("正在计算 benchmark、运行代码和模型内容指纹……")
    run_metadata = build_run_metadata(args, cases)
    if (
        args.resume
        and run_metadata["model_content_sha256"] is None
        and not args.model_revision
    ):
        raise ValueError(
            "远程模型续跑必须通过 --model-revision 固定 commit，"
            "以避免混合不同权重。"
        )
    completed_ids = (
        load_completed_ids(
            args.output_path,
            expected_fingerprint=run_metadata["fingerprint"],
        )
        if args.resume
        else set()
    )

    pending = [
        case
        for case in cases
        if case["id"] not in completed_ids
    ]
    if not pending:
        print("没有待运行的 case。")
        return

    from agent.baseline_runner import BaselineAgent

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if args.resume else "w"

    agent = BaselineAgent(
        args.model_path,
        max_steps=args.max_steps or DEFAULT_MAX_STEPS,
        max_new_tokens=args.max_new_tokens,
        revision=args.model_revision,
    )

    with args.output_path.open(file_mode, encoding="utf-8") as handle:
        for index, case in enumerate(pending, start=1):
            print("=" * 70)
            print(f"[{index}/{len(pending)}] {case['id']}")
            print(case["query"])

            case_max_steps = (
                args.max_steps
                if args.max_steps is not None
                else case.get("max_steps")
            )
            result = agent.run(
                case["query"],
                environment=case.get("environment"),
                max_steps=case_max_steps,
            )
            metrics = evaluate_case(case, result)

            print("预测工具:", metrics["predicted_tools"])
            print("期望工具:", case.get("expected_tools"))
            print("任务成功:", metrics.get("task_success"))
            print("指标:", metrics)

            record = {
                "evaluator_version": run_metadata["evaluator_version"],
                "run_metadata": run_metadata,
                "case": case,
                "result": result,
                "metrics": metrics,
            }
            handle.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            handle.flush()

    print()
    print("结果已保存:", args.output_path)


if __name__ == "__main__":
    main()
