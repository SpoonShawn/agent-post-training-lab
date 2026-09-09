import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.generate_benchmark_v2 import (
    DEFAULT_OUTPUT,
    EXPECTED_CASE_COUNT,
    EXPECTED_CATEGORY_COUNTS,
    FAMILY_ORDER,
    SEMANTIC_SCENARIOS_PER_FAMILY,
    SURFACES_PER_SCENARIO,
    VARIANTS_PER_FAMILY,
    build_cases,
    case_signature,
    final_answer_matches,
    serialize_cases,
    validate_cases,
    validate_reference_oracles,
    write_benchmark,
)


class BenchmarkV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = build_cases()

    def test_schema_counts_and_unique_held_out_metadata(self):
        validate_cases(self.cases)

        self.assertEqual(len(self.cases), EXPECTED_CASE_COUNT)
        self.assertEqual(len({case["id"] for case in self.cases}), 360)
        self.assertEqual(len({case["query"] for case in self.cases}), 360)
        self.assertEqual(
            Counter(case["scenario_family"] for case in self.cases),
            Counter({family: VARIANTS_PER_FAMILY for family in FAMILY_ORDER}),
        )
        self.assertEqual(
            Counter(case["category"] for case in self.cases),
            Counter(EXPECTED_CATEGORY_COUNTS),
        )
        self.assertTrue(all(case["schema_version"] == 2 for case in self.cases))
        self.assertTrue(all(case["split"] == "held_out" for case in self.cases))

    def test_semantic_scenarios_and_execution_signatures_are_diverse(self):
        scenario_counts = Counter(case["scenario_id"] for case in self.cases)
        signature_counts = Counter(case_signature(case) for case in self.cases)

        self.assertEqual(
            len(scenario_counts),
            len(FAMILY_ORDER) * SEMANTIC_SCENARIOS_PER_FAMILY,
        )
        self.assertEqual(set(scenario_counts.values()), {SURFACES_PER_SCENARIO})
        self.assertEqual(len(signature_counts), 180)
        self.assertLessEqual(max(signature_counts.values()), 2)

        for family in FAMILY_ORDER:
            family_cases = [
                case for case in self.cases
                if case["scenario_family"] == family
            ]
            self.assertEqual(
                len({case["scenario_id"] for case in family_cases}),
                SEMANTIC_SCENARIOS_PER_FAMILY,
            )
            self.assertEqual(
                len({case_signature(case) for case in family_cases}),
                SEMANTIC_SCENARIOS_PER_FAMILY,
            )

        grouped = {}
        for case in self.cases:
            grouped.setdefault(case["scenario_id"], []).append(case)
        for scenario_id, rows in grouped.items():
            self.assertEqual(
                {row["surface_variant"] for row in rows},
                {"surface_01", "surface_02"},
                scenario_id,
            )
            self.assertEqual(
                len({case_signature(row) for row in rows}),
                1,
                scenario_id,
            )

    def test_output_is_interleaved_for_useful_smoke_limits(self):
        self.assertEqual(
            [case["scenario_family"] for case in self.cases[:len(FAMILY_ORDER)]],
            FAMILY_ORDER,
        )
        self.assertEqual(
            [case["scenario_family"] for case in self.cases[:5]],
            [
                "no_tool", "build_info", "ui_state",
                "knowledge_black_screen", "knowledge_hud",
            ],
        )

    def test_required_tool_views_come_from_executable_calls(self):
        for case in self.cases:
            names = [call["name"] for call in case["required_calls"]]
            self.assertEqual(case["expected_tools"], names, case["id"])
            self.assertEqual(case["required_tools"], names, case["id"])

        no_tool_cases = [
            case for case in self.cases
            if case["scenario_family"] == "no_tool"
        ]
        self.assertTrue(all(not case["required_calls"] for case in no_tool_cases))
        self.assertTrue(all(not case["expected_tools"] for case in no_tool_cases))
        self.assertTrue(all(
            case["success_criteria"].get("max_tool_calls") == 0
            for case in no_tool_cases
        ))

    def test_recovery_cases_encode_expected_failure_and_retry(self):
        recovery_cases = [case for case in self.cases if case["requires_recovery"]]
        self.assertEqual(len(recovery_cases), 60)

        for case in recovery_cases:
            fault = case["environment"]["fault_injections"][0]
            self.assertEqual(fault["times"], 1)
            self.assertEqual(fault["error"], "transient_failure")

            matching_calls = []
            for call in case["required_calls"]:
                if fault["operation"] == "navigate":
                    matches = (
                        call["name"] == "navigate_ui"
                        and call["arguments"].get("target_page") == fault["target"]
                    )
                else:
                    matches = (
                        call["name"] == "execute_action"
                        and call["arguments"].get("action") == fault["target"]
                    )
                if matches:
                    matching_calls.append(call)

            self.assertGreaterEqual(len(matching_calls), 2, case["id"])
            self.assertEqual(matching_calls[0], matching_calls[1], case["id"])

    def test_all_reference_oracles_are_satisfiable(self):
        validate_reference_oracles(self.cases)

    def test_final_answer_polarity_rejects_contradictory_conclusions(self):
        by_family = {
            family: next(
                case for case in self.cases
                if case["scenario_family"] == family
            )
            for family in (
                "reproduce_black_screen",
                "negative_control_ios",
                "workaround_black_screen",
            )
        }

        repro = by_family["reproduce_black_screen"]["success_criteria"]["final_answer"]
        self.assertTrue(final_answer_matches("复现成功，最终出现了黑屏。", repro))
        self.assertFalse(final_answer_matches("复现成功，但最终未复现。", repro))
        self.assertFalse(final_answer_matches("复现成功，但未出现黑屏。", repro))

        negative = by_family["negative_control_ios"]["success_criteria"]["final_answer"]
        self.assertTrue(final_answer_matches("未复现，战斗场景正常加载。", negative))
        self.assertFalse(final_answer_matches("正常加载，但已复现并发生黑屏。", negative))
        self.assertFalse(final_answer_matches("正常加载，但出现黑屏。", negative))

        workaround = by_family["workaround_black_screen"]["success_criteria"]["final_answer"]
        self.assertTrue(final_answer_matches("规避成功，战斗场景正常加载。", workaround))
        self.assertFalse(final_answer_matches("规避成功，但规避失败且仍然黑屏。", workaround))

        normal_return = next(
            case for case in self.cases
            if case["scenario_family"] == "return_home_then_dungeon"
            and case["success_criteria"]["final_state"]["black_screen"] is False
        )["success_criteria"]["final_answer"]
        self.assertTrue(final_answer_matches("没有复现，加载正常。", normal_return))
        self.assertFalse(final_answer_matches("正常加载，但发生黑屏。", normal_return))

    def test_checked_in_jsonl_is_deterministic_and_parseable(self):
        expected = serialize_cases(self.cases)
        self.assertTrue(DEFAULT_OUTPUT.exists())
        self.assertEqual(DEFAULT_OUTPUT.read_text(encoding="utf-8"), expected)

        parsed = [
            json.loads(line)
            for line in expected.splitlines()
            if line.strip()
        ]
        self.assertEqual(parsed, self.cases)

    def test_generator_can_write_an_identical_temporary_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "bugops_eval_v2.jsonl"
            generated = write_benchmark(output)
            self.assertEqual(generated, self.cases)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                DEFAULT_OUTPUT.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
