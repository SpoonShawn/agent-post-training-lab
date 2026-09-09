import math
import unittest

from evaluation.evaluator import (
    aggregate_results,
    evaluate_case,
    extract_tool_calls,
    extract_tools,
)


def parser_call(name, arguments=None, valid=True, **extra_call_fields):
    call = {"name": name}
    if arguments is not None:
        call["arguments"] = arguments
    call.update(extra_call_fields)
    return {"valid": valid, "tool_call": call}


def step(name=None, arguments=None, outcome=None, parsed=None):
    if parsed is None:
        parsed = [] if name is None else [parser_call(name, arguments)]
    tool_results = []
    if outcome is not None:
        call = {"name": name, "arguments": arguments or {}}
        tool_results.append({"tool_call": call, "result": outcome})
    return {
        "parsed_tool_calls": parsed,
        "tool_results": tool_results,
    }


def result_from_steps(steps, final_answer=None, final_state=None):
    result = {
        "trajectory": steps,
        "final_answer": final_answer,
    }
    if final_state is not None:
        result["final_environment_state"] = final_state
    return result


class EvaluatorV1CompatibilityTests(unittest.TestCase):
    def test_v1_exact_metrics_and_new_selection_metrics(self):
        case = {"expected_tools": ["search_knowledge"]}
        result = result_from_steps([
            step("get_build_info", {}, {"ok": True, "result": {}}),
            step("search_knowledge", {"query": "黑屏"}, {
                "ok": True,
                "result": [],
            }),
            step(),
        ], final_answer="这是已知问题")

        metrics = evaluate_case(case, result)

        self.assertEqual(
            metrics["predicted_tools"],
            ["get_build_info", "search_knowledge"],
        )
        self.assertEqual(metrics["num_steps"], 3)
        self.assertFalse(metrics["ordered_tool_match"])
        self.assertFalse(metrics["tool_set_match"])
        self.assertEqual(metrics["tool_true_positives"], 1)
        self.assertEqual(metrics["tool_false_positives"], 1)
        self.assertEqual(metrics["tool_false_negatives"], 0)
        self.assertAlmostEqual(metrics["tool_precision"], 0.5)
        self.assertAlmostEqual(metrics["tool_recall"], 1.0)
        self.assertAlmostEqual(metrics["tool_f1"], 2 / 3)
        self.assertTrue(metrics["ordered_subsequence_match"])
        self.assertIsNone(metrics["task_success"])

    def test_empty_v1_no_tool_case_has_safe_rates(self):
        metrics = evaluate_case(
            {"expected_tools": []},
            result_from_steps([step()], final_answer="明确步骤便于验证。"),
        )

        self.assertEqual(metrics["tool_precision"], 1.0)
        self.assertEqual(metrics["tool_recall"], 1.0)
        self.assertEqual(metrics["tool_f1"], 1.0)
        self.assertEqual(metrics["valid_call_rate"], 1.0)
        self.assertEqual(metrics["execution_success_rate"], 1.0)
        self.assertEqual(metrics["unknown_tool_rate"], 0.0)
        self.assertEqual(metrics["repeated_tool_call_rate"], 0.0)
        self.assertEqual(metrics["invalid_transition_rate"], 0.0)
        self.assertTrue(metrics["ordered_tool_match"])


class ToolSelectionAndArgumentTests(unittest.TestCase):
    def test_multiset_required_calls_optional_tools_and_subsequence(self):
        case = {
            "required_tools": ["navigate_ui", "navigate_ui", "verify_state"],
            "required_calls": [
                {"name": "navigate_ui", "arguments": {"target_page": "settings"}},
                {"name": "navigate_ui", "arguments": {"target_page": "graphics"}},
                {
                    "name": "verify_state",
                    "arguments": {"expected": {"current_page": "graphics"}},
                },
            ],
            "optional_tools": ["inspect_ui_state"],
            "expected_tools": ["navigate_ui", "navigate_ui", "verify_state"],
        }
        calls = [
            ("inspect_ui_state", {}),
            ("navigate_ui", {"target_page": "settings"}),
            ("navigate_ui", {"target_page": "graphics"}),
            ("verify_state", {
                "expected": {"current_page": "graphics", "black_screen": False},
            }),
            ("navigate_ui", {"target_page": "home"}),
        ]
        result = result_from_steps([
            step(name, arguments, {"ok": True, "result": {}})
            for name, arguments in calls
        ])

        metrics = evaluate_case(case, result)

        self.assertEqual(metrics["tool_true_positives"], 3)
        self.assertEqual(metrics["tool_false_positives"], 1)
        self.assertEqual(metrics["tool_false_negatives"], 0)
        self.assertAlmostEqual(metrics["tool_precision"], 0.75)
        self.assertEqual(metrics["tool_recall"], 1.0)
        self.assertEqual(metrics["argument_matches"], 3)
        self.assertEqual(metrics["argument_accuracy"], 1.0)
        self.assertTrue(metrics["ordered_subsequence_match"])
        self.assertFalse(metrics["ordered_tool_match"])

    def test_duplicate_requirement_is_not_collapsed(self):
        case = {
            "required_tools": ["navigate_ui"],
            "required_calls": [
                {"name": "navigate_ui", "arguments": {"target_page": "settings"}},
                {"name": "navigate_ui", "arguments": {"target_page": "graphics"}},
            ],
        }
        result = result_from_steps([
            step("navigate_ui", {"target_page": "settings"}, {
                "ok": True,
                "result": {},
            }),
        ])

        metrics = evaluate_case(case, result)

        self.assertEqual(metrics["tool_true_positives"], 1)
        self.assertEqual(metrics["tool_false_negatives"], 1)
        self.assertEqual(metrics["tool_recall"], 0.5)
        self.assertEqual(metrics["argument_accuracy"], 0.5)

    def test_each_optional_tool_entry_allows_only_one_extra_call(self):
        case = {
            "required_tools": [],
            "optional_tools": ["search_knowledge"],
        }
        result = result_from_steps([
            step("search_knowledge", {"query": str(index)}, {
                "ok": True,
                "result": [],
            })
            for index in range(3)
        ])

        metrics = evaluate_case(case, result)

        self.assertEqual(metrics["tool_true_positives"], 0)
        self.assertEqual(metrics["tool_false_positives"], 2)
        self.assertEqual(metrics["tool_precision"], 0.0)

    def test_recursive_subset_and_explicit_exact_argument_matching(self):
        actual = {
            "expected": {
                "current_page": "battle",
                "black_screen": True,
            },
            "note": "extra",
        }
        result = result_from_steps([
            step("verify_state", actual, {"ok": True, "result": {}}),
        ])
        subset_case = {
            "required_calls": [{
                "name": "verify_state",
                "arguments": {"expected": {"current_page": "battle"}},
            }],
        }
        exact_case = {
            "required_calls": [{
                "name": "verify_state",
                "arguments": {"expected": {"current_page": "battle"}},
                "argument_match": "exact",
            }],
        }

        self.assertEqual(evaluate_case(subset_case, result)["argument_accuracy"], 1.0)
        self.assertEqual(evaluate_case(exact_case, result)["argument_accuracy"], 0.0)

    def test_json_types_and_array_order_are_strict(self):
        result = result_from_steps([
            step("execute_action", {
                "action": "set_graphics",
                "value": True,
                "items": [1, 2],
            }, {"ok": True, "result": {}}),
        ])
        case = {
            "required_calls": [{
                "name": "execute_action",
                "arguments": {"value": 1, "items": [2, 1]},
            }],
        }

        self.assertEqual(evaluate_case(case, result)["argument_accuracy"], 0.0)


class FormatExecutionAndRepeatTests(unittest.TestCase):
    def test_unparsed_tool_marker_counts_as_invalid_intent(self):
        result = {
            "trajectory": [{
                "model_output": '<tool_call>{"name":"get_build_info"}',
                "parsed_tool_calls": [],
                "tool_results": [],
            }],
            "final_answer": None,
        }

        metrics = evaluate_case({"required_tools": ["get_build_info"]}, result)

        self.assertEqual(metrics["malformed_tool_call_markers"], 1)
        self.assertEqual(metrics["invalid_tool_calls"], 1)
        self.assertEqual(metrics["valid_call_rate"], 0.0)
        self.assertEqual(metrics["predicted_tools"], [])

    def test_ordinary_final_answer_is_not_an_invalid_call_intent(self):
        result = {
            "trajectory": [{
                "model_output": "无需调用工具，直接回答。",
                "parsed_tool_calls": [],
                "tool_results": [],
            }],
            "final_answer": "无需调用工具，直接回答。",
        }

        metrics = evaluate_case({"required_tools": []}, result)

        self.assertEqual(metrics["malformed_tool_call_markers"], 0)
        self.assertEqual(metrics["invalid_tool_calls"], 0)
        self.assertEqual(metrics["valid_call_rate"], 1.0)

    def test_invalid_parser_item_is_separate_from_bad_arguments(self):
        invalid = {"valid": False, "raw": "{bad json"}
        # This mirrors the baseline bad case: parser-valid JSON, but query is at
        # the wrong level and the executor therefore receives arguments={ }.
        misplaced = {
            "valid": True,
            "tool_call": {"name": "search_knowledge", "query": "黑屏"},
        }
        result = result_from_steps([
            step(parsed=[invalid]),
            step(
                "search_knowledge",
                {},
                {"ok": False, "error_type": "TypeError", "error": "missing query"},
                parsed=[misplaced],
            ),
        ])
        case = {
            "required_calls": [{
                "name": "search_knowledge",
                "arguments": {"query": "黑屏"},
            }],
        }

        metrics = evaluate_case(case, result)

        self.assertEqual(metrics["invalid_tool_calls"], 1)
        self.assertEqual(metrics["valid_call_rate"], 0.5)
        self.assertEqual(metrics["predicted_tools"], ["search_knowledge"])
        self.assertEqual(metrics["argument_accuracy"], 0.0)
        self.assertEqual(metrics["failed_tool_calls"], 1)

    def test_execution_unknown_and_invalid_transition_rates(self):
        result = result_from_steps([
            step("get_build_info", {}, {"ok": True, "result": {}}),
            step("navigate_ui", {"target_page": "graphics"}, {
                "ok": False,
                "error_type": "ValueError",
                "error": "无效页面跳转: home -> graphics",
            }),
            step("invented_tool", {}, {
                "ok": False,
                "error_type": "unknown_tool",
                "error": "missing",
            }),
        ])

        metrics = evaluate_case({"required_tools": []}, result)

        self.assertEqual(metrics["failed_tool_calls"], 2)
        self.assertEqual(metrics["unknown_tool_calls"], 1)
        self.assertAlmostEqual(metrics["execution_success_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["unknown_tool_rate"], 1 / 3)
        self.assertEqual(metrics["invalid_transition_count"], 1)
        self.assertAlmostEqual(metrics["invalid_transition_rate"], 1 / 3)

    def test_exact_repeat_after_success_is_redundant(self):
        args_one = {"expected": {"current_page": "home", "black_screen": False}}
        args_reordered = {"expected": {"black_screen": False, "current_page": "home"}}
        result = result_from_steps([
            step("verify_state", args_one, {"ok": True, "result": {}}),
            step("inspect_ui_state", {}, {"ok": True, "result": {}}),
            step("verify_state", args_reordered, {"ok": True, "result": {}}),
            step("verify_state", {"expected": {"current_page": "battle"}}, {
                "ok": True,
                "result": {},
            }),
        ])

        metrics = evaluate_case({"required_tools": []}, result)

        self.assertEqual(metrics["repeated_tool_calls"], 1)
        self.assertAlmostEqual(metrics["repeated_tool_call_rate"], 0.25)
        self.assertEqual(metrics["retry_calls"], 0)

    def test_static_tool_repeat_stays_redundant_after_state_change(self):
        result = result_from_steps([
            step("get_build_info", {}, {"ok": True, "result": {}}),
            step("navigate_ui", {"target_page": "settings"}, {
                "ok": True,
                "result": {"current_page": "settings"},
            }),
            step("get_build_info", {}, {"ok": True, "result": {}}),
        ])

        metrics = evaluate_case({"required_tools": []}, result)

        self.assertEqual(metrics["repeated_tool_calls"], 1)

    def test_first_failure_retry_is_not_redundant(self):
        arguments = {"target_page": "settings"}
        result = result_from_steps([
            step("navigate_ui", arguments, {
                "ok": False,
                "error_type": "RuntimeError",
                "error": "注入的临时故障: navigate:settings:timeout",
            }),
            step("navigate_ui", arguments, {"ok": True, "result": {}}),
        ])

        metrics = evaluate_case({"required_tools": ["navigate_ui"]}, result)

        self.assertEqual(metrics["retry_calls"], 1)
        self.assertEqual(metrics["repeated_tool_calls"], 0)

    def test_third_failed_attempt_is_redundant(self):
        arguments = {"target_page": "settings"}
        failure = {"ok": False, "error_type": "RuntimeError", "error": "timeout"}
        result = result_from_steps([
            step("navigate_ui", arguments, failure),
            step("navigate_ui", arguments, failure),
            step("navigate_ui", arguments, failure),
        ])

        metrics = evaluate_case({"required_tools": ["navigate_ui"]}, result)

        self.assertEqual(metrics["retry_calls"], 2)
        self.assertEqual(metrics["repeated_tool_calls"], 1)

    def test_repeated_observation_after_intervening_failure_is_not_redundant(self):
        result = result_from_steps([
            step("inspect_ui_state", {}, {"ok": True, "result": {}}),
            step("navigate_ui", {"target_page": "settings"}, {
                "ok": False,
                "error_type": "RuntimeError",
                "error": "timeout",
            }),
            step("inspect_ui_state", {}, {"ok": True, "result": {}}),
        ])

        metrics = evaluate_case({"required_tools": []}, result)

        self.assertEqual(metrics["repeated_tool_calls"], 0)
        self.assertEqual(metrics["retry_calls"], 0)

    def test_same_action_after_intervening_state_changes_is_not_redundant(self):
        result = result_from_steps([
            step("execute_action", {"action": "back_home"}, {
                "ok": True,
                "result": {"current_page": "home", "graphics_preset": "high"},
            }),
            step("navigate_ui", {"target_page": "settings"}, {
                "ok": True,
                "result": {"current_page": "settings", "graphics_preset": "high"},
            }),
            step("navigate_ui", {"target_page": "graphics"}, {
                "ok": True,
                "result": {"current_page": "graphics", "graphics_preset": "high"},
            }),
            step("execute_action", {"action": "back_home"}, {
                "ok": True,
                "result": {"current_page": "home", "graphics_preset": "high"},
            }),
        ])

        metrics = evaluate_case({"required_tools": []}, result)

        self.assertEqual(metrics["repeated_tool_calls"], 0)

    def test_fault_adjustment_only_matches_configured_target(self):
        case = {
            "required_tools": ["navigate_ui"],
            "environment": {
                "fault_injections": [{
                    "operation": "navigate",
                    "target": "settings",
                    "times": 1,
                    "error": "timeout",
                }],
            },
        }
        result = result_from_steps([
            step("invented_tool", {}, {
                "ok": False,
                "error_type": "unknown_tool",
                "error": "missing",
            }),
            step("navigate_ui", {"target_page": "settings"}, {
                "ok": False,
                "error_type": "RuntimeError",
                "error": "注入的临时故障: navigate:settings:timeout",
            }),
            step("navigate_ui", {"target_page": "settings"}, {
                "ok": True,
                "result": {},
            }),
        ])

        metrics = evaluate_case(case, result)

        self.assertEqual(metrics["expected_failed_calls"], 1)
        self.assertEqual(metrics["observed_expected_failed_calls"], 1)
        self.assertEqual(metrics["unexpected_failed_calls"], 1)
        self.assertAlmostEqual(metrics["execution_success_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["adjusted_execution_success_rate"], 2 / 3)


class TaskAndRecoverySuccessTests(unittest.TestCase):
    def test_conjunctive_task_success_from_state_tool_results_and_answer(self):
        final_state = {
            "version": "2.3.1",
            "platform": "android",
            "current_page": "battle",
            "graphics_preset": "high",
            "battle_hud_visible": False,
            "black_screen": True,
        }
        result = result_from_steps([
            step("query_logs", {}, {
                "ok": True,
                "result": ["navigate:battle", "ERROR: graphics context restore failure"],
            }),
            step("verify_state", {"expected": {"black_screen": True}}, {
                "ok": True,
                "result": {
                    "success": True,
                    "matches": {"black_screen": True},
                    "current_state": final_state,
                },
            }),
        ], final_answer="结论：已成功复现黑屏。", final_state=final_state)
        case = {
            "required_tools": ["query_logs", "verify_state"],
            "success_criteria": {
                "final_state": {"current_page": "battle", "black_screen": True},
                "required_tool_results": [
                    {
                        "tool": "query_logs",
                        "path": "$",
                        "contains": "graphics context restore failure",
                    },
                    {"tool": "verify_state", "path": "success", "equals": True},
                ],
                "final_answer": {
                    "required": True,
                    "contains_all": ["结论", "黑屏"],
                    "contains_any": ["复现", "触发"],
                    "regex_any": [r"(成功|已经).*复现"],
                },
            },
        }

        metrics = evaluate_case(case, result)

        self.assertTrue(metrics["task_success"])
        self.assertTrue(metrics["final_state_match"])
        self.assertEqual(metrics["required_tool_result_matches"], 2)
        self.assertTrue(metrics["final_answer_match"])
        self.assertEqual(metrics["success_criteria_passed"], 4)

    def test_explicit_final_environment_state_beats_stale_trajectory_state(self):
        result = result_from_steps([
            step("inspect_ui_state", {}, {
                "ok": True,
                "result": {"current_page": "home"},
            }),
        ], final_state={"current_page": "settings"})
        case = {
            "success_criteria": {"final_state": {"current_page": "settings"}},
        }

        self.assertTrue(evaluate_case(case, result)["task_success"])

    def test_latest_state_bearing_result_is_fallback_for_old_records(self):
        result = result_from_steps([
            step("inspect_ui_state", {}, {
                "ok": True,
                "result": {"current_page": "home", "black_screen": False},
            }),
            step("navigate_ui", {"target_page": "settings"}, {
                "ok": True,
                "result": {"current_page": "settings", "black_screen": False},
            }),
            step("query_logs", {}, {"ok": True, "result": ["navigate:settings"]}),
        ])
        case = {
            "success_criteria": {"final_state": {"current_page": "settings"}},
        }

        self.assertTrue(evaluate_case(case, result)["task_success"])

    def test_full_state_fallback_beats_newer_partial_build_payload(self):
        result = result_from_steps([
            step("navigate_ui", {"target_page": "settings"}, {
                "ok": True,
                "result": {
                    "version": "2.3.1",
                    "platform": "android",
                    "current_page": "settings",
                    "graphics_preset": "high",
                    "battle_hud_visible": False,
                    "black_screen": False,
                },
            }),
            step("get_build_info", {}, {
                "ok": True,
                "result": {"version": "2.3.1", "platform": "android"},
            }),
        ])
        case = {
            "success_criteria": {
                "final_state": {"current_page": "settings"},
            },
        }

        self.assertTrue(evaluate_case(case, result)["task_success"])

    def test_outer_execution_success_does_not_imply_verify_criterion_success(self):
        result = result_from_steps([
            step("verify_state", {"expected": {"black_screen": True}}, {
                "ok": True,
                "result": {"success": False},
            }),
        ])
        case = {
            "success_criteria": {
                "required_tool_results": [{
                    "tool": "verify_state",
                    "path": "success",
                    "equals": True,
                }],
            },
        }

        metrics = evaluate_case(case, result)

        self.assertEqual(metrics["execution_success_rate"], 1.0)
        self.assertFalse(metrics["task_success"])

    def test_forbidden_final_answer_phrases_block_contradictory_success(self):
        examples = [
            ("复现", "没有复现黑屏", "没有复现"),
            ("正常", "结果不正常", "不正常"),
            ("规避", "规避失败", "规避失败"),
        ]
        for positive, contradictory, forbidden in examples:
            with self.subTest(answer=contradictory):
                case = {
                    "success_criteria": {
                        "final_answer": {
                            "required": True,
                            "contains_any": [positive],
                            "forbidden_any": [forbidden],
                        },
                    },
                }
                result = result_from_steps(
                    [step()], final_answer=contradictory
                )

                self.assertFalse(evaluate_case(case, result)["task_success"])

    def test_regex_none_blocks_a_negative_conclusion(self):
        case = {
            "success_criteria": {
                "final_answer": {
                    "required": True,
                    "contains_any": ["复现"],
                    "regex_none": [r"(没有|未能|无法).*复现"],
                },
            },
        }
        result = result_from_steps([step()], final_answer="本次未能成功复现黑屏。")

        self.assertFalse(evaluate_case(case, result)["task_success"])

    def test_recovery_requires_planned_failure_later_success_and_task_success(self):
        case = {
            "requires_recovery": True,
            "environment": {
                "fault_injections": [{
                    "operation": "navigate",
                    "target": "settings",
                    "times": 1,
                }],
            },
            "success_criteria": {
                "final_state": {"current_page": "settings"},
            },
        }
        arguments = {"target_page": "settings"}
        result = result_from_steps([
            step("navigate_ui", arguments, {
                "ok": False,
                "error_type": "RuntimeError",
                "error": "注入的临时故障: navigate:settings:transient_failure",
            }),
            step("navigate_ui", arguments, {
                "ok": True,
                "result": {"current_page": "settings"},
            }),
        ], final_state={"current_page": "settings"})

        metrics = evaluate_case(case, result)

        self.assertTrue(metrics["recovery_attempted"])
        self.assertTrue(metrics["recovery_success"])
        self.assertEqual(metrics["retry_calls"], 1)
        self.assertEqual(metrics["repeated_tool_calls"], 0)
        self.assertEqual(metrics["adjusted_execution_success_rate"], 1.0)

    def test_unrelated_failure_does_not_count_as_planned_recovery(self):
        case = {
            "requires_recovery": True,
            "environment": {
                "fault_injections": [{
                    "operation": "navigate",
                    "target": "settings",
                    "times": 1,
                }],
            },
            "success_criteria": {
                "final_state": {"current_page": "home"},
            },
        }
        result = result_from_steps([
            step("invented", {}, {
                "ok": False,
                "error_type": "unknown_tool",
                "error": "missing",
            }),
            step("inspect_ui_state", {}, {
                "ok": True,
                "result": {"current_page": "home"},
            }),
        ], final_state={"current_page": "home"})

        metrics = evaluate_case(case, result)

        self.assertTrue(metrics["task_success"])
        self.assertFalse(metrics["recovery_attempted"])
        self.assertFalse(metrics["recovery_success"])
        self.assertEqual(metrics["observed_expected_failed_calls"], 0)
        self.assertEqual(metrics["unexpected_failed_calls"], 1)

    def test_same_turn_blind_duplicate_is_not_adaptive_recovery(self):
        arguments = {"target_page": "settings"}
        call = {"name": "navigate_ui", "arguments": arguments}
        result = {
            "trajectory": [
                {
                    "parsed_tool_calls": [
                        {"valid": True, "tool_call": call},
                        {"valid": True, "tool_call": call},
                    ],
                    "tool_results": [
                        {
                            "tool_call": call,
                            "result": {
                                "ok": False,
                                "error_type": "RuntimeError",
                                "error": "注入的临时故障: navigate:settings:timeout",
                            },
                        },
                        {
                            "tool_call": call,
                            "result": {
                                "ok": True,
                                "result": {"current_page": "settings"},
                            },
                        },
                    ],
                },
                step("inspect_ui_state", {}, {
                    "ok": True,
                    "result": {"current_page": "settings"},
                }),
            ],
            "final_answer": "已进入设置页",
            "final_environment_state": {"current_page": "settings"},
        }
        case = {
            "requires_recovery": True,
            "environment": {
                "fault_injections": [{
                    "operation": "navigate",
                    "target": "settings",
                    "times": 1,
                }],
            },
            "success_criteria": {
                "final_state": {"current_page": "settings"},
            },
        }

        metrics = evaluate_case(case, result)

        self.assertTrue(metrics["task_success"])
        self.assertFalse(metrics["recovery_success"])
        self.assertEqual(metrics["retry_calls"], 0)
        self.assertEqual(metrics["repeated_tool_calls"], 1)

    def test_max_tool_calls_can_enforce_a_no_tool_task(self):
        case = {
            "required_tools": [],
            "success_criteria": {
                "max_tool_calls": 0,
                "final_answer": {"required": True},
            },
        }
        result = result_from_steps([
            step("get_build_info", {}, {"ok": True, "result": {}}),
            step(),
        ], final_answer="软件缺陷需要明确复现步骤。")

        metrics = evaluate_case(case, result)

        self.assertFalse(metrics["max_tool_calls_match"])
        self.assertFalse(metrics["task_success"])

    def test_malformed_tool_intent_fails_a_no_tool_task(self):
        case = {
            "required_tools": [],
            "success_criteria": {
                "final_answer": {
                    "required": True,
                    "contains_any": ["复现"],
                },
                "max_tool_calls": 0,
            },
        }
        result = {
            "trajectory": [{
                "model_output": "<tool_call>{bad json 复现步骤有助于定位",
                "parsed_tool_calls": [],
                "tool_results": [],
            }],
            "final_answer": "<tool_call>{bad json 复现步骤有助于定位",
        }

        metrics = evaluate_case(case, result)

        self.assertEqual(metrics["invalid_tool_calls"], 1)
        self.assertEqual(metrics["tool_attempt_count"], 1)
        self.assertEqual(metrics["tool_call_count"], 0)
        self.assertFalse(metrics["max_tool_calls_match"])
        self.assertFalse(metrics["task_success"])

    def test_final_answer_text_matching_is_case_insensitive(self):
        case = {
            "success_criteria": {
                "final_answer": {
                    "required": True,
                    "contains_any": ["iOS"],
                    "regex_any": [r"platform:\s*ios"],
                    "forbidden_any": ["FAILED"],
                    "regex_none": [r"not\s+ios"],
                },
            },
        }
        result = result_from_steps(
            [step()], final_answer="PLATFORM: IOS，运行正常。"
        )

        self.assertTrue(evaluate_case(case, result)["task_success"])


class ExtractionAndAggregationTests(unittest.TestCase):
    def test_flattened_result_format_is_supported(self):
        result = {
            "tool_calls": [
                {"name": "get_build_info", "arguments": "{}"},
                {"name": "inspect_ui_state", "arguments": {}},
            ],
            "tool_results": [
                {
                    "tool_call": {"name": "get_build_info", "arguments": {}},
                    "result": {"ok": True, "result": {"version": "2.3.1"}},
                },
                {
                    "tool_call": {"name": "inspect_ui_state", "arguments": {}},
                    "result": {"ok": True, "result": {"current_page": "home"}},
                },
            ],
            "final_answer": "done",
        }

        self.assertEqual(
            extract_tools(result),
            ["get_build_info", "inspect_ui_state"],
        )
        self.assertEqual(extract_tool_calls(result)[0]["arguments"], {})
        self.assertEqual(
            evaluate_case({"required_tools": ["get_build_info"]}, result)[
                "execution_success_rate"
            ],
            1.0,
        )

    def test_malformed_structures_do_not_crash(self):
        result = {
            "trajectory": [{
                "parsed_tool_calls": [None, [], {"valid": True, "tool_call": []}],
                "tool_results": [None],
            }],
        }

        metrics = evaluate_case({}, result)

        self.assertEqual(metrics["predicted_tools"], [])
        self.assertEqual(metrics["tool_execution_count"], 1)
        self.assertFalse(math.isnan(metrics["execution_success_rate"]))

    def test_aggregate_is_macro_and_grouped_with_none_excluded(self):
        passing_case = {
            "category": "single_tool",
            "difficulty": "easy",
            "scenario_family": "build_info",
            "required_calls": [{"name": "get_build_info", "arguments": {}}],
            "success_criteria": {
                "required_tool_results": [{
                    "tool": "get_build_info",
                    "path": "version",
                    "equals": "2.3.1",
                }],
            },
        }
        passing_result = result_from_steps([
            step("get_build_info", {}, {
                "ok": True,
                "result": {"version": "2.3.1", "platform": "android"},
            }),
        ])
        failing_case = {
            "category": "long_horizon",
            "difficulty": "hard",
            "scenario_family": "repro",
            "tags": ["long_horizon"],
            "required_tools": ["navigate_ui"],
            "success_criteria": {
                "final_state": {"current_page": "battle"},
            },
        }
        failing_result = result_from_steps(
            [step()], final_state={"current_page": "home"}
        )
        legacy_case = {
            "category": "single_tool",
            "difficulty": "easy",
            "scenario_family": "legacy",
        }
        records = [
            {"case": passing_case, "result": passing_result},
            {"case": failing_case, "result": failing_result},
            {
                "case": legacy_case,
                "metrics": evaluate_case(legacy_case, result_from_steps([step()])),
            },
        ]

        aggregate = aggregate_results(records)

        self.assertEqual(aggregate["overall"]["num_cases"], 3)
        self.assertEqual(aggregate["overall"]["task_success_rate"], 0.5)
        self.assertEqual(
            aggregate["overall"]["task_success_rate_evaluated_cases"], 2
        )
        self.assertEqual(aggregate["overall"]["argument_accuracy"], 1.0)
        self.assertEqual(
            aggregate["overall"]["argument_accuracy_evaluated_cases"], 1
        )
        self.assertIn("single_tool", aggregate["by_category"])
        self.assertIn("hard", aggregate["by_difficulty"])
        self.assertIn("repro", aggregate["by_scenario_family"])
        self.assertEqual(
            aggregate["overall"]["long_horizon_success_rate"], 0.0
        )
        self.assertEqual(aggregate["category_macro"]["num_categories"], 2)


if __name__ == "__main__":
    unittest.main()
