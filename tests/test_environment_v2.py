import unittest

from agent.environment import SyntheticAppEnvironment
from tools.environment_tools import ENV, get_build_info, reset_environment


class SyntheticAppEnvironmentV2Tests(unittest.TestCase):
    def tearDown(self):
        reset_environment()

    def test_no_arg_reset_restores_all_defaults(self):
        reset_environment({
            "initial_state": {
                "version": "9.9.9",
                "platform": "ios",
                "current_page": "graphics",
                "graphics_preset": "standard",
                "battle_hud_visible": False,
                "black_screen": True,
            },
        })

        state = reset_environment()

        self.assertEqual(state["version"], "2.3.1")
        self.assertEqual(state["platform"], "android")
        self.assertEqual(state["current_page"], "home")
        self.assertEqual(state["graphics_preset"], "high")
        self.assertFalse(state["battle_hud_visible"])
        self.assertFalse(state["black_screen"])

    def test_nested_initial_state_is_visible_to_existing_tools(self):
        reset_environment({
            "initial_state": {
                "version": "2.3.1",
                "platform": "ios",
                "current_page": "settings",
            },
        })

        self.assertEqual(
            get_build_info(),
            {"version": "2.3.1", "platform": "ios"},
        )
        self.assertEqual(ENV.state()["current_page"], "settings")

    def test_navigation_fault_is_consumed_without_state_mutation(self):
        environment = SyntheticAppEnvironment()
        environment.reset({
            "fault_injections": [{
                "operation": "navigate",
                "target": "settings",
                "times": 1,
                "error": "navigation_timeout",
            }],
        })

        with self.assertRaisesRegex(RuntimeError, "navigation_timeout"):
            environment.navigate("settings")

        self.assertEqual(environment.current_page, "home")
        state = environment.navigate("settings")
        self.assertEqual(state["current_page"], "settings")

    def test_action_fault_is_consumed_without_starting_battle(self):
        environment = SyntheticAppEnvironment()
        environment.reset({
            "initial_state": {"current_page": "multiplayer_dungeon"},
            "fault_injections": [{
                "operation": "action",
                "target": "start_multiplayer_dungeon",
                "times": 1,
                "error": "service_busy",
            }],
        })

        with self.assertRaisesRegex(RuntimeError, "service_busy"):
            environment.action("start_multiplayer_dungeon")

        self.assertEqual(environment.current_page, "multiplayer_dungeon")
        state = environment.action("start_multiplayer_dungeon")
        self.assertEqual(state["current_page"], "battle")
        self.assertTrue(state["black_screen"])

    def test_invalid_operation_does_not_consume_planned_fault(self):
        environment = SyntheticAppEnvironment()
        environment.reset({
            "initial_state": {"current_page": "graphics"},
            "fault_injections": [{
                "operation": "navigate",
                "target": "settings",
                "times": 1,
                "error": "navigation_timeout",
            }],
        })

        with self.assertRaisesRegex(ValueError, "无效页面跳转"):
            environment.navigate("settings")

        environment.action("back_home")
        with self.assertRaisesRegex(RuntimeError, "navigation_timeout"):
            environment.navigate("settings")
        self.assertEqual(environment.current_page, "home")

    def test_back_home_clears_battle_only_display_flags(self):
        environment = SyntheticAppEnvironment()
        environment.reset({
            "initial_state": {
                "current_page": "battle",
                "black_screen": True,
                "battle_hud_visible": False,
            },
        })

        state = environment.action("back_home")

        self.assertEqual(state["current_page"], "home")
        self.assertFalse(state["black_screen"])
        self.assertFalse(state["battle_hud_visible"])

        environment.reset({
            "initial_state": {
                "current_page": "battle",
                "black_screen": False,
                "battle_hud_visible": True,
            },
        })
        state = environment.action("back_home")
        self.assertFalse(state["black_screen"])
        self.assertFalse(state["battle_hud_visible"])

    def test_existing_black_screen_scenario_is_unchanged(self):
        environment = SyntheticAppEnvironment()
        environment.reset()
        environment.navigate("dungeon_select")
        environment.navigate("multiplayer_dungeon")

        state = environment.action("start_multiplayer_dungeon")

        self.assertTrue(state["black_screen"])
        self.assertFalse(state["battle_hud_visible"])

    def test_unknown_environment_field_is_rejected(self):
        environment = SyntheticAppEnvironment()

        with self.assertRaisesRegex(ValueError, "未知环境配置字段"):
            environment.reset({"surprise": True})

    def test_invalid_typed_state_is_not_truth_coerced(self):
        environment = SyntheticAppEnvironment()

        with self.assertRaisesRegex(ValueError, "必须是 boolean"):
            environment.reset({
                "initial_state": {"black_screen": "false"},
            })

    def test_invalid_platform_and_preset_are_rejected(self):
        environment = SyntheticAppEnvironment()

        with self.assertRaisesRegex(ValueError, "未知平台"):
            environment.reset({"initial_state": {"platform": "windows"}})
        with self.assertRaisesRegex(ValueError, "未知画质预设"):
            environment.reset({
                "initial_state": {"graphics_preset": "cinematic"},
            })

    def test_set_graphics_rejects_missing_or_unknown_value(self):
        environment = SyntheticAppEnvironment()
        environment.reset({"initial_state": {"current_page": "graphics"}})

        with self.assertRaisesRegex(ValueError, "画质预设必须"):
            environment.action("set_graphics")
        with self.assertRaisesRegex(ValueError, "画质预设必须"):
            environment.action("set_graphics", "cinematic")


if __name__ == "__main__":
    unittest.main()
