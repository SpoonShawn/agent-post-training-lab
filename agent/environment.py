from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


DEFAULT_STATE = {
    "version": "2.3.1",
    "platform": "android",
    "current_page": "home",
    "graphics_preset": "high",
    "battle_hud_visible": False,
    "black_screen": False,
}

VALID_PAGES = {
    "home",
    "settings",
    "graphics",
    "dungeon_select",
    "multiplayer_dungeon",
    "battle",
}

VALID_PLATFORMS = {"android", "ios"}
VALID_GRAPHICS_PRESETS = {"low", "standard", "high"}
VALID_FAULT_OPERATIONS = {"navigate", "action"}


@dataclass
class SyntheticAppEnvironment:
    version: str = "2.3.1"
    platform: str = "android"
    current_page: str = "home"
    graphics_preset: str = "high"
    battle_hud_visible: bool = False
    black_screen: bool = False
    logs: List[str] = field(default_factory=list)
    _fault_injections: List[Dict[str, Any]] = field(
        default_factory=list,
        repr=False,
    )

    def reset(self, config: Optional[Mapping[str, Any]] = None):
        """Reset the synthetic app and optionally apply a benchmark scenario.

        ``config`` accepts state keys directly or under ``initial_state``.  A
        small fault plan can be supplied through ``fault_injections`` to make a
        normally valid operation fail a fixed number of times.  This gives the
        recovery benchmark a deterministic failure without changing the public
        tool API seen by the model.
        """

        config = dict(config or {})
        initial_state = config.pop("initial_state", {})

        if not isinstance(initial_state, Mapping):
            raise ValueError("environment.initial_state 必须是 object。")

        state_overrides = dict(initial_state)
        for key in DEFAULT_STATE:
            if key in config:
                state_overrides[key] = config.pop(key)

        fault_injections = config.pop("fault_injections", [])
        if config:
            unknown = ", ".join(sorted(config))
            raise ValueError(f"未知环境配置字段: {unknown}")

        unknown_state = set(state_overrides) - set(DEFAULT_STATE)
        if unknown_state:
            unknown = ", ".join(sorted(unknown_state))
            raise ValueError(f"未知初始状态字段: {unknown}")

        state = {**DEFAULT_STATE, **state_overrides}
        if not isinstance(state["version"], str) or not state["version"].strip():
            raise ValueError("version 必须是非空字符串。")
        if (
            not isinstance(state["platform"], str)
            or state["platform"].lower() not in VALID_PLATFORMS
        ):
            raise ValueError(
                f"未知平台: {state['platform']}"
            )
        if state["current_page"] not in VALID_PAGES:
            raise ValueError(
                f"未知初始页面: {state['current_page']}"
            )
        if state["graphics_preset"] not in VALID_GRAPHICS_PRESETS:
            raise ValueError(
                f"未知画质预设: {state['graphics_preset']}"
            )
        for boolean_field in ("battle_hud_visible", "black_screen"):
            if not isinstance(state[boolean_field], bool):
                raise ValueError(f"{boolean_field} 必须是 boolean。")
        if state["battle_hud_visible"] and state["black_screen"]:
            raise ValueError("HUD 可见与黑屏状态不能同时为 true。")

        self.version = str(state["version"])
        self.platform = str(state["platform"]).lower()
        self.current_page = state["current_page"]
        self.graphics_preset = state["graphics_preset"]
        self.battle_hud_visible = state["battle_hud_visible"]
        self.black_screen = state["black_screen"]
        self.logs = []
        self._fault_injections = self._validate_faults(fault_injections)

        return self.state()

    @staticmethod
    def _validate_faults(faults: Any) -> List[Dict[str, Any]]:
        if not isinstance(faults, list):
            raise ValueError("environment.fault_injections 必须是 array。")

        validated = []
        for index, raw_fault in enumerate(faults):
            if not isinstance(raw_fault, Mapping):
                raise ValueError(
                    f"fault_injections[{index}] 必须是 object。"
                )

            fault = dict(raw_fault)
            operation = fault.get("operation")
            target = fault.get("target")
            times = fault.get("times", 1)
            error = fault.get("error", "transient_failure")

            if operation not in VALID_FAULT_OPERATIONS:
                raise ValueError(
                    f"未知故障注入操作: {operation}"
                )
            if not isinstance(target, str) or not target:
                raise ValueError("故障注入 target 必须是非空字符串。")
            if not isinstance(times, int) or isinstance(times, bool) or times < 1:
                raise ValueError("故障注入 times 必须是正整数。")

            validated.append({
                "operation": operation,
                "target": target,
                "times": times,
                "error": str(error),
            })

        return validated

    def _maybe_fail(self, operation: str, target: str):
        for fault in self._fault_injections:
            if (
                fault["operation"] == operation
                and fault["target"] == target
                and fault["times"] > 0
            ):
                fault["times"] -= 1
                self.logs.append(
                    f"INJECTED_FAILURE:{operation}:{target}:"
                    f"{fault['error']}"
                )
                raise RuntimeError(
                    "注入的临时故障: "
                    f"{operation}:{target}:{fault['error']}"
                )

    def state(self) -> Dict:
        return {
            "version": self.version,
            "platform": self.platform,
            "current_page": self.current_page,
            "graphics_preset": self.graphics_preset,
            "battle_hud_visible": self.battle_hud_visible,
            "black_screen": self.black_screen,
        }

    def navigate(self, target: str):
        transitions = {
            ("home", "settings"): "settings",
            ("settings", "graphics"): "graphics",
            ("home", "dungeon_select"): "dungeon_select",
            ("dungeon_select", "multiplayer_dungeon"): "multiplayer_dungeon",
        }

        key = (self.current_page, target)

        if key not in transitions:
            raise ValueError(f"无效页面跳转: {self.current_page} -> {target}")

        # A transient injection represents failure of an otherwise valid
        # operation.  Validate the state transition first so an invalid model
        # action cannot consume the planned recovery fault.
        self._maybe_fail("navigate", target)

        self.current_page = transitions[key]
        self.logs.append(f"navigate:{target}")
        return self.state()

    def action(self, action: str, value=None):
        if action == "set_graphics":
            if self.current_page != "graphics":
                raise ValueError("只能在 graphics 页面修改画质设置。")
            if value not in VALID_GRAPHICS_PRESETS:
                raise ValueError(
                    "画质预设必须是 low、standard 或 high。"
                )

            self._maybe_fail("action", action)
            self.graphics_preset = value
            self.logs.append(f"set_graphics:{value}")

        elif action == "back_home":
            self._maybe_fail("action", action)
            self.current_page = "home"
            self.battle_hud_visible = False
            self.black_screen = False
            self.logs.append("back_home")

        elif action == "start_multiplayer_dungeon":
            if self.current_page != "multiplayer_dungeon":
                raise ValueError("必须先进入 multiplayer_dungeon 页面。")

            self._maybe_fail("action", action)
            self.current_page = "battle"

            if (
                self.version == "2.3.1"
                and self.platform == "android"
                and self.graphics_preset == "high"
            ):
                self.black_screen = True
                self.battle_hud_visible = False
                self.logs.append("ERROR: graphics context restore failure")
            else:
                self.black_screen = False
                self.battle_hud_visible = True
                self.logs.append("battle scene loaded successfully")

        else:
            raise ValueError(f"未知动作: {action}")

        return self.state()
