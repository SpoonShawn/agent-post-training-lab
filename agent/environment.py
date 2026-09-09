from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SyntheticAppEnvironment:
    version: str = "2.3.1"
    platform: str = "android"
    current_page: str = "home"
    graphics_preset: str = "high"
    battle_hud_visible: bool = False
    black_screen: bool = False
    logs: List[str] = field(default_factory=list)

    def reset(self):
        self.current_page = "home"
        self.graphics_preset = "high"
        self.battle_hud_visible = False
        self.black_screen = False
        self.logs = []

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

        self.current_page = transitions[key]
        self.logs.append(f"navigate:{target}")
        return self.state()

    def action(self, action: str, value=None):
        if action == "set_graphics":
            if self.current_page != "graphics":
                raise ValueError("只能在 graphics 页面修改画质设置。")

            self.graphics_preset = value
            self.logs.append(f"set_graphics:{value}")

        elif action == "back_home":
            self.current_page = "home"
            self.logs.append("back_home")

        elif action == "start_multiplayer_dungeon":
            if self.current_page != "multiplayer_dungeon":
                raise ValueError("必须先进入 multiplayer_dungeon 页面。")

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
