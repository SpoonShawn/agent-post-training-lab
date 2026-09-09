from tools.environment_tools import (
    reset_environment,
    inspect_ui_state,
    navigate_ui,
    execute_action,
    query_logs,
)

print("RESET")
print(reset_environment())

print("\nNAVIGATE TO DUNGEON")
print(navigate_ui("dungeon_select"))
print(navigate_ui("multiplayer_dungeon"))

print("\nSTART DUNGEON")
print(execute_action("start_multiplayer_dungeon"))

print("\nLOGS")
print(query_logs())

print("\nFINAL STATE")
print(inspect_ui_state())
