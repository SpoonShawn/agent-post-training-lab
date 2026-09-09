from agent.environment import SyntheticAppEnvironment

ENV = SyntheticAppEnvironment()


def reset_environment():
    ENV.reset()
    return ENV.state()


def get_build_info():
    return {
        "version": ENV.version,
        "platform": ENV.platform
    }


def inspect_ui_state():
    return ENV.state()


def navigate_ui(target_page: str):
    return ENV.navigate(target_page)


def execute_action(action: str, value=None):
    return ENV.action(action, value)


def query_logs():
    return list(ENV.logs)


def verify_state(expected: dict):
    current = ENV.state()

    matches = {
        key: current.get(key) == value
        for key, value in expected.items()
    }

    return {
        "success": all(matches.values()),
        "matches": matches,
        "current_state": current,
    }
