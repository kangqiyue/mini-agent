import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("start.sh")


@pytest.fixture
def startup_environment(tmp_path: Path) -> dict[str, str]:
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    compose_dir = tmp_path / "gateway"
    compose_dir.mkdir()
    commands = {
        "colima": """#!/bin/bash
printf 'colima %s\\n' "$*" >> "$STARTUP_TEST_CALL_LOG"
case "$1" in
    status) exit "${STARTUP_TEST_COLIMA_STATUS:-0}" ;;
    start) exit "${STARTUP_TEST_COLIMA_START_EXIT:-0}" ;;
esac
""",
        "docker": """#!/bin/bash
printf 'docker %s\\n' "$*" >> "$STARTUP_TEST_CALL_LOG"
[[ "$1" == --context && "$2" == colima ]] || exit 90
if [[ "$3" == compose && "$4" == ps ]]; then
    printf '%s\\n' "${STARTUP_TEST_SERVICES:-db}" litellm
elif [[ "$3" == compose && "$4" == start ]]; then
    exit "${STARTUP_TEST_COMPOSE_START_EXIT:-0}"
fi
""",
        "curl": """#!/bin/bash
printf 'curl %s\\n' "$*" >> "$STARTUP_TEST_CALL_LOG"
exit "${STARTUP_TEST_HEALTH_EXIT:-0}"
""",
    }
    for name, source in commands.items():
        path = binary_dir / name
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)
    return {
        "PATH": f"{binary_dir}:/usr/bin:/bin",
        "LITELLM_COMPOSE_DIR": str(compose_dir),
        "LITELLM_STARTUP_TIMEOUT_SECONDS": "1",
        "STARTUP_TEST_CALL_LOG": str(tmp_path / "calls.log"),
    }


def run_startup(environment: dict[str, str], *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(SCRIPT), *arguments],
        env=environment,
        capture_output=True,
        text=True,
        timeout=8,
    )


def test_stopped_vm_starts_before_existing_containers_and_health_check(
    startup_environment: dict[str, str],
) -> None:
    startup_environment["STARTUP_TEST_COLIMA_STATUS"] = "1"
    result = run_startup(startup_environment)
    calls = Path(startup_environment["STARTUP_TEST_CALL_LOG"]).read_text()

    assert result.returncode == 0, result.stderr
    assert (
        calls.index("colima start") < calls.index("compose start db litellm") < calls.index("curl ")
    )
    assert "--noproxy *" in calls
    assert "Gateway ready" in result.stdout


def test_repeated_runs_do_not_restart_the_vm_or_recreate_containers(
    startup_environment: dict[str, str],
) -> None:
    for _ in range(2):
        assert run_startup(startup_environment).returncode == 0
    calls = Path(startup_environment["STARTUP_TEST_CALL_LOG"]).read_text()
    assert "colima start" not in calls
    assert "compose up" not in calls
    assert calls.count("compose start db litellm") == 2


@pytest.mark.parametrize(
    ("settings", "error_text", "forbidden_call"),
    [
        (
            {"STARTUP_TEST_COLIMA_STATUS": "1", "STARTUP_TEST_COLIMA_START_EXIT": "1"},
            "Colima could not start",
            "docker ",
        ),
        ({"STARTUP_TEST_SERVICES": "other"}, "containers are missing", "compose start"),
        ({"STARTUP_TEST_COMPOSE_START_EXIT": "1"}, "Compose startup failed", "curl "),
        ({"STARTUP_TEST_HEALTH_EXIT": "7"}, "before the timeout", "Gateway ready"),
    ],
)
def test_startup_failures_are_explicit_and_stop_later_steps(
    startup_environment: dict[str, str],
    settings: dict[str, str],
    error_text: str,
    forbidden_call: str,
) -> None:
    startup_environment.update(settings)
    result = run_startup(startup_environment)
    calls = Path(startup_environment["STARTUP_TEST_CALL_LOG"]).read_text()

    assert result.returncode == 1
    assert error_text in result.stderr
    assert forbidden_call not in calls + result.stdout


@pytest.mark.parametrize("timeout", ["0", "-1", "601", "oops", "99999999999999999999"])
def test_invalid_timeout_fails_before_starting_any_service(
    startup_environment: dict[str, str], timeout: str
) -> None:
    startup_environment["LITELLM_STARTUP_TIMEOUT_SECONDS"] = timeout
    result = run_startup(startup_environment)

    assert result.returncode == 1
    assert not Path(startup_environment["STARTUP_TEST_CALL_LOG"]).exists()


def test_help_does_not_need_a_gateway_or_touch_services() -> None:
    result = run_startup({"PATH": os.defpath}, "--help")
    assert result.returncode == 0
    assert "LITELLM_COMPOSE_DIR" in result.stdout
