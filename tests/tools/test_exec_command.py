import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

import mini_agent.workspace_subprocess as workspace_subprocess
from mini_agent.exec_command_safety import sanitize_exec_command_arguments
from mini_agent.redaction_types import RedactionKind
from mini_agent.tools import exec_command
from mini_agent.tools.base import ToolError
from mini_agent.tools.exec_command import ExecCommandTool
from mini_agent.workspace import Workspace
from mini_agent.workspace_subprocess import (
    WorkspaceSubprocessLauncherError,
)


def test_exec_command_treats_shell_metacharacters_as_an_argv_argument(tmp_path: Path) -> None:
    tool = ExecCommandTool(Workspace(tmp_path))
    command = [
        sys.executable,
        "-c",
        "import sys; print(sys.argv[1])",
        "; echo shell_was_not_started",
    ]

    result = tool.execute(_arguments(command))

    assert "; echo shell_was_not_started" in result.content
    assert result.content.count("shell_was_not_started") == 1


def test_exec_command_resolves_user_path_without_inheriting_sensitive_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command_directory = tmp_path / "commands"
    command_directory.mkdir()
    command = command_directory / "from-user-path"
    command.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"${MINI_AGENT_TEST_SECRET-unset}\" \"$PATH\"\n"
    )
    command.chmod(command.stat().st_mode | 0o111)
    monkeypatch.setenv("PATH", str(command_directory))
    monkeypatch.setenv("MINI_AGENT_TEST_SECRET", "must-not-reach-child")
    tool = ExecCommandTool(Workspace(tmp_path))

    result = tool.execute(_arguments([command.name]))

    assert "stdout:\nunset\n" in result.content
    assert str(command_directory) in result.content
    assert "must-not-reach-child" not in result.content


def test_exec_command_interprets_relative_path_from_command_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher_directory = tmp_path / "launcher"
    command_directory = tmp_path / "workspace" / "commands"
    launcher_directory.mkdir()
    command_directory.mkdir(parents=True)
    _write_executable(launcher_directory / "run-me", "printf 'launcher\\n'\n")
    _write_executable(
        command_directory / "run-me",
        "printf 'command-cwd\\n'\nprintf 'path=%s\\n' \"$PATH\"\npeer\n",
    )
    _write_executable(command_directory / "peer", "printf 'nested-command-cwd\\n'\n")
    monkeypatch.chdir(launcher_directory)
    monkeypatch.setenv("PATH", ".")
    tool = ExecCommandTool(Workspace(tmp_path / "workspace"))

    result = tool.execute(_arguments(["run-me"], cwd="commands"))

    assert "command-cwd" in result.content
    assert "nested-command-cwd" in result.content
    assert "launcher" not in result.content
    assert "path=." in result.content


def test_exec_command_interprets_empty_path_entry_from_command_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher_directory = tmp_path / "launcher"
    command_directory = tmp_path / "workspace" / "commands"
    launcher_directory.mkdir()
    command_directory.mkdir(parents=True)
    _write_executable(launcher_directory / "run-me", "printf 'launcher\\n'\n")
    _write_executable(
        command_directory / "run-me",
        "printf 'command-cwd\\n'\nprintf 'path=%s\\n' \"$PATH\"\npeer\n",
    )
    _write_executable(command_directory / "peer", "printf 'nested-command-cwd\\n'\n")
    monkeypatch.chdir(launcher_directory)
    monkeypatch.setenv("PATH", "")
    tool = ExecCommandTool(Workspace(tmp_path / "workspace"))

    result = tool.execute(_arguments(["run-me"], cwd="commands"))

    assert "command-cwd" in result.content
    assert "nested-command-cwd" in result.content
    assert "launcher" not in result.content
    assert "path=\n" in result.content


def test_exec_command_searches_absolute_path_entries_before_workspace_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A ``.`` entry placed ahead of system directories must not let an
    # executable planted in the workspace shadow a real command: absolute PATH
    # entries are searched first, so the system binary wins even though a
    # same-named file exists in the command cwd.
    system_directory = tmp_path / "system"
    workspace = tmp_path / "workspace"
    commands = workspace / "commands"
    system_directory.mkdir()
    commands.mkdir(parents=True)
    _write_executable(system_directory / "run-me", "printf 'system-bin\\n'\n")
    _write_executable(commands / "run-me", "printf 'planted-workspace\\n'\n")
    monkeypatch.setenv("PATH", f"{system_directory}:.")
    tool = ExecCommandTool(Workspace(workspace))

    result = tool.execute(_arguments(["run-me"], cwd="commands"))

    assert "system-bin" in result.content
    assert "planted-workspace" not in result.content


def test_exec_command_reports_missing_executable_as_command_start_failure(tmp_path: Path) -> None:
    tool = ExecCommandTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments(["does-not-exist-anywhere"]))

    assert error_info.value.code == "command_start_failed"
    assert "not found" in str(error_info.value).lower()


def test_exec_command_rejects_cwd_escape_via_symlink(tmp_path: Path) -> None:
    outside_directory = tmp_path.parent / f"{tmp_path.name}-outside"
    outside_directory.mkdir()
    (tmp_path / "escape").symlink_to(outside_directory, target_is_directory=True)
    tool = ExecCommandTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments([sys.executable, "-c", "pass"], cwd="escape"))

    assert error_info.value.code == "invalid_path"


def test_exec_command_keeps_held_cwd_after_intermediate_directory_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    command_directory = workspace_root / "nested" / "command"
    command_directory.mkdir(parents=True)
    (command_directory / "marker.txt").write_text("held-cwd-content", encoding="utf-8")
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    secret = "outside-cwd-private-content"
    (outside_directory / "marker.txt").write_text(secret, encoding="utf-8")
    moved_directory = workspace_root / "nested" / "command-moved"
    observer = tmp_path / "observed-cwd-content"
    tool = ExecCommandTool(Workspace(workspace_root))
    original_start: Any = workspace_subprocess.WorkspaceSubprocessLauncher.start
    has_swapped = False

    def swap_cwd_before_bootstrap(
        launcher: workspace_subprocess.WorkspaceSubprocessLauncher,
        **kwargs: Any,
    ) -> subprocess.Popen[bytes]:
        nonlocal has_swapped
        has_swapped = True
        command_directory.rename(moved_directory)
        command_directory.symlink_to(outside_directory, target_is_directory=True)
        return original_start(launcher, **kwargs)

    monkeypatch.setattr(
        workspace_subprocess.WorkspaceSubprocessLauncher,
        "start",
        swap_cwd_before_bootstrap,
    )
    command = [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; "
            f"Path({str(observer)!r}).write_text(Path('marker.txt').read_text())"
        ),
    ]

    result = tool.execute(_arguments(command, cwd="nested/command"))

    assert has_swapped is True
    assert "Exit code: 0" in result.content
    assert observer.read_text(encoding="utf-8") == "held-cwd-content"
    assert secret not in result.content


def test_exec_command_passes_untrusted_arguments_outside_fixed_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = ExecCommandTool(Workspace(tmp_path))
    original_popen = subprocess.Popen
    observed_commands: list[list[str]] = []

    def record_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        command = cast(list[str], args[0])
        observed_commands.append(command)
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(exec_command.subprocess, "Popen", record_popen)
    untrusted_argument = "literal;not-bootstrap-code"
    result = tool.execute(
        _arguments(
            [
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                untrusted_argument,
            ]
        )
    )

    assert untrusted_argument in result.content
    assert len(observed_commands) == 1
    bootstrap_command = observed_commands[0]
    assert bootstrap_command[4] == workspace_subprocess.DIRECTORY_FD_BOOTSTRAP
    assert untrusted_argument not in bootstrap_command[4]
    assert bootstrap_command.count(untrusted_argument) == 1
    assert "-I" in bootstrap_command
    assert "-P" in bootstrap_command


def test_exec_command_rejects_active_interpreter_inside_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace_interpreter = workspace_root / "python"
    _write_executable(workspace_interpreter, "exit 0\n")
    monkeypatch.setattr(workspace_subprocess.sys, "executable", str(workspace_interpreter))

    with pytest.raises(WorkspaceSubprocessLauncherError) as error_info:
        ExecCommandTool(Workspace(workspace_root))

    assert str(error_info.value) == "Safe workspace subprocess launching is unavailable"


def test_exec_command_reports_timeout_without_child_output(tmp_path: Path) -> None:
    tool = ExecCommandTool(Workspace(tmp_path))
    command = [sys.executable, "-c", "import time; print('secret'); time.sleep(1)"]

    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments(command, timeout_seconds=0.01))

    assert error_info.value.code == "command_timeout"
    assert "secret" not in str(error_info.value)


def test_exec_command_timeout_kills_descendant_processes(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "descendant-survived"
    descendant_code = (
        f"import pathlib, time; time.sleep(0.3); pathlib.Path({str(marker)!r}).write_text('alive')"
    )
    command = [
        sys.executable,
        "-c",
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {descendant_code!r}]); "
        "time.sleep(10)",
    ]
    tool = ExecCommandTool(Workspace(tmp_path))
    processes, record_popen = _record_process_creation()

    with (
        patch.object(exec_command.subprocess, "Popen", side_effect=record_popen),
        pytest.raises(ToolError, match="time limit"),
    ):
        tool.execute(_arguments(command, timeout_seconds=0.05))

    assert processes[0].returncode is not None
    time.sleep(0.5)
    assert not marker.exists()


def test_exec_command_timeout_kills_descendant_after_main_process_exits(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "pipe-holder-survived"
    descendant_code = (
        f"import pathlib, time; time.sleep(0.5); pathlib.Path({str(marker)!r}).write_text('alive')"
    )
    command = [
        sys.executable,
        "-c",
        f"import subprocess, sys; subprocess.Popen([sys.executable, '-c', {descendant_code!r}])",
    ]
    tool = ExecCommandTool(Workspace(tmp_path))

    started_at = time.monotonic()
    with pytest.raises(ToolError, match="time limit") as error_info:
        tool.execute(_arguments(command, timeout_seconds=0.07))
    elapsed_seconds = time.monotonic() - started_at

    assert error_info.value.code == "command_timeout"
    assert elapsed_seconds < 0.35
    time.sleep(0.55)
    assert not marker.exists()


def test_exec_command_abort_retries_group_kill_before_reaping_direct_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    remaining_members = {"direct", "late-descendant"}
    tool = ExecCommandTool(Workspace(tmp_path))
    process = _FakeProcess(pid=1234, events=events)

    def killpg(process_group_id: int, kill_signal: signal.Signals) -> None:
        assert process_group_id == process.pid
        assert kill_signal is signal.SIGKILL
        events.append("kill-group")
        if len(events) == 1:
            remaining_members.remove("direct")
            return
        remaining_members.discard("late-descendant")
        if not remaining_members:
            raise ProcessLookupError

    def record_sleep(seconds: float) -> None:
        assert seconds == 0.01
        events.append("sleep")

    monkeypatch.setattr(exec_command.os, "killpg", killpg)
    monkeypatch.setattr(exec_command.time, "sleep", record_sleep)

    tool._abort_process_group(  # pyright: ignore[reportPrivateUsage]
        cast(subprocess.Popen[bytes], process)
    )

    assert remaining_members == set()
    assert events == [
        "kill-group",
        "sleep",
        "kill-group",
        "close",
        "close",
        "reap",
    ]


def test_exec_command_abort_stops_group_sweeps_when_group_is_already_gone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    tool = ExecCommandTool(Workspace(tmp_path))
    process = _FakeProcess(pid=1234, events=events)

    def missing_group(process_group_id: int, kill_signal: signal.Signals) -> None:
        assert process_group_id == process.pid
        assert kill_signal is signal.SIGKILL
        events.append("kill-group")
        raise ProcessLookupError

    monkeypatch.setattr(exec_command.os, "killpg", missing_group)

    tool._abort_process_group(  # pyright: ignore[reportPrivateUsage]
        cast(subprocess.Popen[bytes], process)
    )

    assert events == ["kill-group", "close", "close", "reap"]
    assert process.kill_count == 0


def test_exec_command_abort_falls_back_to_direct_child_on_group_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    tool = ExecCommandTool(Workspace(tmp_path))
    process = _FakeProcess(pid=1234, events=events)

    def denied_group(process_group_id: int, kill_signal: signal.Signals) -> None:
        assert process_group_id == process.pid
        assert kill_signal is signal.SIGKILL
        events.append("kill-group")
        raise PermissionError

    monkeypatch.setattr(exec_command.os, "killpg", denied_group)

    tool._abort_process_group(  # pyright: ignore[reportPrivateUsage]
        cast(subprocess.Popen[bytes], process)
    )

    assert events == ["kill-group", "close", "close", "reap"]
    assert process.kill_count == 1


def test_exec_command_interrupt_kills_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "interrupt-descendant-survived"
    descendant_code = (
        f"import pathlib, time; time.sleep(0.3); pathlib.Path({str(marker)!r}).write_text('alive')"
    )
    command = [
        sys.executable,
        "-c",
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {descendant_code!r}]); "
        "time.sleep(10)",
    ]
    tool = ExecCommandTool(Workspace(tmp_path))
    processes, record_popen = _record_process_creation()

    def raise_interrupt(*args: object, **kwargs: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(exec_command.select, "select", raise_interrupt)

    with (
        patch.object(exec_command.subprocess, "Popen", side_effect=record_popen),
        pytest.raises(KeyboardInterrupt),
    ):
        tool.execute(_arguments(command))

    assert processes[0].returncode is not None
    time.sleep(0.35)
    assert not marker.exists()


def test_exec_command_detached_pipe_holder_does_not_extend_deadline(tmp_path: Path) -> None:
    marker = tmp_path / "detached-pipe-holder-survived"
    descendant_code = (
        "import os, pathlib, time; "
        "os.setsid(); "
        "print('detached', flush=True); "
        "time.sleep(0.5); "
        f"pathlib.Path({str(marker)!r}).write_text('alive')"
    )
    command = [
        sys.executable,
        "-c",
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {descendant_code!r}]); "
        "time.sleep(0.05)",
    ]
    tool = ExecCommandTool(Workspace(tmp_path))

    started_at = time.monotonic()
    with pytest.raises(ToolError, match="time limit") as error_info:
        tool.execute(_arguments(command, timeout_seconds=0.2))
    elapsed_seconds = time.monotonic() - started_at

    assert error_info.value.code == "command_timeout"
    assert elapsed_seconds < 0.45
    time.sleep(0.55)
    # A descendant can call setsid() and leave the command's process group.
    assert marker.exists()


def test_exec_command_returns_exit_code_stdout_stderr_and_duration(tmp_path: Path) -> None:
    tool = ExecCommandTool(Workspace(tmp_path))
    command = [
        sys.executable,
        "-c",
        "import sys; print('out'); print('err', file=sys.stderr); raise SystemExit(7)",
    ]

    result = tool.execute(_arguments(command))

    assert "Exit code: 7" in result.content
    assert "Duration: " in result.content
    assert "stdout:\nout\n" in result.content
    assert "stderr:\nerr\n" in result.content
    assert result.is_truncated is False


def test_exec_command_redacts_child_output_before_returning_it(tmp_path: Path) -> None:
    tool = ExecCommandTool(Workspace(tmp_path))
    output_path = tmp_path / "child-output.txt"
    output_path.write_text("api_key=not-for-the-transcript", encoding="utf-8")
    command = [
        sys.executable,
        "-c",
        "import sys; from pathlib import Path; print(Path(sys.argv[1]).read_text())",
        str(output_path),
    ]

    result = tool.execute(_arguments(command))

    assert "not-for-the-transcript" not in result.content
    assert "[REDACTED]" in result.content
    assert result.source_redaction_summary.match_count == 1
    assert result.source_redaction_summary.kinds == (RedactionKind.NAMED_SECRET,)


def test_exec_command_keeps_output_up_to_a_hard_per_stream_limit(tmp_path: Path) -> None:
    tool = ExecCommandTool(Workspace(tmp_path), max_output_bytes=80)
    command = [
        sys.executable,
        "-c",
        "import sys; print('x' * 200); print('y' * 200, file=sys.stderr)",
    ]

    result = tool.execute(_arguments(command))

    assert result.is_truncated is True
    assert "[output truncated at 80 bytes per stream]" in result.content
    assert "x" * 80 in result.content
    assert "y" * 80 in result.content
    assert "x" * 81 not in result.content
    assert "y" * 81 not in result.content


@pytest.mark.parametrize(
    ("stream_name", "partial_secret"),
    (
        ("stdout", "sk_live_" + "0" * 3),
        ("stderr", "rk_test_" + "0" * 3),
    ),
)
def test_exec_command_redacts_partial_secret_at_each_truncated_stream_boundary(
    tmp_path: Path,
    stream_name: str,
    partial_secret: str,
) -> None:
    maximum_bytes = 80
    full_secret = partial_secret + "0" * 32
    filler = "x" * (maximum_bytes - len(partial_secret) - 1)
    payload = f"{filler} {full_secret}"
    output_path = tmp_path / f"{stream_name}-output.txt"
    output_path.write_text(payload, encoding="utf-8")
    write_expression = (
        f"sys.{stream_name}.write(open(sys.argv[1]).read()); sys.{stream_name}.flush()"
    )
    tool = ExecCommandTool(Workspace(tmp_path), max_output_bytes=maximum_bytes)

    result = tool.execute(
        _arguments([sys.executable, "-c", f"import sys; {write_expression}", str(output_path)])
    )
    context_result = result.render_for_context()

    assert partial_secret not in result.content
    assert partial_secret not in context_result
    assert f"{filler} [REDACTED]" in result.content
    assert f"[output truncated at {maximum_bytes} bytes per stream]" in result.content
    assert "[tool output was truncated at hard limit]" in context_result
    assert result.source_redaction_summary.match_count == 1
    assert result.source_redaction_summary.kinds == (RedactionKind.SECRET_PREFIX,)


def test_exec_command_combines_redaction_summaries_without_double_counting(
    tmp_path: Path,
) -> None:
    maximum_bytes = 80
    stdout_partial = "sk_live_" + "0" * 3
    stderr_partial = "AKIA" + "0" * 3
    stdout = (
        "x" * (maximum_bytes - len(stdout_partial) - 1)
        + " "
        + stdout_partial
        + "0" * 32
    )
    stderr = (
        "y" * (maximum_bytes - len(stderr_partial) - 1)
        + " "
        + stderr_partial
        + "0" * 32
    )
    stdout_path = tmp_path / "stdout.txt"
    stderr_path = tmp_path / "stderr.txt"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "sys.stdout.write(open(sys.argv[1]).read()); sys.stdout.flush(); "
            "sys.stderr.write(open(sys.argv[2]).read()); sys.stderr.flush()"
        ),
        str(stdout_path),
        str(stderr_path),
    ]
    tool = ExecCommandTool(Workspace(tmp_path), max_output_bytes=maximum_bytes)

    result = tool.execute(_arguments(command))

    assert stdout_partial not in result.content
    assert stderr_partial not in result.content
    assert result.content.count("[REDACTED]") == 2
    assert result.source_redaction_summary.match_count == 2
    assert result.source_redaction_summary.kinds == (RedactionKind.SECRET_PREFIX,)


def test_exec_command_retains_short_prefixes_from_complete_streams(tmp_path: Path) -> None:
    stdout_prefix = "sk_live_" + "0" * 3
    stderr_prefix = "rk_test_" + "0" * 3
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            f"sys.stdout.write({stdout_prefix!r}); "
            f"sys.stderr.write({stderr_prefix!r})"
        ),
    ]
    tool = ExecCommandTool(Workspace(tmp_path), max_output_bytes=80)

    result = tool.execute(_arguments(command))

    assert stdout_prefix in result.content
    assert stderr_prefix in result.content
    assert result.is_truncated is False
    assert result.source_redaction_summary.match_count == 0
    assert result.source_redaction_summary.kinds == ()


def test_exec_command_rejects_unknown_arguments(tmp_path: Path) -> None:
    tool = ExecCommandTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"argv":["echo","hello"],"unexpected":true}')

    assert error_info.value.code == "invalid_arguments"


@pytest.mark.parametrize(
    ("argv", "cwd"),
    (
        ([sys.executable, "-c", "print('sk_live_" + "0" * 24 + "')"], "."),
        ([sys.executable, "-c", "print('[REDACTED]')"], "."),
        ([sys.executable, "-c", "pass"], "[REDACTED_KEY_1]"),
    ),
)
def test_exec_command_rejects_sensitive_or_redacted_inputs_before_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    cwd: str,
) -> None:
    tool = ExecCommandTool(Workspace(tmp_path))

    def fail_popen(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("sensitive command must not start")

    monkeypatch.setattr(exec_command.subprocess, "Popen", fail_popen)

    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments(argv, cwd=cwd))

    assert error_info.value.code == "sensitive_command_arguments"
    assert "sk_live_" not in str(error_info.value)


@pytest.mark.parametrize(
    "argv",
    (
        ["echo", "--password", "example-password-value"],
        ["echo", "--api-key=example-api-key-value"],
        ["echo", "--header", "Authorization: Bearer example-token-value"],
        ["echo", "--url=https://example-user:example-password@example.test/path"],
    ),
)
def test_exec_command_rejects_sensitive_long_option_forms_before_popen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    tool = ExecCommandTool(Workspace(tmp_path))

    def fail_popen(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("sensitive command must not start")

    monkeypatch.setattr(exec_command.subprocess, "Popen", fail_popen)

    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments(argv))

    assert error_info.value.code == "sensitive_command_arguments"
    assert "example-" not in str(error_info.value)


def test_exec_command_sanitizer_redacts_long_option_value_and_preserves_normal_argv() -> None:
    raw = _arguments(["tool", "--password", "example-password-value", "--port", "4000"])

    safe = sanitize_exec_command_arguments(raw)

    assert safe.has_sensitive_content is True
    assert "example-password-value" not in safe.arguments_json
    assert "--password" not in safe.arguments_json
    assert '"--port","4000"' in safe.arguments_json


def test_exec_command_sanitizer_fail_closes_unknown_env_field_without_retaining_value() -> None:
    raw_value = "example-environment-value"
    raw = json.dumps(
        {"argv": ["echo", "normal"], "env": ["API_KEY", raw_value]}
    )

    safe = sanitize_exec_command_arguments(raw)

    assert safe.has_sensitive_content is True
    assert raw_value not in safe.arguments_json
    assert safe.arguments_json == '{"argv":["[REDACTED]"],"cwd":"[REDACTED]"}'


def test_exec_command_sanitizer_keeps_ordinary_long_option() -> None:
    safe = sanitize_exec_command_arguments(_arguments(["echo", "--port", "4000"]))

    assert safe.has_sensitive_content is False
    assert json.loads(safe.arguments_json)["argv"] == ["echo", "--port", "4000"]


@pytest.mark.parametrize(
    "argv",
    (
        ["curl", "-u", "account:{credential}"],
        ["curl", "-Uproxy:{credential}"],
        ["curl", "--user", "account:{credential}"],
        ["curl", "--proxy-user=proxy:{credential}"],
        ["docker", "login", "-p", "{credential}"],
        ["docker", "login", "--password={credential}"],
        ["podman", "login", "-p{credential}"],
        ["mysql", "-p{credential}"],
        ["mariadb", "-p", "{credential}"],
        ["mongosh", "-p", "{credential}"],
        ["redis-cli", "-a{credential}"],
        ["redis-cli", "--pass", "{credential}"],
        ["sshpass", "-p", "{credential}"],
    ),
)
def test_exec_command_short_credential_options_are_redacted_and_rejected_preflight(
    tmp_path: Path,
    argv: list[str],
) -> None:
    """Opaque values must not reach approval records or command execution."""

    tool = ExecCommandTool(Workspace(tmp_path))
    credential = _opaque_credential_value()
    raw = _arguments([value.replace("{credential}", credential) for value in argv])

    safe = sanitize_exec_command_arguments(raw)

    assert safe.has_sensitive_content is True
    assert credential not in safe.arguments_json
    with pytest.raises(ToolError) as error_info:
        tool.preflight(raw)
    assert error_info.value.code == "sensitive_command_arguments"
    assert credential not in str(error_info.value)


@pytest.mark.parametrize(
    "argv",
    (
        ["curl", "-uaccount:{credential}"],
        ["curl", "-suaccount:{credential}"],
        ["curl", "-su", "account:{credential}"],
        ["/usr/bin/curl", "-suaccount:{credential}"],
        [
            "sudo",
            "-u",
            "ordinary-user",
            "/usr/bin/curl",
            "-suaccount:{credential}",
        ],
        ["sshpass", "-vp{credential}", "/usr/bin/true"],
        ["sshpass", "-vp", "{credential}", "/usr/bin/true"],
        ["mysql", "-vp{credential}"],
        ["mariadb", "-vp", "{credential}"],
        ["redis-cli", "-xa{credential}"],
        ["redis-cli", "-xa", "{credential}"],
        ["docker", "login", "-xp{credential}"],
        ["docker", "login", "-xp", "{credential}"],
    ),
)
def test_exec_command_combined_short_credential_options_are_redacted_and_rejected(
    tmp_path: Path,
    argv: list[str],
) -> None:
    tool = ExecCommandTool(Workspace(tmp_path))
    credential = _opaque_credential_value()
    concrete_argv = [value.replace("{credential}", credential) for value in argv]
    raw = _arguments(concrete_argv)

    safe = sanitize_exec_command_arguments(raw)

    assert safe.has_sensitive_content is True
    assert credential not in safe.arguments_json
    with pytest.raises(ToolError) as error_info:
        tool.preflight(raw)
    assert error_info.value.code == "sensitive_command_arguments"
    assert credential not in str(error_info.value)


def test_exec_command_conservatively_rejects_ambiguous_combined_short_option_value(
    tmp_path: Path,
) -> None:
    """A preceding curl option value can contain ``u``; fail closed by design."""

    tool = ExecCommandTool(Workspace(tmp_path))
    raw = _arguments(["curl", "-duser=ordinary"])

    safe = sanitize_exec_command_arguments(raw)

    assert safe.has_sensitive_content is True
    assert json.loads(safe.arguments_json)["argv"] == ["curl", "[REDACTED]"]
    with pytest.raises(ToolError) as error_info:
        tool.preflight(raw)
    assert error_info.value.code == "sensitive_command_arguments"


@pytest.mark.parametrize(
    "argv",
    (
        ["env", "curl", "-u", "account:{credential}"],
        ["env", "-i", "curl", "-u", "account:{credential}"],
        [
            "env",
            "--ignore-environment",
            "--unset",
            "SAFE_VARIABLE",
            "curl",
            "-u",
            "account:{credential}",
        ],
        ["env", "--unset=SAFE_VARIABLE", "curl", "-u", "account:{credential}"],
        ["/usr/bin/env", "/usr/bin/curl", "-u", "account:{credential}"],
        ["env", "--", "curl", "-u", "account:{credential}"],
        ["env", "--", "curl", "--password", "{credential}"],
        ["env", "--", "RUNTIME_VALUE={credential}", "echo", "ordinary-output"],
        ["env", "RUNTIME_VALUE={credential}", "echo", "ordinary-output"],
    ),
)
def test_exec_command_env_wrapper_redacts_credentials_before_durable_use(
    tmp_path: Path,
    argv: list[str],
) -> None:
    tool = ExecCommandTool(Workspace(tmp_path))
    credential = _opaque_credential_value()
    concrete_argv = [value.replace("{credential}", credential) for value in argv]
    raw = _arguments(concrete_argv)

    safe = sanitize_exec_command_arguments(raw)

    assert safe.has_sensitive_content is True
    assert credential not in safe.arguments_json
    with pytest.raises(ToolError) as error_info:
        tool.preflight(raw)
    assert error_info.value.code == "sensitive_command_arguments"
    assert credential not in str(error_info.value)


@pytest.mark.parametrize(
    "argv",
    (
        ["sudo", "-u", "ordinary-user", "curl", "-u", "account:{credential}"],
        [
            "/usr/bin/sudo",
            "-u",
            "ordinary-user",
            "/usr/bin/curl",
            "-u",
            "account:{credential}",
        ],
        ["nice", "-n", "10", "curl", "-u", "account:{credential}"],
        ["nohup", "curl", "-u", "account:{credential}"],
        ["timeout", "5", "curl", "-u", "account:{credential}"],
        ["unknown-wrapper", "--mode", "safe", "curl", "-u", "account:{credential}"],
        ["sudo", "--", "curl", "--password", "{credential}"],
        ["sudo", "--", "arbitrary-command", "--password", "{credential}"],
        ["unknown-wrapper", "--", "arbitrary-command", "--api-key", "{credential}"],
        ["sudo", "-u", "ordinary-user", "docker", "login", "-p", "{credential}"],
        ["nice", "mysql", "-p{credential}"],
        ["nohup", "redis-cli", "-a{credential}"],
        ["timeout", "10", "sshpass", "-p", "{credential}"],
    ),
)
def test_exec_command_detects_credential_commands_inside_any_wrapper(
    tmp_path: Path,
    argv: list[str],
) -> None:
    tool = ExecCommandTool(Workspace(tmp_path))
    credential = _opaque_credential_value()
    concrete_argv = [value.replace("{credential}", credential) for value in argv]
    raw = _arguments(concrete_argv)

    safe = sanitize_exec_command_arguments(raw)

    assert safe.has_sensitive_content is True
    assert credential not in safe.arguments_json
    with pytest.raises(ToolError) as error_info:
        tool.preflight(raw)
    assert error_info.value.code == "sensitive_command_arguments"
    assert credential not in str(error_info.value)


@pytest.mark.parametrize(
    "argv",
    (
        ["env"],
        ["env", "-i"],
        ["env", "-u"],
        ["env", "-S", "curl -u opaque-value"],
        ["env", "-C", "other-directory", "curl"],
        ["env", "env", "curl", "-u", "opaque-value"],
        ["/usr/bin/env", "/usr/bin/env", "curl", "-u", "opaque-value"],
    ),
)
def test_exec_command_rejects_ambiguous_env_wrapper_without_retaining_argv(
    tmp_path: Path,
    argv: list[str],
) -> None:
    tool = ExecCommandTool(Workspace(tmp_path))
    raw = _arguments(argv)

    safe = sanitize_exec_command_arguments(raw)

    assert safe.has_sensitive_content is True
    assert json.loads(safe.arguments_json)["argv"] == ["[REDACTED]"] * len(argv)
    with pytest.raises(ToolError) as error_info:
        tool.preflight(raw)
    assert error_info.value.code == "sensitive_command_arguments"


@pytest.mark.parametrize(
    "argv",
    (
        ["mkdir", "-p", "nested/path"],
        ["docker", "run", "-p", "8080:80", "example-image"],
        ["env", "-i", "mkdir", "-p", "nested/path"],
        ["env", "--unset=SAFE_VARIABLE", "docker", "run", "-p", "8080:80", "example-image"],
        ["/usr/bin/env", "docker", "run", "-p", "8080:80", "example-image"],
        ["sudo", "-u", "ordinary-user", "mkdir", "-p", "nested/path"],
        ["nice", "docker", "run", "-p", "8080:80", "example-image"],
        ["echo", "-u", "ordinary-user"],
        ["curl", "--", "-u", "{credential}"],
        ["curl", "--", "-su", "account:{credential}"],
        ["env", "--", "curl", "--", "-u", "{credential}"],
        ["sudo", "--", "curl", "--", "-u", "{credential}"],
        ["echo", "--", "--password", "{credential}"],
    ),
)
def test_exec_command_short_options_keep_noncredential_arguments(
    tmp_path: Path,
    argv: list[str],
) -> None:
    tool = ExecCommandTool(Workspace(tmp_path))
    credential = _opaque_credential_value()
    concrete_argv = [value.replace("{credential}", credential) for value in argv]
    raw = _arguments(concrete_argv)

    safe = sanitize_exec_command_arguments(raw)

    assert safe.has_sensitive_content is False
    assert json.loads(safe.arguments_json)["argv"] == concrete_argv
    tool.preflight(raw)


def test_exec_command_does_not_treat_options_after_double_dash_as_sensitive_flags(
    tmp_path: Path,
) -> None:
    tool = ExecCommandTool(Workspace(tmp_path))

    result = tool.execute(_arguments(["echo", "--", "--password"]))

    assert "--password" in result.content


def _arguments(
    argv: list[str],
    *,
    cwd: str = ".",
    timeout_seconds: float = 30.0,
) -> str:
    return json.dumps({"argv": argv, "cwd": cwd, "timeout_seconds": timeout_seconds})


def _opaque_credential_value() -> str:
    """Construct a runtime-only opaque value for command preflight coverage."""

    return "opaque-" + "credential"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}")
    path.chmod(path.stat().st_mode | 0o111)


def _record_process_creation() -> tuple[
    list[subprocess.Popen[Any]],
    Any,
]:
    processes: list[subprocess.Popen[Any]] = []
    original_popen = subprocess.Popen

    def record_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    return processes, record_popen


class _FakePipe:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def close(self) -> None:
        self._events.append("close")


class _FakeProcess:
    def __init__(self, *, pid: int, events: list[str]) -> None:
        self.pid = pid
        self._events = events
        self.stdout = _FakePipe(events)
        self.stderr = _FakePipe(events)
        self.kill_count = 0

    def kill(self) -> None:
        self.kill_count += 1

    def wait(self, *, timeout: float) -> int:
        assert timeout == 1
        self._events.append("reap")
        return 0
