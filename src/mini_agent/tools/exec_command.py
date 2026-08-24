"""Approval-required execution of explicit commands inside a workspace."""

from __future__ import annotations

import os
import select
import signal
import stat
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError

from mini_agent.exec_command_safety import has_sensitive_exec_command_arguments
from mini_agent.redaction import redact_text
from mini_agent.redaction_types import RedactionSummary
from mini_agent.tool_facts import ToolCompletionFacts
from mini_agent.tools.base import ToolDefinition, ToolError, ToolResult
from mini_agent.workspace import Workspace, WorkspacePathError
from mini_agent.workspace_directory_fd import WorkspaceDirectoryFdError
from mini_agent.workspace_subprocess import WorkspaceSubprocessLauncher

_PROCESS_GROUP_KILL_SWEEP_COUNT = 3
_PROCESS_GROUP_KILL_SWEEP_INTERVAL_SECONDS = 0.01


class ExecCommandArguments(BaseModel):
    """One command invocation expressed as an argument vector, never a shell string."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    argv: tuple[StrictStr, ...] = Field(min_length=1)
    cwd: StrictStr = "."
    timeout_seconds: Annotated[float, Field(strict=True, gt=0, le=120)] = 30.0


class ExecCommandTool:
    """Run an argv command after the agent's non-read-only approval flow."""

    def __init__(self, workspace: Workspace, *, max_output_bytes: int = 1_048_576) -> None:
        if max_output_bytes < 64:
            raise ValueError("max_output_bytes must be at least 64")
        self._workspace = workspace
        self._max_output_bytes = max_output_bytes
        self._user_path = os.environ.get("PATH", os.defpath)
        self._subprocess_launcher = WorkspaceSubprocessLauncher(workspace.root)

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="exec_command",
            description=(
                "Run an explicit argv command in a workspace directory. "
                "Shell command strings are unsupported. Every execution requires approval."
            ),
            parameters=ExecCommandArguments.model_json_schema(),
            is_read_only=False,
        )

    def execute(self, arguments_json: str) -> ToolResult:
        arguments, cwd = self._parse_arguments(arguments_json)
        started_at = time.monotonic()
        try:
            with self._workspace.directory_anchor.open_existing_directory(
                arguments.cwd
            ) as cwd_descriptor:
                executable = self._resolve_executable(arguments.argv[0], cwd=cwd)
                process = self._subprocess_launcher.start(
                    executable=executable,
                    argv=(executable, *arguments.argv[1:]),
                    directory_descriptor=cwd_descriptor,
                    environment=self._environment(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
        except WorkspaceDirectoryFdError as error:
            raise ToolError(
                "invalid_path", "Command cwd cannot be opened safely"
            ) from error
        except OSError as error:
            raise ToolError("command_start_failed", "Could not start command") from error

        stdout_capture = _BoundedCapture(self._max_output_bytes)
        stderr_capture = _BoundedCapture(self._max_output_bytes)
        deadline = started_at + arguments.timeout_seconds
        completed = False
        try:
            completed = self._capture_until_deadline(
                process,
                stdout_capture=stdout_capture,
                stderr_capture=stderr_capture,
                deadline=deadline,
            )
            if not completed:
                duration_seconds = time.monotonic() - started_at
                message = f"Command exceeded its time limit after {duration_seconds:.3f} seconds"
                raise ToolError("command_timeout", message)

            duration_seconds = time.monotonic() - started_at
            return self._result_from_captures(
                exit_code=process.returncode,
                duration_seconds=duration_seconds,
                stdout_capture=stdout_capture,
                stderr_capture=stderr_capture,
            )
        finally:
            if not completed:
                self._abort_process_group(process)

    def preflight(self, arguments_json: str) -> None:
        """Reject unsafe command text before the agent asks for approval."""

        self._parse_arguments(arguments_json)

    def _parse_arguments(self, arguments_json: str) -> tuple[ExecCommandArguments, str]:
        try:
            arguments = ExecCommandArguments.model_validate_json(arguments_json)
        except ValidationError as error:
            raise ToolError("invalid_arguments", "Invalid exec_command arguments") from error

        self._reject_sensitive_command_arguments(arguments)
        try:
            cwd = self._workspace.resolve_existing(arguments.cwd, allow_directory=True)
        except (WorkspacePathError, FileNotFoundError) as error:
            raise ToolError("invalid_path", str(error)) from error

        if not cwd.is_dir():
            raise ToolError("invalid_path", "Command cwd must be a workspace directory")
        return arguments, str(cwd)

    def _reject_sensitive_command_arguments(self, arguments: ExecCommandArguments) -> None:
        if has_sensitive_exec_command_arguments(argv=arguments.argv, cwd=arguments.cwd):
            raise ToolError(
                "sensitive_command_arguments",
                "Command arguments contain sensitive or redacted text",
            )

    def _resolve_executable(self, requested_executable: str, *, cwd: str) -> str:
        """Resolve the requested executable before starting the minimal-env child."""
        candidate = self._find_executable(requested_executable, cwd=cwd)
        if candidate is None:
            message = "Command executable was not found or is not executable"
            raise ToolError("command_start_failed", message)

        try:
            executable = candidate.resolve(strict=True)
            mode = executable.stat().st_mode
        except OSError as error:
            message = "Command executable could not be resolved"
            raise ToolError("command_start_failed", message) from error

        if not stat.S_ISREG(mode) or not os.access(executable, os.X_OK):
            message = "Command executable was not found or is not executable"
            raise ToolError("command_start_failed", message)
        return str(executable)

    def _find_executable(self, requested_executable: str, *, cwd: str) -> Path | None:
        if os.sep not in requested_executable:
            for directory in self._path_entries_for_cwd(cwd):
                candidate = directory / requested_executable
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return candidate
            return None

        requested_path = Path(requested_executable)
        return requested_path if requested_path.is_absolute() else Path(cwd) / requested_path

    def _environment(self) -> dict[str, str]:
        """Use a minimal, credential-free environment for the child process."""
        # Relative and empty PATH entries are intentionally retained. The
        # fixed bootstrap changes to the held directory descriptor before the
        # approved process starts, so POSIX resolves those entries from that
        # same directory without a vulnerable absolute pathname rewrite.
        return {"PATH": self._user_path, "LANG": "C", "LC_ALL": "C"}

    def _path_entries_for_cwd(self, cwd: str) -> tuple[Path, ...]:
        """Interpret relative PATH entries from the command's working directory.

        ``Popen(..., cwd=...)`` makes a relative (including empty) PATH entry
        relative to that directory in the child.  Resolve the initial executable
        with the same rule, then pass that normalized PATH to the child so
        nested command lookups retain identical semantics.
        """
        command_cwd = Path(cwd)
        entries: list[Path] = []
        for entry in self._user_path.split(os.pathsep):
            path = Path(entry) if entry else command_cwd
            entries.append(path if path.is_absolute() else command_cwd / path)
        return tuple(entries)

    def _capture_until_deadline(
        self,
        process: subprocess.Popen[bytes],
        *,
        stdout_capture: _BoundedCapture,
        stderr_capture: _BoundedCapture,
        deadline: float,
    ) -> bool:
        """Drain both pipes without threads while respecting one total deadline."""
        streams = self._output_streams(
            process,
            stdout_capture=stdout_capture,
            stderr_capture=stderr_capture,
        )
        while streams or process.poll() is None:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                return False
            if not streams:
                try:
                    process.wait(timeout=remaining_seconds)
                except subprocess.TimeoutExpired:
                    return False
                continue

            readable, _, _ = select.select(list(streams), [], [], remaining_seconds)
            if not readable:
                return False
            for descriptor in readable:
                chunk = os.read(descriptor, 65_536)
                if chunk:
                    streams[descriptor].append(chunk)
                else:
                    streams.pop(descriptor)
        return True

    def _output_streams(
        self,
        process: subprocess.Popen[bytes],
        *,
        stdout_capture: _BoundedCapture,
        stderr_capture: _BoundedCapture,
    ) -> dict[int, _BoundedCapture]:
        streams: dict[int, _BoundedCapture] = {}
        for pipe, capture in (
            (process.stdout, stdout_capture),
            (process.stderr, stderr_capture),
        ):
            if pipe is None:
                raise RuntimeError("Command output pipe is unavailable")
            streams[pipe.fileno()] = capture
        return streams

    def _abort_process_group(self, process: subprocess.Popen[bytes]) -> None:
        """Abort a group before reaping its direct child or closing its pipes."""
        self._kill_process_group_with_sweeps(process)
        self._close_output_pipes(process)
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired as error:
            message = "Could not reap command process after terminating its process group"
            raise RuntimeError(message) from error

    def _kill_process_group_with_sweeps(
        self,
        process: subprocess.Popen[bytes],
    ) -> None:
        """Kill late group members while the direct child PID cannot be reused.

        A child can begin a fork just before the first group signal is delivered.
        Keep the direct child unreaped and issue a small number of follow-up
        signals, giving that late member a bounded chance to join and be killed.
        Once the direct child is reaped, never address its former PGID again.
        """

        for sweep_index in range(_PROCESS_GROUP_KILL_SWEEP_COUNT):
            if not self._kill_process_group_once(process):
                return
            if sweep_index + 1 < _PROCESS_GROUP_KILL_SWEEP_COUNT:
                time.sleep(_PROCESS_GROUP_KILL_SWEEP_INTERVAL_SECONDS)

    def _kill_process_group_once(self, process: subprocess.Popen[bytes]) -> bool:
        """Kill the original process group and report whether it still existed."""
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return False
        except PermissionError:
            # The original group can disappear or become unaddressable after a
            # descendant calls setsid(). The direct child is still ours.
            with suppress(ProcessLookupError):
                process.kill()
            return False
        return True

    def _close_output_pipes(self, process: subprocess.Popen[bytes]) -> None:
        for pipe in (process.stdout, process.stderr):
            if pipe is None:
                continue
            with suppress(OSError, ValueError):
                pipe.close()

    def _result_from_captures(
        self,
        *,
        exit_code: int | None,
        duration_seconds: float,
        stdout_capture: _BoundedCapture,
        stderr_capture: _BoundedCapture,
    ) -> ToolResult:
        if exit_code is None:
            raise RuntimeError("Completed command has no exit code")

        stdout_redaction = redact_text(
            stdout_capture.text,
            truncated_at_end=stdout_capture.was_truncated,
        )
        stderr_redaction = redact_text(
            stderr_capture.text,
            truncated_at_end=stderr_capture.was_truncated,
        )
        was_truncated = stdout_capture.was_truncated or stderr_capture.was_truncated
        truncation_notice = ""
        if was_truncated:
            truncation_notice = (
                f"\n\n[output truncated at {self._max_output_bytes} bytes per stream]"
            )
        captured_content = (
            self._render_result(
                exit_code=exit_code,
                duration_seconds=duration_seconds,
                stdout=stdout_redaction.text,
                stderr=stderr_redaction.text,
            )
            + truncation_notice
        )
        return ToolResult(
            content=captured_content,
            is_truncated=was_truncated,
            source_redaction_summary=RedactionSummary(
                match_count=(
                    stdout_redaction.match_count + stderr_redaction.match_count
                ),
                kinds=tuple(
                    sorted(
                        {
                            *stdout_redaction.matched_kinds,
                            *stderr_redaction.matched_kinds,
                        },
                        key=lambda kind: kind.value,
                    )
                ),
            ),
            facts=ToolCompletionFacts(command_exit_code=exit_code),
        )

    def _render_result(
        self,
        *,
        exit_code: int,
        duration_seconds: float,
        stdout: str,
        stderr: str,
    ) -> str:
        return (
            f"Exit code: {exit_code}\n"
            f"Duration: {duration_seconds:.3f}s\n\n"
            f"stdout:\n{stdout}\n\n"
            f"stderr:\n{stderr}"
        )


class _BoundedCapture:
    """Drain one pipe while retaining no more than a fixed number of bytes."""

    def __init__(self, maximum_bytes: int) -> None:
        self._maximum_bytes = maximum_bytes
        self._data = bytearray()
        self.was_truncated = False

    @property
    def text(self) -> str:
        return bytes(self._data).decode("utf-8", errors="replace")

    def append(self, chunk: bytes) -> None:
        remaining = self._maximum_bytes - len(self._data)
        if remaining <= 0:
            self.was_truncated = True
            return
        self._data.extend(chunk[:remaining])
        if len(chunk) > remaining:
            self.was_truncated = True
