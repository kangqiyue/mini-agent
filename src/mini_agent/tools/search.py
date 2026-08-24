"""Bounded ripgrep search inside a workspace."""

from __future__ import annotations

import os
import select
import subprocess
import time
from dataclasses import dataclass
from json import JSONDecodeError, loads
from os import read
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mini_agent.private_key_material import (
    FileScanLimitExceededError,
    contains_private_key_pem,
    read_bounded_open_file_bytes,
)
from mini_agent.tools.base import ToolDefinition, ToolError, ToolResult
from mini_agent.workspace import (
    SensitiveWorkspacePathError,
    Workspace,
    WorkspacePathError,
)
from mini_agent.workspace_directory_fd import WorkspaceDirectoryFdError
from mini_agent.workspace_subprocess import WorkspaceSubprocessLauncher

_GLOB_METACHARACTERS = frozenset("\\*?[]{}!")
_DEFAULT_MAX_SCAN_BYTES = 4_000_000
_DEFAULT_MAX_SCANNED_FILES = 128
_DEFAULT_MAX_TOTAL_SCAN_BYTES = 16_000_000


@dataclass(frozen=True)
class _SearchMatch:
    relative_path: str
    line_number: int
    column_number: int
    line: str


class SearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, max_length=512)
    path: str = "."
    globs: tuple[str, ...] = Field(default=(), max_length=20)


class SearchTool:
    def __init__(
        self,
        workspace: Workspace,
        *,
        max_matches: int = 2_000,
        max_chars: int = 256_000,
        timeout_seconds: float = 10,
        executable: str = "rg",
        max_scan_bytes: int = _DEFAULT_MAX_SCAN_BYTES,
        max_scanned_files: int = _DEFAULT_MAX_SCANNED_FILES,
        max_total_scan_bytes: int = _DEFAULT_MAX_TOTAL_SCAN_BYTES,
    ) -> None:
        if (
            max_matches < 1
            or max_chars < 1
            or timeout_seconds <= 0
            or max_scan_bytes < 1
            or max_scanned_files < 1
            or max_total_scan_bytes < 1
        ):
            raise ValueError("Search limits must be positive")
        self._workspace = workspace
        self._max_matches = max_matches
        self._max_chars = max_chars
        self._timeout_seconds = timeout_seconds
        self._executable = executable
        self._max_scan_bytes = max_scan_bytes
        self._max_scanned_files = max_scanned_files
        self._max_total_scan_bytes = max_total_scan_bytes
        self._launch_cwd = Path.cwd()
        self._launch_path = os.environ.get("PATH", os.defpath)
        self._subprocess_launcher = WorkspaceSubprocessLauncher(workspace.root)

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search",
            description=(
                "Search workspace text with hard match and output limits. "
                "The complete bounded result can be saved as an artifact."
            ),
            parameters=SearchArguments.model_json_schema(),
            is_read_only=True,
        )

    def execute(self, arguments_json: str) -> ToolResult:
        try:
            arguments = SearchArguments.model_validate_json(arguments_json)
            search_path = self._workspace.resolve_existing(arguments.path, allow_directory=True)
        except ValidationError as error:
            raise ToolError("invalid_arguments", "Invalid search arguments") from error
        except SensitiveWorkspacePathError as error:
            raise ToolError(
                "sensitive_path", "Sensitive workspace paths cannot be searched"
            ) from error
        except (WorkspacePathError, FileNotFoundError) as error:
            raise ToolError("invalid_path", str(error)) from error

        executable = self._resolve_executable()

        search_relative_path = self._workspace.relative(search_path)
        command = self._command(executable, arguments, search_path, search_relative_path)
        output, reached_byte_limit, returncode = self._run_bounded(command)
        if returncode not in (0, 1) and not reached_byte_limit:
            raise ToolError("search_failed", "ripgrep could not complete the search")
        if returncode == 1 and not output:
            return ToolResult(content="No matches found.")
        matches = self._parse_matches(output, reached_byte_limit=reached_byte_limit)
        return self._bounded_result(matches, reached_byte_limit=reached_byte_limit)

    def _resolve_executable(self) -> str:
        for candidate in self._executable_candidates():
            try:
                resolved_candidate = candidate.expanduser().resolve(strict=True)
            except OSError:
                continue
            if not resolved_candidate.is_file() or not os.access(resolved_candidate, os.X_OK):
                continue
            if resolved_candidate.is_relative_to(self._workspace.root):
                raise ToolError(
                    "unsafe_dependency",
                    "ripgrep executable cannot be located inside the workspace",
                )
            return str(resolved_candidate)
        raise ToolError("missing_dependency", "ripgrep is not installed")

    def _executable_candidates(self) -> tuple[Path, ...]:
        executable_path = Path(self._executable)
        if executable_path.is_absolute() or executable_path.parent != Path("."):
            return (self._resolve_launch_relative_path(executable_path),)

        candidates: list[Path] = []
        for path_entry in self._launch_path.split(os.pathsep):
            base_directory = self._launch_cwd if not path_entry else Path(path_entry)
            candidates.append(self._resolve_launch_relative_path(base_directory / executable_path))
        return tuple(candidates)

    def _resolve_launch_relative_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self._launch_cwd / path

    def _command(
        self,
        executable: str,
        arguments: SearchArguments,
        search_path: Path,
        search_relative_path: str,
    ) -> list[str]:
        command = [
            executable,
            "--no-config",
            "--no-follow",
            "--json",
        ]
        for glob in arguments.globs:
            command.extend(("--glob", glob))
        for excluded_root in self._workspace.excluded_roots_below(search_path):
            excluded_path = excluded_root.relative_to(self._workspace.root).as_posix()
            escaped_path = _escape_glob_literal(excluded_path)
            command.extend(("--glob", f"!{escaped_path}"))
            command.extend(("--glob", f"!{escaped_path}/**"))
        command.extend(("--", arguments.query, search_relative_path))
        return command

    def _run_bounded(self, command: list[str]) -> tuple[str, bool, int]:
        """Collect at most a fixed byte count; terminate rg before pipe buffering grows."""
        maximum_bytes = self._max_chars * 4
        try:
            with self._workspace.directory_anchor.open_existing_directory(
                "."
            ) as root_descriptor:
                process = self._subprocess_launcher.start(
                    executable=command[0],
                    argv=command,
                    directory_descriptor=root_descriptor,
                    environment=self._environment(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
        except (OSError, WorkspaceDirectoryFdError) as error:
            raise ToolError("search_failed", "ripgrep could not start safely") from error
        assert process.stdout is not None

        output = bytearray()
        deadline = time.monotonic() + self._timeout_seconds
        reached_byte_limit = False
        completed = False
        try:
            while True:
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise ToolError("search_timeout", "Search exceeded its time limit")
                readable, _, _ = select.select([process.stdout], [], [], remaining_seconds)
                if not readable:
                    raise ToolError("search_timeout", "Search exceeded its time limit")
                chunk = read(
                    process.stdout.fileno(),
                    min(8_192, maximum_bytes - len(output) + 1),
                )
                if not chunk:
                    break
                output.extend(chunk)
                if len(output) > maximum_bytes:
                    reached_byte_limit = True
                    del output[maximum_bytes:]
                    break
            if reached_byte_limit:
                process.terminate()
                process.wait(timeout=1)
                completed = True
                return (
                    output.decode("utf-8", errors="replace"),
                    reached_byte_limit,
                    process.returncode,
                )
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise ToolError("search_timeout", "Search exceeded its time limit")
            try:
                process.wait(timeout=remaining_seconds)
            except subprocess.TimeoutExpired as error:
                raise ToolError("search_timeout", "Search exceeded its time limit") from error
            completed = True
        finally:
            if not completed and process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        return output.decode("utf-8", errors="replace"), reached_byte_limit, process.returncode

    @staticmethod
    def _environment() -> dict[str, str]:
        """Run ripgrep without inherited credentials or user configuration."""

        return {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"}

    def _parse_matches(
        self,
        output: str,
        *,
        reached_byte_limit: bool,
    ) -> tuple[_SearchMatch, ...]:
        matches: list[_SearchMatch] = []
        output_lines = output.splitlines()
        if reached_byte_limit and output and not output.endswith(("\n", "\r")):
            # The byte cap can stop midway through one JSON event. It cannot
            # represent a complete match, so omit it rather than treating a
            # normal truncated search as a provider/tool failure.
            output_lines = output_lines[:-1]
        for line in output_lines:
            try:
                event = loads(line)
            except JSONDecodeError as error:
                raise ToolError("search_failed", "ripgrep returned an invalid result") from error
            match = self._match_from_event(event)
            if match is not None:
                matches.append(match)
        return tuple(matches)

    @staticmethod
    def _match_from_event(event: object) -> _SearchMatch | None:
        if not isinstance(event, dict):
            raise ToolError("search_failed", "ripgrep returned an invalid result")
        event_mapping = cast(dict[str, object], event)
        event_type = event_mapping.get("type")
        if event_type != "match":
            return None
        data = event_mapping.get("data")
        if not isinstance(data, dict):
            raise ToolError("search_failed", "ripgrep returned an invalid result")
        data_mapping = cast(dict[str, object], data)
        relative_path = _json_text_value(data_mapping.get("path"))
        line = _json_text_value(data_mapping.get("lines"))
        line_number = data_mapping.get("line_number")
        submatches = data_mapping.get("submatches")
        if (
            relative_path is None
            or line is None
            or not isinstance(line_number, int)
            or isinstance(line_number, bool)
            or line_number < 1
            or not isinstance(submatches, list)
            or not submatches
        ):
            raise ToolError("search_failed", "ripgrep returned an invalid result")
        first_submatch = cast(list[object], submatches)[0]
        if not isinstance(first_submatch, dict):
            raise ToolError("search_failed", "ripgrep returned an invalid result")
        start = cast(dict[str, object], first_submatch).get("start")
        if not isinstance(start, int) or isinstance(start, bool) or start < 0:
            raise ToolError("search_failed", "ripgrep returned an invalid result")
        return _SearchMatch(
            relative_path=relative_path,
            line_number=line_number,
            column_number=start + 1,
            line=line,
        )

    def _bounded_result(
        self,
        matches: tuple[_SearchMatch, ...],
        *,
        reached_byte_limit: bool,
    ) -> ToolResult:
        selected_matches = matches[: self._max_matches]
        self._reject_private_key_material(selected_matches)
        output_parts: list[str] = []
        output_char_count = 0
        is_truncated = reached_byte_limit or len(matches) > self._max_matches

        for match in selected_matches:
            line = (
                f"{match.relative_path}:{match.line_number}:{match.column_number}:"
                f"{match.line}"
            )
            remaining_chars = self._max_chars - output_char_count
            if len(line) > remaining_chars:
                if remaining_chars > 0:
                    output_parts.append(line[:remaining_chars])
                is_truncated = True
                break
            output_parts.append(line)
            output_char_count += len(line)

        content = "".join(output_parts)
        if not content:
            return ToolResult(content="No matches found.", is_truncated=is_truncated)
        return ToolResult(content=content, is_truncated=is_truncated)

    def _reject_private_key_material(self, matches: tuple[_SearchMatch, ...]) -> None:
        scanned_relative_paths: set[str] = set()
        total_scan_bytes = 0
        for match in matches:
            if match.relative_path in scanned_relative_paths:
                continue
            if len(scanned_relative_paths) >= self._max_scanned_files:
                raise ToolError(
                    "search_scan_limit_exceeded",
                    "Search result file classification exceeded its limit",
                )
            try:
                self._workspace.resolve_existing(match.relative_path)
                with self._workspace.directory_anchor.open_existing_regular_file(
                    match.relative_path
                ) as file_descriptor:
                    size = os.fstat(file_descriptor).st_size
                    if (
                        size > self._max_scan_bytes
                        or total_scan_bytes + size > self._max_total_scan_bytes
                    ):
                        raise ToolError(
                            "search_scan_limit_exceeded",
                            "Search result file classification exceeded its limit",
                        )
                    file_bytes = read_bounded_open_file_bytes(
                        file_descriptor, max_bytes=self._max_scan_bytes
                    )
            except SensitiveWorkspacePathError as error:
                raise ToolError(
                    "sensitive_path", "Sensitive workspace paths cannot be searched"
                ) from error
            except FileScanLimitExceededError as error:
                raise ToolError(
                    "search_scan_limit_exceeded",
                    "Search result file classification exceeded its limit",
                ) from error
            except (OSError, WorkspacePathError, FileNotFoundError) as error:
                raise ToolError(
                    "search_failed", "Search result file could not be classified"
                ) from error
            total_scan_bytes += size
            scanned_relative_paths.add(match.relative_path)
            if contains_private_key_pem(file_bytes):
                raise ToolError(
                    "private_key_material", "Refusing to return private key material"
                )


def _escape_glob_literal(value: str) -> str:
    """Escape a workspace path before placing it in an rg glob exclusion."""

    escaped_characters = (
        f"\\{character}" if character in _GLOB_METACHARACTERS else character
        for character in value
    )
    return "".join(escaped_characters)


def _json_text_value(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    text = cast(dict[str, object], value).get("text")
    return text if isinstance(text, str) else None
