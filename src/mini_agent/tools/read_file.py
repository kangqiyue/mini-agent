"""Bounded UTF-8 file reading inside a workspace."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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

_MAX_START_LINE = 100_000
_DEFAULT_MAX_SCAN_BYTES = 4_000_000

class _PrivateKeyMaterialError(RuntimeError):
    pass


class ReadFileArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    start_line: int = Field(default=1, ge=1, le=_MAX_START_LINE)
    end_line: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def end_line_follows_start_line(self) -> Self:
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        return self


class ReadFileTool:
    def __init__(
        self,
        workspace: Workspace,
        *,
        max_lines: int = 2_000,
        max_chars: int = 256_000,
        max_scan_bytes: int = _DEFAULT_MAX_SCAN_BYTES,
    ) -> None:
        if max_lines < 1 or max_chars < 1 or max_scan_bytes < 1:
            raise ValueError("Read limits must be positive")
        self._workspace = workspace
        self._max_lines = max_lines
        self._max_chars = max_chars
        self._max_scan_bytes = max_scan_bytes

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description=(
                "Read a workspace UTF-8 text range. Full output is bounded by "
                "hard line and character limits."
            ),
            parameters=ReadFileArguments.model_json_schema(),
            is_read_only=True,
        )

    def execute(self, arguments_json: str) -> ToolResult:
        try:
            arguments = ReadFileArguments.model_validate_json(arguments_json)
            self._workspace.resolve_existing(arguments.path)
        except ValidationError as error:
            raise ToolError("invalid_arguments", "Invalid read_file arguments") from error
        except SensitiveWorkspacePathError as error:
            raise ToolError(
                "sensitive_path", "Sensitive workspace paths cannot be read"
            ) from error
        except (WorkspacePathError, FileNotFoundError) as error:
            raise ToolError("invalid_path", str(error)) from error

        try:
            return self._read_lines(arguments.path, arguments)
        except FileScanLimitExceededError as error:
            raise ToolError("read_limit_exceeded", str(error)) from error
        except _PrivateKeyMaterialError as error:
            raise ToolError("private_key_material", str(error)) from error
        except UnicodeDecodeError as error:
            raise ToolError(
                "invalid_encoding", f"File is not valid UTF-8: {arguments.path}"
            ) from error
        except OSError as error:
            raise ToolError("read_failed", f"Could not read file: {arguments.path}") from error

    def _read_lines(self, relative_path: str, arguments: ReadFileArguments) -> ToolResult:
        # Read the entire bounded file before rendering a requested range. The
        # previous streaming renderer could stop at ``end_line`` and therefore
        # never see a private-key boundary earlier or later in the file.
        with self._workspace.directory_anchor.open_existing_regular_file(
            relative_path
        ) as file_descriptor:
            file_bytes = read_bounded_open_file_bytes(
                file_descriptor, max_bytes=self._max_scan_bytes
            )

        # Keep the existing invalid-UTF-8 semantics: an invalid text file is
        # rejected as an encoding error before its content is classified.
        text = file_bytes.decode("utf-8", errors="strict")
        if contains_private_key_pem(file_bytes):
            raise _PrivateKeyMaterialError("Refusing to read private key material")

        renderer = _BoundedLineRenderer(
            start_line=arguments.start_line,
            end_line=arguments.end_line,
            max_lines=self._max_lines,
            max_chars=self._max_chars,
        )
        renderer.consume(text)
        renderer.finish()

        return renderer.result()


class _BoundedLineRenderer:
    """Render decoded text without retaining an unbounded physical line."""

    def __init__(
        self,
        *,
        start_line: int,
        end_line: int | None,
        max_lines: int,
        max_chars: int,
    ) -> None:
        self._start_line = start_line
        self._end_line = end_line
        self._max_lines = max_lines
        self._max_chars = max_chars
        self._output_parts: list[str] = []
        self._output_char_count = 0
        self._selected_line_count = 0
        self._line_number = 1
        self._line_started = False
        self._line_has_input = False
        self._pending_carriage_returns = 0
        self._is_truncated = False
        self._is_finished = False

    @property
    def is_finished(self) -> bool:
        return self._is_finished

    def consume(self, text: str) -> None:
        for character in text:
            if self._is_finished:
                return
            self._line_has_input = True
            if self._line_is_selected() and not self._begin_selected_line():
                return

            if character == "\n":
                if self._line_is_selected():
                    self._pending_carriage_returns = 0
                    if not self._append("\n"):
                        return
                    if self._end_line == self._line_number:
                        self._is_finished = True
                        return
                self._next_line()
                continue

            if not self._line_is_selected():
                continue
            if character == "\r":
                self._pending_carriage_returns += 1
                continue
            if not self._flush_pending_carriage_returns():
                return
            if not self._append(character):
                return

    def finish(self) -> None:
        if self._is_finished or not self._line_has_input or not self._line_is_selected():
            return
        self._pending_carriage_returns = 0
        if self._begin_selected_line():
            self._append("\n")

    def result(self) -> ToolResult:
        content = "".join(self._output_parts)
        if not content:
            content = f"No lines found at or after line {self._start_line}."
        return ToolResult(content=content, is_truncated=self._is_truncated)

    def _line_is_selected(self) -> bool:
        return self._line_number >= self._start_line and (
            self._end_line is None or self._line_number <= self._end_line
        )

    def _begin_selected_line(self) -> bool:
        if self._line_started:
            return True
        if self._selected_line_count >= self._max_lines:
            self._is_truncated = True
            self._is_finished = True
            return False
        self._selected_line_count += 1
        self._line_started = True
        return self._append(f"{self._line_number:>6} | ")

    def _flush_pending_carriage_returns(self) -> bool:
        if self._pending_carriage_returns == 0:
            return True
        pending = "\r" * self._pending_carriage_returns
        self._pending_carriage_returns = 0
        return self._append(pending)

    def _append(self, value: str) -> bool:
        remaining_chars = self._max_chars - self._output_char_count
        if len(value) > remaining_chars:
            if remaining_chars > 0:
                self._output_parts.append(value[:remaining_chars])
            self._is_truncated = True
            self._is_finished = True
            return False
        self._output_parts.append(value)
        self._output_char_count += len(value)
        return True

    def _next_line(self) -> None:
        self._line_number += 1
        self._line_started = False
        self._line_has_input = False
        self._pending_carriage_returns = 0
