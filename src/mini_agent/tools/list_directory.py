"""Bounded directory listing inside a workspace."""

from __future__ import annotations

import os
import stat
from itertools import islice

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mini_agent.redaction import redact_text
from mini_agent.tools.base import ToolDefinition, ToolError, ToolResult
from mini_agent.workspace import (
    SensitiveWorkspacePathError,
    Workspace,
    WorkspacePathError,
)

# Listing shares the bounded-output discipline of the other read-only tools:
# one hard entry cap regardless of directory size.
MAX_DIRECTORY_ENTRIES = 500


class ListDirectoryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(default=".", min_length=1)


class ListDirectoryTool:
    """List one workspace directory without reading any file content."""

    def __init__(self, workspace: Workspace, *, max_entries: int = MAX_DIRECTORY_ENTRIES) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._workspace = workspace
        self._max_entries = max_entries
        self._directory_anchor = workspace.directory_anchor

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_directory",
            description=(
                "List the entries of one workspace directory (name and type), "
                "bounded to a fixed maximum number of entries."
            ),
            parameters=ListDirectoryArguments.model_json_schema(),
            is_read_only=True,
        )

    def execute(self, arguments_json: str) -> ToolResult:
        try:
            arguments = ListDirectoryArguments.model_validate_json(arguments_json)
            self._workspace.resolve_existing(arguments.path, allow_directory=True)
        except ValidationError as error:
            raise ToolError("invalid_arguments", "Invalid list_directory arguments") from error
        except SensitiveWorkspacePathError as error:
            raise ToolError(
                "sensitive_path", "Sensitive workspace paths cannot be read"
            ) from error
        except (WorkspacePathError, FileNotFoundError) as error:
            raise ToolError("invalid_path", str(error)) from error

        try:
            entries, is_truncated = self._list_entries(arguments.path)
        except OSError as error:
            raise ToolError(
                "read_failed", f"Could not list directory: {arguments.path}"
            ) from error

        return self._render(entries, is_truncated=is_truncated)

    def _list_entries(self, relative_path: str) -> tuple[list[tuple[str, str]], bool]:
        """Return sorted (name, type) pairs through a held directory descriptor.

        Reading entries by descriptor keeps the boundary consistent with the
        other anchored tools: no pathname re-resolution.
        """

        with self._directory_anchor.open_existing_directory(
            relative_path
        ) as directory_descriptor:
            with os.scandir(directory_descriptor) as entries:
                names = [entry.name for entry in islice(entries, self._max_entries)]
                is_truncated = next(entries, None) is not None
            visible = [
                (name, _entry_type(directory_descriptor, name)) for name in sorted(names)
            ]
            return visible, is_truncated

    def _render(self, entries: list[tuple[str, str]], *, is_truncated: bool) -> ToolResult:
        if not entries:
            return ToolResult(content="(empty directory)", is_truncated=False)
        lines = [
            f"{entry_type} {redact_text(name).text}" for name, entry_type in entries
        ]
        content = "\n".join(lines)
        if is_truncated:
            content += (
                f"\n\n[list truncated at {self._max_entries} entries; "
                "more entries not shown]"
            )
        return ToolResult(content=content, is_truncated=is_truncated)


def _entry_type(directory_descriptor: int, name: str) -> str:
    """Classify one entry without following symlinks; '?' when unavailable."""

    try:
        mode = os.stat(
            name, dir_fd=directory_descriptor, follow_symlinks=False
        ).st_mode
    except OSError:
        return "?"
    if stat.S_ISDIR(mode):
        return "d"
    if stat.S_ISREG(mode):
        return "f"
    if stat.S_ISLNK(mode):
        return "l"
    return "o"
