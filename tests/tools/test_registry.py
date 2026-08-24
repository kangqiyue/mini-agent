from pathlib import Path

import pytest

from mini_agent.tools.base import MAX_TOOL_DEFINITIONS, ToolError
from mini_agent.tools.read_file import ReadFileTool
from mini_agent.tools.registry import ToolRegistry
from mini_agent.workspace import Workspace


def test_registry_exposes_definitions_and_lookup(tmp_path: Path) -> None:
    read_file = ReadFileTool(Workspace(tmp_path))
    registry = ToolRegistry((read_file,))

    assert registry.definitions == (read_file.definition,)
    assert registry.get("read_file") is read_file


def test_registry_rejects_duplicate_names(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)

    with pytest.raises(ValueError, match="duplicate tool names") as error_info:
        ToolRegistry((ReadFileTool(workspace), ReadFileTool(workspace)))

    assert "read_file" not in str(error_info.value)


def test_registry_reports_unknown_tool(tmp_path: Path) -> None:
    registry = ToolRegistry((ReadFileTool(Workspace(tmp_path)),))

    with pytest.raises(ToolError) as error_info:
        registry.get("missing")

    assert error_info.value.code == "unknown_tool"


def test_registry_rejects_excessive_tool_count_without_reading_definitions() -> None:
    tools = tuple(object() for _ in range(MAX_TOOL_DEFINITIONS + 1))

    with pytest.raises(ValueError, match="tool count limit"):
        ToolRegistry(tools)  # type: ignore[arg-type]
