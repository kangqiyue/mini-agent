"""Name-based lookup for a small explicit set of local tools."""

from collections.abc import Sequence

from mini_agent.tools.base import (
    MAX_TOOL_DEFINITIONS,
    LocalTool,
    ToolDefinition,
    ToolError,
)


class ToolRegistry:
    def __init__(self, tools: Sequence[LocalTool]) -> None:
        if len(tools) > MAX_TOOL_DEFINITIONS:
            raise ValueError("Tool registry exceeds the tool count limit")
        tools_by_name: dict[str, LocalTool] = {}
        for tool in tools:
            name = tool.definition.name
            if name in tools_by_name:
                raise ValueError("Tool registry contains duplicate tool names")
            tools_by_name[name] = tool
        self._tools_by_name = tools_by_name

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool.definition for tool in self._tools_by_name.values())

    def get(self, name: str) -> LocalTool:
        tool = self._tools_by_name.get(name)
        if tool is None:
            raise ToolError("unknown_tool", f"Unknown tool: {name}")
        return tool
