"""Read-only, bounded search over one injected session-history snapshot."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from mini_agent.history import HistoryEventKind, SessionHistory
from mini_agent.tools.base import ToolDefinition, ToolError, ToolResult


class HistorySearchArguments(BaseModel):
    """Arguments for searching the current session only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, max_length=512)
    limit: int = Field(default=10, ge=1, le=50)
    kinds: tuple[HistoryEventKind, ...] | None = None
    before_event_id: int | None = Field(default=None, ge=1)
    after_event_id: int | None = Field(default=None, ge=1)
    snippet_chars: int = Field(default=500, ge=1, le=1_000)

    @model_validator(mode="after")
    def validate_event_bounds(self) -> Self:
        if (
            self.before_event_id is not None
            and self.after_event_id is not None
            and self.after_event_id >= self.before_event_id
        ):
            raise ValueError("after_event_id must be less than before_event_id")
        return self


class HistorySearchTool:
    """Expose only a bounded, preselected session snapshot to the model."""

    def __init__(self, history: Callable[[], SessionHistory]) -> None:
        self._history = history

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="history_search",
            description="Search bounded, persisted records from the current session.",
            parameters=HistorySearchArguments.model_json_schema(),
            is_read_only=True,
        )

    def execute(self, arguments_json: str) -> ToolResult:
        return self._execute(arguments_json, history=self._history())

    def execute_with_transcript_bound(
        self,
        arguments_json: str,
        *,
        before_event_id: int,
    ) -> ToolResult:
        return self._execute(
            arguments_json,
            history=self._history().bounded_before(before_event_id),
        )

    @staticmethod
    def _execute(arguments_json: str, *, history: SessionHistory) -> ToolResult:
        try:
            arguments = HistorySearchArguments.model_validate_json(arguments_json)
            page = history.search(
                arguments.query,
                limit=arguments.limit,
                kinds=arguments.kinds,
                before_event_id=arguments.before_event_id,
                after_event_id=arguments.after_event_id,
                snippet_chars=arguments.snippet_chars,
            )
        except ValidationError as error:
            raise ToolError("invalid_arguments", "Invalid history_search arguments") from error
        except ValueError as error:
            raise ToolError("invalid_arguments", str(error)) from error

        return ToolResult(content=page.model_dump_json(), is_truncated=page.is_truncated)
