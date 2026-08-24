"""Read-only, bounded range reads over one injected session-history snapshot."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from mini_agent.history import SessionHistory
from mini_agent.tools.base import ToolDefinition, ToolError, ToolResult


class HistoryReadArguments(BaseModel):
    """Arguments for a bounded inclusive range in the current session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    from_event_id: int = Field(ge=1)
    to_event_id: int = Field(ge=1)
    max_events: int = Field(default=20, ge=1, le=100)
    max_chars: int = Field(default=8_000, ge=1, le=20_000)
    start_offset: int = Field(
        default=0,
        ge=0,
        description="Characters to skip in from_event_id before reading.",
    )

    @model_validator(mode="after")
    def validate_event_range(self) -> Self:
        if self.from_event_id > self.to_event_id:
            raise ValueError("from_event_id must be less than or equal to to_event_id")
        return self


class HistoryReadTool:
    """Expose only an inclusive, bounded range from the current session."""

    def __init__(self, history: Callable[[], SessionHistory]) -> None:
        self._history = history

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="history_read",
            description=(
                "Read a bounded event range from the current session. "
                "To continue a partial event, reuse next_cursor.event_id and "
                "next_cursor.char_offset as from_event_id and start_offset."
            ),
            parameters=HistoryReadArguments.model_json_schema(),
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
            arguments = HistoryReadArguments.model_validate_json(arguments_json)
            page = history.read(
                from_event_id=arguments.from_event_id,
                to_event_id=arguments.to_event_id,
                max_events=arguments.max_events,
                max_chars=arguments.max_chars,
                start_offset=arguments.start_offset,
            )
        except ValidationError as error:
            raise ToolError("invalid_arguments", "Invalid history_read arguments") from error
        except ValueError as error:
            raise ToolError("invalid_arguments", str(error)) from error

        return ToolResult(content=page.model_dump_json(), is_truncated=page.is_truncated)
