"""Typed mechanical facts produced by completed local tools."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ToolCompletionFacts(BaseModel):
    """Stable completion facts kept separate from human-facing tool output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    command_exit_code: int | None = None
    modified_paths: tuple[str, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def require_valid_modified_paths(self) -> Self:
        if any(not path or len(path) > 1_024 for path in self.modified_paths):
            raise ValueError("Modified paths must be non-empty and bounded")
        if len(set(self.modified_paths)) != len(self.modified_paths):
            raise ValueError("Modified paths must be unique")
        return self
