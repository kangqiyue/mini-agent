"""Read-only, bounded artifact retrieval from one injected artifact store."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mini_agent.artifacts import (
    MAX_ARTIFACT_READ_CHAR_COUNT,
    ArtifactNotFoundError,
    ArtifactStore,
    ArtifactStoreCorruptionError,
)
from mini_agent.tools.base import ToolDefinition, ToolError, ToolResult


class ArtifactReadArguments(BaseModel):
    """A bounded character window for one registered current-session artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=8_000, ge=1, le=MAX_ARTIFACT_READ_CHAR_COUNT)


ARTIFACT_READ_DEFINITION = ToolDefinition(
    name="artifact_read",
    description="Read a bounded character range from a current-session artifact.",
    parameters=ArtifactReadArguments.model_json_schema(),
    is_read_only=True,
)


class ArtifactReadTool:
    """Read only artifacts registered by the injected current-session store."""

    def __init__(self, artifacts: ArtifactStore) -> None:
        self._artifacts = artifacts

    @property
    def definition(self) -> ToolDefinition:
        return ARTIFACT_READ_DEFINITION

    def execute(self, arguments_json: str) -> ToolResult:
        try:
            arguments = ArtifactReadArguments.model_validate_json(arguments_json)
            content = self._artifacts.read(
                arguments.artifact_id,
                offset=arguments.offset,
                limit=arguments.limit,
            )
        except ValidationError as error:
            raise ToolError("invalid_arguments", "Invalid artifact_read arguments") from error
        except ValueError as error:
            raise ToolError("invalid_arguments", str(error)) from error
        except ArtifactNotFoundError as error:
            raise ToolError("artifact_not_found", str(error)) from error
        except ArtifactStoreCorruptionError as error:
            raise ToolError("artifact_corrupt", str(error)) from error
        except OSError as error:
            raise ToolError("artifact_read_failed", "Could not read artifact") from error

        if not content:
            return ToolResult(content="No artifact content in this range.")
        record = next(
            record
            for record in self._artifacts.records
            if record.artifact_id == arguments.artifact_id
        )
        return ToolResult(
            content=content,
            is_truncated=arguments.offset + len(content) < record.char_count,
        )
