"""Typed local tool protocol and execution results."""

from __future__ import annotations

import json
import math
from typing import Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mini_agent.redaction import is_sensitive_key, redact_text
from mini_agent.redaction_types import RedactionSummary
from mini_agent.tool_facts import ToolCompletionFacts

TOOL_OUTPUT_TRUNCATED_NOTICE = "[tool output was truncated at hard limit]"
TOOL_CONTEXT_PREVIEW_TRUNCATED_NOTICE = (
    "[tool output preview was truncated for active context; no artifact was saved]"
)

# Tool definitions are supplied by local extensions but are copied verbatim to
# model providers.  Bound them at this publication boundary so a malformed or
# secret-bearing extension cannot turn into an oversized provider request or a
# durable disclosure.
MAX_TOOL_DEFINITIONS = 32
MAX_TOOL_NAME_UTF8_BYTES = 128
MAX_TOOL_DESCRIPTION_UTF8_BYTES = 16 * 1024
MAX_TOOL_SCHEMA_DEPTH = 16
MAX_TOOL_SCHEMA_NODES = 4_096
MAX_TOOL_SCHEMA_KEY_UTF8_BYTES = 512
MAX_TOOL_SCHEMA_UTF8_BYTES = 64 * 1024

_TOOL_DEFINITION_CREDENTIAL_ERROR = (
    "Tool definition cannot contain credential-shaped text"
)
_TOOL_DEFINITION_SCHEMA_ERROR = "Tool definition parameters must be JSON-compatible"


class ToolDefinition(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        hide_input_in_errors=True,
    )

    name: str = Field(
        min_length=1,
        max_length=MAX_TOOL_NAME_UTF8_BYTES,
        pattern=r"^[A-Za-z][A-Za-z0-9_-]*$",
    )
    description: str = Field(min_length=1)
    parameters: dict[str, object]
    is_read_only: bool

    @field_validator("name")
    @classmethod
    def name_must_not_contain_credentials(cls, value: str) -> str:
        _require_safe_tool_definition_field_name(value)
        return value

    @field_validator("description")
    @classmethod
    def description_must_be_bounded_and_safe(cls, value: str) -> str:
        if _utf8_byte_count(value, _TOOL_DEFINITION_SCHEMA_ERROR) > (
            MAX_TOOL_DESCRIPTION_UTF8_BYTES
        ):
            raise ValueError("Tool definition description exceeds the UTF-8 byte limit")
        _require_safe_tool_definition_text(value)
        return value

    @field_validator("parameters", mode="before")
    @classmethod
    def parameters_must_be_bounded_safe_json_object(cls, value: object) -> object:
        _validate_tool_schema(value)
        return value


def _validate_tool_schema(value: object) -> None:
    """Validate a provider-visible JSON schema without recursive traversal."""

    if not isinstance(value, dict):
        raise ValueError("Tool definition parameters must be a JSON object")

    node_count = 0
    pending: list[tuple[object, int]] = [(value, 1)]
    while pending:
        current, depth = pending.pop()
        node_count += 1
        if node_count > MAX_TOOL_SCHEMA_NODES:
            raise ValueError("Tool definition parameters exceed the JSON node limit")
        if depth > MAX_TOOL_SCHEMA_DEPTH:
            raise ValueError("Tool definition parameters exceed the JSON nesting limit")

        if isinstance(current, dict):
            mapping = cast(dict[object, object], current)
            for key, item in mapping.items():
                if not isinstance(key, str):
                    raise ValueError(_TOOL_DEFINITION_SCHEMA_ERROR)
                if _utf8_byte_count(key, _TOOL_DEFINITION_SCHEMA_ERROR) > (
                    MAX_TOOL_SCHEMA_KEY_UTF8_BYTES
                ):
                    raise ValueError("Tool definition schema key exceeds the UTF-8 byte limit")
                _require_safe_tool_definition_field_name(key)
                pending.append((item, depth + 1))
            continue

        if isinstance(current, list):
            items = cast(list[object], current)
            pending.extend((item, depth + 1) for item in items)
            continue

        if current is None or isinstance(current, bool):
            continue
        if isinstance(current, int):
            continue
        if isinstance(current, float):
            if math.isfinite(current):
                continue
            raise ValueError(_TOOL_DEFINITION_SCHEMA_ERROR)
        if isinstance(current, str):
            _utf8_byte_count(current, _TOOL_DEFINITION_SCHEMA_ERROR)
            _require_safe_tool_definition_text(current)
            continue
        raise ValueError(_TOOL_DEFINITION_SCHEMA_ERROR)

    try:
        encoded_schema = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError) as error:
        raise ValueError(_TOOL_DEFINITION_SCHEMA_ERROR) from error
    if len(encoded_schema) > MAX_TOOL_SCHEMA_UTF8_BYTES:
        raise ValueError("Tool definition parameters exceed the UTF-8 byte limit")


def _utf8_byte_count(value: str, error_message: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ValueError(error_message) from error


def _require_safe_tool_definition_text(value: str) -> None:
    if redact_text(value).match_count > 0:
        raise ValueError(_TOOL_DEFINITION_CREDENTIAL_ERROR)


def _require_safe_tool_definition_field_name(value: str) -> None:
    if is_sensitive_key(value) or redact_text(value).match_count > 0:
        raise ValueError(_TOOL_DEFINITION_CREDENTIAL_ERROR)


class ToolResult(BaseModel):
    """Complete bounded tool output, with a derived model-context preview.

    ``content`` is the tool's full result up to its own hard output limit.  It
    is deliberately distinct from the shorter representation placed in model
    context: the agent can persist this complete value as an artifact first.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str = Field(
        min_length=1,
        description="Complete output up to this tool's hard output limit.",
    )
    is_truncated: bool = Field(
        default=False,
        description="Whether the tool's hard output limit omitted remaining output.",
    )
    source_redaction_summary: RedactionSummary = Field(default_factory=RedactionSummary)
    facts: ToolCompletionFacts = Field(default_factory=ToolCompletionFacts)

    def render_for_context(self, content: str | None = None) -> str:
        """Render content while preserving the tool's completeness state."""
        rendered_content = self.content if content is None else content
        if not self.is_truncated:
            return rendered_content
        return f"{rendered_content}\n\n{TOOL_OUTPUT_TRUNCATED_NOTICE}"

    def preview_for_context(self, maximum_chars: int) -> str:
        """Return a bounded body preview with explicit active-context omission."""
        if maximum_chars < 1:
            raise ValueError("Context preview limit must be positive")
        preview = self.content[:maximum_chars]
        if len(self.content) > maximum_chars:
            preview = f"{preview}\n\n{TOOL_CONTEXT_PREVIEW_TRUNCATED_NOTICE}"
        return self.render_for_context(preview)


class ToolError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class LocalTool(Protocol):
    @property
    def definition(self) -> ToolDefinition: ...

    def execute(self, arguments_json: str) -> ToolResult: ...


@runtime_checkable
class TranscriptBoundTool(LocalTool, Protocol):
    """A retrieval tool that can be limited to a prior transcript boundary.

    ``before_event_id`` is exclusive. The agent supplies the id of the durable
    assistant event that requested the tool, so a retrieval call cannot inspect
    its own arguments or any event that was created after it.
    """

    def execute_with_transcript_bound(
        self,
        arguments_json: str,
        *,
        before_event_id: int,
    ) -> ToolResult: ...
