"""Canonical messages shared by the agent loop and provider adapters."""

import json
from enum import StrEnum
from typing import Self, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mini_agent.redaction import redact_text
from mini_agent.tools.base import MAX_TOOL_DEFINITIONS, ToolDefinition

MAX_TOOL_ARGUMENT_BYTES = 256 * 1024
MAX_TOOL_ARGUMENT_DEPTH = 16
MAX_TOOL_ARGUMENT_ELEMENTS = 4_096
# Kept distinct from the registry limit: one bound constrains advertised tools,
# while this one constrains provider-controlled execution requests.
MAX_TOOL_CALLS_PER_RESPONSE = 32


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    OTHER = "other"


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    arguments_json: str = Field(min_length=2)

    @field_validator("id")
    @classmethod
    def id_must_not_be_credential_shaped(cls, value: str) -> str:
        return validate_tool_call_id(value)

    @field_validator("name")
    @classmethod
    def name_must_not_be_credential_shaped(cls, value: str) -> str:
        return validate_tool_name(value)

    @field_validator("arguments_json")
    @classmethod
    def arguments_must_be_a_json_object(cls, value: str) -> str:
        try:
            encoded_value = value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("Tool arguments must be valid UTF-8 text") from error

        if len(encoded_value) > MAX_TOOL_ARGUMENT_BYTES:
            raise ValueError("Tool arguments exceed the UTF-8 byte limit")

        try:
            parsed_arguments: object = json.loads(value, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError, RecursionError) as error:
            raise ValueError("Tool arguments must be valid JSON") from error
        if not isinstance(parsed_arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")
        _validate_tool_argument_shape(cast(dict[str, object], parsed_arguments))
        return value


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: MessageRole
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    @field_validator("tool_call_id")
    @classmethod
    def tool_result_id_must_not_be_credential_shaped(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_tool_call_id(value)

    @model_validator(mode="after")
    def validate_role_shape(self) -> Self:
        has_content = self.content is not None and bool(self.content.strip())

        if self.role is MessageRole.ASSISTANT:
            if not has_content and not self.tool_calls:
                raise ValueError("Assistant message requires content or tool calls")
            if self.tool_call_id is not None:
                raise ValueError("Assistant message cannot reference a tool call")
            _require_unique_tool_call_ids(self.tool_calls)
            return self

        if self.role is MessageRole.TOOL:
            if not has_content or self.tool_call_id is None:
                raise ValueError("Tool message requires content and tool_call_id")
            if self.tool_calls:
                raise ValueError("Tool message cannot request tool calls")
            return self

        if not has_content:
            raise ValueError(f"{self.role.value} message requires content")
        if self.tool_calls or self.tool_call_id is not None:
            raise ValueError(f"{self.role.value} message cannot contain tool call fields")
        return self


class ModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    messages: tuple[ConversationMessage, ...] = Field(min_length=1)
    tools: tuple[ToolDefinition, ...] = Field(max_length=MAX_TOOL_DEFINITIONS, default=())
    max_output_tokens: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_tool_call_pairs(self) -> Self:
        pending_tool_call_ids: set[str] = set()
        seen_tool_call_ids: set[str] = set()

        for message in self.messages:
            if message.role is MessageRole.ASSISTANT:
                if pending_tool_call_ids:
                    raise ValueError("Assistant message appeared before pending tool results")
                new_ids = {tool_call.id for tool_call in message.tool_calls}
                if len(new_ids) != len(message.tool_calls):
                    raise ValueError("Assistant message has duplicate tool call ids")
                if new_ids & seen_tool_call_ids:
                    raise ValueError("Tool call id was reused")
                seen_tool_call_ids.update(new_ids)
                pending_tool_call_ids.update(new_ids)
                continue

            if message.role is MessageRole.TOOL:
                tool_call_id = message.tool_call_id
                if tool_call_id not in pending_tool_call_ids:
                    raise ValueError("Tool result does not match a pending tool call")
                pending_tool_call_ids.remove(tool_call_id)
                continue

            if pending_tool_call_ids:
                raise ValueError("Non-tool message appeared before pending tool results")

        if pending_tool_call_ids:
            raise ValueError("Model request contains unresolved tool calls")
        return self


class ModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = Field(max_length=MAX_TOOL_CALLS_PER_RESPONSE, default=())
    finish_reason: FinishReason = FinishReason.STOP

    @model_validator(mode="after")
    def require_content_or_tool_calls(self) -> Self:
        has_content = self.content is not None and bool(self.content.strip())
        if not has_content and not self.tool_calls:
            raise ValueError("Model response requires content or tool calls")
        _require_unique_tool_call_ids(self.tool_calls)
        if self.tool_calls and self.finish_reason is not FinishReason.TOOL_CALLS:
            raise ValueError("Model response with tool calls requires tool_calls finish reason")
        if self.finish_reason is FinishReason.TOOL_CALLS and not self.tool_calls:
            raise ValueError("tool_calls finish reason requires at least one tool call")
        return self


def _require_unique_tool_call_ids(tool_calls: tuple[ToolCall, ...]) -> None:
    tool_call_ids = {tool_call.id for tool_call in tool_calls}
    if len(tool_call_ids) != len(tool_calls):
        raise ValueError("Tool call ids must be unique")


def validate_tool_call_id(value: str) -> str:
    """Reject provider identifiers that would leak through event correlation."""

    _require_non_secret_identifier(value, field_name="Tool call id")
    return value


def validate_tool_name(value: str) -> str:
    """Reject provider tool names that look like credentials."""

    _require_non_secret_identifier(value, field_name="Tool name")
    return value


def _require_non_secret_identifier(value: str, *, field_name: str) -> None:
    if redact_text(value).match_count > 0:
        raise ValueError(f"{field_name} cannot be credential-shaped")


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def _validate_tool_argument_shape(arguments: dict[str, object]) -> None:
    """Bound provider-controlled JSON complexity without recursive traversal."""
    element_count = 0
    pending: list[tuple[object, int]] = [(arguments, 1)]

    while pending:
        value, depth = pending.pop()
        if depth > MAX_TOOL_ARGUMENT_DEPTH:
            raise ValueError("Tool arguments exceed the JSON nesting limit")

        if isinstance(value, dict):
            object_value = cast(dict[str, object], value)
            element_count += len(object_value)
            pending.extend((item, depth + 1) for item in object_value.values())
        elif isinstance(value, list):
            object_values = cast(list[object], value)
            element_count += len(object_values)
            pending.extend((item, depth + 1) for item in object_values)

        if element_count > MAX_TOOL_ARGUMENT_ELEMENTS:
            raise ValueError("Tool arguments exceed the JSON element limit")
