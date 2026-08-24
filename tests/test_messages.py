import pytest
from pydantic import ValidationError

import mini_agent.messages as messages
from mini_agent.messages import (
    MAX_TOOL_ARGUMENT_BYTES,
    MAX_TOOL_ARGUMENT_DEPTH,
    MAX_TOOL_ARGUMENT_ELEMENTS,
    MAX_TOOL_CALLS_PER_RESPONSE,
    ConversationMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from mini_agent.tools.base import MAX_TOOL_DEFINITIONS, ToolDefinition


def _nested_object_json(depth: int) -> str:
    value = "1"
    for _ in range(depth):
        value = '{"value":' + value + "}"
    return value


def test_tool_call_requires_json_object_arguments() -> None:
    with pytest.raises(ValidationError, match="JSON object"):
        ToolCall(id="call-1", name="read_file", arguments_json='["README.md"]')


@pytest.mark.parametrize(
    ("arguments_json", "error_message"),
    [
        ('{"score":NaN}', "valid JSON"),
        ("{\"value\":\"" + "x" * MAX_TOOL_ARGUMENT_BYTES + "\"}", "UTF-8 byte"),
        (_nested_object_json(MAX_TOOL_ARGUMENT_DEPTH + 1), "nesting"),
        (
            '{"values":[' + ",".join("0" for _ in range(MAX_TOOL_ARGUMENT_ELEMENTS)) + ",0]}",
            "element",
        ),
    ],
)
def test_tool_call_rejects_unbounded_or_non_standard_arguments(
    arguments_json: str,
    error_message: str,
) -> None:
    with pytest.raises(ValidationError, match=error_message):
        ToolCall(id="call-1", name="read_file", arguments_json=arguments_json)


def test_tool_call_converts_json_recursion_error_to_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_recursion_error(*args: object, **kwargs: object) -> object:
        raise RecursionError("provider JSON was too deep")

    monkeypatch.setattr(messages.json, "loads", raise_recursion_error)

    with pytest.raises(ValidationError, match="valid JSON"):
        ToolCall(id="call-1", name="read_file", arguments_json='{"path":"README.md"}')


def test_tool_call_accepts_bounded_nested_arguments() -> None:
    tool_call = ToolCall(
        id="call-1",
        name="read_file",
        arguments_json='{"path":"README.md","options":{"lines":[1,2,3]}}',
    )

    assert tool_call.arguments_json.endswith("}}")


@pytest.mark.parametrize(
    ("field_name", "credential_shaped_value"),
    (
        ("id", "sk_live_" + "0" * 24),
        ("id", "AKIA" + "0" * 16),
        ("name", "sk_live_" + "0" * 24),
        ("name", "AKIA" + "0" * 16),
    ),
)
def test_tool_call_rejects_credential_shaped_identifier_fields(
    field_name: str,
    credential_shaped_value: str,
) -> None:
    values = {
        "id": "call-1",
        "name": "read_file",
        "arguments_json": "{}",
    }
    values[field_name] = credential_shaped_value

    with pytest.raises(ValidationError, match="credential-shaped"):
        ToolCall(**values)


def test_tool_call_keeps_ordinary_identifiers_and_tool_result_references() -> None:
    tool_call = ToolCall(id="call_2026-08-12", name="read_file_v2", arguments_json="{}")
    tool_result = ConversationMessage(
        role=MessageRole.TOOL,
        content="ok",
        tool_call_id=tool_call.id,
    )

    assert tool_call.id == "call_2026-08-12"
    assert tool_result.tool_call_id == tool_call.id


def test_tool_result_rejects_credential_shaped_reference() -> None:
    with pytest.raises(ValidationError, match="credential-shaped"):
        ConversationMessage(
            role=MessageRole.TOOL,
            content="ok",
            tool_call_id="sk_live_" + "0" * 24,
        )


def test_assistant_message_can_contain_only_tool_calls() -> None:
    tool_call = ToolCall(id="call-1", name="read_file", arguments_json='{"path":"README.md"}')

    message = ConversationMessage(role=MessageRole.ASSISTANT, tool_calls=(tool_call,))

    assert message.content is None
    assert message.tool_calls == (tool_call,)


def test_tool_message_requires_tool_call_id() -> None:
    with pytest.raises(ValidationError, match="tool_call_id"):
        ConversationMessage(role=MessageRole.TOOL, content="result")


def test_model_response_requires_matching_tool_call_finish_reason() -> None:
    tool_call = ToolCall(id="call-1", name="read_file", arguments_json='{"path":"README.md"}')

    with pytest.raises(ValidationError, match="finish reason"):
        ModelResponse(content=None, tool_calls=(tool_call,), finish_reason=FinishReason.STOP)


def test_model_request_rejects_an_orphan_tool_result() -> None:
    with pytest.raises(ValidationError, match="pending tool call"):
        ModelRequest(
            model="m",
            messages=(
                ConversationMessage(
                    role=MessageRole.TOOL,
                    content="result",
                    tool_call_id="call-1",
                ),
            ),
        )


def test_model_request_accepts_a_complete_tool_pair() -> None:
    tool_call = ToolCall(id="call-1", name="read_file", arguments_json='{"path":"README.md"}')

    request = ModelRequest(
        model="m",
        messages=(
            ConversationMessage(role=MessageRole.USER, content="read it"),
            ConversationMessage(role=MessageRole.ASSISTANT, tool_calls=(tool_call,)),
            ConversationMessage(
                role=MessageRole.TOOL,
                content="result",
                tool_call_id=tool_call.id,
            ),
        ),
    )

    assert request.messages[-1].tool_call_id == tool_call.id


def test_model_response_rejects_duplicate_tool_call_ids() -> None:
    first = ToolCall(id="call-1", name="read_file", arguments_json='{"path":"a"}')
    duplicate = ToolCall(id="call-1", name="read_file", arguments_json='{"path":"b"}')

    with pytest.raises(ValidationError, match="unique"):
        ModelResponse(
            tool_calls=(first, duplicate),
            finish_reason=FinishReason.TOOL_CALLS,
        )


@pytest.mark.parametrize(
    "tool_call_count",
    (MAX_TOOL_CALLS_PER_RESPONSE - 1, MAX_TOOL_CALLS_PER_RESPONSE),
)
def test_model_response_accepts_bounded_tool_calls(tool_call_count: int) -> None:
    response = ModelResponse(
        tool_calls=tuple(
            ToolCall(id=f"call-{index}", name="read_file", arguments_json="{}")
            for index in range(tool_call_count)
        ),
        finish_reason=FinishReason.TOOL_CALLS,
    )

    assert len(response.tool_calls) == tool_call_count


def test_model_response_rejects_too_many_tool_calls() -> None:
    with pytest.raises(ValidationError, match="at most"):
        ModelResponse(
            tool_calls=tuple(
                ToolCall(id=f"call-{index}", name="read_file", arguments_json="{}")
                for index in range(MAX_TOOL_CALLS_PER_RESPONSE + 1)
            ),
            finish_reason=FinishReason.TOOL_CALLS,
        )


def test_model_request_rejects_excessive_tool_count() -> None:
    tool = ToolDefinition(
        name="read_file",
        description="Read one file.",
        parameters={"type": "object"},
        is_read_only=True,
    )

    with pytest.raises(ValidationError, match="at most"):
        ModelRequest(
            model="m",
            messages=(ConversationMessage(role=MessageRole.USER, content="hello"),),
            tools=(tool,) * (MAX_TOOL_DEFINITIONS + 1),
        )
