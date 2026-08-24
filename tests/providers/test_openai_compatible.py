import json

import httpx
import pytest
from pydantic import HttpUrl

from mini_agent.config import ModelConfig
from mini_agent.messages import (
    MAX_TOOL_ARGUMENT_DEPTH,
    MAX_TOOL_CALLS_PER_RESPONSE,
    ConversationMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ToolCall,
)
from mini_agent.provider import ProviderError
from mini_agent.providers.openai_compatible import OpenAICompatibleProvider
from mini_agent.tools import ToolDefinition


def _config() -> ModelConfig:
    return ModelConfig(
        model="test-model",
        base_url=HttpUrl("https://provider.example/v1"),
        api_key_env="TEST_MINI_AGENT_KEY",
    )


def _request() -> ModelRequest:
    return ModelRequest(
        model="test-model",
        messages=(ConversationMessage(role=MessageRole.USER, content="hello"),),
    )


def _assert_exception_chain_is_safe(error: BaseException, secret: str) -> None:
    pending = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        assert not isinstance(current, httpx.Request)
        assert secret not in str(current)
        assert secret not in repr(current)
        pending.extend(
            linked_error
            for linked_error in (current.__cause__, current.__context__)
            if linked_error is not None
        )


def test_provider_exposes_configured_capabilities() -> None:
    provider = OpenAICompatibleProvider(_config())

    assert provider.capabilities.context_window == 128_000
    assert provider.capabilities.max_output_tokens == 16_384


def _nested_object_json(depth: int) -> str:
    value = "1"
    for _ in range(depth):
        value = '{"value":' + value + "}"
    return value


@pytest.mark.asyncio
async def test_complete_translates_successful_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_MINI_AGENT_KEY", "test-value")

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-value"
        assert request.url == "https://provider.example/v1/chat/completions"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "world"}, "finish_reason": "length"}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    provider = OpenAICompatibleProvider(_config(), client=client)

    response = await provider.complete(_request())

    assert response.content == "world"
    assert response.finish_reason is FinishReason.LENGTH
    await client.aclose()


@pytest.mark.asyncio
async def test_complete_preserves_base_url_query_when_appending_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MINI_AGENT_KEY", "test-value")
    config = _config().model_copy(
        update={
            "base_url": HttpUrl(
                "https://provider.example/v1/?api-version=2026-01-01"
            )
        }
    )

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.url.params["api-version"] == "2026-01-01"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    provider = OpenAICompatibleProvider(config, client=client)

    response = await provider.complete(_request())

    assert response.content == "ok"
    await client.aclose()


@pytest.mark.asyncio
async def test_complete_rejects_an_oversized_provider_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MINI_AGENT_KEY", "test-value")

    def respond(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, content=b"x" * 1_048_577)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    provider = OpenAICompatibleProvider(_config(), client=client)

    with pytest.raises(ProviderError) as error_info:
        await provider.complete(_request())

    assert error_info.value.code == "provider_response_too_large"
    assert error_info.value.is_retryable is False
    await client.aclose()


@pytest.mark.asyncio
async def test_complete_honors_a_request_output_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MINI_AGENT_KEY", "test-value")

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["max_tokens"] == 512
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    provider = OpenAICompatibleProvider(_config(), client=client)
    request = _request().model_copy(update={"max_output_tokens": 512})

    response = await provider.complete(request)

    assert response.content == "ok"
    await client.aclose()


@pytest.mark.asyncio
async def test_complete_reports_missing_credential_without_a_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_MINI_AGENT_KEY", raising=False)
    provider = OpenAICompatibleProvider(_config())

    with pytest.raises(ProviderError) as error_info:
        await provider.complete(_request())

    assert error_info.value.code == "missing_credential"
    assert error_info.value.is_retryable is False
    await provider.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_error_type", "expected_code", "expected_message"),
    (
        (httpx.ReadTimeout, "provider_timeout", "Model provider request timed out"),
        (httpx.ConnectError, "provider_transport", "Model provider connection failed"),
    ),
)
async def test_complete_does_not_chain_transport_requests_with_credentials(
    monkeypatch: pytest.MonkeyPatch,
    request_error_type: type[httpx.RequestError],
    expected_code: str,
    expected_message: str,
) -> None:
    secret = "synthetic-provider-credential"
    monkeypatch.setenv("TEST_MINI_AGENT_KEY", secret)

    def fail(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {secret}"
        raise request_error_type("synthetic transport failure", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
    provider = OpenAICompatibleProvider(_config(), client=client)

    with pytest.raises(ProviderError) as error_info:
        await provider.complete(_request())

    error = error_info.value
    assert error.code == expected_code
    assert str(error) == expected_message
    assert error.is_retryable is True
    assert error.__cause__ is None
    assert error.__context__ is None
    _assert_exception_chain_is_safe(error, secret)
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_body", "expected_message"),
    (
        (
            b'{"choices":[{"message":{"content":{"token":"synthetic-provider-secret"}}}]}',
            "Model provider returned an invalid response",
        ),
        (
            b'{"choices":[{"message":{"content":""},"finish_reason":"stop"}]}',
            "Model provider returned invalid message content",
        ),
        (
            (
                b'{"choices":[{"message":{"tool_calls":[{"id":"invalid id",'
                b'"type":"function","function":{"name":"read_file",'
                b'"arguments":"{}"}}]},"finish_reason":"tool_calls"}]}'
            ),
            "Model provider returned invalid message content",
        ),
    ),
)
async def test_complete_does_not_chain_provider_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
    response_body: bytes,
    expected_message: str,
) -> None:
    secret = "synthetic-provider-secret"
    monkeypatch.setenv("TEST_MINI_AGENT_KEY", "synthetic-provider-credential")

    def respond(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, content=response_body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    provider = OpenAICompatibleProvider(_config(), client=client)

    with pytest.raises(ProviderError) as error_info:
        await provider.complete(_request())

    error = error_info.value
    assert error.code == "invalid_provider_response"
    assert str(error) == expected_message
    assert error.is_retryable is False
    assert error.__cause__ is None
    assert error.__context__ is None
    _assert_exception_chain_is_safe(error, secret)
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", (408, 425, 429, 500))
async def test_complete_marks_transient_http_statuses_as_retryable(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    monkeypatch.setenv("TEST_MINI_AGENT_KEY", "test-value")

    def respond(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(status_code, text="body-that-must-not-be-propagated")

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    provider = OpenAICompatibleProvider(_config(), client=client)

    with pytest.raises(ProviderError) as error_info:
        await provider.complete(_request())

    error = error_info.value
    assert error.code == f"provider_http_{status_code}"
    assert error.is_retryable is True
    assert "body-that-must-not-be-propagated" not in str(error)
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", (400, 413))
async def test_complete_classifies_context_overflow_without_exposing_body(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    monkeypatch.setenv("TEST_MINI_AGENT_KEY", "test-value")

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            text="maximum context length exceeded: api_key=synthetic-body-secret",
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    provider = OpenAICompatibleProvider(_config(), client=client)

    with pytest.raises(ProviderError) as error_info:
        await provider.complete(_request())

    assert error_info.value.code == "context_overflow"
    assert error_info.value.is_context_overflow is True
    assert error_info.value.is_retryable is False
    assert "synthetic-body-secret" not in str(error_info.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_complete_translates_native_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_MINI_AGENT_KEY", "test-value")

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    provider = OpenAICompatibleProvider(_config(), client=client)

    response = await provider.complete(_request())

    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].arguments_json == '{"path":"README.md"}'
    await client.aclose()


@pytest.mark.asyncio
async def test_complete_rejects_excessive_raw_tool_calls_before_canonical_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MINI_AGENT_KEY", "test-value")

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": f"call-{index}",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": "{}",
                                    },
                                }
                                for index in range(MAX_TOOL_CALLS_PER_RESPONSE + 1)
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    provider = OpenAICompatibleProvider(_config(), client=client)

    with pytest.raises(ProviderError) as error_info:
        await provider.complete(_request())

    assert error_info.value.code == "invalid_provider_response"
    assert error_info.value.is_retryable is False
    assert "call-32" not in str(error_info.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_complete_translates_invalid_tool_arguments_to_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MINI_AGENT_KEY", "test-value")
    nested_arguments = _nested_object_json(MAX_TOOL_ARGUMENT_DEPTH + 1)

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": nested_arguments,
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    provider = OpenAICompatibleProvider(_config(), client=client)

    with pytest.raises(ProviderError) as error_info:
        await provider.complete(_request())

    assert error_info.value.code == "invalid_provider_response"
    assert error_info.value.is_retryable is False
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_id", "tool_name"),
    (
        ("sk_live_" + "0" * 24, "read_file"),
        ("call-1", "AKIA" + "0" * 16),
    ),
)
async def test_complete_rejects_credential_shaped_tool_identifiers(
    monkeypatch: pytest.MonkeyPatch,
    tool_id: str,
    tool_name: str,
) -> None:
    monkeypatch.setenv("TEST_MINI_AGENT_KEY", "test-value")

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": tool_id,
                                    "type": "function",
                                    "function": {"name": tool_name, "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    provider = OpenAICompatibleProvider(_config(), client=client)

    with pytest.raises(ProviderError) as error_info:
        await provider.complete(_request())

    assert error_info.value.code == "invalid_provider_response"
    assert tool_id not in str(error_info.value)
    assert tool_name not in str(error_info.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_complete_serializes_tool_call_and_result_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MINI_AGENT_KEY", "test-value")
    tool_call = ToolCall(id="call-1", name="read_file", arguments_json='{"path":"README.md"}')
    request = ModelRequest(
        model="test-model",
        messages=(
            ConversationMessage(role=MessageRole.USER, content="read it"),
            ConversationMessage(role=MessageRole.ASSISTANT, tool_calls=(tool_call,)),
            ConversationMessage(
                role=MessageRole.TOOL,
                content="contents",
                tool_call_id=tool_call.id,
            ),
        ),
    )

    def respond(http_request: httpx.Request) -> httpx.Response:
        payload = json.loads(http_request.content)
        assert payload["messages"][1]["tool_calls"][0]["id"] == tool_call.id
        assert payload["messages"][2] == {
            "role": "tool",
            "content": "contents",
            "tool_call_id": tool_call.id,
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    provider = OpenAICompatibleProvider(_config(), client=client)

    response = await provider.complete(request)

    assert response.content == "done"
    await client.aclose()


@pytest.mark.asyncio
async def test_complete_serializes_tool_definitions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_MINI_AGENT_KEY", "test-value")
    request = ModelRequest(
        model="test-model",
        messages=(ConversationMessage(role=MessageRole.USER, content="read it"),),
        tools=(
            ToolDefinition(
                name="read_file",
                description="Read a file.",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                is_read_only=True,
            ),
        ),
    )

    def respond(http_request: httpx.Request) -> httpx.Response:
        payload = json.loads(http_request.content)
        assert payload["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            }
        ]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    provider = OpenAICompatibleProvider(_config(), client=client)

    await provider.complete(request)

    await client.aclose()
