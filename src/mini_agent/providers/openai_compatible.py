"""OpenAI-compatible chat completions adapter."""

from __future__ import annotations

import json
import os
from typing import Literal, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError

from mini_agent.config import ModelConfig
from mini_agent.messages import (
    MAX_TOOL_CALLS_PER_RESPONSE,
    ConversationMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolCall,
)
from mini_agent.provider import ProviderCapabilities, ProviderError

_MINIMUM_PROVIDER_RESPONSE_BYTES = 1_048_576
_MAXIMUM_PROVIDER_RESPONSE_BYTES = 16_777_216
_BYTES_PER_OUTPUT_TOKEN = 16


class _ResponseFunction(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    name: str
    arguments: str


class _ResponseToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    type: Literal["function"]
    function: _ResponseFunction


class _ResponseMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    content: str | None = None
    tool_calls: tuple[_ResponseToolCall, ...] = ()


class _ResponseChoice(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    message: _ResponseMessage
    finish_reason: str | None = None


class _ResponseUsage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class _ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    choices: tuple[_ResponseChoice, ...] = Field(min_length=1)
    usage: _ResponseUsage | None = None


class OpenAICompatibleProvider:
    def __init__(
        self,
        config: ModelConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._owns_client = client is None

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            context_window=self._config.context_window,
            max_input_tokens=self._config.max_input_tokens,
            max_output_tokens=self._config.max_output_tokens,
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        api_key = os.environ.get(self._config.api_key_env)
        if not api_key:
            raise ProviderError(
                "missing_credential",
                f"Missing credential environment variable: {self._config.api_key_env}",
                is_retryable=False,
            )

        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._config.timeout_seconds,
                trust_env=False,
            )

        payload = {
            "model": request.model,
            "messages": [_serialize_message(message) for message in request.messages],
            "max_tokens": request.max_output_tokens or self._config.max_output_tokens,
            "temperature": self._config.temperature,
        }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": definition.name,
                        "description": definition.description,
                        "parameters": definition.parameters,
                    },
                }
                for definition in request.tools
            ]
        url = _chat_completions_url(self._config.base_url)

        transport_failure: ProviderError | None = None
        status_code: int | None = None
        response_body: bytes | None = None
        try:
            async with self._client.stream(
                "POST",
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            ) as response:
                status_code = response.status_code
                response_body = await _read_bounded_response(
                    response,
                    maximum_bytes=_provider_response_limit(self._config),
                )
        except httpx.TimeoutException:
            transport_failure = ProviderError(
                "provider_timeout", "Model provider request timed out", is_retryable=True
            )
        except httpx.TransportError:
            transport_failure = ProviderError(
                "provider_transport", "Model provider connection failed", is_retryable=True
            )

        # Raise after the transport exception handler exits.  httpx request
        # errors retain the complete request (including its headers), so
        # chaining one would expose credentials through ProviderError.
        if transport_failure is not None:
            raise transport_failure
        if status_code is None or response_body is None:
            raise RuntimeError("Provider transport ended without a response")

        if status_code >= 400:
            response_text = response_body.decode("utf-8", errors="replace")
            is_context_overflow = status_code in {400, 413} and _is_context_overflow(
                response_text
            )
            is_retryable = not is_context_overflow and (
                status_code in {408, 425, 429} or status_code >= 500
            )
            raise ProviderError(
                "context_overflow"
                if is_context_overflow
                else f"provider_http_{status_code}",
                "Model provider rejected the request context"
                if is_context_overflow
                else f"Model provider returned HTTP {status_code}",
                is_retryable=is_retryable,
                is_context_overflow=is_context_overflow,
            )

        response_parsing_failure: ProviderError | None = None
        parsed_response: _ChatCompletionResponse | None = None
        try:
            raw_response = json.loads(response_body)
            _validate_raw_tool_call_limits(raw_response)
            parsed_response = _ChatCompletionResponse.model_validate(raw_response)
        except ProviderError:
            raise
        except (ValueError, ValidationError, RecursionError):
            response_parsing_failure = ProviderError(
                "invalid_provider_response",
                "Model provider returned an invalid response",
                is_retryable=False,
            )

        # Provider payload validation errors can embed the rejected input.
        # Do not make them reachable through an exception chain.
        if response_parsing_failure is not None:
            raise response_parsing_failure
        if parsed_response is None:
            raise RuntimeError("Provider response parsing ended without a result")

        choice = parsed_response.choices[0]
        response_conversion_failure: ProviderError | None = None
        model_response: ModelResponse | None = None
        try:
            tool_calls = tuple(
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments_json=tool_call.function.arguments,
                )
                for tool_call in choice.message.tool_calls
            )
            model_response = ModelResponse(
                content=choice.message.content,
                tool_calls=tool_calls,
                finish_reason=_finish_reason(choice.finish_reason),
                usage=_usage(parsed_response.usage),
            )
        except (ValidationError, ValueError, RecursionError):
            response_conversion_failure = ProviderError(
                "invalid_provider_response",
                "Model provider returned invalid message content",
                is_retryable=False,
            )

        # Canonical model validation may similarly retain provider-controlled
        # values in its exception details.
        if response_conversion_failure is not None:
            raise response_conversion_failure
        if model_response is None:
            raise RuntimeError("Provider response conversion ended without a result")
        return model_response

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()


def _chat_completions_url(base_url: HttpUrl) -> httpx.URL:
    """Append the endpoint path without disturbing ordinary query parameters."""

    url = httpx.URL(str(base_url))
    return url.copy_with(path=f"{url.path.rstrip('/')}/chat/completions")


def _provider_response_limit(config: ModelConfig) -> int:
    estimated_limit = config.max_output_tokens * _BYTES_PER_OUTPUT_TOKEN
    return min(
        _MAXIMUM_PROVIDER_RESPONSE_BYTES,
        max(_MINIMUM_PROVIDER_RESPONSE_BYTES, estimated_limit),
    )


def _validate_raw_tool_call_limits(raw_response: object) -> None:
    """Reject oversized tool-call lists before constructing canonical calls."""

    if not isinstance(raw_response, dict):
        return
    raw_object = cast(dict[str, object], raw_response)
    raw_choices = raw_object.get("choices")
    if not isinstance(raw_choices, list):
        return
    for raw_choice in cast(list[object], raw_choices):
        if not isinstance(raw_choice, dict):
            continue
        raw_choice_object = cast(dict[str, object], raw_choice)
        raw_message = raw_choice_object.get("message")
        if not isinstance(raw_message, dict):
            continue
        raw_message_object = cast(dict[str, object], raw_message)
        raw_tool_calls = raw_message_object.get("tool_calls")
        if (
            isinstance(raw_tool_calls, list)
            and len(cast(list[object], raw_tool_calls)) > MAX_TOOL_CALLS_PER_RESPONSE
        ):
            raise ProviderError(
                "invalid_provider_response",
                "Model provider returned too many tool calls",
                is_retryable=False,
            )


async def _read_bounded_response(
    response: httpx.Response,
    *,
    maximum_bytes: int,
) -> bytes:
    chunks: list[bytes] = []
    byte_count = 0
    async for chunk in response.aiter_bytes():
        byte_count += len(chunk)
        if byte_count > maximum_bytes:
            raise ProviderError(
                "provider_response_too_large",
                "Model provider response exceeded the configured safety limit",
                is_retryable=False,
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _usage(raw_usage: _ResponseUsage | None) -> TokenUsage | None:
    """Normalize provider usage once; report None when nothing was reported."""

    if raw_usage is None:
        return None
    if (
        raw_usage.prompt_tokens is None
        and raw_usage.completion_tokens is None
        and raw_usage.total_tokens is None
    ):
        return None
    return TokenUsage(
        prompt_tokens=raw_usage.prompt_tokens,
        completion_tokens=raw_usage.completion_tokens,
        total_tokens=raw_usage.total_tokens,
    )


def _finish_reason(value: str | None) -> FinishReason:
    if value == "stop":
        return FinishReason.STOP
    if value == "length":
        return FinishReason.LENGTH
    if value == "tool_calls":
        return FinishReason.TOOL_CALLS
    return FinishReason.OTHER


def _is_context_overflow(response_text: str) -> bool:
    normalized = response_text.lower()
    return any(
        marker in normalized
        for marker in (
            "context_length_exceeded",
            "maximum context length",
            "context window",
            "too many tokens",
        )
    )


def _serialize_message(message: ConversationMessage) -> dict[str, object]:
    serialized: dict[str, object] = {"role": message.role.value}
    if message.content is not None:
        serialized["content"] = message.content

    if message.role is MessageRole.ASSISTANT and message.tool_calls:
        serialized["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments_json,
                },
            }
            for tool_call in message.tool_calls
        ]
    if message.role is MessageRole.TOOL:
        serialized["tool_call_id"] = message.tool_call_id
    return serialized
