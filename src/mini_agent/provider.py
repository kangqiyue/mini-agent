"""Provider-independent model interface and physical model limits."""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from mini_agent.config import ModelConfig
from mini_agent.messages import ModelRequest, ModelResponse
from mini_agent.redaction import redact_text


class ProviderError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        is_retryable: bool,
        is_context_overflow: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.is_retryable = is_retryable
        self.is_context_overflow = is_context_overflow


def safe_provider_error(error: ProviderError) -> ProviderError:
    """Copy provider failures across the programmatic boundary without secrets."""

    code_redaction = redact_text(error.code)
    code = error.code
    if (
        code_redaction.match_count > 0
        or "[REDACTED]" in code
        or "[REDACTED_KEY_" in code
    ):
        code = "invalid_provider_error_code"
    message = redact_text(str(error)).text
    return ProviderError(
        code,
        message,
        is_retryable=error.is_retryable,
        is_context_overflow=error.is_context_overflow,
    )


class ProviderCapabilities(BaseModel):
    """Physical limits used to derive a safe request budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context_window: int = Field(ge=4_096)
    max_input_tokens: int | None = Field(default=None, ge=1)
    max_output_tokens: int = Field(ge=1)
    native_tool_calling: bool = True
    usage_reporting: bool = False
    reasoning_support: bool = False


@runtime_checkable
class CapabilityProvider(Protocol):
    @property
    def capabilities(self) -> ProviderCapabilities: ...


class ModelProvider(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return one canonical response or raise ProviderError."""
        ...


def resolve_provider_capabilities(
    provider: ModelProvider,
    config: ModelConfig,
) -> ProviderCapabilities:
    """Use provider-declared limits, falling back to conservative config limits."""

    if isinstance(provider, CapabilityProvider):
        return provider.capabilities
    return ProviderCapabilities(
        context_window=config.context_window,
        max_input_tokens=config.max_input_tokens,
        max_output_tokens=config.max_output_tokens,
    )
