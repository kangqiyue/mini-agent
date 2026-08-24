"""Validated configuration loading."""

import tomllib
from ipaddress import ip_address
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Self
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from mini_agent.redaction import is_sensitive_key, redact_text


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    provider: Literal["openai-compatible"] = "openai-compatible"
    model: str = Field(min_length=1)
    base_url: HttpUrl
    api_key_env: str = Field(default="MINI_AGENT_API_KEY", pattern=r"^[A-Z][A-Z0-9_]*$")
    context_window: int = Field(default=128_000, ge=4_096)
    max_context: int | None = Field(default=None, ge=4_096)
    max_input_tokens: int | None = Field(default=None, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    max_output_tokens: int = Field(default=16_384, ge=1)
    temperature: float = Field(default=0.0, ge=0, le=2)

    @field_validator("model")
    @classmethod
    def reject_credential_shaped_model_identifier(cls, value: str) -> str:
        if redact_text(value).match_count:
            raise ValueError("Model identifier must not contain credential-shaped content")
        _require_provider_model_identifier(value)
        return value

    @field_validator("base_url", mode="before")
    @classmethod
    def reject_credential_bearing_base_url(cls, value: object) -> object:
        """Reject URLs that could persist credentials in ordinary config text."""

        if not isinstance(value, (str, HttpUrl)):
            return value
        try:
            parts = urlsplit(str(value))
        except ValueError as error:
            raise ValueError("Base URL must be a valid URL without credentials") from error

        if parts.username is not None or parts.password is not None:
            raise ValueError("Base URL must not contain userinfo credentials")
        if parts.fragment:
            raise ValueError("Base URL must not contain a fragment")

        for query_key, query_value in parse_qsl(parts.query, keep_blank_values=True):
            if is_sensitive_key(query_key) or redact_text(query_key).match_count:
                raise ValueError("Base URL must not contain sensitive query parameters")
            if redact_text(query_value).match_count:
                raise ValueError("Base URL query values must not contain credential-shaped content")
        if redact_text(str(value)).match_count:
            raise ValueError("Base URL must not contain credential-shaped content")
        return value

    @field_validator("base_url")
    @classmethod
    def require_https_for_remote_provider(cls, value: HttpUrl) -> HttpUrl:
        if value.scheme == "https":
            return value

        if value.host is None:
            raise ValueError("Provider base URL must include a host")
        host = value.host.strip("[]")
        if host == "localhost":
            return value
        try:
            address = ip_address(host)
        except ValueError:
            address = None
        if address is not None and address.is_loopback:
            return value
        raise ValueError("Remote provider base URL must use HTTPS")

    @field_validator("api_key_env", mode="before")
    @classmethod
    def reject_credential_shaped_api_key_env(cls, value: object) -> object:
        if isinstance(value, str) and redact_text(value).match_count:
            raise ValueError("Credential environment variable name must not contain credentials")
        return value

    @model_validator(mode="after")
    def require_context_overrides_within_physical_window(self) -> Self:
        if self.max_context is not None and self.max_context > self.context_window:
            raise ValueError("Maximum context cannot exceed the provider context window")
        if self.max_input_tokens is not None and self.max_input_tokens > self.context_window:
            raise ValueError("Maximum input tokens cannot exceed the provider context window")
        return self


class ContextConfig(BaseModel):
    """Validated policy for active-context budgeting and rebuild timing."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    checkpoint_milestones: tuple[float, ...] = (0.20, 0.45, 0.70)
    rebuild_ratio: float = Field(default=0.85, gt=0, lt=1)
    reserve_tokens: int = Field(default=16_384, ge=1)
    rebuild_seed_max_tokens: int = Field(default=65_536, ge=1)
    checkpoint_max_tokens: int = Field(default=4_096, ge=1_024)
    checkpoint_max_bytes: int = Field(default=65_536, ge=4_096, le=1_048_576)
    checkpoint_model: str | None = Field(default=None, min_length=1)
    minimum_input_tokens: int = Field(default=1_024, ge=256)

    @field_validator("checkpoint_model")
    @classmethod
    def reject_credential_shaped_checkpoint_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if redact_text(value).match_count:
            raise ValueError("Checkpoint model identifier must not contain credentials")
        _require_provider_model_identifier(value)
        return value

    @model_validator(mode="after")
    def require_ordered_milestones_below_rebuild(self) -> Self:
        if any(value <= 0 or value >= self.rebuild_ratio for value in self.checkpoint_milestones):
            raise ValueError("Checkpoint milestones must be positive and below rebuild ratio")
        if any(
            right <= left
            for left, right in zip(
                self.checkpoint_milestones,
                self.checkpoint_milestones[1:],
                strict=False,
            )
        ):
            raise ValueError("Checkpoint milestones must be strictly increasing")
        return self


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    data_dir: Path | None = None
    provider_retry_count: int = Field(default=2, ge=0, le=10)
    retry_backoff_seconds: float = Field(default=0.25, ge=0, le=30)
    max_model_calls_per_turn: int = Field(default=50, ge=1, le=500)
    inline_tool_result_max_chars: int = Field(default=12_000, ge=1, le=1_000_000)

    @field_validator("data_dir", mode="before")
    @classmethod
    def reject_credential_shaped_data_dir(cls, value: object) -> object:
        if value is not None and redact_text(str(value)).match_count:
            raise ValueError("Runtime data directory must not contain credential-shaped content")
        return value


class SystemPromptConfig(BaseModel):
    """Stable identity plus optional workspace-owned instructions."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    identity: str = Field(
        default=(
            "You are Mini Agent, a local coding agent. Work in the current workspace "
            "through explicit, verifiable actions."
        ),
        min_length=1,
        max_length=2_048,
    )
    instructions_file: Path | None = None
    include_git_status: bool = True
    maximum_instructions_chars: int = Field(default=32_768, ge=1_024, le=262_144)

    @field_validator("identity")
    @classmethod
    def reject_credential_shaped_identity(cls, value: str) -> str:
        if redact_text(value).match_count:
            raise ValueError("System identity must not contain credential-shaped content")
        return value

    @field_validator("instructions_file")
    @classmethod
    def require_workspace_relative_instructions_file(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("System instructions file must be a workspace-relative path")
        if redact_text(value.as_posix()).match_count:
            raise ValueError("System instructions path must not contain credentials")
        return value


class MiniAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    model: ModelConfig
    context: ContextConfig = Field(default_factory=ContextConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    system_prompt: SystemPromptConfig = Field(default_factory=SystemPromptConfig)


class ConfigNotFoundError(FileNotFoundError):
    pass


def load_config(path: Path) -> MiniAgentConfig:
    if not path.is_file():
        raise ConfigNotFoundError(f"Config file does not exist: {path}")

    with path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)

    config = MiniAgentConfig.model_validate(raw_config)
    data_dir = config.runtime.data_dir
    if data_dir is None:
        return config

    resolved_data_dir = data_dir if data_dir.is_absolute() else path.parent / data_dir
    _reject_credential_shaped_data_dir(resolved_data_dir)
    if data_dir.is_absolute():
        return config

    resolved_runtime = config.runtime.model_copy(update={"data_dir": resolved_data_dir})
    return config.model_copy(update={"runtime": resolved_runtime})


def _reject_credential_shaped_data_dir(data_dir: Path) -> None:
    if redact_text(str(data_dir)).match_count:
        raise ValueError("Runtime data directory must not contain credential-shaped content")


def _require_provider_model_identifier(value: str) -> None:
    """Reject local-path routes without rejecting ordinary provider/model ids."""

    normalized = value.strip()
    if normalized != value:
        raise ValueError("Model identifier must not have surrounding whitespace")
    is_local_path = (
        PurePosixPath(normalized).is_absolute()
        or PureWindowsPath(normalized).is_absolute()
        or normalized.startswith(("~/", "~\\", "./", ".\\", "../", "..\\"))
        or normalized.casefold().startswith("file:")
    )
    if is_local_path:
        raise ValueError("Model identifier must not be a local path")
