"""Small dependency-free types shared by redaction producers and consumers."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RedactionKind(StrEnum):
    AUTHORIZATION = "authorization"
    HOST_PATH = "host_path"
    NAMED_SECRET = "named_secret"
    SECRET_PREFIX = "secret_prefix"
    PRIVATE_KEY = "private_key"


def validate_redaction_summary_fields(
    *,
    match_count: int,
    kinds: tuple[RedactionKind, ...],
    subject: str,
) -> None:
    """Reject ambiguous aggregate redaction metadata at every persistence boundary."""
    expected_kinds = tuple(sorted(set(kinds), key=lambda kind: kind.value))
    if kinds != expected_kinds:
        raise ValueError(f"{subject} kinds must be sorted and unique")
    if (match_count == 0) != (not kinds):
        raise ValueError(
            f"{subject} match count must be zero exactly when kinds are empty"
        )


class RedactionSummary(BaseModel):
    """Aggregate-only record of material removed before durable storage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    match_count: int = Field(default=0, ge=0)
    kinds: tuple[RedactionKind, ...] = ()

    @model_validator(mode="after")
    def require_sorted_unique_kinds(self) -> Self:
        validate_redaction_summary_fields(
            match_count=self.match_count,
            kinds=self.kinds,
            subject="Redaction summary",
        )
        return self
