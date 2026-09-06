"""Typed event records persisted in a session transcript."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mini_agent.goal import (
    AcceptanceCriterion,
    CompletionEvidence,
    GoalStatus,
)
from mini_agent.messages import (
    FinishReason,
    TokenUsage,
    ToolCall,
    validate_tool_call_id,
    validate_tool_name,
)
from mini_agent.redaction import redact_text
from mini_agent.redaction_types import (
    RedactionKind,
    RedactionSummary,
    validate_redaction_summary_fields,
)
from mini_agent.tool_facts import ToolCompletionFacts

CURRENT_EVENT_SCHEMA_VERSION = 2
LEGACY_EVENT_SCHEMA_VERSION = 1


class EventDataBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    @field_validator("tool_call_id", check_fields=False)
    @classmethod
    def tool_call_id_must_not_be_credential_shaped(cls, value: str) -> str:
        return validate_tool_call_id(value)

    @field_validator("tool_name", check_fields=False)
    @classmethod
    def tool_name_must_not_be_credential_shaped(cls, value: str) -> str:
        return validate_tool_name(value)


class SessionStartedData(EventDataBase):
    kind: Literal["session_started"] = "session_started"
    workspace: str
    model: str


class SessionResumedData(EventDataBase):
    kind: Literal["session_resumed"] = "session_resumed"
    previous_last_event_id: int = Field(ge=1)
    recovered_tail_byte_count: int = Field(default=0, ge=0)
    recovery_backup_name: str | None = None


class SessionStoppedData(EventDataBase):
    kind: Literal["session_stopped"] = "session_stopped"
    reason: str = Field(min_length=1)


class GoalCreatedData(EventDataBase):
    kind: Literal["goal_created"] = "goal_created"
    goal_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    objective: str = Field(min_length=1, max_length=4_096)
    acceptance_criteria: tuple[AcceptanceCriterion, ...] = Field(max_length=16)

    @model_validator(mode="after")
    def require_unique_criterion_ids(self) -> Self:
        _require_unique_goal_criterion_ids(self.acceptance_criteria)
        return self


class GoalUpdatedData(EventDataBase):
    kind: Literal["goal_updated"] = "goal_updated"
    goal_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    objective: str = Field(min_length=1, max_length=4_096)
    acceptance_criteria: tuple[AcceptanceCriterion, ...] = Field(max_length=16)

    @model_validator(mode="after")
    def require_unique_criterion_ids(self) -> Self:
        _require_unique_goal_criterion_ids(self.acceptance_criteria)
        return self


class GoalStatusChangedData(EventDataBase):
    kind: Literal["goal_status_changed"] = "goal_status_changed"
    goal_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    status: GoalStatus
    completion_evidence: tuple[CompletionEvidence, ...] = Field(max_length=16)
    blocked_reason: str | None = Field(default=None, min_length=1, max_length=2_048)

    @model_validator(mode="after")
    def require_status_payload_shape(self) -> Self:
        evidence_ids = [item.criterion_id for item in self.completion_evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("Goal completion evidence criterion ids must be unique")
        if self.status is GoalStatus.COMPLETED and not self.completion_evidence:
            raise ValueError("Completed goal status requires evidence")
        if self.status is not GoalStatus.COMPLETED and self.completion_evidence:
            raise ValueError("Only completed goal status can carry evidence")
        if self.status is GoalStatus.BLOCKED and self.blocked_reason is None:
            raise ValueError("Blocked goal status requires a reason")
        if self.status is not GoalStatus.BLOCKED and self.blocked_reason is not None:
            raise ValueError("Only blocked goal status can carry a reason")
        return self


def _require_unique_goal_criterion_ids(
    criteria: tuple[AcceptanceCriterion, ...],
) -> None:
    criterion_ids = [item.criterion_id for item in criteria]
    if len(set(criterion_ids)) != len(criterion_ids):
        raise ValueError("Goal acceptance criterion ids must be unique")


class UserMessageData(EventDataBase):
    kind: Literal["user_message"] = "user_message"
    content: str = Field(min_length=1)


class ModelRequestStartedData(EventDataBase):
    kind: Literal["model_request_started"] = "model_request_started"
    model: str = Field(min_length=1)
    message_count: int = Field(ge=1)
    attempt: int = Field(ge=1)


class AssistantMessageData(EventDataBase):
    kind: Literal["assistant_message"] = "assistant_message"
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: FinishReason
    usage: TokenUsage | None = None

    @model_validator(mode="after")
    def require_content_or_tool_calls(self) -> Self:
        has_content = self.content is not None and bool(self.content.strip())
        if not has_content and not self.tool_calls:
            raise ValueError("Assistant event requires content or tool calls")
        tool_call_ids = {tool_call.id for tool_call in self.tool_calls}
        if len(tool_call_ids) != len(self.tool_calls):
            raise ValueError("Assistant event tool call ids must be unique")
        if self.tool_calls and self.finish_reason is not FinishReason.TOOL_CALLS:
            raise ValueError("Assistant event with tool calls requires tool_calls finish reason")
        if self.finish_reason is FinishReason.TOOL_CALLS and not self.tool_calls:
            raise ValueError("tool_calls finish reason requires at least one tool call")
        return self


class ModelRequestFailedData(EventDataBase):
    kind: Literal["model_request_failed"] = "model_request_failed"
    error_code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    is_retryable: bool
    attempt: int = Field(ge=1)


class ToolRecoveryStatus(StrEnum):
    INTERRUPTED = "interrupted"
    UNKNOWN = "unknown"


class ToolRequestedData(EventDataBase):
    kind: Literal["tool_requested"] = "tool_requested"
    tool_call: ToolCall
    is_read_only: bool


class ToolStartedData(EventDataBase):
    kind: Literal["tool_started"] = "tool_started"
    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)


class ToolCompletedData(EventDataBase):
    kind: Literal["tool_completed"] = "tool_completed"
    tool_call_id: str = Field(min_length=1)
    output: str = Field(min_length=1)
    artifact_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    source_redaction_summary: RedactionSummary = Field(default_factory=RedactionSummary)
    facts: ToolCompletionFacts
    legacy_facts_missing: bool = Field(default=False, exclude=True)

    @model_validator(mode="after")
    def require_artifact_redaction_summary_to_be_empty(self) -> Self:
        if self.artifact_id is not None and self.source_redaction_summary != RedactionSummary():
            raise ValueError(
                "Artifact-backed tool completion must not duplicate redaction metadata"
            )
        return self


class ArtifactCreatedData(EventDataBase):
    kind: Literal["artifact_created"] = "artifact_created"
    artifact_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source_event_id: int = Field(ge=1)
    media_type: str = Field(
        min_length=3,
        max_length=127,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*$",
    )
    char_count: int = Field(ge=0)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    redaction_match_count: int = Field(ge=0)
    redaction_kinds: tuple[RedactionKind, ...] = ()
    # M1 recorded the aggregate count but not its categories.  EventStore
    # supplies this flag only while decoding those already-persisted records;
    # new events retain the stricter complete-summary requirement below.
    legacy_redaction_kinds_unknown: bool = Field(default=False, exclude=True)

    @field_validator("media_type")
    @classmethod
    def media_type_must_not_be_credential_shaped(cls, value: str) -> str:
        if redact_text(value).match_count > 0:
            raise ValueError("Media type cannot be credential-shaped")
        return value

    @model_validator(mode="after")
    def require_sorted_unique_redaction_kinds(self) -> Self:
        if self.legacy_redaction_kinds_unknown:
            if self.redaction_match_count == 0 or self.redaction_kinds:
                raise ValueError(
                    "Legacy artifact redaction metadata must have a positive count and no kinds"
                )
            return self
        validate_redaction_summary_fields(
            match_count=self.redaction_match_count,
            kinds=self.redaction_kinds,
            subject="Artifact redaction",
        )
        return self


class ToolFailedData(EventDataBase):
    kind: Literal["tool_failed"] = "tool_failed"
    tool_call_id: str = Field(min_length=1)
    error_code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    before_start: bool = False


class ToolInterruptedData(EventDataBase):
    kind: Literal["tool_interrupted"] = "tool_interrupted"
    tool_call_id: str = Field(min_length=1)
    recovery_status: ToolRecoveryStatus
    reason: str = Field(min_length=1)


class ApprovalDecision(StrEnum):
    ALLOW_ONCE = "allow_once"
    ALLOW_SESSION = "allow_session"
    DENY = "deny"


class ApprovalRequestedData(EventDataBase):
    kind: Literal["approval_requested"] = "approval_requested"
    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    redacted_arguments: str = Field(min_length=2)
    scope_descriptor: str = Field(min_length=1, max_length=256)
    scope_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    can_allow_session: bool


class ApprovalResolvedData(EventDataBase):
    kind: Literal["approval_resolved"] = "approval_resolved"
    tool_call_id: str = Field(min_length=1)
    scope_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: ApprovalDecision


class CheckpointStartedData(EventDataBase):
    kind: Literal["checkpoint_started"] = "checkpoint_started"
    checkpoint_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source_from_event_id: int = Field(ge=1)
    source_through_event_id: int = Field(ge=1)

    @model_validator(mode="after")
    def require_non_empty_source_range(self) -> Self:
        if self.source_from_event_id > self.source_through_event_id:
            raise ValueError("Checkpoint source range cannot be reversed")
        return self


class CheckpointCommittedData(EventDataBase):
    kind: Literal["checkpoint_committed"] = "checkpoint_committed"
    checkpoint_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    version: int = Field(ge=1)
    source_through_event_id: int = Field(ge=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class CheckpointFailedData(EventDataBase):
    kind: Literal["checkpoint_failed"] = "checkpoint_failed"
    checkpoint_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    error_code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ContextEstimatedData(EventDataBase):
    kind: Literal["context_estimated"] = "context_estimated"
    projection_version: int = Field(ge=1)
    strategy: str = Field(min_length=1)
    input_limit: int = Field(ge=1)
    estimated_tokens: int = Field(ge=0)
    utilization_ratio: float = Field(ge=0)
    checkpoint_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


class ContextPrunedData(EventDataBase):
    kind: Literal["context_pruned"] = "context_pruned"
    projection_version: int = Field(ge=1)
    source_message_count: int = Field(ge=1)
    projected_message_count: int = Field(ge=1)
    omitted_message_count: int = Field(ge=0)


class RebuildStartedData(EventDataBase):
    kind: Literal["rebuild_started"] = "rebuild_started"
    rebuild_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source_cycle_id: int = Field(ge=1)
    checkpoint_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    checkpoint_watermark: int = Field(default=0, ge=0)
    reason: str = Field(min_length=1)
    source_message_count: int = Field(ge=1)


class RebuildCompletedData(EventDataBase):
    kind: Literal["rebuild_completed"] = "rebuild_completed"
    rebuild_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    projection_version: int = Field(ge=1)
    strategy: str = Field(min_length=1)
    checkpoint_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    checkpoint_watermark: int = Field(default=0, ge=0)
    projected_message_count: int = Field(ge=1)
    estimated_tokens: int = Field(ge=0)
    source_message_count: int = Field(ge=1)


class RebuildFailedData(EventDataBase):
    kind: Literal["rebuild_failed"] = "rebuild_failed"
    rebuild_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    error_code: str = Field(min_length=1)
    message: str = Field(min_length=1)


type EventData = Annotated[
    SessionStartedData
    | SessionResumedData
    | SessionStoppedData
    | GoalCreatedData
    | GoalUpdatedData
    | GoalStatusChangedData
    | UserMessageData
    | ModelRequestStartedData
    | AssistantMessageData
    | ModelRequestFailedData
    | ToolRequestedData
    | ToolStartedData
    | ToolCompletedData
    | ArtifactCreatedData
    | ToolFailedData
    | ToolInterruptedData
    | ApprovalRequestedData
    | ApprovalResolvedData
    | CheckpointStartedData
    | CheckpointCommittedData
    | CheckpointFailedData
    | ContextEstimatedData
    | ContextPrunedData
    | RebuildStartedData
    | RebuildCompletedData
    | RebuildFailedData,
    Field(discriminator="kind"),
]


class StoredEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[2] = CURRENT_EVENT_SCHEMA_VERSION
    id: int = Field(ge=1)
    session_id: str = Field(min_length=1)
    cycle_id: int = Field(default=1, ge=1)
    turn_id: str | None = None
    timestamp: datetime
    correlation_id: str | None = None
    redaction_summary: RedactionSummary = Field(default_factory=RedactionSummary)
    data: EventData

    @field_validator("session_id")
    @classmethod
    def session_id_must_not_be_credential_shaped(cls, value: str) -> str:
        return validate_tool_call_id(value)

    @field_validator("turn_id")
    @classmethod
    def turn_id_must_not_be_credential_shaped(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_tool_call_id(value)

    @field_validator("correlation_id")
    @classmethod
    def correlation_id_must_not_be_credential_shaped(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_tool_call_id(value)
