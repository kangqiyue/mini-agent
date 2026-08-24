"""Credential redaction at the durable-event boundary."""

from mini_agent.events import (
    ApprovalRequestedData,
    ApprovalResolvedData,
    ArtifactCreatedData,
    AssistantMessageData,
    CheckpointCommittedData,
    CheckpointFailedData,
    CheckpointStartedData,
    ContextEstimatedData,
    ContextPrunedData,
    EventData,
    GoalCreatedData,
    GoalStatusChangedData,
    GoalUpdatedData,
    ModelRequestFailedData,
    ModelRequestStartedData,
    RebuildCompletedData,
    RebuildFailedData,
    RebuildStartedData,
    SessionResumedData,
    SessionStartedData,
    SessionStoppedData,
    ToolCompletedData,
    ToolFailedData,
    ToolRequestedData,
    ToolStartedData,
    UserMessageData,
)
from mini_agent.exec_command_safety import sanitize_exec_command_arguments
from mini_agent.messages import FinishReason, ToolCall
from mini_agent.redaction import RedactionResult, redact_json_text, redact_text
from mini_agent.redaction_types import (
    RedactionKind,
    RedactionSummary,
)
from mini_agent.tool_facts import ToolCompletionFacts


def redact_event_data(data: EventData) -> tuple[EventData, RedactionSummary]:
    """Return one redacted event payload and a non-sensitive audit summary."""
    redactor = _EventRedactor()
    if isinstance(data, ApprovalRequestedData):
        safe_data = data.model_copy(
            update={
                "redacted_arguments": redactor.json_text(data.redacted_arguments),
                "scope_descriptor": redactor.text(data.scope_descriptor),
            }
        )
    elif isinstance(data, ApprovalResolvedData):
        safe_data = data
    elif isinstance(data, SessionStartedData):
        safe_data = data.model_copy(
            update={"workspace": redactor.text(data.workspace), "model": redactor.text(data.model)}
        )
    elif isinstance(data, SessionResumedData):
        backup_name = data.recovery_backup_name
        safe_data = data.model_copy(
            update={
                "recovery_backup_name": redactor.text(backup_name) if backup_name else None
            }
        )
    elif isinstance(data, SessionStoppedData):
        safe_data = data.model_copy(update={"reason": redactor.text(data.reason)})
    elif isinstance(data, UserMessageData):
        safe_data = data.model_copy(update={"content": redactor.text(data.content)})
    elif isinstance(data, GoalCreatedData | GoalUpdatedData):
        safe_data = data.model_copy(
            update={
                "objective": redactor.text(data.objective),
                "acceptance_criteria": tuple(
                    item.model_copy(update={"description": redactor.text(item.description)})
                    for item in data.acceptance_criteria
                ),
            }
        )
    elif isinstance(data, GoalStatusChangedData):
        safe_data = data.model_copy(
            update={
                "completion_evidence": tuple(
                    item.model_copy(update={"note": redactor.text(item.note)})
                    for item in data.completion_evidence
                ),
                "blocked_reason": (
                    redactor.text(data.blocked_reason)
                    if data.blocked_reason is not None
                    else None
                ),
            }
        )
    elif isinstance(data, ModelRequestStartedData):
        safe_data = data.model_copy(update={"model": redactor.text(data.model)})
    elif isinstance(data, AssistantMessageData):
        safe_data = data.model_copy(
            update={
                "content": (
                    redactor.text(
                        data.content,
                        truncated_at_end=data.finish_reason in {
                            FinishReason.LENGTH,
                            FinishReason.OTHER,
                        },
                    )
                    if data.content is not None
                    else None
                ),
                "tool_calls": tuple(redactor.tool_call(call) for call in data.tool_calls),
            }
        )
    elif isinstance(data, ModelRequestFailedData):
        safe_data = data.model_copy(
            update={
                "error_code": redactor.text(data.error_code),
                "message": redactor.text(data.message),
            }
        )
    elif isinstance(data, ToolRequestedData):
        safe_data = data.model_copy(update={"tool_call": redactor.tool_call(data.tool_call)})
    elif isinstance(
        data,
        ToolStartedData
        | CheckpointStartedData
        | CheckpointCommittedData
        | ContextPrunedData
    ):
        safe_data = data
    elif isinstance(data, ArtifactCreatedData):
        # Artifact content is redacted before its metadata event is constructed.
        # Its own source summary is in ArtifactCreatedData, not this outer event.
        safe_data = data.model_copy(update={"media_type": redactor.text(data.media_type)})
    elif isinstance(data, ContextEstimatedData):
        safe_data = data.model_copy(update={"strategy": redactor.text(data.strategy)})
    elif isinstance(data, ToolCompletedData):
        safe_data = data.model_copy(
            update={
                "output": redactor.text(data.output),
                "facts": ToolCompletionFacts(
                    command_exit_code=data.facts.command_exit_code,
                    modified_paths=tuple(
                        redactor.text(path) for path in data.facts.modified_paths
                    ),
                ),
            }
        )
    elif isinstance(data, ToolFailedData | CheckpointFailedData | RebuildFailedData):
        safe_data = data.model_copy(
            update={
                "error_code": redactor.text(data.error_code),
                "message": redactor.text(data.message),
            }
        )
    elif isinstance(data, RebuildStartedData):
        safe_data = data.model_copy(update={"reason": redactor.text(data.reason)})
    elif isinstance(data, RebuildCompletedData):
        safe_data = data.model_copy(update={"strategy": redactor.text(data.strategy)})
    else:
        safe_data = data.model_copy(update={"reason": redactor.text(data.reason)})
    return safe_data, redactor.summary()


class _EventRedactor:
    """Redact one event without retaining a second representation of secrets."""

    def __init__(self) -> None:
        self._match_count = 0
        self._kinds: set[RedactionKind] = set()

    def text(self, value: str, *, truncated_at_end: bool = False) -> str:
        return self._record(
            redact_text(value, truncated_at_end=truncated_at_end)
        )

    def json_text(self, value: str) -> str:
        return self._record(redact_json_text(value))

    def tool_call(self, tool_call: ToolCall) -> ToolCall:
        arguments_json = tool_call.arguments_json
        if tool_call.name == "exec_command":
            sanitization = sanitize_exec_command_arguments(arguments_json)
            self._match_count += sanitization.redaction_match_count
            self._kinds.update(sanitization.redaction_kinds)
            arguments_json = sanitization.arguments_json
        return tool_call.model_copy(
            update={"arguments_json": self.json_text(arguments_json)}
        )

    def summary(self) -> RedactionSummary:
        return RedactionSummary(
            match_count=self._match_count,
            kinds=tuple(sorted(self._kinds, key=lambda kind: kind.value)),
        )

    def _record(self, result: RedactionResult) -> str:
        self._match_count += result.match_count
        self._kinds.update(result.matched_kinds)
        return result.text
