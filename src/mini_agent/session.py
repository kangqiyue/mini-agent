"""Session lifecycle and typed conversation reconstruction."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mini_agent.artifacts import ArtifactRecord, ArtifactStore
from mini_agent.checkpoint import CheckpointStore
from mini_agent.event_store import EventStore, EventStoreCorruptionError
from mini_agent.events import (
    ApprovalDecision,
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
    StoredEvent,
    ToolCompletedData,
    ToolFailedData,
    ToolInterruptedData,
    ToolRecoveryStatus,
    ToolRequestedData,
    ToolStartedData,
    UserMessageData,
)
from mini_agent.goal import (
    AcceptanceCriterion,
    CompletionEvidence,
    CompletionGateResult,
    GoalCompletionRejected,
    GoalState,
    GoalStatus,
    evaluate_completion_gate,
    validate_goal_lifecycle,
)
from mini_agent.messages import ConversationMessage, FinishReason, MessageRole, ToolCall
from mini_agent.permissions import is_persisted_apply_patch_session_grant
from mini_agent.provider import ProviderError
from mini_agent.redaction import redact_text
from mini_agent.redaction_types import RedactionSummary
from mini_agent.session_recovery import (
    record_incomplete_checkpoint_recovery,
    record_incomplete_model_request_recovery,
    record_incomplete_rebuild_recovery,
    record_incomplete_tool_recovery,
    validate_session_events,
)
from mini_agent.tool_facts import ToolCompletionFacts


class SessionNotFoundError(FileNotFoundError):
    pass


class SessionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    workspace: str = Field(min_length=1)
    model: str = Field(min_length=1)
    created_at: datetime


class SessionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    workspace: str
    model: str
    created_at: datetime
    last_event_at: datetime
    event_count: int = Field(ge=1)
    last_event_kind: str
    user_message_count: int = Field(ge=0)
    last_user_message: str | None = None
    has_work: bool
    goal_status: GoalStatus | None = None


class SessionDiscoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    summaries: tuple[SessionSummary, ...]
    unreadable_session_ids: tuple[str, ...]


@dataclass(frozen=True)
class SessionPaths:
    root: Path
    metadata: Path
    events: Path

    @property
    def published_marker(self) -> Path:
        """The durable commit record which makes a session discoverable."""

        return self.root / ".published"

    @property
    def publication_intent_marker(self) -> Path:
        """Versioned marker that distinguishes new staging from legacy sessions."""

        return self.root / ".publication-v2"


class AgentSession:
    def __init__(self, metadata: SessionMetadata, paths: SessionPaths, store: EventStore) -> None:
        self.metadata = metadata
        self.paths = paths
        self.store = store

    @classmethod
    def create(cls, *, data_dir: Path, workspace: Path, model: str) -> AgentSession:
        try:
            resolved_workspace = workspace.expanduser().resolve(strict=True)
            if not resolved_workspace.is_dir():
                raise ValueError
        except (OSError, RuntimeError, ValueError):
            raise ValueError("Workspace must be an existing directory") from None

        canonical_workspace = str(resolved_workspace)
        _reject_credential_shaped_metadata_for_creation(
            workspace=canonical_workspace,
            model=model,
        )
        _prepare_session_storage(data_dir)
        session_id = uuid4().hex
        paths = _session_paths(data_dir, session_id)
        staging_paths = _staging_session_paths(data_dir, session_id)
        staging_paths.root.mkdir(parents=True, mode=0o700)
        _require_session_root(staging_paths.root)

        metadata = SessionMetadata(
            session_id=session_id,
            workspace=canonical_workspace,
            model=model,
            created_at=datetime.now(UTC),
        )
        store: EventStore | None = None
        try:
            _write_metadata(staging_paths.metadata, metadata)
            store = EventStore.open(
                staging_paths.events, workspace_root=Path(canonical_workspace)
            )
            store.append(
                session_id=session_id,
                data=SessionStartedData(workspace=metadata.workspace, model=model),
            )
            _write_publication_intent(staging_paths.publication_intent_marker)
            _fsync_directory(staging_paths.root)
        except BaseException as error:
            if store is not None:
                _close_store_after_failure(store, error)
            raise

        try:
            if _session_path_exists(paths.root):
                raise FileExistsError(f"Session directory already exists: {paths.root.name}")
            os.replace(staging_paths.root, paths.root)
            _fsync_directory(paths.root.parent)
            _write_published_marker(paths.published_marker)
        except BaseException as error:
            _close_store_after_failure(store, error)
            raise

        # The writer lock is attached to `.writer.lock`, which moved together
        # with the staging directory.  Keep that descriptor through publication
        # instead of reopening and introducing a second-writer window.
        store.path = paths.events
        return cls(metadata=metadata, paths=paths, store=store)

    @classmethod
    def peek_metadata(cls, *, data_dir: Path, session_id: str) -> SessionMetadata:
        """Read and validate metadata without opening or changing event history."""

        if re.fullmatch(r"[0-9a-f]{32}", session_id) is None:
            raise SessionNotFoundError(f"Invalid session id: {session_id}")

        sessions_dir = data_dir / "sessions"
        if _legacy_unpublished_marker(sessions_dir, session_id).exists():
            raise SessionNotFoundError(
                f"Session initialization did not complete: {session_id}"
            )

        paths = _session_paths(data_dir, session_id)
        _require_session_root(paths.root)
        if not _is_discoverable_session(paths):
            raise SessionNotFoundError(
                f"Session initialization did not complete: {session_id}"
            )
        _require_session_regular_file(paths.metadata, "metadata")
        _require_session_regular_file(paths.events, "events")

        try:
            metadata = SessionMetadata.model_validate_json(
                _read_session_regular_bytes(paths.metadata)
            )
        except (OSError, UnicodeError, ValidationError, ValueError) as error:
            raise EventStoreCorruptionError(
                "Session metadata is unreadable or invalid"
            ) from error
        if metadata.session_id != session_id:
            raise EventStoreCorruptionError("Session metadata id does not match its directory")
        _reject_credential_shaped_metadata_from_storage(metadata)
        return metadata

    @classmethod
    def resume(cls, *, data_dir: Path, session_id: str) -> AgentSession:
        session = cls.load(data_dir=data_dir, session_id=session_id, allow_recovery=True)
        try:
            store = session.store
            recovery = store.recovery_report
            if not store.events and recovery.discarded_tail_byte_count > 0:
                store.append(
                    session_id=session_id,
                    data=SessionStartedData(
                        workspace=session.metadata.workspace,
                        model=session.metadata.model,
                    ),
                )

            validate_session_events(
                store.events,
                session.metadata.session_id,
                metadata_workspace=session.metadata.workspace,
                metadata_model=session.metadata.model,
            )
            # Verify each durable checkpoint commit before mutating the transcript.
            # A restart reconstructs a checkpoint whose in-memory activation failed.
            CheckpointStore.open(
                session.paths.root,
                registrations=session.checkpoint_registrations,
                workspace_root=Path(session.metadata.workspace),
            )
            ArtifactStore.open(
                session.paths.root,
                registrations=session.artifact_registrations,
                workspace_root=Path(session.metadata.workspace),
            )
            previous_last_event_id = store.events[-1].id
            record_incomplete_model_request_recovery(store, session.metadata.session_id)
            record_incomplete_rebuild_recovery(store, session.metadata.session_id)
            record_incomplete_tool_recovery(store, session.metadata.session_id)
            record_incomplete_checkpoint_recovery(store, session.metadata.session_id)
            store.append(
                session_id=session_id,
                data=SessionResumedData(
                    previous_last_event_id=previous_last_event_id,
                    recovered_tail_byte_count=recovery.discarded_tail_byte_count,
                    recovery_backup_name=(
                        recovery.backup_path.name if recovery.backup_path else None
                    ),
                ),
            )
        except BaseException as error:
            _close_store_after_failure(session.store, error)
            raise
        return session

    @classmethod
    def load(
        cls,
        *,
        data_dir: Path,
        session_id: str,
        allow_recovery: bool = False,
    ) -> AgentSession:
        if re.fullmatch(r"[0-9a-f]{32}", session_id) is None:
            raise SessionNotFoundError(f"Invalid session id: {session_id}")

        metadata = cls.peek_metadata(data_dir=data_dir, session_id=session_id)
        paths = _session_paths(data_dir, session_id)

        store = EventStore.open(
            paths.events,
            allow_recovery=allow_recovery,
            writable=allow_recovery,
            workspace_root=Path(metadata.workspace),
        )
        try:
            is_recovered_empty_stream = (
                allow_recovery
                and not store.events
                and store.recovery_report.discarded_tail_byte_count > 0
            )
            if not is_recovered_empty_stream:
                validate_session_events(
                    store.events,
                    metadata.session_id,
                    metadata_workspace=metadata.workspace,
                    metadata_model=metadata.model,
                )
        except BaseException as error:
            _close_store_after_failure(store, error)
            raise
        return cls(metadata=metadata, paths=paths, store=store)

    @property
    def events(self) -> tuple[StoredEvent, ...]:
        return self.store.events

    def new_turn_id(self) -> str:
        return uuid4().hex

    @property
    def current_cycle_id(self) -> int:
        return max((event.cycle_id for event in self.store.events), default=1)

    def _append(
        self,
        *,
        data: EventData,
        turn_id: str | None = None,
        correlation_id: str | None = None,
        cycle_id: int | None = None,
    ) -> StoredEvent:
        """Append through the current logical cycle unless explicitly overridden."""

        return self.store.append(
            session_id=self.metadata.session_id,
            cycle_id=cycle_id or self.current_cycle_id,
            turn_id=turn_id,
            correlation_id=correlation_id,
            data=data,
        )

    def append_user_message(self, content: str, *, turn_id: str) -> StoredEvent:
        return self._append(
            turn_id=turn_id,
            data=UserMessageData(content=content),
        )

    @property
    def goal(self) -> GoalState | None:
        return validate_goal_lifecycle(self.events)

    def create_goal(
        self,
        *,
        objective: str,
        acceptance_criteria: tuple[AcceptanceCriterion, ...] = (),
        turn_id: str | None = None,
    ) -> GoalState:
        if self.goal is not None:
            raise ValueError("Session already has a goal")
        goal_id = uuid4().hex
        self._append(
            turn_id=turn_id,
            data=GoalCreatedData(
                goal_id=goal_id,
                objective=objective,
                acceptance_criteria=acceptance_criteria,
            ),
        )
        goal = self.goal
        if goal is None:
            raise AssertionError("Goal creation did not produce a goal")
        return goal

    def update_goal(
        self,
        *,
        objective: str,
        acceptance_criteria: tuple[AcceptanceCriterion, ...],
        turn_id: str | None = None,
    ) -> GoalState:
        goal = self._require_goal()
        if goal.status is not GoalStatus.ACTIVE:
            raise ValueError("Only an active goal can be updated")
        self._append(
            turn_id=turn_id,
            data=GoalUpdatedData(
                goal_id=goal.goal_id,
                objective=objective,
                acceptance_criteria=acceptance_criteria,
            ),
        )
        return self._require_goal()

    def complete_goal(
        self,
        evidence: tuple[CompletionEvidence, ...],
        *,
        turn_id: str | None = None,
    ) -> CompletionGateResult:
        goal = self._require_goal()
        if goal.status is not GoalStatus.ACTIVE:
            raise ValueError("Only an active goal can be completed")
        result = evaluate_completion_gate(goal, self.events, evidence=evidence)
        if not result.is_satisfied:
            raise GoalCompletionRejected(result)
        self._append(
            turn_id=turn_id,
            data=GoalStatusChangedData(
                goal_id=goal.goal_id,
                status=GoalStatus.COMPLETED,
                completion_evidence=evidence,
            ),
        )
        return result

    def block_goal(self, reason: str, *, turn_id: str | None = None) -> GoalState:
        return self._change_goal_status(
            status=GoalStatus.BLOCKED,
            blocked_reason=reason,
            turn_id=turn_id,
        )

    def activate_goal(self, *, turn_id: str | None = None) -> GoalState:
        goal = self._require_goal()
        if goal.status is not GoalStatus.BLOCKED:
            raise ValueError("Only a blocked goal can return to active")
        return self._change_goal_status(status=GoalStatus.ACTIVE, turn_id=turn_id)

    def cancel_goal(self, *, turn_id: str | None = None) -> GoalState:
        return self._change_goal_status(status=GoalStatus.CANCELLED, turn_id=turn_id)

    def _change_goal_status(
        self,
        *,
        status: GoalStatus,
        blocked_reason: str | None = None,
        turn_id: str | None = None,
    ) -> GoalState:
        goal = self._require_goal()
        if status is GoalStatus.BLOCKED and goal.status is not GoalStatus.ACTIVE:
            raise ValueError("Only an active goal can become blocked")
        if status is GoalStatus.CANCELLED and goal.status not in {
            GoalStatus.ACTIVE,
            GoalStatus.BLOCKED,
        }:
            raise ValueError("Only an active or blocked goal can be cancelled")
        self._append(
            turn_id=turn_id,
            data=GoalStatusChangedData(
                goal_id=goal.goal_id,
                status=status,
                completion_evidence=(),
                blocked_reason=blocked_reason,
            ),
        )
        return self._require_goal()

    def _require_goal(self) -> GoalState:
        goal = self.goal
        if goal is None:
            raise ValueError("Session has no goal")
        return goal

    def append_model_request_started(
        self,
        *,
        turn_id: str,
        message_count: int,
        attempt: int,
    ) -> StoredEvent:
        return self._append(
            turn_id=turn_id,
            data=ModelRequestStartedData(
                model=self.metadata.model,
                message_count=message_count,
                attempt=attempt,
            ),
        )

    def append_assistant_message(
        self,
        content: str | None,
        *,
        finish_reason: FinishReason,
        tool_calls: tuple[ToolCall, ...] = (),
        turn_id: str,
    ) -> StoredEvent:
        assistant_data = AssistantMessageData(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )
        existing_tool_call_ids = {
            tool_call.id
            for event in self.store.events
            if isinstance(event.data, AssistantMessageData)
            for tool_call in event.data.tool_calls
        }
        reused_tool_call_id = next(
            (tool_call.id for tool_call in tool_calls if tool_call.id in existing_tool_call_ids),
            None,
        )
        if reused_tool_call_id is not None:
            raise ValueError(f"Tool call id was already used: {reused_tool_call_id}")

        return self._append(
            turn_id=turn_id,
            data=assistant_data,
        )

    def append_model_request_failed(
        self,
        error: ProviderError,
        *,
        attempt: int,
        turn_id: str,
    ) -> StoredEvent:
        return self._append(
            turn_id=turn_id,
            data=ModelRequestFailedData(
                error_code=error.code,
                message=str(error),
                is_retryable=error.is_retryable,
                attempt=attempt,
            ),
        )

    def append_checkpoint_started(
        self,
        *,
        checkpoint_id: str,
        source_from_event_id: int,
        source_through_event_id: int,
    ) -> StoredEvent:
        return self._append(
            data=CheckpointStartedData(
                checkpoint_id=checkpoint_id,
                source_from_event_id=source_from_event_id,
                source_through_event_id=source_through_event_id,
            ),
        )

    def append_checkpoint_committed(
        self,
        *,
        checkpoint_id: str,
        version: int,
        source_through_event_id: int,
        content_hash: str,
    ) -> StoredEvent:
        return self._append(
            data=CheckpointCommittedData(
                checkpoint_id=checkpoint_id,
                version=version,
                source_through_event_id=source_through_event_id,
                content_hash=content_hash,
            ),
        )

    def append_checkpoint_failed(
        self,
        *,
        checkpoint_id: str,
        error_code: str,
        message: str,
    ) -> StoredEvent:
        return self._append(
            data=CheckpointFailedData(
                checkpoint_id=checkpoint_id,
                error_code=error_code,
                message=message,
            ),
        )

    def append_context_estimated(
        self,
        *,
        projection_version: int,
        strategy: str,
        input_limit: int,
        estimated_tokens: int,
        utilization_ratio: float,
        checkpoint_id: str | None,
        cycle_id: int,
        turn_id: str,
    ) -> StoredEvent:
        return self.store.append(
            session_id=self.metadata.session_id,
            cycle_id=cycle_id,
            turn_id=turn_id,
            data=ContextEstimatedData(
                projection_version=projection_version,
                strategy=strategy,
                input_limit=input_limit,
                estimated_tokens=estimated_tokens,
                utilization_ratio=utilization_ratio,
                checkpoint_id=checkpoint_id,
            ),
        )

    def append_context_pruned(
        self,
        *,
        projection_version: int,
        source_message_count: int,
        projected_message_count: int,
        omitted_message_count: int,
        cycle_id: int,
        turn_id: str,
    ) -> StoredEvent:
        return self.store.append(
            session_id=self.metadata.session_id,
            cycle_id=cycle_id,
            turn_id=turn_id,
            data=ContextPrunedData(
                projection_version=projection_version,
                source_message_count=source_message_count,
                projected_message_count=projected_message_count,
                omitted_message_count=omitted_message_count,
            ),
        )

    def append_rebuild_started(
        self,
        *,
        rebuild_id: str,
        source_cycle_id: int,
        checkpoint_id: str | None,
        checkpoint_watermark: int,
        reason: str,
        source_message_count: int,
        turn_id: str,
    ) -> StoredEvent:
        return self.store.append(
            session_id=self.metadata.session_id,
            cycle_id=source_cycle_id,
            turn_id=turn_id,
            data=RebuildStartedData(
                rebuild_id=rebuild_id,
                source_cycle_id=source_cycle_id,
                checkpoint_id=checkpoint_id,
                checkpoint_watermark=checkpoint_watermark,
                reason=reason,
                source_message_count=source_message_count,
            ),
        )

    def append_rebuild_completed(
        self,
        *,
        rebuild_id: str,
        projection_version: int,
        strategy: str,
        checkpoint_id: str | None,
        checkpoint_watermark: int,
        projected_message_count: int,
        estimated_tokens: int,
        source_message_count: int,
        cycle_id: int,
        turn_id: str,
    ) -> StoredEvent:
        return self.store.append(
            session_id=self.metadata.session_id,
            cycle_id=cycle_id,
            turn_id=turn_id,
            data=RebuildCompletedData(
                rebuild_id=rebuild_id,
                projection_version=projection_version,
                strategy=strategy,
                checkpoint_id=checkpoint_id,
                checkpoint_watermark=checkpoint_watermark,
                projected_message_count=projected_message_count,
                estimated_tokens=estimated_tokens,
                source_message_count=source_message_count,
            ),
        )

    def append_rebuild_failed(
        self,
        *,
        rebuild_id: str,
        error_code: str,
        message: str,
        cycle_id: int,
        turn_id: str,
    ) -> StoredEvent:
        return self.store.append(
            session_id=self.metadata.session_id,
            cycle_id=cycle_id,
            turn_id=turn_id,
            data=RebuildFailedData(
                rebuild_id=rebuild_id,
                error_code=error_code,
                message=message,
            ),
        )

    def stop(self, reason: str = "user_exit") -> StoredEvent:
        return self.store.append(
            session_id=self.metadata.session_id,
            data=SessionStoppedData(reason=reason),
        )

    def append_tool_requested(
        self,
        tool_call: ToolCall,
        *,
        is_read_only: bool,
        turn_id: str,
    ) -> StoredEvent:
        return self.store.append(
            session_id=self.metadata.session_id,
            turn_id=turn_id,
            correlation_id=tool_call.id,
            data=ToolRequestedData(tool_call=tool_call, is_read_only=is_read_only),
        )

    def append_tool_started(self, tool_call: ToolCall, *, turn_id: str) -> StoredEvent:
        return self.store.append(
            session_id=self.metadata.session_id,
            turn_id=turn_id,
            correlation_id=tool_call.id,
            data=ToolStartedData(tool_call_id=tool_call.id, tool_name=tool_call.name),
        )

    def append_tool_completed(
        self,
        tool_call: ToolCall,
        output: str,
        *,
        artifact_id: str | None = None,
        source_redaction_summary: RedactionSummary | None = None,
        facts: ToolCompletionFacts | None = None,
        turn_id: str,
    ) -> StoredEvent:
        return self.store.append(
            session_id=self.metadata.session_id,
            turn_id=turn_id,
            correlation_id=tool_call.id,
            data=ToolCompletedData(
                tool_call_id=tool_call.id,
                output=output,
                artifact_id=artifact_id,
                source_redaction_summary=source_redaction_summary or RedactionSummary(),
                facts=facts or ToolCompletionFacts(),
            ),
        )

    def append_artifact_created(self, record: ArtifactRecord, *, turn_id: str) -> StoredEvent:
        return self.store.append(
            session_id=self.metadata.session_id,
            turn_id=turn_id,
            data=ArtifactCreatedData(
                artifact_id=record.artifact_id,
                source_event_id=record.source_event_id,
                media_type=record.media_type,
                char_count=record.char_count,
                content_hash=record.content_hash,
                redaction_match_count=record.redaction_match_count,
                redaction_kinds=record.redaction_kinds,
            ),
        )

    def append_tool_failed(
        self,
        tool_call: ToolCall,
        *,
        error_code: str,
        message: str,
        before_start: bool = False,
        turn_id: str,
    ) -> StoredEvent:
        return self.store.append(
            session_id=self.metadata.session_id,
            turn_id=turn_id,
            correlation_id=tool_call.id,
            data=ToolFailedData(
                tool_call_id=tool_call.id,
                error_code=error_code,
                message=message,
                before_start=before_start,
            ),
        )

    def append_tool_interrupted(
        self,
        tool_call: ToolCall,
        *,
        recovery_status: ToolRecoveryStatus,
        reason: str,
        turn_id: str,
    ) -> StoredEvent:
        return self.store.append(
            session_id=self.metadata.session_id,
            turn_id=turn_id,
            correlation_id=tool_call.id,
            data=ToolInterruptedData(
                tool_call_id=tool_call.id,
                recovery_status=recovery_status,
                reason=reason,
            ),
        )

    def append_approval_requested(
        self,
        *,
        tool_call: ToolCall,
        redacted_arguments: str,
        scope_descriptor: str,
        scope_fingerprint: str,
        can_allow_session: bool,
        turn_id: str,
    ) -> StoredEvent:
        return self.store.append(
            session_id=self.metadata.session_id,
            turn_id=turn_id,
            correlation_id=tool_call.id,
            data=ApprovalRequestedData(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                redacted_arguments=redacted_arguments,
                scope_descriptor=scope_descriptor,
                scope_fingerprint=scope_fingerprint,
                can_allow_session=can_allow_session,
            ),
        )

    def append_approval_resolved(
        self,
        *,
        tool_call: ToolCall,
        scope_fingerprint: str,
        decision: ApprovalDecision,
        turn_id: str,
    ) -> StoredEvent:
        return self.store.append(
            session_id=self.metadata.session_id,
            turn_id=turn_id,
            correlation_id=tool_call.id,
            data=ApprovalResolvedData(
                tool_call_id=tool_call.id,
                scope_fingerprint=scope_fingerprint,
                decision=decision,
            ),
        )

    def close(self) -> None:
        self.store.close()

    @property
    def session_grant_fingerprints(self) -> frozenset[str]:
        """Return exact write scopes durably granted for this session."""

        approval_requests = {
            event.data.tool_call_id: event.data
            for event in self.store.events
            if isinstance(event.data, ApprovalRequestedData)
        }
        active_grants: set[str] = set()
        for event in self.store.events:
            data = event.data
            if isinstance(data, ApprovalResolvedData):
                request = approval_requests.get(data.tool_call_id)
                if (
                    data.decision is ApprovalDecision.ALLOW_SESSION
                    and request is not None
                    and is_persisted_apply_patch_session_grant(
                        tool_name=request.tool_name,
                        scope_descriptor=request.scope_descriptor,
                        can_allow_session=request.can_allow_session,
                    )
                    and request.scope_fingerprint == data.scope_fingerprint
                ):
                    active_grants.add(data.scope_fingerprint)
            elif (
                isinstance(data, ToolInterruptedData)
                and data.recovery_status is ToolRecoveryStatus.UNKNOWN
            ):
                request = approval_requests.get(data.tool_call_id)
                if request is not None:
                    active_grants.discard(request.scope_fingerprint)
        return frozenset(active_grants)

    @property
    def artifact_registrations(self) -> tuple[ArtifactCreatedData, ...]:
        """Typed artifact metadata committed by this session's event stream."""

        return tuple(
            event.data
            for event in self.store.events
            if isinstance(event.data, ArtifactCreatedData)
        )

    @property
    def checkpoint_registrations(self) -> tuple[CheckpointCommittedData, ...]:
        """Typed checkpoint commit markers from the durable transcript."""

        return tuple(
            event.data
            for event in self.store.events
            if isinstance(event.data, CheckpointCommittedData)
        )

    def conversation_messages(self) -> tuple[ConversationMessage, ...]:
        messages: list[ConversationMessage] = []
        for event in self.store.events:
            if isinstance(event.data, UserMessageData):
                messages.append(
                    ConversationMessage(role=MessageRole.USER, content=event.data.content)
                )
            elif isinstance(event.data, AssistantMessageData):
                messages.append(
                    ConversationMessage(
                        role=MessageRole.ASSISTANT,
                        content=event.data.content,
                        tool_calls=event.data.tool_calls,
                    )
                )
            elif isinstance(event.data, ToolCompletedData):
                messages.append(
                    ConversationMessage(
                        role=MessageRole.TOOL,
                        content=event.data.output,
                        tool_call_id=event.data.tool_call_id,
                    )
                )
            elif isinstance(event.data, ToolFailedData):
                messages.append(
                    ConversationMessage(
                        role=MessageRole.TOOL,
                        content=f"Tool failed [{event.data.error_code}]: {event.data.message}",
                        tool_call_id=event.data.tool_call_id,
                    )
                )
            elif isinstance(event.data, ToolInterruptedData):
                messages.append(
                    ConversationMessage(
                        role=MessageRole.TOOL,
                        content=f"Tool execution status is {event.data.recovery_status.value}: "
                        f"{event.data.reason}",
                        tool_call_id=event.data.tool_call_id,
                    )
                )
        return tuple(messages)


def discover_sessions(data_dir: Path) -> SessionDiscoveryResult:
    sessions_dir = data_dir / "sessions"
    if not sessions_dir.is_dir():
        return SessionDiscoveryResult(summaries=(), unreadable_session_ids=())

    summaries: list[SessionSummary] = []
    unreadable_session_ids: list[str] = []
    for session_dir in sessions_dir.iterdir():
        if re.fullmatch(r"[0-9a-f]{32}", session_dir.name) is None:
            continue
        if not _is_regular_directory(session_dir):
            continue

        paths = _session_paths(data_dir, session_dir.name)
        if (
            _legacy_unpublished_marker(sessions_dir, session_dir.name).exists()
            or not _is_discoverable_session(paths)
        ):
            continue

        try:
            metadata = AgentSession.peek_metadata(
                data_dir=data_dir,
                session_id=session_dir.name,
            )
            store = EventStore.open(paths.events, allow_recovery=False, writable=False)
            try:
                validate_session_events(
                    store.events,
                    metadata.session_id,
                    metadata_workspace=metadata.workspace,
                    metadata_model=metadata.model,
                )
                user_messages = tuple(
                    event.data.content
                    for event in store.events
                    if isinstance(event.data, UserMessageData)
                )
                has_work = any(
                    event.data.kind
                    not in {"session_started", "session_resumed", "session_stopped"}
                    for event in store.events
                )
                goal = validate_goal_lifecycle(store.events)
                summaries.append(
                    SessionSummary(
                        session_id=metadata.session_id,
                        workspace=metadata.workspace,
                        model=metadata.model,
                        created_at=metadata.created_at,
                        last_event_at=store.events[-1].timestamp,
                        event_count=len(store.events),
                        last_event_kind=store.events[-1].data.kind,
                        user_message_count=len(user_messages),
                        last_user_message=user_messages[-1] if user_messages else None,
                        has_work=has_work,
                        goal_status=goal.status if goal is not None else None,
                    )
                )
            finally:
                store.close()
        except (EventStoreCorruptionError, SessionNotFoundError, OSError):
            unreadable_session_ids.append(session_dir.name)

    return SessionDiscoveryResult(
        summaries=tuple(
            sorted(summaries, key=lambda summary: summary.last_event_at, reverse=True)
        ),
        unreadable_session_ids=tuple(sorted(unreadable_session_ids)),
    )


def list_sessions(data_dir: Path) -> tuple[SessionSummary, ...]:
    return discover_sessions(data_dir).summaries


def _session_paths(data_dir: Path, session_id: str) -> SessionPaths:
    root = data_dir / "sessions" / session_id
    return SessionPaths(root=root, metadata=root / "metadata.json", events=root / "events.jsonl")


def _staging_session_paths(data_dir: Path, session_id: str) -> SessionPaths:
    """Return a hidden, non-session directory for uncommitted initialization."""

    root = data_dir / "sessions" / f".{session_id}.creating-{uuid4().hex}"
    return SessionPaths(root=root, metadata=root / "metadata.json", events=root / "events.jsonl")


def _legacy_unpublished_marker(sessions_dir: Path, session_id: str) -> Path:
    return sessions_dir / f".{session_id}.unpublished"


def _reject_credential_shaped_metadata_for_creation(*, workspace: str, model: str) -> None:
    for field_name, value in (("workspace", workspace), ("model", model)):
        if redact_text(value).match_count:
            raise ValueError(
                f"Session metadata {field_name} contains credential-shaped content"
            )


def _reject_credential_shaped_metadata_from_storage(metadata: SessionMetadata) -> None:
    for field_name, value in (("workspace", metadata.workspace), ("model", metadata.model)):
        if redact_text(value).match_count:
            raise EventStoreCorruptionError(
                f"Session metadata {field_name} contains credential-shaped content"
            )


def _close_store_after_failure(store: EventStore, error: BaseException) -> None:
    """Release a store's writer lock without masking the triggering error."""

    try:
        store.close()
    except BaseException as close_error:
        raise error from close_error


def _session_path_exists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise EventStoreCorruptionError("Session storage is unsafe or unreadable") from error
    return True


def _is_regular_directory(path: Path) -> bool:
    try:
        path_status = os.lstat(path)
    except OSError:
        return False
    return not stat.S_ISLNK(path_status.st_mode) and stat.S_ISDIR(path_status.st_mode)


def _require_session_root(path: Path) -> None:
    if not _is_regular_directory(path):
        raise EventStoreCorruptionError("Session storage is unsafe or unreadable")


def _prepare_session_storage(data_dir: Path) -> None:
    _create_missing_session_directories(data_dir, mode=0o777)
    _require_session_root(data_dir)

    sessions_dir = data_dir / "sessions"
    _create_missing_session_directories(sessions_dir, mode=0o700)
    _require_session_root(sessions_dir)


def _create_missing_session_directories(path: Path, *, mode: int) -> None:
    """Create a missing directory chain and durably record each new entry."""

    missing_directories: list[Path] = []
    current_path = path
    while not _session_path_exists(current_path):
        missing_directories.append(current_path)
        parent_path = current_path.parent
        if parent_path == current_path:
            raise EventStoreCorruptionError("Session storage is unsafe or unreadable")
        current_path = parent_path

    _require_session_root(current_path)
    for missing_path in reversed(missing_directories):
        try:
            missing_path.mkdir(mode=mode)
        except FileExistsError:
            # A concurrent creator is safe only when it produced a real directory.
            _require_session_root(missing_path)
            _fsync_directory(missing_path.parent)
            continue
        except OSError as error:
            raise EventStoreCorruptionError("Session storage is unsafe or unreadable") from error

        _fsync_directory(missing_path.parent)
        _require_session_root(missing_path)


def _require_session_regular_file(path: Path, label: str) -> None:
    try:
        path_status = os.lstat(path)
    except FileNotFoundError as error:
        raise SessionNotFoundError(f"Session {label} does not exist") from error
    except OSError as error:
        raise EventStoreCorruptionError("Session storage is unsafe or unreadable") from error
    if stat.S_ISLNK(path_status.st_mode) or not stat.S_ISREG(path_status.st_mode):
        raise EventStoreCorruptionError("Session storage is unsafe or unreadable")


def _require_session_regular_file_or_missing(path: Path) -> None:
    try:
        path_status = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as error:
        raise EventStoreCorruptionError("Session storage is unsafe or unreadable") from error
    if stat.S_ISLNK(path_status.st_mode) or not stat.S_ISREG(path_status.st_mode):
        raise EventStoreCorruptionError("Session storage is unsafe or unreadable")


def _open_session_regular_file(path: Path, flags: int, *, mode: int = 0o600) -> int:
    try:
        descriptor = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), mode)
    except OSError as error:
        raise EventStoreCorruptionError("Session storage is unsafe or unreadable") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise EventStoreCorruptionError("Session storage is unsafe or unreadable")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_session_regular_bytes(path: Path) -> bytes:
    descriptor = _open_session_regular_file(path, os.O_RDONLY)
    try:
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_metadata(path: Path, metadata: SessionMetadata) -> None:
    encoded_metadata = metadata.model_dump_json(indent=2).encode("utf-8") + b"\n"
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.pending")
    _require_session_regular_file_or_missing(path)
    file_descriptor = _open_session_regular_file(
        temporary_path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
    )
    try:
        bytes_written = 0
        while bytes_written < len(encoded_metadata):
            bytes_written += os.write(file_descriptor, encoded_metadata[bytes_written:])
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
    os.replace(temporary_path, path)

    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    try:
        directory_descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise EventStoreCorruptionError("Session storage is unsafe or unreadable") from error
    try:
        if not stat.S_ISDIR(os.fstat(directory_descriptor).st_mode):
            raise EventStoreCorruptionError("Session storage is unsafe or unreadable")
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


_PUBLISHED_MARKER_CONTENT = b"mini-agent-session-published-v1\n"
_PUBLICATION_INTENT_MARKER_CONTENT = b"mini-agent-session-publication-v2\n"


def _is_discoverable_session(paths: SessionPaths) -> bool:
    """Accept legacy sessions while requiring the new publication commit.

    v0.1 sessions written before this protocol have neither marker.  A session
    carrying either v2 marker, however, was created by the new protocol and is
    exposed only after both exact records are present.
    """

    try:
        has_intent = paths.publication_intent_marker.exists()
        has_publication = paths.published_marker.exists()
    except OSError:
        return False
    if not has_intent and not has_publication:
        return True
    return _has_marker(paths.publication_intent_marker, _PUBLICATION_INTENT_MARKER_CONTENT) and (
        _has_marker(paths.published_marker, _PUBLISHED_MARKER_CONTENT)
    )


def _has_marker(marker: Path, expected_content: bytes) -> bool:
    """Return whether one private protocol marker has its exact content."""

    try:
        return marker.is_file() and marker.read_bytes() == expected_content
    except OSError:
        return False


def _write_publication_intent(marker: Path) -> None:
    """Durably declare that a staging directory follows the v2 protocol."""

    _write_marker(marker, _PUBLICATION_INTENT_MARKER_CONTENT)


def _write_marker(marker: Path, content: bytes) -> None:
    """Create one durable marker without ever overwriting an existing file."""

    descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        bytes_written = 0
        while bytes_written < len(content):
            bytes_written += os.write(descriptor, content[bytes_written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(marker.parent)


def _write_published_marker(marker: Path) -> None:
    """Atomically publish an already-durable session directory.

    The final directory rename and parent fsync happen before this operation.
    Therefore a failure before the visibility commit leaves a formal but
    undiscoverable directory.  The marker is written and its *pending* entry is
    fsynced before the final rename.  The session directory is fsynced again
    after that rename, so a successful `create()` never relies on an
    unpersisted publication entry.  A final fsync error is necessarily an
    explicit ambiguous failure: the marker may be visible, but `create()` does
    not report success.
    """

    pending_marker = marker.with_name(f".{marker.name}.{uuid4().hex}.pending")
    descriptor = os.open(pending_marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        bytes_written = 0
        while bytes_written < len(_PUBLISHED_MARKER_CONTENT):
            bytes_written += os.write(
                descriptor,
                _PUBLISHED_MARKER_CONTENT[bytes_written:],
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(marker.parent)
    try:
        os.replace(pending_marker, marker)
    except BaseException:
        # POSIX rename is atomic but a caller can observe any interruption
        # after the kernel committed it.  If the exact observable commit record
        # is present, treat that state as success rather than raising an
        # ambiguous failure which would violate the publication boundary.
        if not _has_marker(marker, _PUBLISHED_MARKER_CONTENT):
            raise
    _fsync_directory(marker.parent)
