"""Session event validation and interrupted tool recovery."""

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
    ModelRequestFailedData,
    ModelRequestStartedData,
    RebuildCompletedData,
    RebuildFailedData,
    RebuildStartedData,
    SessionStartedData,
    StoredEvent,
    ToolCompletedData,
    ToolFailedData,
    ToolInterruptedData,
    ToolRecoveryStatus,
    ToolRequestedData,
    ToolStartedData,
)
from mini_agent.goal import GoalLifecycleError, validate_goal_lifecycle
from mini_agent.messages import ToolCall
from mini_agent.permissions import (
    ApprovalArgumentsTooLargeError,
    ApprovalScopeValidationError,
    is_legacy_generic_session_grant,
    is_persisted_apply_patch_session_grant,
    validate_persisted_approval_scope,
)
from mini_agent.redaction import redact_text
from mini_agent.redaction_types import RedactionSummary


def validate_session_events(
    events: tuple[StoredEvent, ...],
    session_id: str,
    *,
    metadata_workspace: str | None = None,
    metadata_model: str | None = None,
) -> None:
    """Validate transcript lifecycle and its immutable session identity.

    Session metadata is a convenient index, not an authority.  The first
    durable event is the transcript's binding for workspace and model, and
    stores their redacted forms.  Comparing redacted values preserves that
    boundary while detecting a replaced ``metadata.json`` before a session can
    be resumed with mismatched grants.
    """

    if not events:
        raise EventStoreCorruptionError("Session event stream is empty")
    if not isinstance(events[0].data, SessionStartedData):
        raise EventStoreCorruptionError("First event is not session_started")
    if any(event.session_id != session_id for event in events):
        raise EventStoreCorruptionError("Event stream contains a different session id")
    if metadata_workspace is not None or metadata_model is not None:
        if metadata_workspace is None or metadata_model is None:
            raise ValueError("Session metadata binding requires workspace and model")
        started = events[0].data
        if (
            started.workspace != redact_text(metadata_workspace).text
            or started.model != redact_text(metadata_model).text
        ):
            raise EventStoreCorruptionError(
                "Session metadata does not match the durable session_started event"
            )
    _validate_tool_lifecycle(events)
    _validate_model_request_lifecycle(events)
    _validate_checkpoint_lifecycle(events)
    _validate_rebuild_lifecycle(events)
    try:
        validate_goal_lifecycle(events)
    except GoalLifecycleError as error:
        raise EventStoreCorruptionError(str(error)) from error


def _validate_model_request_lifecycle(events: tuple[StoredEvent, ...]) -> None:
    """Require every durable model terminal to close one matching request start."""

    if not isinstance(events[0].data, SessionStartedData):
        raise AssertionError("Session start was validated before model lifecycle")
    _active_model_request(events, session_model=events[0].data.model)


def _active_model_request(
    events: tuple[StoredEvent, ...], *, session_model: str
) -> StoredEvent | None:
    """Return the sole unfinished request after validating request terminals."""

    active_request: StoredEvent | None = None

    for event in events:
        data = event.data
        if isinstance(data, ModelRequestStartedData):
            if active_request is not None:
                raise EventStoreCorruptionError(
                    "Model request started before the previous request reached a terminal event"
                )
            if event.turn_id is None or data.model != session_model:
                raise EventStoreCorruptionError("Invalid model request start")
            active_request = event
            continue
        if isinstance(data, ModelRequestFailedData):
            if active_request is None:
                raise EventStoreCorruptionError("Model request failure has no active start")
            started = active_request.data
            if not isinstance(started, ModelRequestStartedData):
                raise AssertionError("Active model request has the wrong event type")
            if event.turn_id != active_request.turn_id or data.attempt != started.attempt:
                raise EventStoreCorruptionError(
                    "Model request failure does not match its active start"
                )
            active_request = None
            continue
        if isinstance(data, AssistantMessageData):
            if active_request is None:
                raise EventStoreCorruptionError("Assistant response has no active model request")
            if event.turn_id != active_request.turn_id:
                raise EventStoreCorruptionError(
                    "Assistant response does not match its active model request"
                )
            active_request = None
    return active_request


def record_incomplete_model_request_recovery(store: EventStore, session_id: str) -> None:
    """Close a provider request that ended without a durable response or failure."""

    if not store.events or not isinstance(store.events[0].data, SessionStartedData):
        raise EventStoreCorruptionError("First event is not session_started")
    active_request = _active_model_request(
        store.events,
        session_model=store.events[0].data.model,
    )

    if active_request is None:
        return
    started = active_request.data
    if not isinstance(started, ModelRequestStartedData):
        raise AssertionError("Active model request has the wrong event type")
    store.append(
        session_id=session_id,
        cycle_id=active_request.cycle_id,
        turn_id=active_request.turn_id,
        data=ModelRequestFailedData(
            error_code="model_request_interrupted",
            message="Model request stopped before a durable terminal event",
            is_retryable=False,
            attempt=started.attempt,
        ),
    )


def _validate_rebuild_lifecycle(events: tuple[StoredEvent, ...]) -> None:
    started: dict[str, RebuildStartedData] = {}
    terminal: set[str] = set()
    current_cycle = 1
    active_rebuild_id: str | None = None

    for event in events:
        data = event.data
        expected_cycle = (
            current_cycle + 1 if isinstance(data, RebuildCompletedData) else current_cycle
        )
        if event.cycle_id != expected_cycle:
            raise EventStoreCorruptionError(
                "Only a completed rebuild may advance the session cycle"
            )
        if isinstance(data, RebuildStartedData):
            if (
                data.rebuild_id in started
                or active_rebuild_id is not None
                or data.source_cycle_id != current_cycle
            ):
                raise EventStoreCorruptionError("Invalid rebuild start")
            started[data.rebuild_id] = data
            active_rebuild_id = data.rebuild_id
            continue
        if not isinstance(data, RebuildCompletedData | RebuildFailedData):
            continue
        source = started.get(data.rebuild_id)
        if (
            source is None
            or data.rebuild_id in terminal
            or active_rebuild_id != data.rebuild_id
        ):
            raise EventStoreCorruptionError("Rebuild terminal event has no unique start")
        terminal.add(data.rebuild_id)
        active_rebuild_id = None
        if isinstance(data, RebuildFailedData):
            if source.source_cycle_id != current_cycle:
                raise EventStoreCorruptionError("Failed rebuild changed cycle")
            continue
        if source.source_cycle_id != current_cycle:
            raise EventStoreCorruptionError("Completed rebuild did not advance one cycle")
        if data.source_message_count != source.source_message_count:
            raise EventStoreCorruptionError("Completed rebuild changed source message count")
        if (
            data.checkpoint_id != source.checkpoint_id
            or data.checkpoint_watermark != source.checkpoint_watermark
        ):
            raise EventStoreCorruptionError("Completed rebuild changed checkpoint source")
        current_cycle = event.cycle_id


def record_incomplete_checkpoint_recovery(store: EventStore, session_id: str) -> None:
    """Close writer attempts that had no durable terminal event before resume."""

    active = _active_checkpoint_start(store.events)
    if active is None:
        return
    store.append(
        session_id=session_id,
        data=CheckpointFailedData(
            checkpoint_id=active.checkpoint_id,
            error_code="checkpoint_interrupted",
            message="Checkpoint writer stopped before a durable commit",
        ),
    )


def record_incomplete_rebuild_recovery(store: EventStore, session_id: str) -> None:
    """Close rebuild attempts that had no durable terminal event before resume."""

    started: dict[str, RebuildStartedData] = {}
    terminal: set[str] = set()
    for event in store.events:
        data = event.data
        if isinstance(data, RebuildStartedData):
            started[data.rebuild_id] = data
        elif isinstance(data, RebuildCompletedData | RebuildFailedData):
            terminal.add(data.rebuild_id)

    for rebuild_id in started.keys() - terminal:
        store.append(
            session_id=session_id,
            data=RebuildFailedData(
                rebuild_id=rebuild_id,
                error_code="rebuild_interrupted",
                message="Context rebuild stopped before a durable terminal event",
            ),
        )


def _validate_checkpoint_lifecycle(events: tuple[StoredEvent, ...]) -> None:
    _active_checkpoint_start(events)


def _active_checkpoint_start(
    events: tuple[StoredEvent, ...],
) -> CheckpointStartedData | None:
    started: dict[str, CheckpointStartedData] = {}
    active: CheckpointStartedData | None = None
    committed_versions: list[int] = []
    committed_watermarks: list[int] = []

    for event in events:
        data = event.data
        if isinstance(data, CheckpointStartedData):
            if data.checkpoint_id in started:
                raise EventStoreCorruptionError("Duplicate checkpoint start")
            if active is not None:
                raise EventStoreCorruptionError("Checkpoint start overlaps an active checkpoint")
            if data.source_through_event_id >= event.id:
                raise EventStoreCorruptionError(
                    "Checkpoint source watermark must precede its start event"
                )
            started[data.checkpoint_id] = data
            active = data
            continue
        if not isinstance(data, CheckpointCommittedData | CheckpointFailedData):
            continue
        source = started.get(data.checkpoint_id)
        if source is None or active is None or active.checkpoint_id != data.checkpoint_id:
            raise EventStoreCorruptionError("Checkpoint terminal event has no unique start")
        active = None
        if isinstance(data, CheckpointFailedData):
            continue
        if data.source_through_event_id != source.source_through_event_id:
            raise EventStoreCorruptionError("Checkpoint commit watermark does not match start")
        committed_versions.append(data.version)
        committed_watermarks.append(data.source_through_event_id)

    if committed_versions != list(range(1, len(committed_versions) + 1)):
        raise EventStoreCorruptionError("Checkpoint versions are not contiguous")
    if committed_watermarks != sorted(set(committed_watermarks)):
        raise EventStoreCorruptionError("Checkpoint watermarks are not strictly increasing")
    return active


def record_incomplete_tool_recovery(store: EventStore, session_id: str) -> None:
    assistant_calls: dict[str, StoredEvent] = {}
    requested: dict[str, ToolRequestedData] = {}
    started: dict[str, StoredEvent] = {}
    resolved: set[str] = set()
    approval_decisions: dict[str, ApprovalDecision] = {}

    for event in store.events:
        data = event.data
        if isinstance(data, AssistantMessageData):
            for tool_call in data.tool_calls:
                assistant_calls[tool_call.id] = event
        elif isinstance(data, ToolRequestedData):
            requested[data.tool_call.id] = data
        elif isinstance(data, ToolStartedData):
            started[data.tool_call_id] = event
        elif isinstance(data, ToolCompletedData | ToolFailedData | ToolInterruptedData):
            resolved.add(data.tool_call_id)
        elif isinstance(data, ApprovalResolvedData):
            approval_decisions[data.tool_call_id] = data.decision

    pending_tool_call_ids = (
        tool_call_id for tool_call_id in assistant_calls if tool_call_id not in resolved
    )
    for tool_call_id in pending_tool_call_ids:
        assistant_event = assistant_calls[tool_call_id]
        start_event = started.get(tool_call_id)
        request = requested.get(tool_call_id)

        if (
            start_event is None
            and approval_decisions.get(tool_call_id) is ApprovalDecision.DENY
        ):
            store.append(
                session_id=session_id,
                turn_id=assistant_event.turn_id,
                correlation_id=tool_call_id,
                data=ToolFailedData(
                    tool_call_id=tool_call_id,
                    error_code="permission_denied",
                    message="User denied this tool call",
                    before_start=True,
                ),
            )
            continue

        if start_event is None:
            recovery_status = ToolRecoveryStatus.INTERRUPTED
            reason = "tool was not confirmed as started; retry requires an explicit new request"
            source_event = assistant_event
        elif request is not None and request.is_read_only:
            recovery_status = ToolRecoveryStatus.INTERRUPTED
            reason = "read-only tool was interrupted and may be retried explicitly"
            source_event = start_event
        else:
            recovery_status = ToolRecoveryStatus.UNKNOWN
            reason = "tool may have produced side effects and must not be replayed automatically"
            source_event = start_event

        store.append(
            session_id=session_id,
            turn_id=source_event.turn_id,
            correlation_id=tool_call_id,
            data=ToolInterruptedData(
                tool_call_id=tool_call_id,
                recovery_status=recovery_status,
                reason=reason,
            ),
        )


def _validate_tool_lifecycle(events: tuple[StoredEvent, ...]) -> None:
    assistant_calls: dict[str, tuple[str | None, ToolCall]] = {}
    requested: set[str] = set()
    requests_by_id: dict[str, ToolRequestedData] = {}
    approval_requests: dict[str, ApprovalRequestedData] = {}
    approval_decisions: dict[str, ApprovalDecision] = {}
    started: set[str] = set()
    started_events_by_id: dict[int, StoredEvent] = {}
    artifact_events_by_id: dict[str, StoredEvent] = {}
    artifact_ids_by_tool_call: dict[str, str] = {}
    resolved: set[str] = set()

    for event in events:
        data = event.data
        if isinstance(data, AssistantMessageData):
            for tool_call in data.tool_calls:
                if tool_call.id in assistant_calls:
                    raise EventStoreCorruptionError(
                        f"Duplicate assistant tool call: {tool_call.id}"
                    )
                assistant_calls[tool_call.id] = (event.turn_id, tool_call)
            continue

        if isinstance(data, ToolRequestedData):
            correlation_id = _require_tool_correlation(event, data.tool_call.id)
            _reject_after_terminal(correlation_id, resolved, "tool request")
            assistant_call = assistant_calls.get(correlation_id)
            if assistant_call is None:
                raise EventStoreCorruptionError(
                    f"Tool request has no assistant call: {correlation_id}"
                )
            assistant_turn_id, assistant_tool_call = assistant_call
            if event.turn_id != assistant_turn_id or data.tool_call != assistant_tool_call:
                raise EventStoreCorruptionError(
                    f"Tool request does not match assistant call: {correlation_id}"
                )
            if correlation_id in requested:
                raise EventStoreCorruptionError(f"Duplicate tool request: {correlation_id}")
            requested.add(correlation_id)
            requests_by_id[correlation_id] = data
            continue

        if isinstance(data, ApprovalRequestedData):
            correlation_id = _require_tool_correlation(event, data.tool_call_id)
            _reject_after_terminal(correlation_id, resolved, "approval request")
            request = requests_by_id.get(correlation_id)
            if request is None or request.is_read_only:
                raise EventStoreCorruptionError(
                    f"Approval request has no write tool request: {correlation_id}"
                )
            assistant_turn_id, assistant_tool_call = assistant_calls[correlation_id]
            if event.turn_id != assistant_turn_id or data.tool_name != assistant_tool_call.name:
                raise EventStoreCorruptionError(
                    f"Approval request does not match tool call: {correlation_id}"
                )
            try:
                validate_persisted_approval_scope(
                    tool_name=request.tool_call.name,
                    is_read_only=request.is_read_only,
                    arguments_json=request.tool_call.arguments_json,
                    redacted_arguments=data.redacted_arguments,
                    scope_descriptor=data.scope_descriptor,
                    scope_fingerprint=data.scope_fingerprint,
                    can_allow_session=data.can_allow_session,
                )
            except (ApprovalScopeValidationError, ApprovalArgumentsTooLargeError) as error:
                raise EventStoreCorruptionError(
                    f"Approval scope does not match tool call: {correlation_id}"
                ) from error
            if correlation_id in approval_requests or correlation_id in started:
                raise EventStoreCorruptionError(
                    f"Invalid approval request order: {correlation_id}"
                )
            approval_requests[correlation_id] = data
            continue

        if isinstance(data, ApprovalResolvedData):
            correlation_id = _require_tool_correlation(event, data.tool_call_id)
            _reject_after_terminal(correlation_id, resolved, "approval decision")
            approval_request = approval_requests.get(correlation_id)
            assistant_turn_id = assistant_calls.get(correlation_id, (None, None))[0]
            if approval_request is None or event.turn_id != assistant_turn_id:
                raise EventStoreCorruptionError(
                    f"Approval decision has no matching request: {correlation_id}"
                )
            if (
                correlation_id in approval_decisions
                or correlation_id in started
                or data.scope_fingerprint != approval_request.scope_fingerprint
            ):
                raise EventStoreCorruptionError(
                    f"Invalid approval decision: {correlation_id}"
                )
            if (
                data.decision is ApprovalDecision.ALLOW_SESSION
                and not is_persisted_allow_session_decision(approval_request)
            ):
                raise EventStoreCorruptionError(
                    f"Sensitive scope cannot receive a session grant: {correlation_id}"
                )
            approval_decisions[correlation_id] = data.decision
            continue

        if isinstance(data, ToolStartedData):
            correlation_id = _require_tool_correlation(event, data.tool_call_id)
            _reject_after_terminal(correlation_id, resolved, "tool start")
            if correlation_id not in requested or correlation_id in started:
                raise EventStoreCorruptionError(f"Invalid tool start: {correlation_id}")
            assistant_turn_id, assistant_tool_call = assistant_calls[correlation_id]
            if event.turn_id != assistant_turn_id or data.tool_name != assistant_tool_call.name:
                raise EventStoreCorruptionError(
                    f"Tool start does not match assistant call: {correlation_id}"
                )
            request = requests_by_id[correlation_id]
            if not request.is_read_only and approval_decisions.get(correlation_id) not in {
                ApprovalDecision.ALLOW_ONCE,
                ApprovalDecision.ALLOW_SESSION,
            }:
                raise EventStoreCorruptionError(
                    f"Write tool started without approval: {correlation_id}"
                )
            started.add(correlation_id)
            started_events_by_id[event.id] = event
            continue

        if isinstance(data, ArtifactCreatedData):
            if data.artifact_id in artifact_events_by_id:
                raise EventStoreCorruptionError(
                    f"Duplicate artifact registration: {data.artifact_id}"
                )
            source_event = started_events_by_id.get(data.source_event_id)
            if source_event is None or not isinstance(source_event.data, ToolStartedData):
                raise EventStoreCorruptionError(
                    f"Artifact source is not a prior tool start: {data.artifact_id}"
                )
            source_tool_call_id = source_event.data.tool_call_id
            if event.turn_id != source_event.turn_id:
                raise EventStoreCorruptionError(
                    f"Artifact source has a different turn: {data.artifact_id}"
                )
            if source_tool_call_id in resolved:
                raise EventStoreCorruptionError(
                    f"Artifact appeared after its tool terminal event: {data.artifact_id}"
                )
            if source_tool_call_id in artifact_ids_by_tool_call:
                raise EventStoreCorruptionError(
                    "Tool created more than one artifact: "
                    f"{source_tool_call_id}"
                )
            artifact_events_by_id[data.artifact_id] = event
            artifact_ids_by_tool_call[source_tool_call_id] = data.artifact_id
            continue

        if isinstance(data, ToolCompletedData | ToolFailedData | ToolInterruptedData):
            correlation_id = _require_tool_correlation(event, data.tool_call_id)
            is_denied_before_start = (
                isinstance(data, ToolFailedData)
                and data.before_start
                and data.error_code == "permission_denied"
                and approval_decisions.get(correlation_id) is ApprovalDecision.DENY
            )
            is_preflight_failure = (
                isinstance(data, ToolFailedData)
                and data.before_start
                and correlation_id in requested
                and correlation_id not in approval_requests
                and correlation_id not in approval_decisions
                and correlation_id not in started
                # A permission denial is only valid after the user explicitly
                # denied its matching approval request.  Do not let a forged
                # preflight terminal event impersonate that decision.
                and data.error_code != "permission_denied"
            )
            can_recover_before_start = isinstance(data, ToolInterruptedData)
            if correlation_id not in assistant_calls or correlation_id in resolved:
                raise EventStoreCorruptionError(f"Invalid tool terminal event: {correlation_id}")
            if (
                correlation_id not in started
                and not can_recover_before_start
                and not is_denied_before_start
                and not is_preflight_failure
            ):
                raise EventStoreCorruptionError(f"Tool did not start: {correlation_id}")
            if (
                isinstance(data, ToolFailedData)
                and data.before_start
                and (
                    correlation_id in started
                    or not (is_denied_before_start or is_preflight_failure)
                )
            ):
                raise EventStoreCorruptionError(
                    f"Invalid pre-start failure: {correlation_id}"
                )
            if isinstance(data, ToolInterruptedData):
                request = requests_by_id.get(correlation_id)
                is_started_write = (
                    correlation_id in started
                    and request is not None
                    and not request.is_read_only
                )
                expected_status = (
                    ToolRecoveryStatus.UNKNOWN
                    if is_started_write
                    else ToolRecoveryStatus.INTERRUPTED
                )
                if data.recovery_status is not expected_status:
                    raise EventStoreCorruptionError(
                        f"Invalid tool recovery status: {correlation_id}"
                    )
            if isinstance(data, ToolCompletedData):
                _validate_completed_artifact_linkage(
                    event=event,
                    data=data,
                    source_artifact_id=artifact_ids_by_tool_call.get(correlation_id),
                    artifact_events_by_id=artifact_events_by_id,
                )
            if isinstance(data, ToolFailedData) and correlation_id in artifact_ids_by_tool_call:
                raise EventStoreCorruptionError(
                    f"Artifact-producing tool cannot finish as failed: {correlation_id}"
                )
            assistant_turn_id, _ = assistant_calls[correlation_id]
            if event.turn_id != assistant_turn_id:
                raise EventStoreCorruptionError(
                    f"Tool terminal event has wrong turn: {correlation_id}"
                )
            resolved.add(correlation_id)

def is_persisted_allow_session_decision(request: ApprovalRequestedData) -> bool:
    """Accept an old generic grant only as inert historical evidence.

    Session restoration filters those grants out separately.  Keeping the
    transcript readable is safer than treating an old successful operation as
    corrupt, while no new authority can result from it.
    """

    return is_persisted_apply_patch_session_grant(
        tool_name=request.tool_name,
        scope_descriptor=request.scope_descriptor,
        can_allow_session=request.can_allow_session,
    ) or is_legacy_generic_session_grant(
        tool_name=request.tool_name,
        redacted_arguments=request.redacted_arguments,
        scope_descriptor=request.scope_descriptor,
        can_allow_session=request.can_allow_session,
    )


def _require_tool_correlation(event: StoredEvent, tool_call_id: str) -> str:
    if event.correlation_id != tool_call_id:
        raise EventStoreCorruptionError(f"Tool event {event.id} has an invalid correlation id")
    return tool_call_id


def _reject_after_terminal(tool_call_id: str, resolved: set[str], event_name: str) -> None:
    if tool_call_id in resolved:
        raise EventStoreCorruptionError(
            f"{event_name.capitalize()} appeared after terminal event: {tool_call_id}"
        )


def _validate_completed_artifact_linkage(
    *,
    event: StoredEvent,
    data: ToolCompletedData,
    source_artifact_id: str | None,
    artifact_events_by_id: dict[str, StoredEvent],
) -> None:
    if data.artifact_id is None:
        if source_artifact_id is not None:
            raise EventStoreCorruptionError(
                f"Artifact-created tool completed inline: {data.tool_call_id}"
            )
        return

    if data.source_redaction_summary != RedactionSummary():
        raise EventStoreCorruptionError(
            "Artifact-backed tool completion duplicated redaction metadata: "
            f"{data.tool_call_id}"
        )

    artifact_event = artifact_events_by_id.get(data.artifact_id)
    if artifact_event is None:
        raise EventStoreCorruptionError(
            f"Tool completion references unknown artifact: {data.artifact_id}"
        )
    artifact_data = artifact_event.data
    if not isinstance(artifact_data, ArtifactCreatedData):
        raise EventStoreCorruptionError(
            f"Tool completion artifact is not registered: {data.artifact_id}"
        )
    if source_artifact_id != data.artifact_id:
        raise EventStoreCorruptionError(
            f"Tool completion artifact belongs to another tool: {data.artifact_id}"
        )
    if event.turn_id != artifact_event.turn_id:
        raise EventStoreCorruptionError(
            f"Tool completion artifact has a different turn: {data.artifact_id}"
        )
    if data.artifact_id not in data.output:
        raise EventStoreCorruptionError(
            f"Tool completion output omits artifact reference: {data.artifact_id}"
        )
