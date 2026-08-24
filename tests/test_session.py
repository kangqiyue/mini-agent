import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from mini_agent.artifacts import ArtifactStore
from mini_agent.checkpoint import Checkpoint, CheckpointStore, CheckpointStoreCorruptionError
from mini_agent.event_store import EventStoreCorruptionError
from mini_agent.events import (
    ApprovalDecision,
    ArtifactCreatedData,
    AssistantMessageData,
    ModelRequestFailedData,
    SessionResumedData,
    StoredEvent,
    ToolCompletedData,
    ToolInterruptedData,
    ToolRecoveryStatus,
)
from mini_agent.messages import FinishReason, MessageRole, ToolCall
from mini_agent.permissions import recompute_approval_scope
from mini_agent.redaction_types import RedactionKind, RedactionSummary
from mini_agent.session import (
    AgentSession,
    discover_sessions,
    list_sessions,
)
from mini_agent.session_recovery import validate_session_events
from mini_agent.tool_facts import ToolCompletionFacts
from tests.support.synthetic_secrets import synthetic_stripe_access_token

_SYNTHETIC_METADATA_SECRET = synthetic_stripe_access_token("METADATA")


def _start_read_tool(session: AgentSession) -> tuple[str, ToolCall, StoredEvent]:
    tool_call = ToolCall(id="call-artifact", name="read_file", arguments_json='{"path":"x"}')
    turn_id = session.new_turn_id()
    session.append_model_request_started(turn_id=turn_id, message_count=1, attempt=1)
    session.append_assistant_message(
        None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(tool_call,),
        turn_id=turn_id,
    )
    session.append_tool_requested(tool_call, is_read_only=True, turn_id=turn_id)
    started_event = session.append_tool_started(tool_call, turn_id=turn_id)
    return turn_id, tool_call, started_event


def _artifact_created(source_event_id: int, artifact_id: str = "a" * 32) -> ArtifactCreatedData:
    return ArtifactCreatedData(
        artifact_id=artifact_id,
        source_event_id=source_event_id,
        media_type="text/plain",
        char_count=8,
        content_hash="b" * 64,
        redaction_match_count=0,
    )


def test_artifact_backed_completion_rejects_duplicate_source_redaction_summary() -> None:
    summary = RedactionSummary(
        match_count=1,
        kinds=(RedactionKind.NAMED_SECRET,),
    )

    with pytest.raises(ValidationError, match="must not duplicate redaction metadata"):
        ToolCompletedData(
            tool_call_id="call-artifact",
            output="Artifact aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            artifact_id="a" * 32,
            source_redaction_summary=summary,
            facts=ToolCompletionFacts(),
        )

    inline_completion = ToolCompletedData(
        tool_call_id="call-inline",
        output="[REDACTED]",
        source_redaction_summary=summary,
        facts=ToolCompletionFacts(),
    )
    assert inline_completion.source_redaction_summary == summary


def test_create_and_resume_reconstructs_conversation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=workspace, model="m")
    turn_id = session.new_turn_id()
    session.append_user_message("question", turn_id=turn_id)
    session.append_model_request_started(turn_id=turn_id, message_count=1, attempt=1)
    session.append_assistant_message(
        "answer",
        finish_reason=FinishReason.STOP,
        turn_id=turn_id,
    )
    session.stop()
    session.close()

    resumed = AgentSession.resume(
        data_dir=tmp_path / "data",
        session_id=session.metadata.session_id,
    )

    assert isinstance(resumed.events[-1].data, SessionResumedData)
    messages = resumed.conversation_messages()
    assert [(message.role, message.content) for message in messages] == [
        (MessageRole.USER, "question"),
        (MessageRole.ASSISTANT, "answer"),
    ]
    resumed.close()


def test_resume_records_truncated_tail_recovery(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    session.close()
    with session.paths.events.open("ab") as event_file:
        event_file.write(b'{"partial":')

    resumed = AgentSession.resume(
        data_dir=tmp_path / "data",
        session_id=session.metadata.session_id,
    )

    resume_event = resumed.events[-1].data
    assert isinstance(resume_event, SessionResumedData)
    assert resume_event.recovered_tail_byte_count == len(b'{"partial":')
    assert resume_event.recovery_backup_name is not None
    resumed.close()


@pytest.mark.parametrize("corruption", ["index", "content_hash"])
def test_resume_rejects_invalid_committed_checkpoint_before_resume_event(
    tmp_path: Path,
    corruption: str,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id = session.new_turn_id()
    user_event = session.append_user_message("checkpoint source", turn_id=turn_id)
    checkpoint_id = "c" * 32
    session.append_checkpoint_started(
        checkpoint_id=checkpoint_id,
        source_from_event_id=user_event.id,
        source_through_event_id=user_event.id,
    )
    store = CheckpointStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    record = store.stage(
        Checkpoint(
            checkpoint_id=checkpoint_id,
            session_id=session.metadata.session_id,
            cycle_id=1,
            version=1,
            source_from_event_id=user_event.id,
            source_through_event_id=user_event.id,
            current_intent="Continue the active task",
            writer_model="test",
            created_at=datetime.now(UTC),
        )
    )
    session.append_checkpoint_committed(
        checkpoint_id=record.checkpoint_id,
        version=record.version,
        source_through_event_id=record.source_through_event_id,
        content_hash=record.content_hash,
    )
    session_id = session.metadata.session_id
    session.close()

    if corruption == "index":
        (session.paths.root / "checkpoints" / "index.json").write_text(
            "not-json",
            encoding="utf-8",
        )
    else:
        checkpoint_path = session.paths.root / "checkpoints" / record.relative_storage_path
        checkpoint_path.write_text("{}", encoding="utf-8")
    events_before = session.paths.events.read_bytes()

    with pytest.raises(CheckpointStoreCorruptionError):
        AgentSession.resume(data_dir=tmp_path / "data", session_id=session_id)

    assert session.paths.events.read_bytes() == events_before


def test_load_and_resume_reject_overlapping_checkpoint_starts_without_writes(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    source_event = session.append_user_message("checkpoint source", turn_id=session.new_turn_id())
    session.append_checkpoint_started(
        checkpoint_id="a" * 32,
        source_from_event_id=source_event.id,
        source_through_event_id=source_event.id,
    )
    session.append_checkpoint_started(
        checkpoint_id="b" * 32,
        source_from_event_id=source_event.id,
        source_through_event_id=source_event.id,
    )
    session_id = session.metadata.session_id
    session.close()
    events_before = session.paths.events.read_bytes()

    with pytest.raises(EventStoreCorruptionError, match="overlaps an active checkpoint"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session_id)
    assert session.paths.events.read_bytes() == events_before

    with pytest.raises(EventStoreCorruptionError, match="overlaps an active checkpoint"):
        AgentSession.resume(data_dir=tmp_path / "data", session_id=session_id)
    assert session.paths.events.read_bytes() == events_before


def test_load_is_read_only(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    original_event_count = len(session.events)

    loaded = AgentSession.load(
        data_dir=tmp_path / "data",
        session_id=session.metadata.session_id,
    )

    assert len(loaded.events) == original_event_count


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (("workspace", "/other-workspace"), ("model", "other-model")),
)
def test_load_and_resume_reject_metadata_that_disagrees_with_session_started(
    tmp_path: Path,
    field_name: str,
    replacement: str,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    session.close()

    metadata = json.loads(session.paths.metadata.read_text(encoding="utf-8"))
    metadata[field_name] = replacement
    session.paths.metadata.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(EventStoreCorruptionError, match="metadata does not match"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)
    with pytest.raises(EventStoreCorruptionError, match="metadata does not match"):
        AgentSession.resume(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


@pytest.mark.parametrize(
    ("workspace_relative_path", "model"),
    (
        ("workspace/apikey=synthetic-workspace-credential", "ordinary-model"),
        ("workspace", "api_key=synthetic-model-credential"),
    ),
)
def test_create_rejects_credential_shaped_metadata_before_creating_session_paths(
    tmp_path: Path,
    workspace_relative_path: str,
    model: str,
) -> None:
    workspace = tmp_path / workspace_relative_path
    workspace.mkdir(parents=True)
    data_dir = tmp_path / "data"

    with pytest.raises(ValueError, match="credential-shaped content"):
        AgentSession.create(data_dir=data_dir, workspace=workspace, model=model)

    assert not (data_dir / "sessions").exists()


@pytest.mark.parametrize("workspace_kind", ("missing", "file"))
def test_create_rejects_invalid_workspace_before_creating_session_paths(
    tmp_path: Path,
    workspace_kind: str,
) -> None:
    data_dir = tmp_path / "data"
    workspace = tmp_path / "api_key=synthetic-workspace-value"
    if workspace_kind == "file":
        workspace.write_text("not a workspace", encoding="utf-8")

    with pytest.raises(ValueError, match="Workspace must be an existing directory") as error:
        AgentSession.create(data_dir=data_dir, workspace=workspace, model="m")

    assert "synthetic-workspace-value" not in str(error.value)
    assert not data_dir.exists()


def test_create_rejects_sessions_symlink_before_writing_outside_data_dir(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    outside = tmp_path / "outside"
    data_dir.mkdir()
    outside.mkdir()
    (data_dir / "sessions").symlink_to(outside, target_is_directory=True)

    with pytest.raises(EventStoreCorruptionError, match="unsafe or unreadable"):
        AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="m")

    assert tuple(outside.iterdir()) == ()


def test_create_allows_ordinary_workspace_and_model_identifiers(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace" / "token_count=12"
    workspace.mkdir(parents=True)

    session = AgentSession.create(
        data_dir=tmp_path / "data",
        workspace=workspace,
        model="gpt-5.6-mini",
    )

    assert session.metadata.workspace == str(workspace)
    assert session.metadata.model == "gpt-5.6-mini"
    session.close()


def test_load_rejects_tampered_credential_shaped_metadata_before_using_it(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    session.close()
    metadata = json.loads(session.paths.metadata.read_text(encoding="utf-8"))
    metadata["workspace"] = "/safe/apikey=synthetic-tampered-credential"
    session.paths.metadata.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(EventStoreCorruptionError, match="credential-shaped content"):
        AgentSession.peek_metadata(
            data_dir=tmp_path / "data",
            session_id=session.metadata.session_id,
        )
    with pytest.raises(EventStoreCorruptionError, match="credential-shaped content"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)
    discovery = discover_sessions(tmp_path / "data")
    assert discovery.summaries == ()
    assert discovery.unreadable_session_ids == (session.metadata.session_id,)


def test_peek_metadata_wraps_invalid_metadata_without_echoing_contents(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    session.close()
    session.paths.metadata.write_text(
        '{"workspace":"' + _SYNTHETIC_METADATA_SECRET,
        encoding="utf-8",
    )

    with pytest.raises(EventStoreCorruptionError) as error:
        AgentSession.peek_metadata(
            data_dir=tmp_path / "data",
            session_id=session.metadata.session_id,
        )

    assert "Session metadata is unreadable or invalid" in str(error.value)
    assert _SYNTHETIC_METADATA_SECRET not in str(error.value)


def test_list_sessions_returns_latest_first(tmp_path: Path) -> None:
    older = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="older")
    newer = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="newer")
    older.append_user_message("most recently active", turn_id=older.new_turn_id())
    older.append_user_message("continue the implementation", turn_id=older.new_turn_id())

    summaries = list_sessions(tmp_path / "data")

    assert [summary.session_id for summary in summaries] == [
        older.metadata.session_id,
        newer.metadata.session_id,
    ]
    assert summaries[0].event_count == 3
    assert summaries[0].user_message_count == 2
    assert summaries[0].last_user_message == "continue the implementation"
    assert summaries[0].has_work is True
    assert summaries[1].user_message_count == 0
    assert summaries[1].last_user_message is None
    assert summaries[1].has_work is False
    assert summaries[0].last_event_at > summaries[0].created_at


def test_discover_sessions_isolates_an_unreadable_session(tmp_path: Path) -> None:
    good = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="good")
    bad = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="bad")
    good.append_user_message("keep this session", turn_id=good.new_turn_id())
    good.close()
    bad.close()
    with bad.paths.events.open("ab") as event_file:
        event_file.write(b'{"truncated":')

    discovery = discover_sessions(tmp_path / "data")

    assert [summary.session_id for summary in discovery.summaries] == [
        good.metadata.session_id
    ]
    assert discovery.unreadable_session_ids == (bad.metadata.session_id,)
    assert list_sessions(tmp_path / "data") == discovery.summaries


def test_assistant_event_keeps_finish_reason(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id = session.new_turn_id()
    session.append_model_request_started(turn_id=turn_id, message_count=1, attempt=1)
    event = session.append_assistant_message(
        "partial",
        finish_reason=FinishReason.LENGTH,
        turn_id=turn_id,
    )

    assert isinstance(event.data, AssistantMessageData)
    assert event.data.finish_reason is FinishReason.LENGTH


def test_resume_records_an_interrupted_model_request(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id = session.new_turn_id()
    session.append_model_request_started(turn_id=turn_id, message_count=1, attempt=1)
    session.close()

    resumed = AgentSession.resume(
        data_dir=tmp_path / "data",
        session_id=session.metadata.session_id,
    )

    terminal = resumed.events[-2]
    assert terminal.turn_id == turn_id
    assert isinstance(terminal.data, ModelRequestFailedData)
    assert terminal.data.error_code == "model_request_interrupted"
    assert terminal.data.is_retryable is False
    assert terminal.data.attempt == 1
    resumed.close()


def test_load_rejects_overlapping_model_request_starts(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id = session.new_turn_id()
    session.append_model_request_started(turn_id=turn_id, message_count=1, attempt=1)
    session.append_model_request_started(turn_id=turn_id, message_count=1, attempt=2)
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="previous request"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_load_rejects_model_failure_with_mismatched_attempt(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id = session.new_turn_id()
    session.append_model_request_started(turn_id=turn_id, message_count=1, attempt=1)
    session.store.append(
        session_id=session.metadata.session_id,
        turn_id=turn_id,
        data=ModelRequestFailedData(
            error_code="synthetic_failure",
            message="Synthetic provider failure",
            is_retryable=False,
            attempt=2,
        ),
    )
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="does not match"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_load_rejects_orphan_text_assistant_response(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    session.append_assistant_message(
        "orphan response",
        finish_reason=FinishReason.STOP,
        turn_id=session.new_turn_id(),
    )
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="no active model request"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_load_rejects_orphan_tool_call_assistant_response(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    tool_call = ToolCall(id="call-orphan", name="read_file", arguments_json='{"path":"x"}')
    session.append_assistant_message(
        None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(tool_call,),
        turn_id=session.new_turn_id(),
    )
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="no active model request"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_load_rejects_assistant_that_does_not_match_the_active_request(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    active_turn_id = session.new_turn_id()
    session.append_model_request_started(turn_id=active_turn_id, message_count=1, attempt=1)
    session.append_assistant_message(
        "wrong request",
        finish_reason=FinishReason.STOP,
        turn_id=session.new_turn_id(),
    )
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="does not match its active"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_resume_marks_incomplete_read_only_tool_as_interrupted(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    tool_call = ToolCall(id="call-read", name="read_file", arguments_json='{"path":"x"}')
    turn_id = session.new_turn_id()
    session.append_model_request_started(turn_id=turn_id, message_count=1, attempt=1)
    session.append_assistant_message(
        None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(tool_call,),
        turn_id=turn_id,
    )
    session.append_tool_requested(tool_call, is_read_only=True, turn_id=turn_id)
    session.append_tool_started(tool_call, turn_id=turn_id)
    session.close()

    resumed = AgentSession.resume(
        data_dir=tmp_path / "data",
        session_id=session.metadata.session_id,
    )

    interrupted = resumed.events[-2].data
    assert isinstance(interrupted, ToolInterruptedData)
    assert interrupted.recovery_status is ToolRecoveryStatus.INTERRUPTED
    assert resumed.events[-2].correlation_id == tool_call.id
    resumed.close()


def test_resume_marks_incomplete_side_effect_tool_as_unknown(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    tool_call = ToolCall(id="call-write", name="apply_patch", arguments_json='{"patch":"x"}')
    turn_id = session.new_turn_id()
    session.append_model_request_started(turn_id=turn_id, message_count=1, attempt=1)
    session.append_assistant_message(
        None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(tool_call,),
        turn_id=turn_id,
    )
    session.append_tool_requested(tool_call, is_read_only=False, turn_id=turn_id)
    scope = recompute_approval_scope(
        tool_name=tool_call.name,
        is_read_only=False,
        arguments_json=tool_call.arguments_json,
    )
    assert scope is not None
    session.append_approval_requested(
            tool_call=tool_call,
            redacted_arguments=scope.redacted_arguments,
            scope_descriptor=scope.scope_descriptor,
            scope_fingerprint=scope.scope_fingerprint,
        can_allow_session=scope.can_allow_session,
        turn_id=turn_id,
    )
    session.append_approval_resolved(
        tool_call=tool_call,
        scope_fingerprint=scope.scope_fingerprint,
        decision=ApprovalDecision.ALLOW_ONCE,
        turn_id=turn_id,
    )
    session.append_tool_started(tool_call, turn_id=turn_id)
    session.close()

    resumed = AgentSession.resume(
        data_dir=tmp_path / "data",
        session_id=session.metadata.session_id,
    )

    interrupted = resumed.events[-2].data
    assert isinstance(interrupted, ToolInterruptedData)
    assert interrupted.recovery_status is ToolRecoveryStatus.UNKNOWN
    assert "must not be replayed automatically" in interrupted.reason
    recovery_messages = resumed.conversation_messages()
    assert recovery_messages[-1].role is MessageRole.TOOL
    assert recovery_messages[-1].tool_call_id == tool_call.id
    assert "unknown" in (recovery_messages[-1].content or "")
    resumed.close()


def test_completed_tool_result_reconstructs_a_tool_message(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    tool_call = ToolCall(id="call-read", name="read_file", arguments_json='{"path":"x"}')
    turn_id = session.new_turn_id()
    session.append_model_request_started(turn_id=turn_id, message_count=1, attempt=1)
    session.append_assistant_message(
        None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(tool_call,),
        turn_id=turn_id,
    )
    session.append_tool_requested(tool_call, is_read_only=True, turn_id=turn_id)
    session.append_tool_started(tool_call, turn_id=turn_id)
    session.append_tool_completed(tool_call, "file contents", turn_id=turn_id)

    messages = session.conversation_messages()

    assert messages[-2].role is MessageRole.ASSISTANT
    assert messages[-2].tool_calls == (tool_call,)
    assert messages[-1].role is MessageRole.TOOL
    assert messages[-1].tool_call_id == tool_call.id
    assert messages[-1].content == "file contents"
    session.close()


def test_resume_recovers_assistant_tool_call_after_request_event(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    tool_call = ToolCall(id="call-gap", name="read_file", arguments_json='{"path":"x"}')
    turn_id = session.new_turn_id()
    session.append_model_request_started(turn_id=turn_id, message_count=1, attempt=1)
    session.append_assistant_message(
        None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(tool_call,),
        turn_id=turn_id,
    )
    session.close()

    resumed = AgentSession.resume(
        data_dir=tmp_path / "data",
        session_id=session.metadata.session_id,
    )

    interrupted = resumed.events[-2].data
    assert isinstance(interrupted, ToolInterruptedData)
    assert interrupted.recovery_status is ToolRecoveryStatus.INTERRUPTED
    messages = resumed.conversation_messages()
    assert messages[-2].tool_calls == (tool_call,)
    assert messages[-1].role is MessageRole.TOOL
    resumed.close()


def test_load_rejects_tool_request_without_assistant_call(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    tool_call = ToolCall(id="call-orphan", name="read_file", arguments_json='{"path":"x"}')
    session.append_tool_requested(tool_call, is_read_only=True, turn_id=session.new_turn_id())
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="no assistant call"):
        AgentSession.load(
            data_dir=tmp_path / "data",
            session_id=session.metadata.session_id,
        )


def test_append_assistant_rejects_reused_tool_call_id_before_persistence(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    first = ToolCall(id="call-1", name="read_file", arguments_json='{"path":"a"}')
    turn_id = session.new_turn_id()
    session.append_model_request_started(turn_id=turn_id, message_count=1, attempt=1)
    session.append_assistant_message(
        None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(first,),
        turn_id=turn_id,
    )
    session.append_tool_interrupted(
        first,
        recovery_status=ToolRecoveryStatus.INTERRUPTED,
        reason="not started",
        turn_id=turn_id,
    )
    event_count_before_reuse = len(session.events)
    reused = ToolCall(id="call-1", name="read_file", arguments_json='{"path":"b"}')

    with pytest.raises(ValueError, match="already used"):
        session.append_assistant_message(
            None,
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(reused,),
            turn_id=session.new_turn_id(),
        )

    assert len(session.events) == event_count_before_reuse
    session.close()


def test_load_rejects_duplicate_artifact_registration(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id, tool_call, started_event = _start_read_tool(session)
    artifact = _artifact_created(started_event.id)
    session.store.append(
        session_id=session.metadata.session_id,
        turn_id=turn_id,
        data=artifact,
    )
    session.store.append(
        session_id=session.metadata.session_id,
        turn_id=turn_id,
        data=artifact,
    )
    session.append_tool_completed(tool_call, "done", turn_id=turn_id)
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="Duplicate artifact registration"):
        AgentSession.load(
            data_dir=tmp_path / "data",
            session_id=session.metadata.session_id,
        )


def test_load_rejects_artifact_whose_source_is_not_tool_started(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id, tool_call, started_event = _start_read_tool(session)
    session.store.append(
        session_id=session.metadata.session_id,
        turn_id=turn_id,
        data=_artifact_created(started_event.id - 1),
    )
    session.append_tool_completed(tool_call, "done", turn_id=turn_id)
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="not a prior tool start"):
        AgentSession.load(
            data_dir=tmp_path / "data",
            session_id=session.metadata.session_id,
        )


def test_load_rejects_artifact_after_tool_completion(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id, tool_call, started_event = _start_read_tool(session)
    session.append_tool_completed(tool_call, "done", turn_id=turn_id)
    session.store.append(
        session_id=session.metadata.session_id,
        turn_id=turn_id,
        data=_artifact_created(started_event.id),
    )
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="after its tool terminal event"):
        AgentSession.load(
            data_dir=tmp_path / "data",
            session_id=session.metadata.session_id,
        )


def test_load_rejects_artifact_from_a_different_turn(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id, tool_call, started_event = _start_read_tool(session)
    session.store.append(
        session_id=session.metadata.session_id,
        turn_id=session.new_turn_id(),
        data=_artifact_created(started_event.id),
    )
    session.append_tool_completed(tool_call, "done", turn_id=turn_id)
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="different turn"):
        AgentSession.load(
            data_dir=tmp_path / "data",
            session_id=session.metadata.session_id,
        )


def test_load_rejects_artifact_created_but_inline_completion(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id, tool_call, started_event = _start_read_tool(session)
    session.store.append(
        session_id=session.metadata.session_id,
        turn_id=turn_id,
        data=_artifact_created(started_event.id),
    )
    session.append_tool_completed(tool_call, "artifact result", turn_id=turn_id)
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="completed inline"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_load_rejects_completion_with_unknown_artifact_id(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id, tool_call, _ = _start_read_tool(session)
    unknown_artifact_id = "c" * 32
    session.append_tool_completed(
        tool_call,
        f"Artifact {unknown_artifact_id}",
        artifact_id=unknown_artifact_id,
        turn_id=turn_id,
    )
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="unknown artifact"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_load_rejects_completion_that_consumes_another_tools_artifact(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    first_turn_id, _, first_started = _start_read_tool(session)
    artifact = _artifact_created(first_started.id)
    session.store.append(
        session_id=session.metadata.session_id,
        turn_id=first_turn_id,
        data=artifact,
    )
    second_call = ToolCall(id="call-second", name="read_file", arguments_json='{"path":"y"}')
    session.append_model_request_started(turn_id=first_turn_id, message_count=1, attempt=2)
    session.append_assistant_message(
        None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(second_call,),
        turn_id=first_turn_id,
    )
    session.append_tool_requested(second_call, is_read_only=True, turn_id=first_turn_id)
    session.append_tool_started(second_call, turn_id=first_turn_id)
    session.append_tool_completed(
        second_call,
        f"Artifact {artifact.artifact_id}",
        artifact_id=artifact.artifact_id,
        turn_id=first_turn_id,
    )
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="belongs to another tool"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_load_rejects_completion_without_artifact_reference_in_output(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id, tool_call, started_event = _start_read_tool(session)
    artifact = _artifact_created(started_event.id)
    session.store.append(
        session_id=session.metadata.session_id,
        turn_id=turn_id,
        data=artifact,
    )
    session.append_tool_completed(
        tool_call,
        "tool output omitted the artifact reference",
        artifact_id=artifact.artifact_id,
        turn_id=turn_id,
    )
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="output omits artifact reference"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_load_accepts_matching_artifact_completion(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id, tool_call, started_event = _start_read_tool(session)
    artifact = _artifact_created(started_event.id)
    session.store.append(
        session_id=session.metadata.session_id,
        turn_id=turn_id,
        data=artifact,
    )
    session.append_tool_completed(
        tool_call,
        f"Artifact {artifact.artifact_id}",
        artifact_id=artifact.artifact_id,
        turn_id=turn_id,
    )
    session.close()

    loaded = AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)
    completion = loaded.events[-1].data
    assert completion.kind == "tool_completed"
    assert completion.artifact_id == artifact.artifact_id
    loaded.close()


def test_load_and_resume_accept_published_m1_artifact_events_without_rewriting_history(
    tmp_path: Path,
) -> None:
    """M1 had no event-level redaction kinds or explicit completion linkage."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=workspace, model="m")
    turn_id, tool_call, started_event = _start_read_tool(session)
    artifact_store = ArtifactStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    record = artifact_store.create(
        "before\napi_key=legacy-fixture-value\nafter",
        source_event_id=started_event.id,
    )
    assert record.redaction_match_count > 0
    session.append_artifact_created(record, turn_id=turn_id)
    session.append_tool_completed(
        tool_call,
        f"Artifact {record.artifact_id} contains the full captured result.",
        artifact_id=record.artifact_id,
        turn_id=turn_id,
    )
    session.close()

    # These are the exact fields emitted by the published M1 event models.
    raw_lines = session.paths.events.read_text(encoding="utf-8").splitlines()
    m1_events = [json.loads(line) for line in raw_lines]
    for event in m1_events:
        event["schema_version"] = 1
        event.pop("redaction_summary", None)
        data = event["data"]
        if data["kind"] == "artifact_created":
            data.pop("redaction_kinds", None)
            data.pop("legacy_redaction_kinds_unknown", None)
        elif data["kind"] == "tool_completed":
            data.pop("artifact_id", None)
            data.pop("source_redaction_summary", None)
            data.pop("facts", None)
    m1_serialized = "".join(f"{json.dumps(event)}\n" for event in m1_events)
    session.paths.events.write_text(m1_serialized, encoding="utf-8")

    loaded = AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)
    completion = loaded.events[-1].data
    registration = loaded.artifact_registrations[0]
    assert isinstance(completion, ToolCompletedData)
    assert completion.artifact_id == record.artifact_id
    assert completion.facts == ToolCompletionFacts()
    assert completion.legacy_facts_missing is True
    assert registration.legacy_redaction_kinds_unknown is True
    assert registration.redaction_match_count == record.redaction_match_count
    assert (loaded.paths.events.read_text(encoding="utf-8")) == m1_serialized
    ArtifactStore.open(
        loaded.paths.root,
        registrations=loaded.artifact_registrations,
        workspace_root=Path(loaded.metadata.workspace),
    )
    loaded.close()

    resumed = AgentSession.resume(
        data_dir=tmp_path / "data", session_id=session.metadata.session_id
    )
    assert resumed.events[-1].data.kind == "session_resumed"
    assert resumed.events[-2].data.kind == "tool_completed"
    assert resumed.events[-2].data.artifact_id == record.artifact_id
    assert resumed.paths.events.read_text(encoding="utf-8").startswith(m1_serialized)
    resumed.close()


def test_current_v2_event_missing_artifact_fields_is_not_migrated(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id, tool_call, started_event = _start_read_tool(session)
    artifact_store = ArtifactStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    record = artifact_store.create(
        "api_key=current-event-value",
        source_event_id=started_event.id,
    )
    session.append_artifact_created(record, turn_id=turn_id)
    session.append_tool_completed(
        tool_call,
        f"Artifact {record.artifact_id}",
        artifact_id=record.artifact_id,
        turn_id=turn_id,
    )
    session.close()

    event_records = [
        json.loads(line)
        for line in session.paths.events.read_text(encoding="utf-8").splitlines()
    ]
    for event_record in event_records:
        if event_record["data"]["kind"] == "artifact_created":
            event_record["data"].pop("redaction_kinds")
        elif event_record["data"]["kind"] == "tool_completed":
            event_record["data"].pop("artifact_id")
    session.paths.events.write_text(
        "".join(f"{json.dumps(event_record)}\n" for event_record in event_records),
        encoding="utf-8",
    )

    with pytest.raises(EventStoreCorruptionError, match="Invalid event record"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_current_m2_event_missing_tool_facts_is_not_migrated(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id, tool_call, _started_event = _start_read_tool(session)
    session.append_tool_completed(tool_call, "ordinary output", turn_id=turn_id)
    session.close()

    event_records = [
        json.loads(line)
        for line in session.paths.events.read_text(encoding="utf-8").splitlines()
    ]
    for event_record in event_records:
        if event_record["data"]["kind"] == "tool_completed":
            event_record["data"].pop("facts")
    session.paths.events.write_text(
        "".join(f"{json.dumps(event_record)}\n" for event_record in event_records),
        encoding="utf-8",
    )

    with pytest.raises(EventStoreCorruptionError, match="Invalid event record"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_transcript_validator_rejects_artifact_completion_with_duplicate_redaction_summary(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id, tool_call, started_event = _start_read_tool(session)
    artifact = _artifact_created(started_event.id)
    session.store.append(
        session_id=session.metadata.session_id,
        turn_id=turn_id,
        data=artifact,
    )
    completion_event = session.append_tool_completed(
        tool_call,
        f"Artifact {artifact.artifact_id}",
        artifact_id=artifact.artifact_id,
        turn_id=turn_id,
    )
    summary = RedactionSummary(
        match_count=1,
        kinds=(RedactionKind.NAMED_SECRET,),
    )
    tampered_completion = ToolCompletedData.model_construct(
        kind="tool_completed",
        tool_call_id=tool_call.id,
        output=f"Artifact {artifact.artifact_id}",
        artifact_id=artifact.artifact_id,
        source_redaction_summary=summary,
        facts=ToolCompletionFacts(),
    )
    tampered_event = completion_event.model_copy(update={"data": tampered_completion})

    with pytest.raises(EventStoreCorruptionError, match="duplicated redaction metadata"):
        validate_session_events(
            (*session.events[:-1], tampered_event),
            session.metadata.session_id,
        )

    session.close()


def test_load_rejects_tampered_artifact_completion_redaction_summary(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id, tool_call, started_event = _start_read_tool(session)
    artifact = _artifact_created(started_event.id)
    session.store.append(
        session_id=session.metadata.session_id,
        turn_id=turn_id,
        data=artifact,
    )
    session.append_tool_completed(
        tool_call,
        f"Artifact {artifact.artifact_id}",
        artifact_id=artifact.artifact_id,
        turn_id=turn_id,
    )
    session.close()

    event_records = [
        json.loads(line)
        for line in session.paths.events.read_text(encoding="utf-8").splitlines()
    ]
    for event_record in event_records:
        if event_record["data"]["kind"] == "tool_completed":
            event_record["data"]["source_redaction_summary"] = {
                "match_count": 1,
                "kinds": [RedactionKind.NAMED_SECRET.value],
            }
    session.paths.events.write_text(
        "".join(f"{json.dumps(event_record)}\n" for event_record in event_records),
        encoding="utf-8",
    )

    with pytest.raises(EventStoreCorruptionError, match="Invalid event record"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_load_rejects_repeated_artifact_completion(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id, tool_call, started_event = _start_read_tool(session)
    artifact = _artifact_created(started_event.id)
    session.store.append(
        session_id=session.metadata.session_id,
        turn_id=turn_id,
        data=artifact,
    )
    for _ in range(2):
        session.append_tool_completed(
            tool_call,
            f"Artifact {artifact.artifact_id}",
            artifact_id=artifact.artifact_id,
            turn_id=turn_id,
        )
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="Invalid tool terminal event"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_resume_preserves_an_artifact_created_before_crash(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id, _, started_event = _start_read_tool(session)
    artifact_store = ArtifactStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    record = artifact_store.create("artifact", source_event_id=started_event.id)
    session.append_artifact_created(record, turn_id=turn_id)
    session.close()

    resumed = AgentSession.resume(
        data_dir=tmp_path / "data",
        session_id=session.metadata.session_id,
    )

    assert isinstance(resumed.events[-2].data, ToolInterruptedData)
    recovered_store = ArtifactStore.open(
        resumed.paths.root,
        registrations=resumed.artifact_registrations,
        workspace_root=Path(resumed.metadata.workspace),
    )
    assert recovered_store.records == (record,)
    resumed.close()


def test_committed_artifact_event_survives_fsync_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id, _, started_event = _start_read_tool(session)
    artifact_store = ArtifactStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    record = artifact_store.create("artifact", source_event_id=started_event.id)
    real_fsync = os.fsync

    def fsync_then_raise(file_descriptor: int) -> None:
        real_fsync(file_descriptor)
        raise OSError("event fsync outcome is unknown")

    monkeypatch.setattr("mini_agent.event_store.os.fsync", fsync_then_raise)
    with pytest.raises(OSError, match="outcome is unknown"):
        session.append_artifact_created(record, turn_id=turn_id)
    monkeypatch.undo()
    session.close()

    loaded = AgentSession.load(
        data_dir=tmp_path / "data",
        session_id=session.metadata.session_id,
    )
    recovered_store = ArtifactStore.open(
        loaded.paths.root,
        registrations=loaded.artifact_registrations,
        workspace_root=Path(loaded.metadata.workspace),
    )

    assert loaded.artifact_registrations[0].artifact_id == record.artifact_id
    assert recovered_store.records == (record,)
    assert recovered_store.read(record.artifact_id, offset=0, limit=20) == "artifact"
    loaded.close()


def test_committed_artifact_completion_survives_fsync_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    turn_id, tool_call, started_event = _start_read_tool(session)
    artifact_store = ArtifactStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    record = artifact_store.create("artifact", source_event_id=started_event.id)
    session.append_artifact_created(record, turn_id=turn_id)
    real_fsync = os.fsync

    def fsync_then_raise(file_descriptor: int) -> None:
        real_fsync(file_descriptor)
        raise OSError("completion fsync outcome is unknown")

    monkeypatch.setattr("mini_agent.event_store.os.fsync", fsync_then_raise)
    with pytest.raises(OSError, match="completion fsync outcome is unknown"):
        session.append_tool_completed(
            tool_call,
            f"Artifact {record.artifact_id}",
            artifact_id=record.artifact_id,
            turn_id=turn_id,
        )
    monkeypatch.undo()
    session.close()

    loaded = AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)
    completion = loaded.events[-1].data
    assert completion.kind == "tool_completed"
    assert completion.artifact_id == record.artifact_id
    loaded.close()
