import json
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

import mini_agent.event_store as event_store_module
from mini_agent.event_redaction import redact_event_data
from mini_agent.event_store import (
    EventStore,
    EventStoreCorruptionError,
    EventStoreWriterBusyError,
)
from mini_agent.events import (
    ApprovalRequestedData,
    ArtifactCreatedData,
    AssistantMessageData,
    ContextEstimatedData,
    ModelRequestFailedData,
    RebuildCompletedData,
    SessionStartedData,
    StoredEvent,
    ToolCompletedData,
    ToolRequestedData,
    ToolStartedData,
    UserMessageData,
)
from mini_agent.messages import FinishReason, ToolCall
from mini_agent.redaction_types import RedactionKind, RedactionSummary
from mini_agent.tool_facts import ToolCompletionFacts

_STRIPE_LIVE_KEY = "sk_live_" + "0" * 24
_STRIPE_RESTRICTED_LIVE_KEY = "rk_live_" + "0" * 24
_STRIPE_RESTRICTED_TEST_KEY = "rk_test_" + "0" * 24
_STRIPE_WEBHOOK_SECRET = "whsec_" + "0" * 24
_AWS_ACCESS_KEY_ID = "AKIA" + "0" * 16
_COMPACT_SENSITIVE_FIELD_NAMES = (
    "apikey",
    "accesstoken",
    "secretaccesskey",
    "refreshtoken",
    "privatekey",
    "connectionstring",
    "databaseurl",
    "secretkey",
)


def test_append_reopens_typed_events(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    store.append(session_id="session", data=SessionStartedData(workspace="/repo", model="m"))
    store.append(session_id="session", data=UserMessageData(content="hello"), turn_id="t1")
    store.close()

    persisted_records = [
        json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["schema_version"] for record in persisted_records] == [2, 2]

    reopened_store = EventStore.open(event_path)

    assert [event.id for event in reopened_store.events] == [1, 2]
    assert [event.schema_version for event in reopened_store.events] == [2, 2]
    assert isinstance(reopened_store.events[1].data, UserMessageData)
    assert reopened_store.events[1].data.content == "hello"
    reopened_store.close()


def test_event_store_redacts_host_paths_in_user_message_when_workspace_root_given(
    tmp_path: Path,
) -> None:
    # When the store knows the workspace root, free-text fields are persisted
    # under the durable-text policy so host paths do not reach the on-disk
    # transcript. (session_started.workspace keeps credential-only redaction so
    # resume can still match it against metadata.)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path, workspace_root=workspace)
    store.append(
        session_id="session",
        data=UserMessageData(content=f"review {workspace}/src/secret.py please"),
    )
    store.close()

    reopened = EventStore.open(event_path)
    persisted = reopened.events[0].data
    reopened.close()

    assert isinstance(persisted, UserMessageData)
    assert "<workspace-root>/src/secret.py" in persisted.content
    assert str(workspace) not in persisted.content
    assert "please" in persisted.content


def test_append_without_cycle_id_inherits_the_latest_persisted_cycle(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    store.append(
        session_id="session",
        data=SessionStartedData(workspace="/repo", model="m"),
    )
    store.append(
        session_id="session",
        cycle_id=2,
        data=UserMessageData(content="new cycle"),
    )

    inherited = store.append(
        session_id="session",
        data=UserMessageData(content="same cycle"),
    )

    assert inherited.cycle_id == 2
    with pytest.raises(ValueError, match="stay current"):
        store.append(
            session_id="session",
            cycle_id=1,
            data=UserMessageData(content="regression"),
        )
    with pytest.raises(ValueError, match="advance by one"):
        store.append(
            session_id="session",
            cycle_id=4,
            data=UserMessageData(content="gap"),
        )
    store.close()


def test_append_redacts_credentials_before_persistence(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)

    store.append(
        session_id="session",
        data=UserMessageData(content="api_key=example-secret-value"),
    )
    store.close()

    persisted = event_path.read_text(encoding="utf-8")
    assert "example-secret-value" not in persisted
    assert "[REDACTED]" in persisted


def test_append_redacts_sensitive_http_query_values_in_user_and_tool_output(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    unsafe_user_value = "synthetic-user-url-secret"
    unsafe_tool_value = "synthetic-tool-url-secret"

    user_event = store.append(
        session_id="session",
        data=UserMessageData(
            content=f"https://example.test/v1?api_key={unsafe_user_value}&token_count=12"
        ),
    )
    tool_event = store.append(
        session_id="session",
        correlation_id="call-url-output",
        data=ToolCompletedData(
            tool_call_id="call-url-output",
            output=f"https://example.test/v1#access_token={unsafe_tool_value}",
            facts=ToolCompletionFacts(),
        ),
    )
    store.close()

    persisted = event_path.read_text(encoding="utf-8")
    assert unsafe_user_value not in persisted
    assert unsafe_tool_value not in persisted
    assert isinstance(user_event.data, UserMessageData)
    assert isinstance(tool_event.data, ToolCompletedData)
    assert user_event.data.content == "https://example.test/v1?api_key=[REDACTED]&token_count=12"
    assert tool_event.data.output == "https://example.test/v1#access_token=[REDACTED]"


def test_event_store_refuses_credential_shaped_correlation_without_persisting_it(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    credential_shaped_id = "sk_live_" + "0" * 24

    with pytest.raises(ValueError, match="credential-shaped"):
        store.append(
            session_id="session",
            correlation_id=credential_shaped_id,
            data=UserMessageData(content="ordinary"),
        )
    store.close()

    assert credential_shaped_id not in event_path.read_text(encoding="utf-8")


def test_event_data_refuses_credential_shaped_tool_reference_and_name() -> None:
    credential_shaped_identifier = "AKIA" + "0" * 16

    with pytest.raises(ValueError, match="credential-shaped"):
        ToolCompletedData(
            tool_call_id=credential_shaped_identifier,
            output="ordinary",
            facts=ToolCompletionFacts(),
        )
    with pytest.raises(ValueError, match="credential-shaped"):
        ToolStartedData(tool_call_id="call-1", tool_name=credential_shaped_identifier)


def test_event_redaction_summaries_are_typed_aggregate_only_and_round_trip(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    prefix_secret = "sk_live_" + "0" * 24
    event = store.append(
        session_id="session",
        data=UserMessageData(
            content=(
                "api_key=synthetic-value "
                "Authorization: Bearer synthetic-authorization-value "
                f"{prefix_secret}"
            )
        ),
    )
    marker_event = store.append(
        session_id="session",
        data=UserMessageData(content="The literal [REDACTED] marker is safe text."),
    )
    failure_event = store.append(
        session_id="session",
        data=ModelRequestFailedData(
            error_code="provider_error",
            message="token=synthetic-model-error-value",
            is_retryable=False,
            attempt=1,
        ),
    )
    store.close()

    assert event.redaction_summary.match_count == 3
    assert event.redaction_summary.kinds == (
        RedactionKind.AUTHORIZATION,
        RedactionKind.NAMED_SECRET,
        RedactionKind.SECRET_PREFIX,
    )
    assert marker_event.redaction_summary.match_count == 0
    assert marker_event.redaction_summary.kinds == ()
    assert failure_event.redaction_summary.match_count == 1
    assert failure_event.redaction_summary.kinds == (RedactionKind.NAMED_SECRET,)

    reloaded = EventStore.open(event_path)
    assert reloaded.events[0].redaction_summary == event.redaction_summary
    assert reloaded.events[1].redaction_summary == marker_event.redaction_summary
    reloaded.close()


def test_reopen_accepts_redacted_event_even_when_its_original_summary_is_nonempty(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    written = store.append(
        session_id="session",
        data=UserMessageData(content="api_key=synthetic-fixed-point-secret"),
    )
    store.close()

    reopened = EventStore.open(event_path)
    persisted = reopened.events[0]
    safe_data, residual_summary = redact_event_data(persisted.data)

    assert written.redaction_summary.match_count == 1
    assert persisted.redaction_summary == written.redaction_summary
    assert safe_data == persisted.data
    assert residual_summary == RedactionSummary()
    reopened.close()


@pytest.mark.parametrize(
    ("data", "label"),
    [
        (UserMessageData(content="api_key=synthetic-user-secret"), "user"),
        (
            AssistantMessageData(
                content="api_key=synthetic-assistant-secret",
                finish_reason=FinishReason.STOP,
            ),
            "assistant",
        ),
        (
            ToolRequestedData(
                tool_call=ToolCall(
                    id="call-fixed-point",
                    name="read_file",
                    arguments_json='{"api_key":"synthetic-tool-secret"}',
                ),
                is_read_only=True,
            ),
            "tool",
        ),
        (
            ModelRequestFailedData(
                error_code="provider_error",
                message="api_key=synthetic-error-secret",
                is_retryable=False,
                attempt=1,
            ),
            "error",
        ),
        (
            ApprovalRequestedData(
                tool_call_id="call-approval",
                tool_name="read_file",
                redacted_arguments="{}",
                scope_descriptor="api_key=synthetic-scope-secret",
                scope_fingerprint="a" * 64,
                can_allow_session=False,
            ),
            "approval_scope",
        ),
        (
            ContextEstimatedData(
                projection_version=1,
                strategy="api_key=synthetic-context-secret",
                input_limit=100,
                estimated_tokens=10,
                utilization_ratio=0.1,
            ),
            "context_strategy",
        ),
        (
            RebuildCompletedData(
                rebuild_id="b" * 32,
                projection_version=1,
                strategy="api_key=synthetic-rebuild-secret",
                projected_message_count=1,
                estimated_tokens=1,
                source_message_count=1,
            ),
            "rebuild_strategy",
        ),
    ],
    ids=(
        "user",
        "assistant",
        "tool",
        "error",
        "approval_scope",
        "context_strategy",
        "rebuild_strategy",
    ),
)
def test_open_rejects_current_event_with_residual_sensitive_data(
    tmp_path: Path,
    data: object,
    label: str,
) -> None:
    event_path = tmp_path / f"{label}.jsonl"
    event = StoredEvent(
        id=1,
        session_id="session",
        timestamp=datetime.now(UTC),
        data=data,  # pyright: ignore[reportArgumentType]
    )
    encoded = event.model_dump_json().encode("utf-8") + b"\n"
    event_path.write_bytes(encoded)

    with pytest.raises(EventStoreCorruptionError) as error:
        EventStore.open(event_path)

    assert "synthetic" not in str(error.value)
    assert "secret" not in str(error.value)
    assert str(event_path) not in str(error.value)


def test_open_rejects_legacy_event_with_residual_sensitive_data_without_echoing_it(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events.jsonl"
    secret = "synthetic-legacy-fixed-point-secret"
    legacy_event = StoredEvent(
        id=1,
        session_id="session",
        timestamp=datetime.now(UTC),
        data=UserMessageData(content=f"api_key={secret}"),
    ).model_dump(mode="json")
    legacy_event["schema_version"] = 1
    del legacy_event["redaction_summary"]
    event_path.write_text(json.dumps(legacy_event) + "\n", encoding="utf-8")

    with pytest.raises(EventStoreCorruptionError) as error:
        EventStore.open(event_path)

    assert secret not in str(error.value)


def test_open_accepts_safe_legacy_event(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    legacy_event = StoredEvent(
        id=1,
        session_id="session",
        timestamp=datetime.now(UTC),
        data=UserMessageData(content="safe legacy content"),
    ).model_dump(mode="json")
    legacy_event["schema_version"] = 1
    del legacy_event["redaction_summary"]
    event_path.write_text(json.dumps(legacy_event) + "\n", encoding="utf-8")

    store = EventStore.open(event_path)

    assert store.events[0].data == UserMessageData(content="safe legacy content")
    assert store.events[0].schema_version == 2
    store.close()


def test_open_migrates_real_v1_tool_completion_with_existing_summary_and_missing_facts(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events.jsonl"
    original_summary = RedactionSummary(
        match_count=1,
        kinds=(RedactionKind.NAMED_SECRET,),
    )
    v1_event = StoredEvent(
        id=1,
        session_id="session",
        timestamp=datetime.now(UTC),
        redaction_summary=original_summary,
        data=ToolCompletedData(
            tool_call_id="call-v1",
            output="api_key=[REDACTED]",
            facts=ToolCompletionFacts(),
        ),
    ).model_dump(mode="json")
    v1_event["schema_version"] = 1
    data = v1_event["data"]
    assert isinstance(data, dict)
    del data["facts"]
    event_path.write_text(json.dumps(v1_event) + "\n", encoding="utf-8")

    store = EventStore.open(event_path)
    completed = store.events[0]

    assert completed.schema_version == 2
    assert completed.redaction_summary == original_summary
    assert isinstance(completed.data, ToolCompletedData)
    assert completed.data.facts == ToolCompletionFacts()
    assert completed.data.legacy_facts_missing is True
    store.close()


def test_open_migrates_v1_missing_summary_and_preserves_existing_facts(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    expected_facts = ToolCompletionFacts(command_exit_code=7, modified_paths=("safe.txt",))
    v1_event = StoredEvent(
        id=1,
        session_id="session",
        timestamp=datetime.now(UTC),
        data=ToolCompletedData(
            tool_call_id="call-v1-facts",
            output="ordinary output",
            facts=expected_facts,
        ),
    ).model_dump(mode="json")
    v1_event["schema_version"] = 1
    del v1_event["redaction_summary"]
    event_path.write_text(json.dumps(v1_event) + "\n", encoding="utf-8")

    store = EventStore.open(event_path)
    completed = store.events[0]

    assert completed.schema_version == 2
    assert completed.redaction_summary == RedactionSummary()
    assert isinstance(completed.data, ToolCompletedData)
    assert completed.data.facts == expected_facts
    assert completed.data.legacy_facts_missing is False
    store.close()


def test_open_rejects_v2_tool_completion_missing_facts(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    v2_event = StoredEvent(
        id=1,
        session_id="session",
        timestamp=datetime.now(UTC),
        data=ToolCompletedData(
            tool_call_id="call-v2",
            output="ordinary output",
            facts=ToolCompletionFacts(),
        ),
    ).model_dump(mode="json")
    data = v2_event["data"]
    assert isinstance(data, dict)
    del data["facts"]
    event_path.write_text(json.dumps(v2_event) + "\n", encoding="utf-8")

    with pytest.raises(EventStoreCorruptionError, match="Invalid event record"):
        EventStore.open(event_path)


def test_open_rejects_v2_event_missing_redaction_summary(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    v2_event = StoredEvent(
        id=1,
        session_id="session",
        timestamp=datetime.now(UTC),
        data=UserMessageData(content="ordinary content"),
    ).model_dump(mode="json")
    del v2_event["redaction_summary"]
    event_path.write_text(json.dumps(v2_event) + "\n", encoding="utf-8")

    with pytest.raises(EventStoreCorruptionError, match="Invalid event record"):
        EventStore.open(event_path)


def test_open_rejects_unknown_event_schema_version_without_echoing_contents(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events.jsonl"
    v2_event = StoredEvent(
        id=1,
        session_id="session",
        timestamp=datetime.now(UTC),
        data=UserMessageData(content="ordinary content"),
    ).model_dump(mode="json")
    v2_event["schema_version"] = 99
    event_path.write_text(json.dumps(v2_event) + "\n", encoding="utf-8")

    with pytest.raises(EventStoreCorruptionError) as error:
        EventStore.open(event_path)

    assert "99" not in str(error.value)
    assert str(event_path) not in str(error.value)


def test_open_rejects_unknown_complete_schema_record_without_a_final_newline(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events.jsonl"
    v2_event = StoredEvent(
        id=1,
        session_id="session",
        timestamp=datetime.now(UTC),
        data=UserMessageData(content="ordinary content"),
    ).model_dump(mode="json")
    v2_event["schema_version"] = 99
    event_path.write_text(json.dumps(v2_event), encoding="utf-8")

    with pytest.raises(EventStoreCorruptionError, match="Invalid event record"):
        EventStore.open(event_path)

    assert event_path.read_text(encoding="utf-8") == json.dumps(v2_event)


@pytest.mark.parametrize("field_name", ("session_id", "turn_id", "correlation_id"))
def test_stored_event_rejects_credential_shaped_top_level_identifiers(
    field_name: str,
) -> None:
    credential_shaped_value = "sk_live_" + "0" * 24
    fields: dict[str, object] = {
        "id": 1,
        "session_id": "session",
        "timestamp": datetime.now(UTC),
        "data": UserMessageData(content="ordinary"),
    }
    fields[field_name] = credential_shaped_value

    with pytest.raises(ValueError) as error:
        StoredEvent.model_validate(fields)

    assert credential_shaped_value not in str(error.value)


def test_artifact_media_type_rejects_credential_shaped_value_without_echoing_it() -> None:
    credential_shaped_value = "sk_live_" + "0" * 24

    with pytest.raises(ValueError) as error:
        ArtifactCreatedData(
            artifact_id="a" * 32,
            source_event_id=1,
            media_type=f"{credential_shaped_value}/plain",
            char_count=1,
            content_hash="b" * 64,
            redaction_match_count=0,
        )

    assert credential_shaped_value not in str(error.value)


def test_open_backs_up_and_discards_only_truncated_tail(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    store.append(session_id="session", data=SessionStartedData(workspace="/repo", model="m"))
    store.close()
    valid_content = event_path.read_bytes()
    event_path.write_bytes(valid_content + b'{"schema_version":1,"id":2')

    recovered_store = EventStore.open(event_path)

    assert len(recovered_store.events) == 1
    assert event_path.read_bytes() == valid_content
    assert recovered_store.recovery_report.discarded_tail_byte_count > 0
    backup_path = recovered_store.recovery_report.backup_path
    assert backup_path is not None
    assert backup_path.read_bytes() == valid_content + b'{"schema_version":1,"id":2'
    recovered_store.close()


def test_open_syncs_backup_parent_before_truncating_corrupt_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    store.append(session_id="session", data=SessionStartedData(workspace="/repo", model="m"))
    store.close()
    valid_content = event_path.read_bytes()
    event_path.write_bytes(valid_content + b'{"schema_version":1,"id":2')

    real_fsync = event_store_module.os.fsync
    sync_targets: list[str] = []

    def record_fsync(file_descriptor: int) -> None:
        mode = event_store_module.os.fstat(file_descriptor).st_mode
        sync_targets.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(file_descriptor)

    monkeypatch.setattr(event_store_module.os, "fsync", record_fsync)

    recovered_store = EventStore.open(event_path)

    assert sync_targets == ["file", "directory", "file"]
    assert event_path.read_bytes() == valid_content
    recovered_store.close()


def test_open_aborts_without_truncating_when_backup_parent_sync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    store.append(session_id="session", data=SessionStartedData(workspace="/repo", model="m"))
    store.close()
    corrupt_content = event_path.read_bytes() + b'{"schema_version":1,"id":2'
    event_path.write_bytes(corrupt_content)

    real_fsync = event_store_module.os.fsync
    truncate_attempt_count = 0

    def fail_backup_parent_fsync(file_descriptor: int) -> None:
        mode = event_store_module.os.fstat(file_descriptor).st_mode
        if stat.S_ISDIR(mode):
            raise OSError("backup parent sync failed")
        real_fsync(file_descriptor)

    def record_truncate(file_descriptor: int, byte_count: int) -> None:
        nonlocal truncate_attempt_count
        truncate_attempt_count += 1

    monkeypatch.setattr(event_store_module.os, "fsync", fail_backup_parent_fsync)
    monkeypatch.setattr(event_store_module.os, "ftruncate", record_truncate)

    with pytest.raises(OSError, match="backup parent sync failed"):
        EventStore.open(event_path)

    assert truncate_attempt_count == 0
    assert event_path.read_bytes() == corrupt_content
    monkeypatch.undo()

    recovered_store = EventStore.open(event_path)

    assert recovered_store.recovery_report.backup_path is not None
    assert event_path.read_bytes() != corrupt_content
    recovered_store.close()


def test_open_rejects_corruption_in_complete_line(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    event_path.write_bytes(b"not-json\n")

    with pytest.raises(EventStoreCorruptionError, match="byte offset 0"):
        EventStore.open(event_path)


def test_read_only_open_does_not_recover_truncated_tail(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    event_path.write_bytes(b'{"partial":')

    with pytest.raises(EventStoreCorruptionError, match="resume"):
        EventStore.open(event_path, allow_recovery=False)

    assert event_path.read_bytes() == b'{"partial":'


def test_open_rejects_a_second_writer(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    writer = EventStore.open(event_path)

    with pytest.raises(EventStoreWriterBusyError, match="active writer"):
        EventStore.open(event_path)

    writer.close()


def test_append_failure_before_write_poison_store_without_reusing_event_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    write_attempts = 0

    def fail_before_write(path: Path, content: bytes, *, sync: bool) -> None:
        nonlocal write_attempts
        write_attempts += 1
        raise OSError("disk is unavailable")

    monkeypatch.setattr(event_store_module, "_append_bytes", fail_before_write)

    with pytest.raises(OSError, match="disk is unavailable"):
        store.append(session_id="session", data=SessionStartedData(workspace="/repo", model="m"))

    with pytest.raises(RuntimeError, match="resume or reopen"):
        store.append(session_id="session", data=UserMessageData(content="must not write"))

    assert write_attempts == 1
    assert store.events == ()
    store.close()
    monkeypatch.undo()

    recovered_store = EventStore.open(event_path)
    recovered_event = recovered_store.append(
        session_id="session", data=SessionStartedData(workspace="/repo", model="m")
    )

    assert recovered_event.id == 1
    recovered_store.close()


def test_append_failure_after_partial_write_poison_store_and_recovers_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    real_write = event_store_module.os.write
    write_calls = 0

    def write_part_then_fail(file_descriptor: int, content: bytes) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return real_write(file_descriptor, content[:20])
        raise OSError("short write outcome is unknown")

    monkeypatch.setattr(event_store_module.os, "write", write_part_then_fail)

    with pytest.raises(OSError, match="short write outcome is unknown"):
        store.append(session_id="session", data=SessionStartedData(workspace="/repo", model="m"))

    with pytest.raises(RuntimeError, match="resume or reopen"):
        store.append(session_id="session", data=UserMessageData(content="must not write"))

    assert store.events == ()
    assert event_path.read_bytes()
    store.close()
    monkeypatch.undo()

    recovered_store = EventStore.open(event_path)
    recovered_event = recovered_store.append(
        session_id="session", data=SessionStartedData(workspace="/repo", model="m")
    )

    assert recovered_store.recovery_report.discarded_tail_byte_count > 0
    assert recovered_event.id == 1
    recovered_store.close()


def test_append_failure_after_full_write_poison_store_when_sync_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    real_close = event_store_module.os.close
    close_calls = 0

    def close_then_fail(file_descriptor: int) -> None:
        nonlocal close_calls
        close_calls += 1
        real_close(file_descriptor)
        raise OSError("close outcome is unknown")

    monkeypatch.setattr(event_store_module.os, "close", close_then_fail)

    with pytest.raises(OSError, match="close outcome is unknown"):
        store.append(
            session_id="session",
            data=SessionStartedData(workspace="/repo", model="m"),
            sync=False,
        )

    with pytest.raises(RuntimeError, match="resume or reopen"):
        store.append(session_id="session", data=UserMessageData(content="must not write"))

    assert close_calls == 1
    assert store.events == ()
    monkeypatch.undo()
    store.close()

    recovered_store = EventStore.open(event_path)
    resumed_event = recovered_store.append(
        session_id="session", data=UserMessageData(content="after reopen")
    )

    assert [event.id for event in recovered_store.events] == [1, 2]
    assert resumed_event.id == 2
    recovered_store.close()


def test_append_keyboard_interrupt_after_full_write_poison_store_and_reopens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    real_close = event_store_module.os.close

    def close_then_interrupt(file_descriptor: int) -> None:
        real_close(file_descriptor)
        raise KeyboardInterrupt("close outcome is unknown")

    monkeypatch.setattr(event_store_module.os, "close", close_then_interrupt)

    with pytest.raises(KeyboardInterrupt, match="close outcome is unknown"):
        store.append(
            session_id="session",
            data=SessionStartedData(workspace="/repo", model="m"),
            sync=False,
        )

    with pytest.raises(RuntimeError, match="resume or reopen"):
        store.append(session_id="session", data=UserMessageData(content="must not write"))

    assert store.events == ()
    monkeypatch.undo()
    store.close()

    reopened_store = EventStore.open(event_path)
    resumed_event = reopened_store.append(
        session_id="session", data=UserMessageData(content="after reopen")
    )

    assert [event.id for event in reopened_store.events] == [1, 2]
    assert resumed_event.id == 2
    reopened_store.close()


def test_append_fsync_failure_poison_store_and_reopen_from_durable_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    real_fsync = event_store_module.os.fsync
    append_fsync_calls = 0

    def fsync_after_writing(file_descriptor: int) -> None:
        nonlocal append_fsync_calls
        append_fsync_calls += 1
        real_fsync(file_descriptor)
        raise OSError("fsync outcome is unknown")

    monkeypatch.setattr(event_store_module.os, "fsync", fsync_after_writing)

    with pytest.raises(OSError, match="fsync outcome is unknown") as append_error:
        store.append(session_id="session", data=SessionStartedData(workspace="/repo", model="m"))

    with pytest.raises(RuntimeError, match="resume or reopen") as poisoned_error:
        store.append(session_id="session", data=UserMessageData(content="must not write"))

    assert append_fsync_calls == 1
    assert store.events == ()
    assert poisoned_error.value.__cause__ is append_error.value
    assert len(event_path.read_bytes().splitlines()) == 1

    with pytest.raises(EventStoreWriterBusyError, match="active writer"):
        EventStore.open(event_path)
    store.close()
    monkeypatch.undo()

    recovered_store = EventStore.open(event_path)
    resumed_event = recovered_store.append(
        session_id="session", data=UserMessageData(content="after reopen")
    )

    assert [event.id for event in recovered_store.events] == [1, 2]
    assert resumed_event.id == 2
    recovered_store.close()


def test_append_memory_error_after_durable_write_poison_store_and_reopens(
    tmp_path: Path,
) -> None:
    class FailingEventList(list[StoredEvent]):
        def append(self, item: StoredEvent) -> None:
            raise MemoryError("event cache allocation failed")

    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    store._events = FailingEventList()  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(MemoryError, match="event cache allocation failed") as append_error:
        store.append(session_id="session", data=SessionStartedData(workspace="/repo", model="m"))

    with pytest.raises(RuntimeError, match="resume or reopen") as poisoned_error:
        store.append(session_id="session", data=UserMessageData(content="must not write"))

    assert poisoned_error.value.__cause__ is append_error.value
    assert len(event_path.read_bytes().splitlines()) == 1
    store.close()

    reopened_store = EventStore.open(event_path)
    resumed_event = reopened_store.append(
        session_id="session", data=UserMessageData(content="after reopen")
    )

    assert [event.id for event in reopened_store.events] == [1, 2]
    assert resumed_event.id == 2
    reopened_store.close()


def test_open_releases_writer_lock_after_newline_fsync_ambiguity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_path = tmp_path / "events.jsonl"
    original_store = EventStore.open(event_path)
    original_store.append(
        session_id="session", data=SessionStartedData(workspace="/repo", model="m")
    )
    original_store.close()
    event_path.write_bytes(event_path.read_bytes().removesuffix(b"\n"))
    real_fsync = event_store_module.os.fsync

    def fsync_then_fail(file_descriptor: int) -> None:
        real_fsync(file_descriptor)
        raise OSError("newline fsync outcome is unknown")

    monkeypatch.setattr(event_store_module.os, "fsync", fsync_then_fail)
    with pytest.raises(OSError, match="newline fsync outcome is unknown"):
        EventStore.open(event_path)
    monkeypatch.undo()

    reopened_store = EventStore.open(event_path)

    assert [event.id for event in reopened_store.events] == [1]
    reopened_store.close()


def test_append_redacts_nested_tool_arguments(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    tool_call = ToolCall(
        id="call-1",
        name="read_file",
        arguments_json='{"api_key":"example-secret-value","path":"safe.txt"}',
    )

    event = store.append(
        session_id="session",
        correlation_id=tool_call.id,
        data=ToolRequestedData(tool_call=tool_call, is_read_only=True),
    )

    assert isinstance(event.data, ToolRequestedData)
    assert "example-secret-value" not in event.data.tool_call.arguments_json
    assert "[REDACTED]" in event.data.tool_call.arguments_json
    assert "example-secret-value" not in event_path.read_text(encoding="utf-8")
    store.close()


def test_append_redacts_exec_command_sensitive_long_option_value(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    raw_value = "example-password-value"
    tool_call = ToolCall(
        id="call-1",
        name="exec_command",
        arguments_json=json.dumps(
            {"argv": ["echo", "--password", raw_value], "cwd": "."}
        ),
    )

    event = store.append(
        session_id="session",
        correlation_id=tool_call.id,
        data=ToolRequestedData(tool_call=tool_call, is_read_only=False),
    )

    assert isinstance(event.data, ToolRequestedData)
    assert raw_value not in event.data.tool_call.arguments_json
    assert "--password" not in event.data.tool_call.arguments_json
    assert event.redaction_summary.match_count == 2
    assert event_path.read_text(encoding="utf-8").count(raw_value) == 0
    store.close()


def test_append_redacts_vendor_key_and_connection_string_arguments(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    tool_call = ToolCall(
        id="call-1",
        name="read_file",
        arguments_json=(
            '{"openai_api_key":"test-openai-secret",'
            '"database_url":"postgresql://user:test-db-secret@db.example/app"}'
        ),
    )

    event = store.append(
        session_id="session",
        correlation_id=tool_call.id,
        data=ToolRequestedData(tool_call=tool_call, is_read_only=True),
    )

    assert isinstance(event.data, ToolRequestedData)
    persisted = event_path.read_text(encoding="utf-8")
    assert "test-openai-secret" not in persisted
    assert "test-db-secret" not in persisted
    assert event.data.tool_call.arguments_json.count("[REDACTED]") == 2
    store.close()


def test_append_redacts_assistant_content_and_camel_case_tool_arguments(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    tool_call = ToolCall(
        id="call-1",
        name="write_file",
        arguments_json='{"openaiApiKey":"test-tool-secret","path":"safe.txt"}',
    )

    event = store.append(
        session_id="session",
        correlation_id=tool_call.id,
        data=AssistantMessageData(
            content=(
                "AWS_SECRET_ACCESS_KEY=test-assistant-secret "
                "https://user:test-url-password@example.invalid/path"
            ),
            tool_calls=(tool_call,),
            finish_reason=FinishReason.TOOL_CALLS,
        ),
    )

    assert isinstance(event.data, AssistantMessageData)
    persisted = event_path.read_text(encoding="utf-8")
    for unsafe_value in (
        "test-tool-secret",
        "test-assistant-secret",
        "test-url-password",
    ):
        assert unsafe_value not in persisted
    assert persisted.count("[REDACTED]") >= 3
    store.close()


def test_append_redacts_credential_shaped_keys_in_assistant_and_tool_request(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    assistant_secret_key = "sk_live_" + "0" * 24
    requested_secret_key = "rk_test_" + "0" * 24
    assistant_call = ToolCall(
        id="call-assistant",
        name="write_file",
        arguments_json=(
            '{"[REDACTED_KEY_1]":"ordinary",'
            f'"{assistant_secret_key}":"secret-key field"}}'
        ),
    )
    requested_call = ToolCall(
        id="call-requested",
        name="write_file",
        arguments_json=f'{{"{requested_secret_key}":"another secret-key field"}}',
    )

    assistant_event = store.append(
        session_id="session",
        correlation_id=assistant_call.id,
        data=AssistantMessageData(
            tool_calls=(assistant_call,),
            finish_reason=FinishReason.TOOL_CALLS,
        ),
    )
    requested_event = store.append(
        session_id="session",
        correlation_id=requested_call.id,
        data=ToolRequestedData(tool_call=requested_call, is_read_only=False),
    )

    assert isinstance(assistant_event.data, AssistantMessageData)
    assert isinstance(requested_event.data, ToolRequestedData)
    persisted = event_path.read_text(encoding="utf-8")
    assert assistant_secret_key not in persisted
    assert requested_secret_key not in persisted
    assert json.loads(assistant_event.data.tool_calls[0].arguments_json) == {
        "[REDACTED_KEY_1]": "ordinary",
        "[REDACTED_KEY_2]": "secret-key field",
    }
    assert json.loads(requested_event.data.tool_call.arguments_json) == {
        "[REDACTED_KEY_1]": "another secret-key field"
    }
    store.close()


def test_append_redacts_stripe_and_aws_credentials_from_tool_output(tmp_path: Path) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    credentials = (
        _STRIPE_LIVE_KEY,
        _STRIPE_RESTRICTED_LIVE_KEY,
        _STRIPE_RESTRICTED_TEST_KEY,
        _STRIPE_WEBHOOK_SECRET,
        _AWS_ACCESS_KEY_ID,
    )
    output = " ".join(f"credential={credential}" for credential in credentials)

    event = store.append(
        session_id="session",
        data=ToolCompletedData(
            tool_call_id="call-1",
            output=output,
            facts=ToolCompletionFacts(),
        ),
    )

    assert isinstance(event.data, ToolCompletedData)
    persisted = event_path.read_text(encoding="utf-8")
    assert all(credential not in event.data.output for credential in credentials)
    assert all(credential not in persisted for credential in credentials)
    assert event.data.output == " ".join("credential=[REDACTED]" for _ in credentials)
    assert event.redaction_summary.match_count == len(credentials)
    assert event.redaction_summary.kinds == (RedactionKind.SECRET_PREFIX,)
    store.close()


def test_append_redacts_compact_sensitive_fields_in_every_tool_transcript_path(
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "events.jsonl"
    store = EventStore.open(event_path)
    unsafe_values = {
        name: f"synthetic-compact-event-value-{index}"
        for index, name in enumerate(_COMPACT_SENSITIVE_FIELD_NAMES)
    }
    named_text = " ".join(f"{name}={value}" for name, value in unsafe_values.items())
    arguments_json = json.dumps(unsafe_values)
    assistant_call = ToolCall(
        id="call-assistant",
        name="write_file",
        arguments_json=arguments_json,
    )
    requested_call = ToolCall(
        id="call-requested",
        name="write_file",
        arguments_json=arguments_json,
    )

    assistant_event = store.append(
        session_id="session",
        correlation_id=assistant_call.id,
        data=AssistantMessageData(
            content=named_text,
            tool_calls=(assistant_call,),
            finish_reason=FinishReason.TOOL_CALLS,
        ),
    )
    requested_event = store.append(
        session_id="session",
        correlation_id=requested_call.id,
        data=ToolRequestedData(tool_call=requested_call, is_read_only=False),
    )
    completed_event = store.append(
        session_id="session",
        correlation_id=requested_call.id,
        data=ToolCompletedData(
            tool_call_id=requested_call.id,
            output=named_text,
            facts=ToolCompletionFacts(),
        ),
    )
    persisted = event_path.read_text(encoding="utf-8")

    assert isinstance(assistant_event.data, AssistantMessageData)
    assert isinstance(requested_event.data, ToolRequestedData)
    assert isinstance(completed_event.data, ToolCompletedData)
    assert assistant_event.data.content is not None
    for value in unsafe_values.values():
        assert value not in persisted
        assert value not in assistant_event.data.content
        assert value not in requested_event.data.tool_call.arguments_json
        assert value not in completed_event.data.output
    assert json.loads(assistant_event.data.tool_calls[0].arguments_json) == {
        name: "[REDACTED]" for name in unsafe_values
    }
    assert json.loads(requested_event.data.tool_call.arguments_json) == {
        name: "[REDACTED]" for name in unsafe_values
    }
    assert assistant_event.data.content.count("[REDACTED]") == len(unsafe_values)
    assert completed_event.data.output.count("[REDACTED]") == len(unsafe_values)
    assert assistant_event.redaction_summary.match_count == len(unsafe_values) * 2
    assert requested_event.redaction_summary.match_count == len(unsafe_values)
    assert completed_event.redaction_summary.match_count == len(unsafe_values)
    store.close()
