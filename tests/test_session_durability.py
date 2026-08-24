import os
from pathlib import Path

import pytest

from mini_agent.event_store import (
    EventStore,
    EventStoreCorruptionError,
    EventStoreWriterBusyError,
)
from mini_agent.events import StoredEvent
from mini_agent.messages import FinishReason, ToolCall
from mini_agent.session import (
    AgentSession,
    SessionNotFoundError,
    list_sessions,
)


def test_prepare_session_storage_fsyncs_each_new_parent_directory_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "new-parent" / "data"
    synchronized_directories: list[Path] = []

    monkeypatch.setattr(
        "mini_agent.session._fsync_directory",
        synchronized_directories.append,
    )

    session = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="m")
    session.close()

    assert synchronized_directories[:3] == [
        tmp_path,
        tmp_path / "new-parent",
        data_dir,
    ]


def test_prepare_session_storage_fsyncs_data_directory_when_only_sessions_is_new(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    synchronized_directories: list[Path] = []

    monkeypatch.setattr(
        "mini_agent.session._fsync_directory",
        synchronized_directories.append,
    )

    session = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="m")
    session.close()

    assert synchronized_directories[0] == data_dir


def test_prepare_session_storage_fsyncs_parent_after_concurrent_directory_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    synchronized_directories: list[Path] = []
    original_mkdir = Path.mkdir

    def create_then_report_exists(
        path: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        if path == data_dir:
            original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)
            raise FileExistsError
        original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", create_then_report_exists)
    monkeypatch.setattr(
        "mini_agent.session._fsync_directory",
        synchronized_directories.append,
    )

    session = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="m")
    session.close()

    assert synchronized_directories[:2] == [tmp_path, data_dir]


def test_prepare_session_storage_skips_fsync_when_directories_already_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "sessions").mkdir(parents=True)
    synchronized_directories: list[Path] = []

    monkeypatch.setattr(
        "mini_agent.session._fsync_directory",
        synchronized_directories.append,
    )

    session = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="m")
    session.close()

    assert data_dir not in synchronized_directories
    assert data_dir / "sessions" in synchronized_directories


def test_create_fails_before_publication_when_parent_directory_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ParentDirectoryFsyncError(RuntimeError):
        pass

    data_dir = tmp_path / "data"

    def fail_data_parent_sync(path: Path) -> None:
        if path == tmp_path:
            raise ParentDirectoryFsyncError("new data directory is not durable")

    monkeypatch.setattr("mini_agent.session._fsync_directory", fail_data_parent_sync)

    with pytest.raises(ParentDirectoryFsyncError, match="not durable"):
        AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="m")

    assert list_sessions(data_dir) == ()
    assert not (data_dir / "sessions").exists()


def test_create_failure_before_initial_append_stays_hidden_and_releases_writer_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InitialAppendError(BaseException):
        pass

    def fail_initial_append(*args: object, **kwargs: object) -> StoredEvent:
        del args, kwargs
        raise InitialAppendError("initial event was not durably recorded")

    monkeypatch.setattr(EventStore, "append", fail_initial_append)
    with pytest.raises(InitialAppendError) as error:
        AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    monkeypatch.undo()

    assert str(error.value) == "initial event was not durably recorded"
    sessions_dir = tmp_path / "data" / "sessions"
    events_path = next(sessions_dir.glob(".*/events.jsonl"))
    assert events_path.is_file()
    reopened = EventStore.open(events_path)
    reopened.close()

    healthy = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    try:
        assert [summary.session_id for summary in list_sessions(tmp_path / "data")] == [
            healthy.metadata.session_id
        ]
    finally:
        healthy.close()


def test_create_partial_initial_append_stays_hidden_from_session_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PartialAppendError(RuntimeError):
        pass

    def write_partial_then_fail(path: Path, content: bytes, *, sync: bool) -> None:
        del sync
        with path.open("ab") as event_file:
            event_file.write(content[: max(1, len(content) // 2)])
            event_file.flush()
            os.fsync(event_file.fileno())
        raise PartialAppendError("initial event may have a partial durable tail")

    monkeypatch.setattr("mini_agent.event_store._append_bytes", write_partial_then_fail)
    with pytest.raises(PartialAppendError):
        AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    monkeypatch.undo()

    healthy = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    try:
        assert [summary.session_id for summary in list_sessions(tmp_path / "data")] == [
            healthy.metadata.session_id
        ]
    finally:
        healthy.close()


def test_create_full_initial_append_fsync_failure_stays_hidden_and_releases_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FullAppendFsyncError(RuntimeError):
        pass

    def write_full_then_fail(path: Path, content: bytes, *, sync: bool) -> None:
        del sync
        with path.open("ab") as event_file:
            event_file.write(content)
            event_file.flush()
            os.fsync(event_file.fileno())
        raise FullAppendFsyncError("initial event may be durable despite fsync failure")

    monkeypatch.setattr("mini_agent.event_store._append_bytes", write_full_then_fail)
    with pytest.raises(FullAppendFsyncError):
        AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    monkeypatch.undo()

    staging_events = next((tmp_path / "data" / "sessions").glob(".*/events.jsonl"))
    reopened = EventStore.open(staging_events)
    reopened.close()

    healthy = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    try:
        assert [summary.session_id for summary in list_sessions(tmp_path / "data")] == [
            healthy.metadata.session_id
        ]
    finally:
        healthy.close()


@pytest.mark.parametrize("is_terminal_durable", (False, True))
def test_stop_append_failure_keeps_memory_fail_stopped_and_resume_handles_unknown_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    is_terminal_durable: bool,
) -> None:
    """A failed terminal append must not pretend to know whether it reached disk."""

    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")

    def fail_stop_append(path: Path, content: bytes, *, sync: bool) -> None:
        del sync
        if is_terminal_durable:
            with path.open("ab") as event_file:
                event_file.write(content)
                event_file.flush()
                os.fsync(event_file.fileno())
        raise OSError("terminal append fsync outcome is unknown")

    monkeypatch.setattr("mini_agent.event_store._append_bytes", fail_stop_append)
    with pytest.raises(OSError, match="outcome is unknown"):
        session.stop("user_exit")

    # EventStore deliberately does not mutate its in-memory sequence after an
    # append error: the next event id would otherwise be fabricated.
    assert all(event.data.kind != "session_stopped" for event in session.events)
    assert (b'"kind":"session_stopped"' in session.paths.events.read_bytes()) is is_terminal_durable
    session.close()
    monkeypatch.undo()

    resumed = AgentSession.resume(
        data_dir=tmp_path / "data",
        session_id=session.metadata.session_id,
    )
    try:
        event_kinds = [event.data.kind for event in resumed.events]
        assert event_kinds[-1] == "session_resumed"
        assert ("session_stopped" in event_kinds) is is_terminal_durable
    finally:
        resumed.close()
    assert not list(session.paths.root.glob("events.jsonl.recovery-*.bin"))


def test_create_publish_directory_fsync_failure_keeps_formal_directory_unpublished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DirectoryFsyncError(RuntimeError):
        pass

    sessions_dir = tmp_path / "data" / "sessions"

    def fail_publish_fsync(path: Path) -> None:
        if path == sessions_dir:
            raise DirectoryFsyncError("session publication is ambiguous")

    monkeypatch.setattr("mini_agent.session._fsync_directory", fail_publish_fsync)
    with pytest.raises(DirectoryFsyncError):
        AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    monkeypatch.undo()

    assert list_sessions(tmp_path / "data") == ()
    formal_directory = next(
        directory
        for directory in sessions_dir.iterdir()
        if directory.is_dir() and not directory.name.startswith(".")
    )
    unpublished_session_id = formal_directory.name
    assert formal_directory.joinpath(".publication-v2").read_bytes() == (
        b"mini-agent-session-publication-v2\n"
    )
    assert not (formal_directory / ".published").exists()

    with pytest.raises(SessionNotFoundError, match="initialization did not complete"):
        AgentSession.peek_metadata(
            data_dir=tmp_path / "data",
            session_id=unpublished_session_id,
        )
    with pytest.raises(SessionNotFoundError, match="initialization did not complete"):
        AgentSession.load(
            data_dir=tmp_path / "data",
            session_id=unpublished_session_id,
        )
    with pytest.raises(SessionNotFoundError, match="initialization did not complete"):
        AgentSession.resume(
            data_dir=tmp_path / "data",
            session_id=unpublished_session_id,
        )

    healthy = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    try:
        assert AgentSession.peek_metadata(
            data_dir=tmp_path / "data",
            session_id=healthy.metadata.session_id,
        ) == healthy.metadata
        assert [summary.session_id for summary in list_sessions(tmp_path / "data")] == [
            healthy.metadata.session_id
        ]
    finally:
        healthy.close()


def test_create_holds_writer_lock_across_publish_without_reopening_window(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")

    assert session.paths.publication_intent_marker.read_bytes() == (
        b"mini-agent-session-publication-v2\n"
    )
    assert session.paths.published_marker.read_bytes() == b"mini-agent-session-published-v1\n"
    with pytest.raises(EventStoreWriterBusyError):
        EventStore.open(session.paths.events)

    session.close()
    reopened = EventStore.open(session.paths.events)
    reopened.close()


def test_create_marker_failure_keeps_formal_directory_unpublished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PublicationError(RuntimeError):
        pass

    def fail_publication(marker: Path) -> None:
        del marker
        raise PublicationError("publication commit did not complete")

    monkeypatch.setattr("mini_agent.session._write_published_marker", fail_publication)
    with pytest.raises(PublicationError):
        AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")

    sessions_dir = tmp_path / "data" / "sessions"
    formal_directory = next(
        directory
        for directory in sessions_dir.iterdir()
        if directory.is_dir() and not directory.name.startswith(".")
    )
    assert not (formal_directory / ".published").exists()
    assert formal_directory.joinpath(".publication-v2").read_bytes() == (
        b"mini-agent-session-publication-v2\n"
    )
    assert list_sessions(tmp_path / "data") == ()
    with pytest.raises(SessionNotFoundError, match="initialization did not complete"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=formal_directory.name)


def test_create_treats_publication_rename_interruption_after_commit_as_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_replace = os.replace

    def replace_then_interrupt(source: Path | str, target: Path | str) -> None:
        actual_replace(source, target)
        if Path(target).name == ".published":
            raise KeyboardInterrupt("publication rename completed before interruption")

    monkeypatch.setattr("mini_agent.session.os.replace", replace_then_interrupt)
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")

    try:
        assert AgentSession.peek_metadata(
            data_dir=tmp_path / "data",
            session_id=session.metadata.session_id,
        ) == session.metadata
        assert [summary.session_id for summary in list_sessions(tmp_path / "data")] == [
            session.metadata.session_id
        ]
    finally:
        session.close()


def test_create_does_not_report_success_when_final_publication_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FinalPublicationFsyncError(RuntimeError):
        pass

    def fail_after_publication(path: Path) -> None:
        if (path / ".published").is_file():
            raise FinalPublicationFsyncError("final publication durability is ambiguous")

    monkeypatch.setattr("mini_agent.session._fsync_directory", fail_after_publication)

    with pytest.raises(FinalPublicationFsyncError, match="ambiguous"):
        AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")

    sessions_dir = tmp_path / "data" / "sessions"
    formal_directory = next(
        directory
        for directory in sessions_dir.iterdir()
        if directory.is_dir() and not directory.name.startswith(".")
    )
    assert (formal_directory / ".published").is_file()


def test_legacy_session_without_publication_markers_remains_loadable(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    session.close()
    session.paths.publication_intent_marker.unlink()
    session.paths.published_marker.unlink()

    loaded = AgentSession.load(
        data_dir=tmp_path / "data",
        session_id=session.metadata.session_id,
    )
    try:
        assert loaded.metadata == session.metadata
        assert [summary.session_id for summary in list_sessions(tmp_path / "data")] == [
            session.metadata.session_id
        ]
    finally:
        loaded.close()

    resumed = AgentSession.resume(
        data_dir=tmp_path / "data",
        session_id=session.metadata.session_id,
    )
    resumed.close()


def test_load_with_recovery_releases_writer_lock_after_semantic_corruption(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    tool_call = ToolCall(id="call-orphan", name="read_file", arguments_json='{"path":"x"}')
    session.append_tool_requested(tool_call, is_read_only=True, turn_id=session.new_turn_id())
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="no assistant call"):
        AgentSession.load(
            data_dir=tmp_path / "data",
            session_id=session.metadata.session_id,
            allow_recovery=True,
        )

    reopened = EventStore.open(session.paths.events)
    reopened.close()


def test_resume_releases_writer_lock_when_recovery_append_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecoveryAppendError(BaseException):
        pass

    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    tool_call = ToolCall(id="call-recovery", name="read_file", arguments_json='{"path":"x"}')
    turn_id = session.new_turn_id()
    session.append_model_request_started(turn_id=turn_id, message_count=1, attempt=1)
    session.append_assistant_message(
        None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(tool_call,),
        turn_id=turn_id,
    )
    session.close()

    def fail_recovery_append(*args: object, **kwargs: object) -> StoredEvent:
        del args, kwargs
        raise RecoveryAppendError("recovery event was not durably recorded")

    monkeypatch.setattr(EventStore, "append", fail_recovery_append)
    with pytest.raises(RecoveryAppendError) as error:
        AgentSession.resume(data_dir=tmp_path / "data", session_id=session.metadata.session_id)
    monkeypatch.undo()

    assert str(error.value) == "recovery event was not durably recorded"
    reopened = EventStore.open(session.paths.events)
    reopened.close()


def test_resume_rejects_orphan_assistant_before_recovery_side_effects(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    tool_call = ToolCall(id="call-orphan", name="read_file", arguments_json='{"path":"x"}')
    session.append_assistant_message(
        None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(tool_call,),
        turn_id=session.new_turn_id(),
    )
    session.close()
    durable_before = session.paths.events.read_bytes()

    with pytest.raises(EventStoreCorruptionError, match="no active model request"):
        AgentSession.resume(data_dir=tmp_path / "data", session_id=session.metadata.session_id)

    assert session.paths.events.read_bytes() == durable_before
