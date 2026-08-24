"""Versioned checkpoint store behavior and commit-marker tests."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import mini_agent.checkpoint as checkpoint_module
from mini_agent.checkpoint import (
    Checkpoint,
    CheckpointItem,
    CheckpointStoreCorruptionError,
    CheckpointStoreReadOnlyError,
)
from mini_agent.checkpoint import CheckpointStore as _CheckpointStore
from mini_agent.events import CheckpointCommittedData


class CheckpointStore:
    """Bind low-level store fixtures to their synthetic workspace root."""

    @staticmethod
    def open(
        session_root: Path,
        *,
        registrations: tuple[CheckpointCommittedData, ...],
        workspace_root: Path | None = None,
    ) -> _CheckpointStore:
        return _CheckpointStore.open(
            session_root,
            registrations=registrations,
            workspace_root=workspace_root or session_root,
        )


def _session_root(tmp_path: Path) -> Path:
    root = tmp_path / ("a" * 32)
    root.mkdir()
    return root


def _checkpoint(
    session_root: Path,
    *,
    checkpoint_id: str = "b" * 32,
    version: int = 1,
    through: int = 4,
    intent: str = "Finish context compression",
) -> Checkpoint:
    return Checkpoint(
        checkpoint_id=checkpoint_id,
        session_id=session_root.name,
        cycle_id=1,
        version=version,
        source_from_event_id=1,
        source_through_event_id=through,
        current_intent=intent,
        active_work=(CheckpointItem(text="Implement checkpoint store", source_event_ids=(3,)),),
        next_actions=(CheckpointItem(text="Run tests", source_event_ids=(4,)),),
        writer_model="checkpoint-model",
        created_at=datetime.now(UTC),
    )


def _registration(checkpoint: Checkpoint, content_hash: str) -> CheckpointCommittedData:
    return CheckpointCommittedData(
        checkpoint_id=checkpoint.checkpoint_id,
        version=checkpoint.version,
        source_through_event_id=checkpoint.source_through_event_id,
        content_hash=content_hash,
    )


def test_staged_checkpoint_is_removed_on_writable_reopen_without_transcript_commit(
    tmp_path: Path,
) -> None:
    session_root = _session_root(tmp_path)
    store = CheckpointStore.open(session_root, registrations=())
    checkpoint = _checkpoint(session_root)

    record = store.stage(checkpoint)

    assert store.current is None
    reopened = CheckpointStore.open(session_root, registrations=())
    assert reopened.checkpoints == ()
    assert not (session_root / "checkpoints" / record.relative_storage_path).exists()
    index = json.loads((session_root / "checkpoints" / "index.json").read_text())
    assert index["checkpoints"] == []


def test_read_only_open_leaves_an_empty_session_tree_unchanged(tmp_path: Path) -> None:
    session_root = _session_root(tmp_path)
    before = tuple(session_root.iterdir())

    store = _CheckpointStore.open(
        session_root,
        registrations=(),
        workspace_root=session_root,
        writable=False,
    )

    assert store.current is None
    assert tuple(session_root.iterdir()) == before
    assert not (session_root / "checkpoints").exists()


def test_read_only_store_rejects_checkpoint_mutation(tmp_path: Path) -> None:
    session_root = _session_root(tmp_path)
    store = _CheckpointStore.open(
        session_root,
        registrations=(),
        workspace_root=session_root,
        writable=False,
    )

    with pytest.raises(CheckpointStoreReadOnlyError, match="read-only"):
        store.stage(_checkpoint(session_root))


def test_read_only_open_ignores_crash_orphan_without_mutating_storage(tmp_path: Path) -> None:
    session_root = _session_root(tmp_path)
    writable_store = CheckpointStore.open(session_root, registrations=())
    record = writable_store.stage(_checkpoint(session_root))
    index_path = session_root / "checkpoints" / "index.json"
    content_path = session_root / "checkpoints" / record.relative_storage_path
    index_before = index_path.read_bytes()

    read_only_store = _CheckpointStore.open(
        session_root,
        registrations=(),
        workspace_root=session_root,
        writable=False,
    )

    assert read_only_store.current is None
    assert index_path.read_bytes() == index_before
    assert content_path.is_file()


def test_read_only_open_rejects_missing_committed_checkpoint_storage(tmp_path: Path) -> None:
    session_root = _session_root(tmp_path)
    registration = CheckpointCommittedData(
        checkpoint_id="b" * 32,
        version=1,
        source_through_event_id=1,
        content_hash="a" * 64,
    )

    with pytest.raises(CheckpointStoreCorruptionError, match="storage is missing"):
        _CheckpointStore.open(
            session_root,
            registrations=(registration,),
            workspace_root=session_root,
            writable=False,
        )


def test_committed_checkpoint_round_trips_and_becomes_current(tmp_path: Path) -> None:
    session_root = _session_root(tmp_path)
    store = CheckpointStore.open(session_root, registrations=())
    checkpoint = _checkpoint(session_root)
    record = store.stage(checkpoint)
    registration = _registration(checkpoint, record.content_hash)

    activated = store.activate(registration)
    reopened = CheckpointStore.open(session_root, registrations=(registration,))

    assert activated == checkpoint
    assert reopened.current == checkpoint


def test_orphan_cleanup_never_deletes_a_durably_committed_checkpoint(tmp_path: Path) -> None:
    session_root = _session_root(tmp_path)
    store = CheckpointStore.open(session_root, registrations=())
    committed = _checkpoint(session_root, checkpoint_id="b" * 32, through=4)
    committed_record = store.stage(committed)
    registration = _registration(committed, committed_record.content_hash)
    store.activate(registration)
    orphan = _checkpoint(
        session_root,
        checkpoint_id="c" * 32,
        version=2,
        through=5,
        intent="interrupted candidate",
    )
    orphan_record = store.stage(orphan)

    reopened = CheckpointStore.open(session_root, registrations=(registration,))

    assert reopened.current == committed
    assert (session_root / "checkpoints" / committed_record.relative_storage_path).is_file()
    assert not (session_root / "checkpoints" / orphan_record.relative_storage_path).exists()
    index = json.loads((session_root / "checkpoints" / "index.json").read_text())
    assert [item["checkpoint_id"] for item in index["checkpoints"]] == [
        committed.checkpoint_id
    ]


def test_checkpoint_content_is_redacted_before_it_is_indexed(tmp_path: Path) -> None:
    session_root = _session_root(tmp_path)
    store = CheckpointStore.open(session_root, registrations=())
    secret = "synthetic-checkpoint-secret"
    checkpoint = _checkpoint(session_root, intent=f"api_key={secret}")

    record = store.stage(checkpoint)
    registration = _registration(checkpoint, record.content_hash)
    visible = store.activate(registration)

    assert visible.current_intent == "api_key=[REDACTED]"
    persisted = (session_root / "checkpoints" / record.relative_storage_path).read_text()
    assert secret not in persisted


def test_checkpoint_persistence_normalizes_known_paths_and_is_idempotent(
    tmp_path: Path,
) -> None:
    session_root = _session_root(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = CheckpointStore.open(
        session_root,
        registrations=(),
        workspace_root=workspace,
    )
    checkpoint = _checkpoint(
        session_root,
        intent=f"read {workspace}/private.txt api_key=synthetic-value",
    )

    record = store.stage(checkpoint)
    visible = store.activate(_registration(checkpoint, record.content_hash))

    assert visible.current_intent == "read <workspace-root>/private.txt api_key=[REDACTED]"
    assert CheckpointStore.open(
        session_root,
        registrations=(_registration(checkpoint, record.content_hash),),
        workspace_root=workspace,
    ).current == visible


def test_checkpoint_discards_uncommitted_rehashed_sensitive_content(tmp_path: Path) -> None:
    session_root = _session_root(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = CheckpointStore.open(session_root, registrations=(), workspace_root=workspace)
    checkpoint = _checkpoint(session_root)
    record = store.stage(checkpoint)
    content_path = session_root / "checkpoints" / record.relative_storage_path
    raw = json.loads(content_path.read_text())
    raw["current_intent"] = f"api_key=synthetic-value {workspace}/hidden"
    encoded = json.dumps(raw, separators=(",", ":")).encode()
    content_path.write_bytes(encoded)
    forged_hash = hashlib.sha256(encoded).hexdigest()
    index_path = session_root / "checkpoints" / "index.json"
    index = json.loads(index_path.read_text())
    index["checkpoints"][0]["content_hash"] = forged_hash
    index_path.write_text(json.dumps(index))
    reopened = CheckpointStore.open(
        session_root,
        registrations=(),
        workspace_root=workspace,
    )
    assert reopened.current is None
    assert not content_path.exists()


def test_checkpoint_rejects_rehashed_sensitive_content_after_transcript_commit(
    tmp_path: Path,
) -> None:
    session_root = _session_root(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = CheckpointStore.open(session_root, registrations=(), workspace_root=workspace)
    checkpoint = _checkpoint(session_root)
    record = store.stage(checkpoint)
    content_path = session_root / "checkpoints" / record.relative_storage_path
    raw = json.loads(content_path.read_text())
    raw["current_intent"] = f"api_key=synthetic-value {workspace}/hidden"
    encoded = json.dumps(raw, separators=(",", ":")).encode()
    content_path.write_bytes(encoded)
    forged_hash = hashlib.sha256(encoded).hexdigest()
    index_path = session_root / "checkpoints" / "index.json"
    index = json.loads(index_path.read_text())
    index["checkpoints"][0]["content_hash"] = forged_hash
    index_path.write_text(json.dumps(index))
    forged_registration = _registration(checkpoint, forged_hash)

    with pytest.raises(CheckpointStoreCorruptionError, match="durable safety"):
        CheckpointStore.open(
            session_root,
            registrations=(forged_registration,),
            workspace_root=workspace,
        )


def test_new_candidate_replaces_an_uncommitted_same_version_orphan(tmp_path: Path) -> None:
    session_root = _session_root(tmp_path)
    store = CheckpointStore.open(session_root, registrations=())
    orphan = _checkpoint(session_root, checkpoint_id="b" * 32, intent="orphan")
    replacement = _checkpoint(session_root, checkpoint_id="c" * 32, intent="replacement")

    orphan_record = store.stage(orphan)
    replacement_record = store.stage(replacement)
    registration = _registration(replacement, replacement_record.content_hash)

    visible = store.activate(registration)
    reopened = CheckpointStore.open(session_root, registrations=(registration,))

    assert visible.current_intent == "replacement"
    assert reopened.current == visible
    assert (session_root / "checkpoints" / orphan_record.relative_storage_path).is_file()


def test_writable_orphan_cleanup_fails_closed_when_final_directory_sync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_root = _session_root(tmp_path)
    store = CheckpointStore.open(session_root, registrations=())
    record = store.stage(_checkpoint(session_root))
    checkpoint_root = session_root / "checkpoints"
    actual_fsync_directory = checkpoint_module._fsync_directory  # pyright: ignore[reportPrivateUsage]
    sync_count = 0

    def fail_after_index_replacement(path: Path) -> None:
        nonlocal sync_count
        if path == checkpoint_root:
            sync_count += 1
            if sync_count == 2:
                raise OSError("orphan removal directory sync failed")
        actual_fsync_directory(path)

    monkeypatch.setattr(
        "mini_agent.checkpoint._fsync_directory",
        fail_after_index_replacement,
    )

    with pytest.raises(CheckpointStoreCorruptionError, match="crash-orphan cleanup"):
        CheckpointStore.open(session_root, registrations=())

    assert json.loads((checkpoint_root / "index.json").read_text())["checkpoints"] == []
    assert not (checkpoint_root / record.relative_storage_path).exists()


def test_registration_mismatch_and_content_tampering_fail_closed(tmp_path: Path) -> None:
    session_root = _session_root(tmp_path)
    store = CheckpointStore.open(session_root, registrations=())
    checkpoint = _checkpoint(session_root)
    record = store.stage(checkpoint)

    with pytest.raises(CheckpointStoreCorruptionError, match="commit"):
        CheckpointStore.open(
            session_root,
            registrations=(_registration(checkpoint, "0" * 64),),
        )

    content_path = session_root / "checkpoints" / record.relative_storage_path
    raw = json.loads(content_path.read_text())
    raw["current_intent"] = "forged"
    content_path.write_text(json.dumps(raw))
    with pytest.raises(CheckpointStoreCorruptionError, match="hash"):
        CheckpointStore.open(
            session_root,
            registrations=(_registration(checkpoint, record.content_hash),),
        )


def test_checkpoint_rejects_provenance_after_its_watermark(tmp_path: Path) -> None:
    session_root = _session_root(tmp_path)
    raw = _checkpoint(session_root).model_dump()
    raw["active_work"] = [
        {"text": "future", "source_event_ids": [5]},
    ]

    with pytest.raises(ValueError, match="after its watermark"):
        Checkpoint.model_validate(raw)
