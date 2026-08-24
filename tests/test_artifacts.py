import hashlib
import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from mini_agent.artifacts import (
    MAX_ARTIFACT_PREVIEW_CHAR_COUNT,
    MAX_ARTIFACT_READ_CHAR_COUNT,
    ArtifactNotFoundError,
    ArtifactRecord,
    ArtifactStoreCorruptionError,
)
from mini_agent.artifacts import ArtifactStore as _ArtifactStore
from mini_agent.event_store import EventStore
from mini_agent.events import ArtifactCreatedData
from mini_agent.redaction_types import RedactionKind, RedactionSummary


class ArtifactStore:
    """Bind low-level store fixtures to their synthetic workspace root."""

    @staticmethod
    def open(
        session_root: Path,
        *,
        registrations: tuple[ArtifactCreatedData, ...],
        workspace_root: Path | None = None,
    ) -> _ArtifactStore:
        return _ArtifactStore.open(
            session_root,
            registrations=registrations,
            workspace_root=workspace_root or session_root,
        )


def _registration(record: ArtifactRecord) -> ArtifactCreatedData:
    return ArtifactCreatedData(
        artifact_id=record.artifact_id,
        source_event_id=record.source_event_id,
        media_type=record.media_type,
        char_count=record.char_count,
        content_hash=record.content_hash,
        redaction_match_count=record.redaction_match_count,
        redaction_kinds=record.redaction_kinds,
    )


def _replace_index_preview(
    tmp_path: Path,
    *,
    preview_name: str,
    replacement: str,
) -> None:
    index_path = tmp_path / "artifacts" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["artifacts"][0][preview_name] = replacement
    index_path.write_text(json.dumps(index), encoding="utf-8")


def test_create_persists_redacted_content_and_versioned_metadata(tmp_path: Path) -> None:
    store = ArtifactStore.open(tmp_path, registrations=())

    record = store.create(
        "first line\napi_key=example-secret-value\nlast line",
        source_event_id=3,
    )

    persisted = (tmp_path / "artifacts" / record.relative_storage_path).read_text(encoding="utf-8")
    index = (tmp_path / "artifacts" / "index.json").read_text(encoding="utf-8")
    assert "example-secret-value" not in persisted
    assert "example-secret-value" not in index
    assert persisted == "first line\napi_key=[REDACTED]\nlast line"
    assert record.schema_version == 1
    assert record.redacted is True
    assert record.char_count == len(persisted)
    assert record.head == persisted
    assert record.tail == persisted
    assert '"schema_version":1' in index
    assert (tmp_path / "artifacts" / record.relative_storage_path).stat().st_mode & 0o777 == 0o600


def test_create_normalizes_known_paths_and_partial_trailing_credentials(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ArtifactStore.open(tmp_path, registrations=(), workspace_root=workspace)

    record = store.create(
        f"{workspace}/note.txt\nsk_live_partial_token",
        source_event_id=3,
        truncated_at_end=True,
    )

    persisted = (tmp_path / "artifacts" / record.relative_storage_path).read_text()
    assert persisted == "<workspace-root>/note.txt\n[REDACTED]"
    assert record.redaction_match_count == 2
    assert record.redaction_kinds == (
        RedactionKind.HOST_PATH,
        RedactionKind.SECRET_PREFIX,
    )


def test_rehashed_sensitive_artifact_is_rejected_at_open(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = ArtifactStore.open(tmp_path, registrations=(), workspace_root=workspace)
    record = store.create("safe", source_event_id=3)
    content_path = tmp_path / "artifacts" / record.relative_storage_path
    forged_content = f"api_key=synthetic-value {workspace}/private".encode()
    content_path.write_bytes(forged_content)
    forged_hash = hashlib.sha256(forged_content).hexdigest()
    index_path = tmp_path / "artifacts" / "index.json"
    index = json.loads(index_path.read_text())
    forged_text = forged_content.decode()
    index_record = index["artifacts"][0]
    index_record.update(
        content_hash=forged_hash,
        char_count=len(forged_text),
        byte_count=len(forged_content),
        head=forged_text,
        tail=forged_text,
    )
    index_path.write_text(json.dumps(index))
    forged_registration = ArtifactCreatedData(
        artifact_id=record.artifact_id,
        source_event_id=record.source_event_id,
        media_type=record.media_type,
        char_count=len(forged_text),
        content_hash=forged_hash,
        redaction_match_count=record.redaction_match_count,
        redaction_kinds=record.redaction_kinds,
    )

    with pytest.raises(ArtifactStoreCorruptionError, match="durable safety"):
        ArtifactStore.open(
            tmp_path,
            registrations=(forged_registration,),
            workspace_root=workspace,
        )


def test_open_fsyncs_session_root_after_creating_artifact_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced_directories: list[Path] = []

    def record_fsync(directory: Path) -> None:
        synced_directories.append(directory)

    monkeypatch.setattr(
        "mini_agent.artifacts._fsync_directory",
        record_fsync,
    )

    ArtifactStore.open(tmp_path, registrations=())

    assert synced_directories == [tmp_path]


def test_open_does_not_fsync_session_root_when_artifact_directory_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "artifacts").mkdir(mode=0o700)
    synced_directories: list[Path] = []

    def record_fsync(directory: Path) -> None:
        synced_directories.append(directory)

    monkeypatch.setattr(
        "mini_agent.artifacts._fsync_directory",
        record_fsync,
    )

    ArtifactStore.open(tmp_path, registrations=())

    assert synced_directories == []


def test_open_fsyncs_session_root_after_concurrent_artifact_directory_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    synced_directories: list[Path] = []
    original_mkdir = Path.mkdir

    def competing_mkdir(
        directory: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        if directory == artifact_root:
            original_mkdir(directory, mode=mode, parents=parents, exist_ok=exist_ok)
            raise FileExistsError("concurrent artifact directory creation")
        original_mkdir(directory, mode=mode, parents=parents, exist_ok=exist_ok)

    def record_fsync(directory: Path) -> None:
        synced_directories.append(directory)

    monkeypatch.setattr(Path, "mkdir", competing_mkdir)
    monkeypatch.setattr("mini_agent.artifacts._fsync_directory", record_fsync)

    ArtifactStore.open(tmp_path, registrations=())

    assert synced_directories == [tmp_path]


def test_open_fails_without_confirmed_artifact_when_parent_directory_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_parent_fsync(directory: Path) -> None:
        assert directory == tmp_path
        raise OSError("parent directory fsync failed")

    monkeypatch.setattr("mini_agent.artifacts._fsync_directory", fail_parent_fsync)

    with pytest.raises(OSError, match="parent directory fsync failed"):
        ArtifactStore.open(tmp_path, registrations=())

    artifact_root = tmp_path / "artifacts"
    assert artifact_root.is_dir()
    assert not (artifact_root / "index.json").exists()
    assert list(artifact_root.glob("*.txt")) == []


def test_artifact_registration_keeps_source_summary_without_redacting_metadata_again(
    tmp_path: Path,
) -> None:
    artifacts = ArtifactStore.open(tmp_path, registrations=())
    record = artifacts.create(
        "already safe text",
        source_event_id=3,
        source_redaction_summary=RedactionSummary(
            match_count=1,
            kinds=(RedactionKind.NAMED_SECRET,),
        ),
    )
    event_store = EventStore.open(tmp_path / "events.jsonl")
    event = event_store.append(
        session_id="session",
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

    assert record.redaction_match_count == 1
    assert record.redaction_kinds == (RedactionKind.NAMED_SECRET,)
    assert event.redaction_summary.match_count == 0
    assert isinstance(event.data, ArtifactCreatedData)
    assert event.data.redaction_match_count == 1
    assert event.data.redaction_kinds == (RedactionKind.NAMED_SECRET,)
    event_store.close()


@pytest.mark.parametrize(
    ("match_count", "kinds"),
    [
        (0, (RedactionKind.NAMED_SECRET,)),
        (1, ()),
    ],
)
def test_artifact_models_reject_ambiguous_redaction_metadata(
    tmp_path: Path,
    match_count: int,
    kinds: tuple[RedactionKind, ...],
) -> None:
    record = ArtifactStore.open(tmp_path, registrations=()).create(
        "safe artifact",
        source_event_id=3,
    )
    record_data = record.model_dump()
    record_data.update(
        redaction_match_count=match_count,
        redaction_kinds=kinds,
    )
    registration_data = _registration(record).model_dump()
    registration_data.update(
        redaction_match_count=match_count,
        redaction_kinds=kinds,
    )

    with pytest.raises(ValidationError, match="match count must be zero"):
        ArtifactRecord.model_validate(record_data)
    with pytest.raises(ValidationError, match="match count must be zero"):
        ArtifactCreatedData.model_validate(registration_data)


def test_open_rejects_persisted_artifact_index_with_ambiguous_redaction_metadata(
    tmp_path: Path,
) -> None:
    store = ArtifactStore.open(tmp_path, registrations=())
    record = store.create("api_key=synthetic-value", source_event_id=3)
    index_path = tmp_path / "artifacts" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["artifacts"][0]["redaction_match_count"] = 0
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ArtifactStoreCorruptionError, match="invalid schema"):
        ArtifactStore.open(tmp_path, registrations=(_registration(record),))


def test_read_returns_bounded_character_segments_after_restart(tmp_path: Path) -> None:
    store = ArtifactStore.open(tmp_path, registrations=())
    record = store.create("0123456789", source_event_id=1)

    reopened_store = ArtifactStore.open(tmp_path, registrations=(_registration(record),))

    assert reopened_store.read(record.artifact_id, offset=3, limit=4) == "3456"
    assert reopened_store.read(record.artifact_id, offset=20, limit=4) == ""


def test_create_records_bounded_head_and_tail(tmp_path: Path) -> None:
    content = (
        "a" * MAX_ARTIFACT_PREVIEW_CHAR_COUNT + "middle" + "b" * MAX_ARTIFACT_PREVIEW_CHAR_COUNT
    )
    store = ArtifactStore.open(tmp_path, registrations=())

    record = store.create(content, source_event_id=2)

    assert record.head == "a" * MAX_ARTIFACT_PREVIEW_CHAR_COUNT
    assert record.tail == "b" * MAX_ARTIFACT_PREVIEW_CHAR_COUNT
    assert record.char_count == len(content)
    assert len(record.head) == MAX_ARTIFACT_PREVIEW_CHAR_COUNT
    assert len(record.tail) == MAX_ARTIFACT_PREVIEW_CHAR_COUNT

    reference = record.context_reference(preview_chars=20)
    assert record.artifact_id in reference
    assert "a" * 10 in reference
    assert "b" * 10 in reference


@pytest.mark.parametrize("content", ["", "short artifact"])
def test_open_accepts_previews_derived_from_empty_and_short_content(
    tmp_path: Path,
    content: str,
) -> None:
    store = ArtifactStore.open(tmp_path, registrations=())
    record = store.create(content, source_event_id=2)

    reopened = ArtifactStore.open(tmp_path, registrations=(_registration(record),))

    assert reopened.records == (record,)


def test_open_rejects_tampered_artifact_head_preview(tmp_path: Path) -> None:
    store = ArtifactStore.open(tmp_path, registrations=())
    record = store.create("a" * MAX_ARTIFACT_PREVIEW_CHAR_COUNT + "suffix", source_event_id=2)
    _replace_index_preview(
        tmp_path,
        preview_name="head",
        replacement="x" * MAX_ARTIFACT_PREVIEW_CHAR_COUNT,
    )

    with pytest.raises(ArtifactStoreCorruptionError, match="head preview does not match"):
        ArtifactStore.open(tmp_path, registrations=(_registration(record),))


def test_open_rejects_tampered_artifact_tail_preview(tmp_path: Path) -> None:
    store = ArtifactStore.open(tmp_path, registrations=())
    record = store.create("prefix" + "b" * MAX_ARTIFACT_PREVIEW_CHAR_COUNT, source_event_id=2)
    _replace_index_preview(
        tmp_path,
        preview_name="tail",
        replacement="y" * MAX_ARTIFACT_PREVIEW_CHAR_COUNT,
    )

    with pytest.raises(ArtifactStoreCorruptionError, match="tail preview does not match"):
        ArtifactStore.open(tmp_path, registrations=(_registration(record),))


@pytest.mark.parametrize(
    ("artifact_id", "offset", "limit"),
    [
        ("../events.jsonl", 0, 1),
        ("A" * 32, 0, 1),
        ("0" * 32, -1, 1),
        ("0" * 32, 0, 0),
        ("0" * 32, 0, MAX_ARTIFACT_READ_CHAR_COUNT + 1),
    ],
)
def test_read_rejects_invalid_identifiers_and_windows(
    tmp_path: Path,
    artifact_id: str,
    offset: int,
    limit: int,
) -> None:
    store = ArtifactStore.open(tmp_path, registrations=())

    with pytest.raises(ValueError):
        store.read(artifact_id, offset=offset, limit=limit)


def test_read_rejects_unregistered_files(tmp_path: Path) -> None:
    store = ArtifactStore.open(tmp_path, registrations=())
    unregistered_id = "a" * 32
    unregistered_path = tmp_path / "artifacts" / f"{unregistered_id}.txt"
    unregistered_path.write_text("unregistered", encoding="utf-8")

    with pytest.raises(ArtifactNotFoundError, match="not registered"):
        store.read(unregistered_id, offset=0, limit=10)


def test_create_does_not_register_an_artifact_when_content_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore.open(tmp_path, registrations=())

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr("mini_agent.artifacts.os.replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        store.create("not persisted", source_event_id=1)

    assert store.records == ()
    assert not (tmp_path / "artifacts" / "index.json").exists()
    assert list((tmp_path / "artifacts").glob("*.txt")) == []


def test_transcript_registration_hides_index_only_artifact_without_deleting_it(
    tmp_path: Path,
) -> None:
    created = ArtifactStore.open(tmp_path, registrations=()).create("orphan", source_event_id=1)

    reopened = ArtifactStore.open(tmp_path, registrations=())

    assert reopened.records == ()
    assert (tmp_path / "artifacts" / created.relative_storage_path).is_file()
    assert created.artifact_id in (tmp_path / "artifacts" / "index.json").read_text(
        encoding="utf-8"
    )
    with pytest.raises(ArtifactNotFoundError):
        reopened.read(created.artifact_id, offset=0, limit=10)


def test_open_rejects_transcript_artifact_missing_from_all_metadata(tmp_path: Path) -> None:
    ArtifactStore.open(tmp_path, registrations=())
    registration = ArtifactCreatedData(
        artifact_id="a" * 32,
        source_event_id=1,
        media_type="text/plain",
        char_count=1,
        content_hash="b" * 64,
        redaction_match_count=0,
    )

    with pytest.raises(ArtifactStoreCorruptionError, match="missing artifact metadata"):
        ArtifactStore.open(
            tmp_path,
            registrations=(registration,),
        )


def test_open_rejects_tampered_registration_source_event(tmp_path: Path) -> None:
    store = ArtifactStore.open(tmp_path, registrations=())
    record = store.create("artifact", source_event_id=4)
    registration = _registration(record).model_copy(
        update={"source_event_id": record.source_event_id + 1}
    )

    with pytest.raises(ArtifactStoreCorruptionError, match="does not match index"):
        ArtifactStore.open(tmp_path, registrations=(registration,))


def test_open_rejects_tampered_registration_content_hash(tmp_path: Path) -> None:
    store = ArtifactStore.open(tmp_path, registrations=())
    record = store.create("artifact", source_event_id=4)
    registration = _registration(record).model_copy(update={"content_hash": "c" * 64})

    with pytest.raises(ArtifactStoreCorruptionError, match="does not match index"):
        ArtifactStore.open(tmp_path, registrations=(registration,))


def test_event_append_ambiguity_preserves_artifact_for_transcript_recovery(
    tmp_path: Path,
) -> None:
    store = ArtifactStore.open(tmp_path, registrations=())
    record = store.create("possibly committed", source_event_id=4)

    artifact_path = tmp_path / "artifacts" / record.relative_storage_path
    assert artifact_path.read_text(encoding="utf-8") == "possibly committed"

    # A complete event write followed by an fsync error is recovered from the transcript.
    recovered = ArtifactStore.open(
        tmp_path,
        registrations=(_registration(record),),
    )
    assert recovered.records == (record,)
    assert recovered.read(record.artifact_id, offset=0, limit=100) == "possibly committed"

    # If the event did not commit, the same index entry remains an invisible orphan.
    uncommitted = ArtifactStore.open(tmp_path, registrations=())
    assert uncommitted.records == ()
    assert artifact_path.is_file()


def test_open_rejects_tampered_unregistered_indexed_artifact(tmp_path: Path) -> None:
    store = ArtifactStore.open(tmp_path, registrations=())
    record = store.create("original", source_event_id=4)
    artifact_path = tmp_path / "artifacts" / record.relative_storage_path
    artifact_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(ArtifactStoreCorruptionError, match="hash does not match"):
        ArtifactStore.open(
            tmp_path,
            registrations=(_registration(record),),
        )
    with pytest.raises(ArtifactStoreCorruptionError, match="hash does not match"):
        ArtifactStore.open(tmp_path, registrations=())


def test_artifact_store_rejects_symlinked_content(tmp_path: Path) -> None:
    store = ArtifactStore.open(tmp_path, registrations=())
    record = store.create("safe", source_event_id=1)
    artifact_path = tmp_path / "artifacts" / record.relative_storage_path
    artifact_path.unlink()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    artifact_path.symlink_to(outside)

    with pytest.raises(ArtifactStoreCorruptionError):
        ArtifactStore.open(tmp_path, registrations=(_registration(record),))


def test_index_write_failure_retains_content_as_an_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore.open(tmp_path, registrations=())
    real_replace = os.replace
    write_count = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise OSError("index write failed")
        real_replace(source, destination)

    monkeypatch.setattr("mini_agent.artifacts.os.replace", fail_second_replace)

    with pytest.raises(OSError, match="index write failed"):
        store.create("orphan candidate", source_event_id=1)

    assert store.records == ()
    retained_content = list((tmp_path / "artifacts").glob("*.txt"))
    assert len(retained_content) == 1
    assert retained_content[0].read_text(encoding="utf-8") == "orphan candidate"


def test_index_directory_fsync_ambiguity_keeps_recoverable_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ArtifactStore.open(tmp_path, registrations=())
    fsync_count = 0

    def fail_after_index_replace(directory: Path) -> None:
        nonlocal fsync_count
        fsync_count += 1
        if fsync_count == 2:
            raise OSError("index directory fsync outcome is unknown")
        directory_descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

    monkeypatch.setattr(
        "mini_agent.artifacts._fsync_directory",
        fail_after_index_replace,
    )

    with pytest.raises(OSError, match="outcome is unknown"):
        store.create("preserve after ambiguous fsync", source_event_id=5)

    reopened = ArtifactStore.open(tmp_path, registrations=())
    assert reopened.records == ()
    indexed_artifact_ids = [
        path.stem for path in (tmp_path / "artifacts").glob("*.txt")
    ]
    assert len(indexed_artifact_ids) == 1
    assert indexed_artifact_ids[0] in (
        tmp_path / "artifacts" / "index.json"
    ).read_text(encoding="utf-8")
