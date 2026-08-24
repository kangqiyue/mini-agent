"""Versioned checkpoint state and transcript-committed local storage."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from mini_agent.events import CheckpointCommittedData
from mini_agent.host_path_redaction import redact_persisted_text

_CHECKPOINT_ID_PATTERN = r"^[0-9a-f]{32}$"


class CheckpointStoreCorruptionError(RuntimeError):
    """Raised when a committed checkpoint cannot be validated."""


class CheckpointStoreReadOnlyError(RuntimeError):
    """Raised when a mutating operation is attempted through a read-only store."""


class CheckpointItem(BaseModel):
    """One state atom with explicit transcript provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=8_000)
    source_event_ids: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_sorted_unique_source_ids(self) -> Self:
        if self.source_event_ids != tuple(sorted(set(self.source_event_ids))):
            raise ValueError("Checkpoint source event ids must be sorted and unique")
        if any(event_id < 1 for event_id in self.source_event_ids):
            raise ValueError("Checkpoint source event ids must be positive")
        return self


class Checkpoint(BaseModel):
    """Canonical, machine-readable working state for one logical session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    checkpoint_id: str = Field(pattern=_CHECKPOINT_ID_PATTERN)
    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    cycle_id: int = Field(ge=1)
    version: int = Field(ge=1)
    source_from_event_id: int = Field(ge=1)
    source_through_event_id: int = Field(ge=1)
    current_intent: str = Field(min_length=1, max_length=8_000)
    acceptance_criteria: tuple[CheckpointItem, ...] = ()
    constraints_and_preferences: tuple[CheckpointItem, ...] = ()
    task_tree: tuple[CheckpointItem, ...] = ()
    completed: tuple[CheckpointItem, ...] = ()
    active_work: tuple[CheckpointItem, ...] = ()
    blocked: tuple[CheckpointItem, ...] = ()
    next_actions: tuple[CheckpointItem, ...] = ()
    relevant_files: tuple[CheckpointItem, ...] = ()
    cross_task_findings: tuple[CheckpointItem, ...] = ()
    errors_and_fixes: tuple[CheckpointItem, ...] = ()
    runtime_state: tuple[CheckpointItem, ...] = ()
    key_decisions: tuple[CheckpointItem, ...] = ()
    artifact_references: tuple[CheckpointItem, ...] = ()
    miscellaneous_notes: tuple[CheckpointItem, ...] = ()
    writer_model: str = Field(min_length=1)
    created_at: datetime

    @model_validator(mode="after")
    def require_valid_watermark_and_provenance(self) -> Self:
        if self.source_from_event_id > self.source_through_event_id:
            raise ValueError("Checkpoint source range cannot be reversed")
        for item in self.items:
            if any(event_id > self.source_through_event_id for event_id in item.source_event_ids):
                raise ValueError("Checkpoint item references an event after its watermark")
        return self

    @property
    def items(self) -> tuple[CheckpointItem, ...]:
        return (
            *self.acceptance_criteria,
            *self.constraints_and_preferences,
            *self.task_tree,
            *self.completed,
            *self.active_work,
            *self.blocked,
            *self.next_actions,
            *self.relevant_files,
            *self.cross_task_findings,
            *self.errors_and_fixes,
            *self.runtime_state,
            *self.key_decisions,
            *self.artifact_references,
            *self.miscellaneous_notes,
        )


class CheckpointRecord(BaseModel):
    """Small integrity record mirrored by a committed transcript event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    checkpoint_id: str = Field(pattern=_CHECKPOINT_ID_PATTERN)
    version: int = Field(ge=1)
    source_through_event_id: int = Field(ge=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    relative_storage_path: str

    @model_validator(mode="after")
    def require_storage_path_to_match_id(self) -> Self:
        expected = f"checkpoint-{self.checkpoint_id}.json"
        if self.relative_storage_path != expected:
            raise ValueError("Checkpoint storage path must match checkpoint id")
        return self


class CheckpointIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    checkpoints: tuple[CheckpointRecord, ...] = ()

    @model_validator(mode="after")
    def require_unique_versions_and_ids(self) -> Self:
        ids = {record.checkpoint_id for record in self.checkpoints}
        versions = {record.version for record in self.checkpoints}
        if len(ids) != len(self.checkpoints) or len(versions) != len(self.checkpoints):
            raise ValueError("Checkpoint index contains duplicate ids or versions")
        if tuple(record.version for record in self.checkpoints) != tuple(
            sorted(versions)
        ):
            raise ValueError("Checkpoint versions must be strictly increasing")
        return self


class CheckpointStore:
    """Keeps checkpoint files immutable; transcript events decide visibility."""

    def __init__(
        self,
        root: Path,
        *,
        workspace_root: Path,
        is_writable: bool,
        indexed_records: tuple[CheckpointRecord, ...],
        visible_checkpoints: tuple[Checkpoint, ...],
    ) -> None:
        self._root = root
        self._workspace_root = workspace_root
        self._is_writable = is_writable
        self._index_path = root / "index.json"
        self._records_by_id = {record.checkpoint_id: record for record in indexed_records}
        self._visible_by_id = {
            checkpoint.checkpoint_id: checkpoint for checkpoint in visible_checkpoints
        }

    @classmethod
    def open(
        cls,
        session_root: Path,
        *,
        registrations: tuple[CheckpointCommittedData, ...],
        workspace_root: Path,
        writable: bool = True,
    ) -> CheckpointStore:
        """Open one store bound to its session's validated workspace."""
        _require_regular_directory(session_root, "Session root")
        root = session_root / "checkpoints"
        if root.exists():
            _require_regular_directory(root, "Checkpoint storage")
        else:
            if registrations:
                raise CheckpointStoreCorruptionError(
                    "Checkpoint storage is missing for committed state"
                )
            if writable:
                root.mkdir(mode=0o700)
                _fsync_directory(root.parent)

        bound_workspace_root = _resolve_workspace_root(workspace_root)
        index = _load_index(root / "index.json")
        _validate_registration_sequence(registrations)
        _validate_registered_checkpoints(
            root,
            index,
            registrations,
            workspace_root=bound_workspace_root,
        )
        index = _reconcile_crash_orphans(
            root,
            index,
            registrations,
            writable=writable,
        )
        visible = _select_visible(
            root,
            index,
            registrations,
            workspace_root=bound_workspace_root,
        )
        return cls(
            root,
            workspace_root=bound_workspace_root,
            is_writable=writable,
            indexed_records=index.checkpoints,
            visible_checkpoints=visible,
        )

    @property
    def checkpoints(self) -> tuple[Checkpoint, ...]:
        return tuple(sorted(self._visible_by_id.values(), key=lambda item: item.version))

    @property
    def current(self) -> Checkpoint | None:
        checkpoints = self.checkpoints
        return checkpoints[-1] if checkpoints else None

    def stage(self, checkpoint: Checkpoint) -> CheckpointRecord:
        """Persist an immutable candidate before its commit event is appended."""

        self._require_writable()
        if checkpoint.checkpoint_id in self._records_by_id:
            raise ValueError("Checkpoint id is already indexed")
        expected_version = self.current.version + 1 if self.current is not None else 1
        if checkpoint.version != expected_version:
            raise ValueError("Checkpoint version is not the next index version")
        safe_checkpoint = _redact_checkpoint(
            checkpoint,
            workspace_root=self._workspace_root,
        )
        encoded = safe_checkpoint.model_dump_json().encode()
        record = CheckpointRecord(
            checkpoint_id=safe_checkpoint.checkpoint_id,
            version=safe_checkpoint.version,
            source_through_event_id=safe_checkpoint.source_through_event_id,
            content_hash=hashlib.sha256(encoded).hexdigest(),
            relative_storage_path=f"checkpoint-{safe_checkpoint.checkpoint_id}.json",
        )
        checkpoint_path = self._root / record.relative_storage_path
        if checkpoint_path.exists() or checkpoint_path.is_symlink():
            raise ValueError("Checkpoint content path is already present")
        _write_atomically(checkpoint_path, encoded)
        retained_records = tuple(
            existing
            for existing in self._records_by_id.values()
            if existing.version < expected_version
        )
        new_index = CheckpointIndex(checkpoints=(*retained_records, record))
        _write_atomically(self._index_path, new_index.model_dump_json().encode())
        self._records_by_id = {
            existing.checkpoint_id: existing for existing in new_index.checkpoints
        }
        return record

    def load_staged(self, checkpoint_id: str) -> Checkpoint:
        """Load one indexed candidate for the coordinator's commit step."""

        if not is_checkpoint_id(checkpoint_id):
            raise ValueError("Checkpoint id is invalid")
        record = self._records_by_id.get(checkpoint_id)
        if record is None:
            raise CheckpointStoreCorruptionError("Staged checkpoint is missing")
        return _load_checkpoint(
            self._root / record.relative_storage_path,
            record,
            workspace_root=self._workspace_root,
        )

    def activate(self, registration: CheckpointCommittedData) -> Checkpoint:
        """Expose a staged checkpoint only after its transcript commit exists."""

        self._require_writable()
        record = self._records_by_id.get(registration.checkpoint_id)
        if record is None or (
            record.version != registration.version
            or record.source_through_event_id != registration.source_through_event_id
            or record.content_hash != registration.content_hash
        ):
            raise CheckpointStoreCorruptionError(
                "Checkpoint commit does not match its durable index"
            )
        checkpoint = _load_checkpoint(
            self._root / record.relative_storage_path,
            record,
            workspace_root=self._workspace_root,
        )
        current = self.current
        if current is not None and (
            checkpoint.version != current.version + 1
            or checkpoint.source_through_event_id <= current.source_through_event_id
        ):
            raise CheckpointStoreCorruptionError(
                "Checkpoint commit does not advance the visible state"
            )
        self._visible_by_id[checkpoint.checkpoint_id] = checkpoint
        return checkpoint

    @staticmethod
    def new_checkpoint_id() -> str:
        return uuid4().hex

    def _require_writable(self) -> None:
        if not self._is_writable:
            raise CheckpointStoreReadOnlyError("Checkpoint store is read-only")


def _redact_checkpoint(
    checkpoint: Checkpoint,
    *,
    workspace_root: Path,
) -> Checkpoint:
    def safe_items(items: tuple[CheckpointItem, ...]) -> tuple[CheckpointItem, ...]:
        return tuple(
            item.model_copy(
                update={
                    "text": redact_persisted_text(
                        item.text,
                        workspace_root=workspace_root,
                    ).text
                }
            )
            for item in items
        )

    return checkpoint.model_copy(
        update={
            "current_intent": redact_persisted_text(
                checkpoint.current_intent,
                workspace_root=workspace_root,
            ).text,
            "acceptance_criteria": safe_items(checkpoint.acceptance_criteria),
            "constraints_and_preferences": safe_items(
                checkpoint.constraints_and_preferences
            ),
            "task_tree": safe_items(checkpoint.task_tree),
            "completed": safe_items(checkpoint.completed),
            "active_work": safe_items(checkpoint.active_work),
            "blocked": safe_items(checkpoint.blocked),
            "next_actions": safe_items(checkpoint.next_actions),
            "relevant_files": safe_items(checkpoint.relevant_files),
            "cross_task_findings": safe_items(checkpoint.cross_task_findings),
            "errors_and_fixes": safe_items(checkpoint.errors_and_fixes),
            "runtime_state": safe_items(checkpoint.runtime_state),
            "key_decisions": safe_items(checkpoint.key_decisions),
            "artifact_references": safe_items(checkpoint.artifact_references),
            "miscellaneous_notes": safe_items(checkpoint.miscellaneous_notes),
            "writer_model": redact_persisted_text(
                checkpoint.writer_model,
                workspace_root=workspace_root,
            ).text,
        }
    )


def _load_index(path: Path) -> CheckpointIndex:
    if not _path_exists(path):
        return CheckpointIndex()
    _require_regular_file(path, "Checkpoint index")
    try:
        return CheckpointIndex.model_validate_json(path.read_bytes())
    except (ValidationError, ValueError) as error:
        raise CheckpointStoreCorruptionError("Checkpoint index is invalid") from error


def _select_visible(
    root: Path,
    index: CheckpointIndex,
    registrations: tuple[CheckpointCommittedData, ...],
    *,
    workspace_root: Path,
) -> tuple[Checkpoint, ...]:
    records = {record.checkpoint_id: record for record in index.checkpoints}
    visible: list[Checkpoint] = []
    previous_watermark = 0
    for registration in registrations:
        record = records.get(registration.checkpoint_id)
        if record is None or (
            record.version != registration.version
            or record.source_through_event_id != registration.source_through_event_id
            or record.content_hash != registration.content_hash
        ):
            raise CheckpointStoreCorruptionError(
                "Checkpoint commit does not match its durable index"
            )
        checkpoint = _load_checkpoint(
            root / record.relative_storage_path,
            record,
            workspace_root=workspace_root,
        )
        if checkpoint.source_through_event_id <= previous_watermark:
            raise CheckpointStoreCorruptionError("Checkpoint watermark did not advance")
        previous_watermark = checkpoint.source_through_event_id
        visible.append(checkpoint)
    return tuple(visible)


def _validate_registration_sequence(
    registrations: tuple[CheckpointCommittedData, ...],
) -> None:
    if len({item.checkpoint_id for item in registrations}) != len(registrations):
        raise CheckpointStoreCorruptionError("Duplicate checkpoint commit event")
    if tuple(item.version for item in registrations) != tuple(
        range(1, len(registrations) + 1)
    ):
        raise CheckpointStoreCorruptionError("Checkpoint commit versions are not contiguous")
    watermarks = tuple(item.source_through_event_id for item in registrations)
    if watermarks != tuple(sorted(set(watermarks))):
        raise CheckpointStoreCorruptionError("Checkpoint watermarks are not strictly increasing")


def _validate_registered_checkpoints(
    root: Path,
    index: CheckpointIndex,
    registrations: tuple[CheckpointCommittedData, ...],
    *,
    workspace_root: Path,
) -> None:
    """Validate only checkpoint files made durable by transcript commits.

    A successful stage can survive a process crash before its commit event.  Such
    a candidate has no durable authority and must not make a later resume fail.
    Conversely, every committed registration is checked before an orphan is
    ignored or removed, so recovery never discards a committed checkpoint.
    """

    records = {record.checkpoint_id: record for record in index.checkpoints}
    for registration in registrations:
        record = records.get(registration.checkpoint_id)
        if record is None or (
            record.version != registration.version
            or record.source_through_event_id != registration.source_through_event_id
            or record.content_hash != registration.content_hash
        ):
            raise CheckpointStoreCorruptionError(
                "Checkpoint commit does not match its durable index"
            )
        _load_checkpoint(
            root / record.relative_storage_path,
            record,
            workspace_root=workspace_root,
        )


def _reconcile_crash_orphans(
    root: Path,
    index: CheckpointIndex,
    registrations: tuple[CheckpointCommittedData, ...],
    *,
    writable: bool,
) -> CheckpointIndex:
    """Discard indexed candidates that lack a durable committed registration.

    Transcript commit events are the visibility authority.  This handles the
    crash boundary after ``stage()`` has durably updated the index but before
    the commit event was persisted.  Read-only consumers receive the filtered
    in-memory view; writable recovery also durably replaces the index and then
    removes only the exact orphan content paths named by that old index.
    """

    registered_ids = {registration.checkpoint_id for registration in registrations}
    orphan_records = tuple(
        record for record in index.checkpoints if record.checkpoint_id not in registered_ids
    )
    if not orphan_records:
        return index

    reconciled = CheckpointIndex(
        checkpoints=tuple(
            record for record in index.checkpoints if record.checkpoint_id in registered_ids
        )
    )
    if not writable:
        return reconciled

    try:
        _write_atomically(root / "index.json", reconciled.model_dump_json().encode())
    except OSError as error:
        raise CheckpointStoreCorruptionError(
            "Checkpoint crash-orphan cleanup is unavailable"
        ) from error
    _remove_orphan_checkpoint_files(root, orphan_records)
    return reconciled


def _remove_orphan_checkpoint_files(
    root: Path,
    orphan_records: tuple[CheckpointRecord, ...],
) -> None:
    removed_file_count = 0
    for record in orphan_records:
        try:
            os.unlink(root / record.relative_storage_path)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise CheckpointStoreCorruptionError(
                "Checkpoint crash-orphan cleanup is unavailable"
            ) from error
        removed_file_count += 1
    if removed_file_count:
        try:
            _fsync_directory(root)
        except OSError as error:
            raise CheckpointStoreCorruptionError(
                "Checkpoint crash-orphan cleanup is unavailable"
            ) from error


def _load_checkpoint(
    path: Path,
    record: CheckpointRecord,
    *,
    workspace_root: Path,
) -> Checkpoint:
    _require_regular_file(path, "Checkpoint content")
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != record.content_hash:
        raise CheckpointStoreCorruptionError("Checkpoint content hash does not match")
    try:
        checkpoint = Checkpoint.model_validate_json(encoded)
    except (ValidationError, ValueError) as error:
        raise CheckpointStoreCorruptionError("Checkpoint content is invalid") from error
    if (
        checkpoint.checkpoint_id != record.checkpoint_id
        or checkpoint.version != record.version
        or checkpoint.source_through_event_id != record.source_through_event_id
        or checkpoint.session_id != path.parent.parent.name
    ):
        raise CheckpointStoreCorruptionError("Checkpoint content does not match its index")
    if _redact_checkpoint(checkpoint, workspace_root=workspace_root) != checkpoint:
        raise CheckpointStoreCorruptionError("Checkpoint content fails durable safety validation")
    return checkpoint


def _resolve_workspace_root(workspace_root: Path) -> Path:
    try:
        resolved = workspace_root.expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise CheckpointStoreCorruptionError("Checkpoint workspace is unavailable") from error
    if not resolved.is_dir():
        raise CheckpointStoreCorruptionError("Checkpoint workspace is unavailable")
    return resolved


def _require_regular_directory(path: Path, subject: str) -> None:
    try:
        status = os.lstat(path)
    except OSError as error:
        raise CheckpointStoreCorruptionError(f"{subject} is unavailable") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise CheckpointStoreCorruptionError(f"{subject} is not a regular directory")


def _require_regular_file(path: Path, subject: str) -> None:
    try:
        status = os.lstat(path)
    except OSError as error:
        raise CheckpointStoreCorruptionError(f"{subject} is unavailable") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise CheckpointStoreCorruptionError(f"{subject} is not a regular file")


def _path_exists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise CheckpointStoreCorruptionError("Checkpoint path is unavailable") from error
    return True


def _write_atomically(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        _write_all(descriptor, content)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()


def _write_all(descriptor: int, content: bytes) -> None:
    written = 0
    while written < len(content):
        written += os.write(descriptor, content[written:])


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def is_checkpoint_id(value: str) -> bool:
    return re.fullmatch(_CHECKPOINT_ID_PATTERN, value) is not None


def new_checkpoint(
    *,
    session_id: str,
    cycle_id: int,
    version: int,
    source_from_event_id: int,
    source_through_event_id: int,
    current_intent: str,
    writer_model: str,
) -> Checkpoint:
    return Checkpoint(
        checkpoint_id=uuid4().hex,
        session_id=session_id,
        cycle_id=cycle_id,
        version=version,
        source_from_event_id=source_from_event_id,
        source_through_event_id=source_through_event_id,
        current_intent=current_intent,
        writer_model=writer_model,
        created_at=datetime.now(UTC),
    )
