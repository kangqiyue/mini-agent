"""Durable, redacted storage for large session artifacts."""

from __future__ import annotations

import hashlib
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from mini_agent.events import ArtifactCreatedData
from mini_agent.host_path_redaction import redact_persisted_text
from mini_agent.redaction_types import (
    RedactionKind,
    RedactionSummary,
    validate_redaction_summary_fields,
)

MAX_ARTIFACT_PREVIEW_CHAR_COUNT = 512
MAX_ARTIFACT_READ_CHAR_COUNT = 65_536

_ARTIFACT_ID_PATTERN = r"^[0-9a-f]{32}$"
_MEDIA_TYPE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*$"


class ArtifactStoreCorruptionError(RuntimeError):
    """Raised when durable artifact metadata or content is inconsistent."""


class ArtifactNotFoundError(FileNotFoundError):
    """Raised when an artifact is not registered in this session."""


class ArtifactRecord(BaseModel):
    """A versioned, redacted artifact entry safe to include in active context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    source_event_id: int = Field(ge=1)
    media_type: str = Field(min_length=3, max_length=127, pattern=_MEDIA_TYPE_PATTERN)
    redacted: bool = True
    redaction_match_count: int = Field(ge=0)
    redaction_kinds: tuple[RedactionKind, ...] = ()
    char_count: int = Field(ge=0)
    byte_count: int = Field(ge=0)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    relative_storage_path: str = Field(pattern=r"^[0-9a-f]{32}\.txt$")
    head: str = Field(max_length=MAX_ARTIFACT_PREVIEW_CHAR_COUNT)
    tail: str = Field(max_length=MAX_ARTIFACT_PREVIEW_CHAR_COUNT)
    created_at: datetime

    @model_validator(mode="after")
    def validate_storage_path_and_previews(self) -> ArtifactRecord:
        expected_path = f"{self.artifact_id}.txt"
        if self.relative_storage_path != expected_path:
            raise ValueError("Artifact storage path must match its artifact id")
        if not self.redacted:
            raise ValueError("Artifacts must be persisted after redaction")
        validate_redaction_summary_fields(
            match_count=self.redaction_match_count,
            kinds=self.redaction_kinds,
            subject="Artifact redaction",
        )
        if len(self.head) > self.char_count or len(self.tail) > self.char_count:
            raise ValueError("Artifact previews cannot exceed artifact length")
        return self

    def context_reference(self, *, preview_chars: int) -> str:
        if preview_chars < 1:
            raise ValueError("Artifact preview budget must be positive")
        preview_per_side = max(1, preview_chars // 2)
        head = self.head[:preview_per_side]
        tail = self.tail[-preview_per_side:]
        return (
            f"Artifact {self.artifact_id} stores the {self.char_count}-character captured result.\n"
            f"Head:\n{head}\nTail:\n{tail}"
        )


class ArtifactIndex(BaseModel):
    """The versioned metadata index for one session's artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    artifacts: tuple[ArtifactRecord, ...] = ()

    @model_validator(mode="after")
    def require_unique_artifact_ids(self) -> ArtifactIndex:
        artifact_ids = {record.artifact_id for record in self.artifacts}
        if len(artifact_ids) != len(self.artifacts):
            raise ValueError("Artifact index contains duplicate artifact ids")
        return self


class ArtifactStore:
    """Stores only registered, redacted artifact text under one session root."""

    def __init__(
        self,
        artifact_root: Path,
        indexed_records: tuple[ArtifactRecord, ...],
        *,
        workspace_root: Path,
        visible_records: tuple[ArtifactRecord, ...],
    ) -> None:
        self._artifact_root = artifact_root
        self._workspace_root = workspace_root
        self._index_path = artifact_root / "index.json"
        self._indexed_records_by_id = {record.artifact_id: record for record in indexed_records}
        self._records_by_id = {record.artifact_id: record for record in visible_records}

    @classmethod
    def open(
        cls,
        session_root: Path,
        *,
        registrations: tuple[ArtifactCreatedData, ...],
        workspace_root: Path,
    ) -> ArtifactStore:
        """Open the artifact directory belonging to ``session_root`` only."""
        if not session_root.is_dir():
            raise FileNotFoundError(f"Session root does not exist: {session_root}")

        artifact_root = session_root / "artifacts"
        if artifact_root.is_symlink():
            raise ArtifactStoreCorruptionError("Artifact storage path cannot be a symlink")
        existed_at_open = artifact_root.exists()
        try:
            artifact_root.mkdir(mode=0o700)
        except FileExistsError as error:
            if not existed_at_open:
                if artifact_root.is_symlink() or not artifact_root.is_dir():
                    raise ArtifactStoreCorruptionError(
                        "Artifact storage path is not a directory"
                    ) from error
                _fsync_directory(session_root)
        else:
            _fsync_directory(session_root)
        if not artifact_root.is_dir():
            raise ArtifactStoreCorruptionError("Artifact storage path is not a directory")

        bound_workspace_root = _resolve_workspace_root(workspace_root)
        index_path = artifact_root / "index.json"
        if index_path.is_symlink():
            raise ArtifactStoreCorruptionError("Artifact index cannot be a symlink")
        index = _load_index(index_path)
        visible_records = _select_visible_records(index, registrations)
        _validate_artifact_files(
            artifact_root,
            index.artifacts,
            workspace_root=bound_workspace_root,
        )

        return cls(
            artifact_root=artifact_root,
            workspace_root=bound_workspace_root,
            indexed_records=index.artifacts,
            visible_records=visible_records,
        )

    @property
    def records(self) -> tuple[ArtifactRecord, ...]:
        return tuple(self._records_by_id.values())

    def create(
        self,
        content: str,
        source_event_id: int,
        media_type: str = "text/plain",
        *,
        source_redaction_summary: RedactionSummary | None = None,
        truncated_at_start: bool = False,
        truncated_at_end: bool = False,
    ) -> ArtifactRecord:
        """Persist redacted text and atomically register its metadata."""
        redaction = redact_persisted_text(
            content,
            workspace_root=self._workspace_root,
            truncated_at_start=truncated_at_start,
            truncated_at_end=truncated_at_end,
        )
        safe_content = redaction.text
        upstream_summary = source_redaction_summary or RedactionSummary()
        total_summary = RedactionSummary(
            match_count=upstream_summary.match_count + redaction.match_count,
            kinds=tuple(
                sorted(
                    {*upstream_summary.kinds, *redaction.matched_kinds},
                    key=lambda kind: kind.value,
                )
            ),
        )
        encoded_content = safe_content.encode("utf-8")
        artifact_id = self._new_artifact_id()
        record = _build_record(
            artifact_id=artifact_id,
            source_event_id=source_event_id,
            media_type=media_type,
            content=safe_content,
            encoded_content=encoded_content,
            redaction_match_count=total_summary.match_count,
            redaction_kinds=total_summary.kinds,
        )
        content_path = self._artifact_path(record.artifact_id)

        _write_atomically(content_path, encoded_content)
        indexed_records = tuple(self._indexed_records_by_id.values())
        updated_index = ArtifactIndex(artifacts=(*indexed_records, record))
        _write_atomically(
            self._index_path,
            updated_index.model_dump_json().encode("utf-8"),
        )
        self._indexed_records_by_id[record.artifact_id] = record
        self._records_by_id[record.artifact_id] = record
        return record

    def read(self, artifact_id: str, offset: int, limit: int) -> str:
        """Read a bounded character segment of a registered artifact."""
        _validate_read_window(offset=offset, limit=limit)
        record = self._registered_record(artifact_id)
        encoded_content = self._artifact_path(record.artifact_id).read_bytes()
        content = _decode_and_validate_content(
            record,
            encoded_content,
            workspace_root=self._workspace_root,
        )
        return content[offset : offset + limit]

    def _new_artifact_id(self) -> str:
        for _ in range(10):
            artifact_id = uuid4().hex
            is_new_id = artifact_id not in self._indexed_records_by_id
            is_new_path = not self._artifact_path(artifact_id).exists()
            if is_new_id and is_new_path:
                return artifact_id
        raise RuntimeError("Could not allocate a unique artifact id")

    def _registered_record(self, artifact_id: str) -> ArtifactRecord:
        _validate_artifact_id(artifact_id)
        record = self._records_by_id.get(artifact_id)
        if record is None:
            raise ArtifactNotFoundError("Artifact is not registered")
        return record

    def _artifact_path(self, artifact_id: str) -> Path:
        _validate_artifact_id(artifact_id)
        return self._artifact_root / f"{artifact_id}.txt"


def _build_record(
    *,
    artifact_id: str,
    source_event_id: int,
    media_type: str,
    content: str,
    encoded_content: bytes,
    redaction_match_count: int,
    redaction_kinds: tuple[RedactionKind, ...],
) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=artifact_id,
        source_event_id=source_event_id,
        media_type=media_type,
        redaction_match_count=redaction_match_count,
        redaction_kinds=redaction_kinds,
        char_count=len(content),
        byte_count=len(encoded_content),
        content_hash=hashlib.sha256(encoded_content).hexdigest(),
        relative_storage_path=f"{artifact_id}.txt",
        head=content[:MAX_ARTIFACT_PREVIEW_CHAR_COUNT],
        tail=content[-MAX_ARTIFACT_PREVIEW_CHAR_COUNT:],
        created_at=datetime.now(UTC),
    )


def _load_index(index_path: Path) -> ArtifactIndex:
    if not index_path.exists():
        return ArtifactIndex()
    if not index_path.is_file():
        raise ArtifactStoreCorruptionError("Artifact index is not a regular file")

    try:
        return ArtifactIndex.model_validate_json(index_path.read_bytes())
    except (ValidationError, ValueError) as error:
        message = "Artifact index has an unsupported or invalid schema"
        raise ArtifactStoreCorruptionError(message) from error


def _select_visible_records(
    index: ArtifactIndex,
    registrations: tuple[ArtifactCreatedData, ...],
) -> tuple[ArtifactRecord, ...]:
    registrations_by_id = {registration.artifact_id: registration for registration in registrations}
    if len(registrations_by_id) != len(registrations):
        raise ArtifactStoreCorruptionError("Transcript contains duplicate artifact registrations")

    records_by_id = {record.artifact_id: record for record in index.artifacts}
    missing_artifact_ids = registrations_by_id.keys() - records_by_id.keys()
    if missing_artifact_ids:
        raise ArtifactStoreCorruptionError("Transcript references missing artifact metadata")

    for artifact_id, registration in registrations_by_id.items():
        _validate_registration(records_by_id[artifact_id], registration)
    return tuple(record for record in index.artifacts if record.artifact_id in registrations_by_id)


def _validate_registration(
    record: ArtifactRecord,
    registration: ArtifactCreatedData,
) -> None:
    expected_registration = ArtifactCreatedData(
        artifact_id=record.artifact_id,
        source_event_id=record.source_event_id,
        media_type=record.media_type,
        char_count=record.char_count,
        content_hash=record.content_hash,
        redaction_match_count=record.redaction_match_count,
        redaction_kinds=record.redaction_kinds,
    )
    # M1 event records did not carry category names.  The index may have
    # retained them, but the historical event cannot be required to know them.
    # Compare all durable facts it did record instead.
    if registration.legacy_redaction_kinds_unknown and (
        registration.artifact_id == record.artifact_id
        and registration.source_event_id == record.source_event_id
        and registration.media_type == record.media_type
        and registration.char_count == record.char_count
        and registration.content_hash == record.content_hash
        and registration.redaction_match_count == record.redaction_match_count
    ):
        return
    if registration != expected_registration:
        raise ArtifactStoreCorruptionError("Artifact registration does not match index")


def _validate_artifact_files(
    artifact_root: Path,
    records: tuple[ArtifactRecord, ...],
    *,
    workspace_root: Path,
) -> None:
    for record in records:
        artifact_path = artifact_root / record.relative_storage_path
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise ArtifactStoreCorruptionError("Registered artifact content is missing")
        _decode_and_validate_content(
            record,
            artifact_path.read_bytes(),
            workspace_root=workspace_root,
        )


def _decode_and_validate_content(
    record: ArtifactRecord,
    encoded_content: bytes,
    *,
    workspace_root: Path,
) -> str:
    try:
        content = encoded_content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArtifactStoreCorruptionError("Artifact content is not valid UTF-8") from error

    content_hash = hashlib.sha256(encoded_content).hexdigest()
    if content_hash != record.content_hash:
        raise ArtifactStoreCorruptionError("Artifact content hash does not match")
    if len(encoded_content) != record.byte_count or len(content) != record.char_count:
        raise ArtifactStoreCorruptionError("Artifact content size does not match")
    expected_head = content[:MAX_ARTIFACT_PREVIEW_CHAR_COUNT]
    if record.head != expected_head:
        raise ArtifactStoreCorruptionError("Artifact content head preview does not match")
    expected_tail = content[-MAX_ARTIFACT_PREVIEW_CHAR_COUNT:]
    if record.tail != expected_tail:
        raise ArtifactStoreCorruptionError("Artifact content tail preview does not match")
    if redact_persisted_text(content, workspace_root=workspace_root).text != content:
        raise ArtifactStoreCorruptionError("Artifact content fails durable safety validation")
    return content


def _resolve_workspace_root(workspace_root: Path) -> Path:
    try:
        resolved = workspace_root.expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ArtifactStoreCorruptionError("Artifact workspace is unavailable") from error
    if not resolved.is_dir():
        raise ArtifactStoreCorruptionError("Artifact workspace is unavailable")
    return resolved


def _validate_artifact_id(artifact_id: str) -> None:
    if re.fullmatch(_ARTIFACT_ID_PATTERN, artifact_id) is None:
        raise ValueError("Artifact id must be a 32-character lowercase hexadecimal value")


def _validate_read_window(*, offset: int, limit: int) -> None:
    if offset < 0:
        raise ValueError("Artifact read offset must be non-negative")
    if limit < 1 or limit > MAX_ARTIFACT_READ_CHAR_COUNT:
        message = f"Artifact read limit must be between 1 and {MAX_ARTIFACT_READ_CHAR_COUNT}"
        raise ValueError(message)


def _write_atomically(path: Path, content: bytes) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    file_descriptor: int | None = None
    try:
        file_descriptor = os.open(temporary_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        _write_all(file_descriptor, content)
        os.fsync(file_descriptor)
        os.close(file_descriptor)
        file_descriptor = None
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if temporary_path.exists():
            temporary_path.unlink()


def _write_all(file_descriptor: int, content: bytes) -> None:
    bytes_written = 0
    while bytes_written < len(content):
        bytes_written += os.write(file_descriptor, content[bytes_written:])


def _fsync_directory(directory: Path) -> None:
    directory_descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
