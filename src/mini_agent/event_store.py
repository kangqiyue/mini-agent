"""Append-only JSONL event storage with conservative tail recovery."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from mini_agent.event_redaction import redact_event_data
from mini_agent.events import (
    CURRENT_EVENT_SCHEMA_VERSION,
    LEGACY_EVENT_SCHEMA_VERSION,
    ArtifactCreatedData,
    EventData,
    StoredEvent,
    ToolStartedData,
)
from mini_agent.redaction_types import RedactionSummary


class EventStoreCorruptionError(RuntimeError):
    """Raised when persisted events cannot be recovered without losing valid history."""


class EventStoreWriterBusyError(RuntimeError):
    """Raised when another process already owns the session writer lock."""


@dataclass(frozen=True)
class RecoveryReport:
    discarded_tail_byte_count: int = 0
    backup_path: Path | None = None


class EventStore:
    def __init__(
        self,
        path: Path,
        events: Sequence[StoredEvent],
        recovery_report: RecoveryReport,
        writer_lock_descriptor: int | None,
        workspace_root: Path | None = None,
    ) -> None:
        self.path = path
        self._events = list(events)
        self.recovery_report = recovery_report
        self._writer_lock_descriptor = writer_lock_descriptor
        self._workspace_root = workspace_root
        self._write_lock = threading.Lock()
        self._append_failure: BaseException | None = None

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        allow_recovery: bool = True,
        writable: bool = True,
        workspace_root: Path | None = None,
    ) -> EventStore:
        if allow_recovery and not writable:
            raise ValueError("Tail recovery requires a writable event store")

        if writable:
            path.parent.mkdir(parents=True, exist_ok=True)
        _require_regular_directory(path.parent)
        _require_regular_file_or_missing(path)
        writer_lock_descriptor = _acquire_writer_lock(path) if writable else None

        try:
            if not _path_exists(path):
                if not writable:
                    raise FileNotFoundError("Event store does not exist")
                _create_regular_file(path)

            raw_events = _read_regular_bytes(path)
            events, valid_byte_count, invalid_tail = _decode_events(path, raw_events)
            recovery_report = RecoveryReport()

            if invalid_tail:
                if not allow_recovery:
                    raise EventStoreCorruptionError(
                        f"Truncated event tail in {path.name}; resume the session to recover it"
                    )
                backup_path = _backup_corrupt_file(path, raw_events)
                _truncate_file(path, valid_byte_count)
                recovery_report = RecoveryReport(
                    discarded_tail_byte_count=len(invalid_tail),
                    backup_path=backup_path,
                )
            elif allow_recovery and raw_events and not raw_events.endswith(b"\n"):
                _append_bytes(path, b"\n", sync=True)

            return cls(
                path=path,
                events=events,
                recovery_report=recovery_report,
                writer_lock_descriptor=writer_lock_descriptor,
                workspace_root=workspace_root,
            )
        except BaseException:
            if writer_lock_descriptor is not None:
                _release_writer_lock(writer_lock_descriptor)
            raise

    @property
    def events(self) -> tuple[StoredEvent, ...]:
        return tuple(self._events)

    def append(
        self,
        *,
        session_id: str,
        data: EventData,
        cycle_id: int | None = None,
        turn_id: str | None = None,
        correlation_id: str | None = None,
        sync: bool = True,
    ) -> StoredEvent:
        with self._write_lock:
            if self._writer_lock_descriptor is None:
                raise RuntimeError("Cannot append through a read-only or closed event store")
            if self._append_failure is not None:
                raise RuntimeError(
                    "Cannot append after an I/O failure; close this store and resume or reopen it"
                ) from self._append_failure
            safe_data, redaction_summary = redact_event_data(
                data, workspace_root=self._workspace_root
            )
            current_cycle_id = self._events[-1].cycle_id if self._events else 1
            resolved_cycle_id = cycle_id
            if resolved_cycle_id is None:
                resolved_cycle_id = current_cycle_id
            if not self._events and resolved_cycle_id != 1:
                raise ValueError("The first event must use cycle 1")
            if self._events and not (
                current_cycle_id <= resolved_cycle_id <= current_cycle_id + 1
            ):
                raise ValueError("Event cycle must stay current or advance by one")
            event = StoredEvent(
                id=len(self._events) + 1,
                session_id=session_id,
                cycle_id=resolved_cycle_id,
                turn_id=turn_id,
                timestamp=datetime.now(UTC),
                correlation_id=correlation_id,
                redaction_summary=redaction_summary,
                data=safe_data,
            )
            encoded_event = event.model_dump_json().encode("utf-8") + b"\n"
            try:
                _append_bytes(self.path, encoded_event, sync=sync)
                self._events.append(event)
                return event
            except BaseException as error:
                # A failed append can have reached the filesystem partially or completely.
                # Keeping this instance fail-stopped prevents it from reusing its stale event id.
                self._append_failure = error
                raise

    def close(self) -> None:
        with self._write_lock:
            if self._writer_lock_descriptor is None:
                return
            _release_writer_lock(self._writer_lock_descriptor)
            self._writer_lock_descriptor = None


def _decode_events(
    _path: Path,
    raw_events: bytes,
) -> tuple[list[StoredEvent], int, bytes]:
    events: list[StoredEvent] = []
    cursor = 0
    started_event_ids_by_tool_call: dict[str, int] = {}
    artifact_ids_by_source_event_id: dict[int, str] = {}

    while cursor < len(raw_events):
        newline_index = raw_events.find(b"\n", cursor)
        is_complete_line = newline_index >= 0
        line_end = newline_index + 1 if is_complete_line else len(raw_events)
        line = raw_events[cursor : newline_index if is_complete_line else line_end]

        try:
            raw_event = cast(object, json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            if not is_complete_line:
                return events, cursor, raw_events[cursor:]
            raise EventStoreCorruptionError(
                f"Invalid event record at byte offset {cursor}"
            ) from error

        try:
            event = StoredEvent.model_validate(
                _upgrade_legacy_event_payload(
                    raw_event,
                    started_event_ids_by_tool_call=started_event_ids_by_tool_call,
                    artifact_ids_by_source_event_id=artifact_ids_by_source_event_id,
                )
            )
            _require_redaction_fixed_point(event)
        except (ValidationError, ValueError) as error:
            raise EventStoreCorruptionError(
                f"Invalid event record at byte offset {cursor}"
            ) from error

        expected_id = len(events) + 1
        if event.id != expected_id:
            raise EventStoreCorruptionError(
                f"Non-contiguous event id: expected {expected_id}, got {event.id}"
            )

        events.append(event)
        if isinstance(event.data, ToolStartedData):
            started_event_ids_by_tool_call[event.data.tool_call_id] = event.id
        elif isinstance(event.data, ArtifactCreatedData):
            artifact_ids_by_source_event_id[event.data.source_event_id] = event.data.artifact_id
        cursor = line_end

    return events, len(raw_events), b""


def _require_redaction_fixed_point(event: StoredEvent) -> None:
    """Reject persisted payloads that would still change at the redaction boundary."""
    safe_data, residual_summary = redact_event_data(event.data)
    if safe_data != event.data or residual_summary != RedactionSummary():
        raise ValueError("Persisted event data is not redaction-safe")


def _upgrade_legacy_event_payload(
    raw_event: object,
    *,
    started_event_ids_by_tool_call: dict[str, int],
    artifact_ids_by_source_event_id: dict[int, str],
) -> object:
    """Upgrade the published v1 event shape into the current v2 model.

    v1 records remain byte-for-byte on disk.  This migration exists only at
    the read boundary.  v2 records are never repaired: they are passed
    directly to the strict current model.
    """

    event_payload = _as_string_keyed_dict(raw_event)
    if event_payload is None:
        raise ValueError("Event record must be an object")

    schema_version = event_payload.get("schema_version")
    if type(schema_version) is not int:
        raise ValueError("Unsupported event schema version")
    if schema_version == CURRENT_EVENT_SCHEMA_VERSION:
        if "redaction_summary" not in event_payload:
            raise ValueError("Current event is missing redaction metadata")
        return event_payload
    if schema_version != LEGACY_EVENT_SCHEMA_VERSION:
        raise ValueError("Unsupported event schema version")

    data_payload = _as_string_keyed_dict(event_payload.get("data"))
    if data_payload is None:
        return raw_event

    data = dict(data_payload)
    kind = data.get("kind")
    if kind == "tool_completed" and "facts" not in data:
        # Tool facts were added after the published event shape. Preserve the
        # old text-based evidence behavior only for records read from disk.
        data["facts"] = {}
        data["legacy_facts_missing"] = True

    if kind == "artifact_created" and "redaction_kinds" not in data:
        # M1 persisted the count but not the categories.  Preserve that fact
        # rather than inventing a category or treating healthy history as bad.
        redaction_match_count = data.get("redaction_match_count")
        if isinstance(redaction_match_count, int) and redaction_match_count > 0:
            data["legacy_redaction_kinds_unknown"] = True
    elif kind == "tool_completed" and "artifact_id" not in data:
        tool_call_id = data.get("tool_call_id")
        if isinstance(tool_call_id, str):
            source_event_id = started_event_ids_by_tool_call.get(tool_call_id)
            artifact_id = (
                artifact_ids_by_source_event_id.get(source_event_id)
                if source_event_id is not None
                else None
            )
            if artifact_id is not None:
                # M1 stored an artifact reference only in its bounded output.
                # The source tool start creates an unambiguous durable link.
                data["artifact_id"] = artifact_id

    upgraded_event = dict(event_payload)
    upgraded_event["schema_version"] = CURRENT_EVENT_SCHEMA_VERSION
    if "redaction_summary" not in upgraded_event:
        upgraded_event["redaction_summary"] = {"match_count": 0, "kinds": []}
    upgraded_event["data"] = data
    return upgraded_event


def _as_string_keyed_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    raw_mapping = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in raw_mapping):
        return None
    return {cast(str, key): item for key, item in raw_mapping.items()}


def _backup_corrupt_file(path: Path, raw_events: bytes) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = path.with_name(f"{path.name}.recovery-{timestamp}.bin")
    file_descriptor = _open_regular_file(
        backup_path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        mode=0o600,
    )
    try:
        _write_all(file_descriptor, raw_events)
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
    _fsync_directory(path.parent)
    return backup_path


def _truncate_file(path: Path, byte_count: int) -> None:
    file_descriptor = _open_regular_file(path, os.O_WRONLY)
    try:
        os.ftruncate(file_descriptor, byte_count)
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry after creating a recovery backup."""

    _require_regular_directory(path)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_descriptor = os.open(path, directory_flags)
    except OSError as error:
        raise EventStoreCorruptionError("Event store directory is unsafe or unreadable") from error
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _append_bytes(path: Path, content: bytes, *, sync: bool) -> None:
    file_descriptor = _open_regular_file(path, os.O_WRONLY | os.O_APPEND)
    try:
        _write_all(file_descriptor, content)
        if sync:
            os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _write_all(file_descriptor: int, content: bytes) -> None:
    bytes_written = 0
    while bytes_written < len(content):
        bytes_written += os.write(file_descriptor, content[bytes_written:])


def _acquire_writer_lock(path: Path) -> int:
    lock_path = path.with_name(".writer.lock")
    _require_regular_file_or_missing(lock_path)
    lock_descriptor = _open_regular_file(lock_path, os.O_CREAT | os.O_RDWR, mode=0o600)
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(lock_descriptor)
        raise EventStoreWriterBusyError(
            f"Session already has an active writer: {path.parent.name}"
        ) from error
    return lock_descriptor


def _release_writer_lock(lock_descriptor: int) -> None:
    fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
    os.close(lock_descriptor)


def _path_exists(path: Path) -> bool:
    """Return existence without following a final symlink."""

    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise EventStoreCorruptionError("Event store path is unsafe or unreadable") from error
    return True


def _require_regular_directory(path: Path) -> None:
    try:
        path_status = os.lstat(path)
    except OSError as error:
        raise EventStoreCorruptionError("Event store directory is unsafe or unreadable") from error
    if stat.S_ISLNK(path_status.st_mode) or not stat.S_ISDIR(path_status.st_mode):
        raise EventStoreCorruptionError("Event store directory is unsafe or unreadable")


def _require_regular_file_or_missing(path: Path) -> None:
    try:
        path_status = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as error:
        raise EventStoreCorruptionError("Event store file is unsafe or unreadable") from error
    if stat.S_ISLNK(path_status.st_mode) or not stat.S_ISREG(path_status.st_mode):
        raise EventStoreCorruptionError("Event store file is unsafe or unreadable")


def _open_regular_file(path: Path, flags: int, *, mode: int = 0o600) -> int:
    """Open one non-symlink regular file and reject special files fail-closed.

    The static namespace check covers paths at the moment of open. Concurrent
    replacement after that check remains outside this local-agent threat model.
    """

    no_follow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags | no_follow, mode)
    except OSError as error:
        raise EventStoreCorruptionError("Event store file is unsafe or unreadable") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise EventStoreCorruptionError("Event store file is unsafe or unreadable")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _create_regular_file(path: Path) -> None:
    descriptor = _open_regular_file(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode=0o600)
    os.close(descriptor)
    # Persist the new directory entry. fsyncing the file's data on append does
    # not commit the filename-to-inode link, which lives in the parent directory;
    # a crash within the journal commit window could otherwise leave already
    # fsync'd events unreachable and silently lost on reopen.
    _fsync_directory(path.parent)


def _read_regular_bytes(path: Path) -> bytes:
    descriptor = _open_regular_file(path, os.O_RDONLY)
    try:
        parts: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            parts.append(chunk)
        return b"".join(parts)
    finally:
        os.close(descriptor)
