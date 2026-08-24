"""Bounded retrieval over the already-persisted events of one session."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mini_agent.events import StoredEvent
from mini_agent.host_path_redaction import redact_host_paths

type HistoryEventKind = Literal[
    "approval_requested",
    "approval_resolved",
    "artifact_created",
    "checkpoint_started",
    "checkpoint_committed",
    "checkpoint_failed",
    "context_estimated",
    "context_pruned",
    "goal_created",
    "goal_updated",
    "goal_status_changed",
    "rebuild_started",
    "rebuild_completed",
    "rebuild_failed",
    "session_started",
    "session_resumed",
    "session_stopped",
    "user_message",
    "model_request_started",
    "assistant_message",
    "model_request_failed",
    "tool_requested",
    "tool_started",
    "tool_completed",
    "tool_failed",
    "tool_interrupted",
]

_MAX_QUERY_CHARS = 512
_MAX_SEARCH_LIMIT = 50
_MAX_SNIPPET_CHARS = 1_000
_MAX_READ_EVENTS = 100
_MAX_READ_CHARS = 20_000


class HistoryMatch(BaseModel):
    """A bounded representation of one persisted event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: int = Field(ge=1)
    timestamp: datetime
    kind: HistoryEventKind
    snippet: str = Field(min_length=1, max_length=_MAX_READ_CHARS)
    is_truncated: bool


class HistoryCursor(BaseModel):
    """Position for continuing a partially returned serialized event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: int = Field(ge=1)
    char_offset: int = Field(
        ge=1,
        description="Serialized JSON character offset at which to resume the event.",
    )


class HistoryPage(BaseModel):
    """A bounded result page.

    ``next_after_event_id`` is the last event returned before further matching
    events. Pass it as ``after_event_id`` to continue a search. For range reads,
    start the next read at ``next_after_event_id + 1``.

    When a range read ends within one oversized event, ``next_cursor`` instead
    identifies the same event and the character offset at which to resume. In
    that case, ``next_after_event_id`` is ``None`` because advancing by event ID
    would skip unread content.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    matches: tuple[HistoryMatch, ...]
    is_truncated: bool
    next_after_event_id: int | None = Field(default=None, ge=1)
    next_cursor: HistoryCursor | None = None


class SessionHistory:
    """Search and read a single in-memory snapshot of validated session events.

    The constructor accepts events rather than a session path or an event-store
    object so this module cannot read arbitrary files. Callers should pass the
    current ``store.events`` snapshot.
    """

    def __init__(
        self,
        events: Sequence[StoredEvent],
        *,
        upper_bound_event_id: int | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        self._events = tuple(events)
        _validate_event_order(self._events)
        if upper_bound_event_id is not None and (
            type(upper_bound_event_id) is not int or upper_bound_event_id < 1
        ):
            raise ValueError("upper_bound_event_id must be a positive integer")
        self._upper_bound_event_id = upper_bound_event_id
        self._workspace_root = workspace_root

    def bounded_before(self, event_id: int) -> SessionHistory:
        """Return an immutable view containing only events before ``event_id``."""

        if type(event_id) is not int or event_id < 1:
            raise ValueError("event_id must be a positive integer")
        upper_bound = event_id
        if self._upper_bound_event_id is not None:
            upper_bound = min(upper_bound, self._upper_bound_event_id)
        return SessionHistory(
            self._events,
            upper_bound_event_id=upper_bound,
            workspace_root=self._workspace_root,
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        kinds: Iterable[HistoryEventKind] | None = None,
        before_event_id: int | None = None,
        after_event_id: int | None = None,
        snippet_chars: int = 500,
    ) -> HistoryPage:
        """Search serialized, redacted event records in ascending event-id order.

        ``before_event_id`` and ``after_event_id`` are exclusive bounds. Search
        snippets are intentionally bounded; use :meth:`read` for a selected event.
        """

        normalized_query = _validate_query(query)
        _validate_limit(limit, maximum=_MAX_SEARCH_LIMIT, name="limit")
        _validate_limit(
            snippet_chars,
            maximum=_MAX_SNIPPET_CHARS,
            name="snippet_chars",
        )
        _validate_event_bounds(before_event_id, after_event_id)
        allowed_kinds = _validate_kinds(kinds)

        matches: list[HistoryMatch] = []
        has_more_matches = False
        for event in self._filtered_events(
            before_event_id=before_event_id,
            after_event_id=after_event_id,
            allowed_kinds=allowed_kinds,
        ):
            serialized_event = self._serialized_event(event)
            if normalized_query.casefold() not in serialized_event.casefold():
                continue
            if len(matches) == limit:
                has_more_matches = True
                break
            matches.append(_search_match(event, serialized_event, normalized_query, snippet_chars))

        next_after_event_id = matches[-1].event_id if has_more_matches else None
        is_truncated = has_more_matches or any(match.is_truncated for match in matches)
        return HistoryPage(
            matches=tuple(matches),
            is_truncated=is_truncated,
            next_after_event_id=next_after_event_id,
        )

    def read(
        self,
        *,
        from_event_id: int,
        to_event_id: int,
        max_events: int = 20,
        max_chars: int = 8_000,
        start_offset: int = 0,
    ) -> HistoryPage:
        """Read an inclusive event-id range under event and character budgets.

        ``start_offset`` skips that many serialized JSON characters from
        ``from_event_id`` only. If a record exceeds the character budget, the
        returned ``next_cursor`` can be used as the next ``from_event_id`` and
        ``start_offset`` pair to read the rest of that same record.
        """

        _validate_read_range(from_event_id, to_event_id)
        _validate_limit(max_events, maximum=_MAX_READ_EVENTS, name="max_events")
        _validate_limit(max_chars, maximum=_MAX_READ_CHARS, name="max_chars")
        _validate_start_offset(start_offset)

        matches: list[HistoryMatch] = []
        remaining_chars = max_chars
        has_more_events = False
        next_cursor: HistoryCursor | None = None
        for event in self._events:
            if self._upper_bound_event_id is not None and event.id >= self._upper_bound_event_id:
                break
            if event.id < from_event_id:
                continue
            if event.id > to_event_id:
                break
            if len(matches) == max_events or remaining_chars == 0:
                has_more_events = True
                break

            serialized_event = self._serialized_event(event)
            event_start_offset = start_offset if event.id == from_event_id else 0
            if event_start_offset >= len(serialized_event):
                continue

            remaining_event = serialized_event[event_start_offset:]
            snippet = _truncate(remaining_event, remaining_chars)
            matches.append(
                HistoryMatch(
                    event_id=event.id,
                    timestamp=event.timestamp,
                    kind=event.data.kind,
                    snippet=snippet,
                    is_truncated=len(snippet) < len(remaining_event),
                )
            )
            remaining_chars -= len(snippet)
            if len(snippet) < len(remaining_event):
                next_cursor = HistoryCursor(
                    event_id=event.id,
                    char_offset=event_start_offset + len(snippet),
                )
                break

        next_after_event_id = (
            matches[-1].event_id if has_more_events and matches and next_cursor is None else None
        )
        is_truncated = has_more_events or next_cursor is not None
        return HistoryPage(
            matches=tuple(matches),
            is_truncated=is_truncated,
            next_after_event_id=next_after_event_id,
            next_cursor=next_cursor,
        )

    def _serialized_event(self, event: StoredEvent) -> str:
        serialized_event = event.model_dump_json()
        if self._workspace_root is None:
            return serialized_event
        return redact_host_paths(
            serialized_event,
            workspace_root=self._workspace_root,
        ).text

    def _filtered_events(
        self,
        *,
        before_event_id: int | None,
        after_event_id: int | None,
        allowed_kinds: frozenset[HistoryEventKind] | None,
    ) -> Iterable[StoredEvent]:
        for event in self._events:
            if self._upper_bound_event_id is not None and event.id >= self._upper_bound_event_id:
                break
            if before_event_id is not None and event.id >= before_event_id:
                continue
            if after_event_id is not None and event.id <= after_event_id:
                continue
            if allowed_kinds is not None and event.data.kind not in allowed_kinds:
                continue
            yield event


def search(
    events: Sequence[StoredEvent],
    query: str,
    *,
    limit: int = 10,
    kinds: Iterable[HistoryEventKind] | None = None,
    before_event_id: int | None = None,
    after_event_id: int | None = None,
    snippet_chars: int = 500,
) -> HistoryPage:
    """Search the current ``EventStore.events`` snapshot without filesystem access."""

    return SessionHistory(events).search(
        query,
        limit=limit,
        kinds=kinds,
        before_event_id=before_event_id,
        after_event_id=after_event_id,
        snippet_chars=snippet_chars,
    )


def read(
    events: Sequence[StoredEvent],
    *,
    from_event_id: int,
    to_event_id: int,
    max_events: int = 20,
    max_chars: int = 8_000,
    start_offset: int = 0,
) -> HistoryPage:
    """Read the current ``EventStore.events`` snapshot without filesystem access."""

    return SessionHistory(events).read(
        from_event_id=from_event_id,
        to_event_id=to_event_id,
        max_events=max_events,
        max_chars=max_chars,
        start_offset=start_offset,
    )


def _search_match(
    event: StoredEvent,
    serialized_event: str,
    query: str,
    snippet_chars: int,
) -> HistoryMatch:
    match_start = serialized_event.casefold().find(query.casefold())
    snippet = _snippet_around(serialized_event, match_start, snippet_chars)
    return HistoryMatch(
        event_id=event.id,
        timestamp=event.timestamp,
        kind=event.data.kind,
        snippet=snippet,
        is_truncated=len(snippet) < len(serialized_event),
    )


def _snippet_around(value: str, match_start: int, maximum_chars: int) -> str:
    if len(value) <= maximum_chars:
        return value

    context_before = maximum_chars // 3
    start = max(0, match_start - context_before)
    end = min(len(value), start + maximum_chars)
    start = max(0, end - maximum_chars)
    return value[start:end]


def _truncate(value: str, maximum_chars: int) -> str:
    return value[:maximum_chars]


def _validate_event_order(events: tuple[StoredEvent, ...]) -> None:
    previous_id = 0
    for event in events:
        if event.id <= previous_id:
            raise ValueError("History events must be strictly ordered by event id")
        previous_id = event.id


def _validate_query(query: str) -> str:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be blank")
    if len(normalized_query) > _MAX_QUERY_CHARS:
        raise ValueError(f"query must be at most {_MAX_QUERY_CHARS} characters")
    return normalized_query


def _validate_limit(value: int, *, maximum: int, name: str) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")


def _validate_event_bounds(before_event_id: int | None, after_event_id: int | None) -> None:
    for name, value in (("before_event_id", before_event_id), ("after_event_id", after_event_id)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    if (
        before_event_id is not None
        and after_event_id is not None
        and after_event_id >= before_event_id
    ):
        raise ValueError("after_event_id must be less than before_event_id")


def _validate_read_range(from_event_id: int, to_event_id: int) -> None:
    for name, value in (("from_event_id", from_event_id), ("to_event_id", to_event_id)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if from_event_id > to_event_id:
        raise ValueError("from_event_id must be less than or equal to to_event_id")


def _validate_start_offset(start_offset: int) -> None:
    if type(start_offset) is not int or start_offset < 0:
        raise ValueError("start_offset must be a non-negative integer")


def _validate_kinds(
    kinds: Iterable[HistoryEventKind] | None,
) -> frozenset[HistoryEventKind] | None:
    if kinds is None:
        return None

    allowed_kinds = frozenset(kinds)
    if not allowed_kinds:
        raise ValueError("kinds must not be empty when provided")

    valid_kinds = {
        "approval_requested",
        "approval_resolved",
        "artifact_created",
        "checkpoint_started",
        "checkpoint_committed",
        "checkpoint_failed",
        "context_estimated",
        "context_pruned",
        "goal_created",
        "goal_updated",
        "goal_status_changed",
        "rebuild_started",
        "rebuild_completed",
        "rebuild_failed",
        "session_started",
        "session_resumed",
        "session_stopped",
        "user_message",
        "model_request_started",
        "assistant_message",
        "model_request_failed",
        "tool_requested",
        "tool_started",
        "tool_completed",
        "tool_failed",
        "tool_interrupted",
    }
    invalid_kinds = allowed_kinds - valid_kinds
    if invalid_kinds:
        raise ValueError(f"Unknown history event kinds: {sorted(invalid_kinds)}")
    return allowed_kinds
