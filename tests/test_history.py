from datetime import UTC, datetime

import pytest

from mini_agent.events import AssistantMessageData, StoredEvent, UserMessageData
from mini_agent.history import HistoryCursor, HistoryEventKind, SessionHistory, read, search
from mini_agent.messages import FinishReason


def test_search_uses_persisted_event_json_and_returns_bounded_snippet() -> None:
    events = (
        _event(1, UserMessageData(content="please investigate context compression")),
        _event(
            2,
            AssistantMessageData(
                content="I will inspect the repository",
                finish_reason=FinishReason.STOP,
            ),
        ),
    )

    page = search(events, "compression", snippet_chars=24)

    assert [match.event_id for match in page.matches] == [1]
    assert page.matches[0].kind == "user_message"
    assert len(page.matches[0].snippet) == 24
    assert page.matches[0].is_truncated is True
    assert page.is_truncated is True
    assert page.next_after_event_id is None


def test_search_applies_kind_and_exclusive_event_id_bounds_in_stable_order() -> None:
    events = (
        _event(1, UserMessageData(content="find issue alpha")),
        _event(
            2,
            AssistantMessageData(
                content="issue alpha resolved",
                finish_reason=FinishReason.STOP,
            ),
        ),
        _event(3, UserMessageData(content="find issue alpha again")),
    )

    page = search(
        events,
        "issue alpha",
        kinds=("user_message",),
        after_event_id=1,
        before_event_id=4,
    )

    assert [match.event_id for match in page.matches] == [3]
    assert page.matches[0].kind == "user_message"


def test_search_exposes_next_cursor_when_limit_leaves_matching_events() -> None:
    events = tuple(_event(index, UserMessageData(content="needle")) for index in range(1, 4))

    first_page = search(events, "needle", limit=2)

    assert [match.event_id for match in first_page.matches] == [1, 2]
    assert first_page.is_truncated is True
    assert first_page.next_after_event_id == 2
    second_page = search(events, "needle", after_event_id=first_page.next_after_event_id)
    assert [match.event_id for match in second_page.matches] == [3]
    assert second_page.is_truncated is False
    assert second_page.next_after_event_id is None


@pytest.mark.parametrize(
    ("query", "limit", "snippet_chars", "message"),
    [
        (" ", 1, 1, "query must not be blank"),
        ("a" * 513, 1, 1, "query must be at most"),
        ("needle", 0, 1, "limit must be between"),
        ("needle", 51, 1, "limit must be between"),
        ("needle", 1, 1_001, "snippet_chars must be between"),
    ],
)
def test_search_rejects_invalid_query_and_budgets(
    query: str,
    limit: int,
    snippet_chars: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        search(
            (_event(1, UserMessageData(content="needle")),),
            query,
            limit=limit,
            snippet_chars=snippet_chars,
        )


def test_search_rejects_invalid_event_bounds_and_unknown_kind() -> None:
    events = (_event(1, UserMessageData(content="needle")),)

    with pytest.raises(ValueError, match="after_event_id must be less"):
        search(events, "needle", before_event_id=1, after_event_id=1)
    with pytest.raises(ValueError, match="Unknown history event kinds"):
        search(events, "needle", kinds=("not_an_event",))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kind",
    (
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
    ),
)
def test_search_accepts_every_declared_runtime_event_kind(kind: HistoryEventKind) -> None:
    events = (_event(1, UserMessageData(content="needle")),)

    page = search(events, "needle", kinds=(kind,))

    assert page.matches == ()


def test_read_is_inclusive_and_has_a_stable_continuation_boundary() -> None:
    events = tuple(
        _event(index, UserMessageData(content=f"event {index}")) for index in range(1, 4)
    )

    first_page = read(events, from_event_id=1, to_event_id=3, max_events=2, max_chars=20_000)

    assert [match.event_id for match in first_page.matches] == [1, 2]
    assert first_page.is_truncated is True
    assert first_page.next_after_event_id == 2
    second_page = read(
        events,
        from_event_id=first_page.next_after_event_id + 1,
        to_event_id=3,
        max_chars=20_000,
    )
    assert [match.event_id for match in second_page.matches] == [3]


def test_read_marks_an_oversized_event_without_exceeding_the_character_budget() -> None:
    history = SessionHistory((_event(1, UserMessageData(content="x" * 100)),))

    page = history.read(from_event_id=1, to_event_id=1, max_chars=25)

    assert len(page.matches) == 1
    assert len(page.matches[0].snippet) == 25
    assert page.matches[0].is_truncated is True
    assert page.is_truncated is True
    assert page.next_after_event_id is None
    assert page.next_cursor == HistoryCursor(event_id=1, char_offset=25)


def test_read_continues_an_oversized_event_from_a_character_cursor() -> None:
    event = _event(1, UserMessageData(content="x" * 100))
    history = SessionHistory((event,))
    expected = event.model_dump_json()
    start_offset = 0
    snippets: list[str] = []

    while True:
        page = history.read(
            from_event_id=1,
            to_event_id=1,
            max_chars=25,
            start_offset=start_offset,
        )
        snippets.extend(match.snippet for match in page.matches)
        if page.next_cursor is None:
            break
        assert page.next_after_event_id is None
        assert page.next_cursor.event_id == 1
        start_offset = page.next_cursor.char_offset

    assert "".join(snippets) == expected


@pytest.mark.parametrize(
    ("from_event_id", "to_event_id", "max_events", "max_chars", "start_offset", "message"),
    [
        (0, 1, 1, 1, 0, "from_event_id must be a positive"),
        (2, 1, 1, 1, 0, "from_event_id must be less than or equal"),
        (1, 1, 0, 1, 0, "max_events must be between"),
        (1, 1, 1, 20_001, 0, "max_chars must be between"),
        (1, 1, 1, 1, -1, "start_offset must be a non-negative"),
    ],
)
def test_read_rejects_invalid_range_and_budgets(
    from_event_id: int,
    to_event_id: int,
    max_events: int,
    max_chars: int,
    start_offset: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        read(
            (_event(1, UserMessageData(content="needle")),),
            from_event_id=from_event_id,
            to_event_id=to_event_id,
            max_events=max_events,
            max_chars=max_chars,
            start_offset=start_offset,
        )


def test_history_rejects_events_that_are_not_strictly_ordered() -> None:
    event = _event(1, UserMessageData(content="needle"))

    with pytest.raises(ValueError, match="strictly ordered"):
        SessionHistory((event, event))


def _event(event_id: int, data: UserMessageData | AssistantMessageData) -> StoredEvent:
    return StoredEvent(
        id=event_id,
        session_id="session",
        timestamp=datetime(2026, 8, 12, tzinfo=UTC),
        data=data,
    )
