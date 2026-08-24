from datetime import UTC, datetime

import pytest

from mini_agent.events import StoredEvent, UserMessageData
from mini_agent.history import HistoryPage, SessionHistory
from mini_agent.tools.base import ToolError
from mini_agent.tools.history_read import HistoryReadTool


def test_history_read_returns_a_bounded_inclusive_range() -> None:
    history = SessionHistory((_event(1, "one"), _event(2, "two")))
    tool = HistoryReadTool(lambda: history)

    result = tool.execute('{"from_event_id":1,"to_event_id":2,"max_events":1}')

    page = HistoryPage.model_validate_json(result.content)
    assert [match.event_id for match in page.matches] == [1]
    assert page.next_after_event_id == 1
    assert result.is_truncated is True


def test_history_read_rejects_reversed_or_unknown_arguments() -> None:
    history = SessionHistory((_event(1, "one"),))
    tool = HistoryReadTool(lambda: history)

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"from_event_id":2,"to_event_id":1}')

    assert error_info.value.code == "invalid_arguments"


def test_history_read_exposes_a_cursor_for_an_oversized_event() -> None:
    event = _event(1, "x" * 100)
    history = SessionHistory((event,))
    tool = HistoryReadTool(lambda: history)

    first_result = tool.execute('{"from_event_id":1,"to_event_id":1,"max_chars":25}')
    first_page = HistoryPage.model_validate_json(first_result.content)
    assert first_page.next_after_event_id is None
    assert first_page.next_cursor is not None

    second_result = tool.execute(
        "{"
        f'\"from_event_id\":{first_page.next_cursor.event_id},'
        '\"to_event_id\":1,'
        f'\"start_offset\":{first_page.next_cursor.char_offset},'
        '\"max_chars\":25}'
    )
    second_page = HistoryPage.model_validate_json(second_result.content)

    assert second_page.matches[0].snippet == event.model_dump_json()[25:50]
    assert second_page.next_cursor is not None
    assert second_page.next_cursor.char_offset == 50


def _event(event_id: int, content: str) -> StoredEvent:
    return StoredEvent(
        id=event_id,
        session_id="session",
        timestamp=datetime(2026, 8, 12, tzinfo=UTC),
        data=UserMessageData(content=content),
    )
