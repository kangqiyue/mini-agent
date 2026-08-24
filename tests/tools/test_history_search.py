from datetime import UTC, datetime

import pytest

from mini_agent.events import StoredEvent, UserMessageData
from mini_agent.history import HistoryPage, SessionHistory
from mini_agent.tools.base import ToolError
from mini_agent.tools.history_search import HistorySearchTool


def test_history_search_reads_only_the_injected_snapshot() -> None:
    history = SessionHistory(
        (
            _event(1, "first session-only record"),
            _event(2, "second record"),
        )
    )
    tool = HistorySearchTool(lambda: history)

    result = tool.execute('{"query":"session-only"}')

    page = HistoryPage.model_validate_json(result.content)
    assert [match.event_id for match in page.matches] == [1]
    assert result.is_truncated is False
    properties = tool.definition.parameters["properties"]
    assert isinstance(properties, dict)
    assert "session_id" not in properties
    assert "path" not in properties


def test_history_search_rejects_unknown_or_invalid_arguments() -> None:
    history = SessionHistory((_event(1, "record"),))
    tool = HistorySearchTool(lambda: history)

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"query":"record","unexpected":true}')

    assert error_info.value.code == "invalid_arguments"


def test_history_search_uses_latest_factory_snapshot() -> None:
    events = [_event(1, "first")]
    tool = HistorySearchTool(lambda: SessionHistory(tuple(events)))
    events.append(_event(2, "newly persisted marker"))

    result = tool.execute('{"query":"newly persisted marker"}')

    page = HistoryPage.model_validate_json(result.content)
    assert [match.event_id for match in page.matches] == [2]


def _event(event_id: int, content: str) -> StoredEvent:
    return StoredEvent(
        id=event_id,
        session_id="session",
        timestamp=datetime(2026, 8, 12, tzinfo=UTC),
        data=UserMessageData(content=content),
    )
