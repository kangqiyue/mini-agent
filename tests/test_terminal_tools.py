from datetime import UTC, datetime
from io import StringIO

import pytest
from rich.console import Console

from mini_agent.events import (
    EventData,
    StoredEvent,
    ToolCompletedData,
    ToolFailedData,
    ToolInterruptedData,
    ToolRecoveryStatus,
    ToolRequestedData,
    ToolStartedData,
)
from mini_agent.messages import ToolCall
from mini_agent.terminal_tools import OutputMode, TerminalToolOutput
from mini_agent.tool_facts import ToolCompletionFacts
from tests.support.synthetic_secrets import synthetic_stripe_access_token


def _event(data: EventData) -> StoredEvent:
    return StoredEvent(id=1, session_id="session", timestamp=datetime.now(UTC), data=data)


@pytest.mark.parametrize("mode", list(OutputMode))
def test_tool_activity_shows_completion_and_limits_body_to_detailed_mode(mode: OutputMode) -> None:
    output = StringIO()
    view = TerminalToolOutput(Console(file=output, width=120), mode)
    call = ToolCall(id="call", name="read_file", arguments_json='{"path":"notes.txt"}')

    for data in (
        ToolRequestedData(tool_call=call, is_read_only=True),
        ToolStartedData(tool_call_id=call.id, tool_name=call.name),
        ToolCompletedData(tool_call_id=call.id, output="saved body", facts=ToolCompletionFacts()),
    ):
        view.show_event(_event(data))

    rendered = output.getvalue()
    assert "→ read_file" in rendered
    assert "notes.txt" in rendered
    assert "✓ read_file · 1 output lines" in rendered
    assert ("saved body" in rendered) is (mode is OutputMode.DETAILED)


@pytest.mark.parametrize(("exit_code", "symbol"), [(0, "✓"), (1, "✗"), (-9, "✗")])
def test_command_exit_status_is_not_confused_with_success(exit_code: int, symbol: str) -> None:
    output = StringIO()
    view = TerminalToolOutput(Console(file=output, width=120))
    call = ToolCall(id="call", name="exec_command", arguments_json='{"argv":["false"]}')
    view.remember_calls((call,))

    view.show_event(
        _event(
            ToolCompletedData(
                tool_call_id=call.id,
                output="command result",
                facts=ToolCompletionFacts(command_exit_code=exit_code),
            )
        )
    )

    assert f"{symbol} exec_command · exit {exit_code}" in output.getvalue()


@pytest.mark.parametrize(
    "data",
    [
        ToolFailedData(tool_call_id="call", error_code="permission_denied", message="User denied"),
        ToolInterruptedData(
            tool_call_id="call",
            recovery_status=ToolRecoveryStatus.UNKNOWN,
            reason="Review required",
        ),
    ],
)
def test_failure_and_unknown_status_are_visible_even_in_brief_mode(data: EventData) -> None:
    output = StringIO()
    view = TerminalToolOutput(Console(file=output, width=120))

    view.show_event(_event(data))

    rendered = output.getvalue()
    assert "✓" not in rendered
    assert "permission_denied" in rendered or "unknown" in rendered


@pytest.mark.parametrize("body", ["line\n" * 100, "long" * 2_000])
def test_detailed_output_is_bounded_without_changing_the_saved_result(body: str) -> None:
    output = StringIO()
    view = TerminalToolOutput(Console(file=output, width=120), OutputMode.DETAILED)
    event = _event(
        ToolCompletedData(
            tool_call_id="call", output=body + "last-sentinel", facts=ToolCompletionFacts()
        )
    )

    view.show_event(event)

    assert "display limited" in output.getvalue()
    assert "last-sentinel" not in output.getvalue()
    assert isinstance(event.data, ToolCompletedData)
    assert event.data.output == body + "last-sentinel"


def test_detailed_output_redacts_credentials_and_escapes_terminal_controls() -> None:
    output = StringIO()
    view = TerminalToolOutput(Console(file=output, width=120), OutputMode.DETAILED)
    secret = synthetic_stripe_access_token("TERMINALTOOL")

    view.show_output(f"{secret}\n\x1b]52;c;clipboard\x1b\\ [bold]literal[/bold]")

    rendered = output.getvalue()
    assert secret not in rendered
    assert "REDACTED" in rendered
    assert "\x1b" not in rendered
    assert "\\x1B]52;c;clipboard" in rendered
    assert "[bold]literal[/bold]" in rendered


def test_brief_completion_shows_full_output_artifact_reference() -> None:
    output = StringIO()
    view = TerminalToolOutput(Console(file=output, width=120))

    view.show_event(
        _event(
            ToolCompletedData(
                tool_call_id="call",
                output="preview",
                artifact_id="a" * 32,
                facts=ToolCompletionFacts(),
            )
        )
    )

    assert "full output: artifact " + "a" * 32 in output.getvalue()
