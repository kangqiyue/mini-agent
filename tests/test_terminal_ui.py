from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.text import Text

from mini_agent.agent import AgentStatus
from mini_agent.context import ContextProjection, RequestTokenEstimate
from mini_agent.goal import GoalStatus
from mini_agent.messages import ConversationMessage, MessageRole, ToolCall
from mini_agent.session import SessionSummary
from mini_agent.system_prompt import GitContext, RuntimeContext
from mini_agent.terminal_tools import OutputMode
from mini_agent.terminal_ui import TerminalChatUI


def test_terminal_ui_renders_session_and_assistant_without_legacy_labels() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=100)
    ui = TerminalChatUI(console)
    runtime = RuntimeContext(
        model="test-model",
        workspace=Path("/workspace"),
        platform="TestOS arm64",
        git=GitContext(is_repository=True, branch="main", change_count=1),
    )

    ui.show_welcome(session_id="a" * 32, runtime=runtime)
    ui.show_assistant("Implemented the change.")

    rendered = output.getvalue()
    assert "Mini Agent" in rendered
    assert "test-model" in rendered
    assert "/workspace" in rendered
    assert "main · dirty (1 changes)" in rendered
    assert "/resume" in rendered
    assert "Implemented the change." in rendered
    assert "you>" not in rendered
    assert "assistant>" not in rendered


def test_terminal_ui_does_not_emit_hyperlink_escape_sequences_from_assistant_text() -> None:
    output = StringIO()
    console = Console(
        file=output,
        force_terminal=True,
        color_system="truecolor",
        width=100,
    )
    ui = TerminalChatUI(console)

    ui.show_assistant("[review this link](https://untrusted.example/path)")

    rendered = output.getvalue()
    assert "\x1b]8;" not in rendered
    assert "https://untrusted.example/path" in rendered


def test_terminal_ui_reads_input_without_a_fixed_width_frame() -> None:
    output = StringIO()
    ui = TerminalChatUI(Console(file=output, force_terminal=False, width=60))

    with patch("builtins.input", return_value="inspect the repository"):
        value = ui.read_input()

    rendered = output.getvalue()
    assert value == "inspect the repository"
    assert "You\n❯ " in rendered
    assert "╭" not in rendered
    assert "╰" not in rendered


def test_terminal_ui_renders_compact_status_before_input() -> None:
    output = StringIO()
    ui = TerminalChatUI(Console(file=output, force_terminal=False, width=120))
    runtime = RuntimeContext(
        model="provider/test-model",
        workspace=Path("/workspace"),
        platform="TestOS arm64",
        git=GitContext(is_repository=True, branch="main", change_count=2),
    )
    projection = ContextProjection(
        estimate=RequestTokenEstimate(
            system_tokens=500,
            tool_tokens=200,
            message_tokens=700,
            protocol_tokens=100,
            total_tokens=1_500,
        ),
        input_limit=8_000,
    )

    ui.show_status(
        AgentStatus(
            runtime=runtime,
            context=projection,
            checkpoint_version=2,
            goal_status=GoalStatus.ACTIVE,
            has_uncommitted_events=True,
        )
    )

    rendered = output.getvalue()
    assert "provider/test-model" in rendered
    assert "git main*" in rendered
    assert "ctx ~1.5k/8.0k 19%" in rendered
    assert "checkpoint v2+" in rendered
    assert "goal active" in rendered


def test_terminal_ui_selects_session_from_readable_history() -> None:
    output = StringIO()
    ui = TerminalChatUI(Console(file=output, force_terminal=False, width=120))
    now = datetime(2026, 8, 13, 9, 30, tzinfo=UTC)
    sessions = (
        SessionSummary(
            session_id="a" * 32,
            workspace="/workspace",
            model="test-model",
            created_at=now,
            last_event_at=now,
            event_count=8,
            last_event_kind="assistant_message",
            user_message_count=3,
            last_user_message="修复 resume 的历史会话选择",
            has_work=True,
        ),
        SessionSummary(
            session_id="b" * 32,
            workspace="/workspace",
            model="test-model",
            created_at=now,
            last_event_at=now,
            event_count=4,
            last_event_kind="session_stopped",
            user_message_count=1,
            last_user_message="研究上下文压缩",
            has_work=True,
        ),
    )

    with patch("builtins.input", return_value="2"):
        selected = ui.select_session(sessions, current_session_id="b" * 32)

    rendered = output.getvalue()
    assert selected == "b" * 32
    assert "Resume a session" in rendered
    assert "修复 resume 的历史会话选择" in rendered
    assert "研究上下文压缩" in rendered
    assert "aaaaaaaa" in rendered
    assert "current" in rendered


def test_terminal_ui_session_selector_defaults_to_latest() -> None:
    output = StringIO()
    ui = TerminalChatUI(Console(file=output, force_terminal=False, width=100))
    now = datetime(2026, 8, 13, 9, 30, tzinfo=UTC)
    latest = SessionSummary(
        session_id="c" * 32,
        workspace="/workspace",
        model="test-model",
        created_at=now,
        last_event_at=now,
        event_count=1,
        last_event_kind="session_started",
        user_message_count=0,
        has_work=False,
    )

    with patch("builtins.input", return_value=""):
        selected = ui.select_session((latest,))

    assert selected == latest.session_id


def test_terminal_ui_replays_only_latest_thirty_visible_messages() -> None:
    output = StringIO()
    ui = TerminalChatUI(Console(file=output, force_terminal=False, width=120))
    conversation = tuple(
        ConversationMessage(
            role=MessageRole.USER if index % 2 == 0 else MessageRole.ASSISTANT,
            content=f"visible-message-{index:02d}",
        )
        for index in range(32)
    )
    tool_message = ConversationMessage(
        role=MessageRole.TOOL,
        content="tool-output-must-not-render",
        tool_call_id="call-1",
    )

    ui.show_recent_conversation((*conversation[:16], tool_message, *conversation[16:]))

    rendered = output.getvalue()
    assert "Recent conversation · 30 messages · 2 earlier omitted" in rendered
    assert "visible-message-00" not in rendered
    assert "visible-message-01" not in rendered
    assert "visible-message-02" in rendered
    assert "visible-message-31" in rendered
    assert "tool-output-must-not-render" not in rendered
    assert "Resume here" in rendered


def test_terminal_ui_does_not_render_empty_resume_history() -> None:
    output = StringIO()
    ui = TerminalChatUI(Console(file=output, force_terminal=False, width=100))

    ui.show_recent_conversation(())

    assert output.getvalue() == ""


@pytest.mark.parametrize("width", [20, 40, 80, 120])
def test_status_reserves_last_column_and_keeps_semantic_blocks_together(width: int) -> None:
    output = StringIO()
    ui = TerminalChatUI(Console(file=output, force_terminal=False, width=width))
    runtime = RuntimeContext(
        model="provider/中文-model-" * 6,
        workspace=Path("/workspace"),
        platform="TestOS",
        git=GitContext(is_repository=True, branch="main"),
    )

    ui.show_status(
        AgentStatus(
            runtime=runtime,
            context=None,
            checkpoint_version=None,
            goal_status=GoalStatus.ACTIVE,
            has_uncommitted_events=False,
        )
    )

    rendered = output.getvalue()
    assert "goal active" in rendered
    assert "checkpoint none" in rendered
    assert "output brief" in rendered
    assert all(Text(line).cell_len < width for line in rendered.splitlines())


def test_detailed_history_includes_arguments_for_tool_only_assistant_messages() -> None:
    output = StringIO()
    ui = TerminalChatUI(Console(file=output, width=120), output_mode=OutputMode.DETAILED)
    call = ToolCall(id="call", name="read_file", arguments_json='{"path":"notes.txt"}')
    messages = (
        ConversationMessage(role=MessageRole.USER, content="read the file"),
        ConversationMessage(role=MessageRole.ASSISTANT, tool_calls=(call,)),
        ConversationMessage(role=MessageRole.TOOL, tool_call_id="call", content="saved body"),
    )

    ui.show_recent_conversation(messages)

    assert "notes.txt" in output.getvalue()
    assert "read_file · saved result" in output.getvalue()
    assert "saved body" in output.getvalue()


@pytest.mark.parametrize("command", ["/output invalid", "/output brief extra"])
def test_invalid_output_mode_keeps_the_current_mode(command: str) -> None:
    output = StringIO()
    ui = TerminalChatUI(Console(file=output), output_mode=OutputMode.DETAILED)

    ui.handle_output_command(command)

    assert ui.output_mode is OutputMode.DETAILED
    assert "Usage: /output brief|detailed" in output.getvalue()


def test_output_mode_can_be_changed_without_restarting_the_ui() -> None:
    output = StringIO()
    ui = TerminalChatUI(Console(file=output))

    ui.handle_output_command("/output detailed")

    assert ui.output_mode is OutputMode.DETAILED
    assert "/history" in output.getvalue()


def test_approval_pause_stops_live_animation_until_input_finishes() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=True)
    ui = TerminalChatUI(console)
    status = console.status("working")
    # Rich exposes its live lifecycle state on the underlying renderable.
    live = status._live  # pyright: ignore[reportPrivateUsage]

    with patch.object(console, "status", return_value=status), ui.thinking():
        assert live.is_started
        with ui.pause_thinking():
            assert not live.is_started
        assert live.is_started
    assert not live.is_started
