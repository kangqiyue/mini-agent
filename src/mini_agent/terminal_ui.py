"""Small Rich terminal surface for an interactive Mini Agent session."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text

from mini_agent.agent import AgentStatus
from mini_agent.events import AssistantMessageData, StoredEvent
from mini_agent.messages import ConversationMessage, MessageRole
from mini_agent.session import SessionSummary
from mini_agent.system_prompt import RuntimeContext
from mini_agent.terminal_safety import safe_terminal_text
from mini_agent.terminal_tools import OutputMode, TerminalToolOutput


class TerminalChatUI:
    """Render conversation boundaries without owning agent behavior."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        output_mode: OutputMode = OutputMode.BRIEF,
        auto_approve: bool = False,
    ) -> None:
        self._console = console or Console()
        self._tools = TerminalToolOutput(self._console, output_mode)
        self._active_status: Status | None = None
        self._can_auto_approve = auto_approve

    @property
    def output_mode(self) -> OutputMode:
        return self._tools.mode

    def handle_output_command(self, command: str) -> None:
        parts = command.split()
        if len(parts) > 2:
            self._console.print("Usage: /output brief|detailed")
            return
        if len(parts) == 2:
            try:
                self._tools.mode = OutputMode(parts[1])
            except ValueError:
                self._console.print("Usage: /output brief|detailed")
                return
        self._console.print(
            f"Output: {self.output_mode.value}. Use /history to review saved results."
        )

    def show_permissions(self, *, active_grant_count: int) -> None:
        if self._can_auto_approve:
            self._console.print(
                Text(
                    "Permissions · AUTO\n"
                    "All tool actions are approved once without prompting.\n"
                    "Commands and file writes execute directly with your user permissions.\n"
                    "Workspace checks, redaction, and recovery rules still apply.\n"
                    "This startup choice is not retained when the process exits.",
                    style="yellow",
                )
            )
            return
        self._console.print(
            Text(
                "Permissions\n"
                "• Workspace reads and search: automatic after path and content checks.\n"
                "• File replacement: approve once, or approve this exact file for the session.\n"
                "• New files and external commands: approve each invocation.\n"
                "• Unknown side effects revoke the affected file grant and require review.\n"
                f"Active file grants: {active_grant_count}\n"
                "Empty or invalid approval input denies the operation.",
            )
        )

    def show_welcome(self, *, session_id: str | None, runtime: RuntimeContext) -> None:
        details = Table.grid(padding=(0, 1))
        details.add_column(style="dim", justify="right")
        details.add_column()
        details.add_row("model", Text(safe_terminal_text(runtime.model), style="bold"))
        details.add_row("workspace", Text(safe_terminal_text(str(runtime.workspace))))
        details.add_row("git", Text(self._git_label(runtime)))
        session_label = session_id or "not saved until the first request"
        details.add_row("session", Text(safe_terminal_text(session_label), style="dim"))
        if self._can_auto_approve:
            details.add_row(
                "approvals", Text("AUTO · commands and writes run without prompts", style="yellow")
            )
        details.add_row(
            "commands",
            Text(
                "/output  /history  /permissions  /system  /context  "
                "/checkpoint  /compact  /goal  /resume  /exit",
                style="cyan",
            ),
        )
        self._console.print(
            Panel(
                details,
                title="[bold cyan]Mini Agent[/bold cyan]",
                title_align="left",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    def show_pending_status(self, runtime: RuntimeContext) -> None:
        """Render the pre-session state without implying durable work exists."""

        self._show_status_blocks(
            runtime.model,
            (
                self._compact_git_label(runtime),
                "ctx empty",
                "session not saved",
                "goal none",
            ),
        )

    def read_input(self) -> str:
        self._console.print(Text("You", style="cyan"))
        self._console.print(Text("❯ ", style="bold cyan"), end="")
        value = input()
        self._console.print()
        return value

    def show_status(self, status: AgentStatus) -> None:
        """Render a compact snapshot without taking ownership of terminal input."""

        context = status.context
        if context is None:
            context_label = "ctx rebuild needed"
        else:
            context_label = (
                f"ctx ~{_format_tokens(context.estimate.total_tokens)}"
                f"/{_format_tokens(context.input_limit)} "
                f"{context.utilization_ratio:.0%}"
            )
        checkpoint_label = (
            f"checkpoint v{status.checkpoint_version}{'+' if status.has_uncommitted_events else ''}"
            if status.checkpoint_version is not None
            else "checkpoint none"
        )
        goal_label = (
            f"goal {status.goal_status.value}" if status.goal_status is not None else "goal none"
        )
        self._show_status_blocks(
            status.runtime.model,
            (
                self._compact_git_label(status.runtime),
                context_label,
                checkpoint_label,
                goal_label,
            ),
        )

    def _show_status_blocks(self, model: str, labels: tuple[str, ...]) -> None:
        # Reserve the terminal's final column to avoid automatic-wrap artifacts.
        width = max(1, self._console.width - 1)
        model_line = Text(safe_terminal_text(model), style="bold cyan")
        model_line.truncate(width, overflow="ellipsis")
        self._console.print(model_line, no_wrap=True)
        line = Text(style="dim")
        approval_label = "approvals AUTO" if self._can_auto_approve else "approvals ask"
        for label in (*labels, f"output {self.output_mode.value}", approval_label):
            block = Text(safe_terminal_text(label))
            if label == "approvals AUTO":
                block.stylize("bold yellow")
            block.truncate(width, overflow="ellipsis")
            if line and line.cell_len + 3 + block.cell_len > width:
                self._console.print(line, no_wrap=True)
                line = Text(style="dim")
            if line:
                line.append(" │ ")
            line.append_text(block)
        if line:
            self._console.print(line, no_wrap=True)

    @contextmanager
    def thinking(self) -> Generator[None]:
        if not self._console.is_terminal:
            yield
            return
        status = self._console.status("[cyan]Mini Agent is working…[/cyan]", spinner="dots")
        self._active_status = status
        try:
            with status:
                yield
        finally:
            self._active_status = None

    @contextmanager
    def pause_thinking(self) -> Generator[None]:
        status = self._active_status
        if status is not None:
            status.stop()
        try:
            yield
        finally:
            if status is not None and self._active_status is status:
                status.start()

    def show_event(self, event: StoredEvent) -> None:
        data = event.data
        if isinstance(data, AssistantMessageData) and data.tool_calls and data.content:
            self.show_assistant(data.content, is_intermediate=True)
        self._tools.show_event(event)

    def show_assistant(self, content: str, *, is_intermediate: bool = False) -> None:
        if is_intermediate and self.output_mode is OutputMode.BRIEF:
            compact = " ".join(safe_terminal_text(content).split())
            self._console.print(
                Text("↳ " + compact[:240] + ("…" if len(compact) > 240 else ""), style="dim")
            )
            return
        rendered = Text(safe_terminal_text(content))
        self._console.print(
            Panel(
                rendered,
                title="[bold green]Mini Agent[/bold green]",
                title_align="left",
                border_style="green",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )
        self._console.print()

    def show_recent_conversation(
        self, messages: tuple[ConversationMessage, ...], *, limit: int = 30
    ) -> None:
        """Replay the latest visible user and assistant messages after resume."""

        if limit < 1:
            raise ValueError("Conversation history limit must be positive")
        visible_indices = tuple(
            index
            for index, message in enumerate(messages)
            if message.role in {MessageRole.USER, MessageRole.ASSISTANT}
            and message.content is not None
        )
        if not visible_indices:
            return
        selected = visible_indices[-limit:]
        omitted_count = len(visible_indices) - len(selected)
        title = f"Recent conversation · {len(selected)} messages"
        if omitted_count:
            title += f" · {omitted_count} earlier omitted"
        self._console.rule(f"[bold cyan]{title}[/bold cyan]")
        for message in messages:
            self._tools.remember_calls(message.tool_calls)
        for message in messages[selected[0] :]:
            if message.role is MessageRole.USER:
                self._console.print(
                    Panel(
                        Text(safe_terminal_text(message.content or "")),
                        title="[bold cyan]You[/bold cyan]",
                        title_align="left",
                        border_style="cyan",
                        box=box.ROUNDED,
                        padding=(0, 1),
                    )
                )
            elif message.role is MessageRole.ASSISTANT:
                if message.content is not None:
                    self.show_assistant(message.content, is_intermediate=bool(message.tool_calls))
                if self.output_mode is OutputMode.DETAILED:
                    for call in message.tool_calls:
                        self._tools.show_arguments(call)
            elif message.role is MessageRole.TOOL:
                self._tools.show_history_result(
                    message.tool_call_id or "tool", message.content or ""
                )
        self._console.rule("[dim]Resume here[/dim]")

    def show_system(self, content: str) -> None:
        self._console.print(
            Panel(
                Text(safe_terminal_text(content)),
                title="[bold yellow]Assembled system prompt[/bold yellow]",
                title_align="left",
                border_style="yellow",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    def select_session(
        self,
        sessions: tuple[SessionSummary, ...],
        *,
        current_session_id: str | None = None,
    ) -> str | None:
        """Show resumable history and return a selected session id."""

        table = Table(
            box=box.SIMPLE_HEAD,
            title="Resume a session",
            title_style="bold cyan",
            show_edge=False,
        )
        table.add_column("#", style="bold cyan", justify="right")
        table.add_column("Last active", style="dim", no_wrap=True)
        table.add_column("Turns", justify="right")
        table.add_column("Conversation")
        for index, session in enumerate(sessions, start=1):
            marker = "  current" if session.session_id == current_session_id else ""
            preview = _conversation_preview(session.last_user_message)
            conversation = f"{preview}  {session.session_id[:8]}{marker}"
            conversation_text = (
                Text(conversation, style="bold") if index == 1 else Text(conversation)
            )
            table.add_row(
                str(index),
                session.last_event_at.astimezone().strftime("%Y-%m-%d %H:%M"),
                str(session.user_message_count),
                conversation_text,
            )
        self._console.print(table)

        while True:
            try:
                answer = input("Resume [1] (number, session id, q to cancel): ").strip()
            except (EOFError, KeyboardInterrupt):
                self._console.print()
                return None
            if not answer:
                return sessions[0].session_id
            if answer.lower() in {"q", "quit", "cancel"}:
                return None
            if answer.isdecimal():
                index = int(answer)
                if 1 <= index <= len(sessions):
                    return sessions[index - 1].session_id
            matches = tuple(
                session for session in sessions if session.session_id.startswith(answer)
            )
            if len(matches) == 1:
                return matches[0].session_id
            self._console.print(
                "Choose a listed number or an unambiguous session id.",
                style="yellow",
            )

    @staticmethod
    def _git_label(runtime: RuntimeContext) -> str:
        if not runtime.git.is_available:
            return "unavailable"
        if not runtime.git.is_repository:
            return "not a repository"
        branch = runtime.git.branch or "detached"
        return safe_terminal_text(f"{branch} · {runtime.git.state}")

    @staticmethod
    def _compact_git_label(runtime: RuntimeContext) -> str:
        if not runtime.git.is_available:
            return "git unavailable"
        if not runtime.git.is_repository:
            return "git none"
        branch = runtime.git.branch or "detached"
        suffix = "*" if runtime.git.change_count else ""
        return f"git {branch}{suffix}"


def _format_tokens(token_count: int) -> str:
    if token_count < 1_000:
        return str(token_count)
    if token_count < 1_000_000:
        return f"{token_count / 1_000:.1f}k"
    return f"{token_count / 1_000_000:.1f}m"


def _conversation_preview(content: str | None, *, maximum_chars: int = 56) -> str:
    if content is None:
        return "(no user message yet)"
    compact = " ".join(safe_terminal_text(content).split())
    if len(compact) <= maximum_chars:
        return compact
    return compact[: maximum_chars - 1].rstrip() + "…"
