"""Small Rich terminal surface for an interactive Mini Agent session."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mini_agent.agent import AgentStatus
from mini_agent.messages import ConversationMessage, MessageRole
from mini_agent.session import SessionSummary
from mini_agent.system_prompt import RuntimeContext
from mini_agent.terminal_safety import safe_terminal_text


class TerminalChatUI:
    """Render conversation boundaries without owning agent behavior."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    def show_welcome(self, *, session_id: str | None, runtime: RuntimeContext) -> None:
        details = Table.grid(padding=(0, 1))
        details.add_column(style="dim", justify="right")
        details.add_column()
        details.add_row("model", Text(safe_terminal_text(runtime.model), style="bold"))
        details.add_row("workspace", Text(safe_terminal_text(str(runtime.workspace))))
        details.add_row("git", Text(self._git_label(runtime)))
        session_label = session_id or "not saved until the first request"
        details.add_row("session", Text(safe_terminal_text(session_label), style="dim"))
        details.add_row(
            "commands",
            Text(
                "/system  /context  /checkpoint  /compact  /goal  /resume  /exit",
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

        line = Text(no_wrap=True, overflow="ellipsis")
        line.append(f" {safe_terminal_text(runtime.model)} ", style="bold cyan")
        for label in (
            self._compact_git_label(runtime),
            "ctx empty",
            "session not saved",
            "goal none",
        ):
            line.append("│", style="dim")
            line.append(f" {safe_terminal_text(label)} ", style="dim")
        self._console.print(line)

    def read_input(self) -> str:
        width = max(24, min(self._console.width, 100))
        self._console.print(Text(f"╭─ You {'─' * (width - 7)}", style="cyan"))
        self._console.print(Text("│ ❯ ", style="bold cyan"), end="")
        value = input()
        self._console.print(Text(f"╰{'─' * (width - 1)}", style="cyan"))
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
            f"checkpoint v{status.checkpoint_version}"
            f"{'+' if status.has_uncommitted_events else ''}"
            if status.checkpoint_version is not None
            else "checkpoint none"
        )
        goal_label = (
            f"goal {status.goal_status.value}"
            if status.goal_status is not None
            else "goal none"
        )
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append(f" {safe_terminal_text(status.runtime.model)} ", style="bold cyan")
        for label in (
            self._compact_git_label(status.runtime),
            context_label,
            checkpoint_label,
            goal_label,
        ):
            line.append("│", style="dim")
            line.append(f" {safe_terminal_text(label)} ", style="dim")
        self._console.print(line)

    def thinking(self) -> AbstractContextManager[object]:
        if not self._console.is_terminal:
            return nullcontext()
        return self._console.status("[cyan]Mini Agent is working…[/cyan]", spinner="dots")

    def show_assistant(self, content: str) -> None:
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
        visible = tuple(
            message
            for message in messages
            if message.role in {MessageRole.USER, MessageRole.ASSISTANT}
            and message.content is not None
        )
        if not visible:
            return
        selected = visible[-limit:]
        omitted_count = len(visible) - len(selected)
        title = f"Recent conversation · {len(selected)} messages"
        if omitted_count:
            title += f" · {omitted_count} earlier omitted"
        self._console.rule(f"[bold cyan]{title}[/bold cyan]")
        for message in selected:
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
            else:
                self.show_assistant(message.content or "")
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
