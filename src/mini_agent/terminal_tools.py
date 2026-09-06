"""Bounded terminal views of durable tool activity."""

from enum import StrEnum

from rich.console import Console
from rich.text import Text

from mini_agent.events import (
    ApprovalResolvedData,
    StoredEvent,
    ToolCompletedData,
    ToolFailedData,
    ToolInterruptedData,
    ToolRequestedData,
    ToolStartedData,
)
from mini_agent.messages import ToolCall
from mini_agent.terminal_safety import safe_terminal_text

MAX_DETAIL_CHARS = 6_000
MAX_DETAIL_LINES = 80


class OutputMode(StrEnum):
    BRIEF = "brief"
    DETAILED = "detailed"


class TerminalToolOutput:
    """Render only committed, redacted facts; never decide tool permissions."""

    def __init__(self, console: Console, mode: OutputMode = OutputMode.BRIEF) -> None:
        self._console = console
        self.mode = mode
        self._calls: dict[str, ToolCall] = {}

    def show_event(self, event: StoredEvent) -> None:
        data = event.data
        if isinstance(data, ToolRequestedData):
            self._calls[data.tool_call.id] = data.tool_call
        elif isinstance(data, ToolStartedData):
            call = self._calls.get(data.tool_call_id)
            self._line(f"→ {data.tool_name}", style="cyan")
            if call is not None:
                self.show_arguments(call)
        elif isinstance(data, ToolCompletedData):
            name = self._name(data.tool_call_id)
            result = f"{len(data.output.splitlines())} output lines"
            if data.facts.command_exit_code is not None:
                result = f"exit {data.facts.command_exit_code}"
            if data.facts.modified_paths:
                result = "modified " + ", ".join(data.facts.modified_paths)
            has_failed_command = data.facts.command_exit_code not in {None, 0}
            symbol = "✗" if has_failed_command else "✓"
            style = "yellow" if has_failed_command else "green"
            self._line(f"{symbol} {name} · {result}", style=style)
            if data.artifact_id is not None:
                self._line(f"  full output: artifact {data.artifact_id}", style="dim")
            self.show_output(data.output)
        elif isinstance(data, ToolFailedData):
            self._line(
                f"✗ {self._name(data.tool_call_id)} · {data.error_code}: {data.message}",
                style="yellow",
            )
        elif isinstance(data, ToolInterruptedData):
            self._line(
                f"! {self._name(data.tool_call_id)} · {data.recovery_status}: {data.reason}",
                style="yellow",
            )
        elif isinstance(data, ApprovalResolvedData) and self.mode is OutputMode.DETAILED:
            self._line(f"  approval: {data.decision.value}", style="dim")

    def show_arguments(self, call: ToolCall) -> None:
        if self.mode is OutputMode.DETAILED:
            self._detail(call.arguments_json, label="arguments")
        else:
            compact = " ".join(safe_terminal_text(call.arguments_json).split())
            self._line("  " + compact[:160] + ("…" if len(compact) > 160 else ""), style="dim")

    def show_history_result(self, tool_call_id: str, output: str) -> None:
        self._line(f"↳ {self._name(tool_call_id)} · saved result", style="cyan")
        self.show_output(output)

    def remember_calls(self, calls: tuple[ToolCall, ...]) -> None:
        self._calls.update((call.id, call) for call in calls)

    def show_output(self, output: str) -> None:
        if self.mode is OutputMode.DETAILED:
            self._detail(output, label="output")

    def _name(self, tool_call_id: str) -> str:
        call = self._calls.get(tool_call_id)
        return call.name if call is not None else tool_call_id

    def _line(self, text: str, *, style: str) -> None:
        self._console.print(Text(safe_terminal_text(text), style=style))

    def _detail(self, content: str, *, label: str) -> None:
        safe = safe_terminal_text(content)
        lines = safe.splitlines()
        bounded = "\n".join(lines[:MAX_DETAIL_LINES])[:MAX_DETAIL_CHARS]
        self._line(f"  {label}:", style="dim")
        self._console.print(Text(bounded))
        if len(lines) > MAX_DETAIL_LINES or len(safe) > MAX_DETAIL_CHARS:
            self._line(
                "  … display limited; the durable result remains in session history.",
                style="dim",
            )
