"""Drive the installed source CLI through a PTY with a deterministic provider.

The harness intentionally patches only the provider construction in the child
process. CLI input, Rich rendering, permission prompts, local tools, and durable
session storage are the production implementations.
"""

from __future__ import annotations

import os
import pty
import select
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from mini_agent.cli import TerminalApprovalPrompt
from mini_agent.events import ApprovalDecision
from mini_agent.permissions import ApprovalRequest
from mini_agent.session import AgentSession

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).resolve().parent / "cache"


class _PauseProbe:
    def __init__(self) -> None:
        self.transitions: list[str] = []

    @contextmanager
    def pause_thinking(self):
        self.transitions.append("paused")
        try:
            yield
        finally:
            self.transitions.append("resumed")


def _check_prompt_pauses_spinner() -> bool:
    probe = _PauseProbe()
    request = ApprovalRequest(
        tool_call_id="acceptance-call",
        tool_name="exec_command",
        redacted_arguments='{"argv":["pwd"],"cwd":"."}',
        scope_descriptor="exec_command:once",
        scope_fingerprint="0" * 64,
        can_allow_session=False,
    )
    with patch("builtins.input", return_value="d"):
        decision = TerminalApprovalPrompt(probe).decide(request)
    return decision is ApprovalDecision.DENY and probe.transitions == ["paused", "resumed"]


def _run(
    arguments: list[str],
    *,
    width: int,
    scripted_input: bytes,
    scenario: str = "read",
    scripted_writes: tuple[tuple[float, bytes], ...] | None = None,
    command: str = "chat",
    auto_approve: bool = False,
    timeout_seconds: float = 12,
    return_partial_on_timeout: bool = False,
    session_id: str | None = None,
) -> str:
    """Run a child Python command in a fixed-size PTY and return its transcript."""

    master_fd, slave_fd = pty.openpty()
    os.set_blocking(master_fd, False)
    process_id = os.fork()
    if process_id == 0:
        os.close(master_fd)
        os.environ["COLUMNS"] = str(width)
        os.environ["LINES"] = "32"
        os.environ["MINI_AGENT_ACCEPTANCE_SCENARIO"] = scenario
        os.environ["MINI_AGENT_ACCEPTANCE_COMMAND"] = command
        os.environ["MINI_AGENT_ACCEPTANCE_AUTO_APPROVE"] = "1" if auto_approve else "0"
        os.environ["MINI_AGENT_ACCEPTANCE_SESSION_ID"] = session_id or ""
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        os.close(slave_fd)
        os.execv(sys.executable, [sys.executable, *arguments])
    os.close(slave_fd)
    transcript = bytearray()
    writes = scripted_writes or tuple(
        (0.15 + index * 0.4, line)
        for index, line in enumerate(scripted_input.splitlines(keepends=True))
    )
    next_write_index = 0
    started_at = time.monotonic()
    try:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            while (
                next_write_index < len(writes)
                and time.monotonic() - started_at >= writes[next_write_index][0]
            ):
                os.write(master_fd, writes[next_write_index][1])
                next_write_index += 1
            readable, _, _ = select.select([master_fd], [], [], 0.05)
            if readable:
                try:
                    transcript.extend(os.read(master_fd, 65_536))
                except OSError:
                    break
            complete_id, _ = os.waitpid(process_id, os.WNOHANG)
            if complete_id == process_id:
                while True:
                    try:
                        chunk = os.read(master_fd, 65_536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    transcript.extend(chunk)
                break
        else:
            os.kill(process_id, 9)
            os.waitpid(process_id, 0)
            if not return_partial_on_timeout:
                raise RuntimeError("CLI child timed out")
    finally:
        os.close(master_fd)
    return transcript.decode("utf-8", errors="replace")


def _write_child_runner(workspace: Path) -> Path:
    """Create a source-only child runner that injects a scripted provider."""

    runner = workspace / "child_runner.py"
    runner.write_text(
        """\
import asyncio
import os
import sys
from pathlib import Path

from mini_agent import cli
from mini_agent.messages import FinishReason, ModelResponse, ToolCall


class ScriptedProvider:
    def __init__(self, _config):
        self.calls = 0

    async def complete(self, _request):
        self.calls += 1
        scenario = os.environ["MINI_AGENT_ACCEPTANCE_SCENARIO"]
        if self.calls == 1:
            if scenario in {"approval", "denied", "approval_slow"}:
                if scenario == "approval_slow":
                    await asyncio.sleep(0.5)
                return ModelResponse(
                    content=None,
                    tool_calls=(ToolCall(
                        id=\"exec-call\", name=\"exec_command\",
                        arguments_json='{"argv":["pwd"],"cwd":"."}',
                    ),),
                    finish_reason=FinishReason.TOOL_CALLS,
                )
            return ModelResponse(
                content=None,
                tool_calls=(ToolCall(
                    id=\"read-call\", name=\"read_file\",
                    arguments_json='{"path":"fixture.txt"}',
                ),),
                finish_reason=FinishReason.TOOL_CALLS,
            )
        if scenario in {"denied", "approval_slow"}:
            answer = \"denied command was not run\"
        elif scenario == "approval":
            answer = \"approved command was consumed\"
        else:
            answer = \"tool output was consumed\"
        return ModelResponse(content=answer, finish_reason=FinishReason.STOP)

    async def aclose(self):
        return None


cli.OpenAICompatibleProvider = ScriptedProvider
command = os.environ[\"MINI_AGENT_ACCEPTANCE_COMMAND\"]
workspace = sys.argv[1]
config = sys.argv[2]
os.chdir(workspace)
sys.argv = [\"mini-agent\"]
if command == \"default\":
    os.environ[\"XDG_CONFIG_HOME\"] = os.path.join(os.path.dirname(config), \"user_config\")
else:
    sys.argv.append(command)
    session_id = os.environ[\"MINI_AGENT_ACCEPTANCE_SESSION_ID\"]
    if command == \"resume\" and session_id:
        sys.argv.append(session_id)
    sys.argv.extend([\"--workspace\", workspace, \"--config\", config])
if os.environ[\"MINI_AGENT_ACCEPTANCE_AUTO_APPROVE\"] == \"1\":
    sys.argv.append(\"--auto-approve\")
cli.app()
""",
        encoding="utf-8",
    )
    return runner


def _prepare_workspace(name: str) -> tuple[Path, Path]:
    workspace = CACHE / name
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    (workspace / "fixture.txt").write_text("fixture tool output\n", encoding="utf-8")
    config = workspace / "config.toml"
    config.write_text(
        """[model]
model = \"acceptance-fake\"
base_url = \"http://127.0.0.1:9/v1\"
context_window = 100000
max_output_tokens = 1024

[runtime]
data_dir = \"data\"
""",
        encoding="utf-8",
    )
    user_config = workspace / "user_config" / "mini-agent" / "config.toml"
    user_config.parent.mkdir(parents=True)
    user_config.write_text(config.read_text(encoding="utf-8"), encoding="utf-8")
    return workspace, _write_child_runner(workspace)


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    for width in (40, 80, 120):
        workspace, runner = _prepare_workspace(f"width_{width}")
        transcript = _run(
            [str(runner), str(workspace), str(workspace / "config.toml")],
            width=width,
            scripted_input=b"inspect fixture\n/exit\n",
        )
        (CACHE / f"baseline_width_{width}.ansi").write_text(transcript, encoding="utf-8")
        required = ("tool output was consumed", "→ read_file", "✓ read_file · 1 output lines")
        missing = [item for item in required if item not in transcript]
        has_hidden_output = "fixture tool output" in transcript
        outcome = "PASS" if not missing and not has_hidden_output else "FAILED"
        details = missing + (["brief output leaked"] if has_hidden_output else [])
        print(f"brief width={width}: {outcome if not details else ', '.join(details)}")

    workspace, runner = _prepare_workspace("approval")
    transcript = _run(
        [str(runner), str(workspace), str(workspace / "config.toml")],
        width=80,
        scripted_input=b"run approved command\no\n/exit\n",
        scenario="approval",
    )
    (CACHE / "baseline_approval.ansi").write_text(transcript, encoding="utf-8")
    required = (
        "Approval required: exec_command",
        "Command: pwd",
        "o = allow once / d = deny (default d):",
    )
    missing = [item for item in required if item not in transcript]
    print(f"approval: {'PASS' if not missing else 'MISSING ' + ', '.join(missing)}")

    workspace, runner = _prepare_workspace("denied_approval")
    transcript = _run(
        [str(runner), str(workspace), str(workspace / "config.toml")],
        width=80,
        scripted_input=b"run denied command\nd\n/exit\n",
        scenario="denied",
    )
    (CACHE / "denied_approval.ansi").write_text(transcript, encoding="utf-8")
    required = ("Approval required: exec_command", "✗ exec_command · permission_denied")
    forbidden = "✓ exec_command"
    missing = [item for item in required if item not in transcript]
    outcome = "PASS" if not missing and forbidden not in transcript else "FAILED"
    details = missing + (["denied command executed"] if forbidden in transcript else [])
    print(f"denied approval: {outcome if not details else ', '.join(details)}")

    for label, response in (("empty approval", b"\n"), ("invalid approval", b"x\n")):
        workspace, runner = _prepare_workspace(label.replace(" ", "_"))
        transcript = _run(
            [str(runner), str(workspace), str(workspace / "config.toml")],
            width=80,
            scripted_input=b"request command\n" + response + b"/exit\n",
            scenario="denied",
        )
        (CACHE / f"{label.replace(' ', '_')}.ansi").write_text(transcript, encoding="utf-8")
        required = ("✗ exec_command · permission_denied",)
        if label == "invalid approval":
            required += ("That choice is not allowed for this operation; denied.",)
        missing = [item for item in required if item not in transcript]
        print(f"{label}: {'PASS' if not missing else 'MISSING ' + ', '.join(missing)}")

    workspace, runner = _prepare_workspace("spinner_pause")
    transcript = _run(
        [str(runner), str(workspace), str(workspace / "config.toml")],
        width=80,
        scripted_input=b"",
        scenario="approval_slow",
        scripted_writes=(
            (0.15, b"request delayed command\n"),
            (1.0, b"d\n"),
            (1.2, b"/exit\n"),
        ),
    )
    (CACHE / "spinner_pause.ansi").write_text(transcript, encoding="utf-8")
    approval_offset = transcript.find("Approval required: exec_command")
    denied_offset = transcript.find("✗ exec_command · permission_denied")
    prompt_segment = (
        transcript[approval_offset:denied_offset] if denied_offset > approval_offset else ""
    )
    has_working_during_prompt = "Mini Agent is working" in prompt_segment
    prompt_visible = approval_offset >= 0 and denied_offset > approval_offset
    print(
        "spinner pause PTY: "
        f"{'PASS' if prompt_visible and not has_working_during_prompt else 'FAILED'}"
    )
    print(f"spinner pause contract: {'PASS' if _check_prompt_pauses_spinner() else 'FAILED'}")

    workspace, runner = _prepare_workspace("detailed")
    transcript = _run(
        [str(runner), str(workspace), str(workspace / "config.toml")],
        width=80,
        scripted_input=b"/output detailed\ninspect fixture\n/history\n/exit\n",
    )
    (CACHE / "detailed_history.ansi").write_text(transcript, encoding="utf-8")
    required = ("Output: detailed.", "fixture tool output", "Recent conversation")
    missing = [item for item in required if item not in transcript]
    print(f"detailed history: {'PASS' if not missing else 'MISSING ' + ', '.join(missing)}")

    workspace, runner = _prepare_workspace("permissions")
    transcript = _run(
        [str(runner), str(workspace), str(workspace / "config.toml")],
        width=80,
        scripted_input=b"/permissions\n/exit\n",
    )
    (CACHE / "permissions.ansi").write_text(transcript, encoding="utf-8")
    required = (
        "Permissions",
        "Workspace reads and search: automatic after path and content checks.",
        "New files and external commands: approve each invocation.",
    )
    missing = [item for item in required if item not in transcript]
    print(f"permission policy: {'PASS' if not missing else 'MISSING ' + ', '.join(missing)}")

    workspace, runner = _prepare_workspace("auto_approve")
    transcript = _run(
        [str(runner), str(workspace), str(workspace / "config.toml")],
        width=80,
        scripted_input=b"run command without a prompt\n/exit\n",
        scenario="approval",
        auto_approve=True,
    )
    (CACHE / "auto_approve.ansi").write_text(transcript, encoding="utf-8")
    required = ("✓ exec_command · exit 0",)
    missing = [item for item in required if item not in transcript]
    prompt_shown = "Approval required:" in transcript
    print("auto approve: PASS" if not missing and not prompt_shown else "auto approve: FAILED")

    workspace, runner = _prepare_workspace("default_auto_approve")
    transcript = _run(
        [str(runner), str(workspace), str(workspace / "config.toml")],
        width=80,
        scripted_input=b"run command through default entry\n/exit\n",
        scenario="approval",
        command="default",
        auto_approve=True,
    )
    (CACHE / "default_auto_approve.ansi").write_text(transcript, encoding="utf-8")
    missing = [item for item in required if item not in transcript]
    prompt_shown = "Approval required:" in transcript
    print(
        "default auto approve: PASS"
        if not missing and not prompt_shown
        else "default auto approve: FAILED"
    )

    workspace, runner = _prepare_workspace("resume_auto_approve")
    session = AgentSession.create(
        data_dir=workspace / "data", workspace=workspace, model="acceptance-fake"
    )
    session.append_user_message("synthetic resume fixture", turn_id="acceptance-turn")
    session.stop()
    session.close()
    resumed = _run(
        [str(runner), str(workspace), str(workspace / "config.toml")],
        width=80,
        scripted_input=b"run resumed command\n/exit\n",
        scenario="approval",
        command="resume",
        auto_approve=True,
        session_id=session.metadata.session_id,
    )
    (CACHE / "resume_auto_approve.ansi").write_text(resumed, encoding="utf-8")
    required = ("✓ exec_command · exit 0",)
    missing = [item for item in required if item not in resumed]
    resume_failed = "Approval required:" in resumed or "not found" in resumed
    passed = not missing and not resume_failed
    print("resume auto approve: PASS" if passed else "resume auto approve: FAILED")


if __name__ == "__main__":
    main()
