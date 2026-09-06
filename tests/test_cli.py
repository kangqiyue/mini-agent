import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import HttpUrl
from typer.testing import CliRunner

from mini_agent import cli as cli_module
from mini_agent.agent import AgentStatus, MiniAgent
from mini_agent.artifacts import ArtifactStoreCorruptionError
from mini_agent.checkpoint import CheckpointStoreCorruptionError
from mini_agent.checkpoint_writer import CheckpointRecoveryRequiredError
from mini_agent.cli import (
    InteractiveResult,
    TerminalApprovalPrompt,
    app,
    build_workspace_for_session,
    handle_goal_command,
    show_context,
)
from mini_agent.config import ContextConfig, MiniAgentConfig, ModelConfig, RuntimeConfig
from mini_agent.context import (
    ContextConfigurationError,
    Utf8TokenEstimator,
    initial_goal_objective,
    resolve_request_budget,
)
from mini_agent.event_store import EventStore, EventStoreWriterBusyError
from mini_agent.events import ApprovalDecision
from mini_agent.goal import GoalState, GoalStatus
from mini_agent.host_path_redaction import (
    normalize_conversation_message_host_paths,
    normalize_host_path_text,
    normalize_model_request_host_paths,
)
from mini_agent.messages import (
    ConversationMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
)
from mini_agent.permissions import ApprovalRequest
from mini_agent.provider import ProviderCapabilities
from mini_agent.session import AgentSession, list_sessions
from mini_agent.system_prompt import GitContext, RuntimeContext
from mini_agent.terminal_tools import OutputMode
from mini_agent.workspace import WorkspacePathError
from tests.support.synthetic_secrets import synthetic_stripe_access_token

runner = CliRunner()
_SYNTHETIC_CLI_SECRET = synthetic_stripe_access_token("CLICONFIG")


class _BenchmarkReportStub:
    """Minimal report surface for CLI boundary tests."""

    def model_dump(self, *, mode: str) -> dict[str, str]:
        assert mode == "json"
        return {"result": "ok"}

    def model_dump_json(self, *, indent: int) -> str:
        assert indent == 2
        return '{\n  "result": "ok"\n}'


def test_no_subcommand_starts_chat_with_current_workspace_defaults() -> None:
    with patch("mini_agent.cli.chat") as chat_mock:
        result = runner.invoke(app, [])

    assert result.exit_code == 0, result.output
    chat_mock.assert_called_once_with(auto_approve=False)


def test_explicit_subcommand_does_not_start_default_chat() -> None:
    with patch("mini_agent.cli.chat") as chat_mock:
        result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    chat_mock.assert_not_called()


def test_init_config_creates_a_complete_config_without_printing_workspace(
    tmp_path: Path,
) -> None:
    result = runner.invoke(app, ["init-config", "--workspace", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert ".mini-agent/config.toml" in result.output
    assert str(tmp_path) not in result.output
    assert (tmp_path / ".mini-agent" / "config.toml").is_file()


def test_init_config_refuses_to_overwrite_an_existing_config(tmp_path: Path) -> None:
    first_result = runner.invoke(app, ["init-config", "--workspace", str(tmp_path)])
    config_path = tmp_path / ".mini-agent" / "config.toml"
    original_contents = config_path.read_text(encoding="utf-8")

    second_result = runner.invoke(app, ["init-config", "--workspace", str(tmp_path)])

    assert first_result.exit_code == 0, first_result.output
    assert second_result.exit_code != 0
    assert "already exists" in second_result.output
    assert config_path.read_text(encoding="utf-8") == original_contents


def test_chat_exit_before_first_request_does_not_create_a_session(tmp_path: Path) -> None:
    data_dir = tmp_path / "agent-data"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, data_dir=data_dir, model="test-model")

    with patch("builtins.input", return_value="/exit"):
        result = runner.invoke(
            app,
            ["chat", "--workspace", str(tmp_path), "--config", str(config_path)],
        )

    assert result.exit_code == 0, result.output
    assert not (data_dir / "sessions").exists()
    assert "not saved until the first request" in result.output


def test_chat_rejects_unsendable_first_request_before_creating_session(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "agent-data"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            (
                "[model]",
                'model = "test-model"',
                'base_url = "https://example.test/v1"',
                "context_window = 4096",
                "max_output_tokens = 1024",
                "",
                "[context]",
                "reserve_tokens = 1024",
                "minimum_input_tokens = 1024",
                "",
                "[runtime]",
                f'data_dir = "{data_dir}"',
                "",
            )
        ),
        encoding="utf-8",
    )

    with patch("builtins.input", return_value="start"):
        result = runner.invoke(
            app,
            ["chat", "--workspace", str(tmp_path), "--config", str(config_path)],
        )

    assert result.exit_code == 1, result.output
    assert "Configuration cannot fit the required model request." in result.output
    assert str(tmp_path) not in result.output
    assert "chat/completions" not in result.output
    assert not data_dir.exists()


def test_chat_first_input_floor_includes_user_and_rebuild_before_session_creation(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "agent-data"
    initial_input = "x" * 128
    provisional = MiniAgentConfig(
        model=ModelConfig(
            model="test-model",
            base_url=HttpUrl("https://example.test/v1"),
            context_window=12_000,
            max_output_tokens=1_024,
        ),
        context=ContextConfig(reserve_tokens=1_024, minimum_input_tokens=1_024),
        runtime=RuntimeConfig(data_dir=data_dir),
    )
    setup = cli_module._prepare_pending_chat(  # pyright: ignore[reportPrivateUsage]
        provisional,
        tmp_path,
    )
    budget = resolve_request_budget(
        model=provisional.model,
        context=provisional.context,
        capabilities=ProviderCapabilities(
            context_window=12_000,
            max_output_tokens=1_024,
        ),
    )
    goal = GoalState(
        goal_id="0" * 32,
        objective=initial_goal_objective(
            normalize_host_path_text(initial_input, workspace_root=tmp_path)
        ),
        acceptance_criteria=(),
    )
    # This one-token boundary needs the same runtime snapshot on both sides.
    # Live Git probes can change size when a subprocess probe times out.
    runtime = RuntimeContext(
        model="test-model", workspace=tmp_path, platform="TestOS",
        git=GitContext(is_repository=False),
    )
    system = normalize_conversation_message_host_paths(
        setup.prompt_assembler.assemble(goal, runtime=runtime),
        workspace_root=tmp_path,
    )
    full_request = normalize_model_request_host_paths(
        ModelRequest(
            model=provisional.model.model,
            messages=(
                system,
                ConversationMessage(role=MessageRole.USER, content=initial_input),
            ),
            tools=setup.tools,
            max_output_tokens=budget.desired_output_tokens,
        ),
        workspace_root=tmp_path,
    )
    input_limit = Utf8TokenEstimator().estimate_request(full_request).total_tokens - 1
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            (
                "[model]",
                'model = "test-model"',
                'base_url = "https://example.test/v1"',
                "context_window = 12000",
                f"max_input_tokens = {input_limit}",
                "max_output_tokens = 1024",
                "",
                "[context]",
                "reserve_tokens = 1024",
                "minimum_input_tokens = 1024",
                "",
                "[runtime]",
                f'data_dir = "{data_dir}"',
                "",
            )
        ),
        encoding="utf-8",
    )

    with (
        patch("builtins.input", return_value=initial_input),
        patch("mini_agent.cli.SystemPromptAssembler.runtime_context", return_value=runtime),
        patch("mini_agent.cli.MiniAgent.run_turn", new_callable=AsyncMock) as turn,
    ):
        result = runner.invoke(
            app,
            ["chat", "--workspace", str(tmp_path), "--config", str(config_path)],
        )

    assert result.exit_code == 1, result.output
    assert "Configuration cannot fit the required model request." in result.output
    assert not data_dir.exists()
    turn.assert_not_awaited()


def test_goal_mutation_preflight_preserves_existing_goal_after_rejection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    handle_goal_command(session, "/goal set keep this")
    events_before = session.events

    def reject(_goal: object) -> None:
        raise ContextConfigurationError("Base model request exceeds the active input budget")

    handle_goal_command(
        session,
        "/goal set oversized",
        validate_goal=reject,
    )

    assert session.events == events_before
    assert session.goal is not None
    assert session.goal.objective == "keep this"
    assert "Goal update cannot fit the required model request." in capsys.readouterr().out

    criteria_events_before = session.events
    handle_goal_command(
        session,
        "/goal criteria "
        '[{"criterion_id":"fit","description":"oversized","evidence_kind":"command_succeeded"}]',
        validate_goal=reject,
    )

    assert session.events == criteria_events_before
    assert session.goal is not None
    assert session.goal.acceptance_criteria == ()

    handle_goal_command(session, "/goal set still usable", validate_goal=lambda _goal: None)

    assert session.goal is not None
    assert session.goal.objective == "still usable"
    session.close()


def test_implicit_resume_candidates_exclude_terminal_goals(tmp_path: Path) -> None:
    data_dir = tmp_path / "agent-data"
    active = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="test-model")
    terminal = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="test-model")
    active.create_goal(objective="continue this work")
    terminal.create_goal(objective="finished work")
    terminal.cancel_goal()
    active.close()
    terminal.close()

    candidates = cli_module._matching_sessions(  # pyright: ignore[reportPrivateUsage]
        data_dir=data_dir,
        workspace=tmp_path.resolve(),
        model="test-model",
    )

    assert [summary.session_id for summary in candidates] == [active.metadata.session_id]


def test_chat_local_system_command_does_not_create_a_session(tmp_path: Path) -> None:
    data_dir = tmp_path / "agent-data"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, data_dir=data_dir, model="test-model")

    with patch("builtins.input", side_effect=("/system", "/exit")):
        result = runner.invoke(
            app,
            ["chat", "--workspace", str(tmp_path), "--config", str(config_path)],
        )

    assert result.exit_code == 0, result.output
    assert "Assembled system prompt" in result.output
    assert not (data_dir / "sessions").exists()


@pytest.mark.parametrize("inputs", (("/system",), ("first request",)))
def test_chat_missing_system_instructions_has_safe_pending_diagnostic(
    tmp_path: Path,
    inputs: tuple[str, ...],
) -> None:
    data_dir = tmp_path / "agent-data"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, data_dir=data_dir, model="test-model")
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + "[system_prompt]\n"
        + 'instructions_file = "missing-project-instructions.md"\n',
        encoding="utf-8",
    )

    with patch("builtins.input", side_effect=inputs):
        result = runner.invoke(
            app,
            ["chat", "--workspace", str(tmp_path), "--config", str(config_path)],
        )

    assert result.exit_code == 1
    assert "Configured system prompt could not be assembled safely." in _plain_cli_output(
        result.output
    )
    assert "Traceback" not in result.output
    assert str(tmp_path) not in result.output
    assert "missing-project-instructions.md" not in result.output
    assert not (data_dir / "sessions").exists()


def test_benchmark_existing_directory_output_has_safe_diagnostic(tmp_path: Path) -> None:
    output = tmp_path / "existing-directory"
    output.mkdir()

    with patch(
        "mini_agent.cli.run_reproducible_benchmark",
        new=AsyncMock(return_value=_BenchmarkReportStub()),
    ):
        result = runner.invoke(app, ["benchmark", "--output", str(output)])

    assert result.exit_code == 1
    assert "Benchmark report could not be written safely." in _plain_cli_output(result.output)
    assert "Traceback" not in result.output
    assert str(tmp_path) not in result.output
    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_benchmark_execution_os_error_has_safe_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "benchmark-execution-private-detail"
    monkeypatch.setattr(
        "mini_agent.cli.run_reproducible_benchmark",
        AsyncMock(side_effect=OSError(secret)),
    )

    result = runner.invoke(app, ["benchmark"])

    assert result.exit_code == 1
    assert "Benchmark could not be completed safely." in _plain_cli_output(result.output)
    assert "Traceback" not in result.output
    assert secret not in result.output


def test_benchmark_report_write_os_error_has_safe_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "benchmark-report-private-detail"
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "mini_agent.cli.run_reproducible_benchmark",
        AsyncMock(return_value=_BenchmarkReportStub()),
    )

    def fail_report_write(*_args: object, **_kwargs: object) -> None:
        raise OSError(secret)

    monkeypatch.setattr("mini_agent.cli.write_benchmark_report", fail_report_write)

    result = runner.invoke(app, ["benchmark", "--output", str(output)])

    assert result.exit_code == 1
    assert "Benchmark report could not be written safely." in _plain_cli_output(result.output)
    assert "Traceback" not in result.output
    assert secret not in result.output
    assert str(tmp_path) not in result.output
    assert not output.exists()


def test_benchmark_prints_report_after_successful_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "mini_agent.cli.run_reproducible_benchmark",
        AsyncMock(return_value=_BenchmarkReportStub()),
    )

    result = runner.invoke(app, ["benchmark"])

    assert result.exit_code == 0, result.output
    assert '"result": "ok"' in result.output


def test_chat_creates_session_for_first_request_and_processes_that_request(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "agent-data"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, data_dir=data_dir, model="test-model")
    interactive = AsyncMock(return_value=InteractiveResult(stop_reason="user_exit"))

    with (
        patch("builtins.input", side_effect=["/output detailed", "inspect this project"]),
        patch("mini_agent.cli._run_interactive", interactive),
    ):
        result = runner.invoke(
            app,
            ["chat", "--workspace", str(tmp_path), "--config", str(config_path), "--auto-approve"],
        )

    assert result.exit_code == 0, result.output
    assert len(list_sessions(data_dir)) == 1
    await_args = interactive.await_args
    assert await_args is not None
    assert await_args.kwargs["initial_input"] == "inspect this project"
    assert await_args.kwargs["show_welcome"] is False
    assert await_args.kwargs["output_mode"] is OutputMode.DETAILED
    assert await_args.kwargs["auto_approve"] is True


def test_approval_prompt_denies_on_end_of_input() -> None:
    request = ApprovalRequest(
        tool_call_id="call-1",
        tool_name="apply_patch",
        redacted_arguments='{"path":"notes.txt"}',
        scope_descriptor="apply_patch:arguments:v1:" + "a" * 64,
        scope_fingerprint="a" * 64,
        can_allow_session=False,
    )

    with patch("builtins.input", side_effect=EOFError):
        decision = TerminalApprovalPrompt().decide(request)

    assert decision is ApprovalDecision.DENY


@pytest.mark.parametrize("answer", ["", " ", "yes", "s", "invalid"])
def test_approval_prompt_denies_missing_or_unsupported_command_approval(answer: str) -> None:
    request = ApprovalRequest(
        tool_call_id="call",
        tool_name="exec_command",
        redacted_arguments='{"argv":["git","status"],"cwd":"."}',
        scope_descriptor="exec_command:arguments:v1:" + "a" * 64,
        scope_fingerprint="a" * 64,
        can_allow_session=False,
    )

    with patch("builtins.input", return_value=answer) as prompt:
        assert TerminalApprovalPrompt().decide(request) is ApprovalDecision.DENY

    assert "o = allow once / d = deny (default d)" in prompt.call_args.args[0]
    assert "[o]" not in prompt.call_args.args[0]


def test_command_approval_explains_executable_directory_and_scope(
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = ApprovalRequest(
        tool_call_id="call",
        tool_name="exec_command",
        redacted_arguments='{"argv":["git","status"],"cwd":"."}',
        scope_descriptor="exec_command:arguments:v1:" + "a" * 64,
        scope_fingerprint="a" * 64,
        can_allow_session=False,
    )

    with patch("builtins.input", return_value="o"):
        assert TerminalApprovalPrompt().decide(request) is ApprovalDecision.ALLOW_ONCE

    rendered = capsys.readouterr().out
    assert "Command: git status" in rendered
    assert "Directory: ." in rendered
    assert "hooks or configured filters" in rendered
    assert "this invocation only" in rendered


def test_approval_prompt_disallows_session_choice_for_high_risk_tool() -> None:
    request = ApprovalRequest(
        tool_call_id="call-1",
        tool_name="exec_command",
        redacted_arguments='{"argv":["git","status"]}',
        scope_descriptor='{"argv":["git","status"]}',
        scope_fingerprint="a" * 64,
        can_allow_session=False,
    )

    with patch("builtins.input", return_value="s"):
        decision = TerminalApprovalPrompt().decide(request)

    assert decision is ApprovalDecision.DENY


def test_approval_prompt_describes_apply_patch_path_scope(
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = ApprovalRequest(
        tool_call_id="call-1",
        tool_name="apply_patch",
        redacted_arguments='{"changes":[{"path":"notes.txt","replacement_content":"new"}]}',
        scope_descriptor="apply_patch:workspace-path:v1:" + "a" * 64,
        scope_fingerprint="a" * 64,
        can_allow_session=True,
    )

    with patch("builtins.input", return_value="s") as input_mock:
        decision = TerminalApprovalPrompt().decide(request)

    assert decision is ApprovalDecision.ALLOW_SESSION
    assert "s = exact same file path this session" in input_mock.call_args.args[0]
    assert "Approval required: apply_patch" in capsys.readouterr().out


def test_approval_prompt_safely_renders_untrusted_request_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = ApprovalRequest(
        tool_call_id="call-1",
        tool_name="apply_patch\x1b]52;c;clipboard\x1b\\",
        redacted_arguments='{"path":"notes.txt\u202e"}',
        scope_descriptor="apply_patch:arguments:v1:" + "a" * 64,
        scope_fingerprint="a" * 64,
        can_allow_session=False,
    )

    with patch("builtins.input", return_value="d"):
        decision = TerminalApprovalPrompt().decide(request)

    displayed = capsys.readouterr().out
    assert decision is ApprovalDecision.DENY
    assert "\\x1B]52;c;clipboard\\x1B\\" in displayed
    assert "\\u202E" in displayed


def test_cli_workspace_excludes_all_runtime_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "agent-data"
    session = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="test-model")
    config = MiniAgentConfig.model_validate(
        {
            "model": {
                "model": "test-model",
                "base_url": "https://example.test/v1",
            },
            "runtime": {"data_dir": data_dir},
        }
    )
    try:
        workspace = build_workspace_for_session(config, session)

        with pytest.raises(WorkspacePathError, match="excluded"):
            workspace.resolve_existing("agent-data")
    finally:
        session.close()


def test_show_context_prints_bounded_projection_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")

    class EmptyProvider:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            raise AssertionError(request)

    config = MiniAgentConfig.model_validate(
        {"model": {"model": "m", "base_url": "https://example.test/v1"}}
    )
    agent = MiniAgent(config=config, provider=EmptyProvider(), session=session)

    show_context(agent)

    output = capsys.readouterr().out
    assert "strategy=full_history" in output
    assert "tokens~=" in output
    assert "checkpoint=none" in output
    session.close()


def test_goal_commands_set_criteria_and_report_gate_gaps(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    handle_goal_command(session, "/goal set Ship a verified change")
    handle_goal_command(
        session,
        "/goal criteria "
        '[{"criterion_id":"tests","description":"Tests pass",'
        '"evidence_kind":"command_succeeded"}]',
    )
    handle_goal_command(session, "/goal complete []")

    output = capsys.readouterr().out
    assert "status=active" in output
    assert "tests [command_succeeded] Tests pass" in output
    assert "Completion gate rejected" in output
    assert "evidence_missing [tests]" in output
    assert session.goal is not None
    assert session.goal.status is GoalStatus.ACTIVE
    session.close()


@pytest.mark.parametrize("workspace_kind", ("missing", "file"))
def test_chat_rejects_invalid_workspace_before_creating_a_session(
    tmp_path: Path,
    workspace_kind: str,
) -> None:
    data_dir = tmp_path / "agent-data"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, data_dir=data_dir, model="test-model")
    workspace = tmp_path / "api_key=synthetic-workspace-value"
    if workspace_kind == "file":
        workspace.write_text("not a workspace", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "chat",
            "--workspace",
            str(workspace),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code != 0
    assert "Workspace must be an existing directory" in result.output
    assert "synthetic-workspace-value" not in result.output
    assert not data_dir.exists()


@pytest.mark.parametrize(
    "contents",
    (
        f"""
[model]
model = "test-model"
base_url = "https://example.test/v1"
api_key = "{_SYNTHETIC_CLI_SECRET}"
""".strip(),
        f"""
[model]
model = "{_SYNTHETIC_CLI_SECRET}
""".strip(),
    ),
)
def test_cli_hides_invalid_configuration_contents(
    tmp_path: Path,
    contents: str,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(contents, encoding="utf-8")

    result = runner.invoke(
        app,
        ["sessions", "--workspace", str(tmp_path), "--config", str(config_path)],
    )

    assert result.exit_code != 0
    assert "Configuration could not be loaded safely." in " ".join(result.output.split())
    assert "SYNTHETICCLICONFIGVALUE" not in result.output
    assert "Traceback" not in result.output
    assert "input_value" not in result.output


def test_evaluate_rejects_a_fixture_that_does_not_match_the_session(tmp_path: Path) -> None:
    data_dir, config_path, session_id = _create_cli_session(tmp_path, model="test-model")
    session_root = data_dir / "sessions" / session_id
    before = _tree_bytes(session_root)
    assert not (session_root / "checkpoints").exists()
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        '{"schema_version":1,"fixture_id":"mismatch","description":"mismatch",'
        '"atoms":[{"atom_id":"missing","category":"exact_detail","text":"x",'
        '"source_event_ids":[99],"is_current":true}]}',
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "evaluate",
            session_id,
            "--fixture",
            str(fixture),
            "--workspace",
            str(tmp_path),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code != 0
    assert "not match the session" in result.output
    assert data_dir.is_dir()
    assert _tree_bytes(session_root) == before
    assert not (session_root / "checkpoints").exists()


def test_chat_stops_safely_after_an_unexpected_turn_and_poisoned_session_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "agent-data"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, data_dir=data_dir, model="test-model")
    secret = synthetic_stripe_access_token("_TURN_FAILURE_")

    class FakeProvider:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    class FailingToolAgent:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def run_turn(self, _user_input: str) -> object:
            raise RuntimeError(f"custom tool failed: {secret}")

        def runtime_context(self) -> RuntimeContext:
            return RuntimeContext(
                model="test-model",
                workspace=tmp_path,
                platform="TestOS",
                git=GitContext(is_repository=False),
            )

        def status(self) -> AgentStatus:
            return AgentStatus(
                runtime=self.runtime_context(),
                context=None,
                checkpoint_version=None,
                goal_status=None,
                has_uncommitted_events=True,
            )

    def fail_stop(self: AgentSession, reason: str = "user_exit") -> object:
        del self, reason
        raise OSError(f"poisoned event store contains {secret}")

    monkeypatch.setattr("mini_agent.cli.OpenAICompatibleProvider", FakeProvider)
    monkeypatch.setattr("mini_agent.cli.MiniAgent", FailingToolAgent)
    monkeypatch.setattr(AgentSession, "stop", fail_stop)

    with patch("builtins.input", return_value="run it"):
        result = runner.invoke(
            app,
            ["chat", "--workspace", str(tmp_path), "--config", str(config_path)],
        )

    assert result.exit_code == 1, result.output
    assert "Agent turn failed; resume the session before continuing." in result.output
    assert (
        "Session state could not be finalized; resume the session before continuing."
        in result.output
    )
    assert secret not in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("input_effect", ("/exit", EOFError, KeyboardInterrupt))
def test_resume_finalization_failure_returns_nonzero_without_stopped_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_effect: object,
) -> None:
    data_dir, config_path, session_id = _create_cli_session(tmp_path, model="test-model")
    secret = "session-finalization-private-detail"

    def fail_stop(self: AgentSession, reason: str = "user_exit") -> object:
        del self, reason
        raise OSError(secret)

    monkeypatch.setattr(AgentSession, "stop", fail_stop)

    input_patch = (
        patch("builtins.input", return_value=input_effect)
        if isinstance(input_effect, str)
        else patch("builtins.input", side_effect=input_effect)
    )
    with input_patch:
        result = runner.invoke(
            app,
            [
                "resume",
                session_id,
                "--workspace",
                str(tmp_path),
                "--config",
                str(config_path),
            ],
        )

    assert result.exit_code == 1
    assert (
        "Session state could not be finalized; resume the session before continuing."
        in _plain_cli_output(result.output)
    )
    assert "Traceback" not in result.output
    assert secret not in result.output
    assert str(tmp_path) not in result.output
    resumed = AgentSession.load(data_dir=data_dir, session_id=session_id)
    try:
        assert all(event.data.kind != "session_stopped" for event in resumed.events)
    finally:
        resumed.close()


def test_resume_recovers_truncated_tail_after_read_only_model_check(tmp_path: Path) -> None:
    data_dir, config_path, session_id = _create_cli_session(tmp_path, model="test-model")
    events_path = data_dir / "sessions" / session_id / "events.jsonl"
    events_path.write_bytes(events_path.read_bytes() + b'{"id":2')
    interactive = AsyncMock()

    with patch("mini_agent.cli._run_interactive", interactive):
        result = runner.invoke(
            app,
            [
                "resume",
                session_id,
                "--workspace",
                str(tmp_path),
                "--config",
                str(config_path),
            ],
        )

    assert result.exit_code == 0, result.output
    interactive.assert_awaited_once()
    await_args = interactive.await_args
    assert await_args is not None
    assert await_args.kwargs["registrations"] == ()
    resumed = AgentSession.load(data_dir=data_dir, session_id=session_id)
    try:
        assert resumed.events[-1].data.kind == "session_resumed"
    finally:
        resumed.close()
    assert list(events_path.parent.glob("events.jsonl.recovery-*.bin"))


@pytest.mark.asyncio
async def test_resumed_interactive_session_replays_recent_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, _config_path, session_id = _create_cli_session(tmp_path, model="test-model")
    session = AgentSession.resume(data_dir=data_dir, session_id=session_id)
    turn_id = session.new_turn_id()
    session.append_user_message("continue the previous task", turn_id=turn_id)
    session.append_assistant_message(
        "the previous result",
        finish_reason=FinishReason.STOP,
        turn_id=turn_id,
    )
    config = MiniAgentConfig.model_validate(
        {
            "model": {"model": "test-model", "base_url": "https://example.test/v1"},
            "runtime": {"data_dir": data_dir},
        }
    )

    class FakeProvider:
        async def aclose(self) -> None:
            pass

    def fake_provider(_config: object) -> FakeProvider:
        return FakeProvider()

    monkeypatch.setattr("mini_agent.cli.OpenAICompatibleProvider", fake_provider)
    with (
        patch("mini_agent.cli.TerminalChatUI.show_recent_conversation") as history,
        patch("builtins.input", return_value="/exit"),
    ):
        result = await cli_module._run_interactive(  # pyright: ignore[reportPrivateUsage]
            config,
            session,
            registrations=(),
        )

    assert result.stop_reason == "user_exit"
    replayed_messages = history.call_args.args[0]
    assert history.call_args.kwargs["limit"] == 30
    assert [message.content for message in replayed_messages] == [
        "continue the previous task",
        "the previous result",
    ]
    session.close()


@pytest.mark.asyncio
async def test_run_interactive_finalizes_session_on_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ctrl-C while the model is thinking must stop the session cleanly so the
    # terminal event is recorded and the writer lock is released, instead of
    # aborting past session.finalize() and leaving the session unresumable.
    session = AgentSession.create(
        data_dir=tmp_path / "data", workspace=tmp_path, model="test-model"
    )
    config = MiniAgentConfig.model_validate(
        {
            "model": {"model": "test-model", "base_url": "https://example.test/v1"},
            "runtime": {"data_dir": tmp_path / "data"},
        }
    )

    class FakeProvider:
        async def aclose(self) -> None:
            pass

    def fake_provider(_config: object) -> FakeProvider:
        return FakeProvider()

    monkeypatch.setattr("mini_agent.cli.OpenAICompatibleProvider", fake_provider)
    monkeypatch.setattr(
        "mini_agent.agent.MiniAgent.run_turn",
        AsyncMock(side_effect=asyncio.CancelledError()),
    )

    with (
        patch("mini_agent.cli.TerminalChatUI.show_recent_conversation"),
        patch("builtins.input", return_value="please continue the work"),
        patch("mini_agent.cli.typer.echo"),
    ):
        result = await cli_module._run_interactive(  # pyright: ignore[reportPrivateUsage]
            config,
            session,
            registrations=(),
        )

    assert result.stop_reason == "user_interrupt"
    assert session.events[-1].data.kind == "session_stopped"
    assert session.events[-1].data.reason == "user_interrupt"
    session.close()


def test_resume_without_id_selects_latest_session_for_workspace_and_model(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "agent-data"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, data_dir=data_dir, model="test-model")
    older = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="test-model")
    older_id = older.metadata.session_id
    older.append_user_message("older work", turn_id=older.new_turn_id())
    older.close()
    other_workspace = tmp_path / "other"
    other_workspace.mkdir()
    other = AgentSession.create(
        data_dir=data_dir,
        workspace=other_workspace,
        model="test-model",
    )
    other.close()
    latest = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="test-model")
    latest_id = latest.metadata.session_id
    latest.append_user_message("latest work", turn_id=latest.new_turn_id())
    latest.close()
    interactive = AsyncMock()

    with (
        patch("mini_agent.cli.AgentSession.resume", wraps=AgentSession.resume) as resume_mock,
        patch("mini_agent.cli._run_interactive", interactive),
    ):
        result = runner.invoke(
            app,
            [
                "resume",
                "--workspace",
                str(tmp_path),
                "--config",
                str(config_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert older_id != latest_id
    assert resume_mock.call_args.kwargs["session_id"] == latest_id


def test_resume_without_id_can_select_an_older_session_interactively(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "agent-data"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, data_dir=data_dir, model="test-model")
    older = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="test-model")
    older_id = older.metadata.session_id
    older.append_user_message("older work", turn_id=older.new_turn_id())
    older.close()
    latest = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="test-model")
    latest.append_user_message("latest work", turn_id=latest.new_turn_id())
    latest.close()
    interactive = AsyncMock()

    with (
        patch("mini_agent.cli._is_interactive_terminal", return_value=True),
        patch("mini_agent.cli.TerminalChatUI.select_session", return_value=older_id),
        patch("mini_agent.cli.AgentSession.resume", wraps=AgentSession.resume) as resume_mock,
        patch("mini_agent.cli._run_interactive", interactive),
    ):
        result = runner.invoke(
            app,
            [
                "resume",
                "--workspace",
                str(tmp_path),
                "--config",
                str(config_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert resume_mock.call_args.kwargs["session_id"] == older_id


def test_resume_without_id_reports_when_workspace_has_no_matching_session(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "agent-data"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, data_dir=data_dir, model="test-model")

    result = runner.invoke(
        app,
        ["resume", "--workspace", str(tmp_path), "--config", str(config_path)],
    )

    assert result.exit_code != 0
    assert "No resumable session was found" in result.output


def test_resume_with_an_exact_id_rejects_a_different_workspace_before_writing(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "agent-data"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, data_dir=data_dir, model="test-model")
    first_workspace = tmp_path / "first-workspace"
    second_workspace = tmp_path / "second-workspace"
    first_workspace.mkdir()
    second_workspace.mkdir()
    session = AgentSession.create(
        data_dir=data_dir,
        workspace=second_workspace,
        model="test-model",
    )
    session_id = session.metadata.session_id
    session.close()

    async def close_resumed_session(
        _config: MiniAgentConfig,
        resumed_session: AgentSession,
        **_kwargs: object,
    ) -> InteractiveResult:
        resumed_session.close()
        return InteractiveResult(stop_reason="user_exit")

    with (
        patch("mini_agent.cli.AgentSession.resume", wraps=AgentSession.resume) as resume_mock,
        patch("mini_agent.cli._run_interactive", side_effect=close_resumed_session),
    ):
        rejected = runner.invoke(
            app,
            [
                "resume",
                session_id,
                "--workspace",
                str(first_workspace),
                "--config",
                str(config_path),
            ],
            terminal_width=200,
        )

        assert rejected.exit_code != 0
        assert "Requested session belongs to a different workspace." in " ".join(
            rejected.output.replace("│", "").split()
        )
        assert session_id not in rejected.output
        assert str(first_workspace) not in rejected.output
        assert str(second_workspace) not in rejected.output
        assert "example.test" not in rejected.output
        resume_mock.assert_not_called()

        accepted = runner.invoke(
            app,
            [
                "resume",
                session_id,
                "--workspace",
                str(second_workspace),
                "--config",
                str(config_path),
            ],
        )

    assert accepted.exit_code == 0, accepted.output
    assert resume_mock.call_count == 1
    assert resume_mock.call_args.kwargs["session_id"] == session_id


def test_empty_legacy_session_is_hidden_but_remains_explicitly_addressable(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "agent-data"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, data_dir=data_dir, model="test-model")
    empty = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="test-model")
    empty_id = empty.metadata.session_id
    empty.close()

    sessions_result = runner.invoke(
        app,
        ["sessions", "--workspace", str(tmp_path), "--config", str(config_path)],
    )
    implicit_resume = runner.invoke(
        app,
        ["resume", "--workspace", str(tmp_path), "--config", str(config_path)],
    )

    assert sessions_result.exit_code == 0, sessions_result.output
    assert "No sessions found." in sessions_result.output
    assert empty_id not in sessions_result.output
    assert implicit_resume.exit_code != 0
    assert "No resumable session was found" in implicit_resume.output
    metadata = AgentSession.peek_metadata(data_dir=data_dir, session_id=empty_id)
    assert metadata.session_id == empty_id


@pytest.mark.asyncio
async def test_slash_resume_requests_current_session_without_calling_the_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, _config_path, session_id = _create_cli_session(tmp_path, model="test-model")
    config = MiniAgentConfig.model_validate(
        {
            "model": {"model": "test-model", "base_url": "https://example.test/v1"},
            "runtime": {"data_dir": data_dir},
        }
    )
    session = AgentSession.resume(data_dir=data_dir, session_id=session_id)

    class FakeProvider:
        async def aclose(self) -> None:
            pass

    def fake_provider(_config: object) -> FakeProvider:
        return FakeProvider()

    class FakeAgent:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def runtime_context(self) -> RuntimeContext:
            return RuntimeContext(
                model="test-model",
                workspace=tmp_path,
                platform="TestOS",
                git=GitContext(is_repository=False),
            )

        def status(self) -> AgentStatus:
            return AgentStatus(
                runtime=self.runtime_context(),
                context=None,
                checkpoint_version=None,
                goal_status=None,
                has_uncommitted_events=True,
            )

        async def run_turn(self, _user_input: str) -> object:
            raise AssertionError("/resume must not call the model")

    monkeypatch.setattr("mini_agent.cli.OpenAICompatibleProvider", fake_provider)
    monkeypatch.setattr("mini_agent.cli.MiniAgent", FakeAgent)
    with patch("builtins.input", return_value="/resume"):
        result = await cli_module._run_interactive(  # pyright: ignore[reportPrivateUsage]
            config,
            session,
            registrations=(),
        )

    assert result.stop_reason == "resume_requested"
    assert result.resume_session_id == session_id
    session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/checkpoint", "/compact retain focus"])
async def test_checkpoint_recovery_required_slash_command_stops_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "data"
    config = MiniAgentConfig.model_validate(
        {
            "model": {"model": "test-model", "base_url": "https://example.test/v1"},
            "runtime": {"data_dir": data_dir},
        }
    )
    session = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="test-model")

    class FakeProvider:
        async def aclose(self) -> None:
            return None

    def fake_provider(_config: MiniAgentConfig) -> FakeProvider:
        return FakeProvider()

    class FailingCheckpointAgent:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def checkpoint(self) -> None:
            raise CheckpointRecoveryRequiredError()

        async def compact(self, *, focus: str | None = None) -> object:
            del focus
            raise CheckpointRecoveryRequiredError()

        def status(self) -> AgentStatus:
            return AgentStatus(
                runtime=RuntimeContext(
                    model="test-model",
                    workspace=tmp_path,
                    platform="TestOS",
                    git=GitContext(is_repository=False),
                ),
                context=None,
                checkpoint_version=None,
                goal_status=None,
                has_uncommitted_events=True,
            )

        async def run_turn(self, _user_input: str) -> object:
            raise AssertionError("checkpoint recovery must stop before another turn")

    monkeypatch.setattr("mini_agent.cli.OpenAICompatibleProvider", fake_provider)
    monkeypatch.setattr("mini_agent.cli.MiniAgent", FailingCheckpointAgent)
    with patch("builtins.input", return_value=command):
        result = await cli_module._run_interactive(  # pyright: ignore[reportPrivateUsage]
            config,
            session,
            registrations=(),
            show_welcome=False,
        )

    assert result.stop_reason == "checkpoint_recovery_required"
    assert "Checkpoint state requires resume; this session will stop." in capsys.readouterr().err
    session.close()


def test_session_chain_closes_current_writer_before_resuming_target(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    config = MiniAgentConfig.model_validate(
        {
            "model": {"model": "test-model", "base_url": "https://example.test/v1"},
            "runtime": {"data_dir": data_dir},
        }
    )
    first = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="test-model")
    second = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="test-model")
    second_id = second.metadata.session_id
    second.close()
    results = iter(
        (
            InteractiveResult(
                stop_reason="resume_requested",
                resume_session_id=second_id,
                output_mode=OutputMode.DETAILED,
            ),
            InteractiveResult(stop_reason="user_exit"),
        )
    )

    modes: list[OutputMode] = []

    def interactive(
        _config: MiniAgentConfig,
        session: AgentSession,
        *,
        registrations: tuple[object, ...],
        output_mode: OutputMode,
        auto_approve: bool,
    ) -> InteractiveResult:
        del _config, registrations
        assert auto_approve
        modes.append(output_mode)
        result = next(results)
        session.close()
        return result

    with patch("mini_agent.cli._run_interactive_safely", side_effect=interactive):
        stop_reason = cli_module._run_session_chain(  # pyright: ignore[reportPrivateUsage]
            config,
            first,
            registrations=(),
            auto_approve=True,
        )

    assert stop_reason == "user_exit"
    assert modes == [OutputMode.BRIEF, OutputMode.DETAILED]


def test_interactive_safe_runner_releases_writer_after_normal_exit(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    config = MiniAgentConfig.model_validate(
        {"model": {"model": "m", "base_url": "https://example.test/v1"}}
    )

    async def normal_exit(*_args: object, **_kwargs: object) -> InteractiveResult:
        return InteractiveResult(stop_reason="user_exit")

    with patch("mini_agent.cli._run_interactive", new=normal_exit):
        result = cli_module._run_interactive_safely(  # pyright: ignore[reportPrivateUsage]
            config,
            session,
            registrations=(),
        )

    assert result.stop_reason == "user_exit"
    reopened = EventStore.open(session.paths.events)
    reopened.close()


def test_interactive_safe_runner_releases_writer_after_unexpected_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    config = MiniAgentConfig.model_validate(
        {"model": {"model": "m", "base_url": "https://example.test/v1"}}
    )
    secret = synthetic_stripe_access_token("_INTERACTIVE_FAILURE_")

    async def fail_interactive(*_args: object, **_kwargs: object) -> InteractiveResult:
        raise RuntimeError(f"unexpected agent failure at {tmp_path}: {secret}")

    with patch("mini_agent.cli._run_interactive", new=fail_interactive):
        result = cli_module._run_interactive_safely(  # pyright: ignore[reportPrivateUsage]
            config,
            session,
            registrations=(),
        )

    assert result.stop_reason == "turn_failure"
    reopened = EventStore.open(session.paths.events)
    reopened.close()
    stderr = capsys.readouterr().err
    assert "Agent session failed; resume the session before continuing." in stderr
    assert secret not in stderr
    assert str(tmp_path) not in stderr
    assert "Traceback" not in stderr


def test_session_chain_does_not_open_resume_target_after_current_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "data"
    config = MiniAgentConfig.model_validate(
        {
            "model": {"model": "test-model", "base_url": "https://example.test/v1"},
            "runtime": {"data_dir": data_dir},
        }
    )
    current = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="test-model")
    target = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="test-model")
    target_id = target.metadata.session_id
    target.close()
    secret = synthetic_stripe_access_token("_CLOSE_FAILURE_")
    original_close = AgentSession.close

    async def request_resume(*_args: object, **_kwargs: object) -> InteractiveResult:
        return InteractiveResult(stop_reason="resume_requested", resume_session_id=target_id)

    def fail_close(_session: AgentSession) -> None:
        raise OSError(f"close failure at {tmp_path}: {secret}")

    monkeypatch.setattr("mini_agent.cli._run_interactive", request_resume)
    monkeypatch.setattr(AgentSession, "close", fail_close)
    try:
        with patch("mini_agent.cli.AgentSession.resume", wraps=AgentSession.resume) as resume:
            stop_reason = cli_module._run_session_chain(  # pyright: ignore[reportPrivateUsage]
                config,
                current,
                registrations=(),
            )
    finally:
        monkeypatch.setattr(AgentSession, "close", original_close)
        current.close()

    assert stop_reason == "session_finalization_failed"
    resume.assert_not_called()
    stderr = capsys.readouterr().err
    assert "Session cleanup failed; resume the session before continuing." in stderr
    assert secret not in stderr
    assert str(tmp_path) not in stderr
    assert "Traceback" not in stderr


def test_chat_provider_cleanup_failure_returns_nonzero_with_redacted_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "agent-data"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, data_dir=data_dir, model="test-model")
    secret = synthetic_stripe_access_token("_PROVIDER_CLOSE_")

    class FailingCloseProvider:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def aclose(self) -> None:
            raise RuntimeError(f"provider cleanup failed at {tmp_path}: {secret}")

    class FakeAgent:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def run_turn(self, _user_input: str) -> ModelResponse:
            return ModelResponse(finish_reason=FinishReason.STOP, content="ok")

        def status(self) -> AgentStatus:
            return AgentStatus(
                runtime=RuntimeContext(
                    model="test-model",
                    workspace=tmp_path,
                    platform="TestOS",
                    git=GitContext(is_repository=False),
                ),
                context=None,
                checkpoint_version=None,
                goal_status=None,
                has_uncommitted_events=True,
            )

    monkeypatch.setattr("mini_agent.cli.OpenAICompatibleProvider", FailingCloseProvider)
    monkeypatch.setattr("mini_agent.cli.MiniAgent", FakeAgent)
    with patch("builtins.input", side_effect=("run", "/exit")):
        result = runner.invoke(
            app,
            ["chat", "--workspace", str(tmp_path), "--config", str(config_path)],
        )

    assert result.exit_code == 1, result.output
    assert "Agent cleanup failed; resume the session before continuing." in result.output
    assert secret not in result.output
    assert str(tmp_path) not in result.output
    assert "Traceback" not in result.output
    session_id = list_sessions(data_dir)[0].session_id
    resumed = AgentSession.load(data_dir=data_dir, session_id=session_id)
    try:
        assert resumed.events[-1].data.kind == "session_stopped"
    finally:
        resumed.close()


def test_slash_resume_rejects_corrupted_target_without_closing_current_session(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "data"
    config = MiniAgentConfig.model_validate(
        {
            "model": {"model": "test-model", "base_url": "https://example.test/v1"},
            "runtime": {"data_dir": data_dir},
        }
    )
    current = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="test-model")
    target = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="test-model")
    target_id = target.metadata.session_id
    target.close()
    target.paths.metadata.write_text("not-json", encoding="utf-8")

    can_switch = cli_module._can_switch_session(  # pyright: ignore[reportPrivateUsage]
        config=config,
        current_session=current,
        candidate_id=target_id,
    )

    assert can_switch is False
    assert "not found or is not readable" in capsys.readouterr().err
    current.append_user_message("still writable", turn_id=current.new_turn_id())
    current.close()


def test_slash_resume_rejects_corrupted_event_stream_before_switching(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "data"
    config = MiniAgentConfig.model_validate(
        {
            "model": {"model": "test-model", "base_url": "https://example.test/v1"},
            "runtime": {"data_dir": data_dir},
        }
    )
    current = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="test-model")
    target = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="test-model")
    target_id = target.metadata.session_id
    target.close()
    with target.paths.events.open("ab") as event_file:
        event_file.write(b'{"id":2')

    can_switch = cli_module._can_switch_session(  # pyright: ignore[reportPrivateUsage]
        config=config,
        current_session=current,
        candidate_id=target_id,
    )

    assert can_switch is False
    assert "not found or is not readable" in capsys.readouterr().err
    current.append_user_message("still writable", turn_id=current.new_turn_id())
    current.close()


def test_resume_model_mismatch_does_not_recover_or_modify_history(tmp_path: Path) -> None:
    data_dir, config_path, session_id = _create_cli_session(tmp_path, model="recorded-model")
    _write_config(config_path, data_dir=data_dir, model="different-model")
    events_path = data_dir / "sessions" / session_id / "events.jsonl"
    events_path.write_bytes(events_path.read_bytes() + b'{"id":2')
    history_before = events_path.read_bytes()

    result = runner.invoke(
        app,
        [
            "resume",
            session_id,
            "--workspace",
            str(tmp_path),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code != 0
    assert "Configured model differs" in result.output
    assert events_path.read_bytes() == history_before
    assert not list(events_path.parent.glob("events.jsonl.recovery-*.bin"))


def test_cli_config_read_error_has_stable_redacted_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "company-internal-config-content"

    def fail_load_config(_path: Path) -> MiniAgentConfig:
        raise OSError(secret)

    monkeypatch.setattr("mini_agent.cli.load_config", fail_load_config)

    result = runner.invoke(
        app,
        ["sessions", "--workspace", str(tmp_path), "--config", str(tmp_path / "config.toml")],
    )

    assert result.exit_code != 0
    assert "Configuration could not be loaded safely." in result.output
    assert "Traceback" not in result.output
    assert secret not in result.output
    assert str(tmp_path) not in result.output


def test_sessions_discovery_io_error_has_stable_redacted_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "agent-data"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, data_dir=data_dir, model="test-model")
    secret = "company-internal-session-discovery-content"

    def fail_discovery(_data_dir: Path) -> object:
        raise OSError(secret)

    monkeypatch.setattr("mini_agent.cli.discover_sessions", fail_discovery)

    result = runner.invoke(
        app,
        ["sessions", "--workspace", str(tmp_path), "--config", str(config_path)],
    )

    assert result.exit_code == 1
    assert "Session discovery could not be completed safely." in result.output
    assert "Traceback" not in result.output
    assert secret not in result.output
    assert str(tmp_path) not in result.output


def test_inspect_corrupt_or_missing_session_has_stable_redacted_diagnostic(
    tmp_path: Path,
) -> None:
    data_dir, config_path, session_id = _create_cli_session(tmp_path, model="test-model")
    secret = "company-internal-corrupted-session-content"
    (data_dir / "sessions" / session_id / "metadata.json").write_text(secret, encoding="utf-8")

    result = runner.invoke(
        app,
        ["inspect", session_id, "--workspace", str(tmp_path), "--config", str(config_path)],
    )

    assert result.exit_code != 0
    assert "Requested session was not found or is not readable." in _plain_cli_output(result.output)
    assert "Traceback" not in result.output
    assert secret not in result.output
    assert session_id not in result.output
    assert str(tmp_path) not in result.output


@pytest.mark.parametrize(
    ("error_type", "expected_message"),
    (
        (EventStoreWriterBusyError, "Requested session is currently in use by another process."),
        (CheckpointStoreCorruptionError, "Requested session was not found or is not readable."),
        (ArtifactStoreCorruptionError, "Requested session was not found or is not readable."),
    ),
)
def test_resume_storage_failures_have_stable_redacted_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
    expected_message: str,
) -> None:
    _data_dir, config_path, session_id = _create_cli_session(tmp_path, model="test-model")
    secret = "company-internal-storage-content"

    def fail_resume(*, data_dir: Path, session_id: str) -> AgentSession:
        del data_dir, session_id
        raise error_type(secret)

    monkeypatch.setattr(AgentSession, "resume", fail_resume)

    result = runner.invoke(
        app,
        ["resume", session_id, "--workspace", str(tmp_path), "--config", str(config_path)],
    )

    assert result.exit_code != 0
    assert expected_message in _plain_cli_output(result.output)
    assert "Traceback" not in result.output
    assert secret not in result.output
    assert session_id not in result.output
    assert str(tmp_path) not in result.output


def test_evaluate_checkpoint_corruption_has_stable_redacted_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _data_dir, config_path, session_id = _create_cli_session(tmp_path, model="test-model")
    secret = "company-internal-checkpoint-content"
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text('{"atoms": []}', encoding="utf-8")

    def fail_open(*_args: object, **_kwargs: object) -> CheckpointStoreCorruptionError:
        raise CheckpointStoreCorruptionError(secret)

    monkeypatch.setattr("mini_agent.cli.CheckpointStore.open", fail_open)

    result = runner.invoke(
        app,
        [
            "evaluate",
            session_id,
            "--fixture",
            str(fixture_path),
            "--workspace",
            str(tmp_path),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code != 0
    assert "Requested session was not found or is not readable." in _plain_cli_output(result.output)
    assert "Traceback" not in result.output
    assert secret not in result.output
    assert session_id not in result.output
    assert str(tmp_path) not in result.output


def _create_cli_session(tmp_path: Path, *, model: str) -> tuple[Path, Path, str]:
    data_dir = tmp_path / "agent-data"
    config_path = tmp_path / "config.toml"
    _write_config(config_path, data_dir=data_dir, model=model)
    session = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model=model)
    try:
        return data_dir, config_path, session.metadata.session_id
    finally:
        session.close()


def _tree_bytes(root: Path) -> dict[Path, bytes | None]:
    """Capture a small session tree without following links."""

    snapshot: dict[Path, bytes | None] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        snapshot[relative] = None if path.is_dir() else path.read_bytes()
    return snapshot


def _plain_cli_output(output: str) -> str:
    """Normalize Rich's visual borders so stable messages can wrap safely."""

    return " ".join(output.replace("│", "").split())


def _write_config(config_path: Path, *, data_dir: Path, model: str) -> None:
    config_path.write_text(
        "\n".join(
            (
                "[model]",
                f'model = "{model}"',
                'base_url = "https://example.test/v1"',
                "",
                "[runtime]",
                f'data_dir = "{data_dir}"',
                "",
            )
        ),
        encoding="utf-8",
    )
