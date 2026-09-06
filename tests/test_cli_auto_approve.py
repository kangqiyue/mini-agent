from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import HttpUrl
from typer.testing import CliRunner

from mini_agent import cli
from mini_agent.config import MiniAgentConfig, ModelConfig, RuntimeConfig
from mini_agent.events import ApprovalDecision, ApprovalResolvedData, ToolFailedData
from mini_agent.messages import FinishReason, ModelResponse, ToolCall
from mini_agent.session import AgentSession
from mini_agent.tools.apply_patch import ApplyPatchArguments, FileReplacement
from tests.support.providers import SequenceProvider

runner = CliRunner()


class InteractiveProvider(SequenceProvider):
    async def aclose(self) -> None:
        pass


def test_default_chat_accepts_explicit_auto_approval() -> None:
    with patch("mini_agent.cli.chat") as chat:
        result = runner.invoke(cli.app, ["--auto-approve"])

    assert result.exit_code == 0, result.output
    chat.assert_called_once_with(auto_approve=True)


def test_misplaced_auto_approval_option_is_never_silently_ignored() -> None:
    with patch("mini_agent.cli._run_interactive", AsyncMock()) as interactive:
        result = runner.invoke(cli.app, ["--auto-approve", "chat"])

    assert result.exit_code == 2
    assert "Place --auto-approve" in result.output
    assert "subcommand" in result.output
    interactive.assert_not_called()


@pytest.mark.parametrize("auto_approve", [False, True])
@pytest.mark.parametrize("path", ["note.txt", "../outside.txt"])
async def test_interactive_auto_approval_skips_prompt_but_preserves_validation_and_audit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    auto_approve: bool,
    path: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = MiniAgentConfig(
        model=ModelConfig(model="test-model", base_url=HttpUrl("https://example.test/v1")),
        runtime=RuntimeConfig(data_dir=tmp_path / "data"),
    )
    session = AgentSession.create(
        data_dir=tmp_path / "data", workspace=workspace, model="test-model"
    )
    call = ToolCall(
        id="write-call",
        name="apply_patch",
        arguments_json=ApplyPatchArguments(
            changes=(FileReplacement(path=path, replacement_content="verified content"),)
        ).model_dump_json(),
    )
    provider = InteractiveProvider(
        (
            ModelResponse(tool_calls=(call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="finished", finish_reason=FinishReason.STOP),
        )
    )
    try:
        with (
            patch("mini_agent.cli.OpenAICompatibleProvider", return_value=provider),
            patch(
                "mini_agent.cli.TerminalChatUI.read_input",
                side_effect=["/permissions", "write the note", "/exit"],
            ),
            patch("builtins.input", return_value="d") as prompt,
        ):
            result = await cli._run_interactive(  # pyright: ignore[reportPrivateUsage]
                config,
                session,
                registrations=(),
                auto_approve=auto_approve,
            )
        assert result.stop_reason == "user_exit"
        assert (workspace / "note.txt").exists() is (auto_approve and path == "note.txt")
        assert not (tmp_path / "outside.txt").exists()
        assert not session.session_grant_fingerprints
        decisions = [
            e.data.decision for e in session.events if isinstance(e.data, ApprovalResolvedData)
        ]
        if path == "../outside.txt":
            assert decisions == []
            assert any(isinstance(e.data, ToolFailedData) for e in session.events)
        else:
            expected = ApprovalDecision.ALLOW_ONCE if auto_approve else ApprovalDecision.DENY
            assert decisions == [expected]
        assert prompt.call_count == int(not auto_approve and path == "note.txt")
        rendered = capsys.readouterr().out
        if auto_approve:
            assert "approvals AUTO" in rendered
            assert "without prompting" in rendered
        else:
            assert "approvals ask" in rendered
    finally:
        session.close()


@pytest.mark.parametrize("auto_approve", [False, True])
def test_explicit_resume_requires_its_own_auto_approve_flag(
    tmp_path: Path,
    auto_approve: bool,
) -> None:
    data_dir = tmp_path / "data"
    session = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="test-model")
    session.append_user_message("existing request", turn_id=session.new_turn_id())
    session.stop()
    session.close()
    config = MiniAgentConfig(
        model=ModelConfig(model="test-model", base_url=HttpUrl("https://example.test/v1")),
        runtime=RuntimeConfig(data_dir=data_dir),
    )
    arguments = ["resume", session.metadata.session_id, "--workspace", str(tmp_path)]
    if auto_approve:
        arguments.append("--auto-approve")
    with (
        patch("mini_agent.cli._load_runtime_config", return_value=config),
        patch(
            "mini_agent.cli._run_interactive",
            AsyncMock(return_value=cli.InteractiveResult(stop_reason="user_exit")),
        ) as interactive,
    ):
        result = runner.invoke(cli.app, arguments)

    assert result.exit_code == 0, result.output
    assert interactive.await_args is not None
    assert interactive.await_args.kwargs["auto_approve"] is auto_approve
