from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import HttpUrl
from typer.testing import CliRunner

from mini_agent.cli import app
from mini_agent.config import MiniAgentConfig, ModelConfig, RuntimeConfig
from mini_agent.events import ArtifactCreatedData, ToolCompletedData
from mini_agent.headless import run_headless_task
from mini_agent.messages import FinishReason, ModelResponse, TokenUsage, ToolCall
from mini_agent.session import AgentSession
from mini_agent.tools.apply_patch import ApplyPatchArguments, ApplyPatchTool, FileReplacement
from mini_agent.tools.read_file import ReadFileArguments, ReadFileTool
from mini_agent.tools.registry import ToolRegistry
from mini_agent.workspace import Workspace
from tests.support.providers import SequenceProvider


@pytest.fixture
def config(tmp_path: Path) -> MiniAgentConfig:
    return MiniAgentConfig(
        model=ModelConfig(model="test-model", base_url=HttpUrl("https://example.test/v1")),
        runtime=RuntimeConfig(data_dir=tmp_path / "data", inline_tool_result_max_chars=100),
    )


@pytest.fixture
def session(config: MiniAgentConfig, tmp_path: Path) -> Iterator[AgentSession]:
    assert config.runtime.data_dir is not None
    session = AgentSession.create(
        data_dir=config.runtime.data_dir, workspace=tmp_path, model=config.model.model
    )
    try:
        yield session
    finally:
        session.close()


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (None, (None, None, None)),
        (TokenUsage(total_tokens=9), (None, None, 9)),
        (TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0), (0, 0, 0)),
        (TokenUsage(prompt_tokens=5, completion_tokens=4, total_tokens=9), (5, 4, 9)),
    ],
)
async def test_headless_preserves_missing_usage_fields_and_current_goal(
    config: MiniAgentConfig,
    session: AgentSession,
    usage: TokenUsage | None,
    expected: tuple[int | None, int | None, int | None],
) -> None:
    result = await run_headless_task(
        config=config,
        session=session,
        tools=ToolRegistry(()),
        task="answer",
        provider=SequenceProvider([ModelResponse(content="done", usage=usage)]),
    )

    assert (result.prompt_tokens, result.completion_tokens, result.total_tokens) == expected
    assert result.goal_status == "active"
    assert result.stop_reason == "completed"


async def test_headless_counts_only_the_current_turn(
    config: MiniAgentConfig, session: AgentSession
) -> None:
    provider = SequenceProvider(
        [
            ModelResponse(content="first", usage=TokenUsage(total_tokens=10)),
            ModelResponse(content="second", usage=TokenUsage(total_tokens=20)),
        ]
    )
    for expected in (10, 20):
        result = await run_headless_task(
            config=config,
            session=session,
            tools=ToolRegistry(()),
            task="answer",
            provider=provider,
        )
        assert result.total_tokens == expected


@pytest.mark.parametrize("final_usage", [None, TokenUsage(total_tokens=9)])
async def test_headless_does_not_report_partial_usage_as_a_complete_total(
    config: MiniAgentConfig, session: AgentSession, tmp_path: Path, final_usage: TokenUsage | None
) -> None:
    (tmp_path / "notes.txt").write_text("notes", encoding="utf-8")
    provider = SequenceProvider(
        [
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="read_notes",
                        name="read_file",
                        arguments_json=ReadFileArguments(path="notes.txt").model_dump_json(),
                    ),
                ),
                usage=TokenUsage(prompt_tokens=5, completion_tokens=4, total_tokens=9),
            ),
            ModelResponse(content="done", usage=final_usage),
        ]
    )
    result = await run_headless_task(
        config=config,
        session=session,
        task="read notes",
        provider=provider,
        tools=ToolRegistry((ReadFileTool(Workspace(tmp_path)),)),
    )
    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert result.total_tokens == (18 if final_usage is not None else None)


@pytest.mark.parametrize("auto_approve", [False, True])
async def test_headless_requires_explicit_approval_for_writes(
    config: MiniAgentConfig, session: AgentSession, tmp_path: Path, auto_approve: bool
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    provider = SequenceProvider(
        [
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="write_notes",
                        name="apply_patch",
                        arguments_json=ApplyPatchArguments(
                            changes=(
                                FileReplacement(
                                    path="notes.txt",
                                    expected_content="before",
                                    replacement_content="after",
                                ),
                            )
                        ).model_dump_json(),
                    ),
                ),
            ),
            ModelResponse(content="done"),
        ]
    )
    result = await run_headless_task(
        config=config,
        provider=provider,
        session=session,
        task="edit notes",
        tools=ToolRegistry((ApplyPatchTool(Workspace(tmp_path)),)),
        auto_approve=auto_approve,
    )
    assert target.read_text(encoding="utf-8") == ("after" if auto_approve else "before")
    assert result.tool_calls == 1
    assert result.tool_failures == (0 if auto_approve else 1)


class _ClosableProvider(SequenceProvider):
    has_close_failure = False

    async def aclose(self) -> None:
        if self.has_close_failure:
            raise OSError("simulated cleanup failure")


@pytest.mark.parametrize("failure_stage", ["provider", "stop", "close"])
def test_headless_cli_returns_nonzero_on_cleanup_failure(
    config: MiniAgentConfig, tmp_path: Path, failure_stage: str
) -> None:
    assert config.runtime.data_dir is not None
    provider = _ClosableProvider([ModelResponse(content="done")])
    provider.has_close_failure = failure_stage == "provider"
    original_stop = AgentSession.stop
    original_close = AgentSession.close

    def stop(session: AgentSession, reason: str) -> None:
        if failure_stage == "stop":
            raise OSError("simulated stop failure")
        original_stop(session, reason)

    def close(session: AgentSession) -> None:
        original_close(session)
        if failure_stage == "close":
            raise OSError("simulated close failure")

    with (
        patch("mini_agent.cli._load_runtime_config", return_value=config),
        patch("mini_agent.cli.OpenAICompatibleProvider", return_value=provider),
        patch.object(AgentSession, "stop", stop),
        patch.object(AgentSession, "close", close),
    ):
        result = CliRunner().invoke(app, ["run", "answer", "--workspace", str(tmp_path), "--json"])

    assert result.exit_code == 1
    assert '"stop_reason":"completed"' not in result.stdout
    assert "simulated" not in result.output
    session_id = next((config.runtime.data_dir / "sessions").iterdir()).name
    reopened = AgentSession.load(data_dir=config.runtime.data_dir, session_id=session_id)
    reopened.close()


def test_headless_cli_persists_large_tool_output_for_artifact_retrieval(
    config: MiniAgentConfig, tmp_path: Path
) -> None:
    assert config.runtime.data_dir is not None
    (tmp_path / "notes.txt").write_text("long output\n" * 40, encoding="utf-8")
    provider = _ClosableProvider(
        [
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="read_notes",
                        name="read_file",
                        arguments_json=ReadFileArguments(path="notes.txt").model_dump_json(),
                    ),
                ),
            ),
            ModelResponse(content="done"),
        ]
    )
    with (
        patch("mini_agent.cli._load_runtime_config", return_value=config),
        patch("mini_agent.cli.OpenAICompatibleProvider", return_value=provider),
    ):
        result = CliRunner().invoke(
            app, ["run", "read notes", "--workspace", str(tmp_path), "--json"]
        )

    assert result.exit_code == 0, result.output
    session_id = next((config.runtime.data_dir / "sessions").iterdir()).name
    reopened = AgentSession.load(data_dir=config.runtime.data_dir, session_id=session_id)
    try:
        artifacts = [
            event.data for event in reopened.events if isinstance(event.data, ArtifactCreatedData)
        ]
        completions = [
            event.data for event in reopened.events if isinstance(event.data, ToolCompletedData)
        ]
        assert len(artifacts) == 1
        assert completions[0].artifact_id == artifacts[0].artifact_id
    finally:
        reopened.close()
