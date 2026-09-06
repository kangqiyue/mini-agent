import asyncio
import json
import os
from pathlib import Path

import pytest
from pydantic import HttpUrl

from mini_agent.agent import MiniAgent, ModelCallLimitExceeded
from mini_agent.artifacts import ArtifactStore
from mini_agent.checkpoint import Checkpoint, CheckpointStore
from mini_agent.checkpoint_writer import (
    CheckpointCoordinator,
    CheckpointRecoveryRequiredError,
    DeterministicCheckpointExtractor,
)
from mini_agent.config import (
    ContextConfig,
    MiniAgentConfig,
    ModelConfig,
    RuntimeConfig,
    SystemPromptConfig,
)
from mini_agent.context import ContextConfigurationError
from mini_agent.events import (
    ApprovalDecision,
    ApprovalRequestedData,
    ApprovalResolvedData,
    AssistantMessageData,
    ContextEstimatedData,
    GoalCreatedData,
    ModelRequestFailedData,
    ModelRequestStartedData,
    ToolCompletedData,
    ToolFailedData,
    ToolInterruptedData,
    ToolRecoveryStatus,
    ToolRequestedData,
    ToolStartedData,
    UserMessageData,
)
from mini_agent.history import HistoryPage, SessionHistory
from mini_agent.messages import (
    MAX_TOOL_CALLS_PER_RESPONSE,
    FinishReason,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolCall,
)
from mini_agent.permissions import ApprovalPrompt, ApprovalRequest, PermissionController
from mini_agent.provider import ProviderError
from mini_agent.redaction_types import RedactionKind
from mini_agent.session import AgentSession
from mini_agent.tools.apply_patch import ApplyPatchTool
from mini_agent.tools.base import (
    TOOL_CONTEXT_PREVIEW_TRUNCATED_NOTICE,
    ToolDefinition,
    ToolError,
    ToolResult,
)
from mini_agent.tools.history_search import HistorySearchTool
from mini_agent.tools.read_file import ReadFileTool
from mini_agent.tools.registry import ToolRegistry
from mini_agent.workspace import Workspace
from tests.support.providers import SequenceProvider


class CountingReadTool:
    def __init__(self) -> None:
        self.execute_count = 0

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="counting_read",
            description="Count executions.",
            parameters={"type": "object"},
            is_read_only=True,
        )

    def execute(self, arguments_json: str) -> ToolResult:
        self.execute_count += 1
        return ToolResult(content="executed")


class FixedResultTool:
    def __init__(self, result: ToolResult) -> None:
        self._result = result

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="fixed_result",
            description="Return one fixed result.",
            parameters={"type": "object"},
            is_read_only=True,
        )

    def execute(self, arguments_json: str) -> ToolResult:
        return self._result


class OversizedDefinitionTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="oversized_definition",
            description="x" * 5_000,
            parameters={"type": "object"},
            is_read_only=True,
        )

    def execute(self, arguments_json: str) -> ToolResult:
        del arguments_json
        return ToolResult(content="unused")


class CountingWriteTool(CountingReadTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="counting_write",
            description="Count write executions.",
            parameters={"type": "object"},
            is_read_only=False,
        )


class RecordingWriteTool(CountingWriteTool):
    def __init__(self) -> None:
        super().__init__()
        self.arguments: list[str] = []

    def execute(self, arguments_json: str) -> ToolResult:
        self.arguments.append(arguments_json)
        return super().execute(arguments_json)


class FailingWriteTool(CountingWriteTool):
    def __init__(self, code: str) -> None:
        super().__init__()
        self._code = code

    def execute(self, arguments_json: str) -> ToolResult:
        self.execute_count += 1
        raise ToolError(self._code, "injected failure")


class FixedApprovalPrompt(ApprovalPrompt):
    def __init__(self, decision: ApprovalDecision) -> None:
        self.decision = decision
        self.requests: list[ApprovalRequest] = []

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return self.decision


class _UnexpectedProviderFailure:
    """Provider double that violates the ProviderError-only contract."""

    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        raise self._error


def _config(tmp_path: Path, *, retry_count: int = 2) -> MiniAgentConfig:
    return MiniAgentConfig(
        model=ModelConfig(model="m", base_url=HttpUrl("https://example.test/v1")),
        runtime=RuntimeConfig(
            data_dir=tmp_path / "data",
            provider_retry_count=retry_count,
            retry_backoff_seconds=0,
        ),
    )


def _tool_calls(prefix: str, count: int) -> tuple[ToolCall, ...]:
    return tuple(
        ToolCall(id=f"{prefix}-{index}", name="counting_read", arguments_json="{}")
        for index in range(count)
    )


@pytest.mark.asyncio
async def test_run_turn_persists_boundaries_in_order(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider([ModelResponse(content="answer", finish_reason=FinishReason.STOP)])
    agent = MiniAgent(config=_config(tmp_path), provider=provider, session=session)

    response = await agent.run_turn(" question ")

    assert response.content == "answer"
    assert [type(event.data) for event in session.events[1:]] == [
        GoalCreatedData,
        UserMessageData,
        ContextEstimatedData,
        ModelRequestStartedData,
        AssistantMessageData,
    ]
    assert provider.requests[0].messages[0].role.value == "system"
    assert "Objective: question" in (provider.requests[0].messages[0].content or "")
    assert provider.requests[0].messages[1].content == "question"
    assert provider.requests[0].max_output_tokens == 16_384


@pytest.mark.asyncio
async def test_run_turn_rejects_model_constructed_response_with_too_many_tool_calls(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    unsafe_response = ModelResponse.model_construct(
        content=None,
        tool_calls=_tool_calls("too-many", MAX_TOOL_CALLS_PER_RESPONSE + 1),
        finish_reason=FinishReason.TOOL_CALLS,
    )
    provider = SequenceProvider([unsafe_response])
    tool = CountingReadTool()
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((tool,)),
    )

    with pytest.raises(ProviderError) as error_info:
        await agent.run_turn("read files")

    assert error_info.value.code == "invalid_provider_response"
    assert tool.execute_count == 0
    event_kinds = [event.data.kind for event in session.events]
    assert "model_request_failed" in event_kinds
    assert "assistant_message" not in event_kinds
    assert "tool_requested" not in event_kinds
    assert "tool_started" not in event_kinds
    assert "artifact_created" not in event_kinds
    session.close()


@pytest.mark.asyncio
async def test_run_turn_rejects_tool_call_fan_out_across_model_responses(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    responses = [
        ModelResponse(
            tool_calls=_tool_calls(f"batch-{batch}", MAX_TOOL_CALLS_PER_RESPONSE),
            finish_reason=FinishReason.TOOL_CALLS,
        )
        for batch in range(4)
    ]
    rejected_tool_call = _tool_calls("over-turn-limit", 1)[0]
    responses.append(
        ModelResponse(
            tool_calls=(rejected_tool_call,),
            finish_reason=FinishReason.TOOL_CALLS,
        )
    )
    provider = SequenceProvider(responses)
    tool = CountingReadTool()
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(update={"max_model_calls_per_turn": 5})
        }
    )
    agent = MiniAgent(
        config=config,
        provider=provider,
        session=session,
        tools=ToolRegistry((tool,)),
    )

    with pytest.raises(ProviderError) as error_info:
        await agent.run_turn("read files")

    assert error_info.value.code == "invalid_provider_response"
    assert tool.execute_count == 4 * MAX_TOOL_CALLS_PER_RESPONSE
    assert not any(
        event.correlation_id == rejected_tool_call.id for event in session.events
    )
    assert [event.data.kind for event in session.events].count("assistant_message") == 4
    assert [event.data.kind for event in session.events].count("model_request_failed") == 1
    session.close()


@pytest.mark.asyncio
async def test_run_turn_stops_before_provider_when_full_history_exceeds_budget(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider([])
    config = _config(tmp_path).model_copy(
        update={
            "model": _config(tmp_path).model.model_copy(
                update={"context_window": 4_096, "max_output_tokens": 1_024}
            ),
        }
    )
    agent = MiniAgent(config=config, provider=provider, session=session)

    with pytest.raises(ContextConfigurationError):
        await agent.run_turn("x" * 4_000)

    assert provider.requests == []


@pytest.mark.asyncio
async def test_run_turn_serializes_concurrent_calls(tmp_path: Path) -> None:
    # Without a serialization guard, two overlapping run_turn calls would
    # interleave at the provider await and clobber shared projection/cycle
    # state. A provider that records how many complete() calls are in flight at
    # once proves the MiniAgent lock serializes the agent-running entry points.

    class _ConcurrencyDetectingProvider:
        def __init__(self) -> None:
            self._in_flight = 0
            self.max_concurrency = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            self._in_flight += 1
            self.max_concurrency = max(self.max_concurrency, self._in_flight)
            try:
                await asyncio.sleep(0.02)
            finally:
                self._in_flight -= 1
            return ModelResponse(content="done", finish_reason=FinishReason.STOP)

    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = _ConcurrencyDetectingProvider()
    agent = MiniAgent(config=_config(tmp_path), provider=provider, session=session)

    await asyncio.gather(agent.run_turn("first"), agent.run_turn("second"))

    assert provider.max_concurrency == 1
    session.close()


@pytest.mark.asyncio
async def test_run_turn_persists_provider_usage_on_assistant_event(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider(
        [
            ModelResponse(
                content="counted",
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
            )
        ]
    )
    agent = MiniAgent(config=_config(tmp_path), provider=provider, session=session)

    await agent.run_turn("say counted")

    assistant_events = [
        event
        for event in session.events
        if isinstance(event.data, AssistantMessageData)
    ]
    assert len(assistant_events) == 1
    assert isinstance(assistant_events[0].data, AssistantMessageData)
    assert assistant_events[0].data.usage == TokenUsage(
        prompt_tokens=11, completion_tokens=7, total_tokens=18
    )
    session.close()


@pytest.mark.asyncio
async def test_run_turn_persists_events_without_usage_forever_valid(
    tmp_path: Path,
) -> None:
    # Responses from providers that report no usage must keep producing
    # assistant events (usage defaults to None), and old records without the
    # field must remain loadable.
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider(
        [ModelResponse(content="answer", finish_reason=FinishReason.STOP)]
    )
    agent = MiniAgent(config=_config(tmp_path), provider=provider, session=session)

    await agent.run_turn("hi")

    assistant_events = [
        event
        for event in session.events
        if isinstance(event.data, AssistantMessageData)
    ]
    assert len(assistant_events) == 1
    assert isinstance(assistant_events[0].data, AssistantMessageData)
    assert assistant_events[0].data.usage is None
    session.close()

    reopened = AgentSession.load(
        data_dir=tmp_path / "data", session_id=session.metadata.session_id
    )
    reloaded_assistant = next(
        event
        for event in reopened.events
        if isinstance(event.data, AssistantMessageData)
    )
    assert isinstance(reloaded_assistant.data, AssistantMessageData)
    assert reloaded_assistant.data.usage is None
    reopened.close()


def test_constructor_rejects_an_unsendable_base_request_before_checkpoint_side_effects(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider([])
    base = _config(tmp_path)
    config = base.model_copy(
        update={
            "model": base.model.model_copy(
                update={"context_window": 4_096, "max_output_tokens": 1_024}
            ),
            "context": ContextConfig(reserve_tokens=1_024, minimum_input_tokens=1_024),
        }
    )
    tools = ToolRegistry((OversizedDefinitionTool(),))
    events_before = session.events

    with pytest.raises(ContextConfigurationError):
        MiniAgent(config=config, provider=provider, session=session, tools=tools)

    assert session.events == events_before
    assert provider.requests == []
    assert not (session.paths.root / "checkpoints").exists()
    session.close()


@pytest.mark.asyncio
async def test_checkpoint_recovery_requirement_blocks_further_agent_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider([])
    checkpoints = CheckpointStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    coordinator = CheckpointCoordinator(
        session=session,
        store=checkpoints,
        extractor=DeterministicCheckpointExtractor(),
    )

    def fail_activation(_registration: object) -> Checkpoint:
        raise OSError("synthetic activation failure")

    monkeypatch.setattr(checkpoints, "activate", fail_activation)
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        checkpoints=checkpoints,
        checkpoint_coordinator=coordinator,
    )

    with pytest.raises(CheckpointRecoveryRequiredError):
        await agent.checkpoint()
    with pytest.raises(RuntimeError, match="must be resumed"):
        await agent.run_turn("continue")
    with pytest.raises(RuntimeError, match="must be resumed"):
        await agent.compact()
    assert provider.requests == []
    session.close()


@pytest.mark.asyncio
async def test_checkpoint_base_exception_after_start_requires_agent_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider([])
    checkpoints = CheckpointStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    extractor = DeterministicCheckpointExtractor()

    async def cancel_extract(**_kwargs: object) -> Checkpoint:
        raise asyncio.CancelledError()

    monkeypatch.setattr(extractor, "extract", cancel_extract)
    coordinator = CheckpointCoordinator(
        session=session,
        store=checkpoints,
        extractor=extractor,
    )
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        checkpoints=checkpoints,
        checkpoint_coordinator=coordinator,
    )

    with pytest.raises(asyncio.CancelledError):
        await agent.checkpoint()
    with pytest.raises(RuntimeError, match="must be resumed"):
        await agent.checkpoint()
    with pytest.raises(RuntimeError, match="must be resumed"):
        await agent.compact()
    with pytest.raises(RuntimeError, match="must be resumed"):
        await agent.run_turn("continue")
    assert provider.requests == []
    session.close()


@pytest.mark.asyncio
async def test_first_goal_is_validated_before_any_goal_or_user_event(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider([])
    base = _config(tmp_path)
    config = base.model_copy(
        update={
            "model": base.model.model_copy(
                update={
                    "context_window": 8_192,
                    "max_input_tokens": 1_800,
                    "max_output_tokens": 1_024,
                }
            ),
            "context": ContextConfig(reserve_tokens=1_024, minimum_input_tokens=1_024),
        }
    )
    agent = MiniAgent(config=config, provider=provider, session=session)
    events_before = session.events

    with pytest.raises(ContextConfigurationError):
        await agent.run_turn("x" * 4_000)

    assert session.events == events_before
    assert provider.requests == []
    session.close()


@pytest.mark.asyncio
async def test_first_turn_floor_includes_the_latest_user_before_writing_events(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider([])
    base = _config(tmp_path)
    config = base.model_copy(
        update={
            "model": base.model.model_copy(
                update={"context_window": 4_096, "max_output_tokens": 1_024}
            )
        }
    )
    agent = MiniAgent(config=config, provider=provider, session=session)
    events_before = session.events

    with pytest.raises(ContextConfigurationError):
        await agent.run_turn("x" * 128)

    assert session.events == events_before
    assert provider.requests == []
    session.close()


@pytest.mark.asyncio
async def test_run_turn_rechecks_live_instructions_before_persisting_input(
    tmp_path: Path,
) -> None:
    instructions = tmp_path / "AGENT_INSTRUCTIONS.md"
    instructions.write_text("short", encoding="utf-8")
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider([])
    base = _config(tmp_path)
    config = base.model_copy(
        update={
            "model": base.model.model_copy(
                update={
                    "context_window": 8_192,
                    "max_input_tokens": 4_096,
                    "max_output_tokens": 1_024,
                }
            ),
            "context": ContextConfig(reserve_tokens=1_024, minimum_input_tokens=1_024),
            "system_prompt": SystemPromptConfig(
                instructions_file=Path("AGENT_INSTRUCTIONS.md"),
                maximum_instructions_chars=12_000,
            ),
        }
    )
    agent = MiniAgent(config=config, provider=provider, session=session)
    instructions.write_text("x" * 10_000, encoding="utf-8")
    events_before = session.events

    with pytest.raises(ContextConfigurationError):
        await agent.run_turn("continue")

    assert session.events == events_before
    assert provider.requests == []
    session.close()


@pytest.mark.asyncio
async def test_provider_tool_arguments_are_safe_in_events_and_replay_but_raw_for_execution(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=workspace, model="m")
    raw_path = workspace / "target.txt"
    tool_call = ToolCall(
        id="call-path",
        name="counting_write",
        arguments_json=json.dumps({"path": str(raw_path)}),
    )
    provider = SequenceProvider(
        [
            ModelResponse(tool_calls=(tool_call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="done"),
        ]
    )
    tool = RecordingWriteTool()
    approval = FixedApprovalPrompt(ApprovalDecision.ALLOW_SESSION)
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((tool,)),
        permissions=PermissionController(prompt=approval),
    )

    await agent.run_turn("write it")

    assistant = next(
        event.data for event in session.events if isinstance(event.data, AssistantMessageData)
    )
    requested = next(
        event.data for event in session.events if isinstance(event.data, ToolRequestedData)
    )
    for payload in (
        assistant.tool_calls[0].arguments_json,
        requested.tool_call.arguments_json,
        provider.requests[1].model_dump_json(),
    ):
        if str(raw_path) in payload:
            pytest.fail("safe payload contains a host path")
    assert tool.arguments == [tool_call.arguments_json]
    assert approval.requests[0].can_allow_session is False
    assert session.session_grant_fingerprints == frozenset()
    session.close()


@pytest.mark.asyncio
async def test_run_turn_retries_only_retryable_provider_errors(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider(
        [
            ProviderError("busy", "temporary", is_retryable=True),
            ModelResponse(content="recovered"),
        ]
    )
    agent = MiniAgent(config=_config(tmp_path), provider=provider, session=session)

    response = await agent.run_turn("hello")

    assert response.content == "recovered"
    assert len(provider.requests) == 2
    event_types = [type(event.data) for event in session.events]
    assert event_types.count(ModelRequestStartedData) == 2
    assert event_types.count(ModelRequestFailedData) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    (
        RuntimeError("provider protocol violation"),
        asyncio.CancelledError(),
        KeyboardInterrupt(),
    ),
)
async def test_non_provider_failure_after_request_start_requires_resume(
    tmp_path: Path,
    error: BaseException,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = _UnexpectedProviderFailure(error)
    agent = MiniAgent(config=_config(tmp_path), provider=provider, session=session)

    with pytest.raises(type(error)):
        await agent.run_turn("hello")

    assert len(provider.requests) == 1
    assert [event.data.kind for event in session.events].count("model_request_started") == 1
    assert not any(isinstance(event.data, ModelRequestFailedData) for event in session.events)
    with pytest.raises(RuntimeError, match="must be resumed"):
        await agent.run_turn("another request")
    with pytest.raises(RuntimeError, match="must be resumed"):
        await agent.checkpoint()
    with pytest.raises(RuntimeError, match="must be resumed"):
        await agent.compact()

    session_id = session.metadata.session_id
    session.close()
    resumed = AgentSession.resume(data_dir=tmp_path / "data", session_id=session_id)
    terminal = resumed.events[-2].data
    assert isinstance(terminal, ModelRequestFailedData)
    assert terminal.error_code == "model_request_interrupted"
    assert terminal.attempt == 1

    continued_provider = SequenceProvider([ModelResponse(content="resumed")])
    resumed_agent = MiniAgent(
        config=_config(tmp_path),
        provider=continued_provider,
        session=resumed,
    )
    response = await resumed_agent.run_turn("continue")

    assert response.content == "resumed"
    assert len(continued_provider.requests) == 1
    resumed.close()


@pytest.mark.asyncio
async def test_assistant_append_failure_requires_resume_without_persisting_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider([ModelResponse(content="provider response")])
    agent = MiniAgent(config=_config(tmp_path), provider=provider, session=session)

    def fail_assistant_append(*_args: object, **_kwargs: object) -> object:
        raise OSError("assistant append fault")

    monkeypatch.setattr(session, "append_assistant_message", fail_assistant_append)

    with pytest.raises(OSError, match="assistant append fault"):
        await agent.run_turn("hello")

    assert len(provider.requests) == 1
    assert not any(isinstance(event.data, AssistantMessageData) for event in session.events)
    assert "provider response" not in session.paths.events.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError, match="must be resumed"):
        await agent.run_turn("another request")

    session_id = session.metadata.session_id
    session.close()
    resumed = AgentSession.resume(data_dir=tmp_path / "data", session_id=session_id)
    terminal = resumed.events[-2].data
    assert isinstance(terminal, ModelRequestFailedData)
    assert terminal.error_code == "model_request_interrupted"
    resumed.close()


@pytest.mark.asyncio
async def test_provider_retries_cannot_exceed_the_turn_model_call_budget(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider(
        [
            ProviderError("busy", "temporary", is_retryable=True),
            ModelResponse(content="must not be called"),
        ]
    )
    base_config = _config(tmp_path, retry_count=2)
    config = base_config.model_copy(
        update={
            "runtime": base_config.runtime.model_copy(
                update={"max_model_calls_per_turn": 1}
            )
        }
    )
    agent = MiniAgent(config=config, provider=provider, session=session)

    with pytest.raises(ProviderError, match="temporary"):
        await agent.run_turn("hello")

    assert len(provider.requests) == 1
    session.close()


@pytest.mark.asyncio
async def test_tool_completion_at_model_call_limit_leaves_a_loadable_session(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    tool = CountingReadTool()
    call = ToolCall(id="call-budget", name="counting_read", arguments_json="{}")
    provider = SequenceProvider(
        [ModelResponse(tool_calls=(call,), finish_reason=FinishReason.TOOL_CALLS)]
    )
    base_config = _config(tmp_path)
    config = base_config.model_copy(
        update={
            "runtime": base_config.runtime.model_copy(
                update={"max_model_calls_per_turn": 1}
            )
        }
    )
    agent = MiniAgent(
        config=config,
        provider=provider,
        session=session,
        tools=ToolRegistry((tool,)),
    )

    with pytest.raises(ModelCallLimitExceeded):
        await agent.run_turn("run the read")

    assert tool.execute_count == 1
    session_id = session.metadata.session_id
    session.close()
    loaded = AgentSession.load(data_dir=tmp_path / "data", session_id=session_id)
    assert loaded.events[-1].data.kind == "tool_completed"
    loaded.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finish_reason",
    (FinishReason.LENGTH, FinishReason.OTHER),
)
async def test_possibly_truncated_model_response_redacts_partial_secret_everywhere(
    tmp_path: Path,
    finish_reason: FinishReason,
) -> None:
    partial_secret = "sk_live_" + "0" * 3
    session = AgentSession.create(
        data_dir=tmp_path / "data",
        workspace=tmp_path,
        model="m",
    )
    provider = SequenceProvider(
        [
            ModelResponse(
                content=f"partial answer: {partial_secret}",
                finish_reason=finish_reason,
            )
        ]
    )
    agent = MiniAgent(config=_config(tmp_path), provider=provider, session=session)

    response = await agent.run_turn("answer within the limit")

    assistant_event = next(
        event
        for event in session.events
        if isinstance(event.data, AssistantMessageData)
    )
    assistant_data = assistant_event.data
    assert isinstance(assistant_data, AssistantMessageData)
    assert response.content == "partial answer: [REDACTED]"
    assert assistant_data.content == "partial answer: [REDACTED]"
    assert partial_secret not in session.paths.events.read_text(encoding="utf-8")
    assert assistant_event.redaction_summary.match_count == 1
    assert assistant_event.redaction_summary.kinds == (RedactionKind.SECRET_PREFIX,)
    session.close()


@pytest.mark.asyncio
async def test_run_turn_does_not_retry_permanent_provider_error(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider([ProviderError("bad_request", "invalid", is_retryable=False)])
    agent = MiniAgent(config=_config(tmp_path), provider=provider, session=session)

    with pytest.raises(ProviderError, match="invalid"):
        await agent.run_turn("hello")

    assert len(provider.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_code", "expected_code", "is_retryable"),
    (
        ("bad_request", "bad_request", False),
        ("sk_live_" + "0" * 24, "invalid_provider_error_code", True),
    ),
)
async def test_provider_error_is_redacted_before_programmatic_rethrow_and_persistence(
    tmp_path: Path,
    raw_code: str,
    expected_code: str,
    is_retryable: bool,
) -> None:
    secret = "synthetic-provider-secret"
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider(
        [ProviderError(raw_code, f"api_key={secret}", is_retryable=is_retryable)]
    )
    agent = MiniAgent(
        config=_config(tmp_path, retry_count=0),
        provider=provider,
        session=session,
    )

    with pytest.raises(ProviderError) as error_info:
        await agent.run_turn("hello")

    error = error_info.value
    assert error.code == expected_code
    assert error.is_retryable is is_retryable
    assert secret not in str(error)
    assert secret not in repr(error)
    assert all(secret not in str(value) for value in error.args)
    assert error.__cause__ is None
    assert error.__context__ is None
    failed = next(
        event.data for event in session.events if isinstance(event.data, ModelRequestFailedData)
    )
    assert failed.error_code == expected_code
    assert failed.is_retryable is is_retryable
    assert secret not in failed.message
    persisted = session.paths.events.read_text(encoding="utf-8")
    assert secret not in persisted
    assert raw_code not in persisted or raw_code == expected_code
    session.close()


@pytest.mark.asyncio
async def test_run_turn_rejects_empty_input_without_writing(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider([])
    agent = MiniAgent(config=_config(tmp_path), provider=provider, session=session)

    with pytest.raises(ValueError, match="empty"):
        await agent.run_turn("   ")

    assert len(session.events) == 1


@pytest.mark.asyncio
async def test_run_turn_persists_canonical_tool_calls(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    tool_call = ToolCall(id="call-1", name="read_file", arguments_json='{"path":"README.md"}')
    (tmp_path / "README.md").write_text("contents", encoding="utf-8")
    provider = SequenceProvider(
        [
            ModelResponse(
                tool_calls=(tool_call,),
                finish_reason=FinishReason.TOOL_CALLS,
            ),
            ModelResponse(content="done"),
        ]
    )
    registry = ToolRegistry((ReadFileTool(Workspace(tmp_path)),))
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=registry,
    )

    response = await agent.run_turn("read it")

    assert response.content == "done"
    assert len(provider.requests) == 2
    assert provider.requests[1].messages[-1].tool_call_id == tool_call.id
    assert "contents" in (provider.requests[1].messages[-1].content or "")
    ModelRequest(model="m", messages=session.conversation_messages())
    session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("has_prior_match", (False, True))
async def test_history_search_cannot_match_its_own_call(
    tmp_path: Path,
    has_prior_match: bool,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    query = "self-call-only-needle"
    if has_prior_match:
        session.append_user_message(
            f"Persisted earlier record: {query}",
            turn_id=session.new_turn_id(),
        )
    tool_call = ToolCall(
        id="history-search",
        name="history_search",
        arguments_json=json.dumps({"query": query}),
    )
    provider = SequenceProvider(
        [
            ModelResponse(tool_calls=(tool_call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="done"),
        ]
    )
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((HistorySearchTool(lambda: SessionHistory(session.events)),)),
    )

    await agent.run_turn("Search the session history.")

    result = provider.requests[1].messages[-1]
    assert result.content is not None
    page = HistoryPage.model_validate_json(result.content)
    assistant_event = next(
        event
        for event in session.events
        if isinstance(event.data, AssistantMessageData)
        and event.data.tool_calls
        and event.data.tool_calls[0].id == tool_call.id
    )
    assert all(match.event_id < assistant_event.id for match in page.matches)
    assert [match.event_id for match in page.matches] == ([2] if has_prior_match else [])
    session.close()


@pytest.mark.asyncio
async def test_tool_is_not_executed_when_started_event_cannot_persist(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    tool_call = ToolCall(id="call-1", name="counting_read", arguments_json="{}")
    provider = SequenceProvider(
        [
            ModelResponse(tool_calls=(tool_call,), finish_reason=FinishReason.TOOL_CALLS),
        ]
    )
    tool = CountingReadTool()
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((tool,)),
    )
    original_append_started = session.append_tool_started

    def fail_before_tool_start(tool_call: ToolCall, *, turn_id: str) -> object:
        raise OSError("event store write failed")

    session.append_tool_started = fail_before_tool_start  # type: ignore[method-assign]

    with pytest.raises(OSError, match="event store"):
        await agent.run_turn("run it")

    assert tool.execute_count == 0
    session.append_tool_started = original_append_started  # type: ignore[method-assign]
    session.close()


@pytest.mark.asyncio
async def test_large_tool_result_is_externalized_before_completion(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    (tmp_path / "large.txt").write_text("x" * 100, encoding="utf-8")
    tool_call = ToolCall(
        id="call-large",
        name="read_file",
        arguments_json='{"path":"large.txt"}',
    )
    provider = SequenceProvider(
        [
            ModelResponse(tool_calls=(tool_call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="done"),
        ]
    )
    base_config = _config(tmp_path)
    config = base_config.model_copy(
        update={
            "runtime": base_config.runtime.model_copy(update={"inline_tool_result_max_chars": 20})
        }
    )
    agent = MiniAgent(
        config=config,
        provider=provider,
        session=session,
        tools=ToolRegistry((ReadFileTool(Workspace(tmp_path), max_chars=1_000),)),
        artifacts=ArtifactStore.open(
            session.paths.root,
            registrations=(),
            workspace_root=Path(session.metadata.workspace),
        ),
    )

    response = await agent.run_turn("read the large file")

    assert response.content == "done"
    event_kinds = [event.data.kind for event in session.events]
    assert event_kinds.index("tool_started") < event_kinds.index("artifact_created")
    assert event_kinds.index("artifact_created") < event_kinds.index("tool_completed")
    artifact = next(event.data for event in session.events if event.data.kind == "artifact_created")
    completed = next(event.data for event in session.events if event.data.kind == "tool_completed")
    assert completed.artifact_id == artifact.artifact_id
    assert artifact.artifact_id in completed.output
    tool_result = provider.requests[1].messages[-1].content or ""
    assert "Artifact " in tool_result
    assert "captured result" in tool_result
    session.close()


@pytest.mark.asyncio
async def test_large_result_without_artifact_marks_context_preview_omission(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    tool_call = ToolCall(id="call-preview", name="fixed_result", arguments_json="{}")
    provider = SequenceProvider(
        [
            ModelResponse(tool_calls=(tool_call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="done"),
        ]
    )
    base_config = _config(tmp_path)
    config = base_config.model_copy(
        update={
            "runtime": base_config.runtime.model_copy(update={"inline_tool_result_max_chars": 20})
        }
    )
    full_content = "x" * 20 + "TAIL-OMITTED-FROM-CONTEXT"
    agent = MiniAgent(
        config=config,
        provider=provider,
        session=session,
        tools=ToolRegistry((FixedResultTool(ToolResult(content=full_content)),)),
    )

    await agent.run_turn("get a large result")

    tool_output = provider.requests[1].messages[-1].content or ""
    assert tool_output.startswith("x" * 20)
    assert "TAIL-OMITTED-FROM-CONTEXT" not in tool_output
    assert TOOL_CONTEXT_PREVIEW_TRUNCATED_NOTICE in tool_output
    assert "hard limit" not in tool_output
    session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("result_path", ["inline", "preview", "artifact"])
async def test_truncated_tool_result_is_explicit_in_every_context_path(
    tmp_path: Path,
    result_path: str,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    partial_secret = "sk_live_" + "0" * 3
    content = (
        f"bounded {partial_secret}"
        if result_path == "inline"
        else f"{'x' * 100} {partial_secret}"
    )
    tool_call = ToolCall(id="call-truncated", name="fixed_result", arguments_json="{}")
    provider = SequenceProvider(
        [
            ModelResponse(tool_calls=(tool_call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="done"),
        ]
    )
    base_config = _config(tmp_path)
    config = base_config.model_copy(
        update={
            "runtime": base_config.runtime.model_copy(update={"inline_tool_result_max_chars": 20})
        }
    )
    artifacts = (
        ArtifactStore.open(
            session.paths.root,
            registrations=(),
            workspace_root=Path(session.metadata.workspace),
        )
        if result_path == "artifact"
        else None
    )
    agent = MiniAgent(
        config=config,
        provider=provider,
        session=session,
        tools=ToolRegistry((FixedResultTool(ToolResult(content=content, is_truncated=True)),)),
        artifacts=artifacts,
    )

    await agent.run_turn("get bounded output")

    tool_output = provider.requests[1].messages[-1].content or ""
    completed = next(
        event.data
        for event in session.events
        if isinstance(event.data, ToolCompletedData)
    )
    assert partial_secret not in tool_output
    assert partial_secret not in session.paths.events.read_text(encoding="utf-8")
    assert "[tool output was truncated at hard limit]" in tool_output
    if result_path == "inline":
        assert "bounded [REDACTED]" in tool_output
        assert completed.source_redaction_summary.match_count == 1
    elif result_path == "preview":
        assert tool_output.startswith("x" * 20)
        assert "x" * 21 not in tool_output
        assert completed.source_redaction_summary.match_count == 1
    else:
        assert tool_output.startswith("Artifact ")
        assert artifacts is not None
        record = artifacts.records[0]
        artifact_content = artifacts.read(
            record.artifact_id,
            offset=0,
            limit=record.char_count,
        )
        assert partial_secret not in artifact_content
        assert artifact_content.endswith("[REDACTED]")
        assert record.redaction_match_count == 1
        assert completed.source_redaction_summary.match_count == 0
    session.close()


@pytest.mark.asyncio
async def test_default_inline_limit_preserves_large_result_tail_in_artifact(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    tail_marker = "TAIL-BEYOND-INLINE-LIMIT"
    (tmp_path / "large.txt").write_text(
        "x" * 13_000 + tail_marker,
        encoding="utf-8",
    )
    call = ToolCall(
        id="call-large-default",
        name="read_file",
        arguments_json='{"path":"large.txt"}',
    )
    provider = SequenceProvider(
        [
            ModelResponse(tool_calls=(call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="done"),
        ]
    )
    artifacts = ArtifactStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((ReadFileTool(Workspace(tmp_path)),)),
        artifacts=artifacts,
    )

    await agent.run_turn("read it")

    assert len(artifacts.records) == 1
    record = artifacts.records[0]
    tail = artifacts.read(
        record.artifact_id,
        offset=max(0, record.char_count - 100),
        limit=100,
    )
    assert tail_marker in tail
    session.close()


@pytest.mark.asyncio
async def test_truncated_read_partial_secret_is_redacted_before_artifact_persistence(
    tmp_path: Path,
) -> None:
    maximum_chars = 80
    partial_secret = "sk_live_" + "0" * 3
    full_secret = "sk_live_" + "0" * 24
    line_prefix_chars = len(f"{1:>6} | ")
    filler = "x" * (maximum_chars - line_prefix_chars - len(partial_secret) - 1)
    (tmp_path / "bounded.txt").write_text(
        f"{filler} {full_secret}",
        encoding="utf-8",
    )
    call = ToolCall(
        id="call-bounded-secret",
        name="read_file",
        arguments_json='{"path":"bounded.txt"}',
    )
    provider = SequenceProvider(
        [
            ModelResponse(tool_calls=(call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="done"),
        ]
    )
    base_config = _config(tmp_path)
    config = base_config.model_copy(
        update={
            "runtime": base_config.runtime.model_copy(
                update={"inline_tool_result_max_chars": 20}
            )
        }
    )
    session = AgentSession.create(
        data_dir=tmp_path / "data",
        workspace=tmp_path,
        model="m",
    )
    artifacts = ArtifactStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    agent = MiniAgent(
        config=config,
        provider=provider,
        session=session,
        tools=ToolRegistry(
            (ReadFileTool(Workspace(tmp_path), max_chars=maximum_chars),)
        ),
        artifacts=artifacts,
    )

    await agent.run_turn("read the bounded file")

    assert len(artifacts.records) == 1
    record = artifacts.records[0]
    artifact_content = artifacts.read(record.artifact_id, offset=0, limit=maximum_chars)
    completed = next(
        event.data
        for event in session.events
        if isinstance(event.data, ToolCompletedData)
    )
    assert partial_secret not in artifact_content
    assert partial_secret not in session.paths.events.read_text(encoding="utf-8")
    assert partial_secret not in (provider.requests[1].messages[-1].content or "")
    assert "[REDACTED]" in artifact_content
    assert record.redaction_match_count == 1
    assert record.redaction_kinds == (RedactionKind.SECRET_PREFIX,)
    assert completed.source_redaction_summary.match_count == 0
    session.close()


@pytest.mark.asyncio
async def test_write_tool_denial_is_persisted_without_starting_or_executing(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    tool_call = ToolCall(id="call-write", name="counting_write", arguments_json="{}")
    provider = SequenceProvider(
        [
            ModelResponse(tool_calls=(tool_call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="understood"),
        ]
    )
    tool = CountingWriteTool()
    prompt = FixedApprovalPrompt(ApprovalDecision.DENY)
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((tool,)),
        permissions=PermissionController(prompt=prompt),
    )

    response = await agent.run_turn("do not run it")

    assert response.content == "understood"
    assert tool.execute_count == 0
    event_data = [event.data for event in session.events]
    assert any(isinstance(data, ApprovalRequestedData) for data in event_data)
    assert any(isinstance(data, ApprovalResolvedData) for data in event_data)
    assert not any(isinstance(data, ToolStartedData) for data in event_data)
    failure = next(data for data in event_data if isinstance(data, ToolFailedData))
    assert failure.error_code == "permission_denied"
    assert failure.before_start is True
    assert "permission_denied" in (provider.requests[1].messages[-1].content or "")
    session.close()


@pytest.mark.asyncio
async def test_write_tool_starts_only_after_allow_decision_is_persisted(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    tool_call = ToolCall(id="call-write", name="counting_write", arguments_json="{}")
    provider = SequenceProvider(
        [
            ModelResponse(tool_calls=(tool_call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="done"),
        ]
    )
    tool = CountingWriteTool()
    prompt = FixedApprovalPrompt(ApprovalDecision.ALLOW_SESSION)
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((tool,)),
        permissions=PermissionController(prompt=prompt),
    )

    await agent.run_turn("run it")

    assert tool.execute_count == 1
    kinds = [event.data.kind for event in session.events]
    assert kinds.index("approval_requested") < kinds.index("approval_resolved")
    assert kinds.index("approval_resolved") < kinds.index("tool_started")
    assert session.session_grant_fingerprints == frozenset()
    session.close()


@pytest.mark.asyncio
async def test_uncommitted_artifact_is_hidden_after_registration_error(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    (tmp_path / "large.txt").write_text("x" * 100, encoding="utf-8")
    call = ToolCall(
        id="call-artifact-fail",
        name="read_file",
        arguments_json='{"path":"large.txt"}',
    )
    provider = SequenceProvider(
        [ModelResponse(tool_calls=(call,), finish_reason=FinishReason.TOOL_CALLS)]
    )
    base_config = _config(tmp_path)
    config = base_config.model_copy(
        update={
            "runtime": base_config.runtime.model_copy(update={"inline_tool_result_max_chars": 20})
        }
    )
    artifacts = ArtifactStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    agent = MiniAgent(
        config=config,
        provider=provider,
        session=session,
        tools=ToolRegistry((ReadFileTool(Workspace(tmp_path)),)),
        artifacts=artifacts,
    )

    def fail_registration(*_args: object, **_kwargs: object) -> object:
        raise OSError("artifact event failed")

    session.append_artifact_created = fail_registration  # type: ignore[method-assign]

    with pytest.raises(OSError, match="artifact event failed"):
        await agent.run_turn("read it")

    assert len(artifacts.records) == 1
    retained_files = list((session.paths.root / "artifacts").glob("*.txt"))
    assert len(retained_files) == 1
    retained_content = retained_files[0].read_text(encoding="utf-8")
    assert "x" * 100 in retained_content
    reopened = ArtifactStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    assert reopened.records == ()
    session.close()


@pytest.mark.asyncio
async def test_command_timeout_records_unknown_and_stops_the_turn(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    call = ToolCall(id="call-timeout", name="counting_write", arguments_json="{}")
    provider = SequenceProvider(
        [ModelResponse(tool_calls=(call,), finish_reason=FinishReason.TOOL_CALLS)]
    )
    tool = FailingWriteTool("command_timeout")
    prompt = FixedApprovalPrompt(ApprovalDecision.ALLOW_SESSION)
    permissions = PermissionController(prompt=prompt)
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((tool,)),
        permissions=permissions,
    )

    with pytest.raises(RuntimeError, match="unknown") as error_info:
        await agent.run_turn("run it")

    assert error_info.value.__cause__ is None
    assert error_info.value.__suppress_context__ is True

    interrupted = next(
        event.data for event in session.events if isinstance(event.data, ToolInterruptedData)
    )
    assert interrupted.recovery_status is ToolRecoveryStatus.UNKNOWN
    assert permissions.session_grant_fingerprints == frozenset()
    assert len(provider.requests) == 1
    session.close()


@pytest.mark.asyncio
async def test_tool_completion_write_failure_stops_before_second_side_effect(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    first_call = ToolCall(id="call-first", name="counting_write", arguments_json="{}")
    second_call = ToolCall(id="call-second", name="second_write", arguments_json="{}")
    provider = SequenceProvider(
        [
            ModelResponse(
                tool_calls=(first_call, second_call),
                finish_reason=FinishReason.TOOL_CALLS,
            )
        ]
    )
    first_tool = CountingWriteTool()

    class SecondWriteTool(CountingWriteTool):
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(
                name="second_write",
                description="A second side effect.",
                parameters={"type": "object"},
                is_read_only=False,
            )

    second_tool = SecondWriteTool()
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((first_tool, second_tool)),
        permissions=PermissionController(prompt=FixedApprovalPrompt(ApprovalDecision.ALLOW_ONCE)),
    )

    def fail_completion(*_args: object, **_kwargs: object) -> object:
        raise OSError("completion event failed")

    session.append_tool_completed = fail_completion  # type: ignore[method-assign]

    with pytest.raises(OSError, match="completion event failed"):
        await agent.run_turn("run both")

    assert first_tool.execute_count == 1
    assert second_tool.execute_count == 0
    assert not any(event.correlation_id == second_call.id for event in session.events)
    session.close()


@pytest.mark.asyncio
async def test_apply_patch_unknown_replace_status_stops_before_second_write_and_requires_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    first_call = ToolCall(
        id="call-first",
        name="apply_patch",
        arguments_json=json.dumps(
            {
                "changes": [
                    {
                        "path": target.name,
                        "expected_content": "before",
                        "replacement_content": "after",
                    }
                ]
            }
        ),
    )
    second_call = ToolCall(id="call-second", name="second_write", arguments_json="{}")
    provider = SequenceProvider(
        [
            ModelResponse(
                tool_calls=(first_call, second_call),
                finish_reason=FinishReason.TOOL_CALLS,
            )
        ]
    )

    class SecondWriteTool(CountingWriteTool):
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(
                name="second_write",
                description="A second side effect.",
                parameters={"type": "object"},
                is_read_only=False,
            )

    second_tool = SecondWriteTool()
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((ApplyPatchTool(Workspace(tmp_path)), second_tool)),
        permissions=PermissionController(prompt=FixedApprovalPrompt(ApprovalDecision.ALLOW_ONCE)),
    )
    original_replace = os.replace

    def replace_then_raise(
        source: str | bytes | Path,
        destination: str | bytes | Path,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        raise OSError("simulated post-replace failure")

    monkeypatch.setattr("mini_agent.tools.apply_patch.os.replace", replace_then_raise)

    with pytest.raises(RuntimeError, match="status is unknown"):
        await agent.run_turn("apply both changes")

    assert target.read_text(encoding="utf-8") == "after"
    interrupted = next(
        event.data for event in session.events if isinstance(event.data, ToolInterruptedData)
    )
    assert interrupted.tool_call_id == first_call.id
    assert interrupted.recovery_status is ToolRecoveryStatus.UNKNOWN
    assert second_tool.execute_count == 0
    assert not any(event.correlation_id == second_call.id for event in session.events)

    with pytest.raises(RuntimeError, match="must be resumed"):
        await agent.run_turn("try another change")

    assert len(provider.requests) == 1
    session.close()
