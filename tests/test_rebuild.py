"""Agent-level checkpoint, rebuild, and resume projection tests."""

import asyncio
from pathlib import Path

import pytest
from pydantic import HttpUrl

from mini_agent.agent import MiniAgent
from mini_agent.config import ContextConfig, MiniAgentConfig, ModelConfig, RuntimeConfig
from mini_agent.event_store import EventStoreCorruptionError
from mini_agent.events import (
    CheckpointCommittedData,
    RebuildCompletedData,
    RebuildFailedData,
    RebuildStartedData,
)
from mini_agent.history import SessionHistory
from mini_agent.messages import FinishReason, ModelRequest, ModelResponse, ToolCall
from mini_agent.provider import ProviderError
from mini_agent.session import AgentSession
from mini_agent.session_recovery import validate_session_events
from mini_agent.tools.history_search import HistorySearchTool
from mini_agent.tools.registry import ToolRegistry


class RecordingProvider:
    def __init__(self, responses: list[ModelResponse | ProviderError]) -> None:
        self.responses = iter(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        result = next(self.responses)
        if isinstance(result, ProviderError):
            raise result
        return result


def test_session_validator_rejects_a_first_event_outside_cycle_one(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    tampered_start = session.events[0].model_copy(update={"cycle_id": 2})

    with pytest.raises(EventStoreCorruptionError, match="completed rebuild"):
        validate_session_events((tampered_start,), session.metadata.session_id)

    session.close()


def test_session_validator_rejects_an_ordinary_event_advancing_cycle(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    session.append_user_message("ordinary event", turn_id=session.new_turn_id())
    tampered_event = session.events[-1].model_copy(update={"cycle_id": 2})

    with pytest.raises(EventStoreCorruptionError, match="completed rebuild"):
        validate_session_events(
            (*session.events[:-1], tampered_event),
            session.metadata.session_id,
        )

    session.close()


def _config(tmp_path: Path) -> MiniAgentConfig:
    return MiniAgentConfig(
        model=ModelConfig(
            model="m",
            base_url=HttpUrl("https://example.test/v1"),
            context_window=8_192,
            max_output_tokens=1_024,
        ),
        context=ContextConfig(
            checkpoint_milestones=(0.60, 0.75, 0.90),
            rebuild_ratio=0.95,
            reserve_tokens=1_024,
            rebuild_seed_max_tokens=3_000,
            checkpoint_max_tokens=2_048,
            minimum_input_tokens=1_024,
        ),
        runtime=RuntimeConfig(data_dir=tmp_path / "data", provider_retry_count=0),
    )


@pytest.mark.asyncio
async def test_agent_rebuilds_without_rewriting_the_transcript(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = RecordingProvider(
        [ModelResponse(content="first answer"), ModelResponse(content="second answer")]
    )
    agent = MiniAgent(config=_config(tmp_path), provider=provider, session=session)

    await agent.run_turn("old context " * 300)
    durable_before = session.conversation_messages()
    await agent.compact(focus="keep the latest task state")
    response = await agent.run_turn("latest request " * 200)

    assert response.content == "second answer"
    assert len(session.conversation_messages()) > len(durable_before)
    assert provider.requests[-1].messages[0].role.value == "system"
    projected_text = "\n".join(
        message.content or ""
        for message in provider.requests[-1].messages
        if message.role.value != "system"
    )
    durable_text = "\n".join(
        message.content or "" for message in session.conversation_messages()
    )
    assert len(projected_text) < len(durable_text)
    assert any(isinstance(event.data, CheckpointCommittedData) for event in session.events)
    assert any(isinstance(event.data, RebuildCompletedData) for event in session.events)
    rebuild = next(
        event.data for event in session.events if isinstance(event.data, RebuildCompletedData)
    )
    assert rebuild.projected_message_count <= rebuild.source_message_count + 1
    assert session.current_cycle_id >= 2
    session.close()


@pytest.mark.asyncio
async def test_resume_restores_rebuilt_projection_instead_of_full_history(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    first_provider = RecordingProvider([ModelResponse(content="first")])
    first_agent = MiniAgent(config=_config(tmp_path), provider=first_provider, session=session)
    await first_agent.run_turn("large prior context " * 350)
    await first_agent.compact(focus="continue after resume")
    session_id = session.metadata.session_id
    session.close()

    resumed = AgentSession.resume(data_dir=tmp_path / "data", session_id=session_id)
    second_provider = RecordingProvider([ModelResponse(content="resumed")])
    second_agent = MiniAgent(config=_config(tmp_path), provider=second_provider, session=resumed)

    response = await second_agent.run_turn("small follow up")

    assert response.content == "resumed"
    assert second_provider.requests[0].messages[0].role.value == "system"
    assert "continue after resume" in (second_provider.requests[0].messages[0].content or "")
    resumed.close()


@pytest.mark.asyncio
async def test_context_status_includes_messages_added_after_rebuild(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = RecordingProvider([ModelResponse(content="first")])
    agent = MiniAgent(config=_config(tmp_path), provider=provider, session=session)
    await agent.run_turn("initial request")
    await agent.compact(focus="continue from the checkpoint")
    before = agent.context_status()

    session.append_user_message("new detail after rebuild", turn_id=session.new_turn_id())
    after = agent.context_status()

    assert len(after.messages) == len(before.messages) + 1
    assert after.estimate.total_tokens > before.estimate.total_tokens
    session.close()


@pytest.mark.asyncio
async def test_provider_overflow_triggers_one_deterministic_rebuild(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = RecordingProvider(
        [
            ProviderError(
                "context_overflow",
                "request context is too large",
                is_retryable=False,
                is_context_overflow=True,
            ),
            ModelResponse(content="recovered"),
        ]
    )
    agent = MiniAgent(config=_config(tmp_path), provider=provider, session=session)

    response = await agent.run_turn("ordinary request")

    assert response.content == "recovered"
    assert len(provider.requests) == 2
    assert provider.requests[1].messages[0].role.value == "system"
    assert session.current_cycle_id == 2
    session.close()


@pytest.mark.asyncio
async def test_provider_overflow_does_not_rebuild_without_a_recovery_call_budget(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = RecordingProvider(
        [
            ProviderError(
                "context_overflow",
                "request context is too large",
                is_retryable=False,
                is_context_overflow=True,
            )
        ]
    )
    base_config = _config(tmp_path)
    config = base_config.model_copy(
        update={
            "runtime": base_config.runtime.model_copy(
                update={"max_model_calls_per_turn": 1}
            )
        }
    )
    agent = MiniAgent(config=config, provider=provider, session=session)

    with pytest.raises(ProviderError) as error_info:
        await agent.run_turn("ordinary request")

    assert error_info.value.is_context_overflow is True
    assert len(provider.requests) == 1
    assert session.current_cycle_id == 1
    assert not any(event.data.kind == "rebuild_started" for event in session.events)
    session.close()


@pytest.mark.asyncio
async def test_fixed_long_trace_continues_across_three_rebuild_cycles(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = RecordingProvider(
        [
            ModelResponse(content="one"),
            ModelResponse(content="two"),
            ModelResponse(content="three"),
            ModelResponse(content="four"),
        ]
    )
    agent = MiniAgent(config=_config(tmp_path), provider=provider, session=session)

    answers: list[str | None] = []
    for index in range(4):
        response = await agent.run_turn(f"request-{index} " + "detail " * 250)
        answers.append(response.content)
        if index < 3:
            await agent.compact(focus=f"cycle-{index + 2}")

    assert answers == ["one", "two", "three", "four"]
    assert session.current_cycle_id >= 4
    completed = [
        event.data for event in session.events if isinstance(event.data, RebuildCompletedData)
    ]
    assert len(completed) >= 3
    completed_cycles = [
        event.cycle_id
        for event in session.events
        if isinstance(event.data, RebuildCompletedData)
    ]
    assert completed_cycles[:3] == [2, 3, 4]
    assert all(request.messages[0].role.value == "system" for request in provider.requests[1:])
    session.close()


@pytest.mark.asyncio
async def test_stop_and_resume_preserve_the_current_rebuild_cycle(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = RecordingProvider(
        [ModelResponse(content="first"), ModelResponse(content="second")]
    )
    agent = MiniAgent(config=_config(tmp_path), provider=provider, session=session)

    await agent.run_turn("first request")
    await agent.compact(focus="cycle two")
    await agent.run_turn("second request")
    await agent.compact(focus="cycle three")

    assert session.current_cycle_id == 3
    stopped = session.stop()
    assert stopped.cycle_id == 3
    session_id = session.metadata.session_id
    session.close()

    resumed = AgentSession.resume(data_dir=tmp_path / "data", session_id=session_id)
    assert resumed.events[-1].data.kind == "session_resumed"
    assert resumed.events[-1].cycle_id == 3
    resumed.close()

    loaded = AgentSession.load(data_dir=tmp_path / "data", session_id=session_id)
    assert loaded.current_cycle_id == 3
    loaded.close()


@pytest.mark.asyncio
async def test_resume_closes_an_interrupted_rebuild_in_the_current_cycle(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = RecordingProvider([ModelResponse(content="first")])
    agent = MiniAgent(config=_config(tmp_path), provider=provider, session=session)
    await agent.run_turn("first request")
    await agent.compact(focus="cycle two")
    source_message_count = len(session.conversation_messages())
    session.append_rebuild_started(
        rebuild_id="f" * 32,
        source_cycle_id=2,
        checkpoint_id=None,
        checkpoint_watermark=0,
        reason="injected interruption",
        source_message_count=source_message_count,
        turn_id=session.new_turn_id(),
    )
    session_id = session.metadata.session_id
    session.close()

    resumed = AgentSession.resume(data_dir=tmp_path / "data", session_id=session_id)

    failed = next(
        event
        for event in resumed.events
        if isinstance(event.data, RebuildFailedData)
        and event.data.rebuild_id == "f" * 32
    )
    failed_data = failed.data
    assert isinstance(failed_data, RebuildFailedData)
    assert failed_data.error_code == "rebuild_interrupted"
    assert failed.cycle_id == 2
    assert resumed.events[-1].data.kind == "session_resumed"
    assert resumed.events[-1].cycle_id == 2
    resumed.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    (asyncio.CancelledError(), KeyboardInterrupt()),
)
async def test_rebuild_computation_interruption_requires_resume_after_stable_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    agent = MiniAgent(config=_config(tmp_path), provider=RecordingProvider([]), session=session)

    def interrupt_rebuild(*_args: object, **_kwargs: object) -> object:
        raise error

    monkeypatch.setattr("mini_agent.agent.build_rebuild_projection", interrupt_rebuild)

    with pytest.raises(type(error)):
        await agent.compact(focus="interrupt rebuild")

    failed = next(
        event.data for event in session.events if isinstance(event.data, RebuildFailedData)
    )
    assert failed.error_code == "rebuild_failed"
    with pytest.raises(RuntimeError, match="must be resumed"):
        await agent.run_turn("another request")
    with pytest.raises(RuntimeError, match="must be resumed"):
        await agent.checkpoint()
    with pytest.raises(RuntimeError, match="must be resumed"):
        await agent.compact()

    session_id = session.metadata.session_id
    session.close()
    resumed = AgentSession.resume(data_dir=tmp_path / "data", session_id=session_id)
    assert not any(
        isinstance(event.data, RebuildFailedData)
        and event.data.error_code == "rebuild_interrupted"
        for event in resumed.events
    )
    resumed.close()


@pytest.mark.asyncio
async def test_rebuild_completion_append_failure_requires_resume_and_recovery_closes_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    agent = MiniAgent(config=_config(tmp_path), provider=RecordingProvider([]), session=session)

    def fail_rebuild_completion(*_args: object, **_kwargs: object) -> object:
        raise OSError("rebuild completion append fault")

    monkeypatch.setattr(session, "append_rebuild_completed", fail_rebuild_completion)

    with pytest.raises(OSError, match="rebuild completion append fault"):
        await agent.compact(focus="fault injection")

    started = next(event for event in session.events if isinstance(event.data, RebuildStartedData))
    started_data = started.data
    assert isinstance(started_data, RebuildStartedData)
    assert not any(isinstance(event.data, RebuildCompletedData) for event in session.events)
    with pytest.raises(RuntimeError, match="must be resumed"):
        await agent.run_turn("another request")
    with pytest.raises(RuntimeError, match="must be resumed"):
        await agent.checkpoint()
    with pytest.raises(RuntimeError, match="must be resumed"):
        await agent.compact()

    session_id = session.metadata.session_id
    session.close()
    resumed = AgentSession.resume(data_dir=tmp_path / "data", session_id=session_id)
    recovered = next(
        event
        for event in resumed.events
        if isinstance(event.data, RebuildFailedData)
        and event.data.rebuild_id == started_data.rebuild_id
    )
    recovered_data = recovered.data
    assert isinstance(recovered_data, RebuildFailedData)
    assert recovered_data.error_code == "rebuild_interrupted"

    resumed_agent = MiniAgent(
        config=_config(tmp_path),
        provider=RecordingProvider([]),
        session=resumed,
    )
    await resumed_agent.compact(focus="after recovery")
    rebuild_starts = [
        event.data.rebuild_id
        for event in resumed.events
        if isinstance(event.data, RebuildStartedData)
    ]
    rebuild_terminals = {
        event.data.rebuild_id
        for event in resumed.events
        if isinstance(event.data, RebuildCompletedData | RebuildFailedData)
    }
    assert set(rebuild_starts) == rebuild_terminals
    resumed.close()


@pytest.mark.asyncio
async def test_rebuilt_agent_recovers_omitted_detail_through_history_search(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    exact_detail = "rare-regression-code-7319"
    search_call = ToolCall(
        id="recover-detail",
        name="history_search",
        arguments_json='{"query":"rare-regression-code-7319"}',
    )
    provider = RecordingProvider(
        [
            ModelResponse(content="noted"),
            ModelResponse(tool_calls=(search_call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content=f"Recovered {exact_detail}"),
        ]
    )
    tools = ToolRegistry((HistorySearchTool(lambda: SessionHistory(session.events)),))
    agent = MiniAgent(config=_config(tmp_path), provider=provider, session=session, tools=tools)

    await agent.run_turn(f"Remember this exact detail: {exact_detail}")
    await agent.compact(focus="continue without copying every exact detail")
    response = await agent.run_turn("Find the exact regression code from history.")

    assert response.content == f"Recovered {exact_detail}"
    search_result = provider.requests[-1].messages[-1]
    assert search_result.tool_call_id == search_call.id
    assert exact_detail in (search_result.content or "")
    session.close()
