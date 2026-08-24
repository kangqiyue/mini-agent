"""Restricted model checkpoint extraction and deterministic fallback tests."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mini_agent.agent import MiniAgent
from mini_agent.checkpoint import Checkpoint
from mini_agent.checkpoint_model import CheckpointExtractionError, ModelCheckpointExtractor
from mini_agent.config import ContextConfig, MiniAgentConfig, ModelConfig, RuntimeConfig
from mini_agent.context import Utf8TokenEstimator
from mini_agent.messages import ModelResponse
from mini_agent.provider import ProviderError
from mini_agent.session import AgentSession
from tests.support.providers import SequenceProvider


def _draft(event_id: int) -> str:
    empty_fields: dict[str, list[object]] = {
        name: []
        for name in (
            "acceptance_criteria",
            "constraints_and_preferences",
            "task_tree",
            "completed",
            "active_work",
            "blocked",
            "next_actions",
            "relevant_files",
            "cross_task_findings",
            "errors_and_fixes",
            "runtime_state",
            "key_decisions",
            "artifact_references",
            "miscellaneous_notes",
        )
    }
    return json.dumps(
        {
            "current_intent": "Implement bounded context",
            **empty_fields,
            "next_actions": [{"text": "Run tests", "source_event_ids": [event_id]}],
        }
    )


@pytest.mark.asyncio
async def test_model_checkpoint_extractor_uses_no_tools_and_runtime_owned_identity(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    session.append_user_message("implement context", turn_id=session.new_turn_id())
    provider = SequenceProvider([ModelResponse(content=_draft(2))])
    extractor = ModelCheckpointExtractor(
        provider=provider,
        model="checkpoint-model",
        maximum_output_tokens=2_048,
        maximum_checkpoint_bytes=16_384,
        maximum_input_tokens=8_000,
        workspace_root=tmp_path,
    )

    checkpoint = await extractor.extract(
        previous=None,
        events=session.events,
        source_from_event_id=1,
        source_through_event_id=2,
        checkpoint_id="a" * 32,
        version=1,
        focus=None,
    )

    assert checkpoint.checkpoint_id == "a" * 32
    assert checkpoint.session_id == session.metadata.session_id
    assert checkpoint.next_actions[0].source_event_ids == (2,)
    request = provider.requests[0]
    assert request.model == "checkpoint-model"
    assert request.tools == ()
    assert request.max_output_tokens == 2_048
    session.close()


@pytest.mark.asyncio
async def test_model_checkpoint_extractor_returns_safe_failure(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider(
        [ProviderError("bad", "api_key=synthetic-secret", is_retryable=False)]
    )
    extractor = ModelCheckpointExtractor(
        provider=provider,
        model="checkpoint-model",
        maximum_output_tokens=2_048,
        maximum_checkpoint_bytes=16_384,
        maximum_input_tokens=8_000,
        workspace_root=tmp_path,
    )

    with pytest.raises(CheckpointExtractionError) as error:
        await extractor.extract(
            previous=None,
            events=session.events,
            source_from_event_id=1,
            source_through_event_id=1,
            checkpoint_id="a" * 32,
            version=1,
            focus=None,
        )

    assert "synthetic-secret" not in str(error.value)
    session.close()


@pytest.mark.asyncio
async def test_model_checkpoint_extractor_applies_checkpoint_storage_byte_budget(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    session.append_user_message("checkpoint", turn_id=session.new_turn_id())
    draft = json.loads(_draft(2))
    draft["current_intent"] = "界" * 2_000
    provider = SequenceProvider([ModelResponse(content=json.dumps(draft, ensure_ascii=False))])
    extractor = ModelCheckpointExtractor(
        provider=provider,
        model="checkpoint-model",
        maximum_output_tokens=1_024,
        maximum_checkpoint_bytes=4_096,
        maximum_input_tokens=8_000,
        workspace_root=tmp_path,
    )

    with pytest.raises(CheckpointExtractionError, match="storage budget"):
        await extractor.extract(
            previous=None,
            events=session.events,
            source_from_event_id=1,
            source_through_event_id=2,
            checkpoint_id="a" * 32,
            version=1,
            focus=None,
        )

    session.close()


@pytest.mark.asyncio
async def test_model_checkpoint_output_tokens_are_distinct_from_storage_bytes(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    session.append_user_message("checkpoint", turn_id=session.new_turn_id())
    draft = json.loads(_draft(2))
    draft["current_intent"] = "界" * 500
    provider = SequenceProvider([ModelResponse(content=json.dumps(draft, ensure_ascii=False))])
    extractor = ModelCheckpointExtractor(
        provider=provider,
        model="checkpoint-model",
        maximum_output_tokens=1_024,
        maximum_checkpoint_bytes=16_384,
        maximum_input_tokens=8_000,
        workspace_root=tmp_path,
    )

    checkpoint = await extractor.extract(
        previous=None,
        events=session.events,
        source_from_event_id=1,
        source_through_event_id=2,
        checkpoint_id="a" * 32,
        version=1,
        focus=None,
    )

    assert checkpoint.current_intent == "界" * 500
    assert provider.requests[0].max_output_tokens == 1_024
    assert len(checkpoint.model_dump_json().encode("utf-8")) > 1_024
    session.close()


@pytest.mark.asyncio
async def test_checkpoint_request_obeys_full_token_budget_and_normalizes_host_paths(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = AgentSession.create(
        data_dir=tmp_path / "data",
        workspace=workspace,
        model="m",
    )
    session.append_user_message(
        f"inspect {workspace.resolve()}/src/main.py " + "界" * 600,
        turn_id=session.new_turn_id(),
    )
    provider = SequenceProvider([ModelResponse(content=_draft(2))])
    maximum_input_tokens = 3_200
    extractor = ModelCheckpointExtractor(
        provider=provider,
        model="checkpoint-model",
        maximum_output_tokens=1_024,
        maximum_checkpoint_bytes=16_384,
        maximum_input_tokens=maximum_input_tokens,
        workspace_root=workspace,
    )
    previous = Checkpoint(
        checkpoint_id="b" * 32,
        session_id=session.metadata.session_id,
        cycle_id=1,
        version=1,
        source_from_event_id=1,
        source_through_event_id=1,
        writer_model="deterministic-v1",
        created_at=datetime.now(UTC),
        current_intent=f"continue {workspace.resolve()}/src/main.py",
    )

    await extractor.extract(
        previous=previous,
        events=session.events,
        source_from_event_id=1,
        source_through_event_id=2,
        checkpoint_id="a" * 32,
        version=2,
        focus=f"verify {workspace.resolve()}/src/main.py",
    )

    request = provider.requests[0]
    assert Utf8TokenEstimator().estimate_request(request).total_tokens <= maximum_input_tokens
    request_text = request.model_dump_json()
    if str(workspace.resolve()) in request_text:
        pytest.fail("checkpoint provider request contains a host path")
    assert "<workspace-root>/src/main.py" in request_text
    session.close()


@pytest.mark.asyncio
async def test_checkpoint_input_that_cannot_fit_source_event_skips_provider(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider([])
    extractor = ModelCheckpointExtractor(
        provider=provider,
        model="checkpoint-model",
        maximum_output_tokens=1_024,
        maximum_checkpoint_bytes=16_384,
        maximum_input_tokens=800,
        workspace_root=tmp_path,
    )

    with pytest.raises(CheckpointExtractionError, match="cannot fit a source event"):
        await extractor.extract(
            previous=None,
            events=session.events,
            source_from_event_id=1,
            source_through_event_id=1,
            checkpoint_id="a" * 32,
            version=1,
            focus=None,
        )

    assert provider.requests == []
    session.close()


@pytest.mark.asyncio
async def test_configured_checkpoint_model_falls_back_without_blocking_compact(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider(
        [
            ModelResponse(content="normal answer"),
            ModelResponse(content="not-json"),
        ]
    )
    config = MiniAgentConfig(
        model=ModelConfig.model_validate(
            {"model": "m", "base_url": "https://example.test/v1"}
        ),
        context=ContextConfig(checkpoint_model="checkpoint-model"),
        runtime=RuntimeConfig(data_dir=tmp_path / "data", provider_retry_count=0),
    )
    agent = MiniAgent(config=config, provider=provider, session=session)
    await agent.run_turn("do the task")

    projection = await agent.compact()

    assert projection.checkpoint_id is not None
    assert len(provider.requests) == 2
    assert provider.requests[1].model == "checkpoint-model"
    session.close()


@pytest.mark.asyncio
async def test_automatic_checkpoint_preserves_the_only_turn_model_call(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider([ModelResponse(content="normal answer")])
    config = MiniAgentConfig(
        model=ModelConfig.model_validate(
            {"model": "m", "base_url": "https://example.test/v1"}
        ),
        context=ContextConfig(
            checkpoint_model="checkpoint-model",
            checkpoint_milestones=(0.0001,),
        ),
        runtime=RuntimeConfig(
            data_dir=tmp_path / "data",
            provider_retry_count=0,
            max_model_calls_per_turn=1,
        ),
    )
    agent = MiniAgent(config=config, provider=provider, session=session)

    response = await agent.run_turn("do the task")

    assert response.content == "normal answer"
    assert len(provider.requests) == 1
    assert provider.requests[0].model == "m"
    session.close()


@pytest.mark.asyncio
async def test_automatic_checkpoint_uses_only_unreserved_turn_capacity(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider(
        [
            ModelResponse(content=_draft(3)),
            ModelResponse(content="normal answer"),
        ]
    )
    config = MiniAgentConfig(
        model=ModelConfig.model_validate(
            {"model": "m", "base_url": "https://example.test/v1"}
        ),
        context=ContextConfig(
            checkpoint_model="checkpoint-model",
            checkpoint_milestones=(0.0001,),
        ),
        runtime=RuntimeConfig(
            data_dir=tmp_path / "data",
            provider_retry_count=0,
            max_model_calls_per_turn=2,
        ),
    )
    agent = MiniAgent(config=config, provider=provider, session=session)

    response = await agent.run_turn("do the task")

    assert response.content == "normal answer"
    assert [request.model for request in provider.requests] == ["checkpoint-model", "m"]
    session.close()
