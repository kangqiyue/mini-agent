"""Request-budget and atomic conversation projection tests."""

import json
from datetime import UTC, datetime

import pytest
from pydantic import HttpUrl

from mini_agent import context as context_module
from mini_agent.checkpoint import Checkpoint, CheckpointItem
from mini_agent.config import ContextConfig, ModelConfig
from mini_agent.context import (
    ContextBudgetExceeded,
    ContextConfigurationError,
    RequestBudget,
    RequestTokenEstimate,
    Utf8TokenEstimator,
    build_rebuild_projection,
    group_tool_rounds,
    project_full_history,
    resolve_request_budget,
    validate_prospective_request_floor,
    validate_request_floor,
)
from mini_agent.messages import ConversationMessage, MessageRole, ModelRequest, ToolCall
from mini_agent.provider import ProviderCapabilities
from mini_agent.tools.base import ToolDefinition


def _model(**updates: object) -> ModelConfig:
    values: dict[str, object] = {
        "model": "m",
        "base_url": HttpUrl("https://example.test/v1"),
        "context_window": 128_000,
        "max_output_tokens": 16_384,
    }
    values.update(updates)
    return ModelConfig.model_validate(values)


def test_resolve_request_budget_clamps_all_provider_and_config_limits() -> None:
    budget = resolve_request_budget(
        model=_model(max_context=80_000, max_output_tokens=20_000),
        context=ContextConfig(
            reserve_tokens=18_000,
            rebuild_seed_max_tokens=65_536,
        ),
        capabilities=ProviderCapabilities(
            context_window=100_000,
            max_input_tokens=70_000,
            max_output_tokens=12_000,
        ),
    )

    assert budget.working_context_limit == 80_000
    assert budget.desired_output_tokens == 12_000
    assert budget.output_reserve == 18_000
    assert budget.request_input_limit == 62_000
    assert budget.rebuild_seed_tokens == 31_000


def test_small_context_clamps_output_before_rejecting_input_space() -> None:
    budget = resolve_request_budget(
        model=_model(context_window=4_096, max_output_tokens=16_384),
        context=ContextConfig(reserve_tokens=16_384, minimum_input_tokens=1_024),
        capabilities=ProviderCapabilities(
            context_window=4_096,
            max_output_tokens=16_384,
        ),
    )

    assert budget.request_input_limit == 1_024
    assert budget.output_reserve == 3_072
    assert budget.desired_output_tokens == 3_072


def test_request_budget_rejects_a_window_below_minimum_input() -> None:
    with pytest.raises(ContextConfigurationError, match="minimum"):
        resolve_request_budget(
            model=_model(context_window=4_096),
            context=ContextConfig(minimum_input_tokens=4_096),
            capabilities=ProviderCapabilities(
                context_window=4_096,
                max_output_tokens=1_024,
            ),
        )


def test_estimator_counts_tools_messages_and_protocol_separately() -> None:
    request = ModelRequest(
        model="m",
        messages=(ConversationMessage(role=MessageRole.USER, content="你好"),),
        tools=(
            ToolDefinition(
                name="read_file",
                description="Read one file",
                parameters={"type": "object"},
                is_read_only=True,
            ),
        ),
    )

    estimate = Utf8TokenEstimator().estimate_request(request)

    assert estimate.message_tokens >= len("你好".encode())
    assert estimate.tool_tokens > 0
    assert estimate.protocol_tokens > 0
    assert estimate.total_tokens == (
        estimate.tool_tokens + estimate.message_tokens + estimate.protocol_tokens
    )


def test_full_history_projection_fails_before_sending_an_oversized_request() -> None:
    request = ModelRequest(
        model="m",
        messages=(ConversationMessage(role=MessageRole.USER, content="x" * 2_000),),
    )
    budget = resolve_request_budget(
        model=_model(context_window=4_096, max_output_tokens=1_024),
        context=ContextConfig(reserve_tokens=1_024, minimum_input_tokens=1_024),
        capabilities=ProviderCapabilities(
            context_window=4_096,
            max_output_tokens=1_024,
            max_input_tokens=1_500,
        ),
    )

    with pytest.raises(ContextBudgetExceeded, match="requires rebuild"):
        project_full_history(
            request,
            budget=budget,
            estimator=Utf8TokenEstimator(),
        )


def test_request_floor_allows_the_exact_resolved_input_boundary() -> None:
    request = ModelRequest(
        model="m",
        messages=(ConversationMessage(role=MessageRole.SYSTEM, content="floor"),),
    )
    estimator = Utf8TokenEstimator()
    estimate = estimator.estimate_request(request)
    budget = RequestBudget(
        working_context_limit=estimate.total_tokens + 1,
        desired_output_tokens=1,
        output_reserve=1,
        request_input_limit=estimate.total_tokens,
        rebuild_seed_tokens=1,
    )

    validate_request_floor(request, budget=budget, estimator=estimator)


def test_request_floor_rejects_an_over_budget_base_request_without_details() -> None:
    request = ModelRequest(
        model="m",
        messages=(ConversationMessage(role=MessageRole.SYSTEM, content="floor"),),
    )
    estimator = Utf8TokenEstimator()
    estimate = estimator.estimate_request(request)
    budget = RequestBudget(
        working_context_limit=estimate.total_tokens,
        desired_output_tokens=1,
        output_reserve=1,
        request_input_limit=estimate.total_tokens - 1,
        rebuild_seed_tokens=1,
    )

    with pytest.raises(
        ContextConfigurationError,
        match="Base model request exceeds the active input budget",
    ):
        validate_request_floor(request, budget=budget, estimator=estimator)


def test_prospective_request_floor_allows_an_exact_full_request_boundary() -> None:
    request = ModelRequest(
        model="m",
        messages=(
            ConversationMessage(role=MessageRole.SYSTEM, content="system"),
            ConversationMessage(role=MessageRole.USER, content="user"),
        ),
    )
    estimator = Utf8TokenEstimator()
    estimate = estimator.estimate_request(request)
    budget = RequestBudget(
        working_context_limit=estimate.total_tokens + 1,
        desired_output_tokens=1,
        output_reserve=1,
        request_input_limit=estimate.total_tokens,
        rebuild_seed_tokens=1,
    )

    validate_prospective_request_floor(
        request,
        budget=budget,
        estimator=estimator,
        maximum_checkpoint_bytes=1_048_576,
    )


def test_prospective_request_floor_rejects_user_when_base_alone_fits() -> None:
    system_message = ConversationMessage(role=MessageRole.SYSTEM, content="system")
    request = ModelRequest(
        model="m",
        messages=(
            system_message,
            ConversationMessage(role=MessageRole.USER, content="x" * 128),
        ),
    )
    estimator = Utf8TokenEstimator()
    base_request = request.model_copy(update={"messages": (system_message,)})
    base_estimate = estimator.estimate_request(base_request)
    budget = RequestBudget(
        working_context_limit=base_estimate.total_tokens + 1,
        desired_output_tokens=1,
        output_reserve=1,
        request_input_limit=base_estimate.total_tokens,
        rebuild_seed_tokens=1,
    )

    validate_request_floor(base_request, budget=budget, estimator=estimator)
    with pytest.raises(ContextConfigurationError):
        validate_prospective_request_floor(
            request,
            budget=budget,
            estimator=estimator,
            maximum_checkpoint_bytes=1_048_576,
        )


def test_prospective_floor_uses_the_full_bounded_checkpoint_view() -> None:
    request = ModelRequest(
        model="m",
        messages=(
            ConversationMessage(role=MessageRole.SYSTEM, content="s" * 50_000),
            ConversationMessage(role=MessageRole.USER, content="u" * 20_000),
        ),
    )
    budget = RequestBudget(
        working_context_limit=65_537,
        desired_output_tokens=1,
        output_reserve=1,
        request_input_limit=65_536,
        rebuild_seed_tokens=32_768,
    )
    estimator = Utf8TokenEstimator()
    legacy_checkpoint = Checkpoint(
        checkpoint_id="a" * 32,
        session_id="b" * 32,
        cycle_id=1,
        version=1,
        source_from_event_id=1,
        source_through_event_id=1,
        current_intent="x" * 8_000,
        writer_model="deterministic-v1",
        created_at=datetime.now(UTC),
    )

    legacy_projection = build_rebuild_projection(
        request,
        checkpoint=legacy_checkpoint,
        budget=budget,
        estimator=estimator,
        projection_version=1,
    )

    assert legacy_projection.estimate.total_tokens <= budget.request_input_limit
    with pytest.raises(ContextConfigurationError):
        validate_prospective_request_floor(
            request,
            budget=budget,
            estimator=estimator,
            maximum_checkpoint_bytes=1_048_576,
        )


def test_prospective_floor_bounds_checkpoint_view_when_rebuild_seed_is_huge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OverflowEstimator:
        def estimate_request(self, request: ModelRequest) -> RequestTokenEstimate:
            del request
            return RequestTokenEstimate(
                system_tokens=1_000_000_001,
                tool_tokens=0,
                message_tokens=0,
                protocol_tokens=0,
                total_tokens=1_000_000_001,
            )

    requested_checkpoint_sizes: list[int] = []

    def bounded_checkpoint_view(*, maximum_content_bytes: int) -> ConversationMessage:
        requested_checkpoint_sizes.append(maximum_content_bytes)
        return ConversationMessage(role=MessageRole.SYSTEM, content="checkpoint")

    monkeypatch.setattr(
        context_module,
        "_maximum_checkpoint_message_for_floor",
        bounded_checkpoint_view,
    )
    request = ModelRequest(
        model="m",
        messages=(
            ConversationMessage(role=MessageRole.SYSTEM, content="system"),
            ConversationMessage(role=MessageRole.USER, content="user"),
        ),
    )
    budget = RequestBudget(
        working_context_limit=1_000_000_001,
        desired_output_tokens=1,
        output_reserve=1,
        request_input_limit=1_000_000_000,
        rebuild_seed_tokens=1_000_000_000,
    )

    with pytest.raises(ContextConfigurationError):
        validate_prospective_request_floor(
            request,
            budget=budget,
            estimator=OverflowEstimator(),  # type: ignore[arg-type]
            maximum_checkpoint_bytes=4_096,
        )

    assert requested_checkpoint_sizes == [4_096 + 8_192]


def test_tool_call_and_every_result_form_one_atomic_group() -> None:
    calls = (
        ToolCall(id="one", name="read_file", arguments_json=json.dumps({"path": "a"})),
        ToolCall(id="two", name="read_file", arguments_json=json.dumps({"path": "b"})),
    )
    messages = (
        ConversationMessage(role=MessageRole.USER, content="read both"),
        ConversationMessage(role=MessageRole.ASSISTANT, tool_calls=calls),
        ConversationMessage(role=MessageRole.TOOL, content="a", tool_call_id="one"),
        ConversationMessage(role=MessageRole.TOOL, content="b", tool_call_id="two"),
        ConversationMessage(role=MessageRole.ASSISTANT, content="done"),
    )

    groups = group_tool_rounds(messages)

    assert tuple(len(group) for group in groups) == (1, 3, 1)
    assert groups[1][0].tool_calls == calls
    assert {message.tool_call_id for message in groups[1][1:]} == {"one", "two"}


def test_rebuild_keeps_checkpoint_latest_user_and_atomic_tool_round() -> None:
    checkpoint = Checkpoint(
        checkpoint_id="a" * 32,
        session_id="b" * 32,
        cycle_id=1,
        version=1,
        source_from_event_id=1,
        source_through_event_id=10,
        current_intent="Keep the exact target stable",
        constraints_and_preferences=(
            CheckpointItem(text="Do not rewrite history", source_event_ids=(2,)),
        ),
        writer_model="deterministic-v1",
        created_at=datetime.now(UTC),
    )
    call = ToolCall(id="read", name="read_file", arguments_json='{"path":"a"}')
    messages = (
        ConversationMessage(role=MessageRole.USER, content="old" * 200),
        ConversationMessage(role=MessageRole.ASSISTANT, content="old answer" * 200),
        ConversationMessage(role=MessageRole.USER, content="latest request"),
        ConversationMessage(role=MessageRole.ASSISTANT, tool_calls=(call,)),
        ConversationMessage(role=MessageRole.TOOL, content="result", tool_call_id="read"),
    )
    request = ModelRequest(model="m", messages=messages)
    budget = resolve_request_budget(
        model=_model(context_window=8_192, max_output_tokens=1_024),
        context=ContextConfig(
            reserve_tokens=1_024,
            minimum_input_tokens=1_024,
            rebuild_seed_max_tokens=1_200,
        ),
        capabilities=ProviderCapabilities(
            context_window=8_192,
            max_output_tokens=1_024,
        ),
    )

    projection = build_rebuild_projection(
        request,
        checkpoint=checkpoint,
        budget=budget,
        estimator=Utf8TokenEstimator(),
        projection_version=2,
    )

    assert projection.strategy == "checkpoint_rebuild"
    assert projection.messages[0].role is MessageRole.SYSTEM
    assert "Do not rewrite history" in (projection.messages[0].content or "")
    assert any(message.content == "latest request" for message in projection.messages)
    assistant_index = next(
        index for index, message in enumerate(projection.messages) if message.tool_calls
    )
    user_index = next(
        index
        for index, message in enumerate(projection.messages)
        if message.content == "latest request"
    )
    assert user_index < assistant_index
    assert projection.messages[assistant_index + 1].tool_call_id == "read"
    assert projection.omitted_message_count >= 1


def test_rebuild_merges_checkpoint_into_the_authoritative_system_message() -> None:
    checkpoint = Checkpoint(
        checkpoint_id="a" * 32,
        session_id="b" * 32,
        cycle_id=1,
        version=1,
        source_from_event_id=1,
        source_through_event_id=2,
        current_intent="Preserve the active task",
        writer_model="deterministic-v1",
        created_at=datetime.now(UTC),
    )
    request = ModelRequest(
        model="m",
        messages=(
            ConversationMessage(
                role=MessageRole.SYSTEM,
                content="## Identity\nMini Agent\n\n## Runtime\n- Model: m",
            ),
            ConversationMessage(role=MessageRole.USER, content="Continue"),
        ),
    )
    budget = resolve_request_budget(
        model=_model(context_window=8_192, max_output_tokens=1_024),
        context=ContextConfig(reserve_tokens=1_024, rebuild_seed_max_tokens=4_000),
        capabilities=ProviderCapabilities(context_window=8_192, max_output_tokens=1_024),
    )

    projection = build_rebuild_projection(
        request,
        checkpoint=checkpoint,
        budget=budget,
        estimator=Utf8TokenEstimator(),
        projection_version=2,
    )

    system_messages = [
        message for message in projection.messages if message.role is MessageRole.SYSTEM
    ]
    assert len(system_messages) == 1
    system_content = system_messages[0].content or ""
    assert "## Identity\nMini Agent" in system_content
    assert "## Checkpoint" in system_content
    assert "Preserve the active task" in system_content


def test_rebuild_bounds_a_multibyte_checkpoint_view_without_changing_storage() -> None:
    long_intent = "界" * 4_000
    checkpoint = Checkpoint(
        checkpoint_id="a" * 32,
        session_id="b" * 32,
        cycle_id=1,
        version=1,
        source_from_event_id=1,
        source_through_event_id=2,
        current_intent=long_intent,
        writer_model="checkpoint-model",
        created_at=datetime.now(UTC),
    )
    request = ModelRequest(
        model="m",
        messages=(ConversationMessage(role=MessageRole.USER, content="Continue"),),
    )
    budget = resolve_request_budget(
        model=_model(context_window=8_192, max_output_tokens=1_024),
        context=ContextConfig(reserve_tokens=1_024, rebuild_seed_max_tokens=2_048),
        capabilities=ProviderCapabilities(context_window=8_192, max_output_tokens=1_024),
    )

    projection = build_rebuild_projection(
        request,
        checkpoint=checkpoint,
        budget=budget,
        estimator=Utf8TokenEstimator(),
        projection_version=2,
    )

    checkpoint_view = projection.messages[0].content or ""
    assert "checkpoint view truncated" in checkpoint_view
    assert len(checkpoint_view.encode("utf-8")) <= budget.rebuild_seed_tokens // 2
    assert checkpoint.current_intent == long_intent
