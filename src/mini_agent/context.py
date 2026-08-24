"""Deterministic request budgeting for an immutable conversation transcript."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mini_agent.checkpoint import Checkpoint, CheckpointItem
from mini_agent.config import ContextConfig, ModelConfig
from mini_agent.messages import ConversationMessage, MessageRole, ModelRequest
from mini_agent.provider import ProviderCapabilities
from mini_agent.tools.base import ToolDefinition


class ContextConfigurationError(ValueError):
    """Raised when physical limits cannot fit a minimally useful request."""


class ContextBudgetExceeded(RuntimeError):
    """Raised when the current projection cannot fit its input budget."""


def initial_goal_objective(user_input: str) -> str:
    """Keep the durable first-goal objective within its schema bound."""

    if len(user_input) <= 4_096:
        return user_input
    return f"{user_input[:4_032]}\n[objective truncated; use history_read for exact request]"


class RequestBudget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    working_context_limit: int = Field(ge=1)
    desired_output_tokens: int = Field(ge=1)
    output_reserve: int = Field(ge=1)
    request_input_limit: int = Field(ge=1)
    rebuild_seed_tokens: int = Field(ge=1)

    @model_validator(mode="after")
    def require_non_overlapping_input_and_output(self) -> Self:
        if self.request_input_limit + self.output_reserve > self.working_context_limit:
            raise ValueError("Request input and output reserve exceed the working context")
        if self.desired_output_tokens > self.output_reserve:
            raise ValueError("Desired output must fit inside the output reserve")
        if self.rebuild_seed_tokens > self.request_input_limit:
            raise ValueError("Rebuild seed must fit inside the request input budget")
        return self


class RequestTokenEstimate(BaseModel):
    """Conservative UTF-8 byte estimate with explicit protocol overhead."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    system_tokens: int = Field(default=0, ge=0)
    tool_tokens: int = Field(ge=0)
    message_tokens: int = Field(ge=0)
    protocol_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def require_total_to_match_sections(self) -> Self:
        expected = (
            self.system_tokens
            + self.tool_tokens
            + self.message_tokens
            + self.protocol_tokens
        )
        if self.total_tokens != expected:
            raise ValueError("Token estimate total does not match its sections")
        return self


class ContextProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(default=1, ge=1)
    strategy: str = Field(default="full_history", min_length=1)
    messages: tuple[ConversationMessage, ...] = ()
    estimate: RequestTokenEstimate
    input_limit: int = Field(ge=1)
    checkpoint_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    checkpoint_watermark: int = Field(default=0, ge=0)
    omitted_message_count: int = Field(default=0, ge=0)

    @property
    def utilization_ratio(self) -> float:
        return self.estimate.total_tokens / self.input_limit


class Utf8TokenEstimator:
    """Stable v0.1 estimator: one token per UTF-8 byte plus fixed framing.

    This intentionally overestimates ordinary text. It is tokenizer-independent
    and therefore safe for an unknown OpenAI-compatible route. Provider usage
    calibration can replace it later without changing the projection contract.
    """

    message_overhead_tokens = 8
    tool_definition_overhead_tokens = 16
    tool_call_overhead_tokens = 12
    request_overhead_tokens = 8

    def estimate_request(self, request: ModelRequest) -> RequestTokenEstimate:
        tool_tokens = sum(self._tool_tokens(tool) for tool in request.tools)
        system_tokens = sum(
            self._message_tokens(message)
            for message in request.messages
            if message.role is MessageRole.SYSTEM
        )
        message_tokens = sum(
            self._message_tokens(message)
            for message in request.messages
            if message.role is not MessageRole.SYSTEM
        )
        protocol_tokens = (
            self.request_overhead_tokens
            + len(request.messages) * self.message_overhead_tokens
            + len(request.tools) * self.tool_definition_overhead_tokens
            + sum(
                len(message.tool_calls) * self.tool_call_overhead_tokens
                for message in request.messages
            )
        )
        total_tokens = system_tokens + tool_tokens + message_tokens + protocol_tokens
        return RequestTokenEstimate(
            system_tokens=system_tokens,
            tool_tokens=tool_tokens,
            message_tokens=message_tokens,
            protocol_tokens=protocol_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _tool_tokens(tool: ToolDefinition) -> int:
        return len(tool.model_dump_json().encode("utf-8"))

    @staticmethod
    def _message_tokens(message: ConversationMessage) -> int:
        token_count = len(message.role.value.encode("utf-8"))
        if message.content is not None:
            token_count += len(message.content.encode("utf-8"))
        if message.tool_call_id is not None:
            token_count += len(message.tool_call_id.encode("utf-8"))
        for tool_call in message.tool_calls:
            token_count += len(tool_call.id.encode("utf-8"))
            token_count += len(tool_call.name.encode("utf-8"))
            token_count += len(tool_call.arguments_json.encode("utf-8"))
        return token_count


def resolve_request_budget(
    *,
    model: ModelConfig,
    context: ContextConfig,
    capabilities: ProviderCapabilities,
) -> RequestBudget:
    configured_context = model.max_context or capabilities.context_window
    working_context_limit = min(capabilities.context_window, configured_context)
    maximum_reserve = working_context_limit - context.minimum_input_tokens
    if maximum_reserve < 1:
        raise ContextConfigurationError("Context window cannot fit the minimum input budget")

    desired_output_tokens = min(
        model.max_output_tokens,
        capabilities.max_output_tokens,
        maximum_reserve,
    )
    output_reserve = min(
        max(context.reserve_tokens, desired_output_tokens),
        maximum_reserve,
    )
    physical_input_limit = capabilities.max_input_tokens or working_context_limit
    request_input_limit = min(
        physical_input_limit,
        working_context_limit - output_reserve,
    )
    if request_input_limit < context.minimum_input_tokens:
        raise ContextConfigurationError("Resolved request input budget is below the minimum")

    return RequestBudget(
        working_context_limit=working_context_limit,
        desired_output_tokens=desired_output_tokens,
        output_reserve=output_reserve,
        request_input_limit=request_input_limit,
        rebuild_seed_tokens=min(
            context.rebuild_seed_max_tokens,
            request_input_limit // 2,
        ),
    )


def validate_request_floor(
    request: ModelRequest,
    *,
    budget: RequestBudget,
    estimator: Utf8TokenEstimator,
) -> None:
    """Reject a configuration whose irreducible request cannot be sent.

    ``request_input_limit`` already excludes the output reserve.  Applying that
    reserve a second time here would incorrectly reject requests at the exact
    physical boundary.
    """

    estimate = estimator.estimate_request(request)
    if estimate.total_tokens > budget.request_input_limit:
        raise ContextConfigurationError(
            "Base model request exceeds the active input budget"
        )


def validate_prospective_request_floor(
    request: ModelRequest,
    *,
    budget: RequestBudget,
    estimator: Utf8TokenEstimator,
    maximum_checkpoint_bytes: int,
) -> None:
    """Validate a new turn before its goal or user message is persisted.

    A full request is preferred. If it overflows, verify that the deterministic
    rebuild path can retain a bounded checkpoint view and the latest user
    request. This is deliberately pure: callers can fail before creating any
    durable session state.
    """

    if estimator.estimate_request(request).total_tokens <= budget.request_input_limit:
        return
    try:
        build_rebuild_projection(
            request,
            checkpoint=None,
            checkpoint_view=_maximum_checkpoint_message_for_floor(
                maximum_content_bytes=min(
                    max(256, budget.rebuild_seed_tokens // 2),
                    maximum_checkpoint_bytes + _CHECKPOINT_RENDER_OVERHEAD_BYTES,
                )
            ),
            budget=budget,
            estimator=estimator,
            projection_version=1,
        )
    except ContextBudgetExceeded as error:
        raise ContextConfigurationError(
            "Initial model request cannot fit the active input budget"
        ) from error


_CHECKPOINT_RENDER_OVERHEAD_BYTES = 8_192


def _maximum_checkpoint_message_for_floor(
    *,
    maximum_content_bytes: int,
) -> ConversationMessage:
    """Build one bounded, non-durable checkpoint view without unbounded items."""

    notice = "[checkpoint view truncated; use history_read for exact durable details]"
    notice_bytes = len(notice.encode("utf-8"))
    prefix_bytes = max(0, maximum_content_bytes - notice_bytes)
    return ConversationMessage(
        role=MessageRole.SYSTEM,
        content=f"{'x' * prefix_bytes}{notice}",
    )


def project_full_history(
    request: ModelRequest,
    *,
    budget: RequestBudget,
    estimator: Utf8TokenEstimator,
) -> ContextProjection:
    """Validate the current full-history view without mutating the transcript."""

    estimate = estimator.estimate_request(request)
    if estimate.total_tokens > budget.request_input_limit:
        raise ContextBudgetExceeded(
            "Full-history request exceeds the active input budget and requires rebuild"
        )
    return ContextProjection(
        messages=request.messages,
        estimate=estimate,
        input_limit=budget.request_input_limit,
    )


def group_tool_rounds(
    messages: Iterable[ConversationMessage],
) -> tuple[tuple[ConversationMessage, ...], ...]:
    """Group assistant tool calls with every matching result as one atomic unit."""

    message_list = tuple(messages)
    _validate_tool_pair_sequence(message_list)
    groups: list[tuple[ConversationMessage, ...]] = []
    cursor = 0
    while cursor < len(message_list):
        message = message_list[cursor]
        if message.role is not MessageRole.ASSISTANT or not message.tool_calls:
            groups.append((message,))
            cursor += 1
            continue

        result_count = len(message.tool_calls)
        group = message_list[cursor : cursor + result_count + 1]
        if len(group) != result_count + 1:
            raise ValueError("Assistant tool round is missing terminal results")
        groups.append(group)
        cursor += len(group)
    return tuple(groups)


def _validate_tool_pair_sequence(messages: tuple[ConversationMessage, ...]) -> None:
    """Validate tool pairing without requiring a user-first provider request."""

    pending: set[str] = set()
    seen: set[str] = set()
    for message in messages:
        if message.role is MessageRole.ASSISTANT:
            if pending:
                raise ValueError("Assistant message appeared before pending tool results")
            new_ids = {call.id for call in message.tool_calls}
            if len(new_ids) != len(message.tool_calls) or new_ids & seen:
                raise ValueError("Tool call ids are duplicate or reused")
            seen.update(new_ids)
            pending.update(new_ids)
        elif message.role is MessageRole.TOOL:
            if message.tool_call_id not in pending:
                raise ValueError("Tool result does not match a pending tool call")
            pending.remove(message.tool_call_id)
        elif pending:
            raise ValueError("Non-tool message appeared before pending tool results")
    if pending:
        raise ValueError("Conversation contains unresolved tool calls")


def build_rebuild_projection(
    request: ModelRequest,
    *,
    checkpoint: Checkpoint | None,
    checkpoint_view: ConversationMessage | None = None,
    budget: RequestBudget,
    estimator: Utf8TokenEstimator,
    projection_version: int,
) -> ContextProjection:
    """Build a bounded seed from checkpoint state and newest atomic message groups."""

    system_context, conversation_messages = _split_system_context(request.messages)
    groups = group_tool_rounds(conversation_messages)
    if checkpoint is not None and checkpoint_view is not None:
        raise ValueError("Rebuild projection accepts either a checkpoint or a checkpoint view")
    checkpoint_message = checkpoint_view or (
        _checkpoint_message(
            checkpoint,
            maximum_content_bytes=max(256, budget.rebuild_seed_tokens // 2),
        )
        if checkpoint is not None
        else None
    )
    selected_group_indexes: set[int] = set()
    latest_user_index = _latest_user_group_index(groups)
    fitted_latest_user: tuple[ConversationMessage, ...] | None = None
    if latest_user_index is not None:
        selected_group_indexes.add(latest_user_index)
        fitted_latest_user = _fit_required_user_group(
            request,
            system_context=system_context,
            checkpoint_message=checkpoint_message,
            group=groups[latest_user_index],
            budget=budget,
            estimator=estimator,
        )

    required_messages = _projection_messages(
        system_context,
        checkpoint_message,
        [fitted_latest_user] if fitted_latest_user is not None else [],
    )
    required_estimate = estimator.estimate_request(
        request.model_copy(update={"messages": required_messages})
    )
    rebuild_target = min(
        budget.request_input_limit,
        max(budget.rebuild_seed_tokens, required_estimate.total_tokens),
    )

    for index in reversed(range(len(groups))):
        if index in selected_group_indexes:
            continue
        candidate_indexes = {*selected_group_indexes, index}
        candidate_groups = _selected_groups(
            groups,
            candidate_indexes,
            latest_user_index=latest_user_index,
            fitted_latest_user=fitted_latest_user,
        )
        candidate_messages = _projection_messages(
            system_context,
            checkpoint_message,
            candidate_groups,
        )
        estimate = estimator.estimate_request(
            request.model_copy(update={"messages": candidate_messages})
        )
        if estimate.total_tokens <= rebuild_target:
            selected_group_indexes = candidate_indexes

    selected_groups = _selected_groups(
        groups,
        selected_group_indexes,
        latest_user_index=latest_user_index,
        fitted_latest_user=fitted_latest_user,
    )
    projected_messages = _projection_messages(
        system_context,
        checkpoint_message,
        selected_groups,
    )
    projected_request = request.model_copy(update={"messages": projected_messages})
    estimate = estimator.estimate_request(projected_request)
    if estimate.total_tokens > budget.request_input_limit:
        raise ContextBudgetExceeded("Rebuild seed cannot fit the active input budget")
    return ContextProjection(
        version=projection_version,
        strategy="checkpoint_rebuild" if checkpoint is not None else "deterministic_recovery",
        messages=projected_messages,
        estimate=estimate,
        input_limit=budget.request_input_limit,
        checkpoint_id=checkpoint.checkpoint_id if checkpoint is not None else None,
        checkpoint_watermark=(
            checkpoint.source_through_event_id if checkpoint is not None else 0
        ),
        omitted_message_count=(
            len(conversation_messages) - sum(len(group) for group in selected_groups)
        ),
    )


def _projection_messages(
    system_context: ConversationMessage | None,
    checkpoint_message: ConversationMessage | None,
    groups: list[tuple[ConversationMessage, ...]],
) -> tuple[ConversationMessage, ...]:
    messages: list[ConversationMessage] = []
    merged_system = _merge_system_context(system_context, checkpoint_message)
    if merged_system is not None:
        messages.append(merged_system)
    for group in groups:
        messages.extend(group)
    return tuple(messages)


def _latest_user_group_index(
    groups: tuple[tuple[ConversationMessage, ...], ...],
) -> int | None:
    return next(
        (
            index
            for index in reversed(range(len(groups)))
            if len(groups[index]) == 1
            and groups[index][0].role is MessageRole.USER
        ),
        None,
    )


def _selected_groups(
    groups: tuple[tuple[ConversationMessage, ...], ...],
    indexes: set[int],
    *,
    latest_user_index: int | None,
    fitted_latest_user: tuple[ConversationMessage, ...] | None,
) -> list[tuple[ConversationMessage, ...]]:
    return [
        (
            fitted_latest_user
            if index == latest_user_index and fitted_latest_user is not None
            else groups[index]
        )
        for index in sorted(indexes)
    ]


def _fit_required_user_group(
    request: ModelRequest,
    *,
    system_context: ConversationMessage | None,
    checkpoint_message: ConversationMessage | None,
    group: tuple[ConversationMessage, ...],
    budget: RequestBudget,
    estimator: Utf8TokenEstimator,
) -> tuple[ConversationMessage, ...]:
    message = group[0]
    content = message.content or ""
    candidate = group
    while True:
        messages = _projection_messages(system_context, checkpoint_message, [candidate])
        estimate = estimator.estimate_request(request.model_copy(update={"messages": messages}))
        if estimate.total_tokens <= budget.request_input_limit:
            return candidate
        if len(content) <= 256:
            raise ContextBudgetExceeded("Latest user request cannot fit the rebuild seed")
        content = content[: max(256, len(content) // 2)]
        candidate = (
            message.model_copy(
                update={
                    "content": (
                        f"{content}\n\n"
                        "[latest user request truncated; use history_read for exact text]"
                    )
                }
            ),
        )


def _split_system_context(
    messages: tuple[ConversationMessage, ...],
) -> tuple[ConversationMessage | None, tuple[ConversationMessage, ...]]:
    first_message = next(iter(messages), None)
    if first_message is None or first_message.role is not MessageRole.SYSTEM:
        return None, messages
    if len(messages) > 1 and messages[1].role is MessageRole.SYSTEM:
        raise ValueError("Model request must have at most one leading system message")
    return first_message, messages[1:]


def _merge_system_context(
    system_context: ConversationMessage | None,
    checkpoint_message: ConversationMessage | None,
) -> ConversationMessage | None:
    if system_context is None:
        return checkpoint_message
    if checkpoint_message is None:
        return system_context
    return system_context.model_copy(
        update={
            "content": (
                f"{system_context.content}\n\n"
                "## Checkpoint\n"
                f"{checkpoint_message.content}"
            )
        }
    )


def _checkpoint_message(
    checkpoint: Checkpoint,
    *,
    maximum_content_bytes: int,
) -> ConversationMessage:
    sections: list[str] = [
        "Session checkpoint (lossy; use history_search/history_read for exact details).",
        f"Watermark: event {checkpoint.source_through_event_id}",
        f"Current intent: {checkpoint.current_intent}",
    ]
    _append_checkpoint_items(sections, "Constraints", checkpoint.constraints_and_preferences)
    _append_checkpoint_items(sections, "Acceptance criteria", checkpoint.acceptance_criteria)
    _append_checkpoint_items(sections, "Completed", checkpoint.completed)
    _append_checkpoint_items(sections, "Active work", checkpoint.active_work)
    _append_checkpoint_items(sections, "Blocked", checkpoint.blocked)
    _append_checkpoint_items(sections, "Next actions", checkpoint.next_actions)
    _append_checkpoint_items(sections, "Relevant files", checkpoint.relevant_files)
    _append_checkpoint_items(sections, "Errors and fixes", checkpoint.errors_and_fixes)
    _append_checkpoint_items(sections, "Key decisions", checkpoint.key_decisions)
    _append_checkpoint_items(sections, "Artifacts", checkpoint.artifact_references)
    content = _fit_utf8_text(
        "\n".join(sections),
        maximum_bytes=maximum_content_bytes,
        truncation_notice=(
            "\n[checkpoint view truncated; use history_read for exact durable details]"
        ),
    )
    return ConversationMessage(role=MessageRole.SYSTEM, content=content)


def _fit_utf8_text(value: str, *, maximum_bytes: int, truncation_notice: str) -> str:
    if len(value.encode("utf-8")) <= maximum_bytes:
        return value
    notice_bytes = truncation_notice.encode("utf-8")
    prefix_byte_count = max(0, maximum_bytes - len(notice_bytes))
    prefix = value.encode("utf-8")[:prefix_byte_count].decode("utf-8", errors="ignore")
    return f"{prefix.rstrip()}{truncation_notice}"


def _append_checkpoint_items(
    sections: list[str],
    title: str,
    items: tuple[CheckpointItem, ...],
) -> None:
    if not items:
        return
    sections.append(f"{title}:")
    sections.extend(
        f"- {item.text} [events {','.join(str(value) for value in item.source_event_ids)}]"
        for item in items
    )
