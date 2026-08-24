"""Minimal conversation loop with durable turn boundaries."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from mini_agent.artifacts import ArtifactStore
from mini_agent.checkpoint import Checkpoint, CheckpointStore
from mini_agent.checkpoint_writer import (
    CheckpointCoordinator,
    DeterministicCheckpointExtractor,
    FallbackCheckpointExtractor,
)
from mini_agent.config import MiniAgentConfig
from mini_agent.context import (
    ContextBudgetExceeded,
    ContextProjection,
    RequestBudget,
    Utf8TokenEstimator,
    build_rebuild_projection,
    initial_goal_objective,
    project_full_history,
    resolve_request_budget,
    validate_prospective_request_floor,
    validate_request_floor,
)
from mini_agent.events import ToolRecoveryStatus
from mini_agent.exec_command_safety import sanitize_exec_command_arguments
from mini_agent.goal import GoalState, GoalStatus
from mini_agent.host_path_redaction import (
    normalize_conversation_message_host_paths,
    normalize_host_path_text,
    normalize_model_request_host_paths,
    normalize_model_response_host_paths,
    redact_host_paths,
)
from mini_agent.messages import (
    MAX_TOOL_CALLS_PER_RESPONSE,
    ConversationMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from mini_agent.permissions import (
    ApprovalArgumentsTooLargeError,
    ApprovalDecision,
    ApprovalRequest,
    PermissionController,
    built_in_apply_patch_session_grant,
)
from mini_agent.provider import (
    ModelProvider,
    ProviderError,
    resolve_provider_capabilities,
    safe_provider_error,
)
from mini_agent.redaction import redact_text
from mini_agent.redaction_types import RedactionKind, RedactionSummary
from mini_agent.session import AgentSession
from mini_agent.storage_paths import resolve_data_dir
from mini_agent.system_prompt import RuntimeContext, SystemPromptAssembler
from mini_agent.tools import (
    ToolDefinition,
    ToolError,
    ToolRegistry,
    ToolResult,
    TranscriptBoundTool,
)


@dataclass(frozen=True)
class AgentStatus:
    """Small read-only snapshot rendered before the next terminal input."""

    runtime: RuntimeContext
    context: ContextProjection | None
    checkpoint_version: int | None
    goal_status: GoalStatus | None
    has_uncommitted_events: bool


# A turn may require several model responses.  This caps their cumulative
# execution fan-out independently from the configured provider-call budget.
MAX_TOOL_CALLS_PER_TURN = 128


class ModelCallLimitExceeded(RuntimeError):
    """Raised when one turn consumes its configured provider-call budget."""


@dataclass
class _ProviderCallBudget:
    maximum_count: int
    used_count: int = 0

    @property
    def has_capacity(self) -> bool:
        return self.used_count < self.maximum_count

    def try_consume(self, *, reserved_count: int = 0) -> bool:
        if reserved_count < 0:
            raise ValueError("Reserved model-call count cannot be negative")
        if self.maximum_count - self.used_count <= reserved_count:
            return False
        self.used_count += 1
        return True

    def consume(self) -> None:
        if not self.try_consume():
            raise ModelCallLimitExceeded(
                "Model call limit exceeded before a final response"
            )


class MiniAgent:
    def __init__(
        self,
        *,
        config: MiniAgentConfig,
        provider: ModelProvider,
        session: AgentSession,
        tools: ToolRegistry | None = None,
        artifacts: ArtifactStore | None = None,
        permissions: PermissionController | None = None,
        checkpoints: CheckpointStore | None = None,
        checkpoint_coordinator: CheckpointCoordinator | None = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self.session = session
        self._workspace_root = Path(session.metadata.workspace)
        self._tools = tools or ToolRegistry(())
        self._artifacts = artifacts
        self._permissions = permissions or PermissionController(
            session_grant_fingerprints=session.session_grant_fingerprints
        )
        self._system_prompt = SystemPromptAssembler(
            config=config.system_prompt,
            model=config.model.model,
            workspace=self._workspace_root,
            tools=self._tools.definitions,
            excluded_roots=(resolve_data_dir(config),),
        )
        self._token_estimator = Utf8TokenEstimator()
        self._request_budget = resolve_request_budget(
            model=config.model,
            context=config.context,
            capabilities=resolve_provider_capabilities(provider, config.model),
        )
        self._runtime_context = self._system_prompt.runtime_context()
        self._system_context = normalize_conversation_message_host_paths(
            self._system_prompt.assemble(
                session.goal,
                runtime=self._runtime_context,
            ),
            workspace_root=self._workspace_root,
        )
        self._validate_request_floor()
        self._checkpoints = checkpoints or CheckpointStore.open(
            session.paths.root,
            registrations=session.checkpoint_registrations,
            workspace_root=self._workspace_root,
        )
        self._checkpoint_coordinator = checkpoint_coordinator or _checkpoint_coordinator(
            config=config,
            provider=provider,
            session=session,
            store=self._checkpoints,
            request_budget=self._request_budget,
        )
        self._requires_resume = False
        self._cycle_id = session.current_cycle_id
        restored_projection = _restore_context_projection(
            session,
            self._checkpoints,
            self._request_budget,
            self._tools.definitions,
            self._system_context,
            workspace_root=self._workspace_root,
        )
        self._projection_version = restored_projection.version if restored_projection else 1
        self._projection = restored_projection
        self._projection_source_message_count = (
            _restored_projection_source_message_count(session)
            if restored_projection is not None
            else 0
        )
        self._passed_milestones: set[float] = set()

    @property
    def request_budget(self) -> RequestBudget:
        return self._request_budget

    async def run_turn(self, user_input: str) -> ModelResponse:
        self._require_resumed_session()
        goal = self.session.goal
        if goal is not None and goal.status is GoalStatus.BLOCKED:
            raise RuntimeError("Goal is blocked; use /goal resume before continuing")
        if goal is not None and goal.status in {GoalStatus.COMPLETED, GoalStatus.CANCELLED}:
            raise RuntimeError("Goal is terminal; start a new session for another task")
        stripped_input = user_input.strip()
        if not stripped_input:
            raise ValueError("User input cannot be empty")
        self._refresh_system_context()
        prospective_objective = initial_goal_objective(
            normalize_host_path_text(
                stripped_input,
                workspace_root=self._workspace_root,
            )
        )
        if self.session.goal is None:
            prospective_goal = GoalState(
                goal_id="0" * 32,
                objective=prospective_objective,
                acceptance_criteria=(),
            )
            prospective_system_context = self._system_context_for_goal(prospective_goal)
            prospective_messages = (
                prospective_system_context,
                ConversationMessage(role=MessageRole.USER, content=stripped_input),
            )
        else:
            prospective_messages = (
                *_session_request_messages(
                    self.session,
                    self._system_context,
                    workspace_root=self._workspace_root,
                ),
                ConversationMessage(role=MessageRole.USER, content=stripped_input),
            )
        self._validate_prospective_turn(prospective_messages)

        turn_id = self.session.new_turn_id()
        if self.session.goal is None:
            self.session.create_goal(
                objective=prospective_objective,
                turn_id=turn_id,
            )
        self.session.append_user_message(stripped_input, turn_id=turn_id)
        return await self._run_model_loop(turn_id)

    async def _run_model_loop(self, turn_id: str) -> ModelResponse:
        maximum_model_calls = self._config.runtime.max_model_calls_per_turn
        call_budget = _ProviderCallBudget(maximum_count=maximum_model_calls)
        used_tool_call_count = 0
        overflow_recovery_used = False
        while call_budget.has_capacity:
            messages = self._request_messages()
            request = normalize_model_request_host_paths(
                ModelRequest(
                    model=self._config.model.model,
                    messages=messages,
                    tools=self._tools.definitions,
                    max_output_tokens=self._request_budget.desired_output_tokens,
                ),
                workspace_root=self._workspace_root,
            )
            projection = await self._prepare_projection(
                request,
                turn_id=turn_id,
                call_budget=call_budget,
            )
            request = request.model_copy(update={"messages": projection.messages})
            try:
                response = await self._complete_with_retry(
                    request,
                    turn_id=turn_id,
                    call_budget=call_budget,
                    maximum_tool_call_count=MAX_TOOL_CALLS_PER_TURN - used_tool_call_count,
                )
            except ProviderError as error:
                if not error.is_context_overflow or overflow_recovery_used:
                    raise
                if not call_budget.has_capacity:
                    raise
                overflow_recovery_used = True
                projection = await self._rebuild(
                    self._full_request(),
                    turn_id=turn_id,
                    reason="provider_context_overflow",
                    call_budget=call_budget,
                    reserved_model_call_count=1,
                )
                self._projection = projection
                self._projection_source_message_count = len(
                    self._request_messages()
                )
                continue
            used_tool_call_count += len(response.tool_calls)
            try:
                path_safe_response = normalize_model_response_host_paths(
                    response,
                    workspace_root=self._workspace_root,
                )
                safe_response = _redact_model_response_at_boundary(
                    path_safe_response,
                    workspace_root=self._workspace_root,
                )
                durable_tool_calls = tuple(
                    _durable_tool_call(
                        tool_call,
                        arguments_normalized=_tool_call_arguments_changed(
                            raw_tool_call.arguments_json,
                            tool_call.arguments_json,
                        ),
                    )
                    for raw_tool_call, tool_call in zip(
                        response.tool_calls,
                        safe_response.tool_calls,
                        strict=True,
                    )
                )
                assistant_event = self.session.append_assistant_message(
                    path_safe_response.content,
                    finish_reason=path_safe_response.finish_reason,
                    tool_calls=durable_tool_calls,
                    turn_id=turn_id,
                )
            except BaseException:
                # The provider may have completed but its durable terminal is
                # now unknown.  Do not allow another request start in this
                # process; resume will close the pending request explicitly.
                self._requires_resume = True
                raise
            if not safe_response.tool_calls:
                return safe_response
            self._execute_tool_calls(
                response,
                durable_tool_calls=durable_tool_calls,
                turn_id=turn_id,
                transcript_before_event_id=assistant_event.id,
            )

        raise ModelCallLimitExceeded("Model call limit exceeded before a final response")

    async def compact(self, *, focus: str | None = None) -> ContextProjection:
        """Synchronously checkpoint and rebuild the next active request view."""

        self._require_resumed_session()
        if len(self.session.events) < 1:
            raise RuntimeError("Cannot compact an empty session")
        turn_id = self.session.new_turn_id()
        request = self._full_request()
        call_budget = _ProviderCallBudget(
            maximum_count=self._config.runtime.max_model_calls_per_turn
        )
        projection = await self._rebuild(
            request,
            turn_id=turn_id,
            reason="manual_compact",
            focus=focus,
            call_budget=call_budget,
        )
        self._projection = projection
        self._projection_source_message_count = len(request.messages)
        return projection

    async def checkpoint(self, *, focus: str | None = None) -> None:
        """Commit current durable state without switching the active cycle."""

        self._require_resumed_session()
        call_budget = _ProviderCallBudget(
            maximum_count=self._config.runtime.max_model_calls_per_turn
        )
        await self._checkpoint(
            through_event_id=self.session.events[-1].id,
            focus=focus,
            model_call_budget=call_budget,
        )

    def _require_resumed_session(self) -> None:
        if self._requires_resume:
            raise RuntimeError("Agent must be resumed before continuing")

    async def _checkpoint(
        self,
        *,
        through_event_id: int,
        focus: str | None = None,
        model_call_budget: _ProviderCallBudget | None = None,
        reserved_model_call_count: int = 0,
    ) -> Checkpoint:
        try:
            return await self._checkpoint_coordinator.checkpoint(
                through_event_id=through_event_id,
                focus=focus,
                model_call_budget=model_call_budget,
                reserved_model_call_count=reserved_model_call_count,
            )
        except BaseException:
            if self._checkpoint_coordinator.last_checkpoint_attempt_started:
                self._requires_resume = True
            raise

    @property
    def context_projection(self) -> ContextProjection | None:
        return self._projection

    def context_status(self) -> ContextProjection:
        """Estimate the next request view without mutating the active projection."""

        request = self._full_request()
        if self._projection is not None:
            messages = self._extend_projection(request.messages)
            estimate = self._token_estimator.estimate_request(
                request.model_copy(update={"messages": messages})
            )
            return self._projection.model_copy(
                update={"messages": messages, "estimate": estimate}
            )
        return project_full_history(
            request,
            budget=self._request_budget,
            estimator=self._token_estimator,
        )

    def system_context(self) -> ConversationMessage:
        """Return the exact structured system message for the next request."""

        self._refresh_system_context()
        if self._projection is not None and self._projection.messages:
            candidate = self._projection.messages[0]
            if candidate.role is MessageRole.SYSTEM:
                return candidate
        return self._system_context

    def runtime_context(self) -> RuntimeContext:
        """Return current model, workspace, platform, and Git summary for the UI."""

        self._refresh_system_context()
        return self._runtime_context

    def status(self) -> AgentStatus:
        """Return one consistent runtime/context snapshot for the terminal UI."""

        try:
            context = self.context_status()
        except ContextBudgetExceeded:
            context = None
        current_checkpoint = self._checkpoints.current
        goal = self.session.goal
        return AgentStatus(
            runtime=self._runtime_context,
            context=context,
            checkpoint_version=(
                current_checkpoint.version if current_checkpoint is not None else None
            ),
            goal_status=goal.status if goal is not None else None,
            has_uncommitted_events=(
                current_checkpoint is None
                or current_checkpoint.source_through_event_id < self.session.events[-1].id
            ),
        )

    def _full_request(self) -> ModelRequest:
        return normalize_model_request_host_paths(
            ModelRequest(
                model=self._config.model.model,
                messages=self._request_messages(),
                tools=self._tools.definitions,
                max_output_tokens=self._request_budget.desired_output_tokens,
            ),
            workspace_root=self._workspace_root,
        )

    def _request_messages(self) -> tuple[ConversationMessage, ...]:
        self._refresh_system_context()
        return _session_request_messages(
            self.session,
            self._system_context,
            workspace_root=self._workspace_root,
        )

    def _refresh_system_context(self) -> None:
        runtime_context = self._system_prompt.runtime_context()
        system_context = normalize_conversation_message_host_paths(
            self._system_prompt.assemble(
                self.session.goal,
                runtime=runtime_context,
            ),
            workspace_root=self._workspace_root,
        )
        self._runtime_context = runtime_context
        if system_context == self._system_context:
            return
        self._system_context = system_context
        self._projection = None
        self._projection_source_message_count = 0

    def _validate_request_floor(self) -> None:
        self._validate_system_request_floor(self._system_context)

    def validate_goal_request_floor(self, goal: GoalState) -> None:
        """Validate one prospective goal before its lifecycle event is written."""

        self._validate_system_request_floor(self._system_context_for_goal(goal))

    def _system_context_for_goal(self, goal: GoalState) -> ConversationMessage:
        runtime = self._system_prompt.runtime_context()
        return normalize_conversation_message_host_paths(
            self._system_prompt.assemble(goal, runtime=runtime),
            workspace_root=self._workspace_root,
        )

    def _validate_system_request_floor(
        self,
        system_context: ConversationMessage,
    ) -> None:
        validate_request_floor(
            normalize_model_request_host_paths(
                ModelRequest(
                    model=self._config.model.model,
                    messages=(system_context,),
                    tools=self._tools.definitions,
                    max_output_tokens=self._request_budget.desired_output_tokens,
                ),
                workspace_root=self._workspace_root,
            ),
            budget=self._request_budget,
            estimator=self._token_estimator,
        )

    def _validate_prospective_turn(
        self,
        messages: tuple[ConversationMessage, ...],
    ) -> None:
        validate_prospective_request_floor(
            normalize_model_request_host_paths(
                ModelRequest(
                    model=self._config.model.model,
                    messages=messages,
                    tools=self._tools.definitions,
                    max_output_tokens=self._request_budget.desired_output_tokens,
                ),
                workspace_root=self._workspace_root,
            ),
            budget=self._request_budget,
            estimator=self._token_estimator,
            maximum_checkpoint_bytes=self._config.context.checkpoint_max_bytes,
        )

    async def _prepare_projection(
        self,
        request: ModelRequest,
        *,
        turn_id: str,
        call_budget: _ProviderCallBudget,
    ) -> ContextProjection:
        if self._projection is not None:
            candidate_request = request.model_copy(
                update={"messages": self._extend_projection(request.messages)}
            )
            estimate = self._token_estimator.estimate_request(candidate_request)
            if estimate.total_tokens <= self._request_budget.request_input_limit:
                projection = self._projection.model_copy(
                    update={
                        "messages": candidate_request.messages,
                        "estimate": estimate,
                    }
                )
                if projection.utilization_ratio >= self._config.context.rebuild_ratio:
                    projection = await self._rebuild(
                        request,
                        turn_id=turn_id,
                        reason="rebuild_threshold_reached",
                        call_budget=call_budget,
                        reserved_model_call_count=1,
                    )
            else:
                projection = await self._rebuild(
                    request,
                    turn_id=turn_id,
                    reason="active_projection_exceeded_budget",
                    call_budget=call_budget,
                    reserved_model_call_count=1,
                )
        else:
            try:
                projection = project_full_history(
                    request,
                    budget=self._request_budget,
                    estimator=self._token_estimator,
                )
            except ContextBudgetExceeded:
                projection = await self._rebuild(
                    request,
                    turn_id=turn_id,
                    reason="full_history_exceeded_budget",
                    call_budget=call_budget,
                    reserved_model_call_count=1,
                )
            else:
                if projection.utilization_ratio >= self._config.context.rebuild_ratio:
                    projection = await self._rebuild(
                        request,
                        turn_id=turn_id,
                        reason="rebuild_threshold_reached",
                        call_budget=call_budget,
                        reserved_model_call_count=1,
                    )

        self._projection = projection
        self._projection_source_message_count = len(request.messages)
        self._record_projection(projection, turn_id=turn_id)
        await self._checkpoint_passed_milestone(
            projection,
            call_budget=call_budget,
        )
        return projection

    def _extend_projection(
        self,
        full_messages: tuple[ConversationMessage, ...],
    ) -> tuple[ConversationMessage, ...]:
        if self._projection is None:
            return full_messages
        return (
            *self._projection.messages,
            *full_messages[self._projection_source_message_count :],
        )

    async def _rebuild(
        self,
        request: ModelRequest,
        *,
        turn_id: str,
        reason: str,
        focus: str | None = None,
        call_budget: _ProviderCallBudget | None = None,
        reserved_model_call_count: int = 0,
    ) -> ContextProjection:
        watermark = self.session.events[-1].id
        checkpoint = await self._checkpoint(
            through_event_id=watermark,
            focus=focus,
            model_call_budget=call_budget,
            reserved_model_call_count=reserved_model_call_count,
        )
        rebuild_id = uuid4().hex
        try:
            self.session.append_rebuild_started(
                rebuild_id=rebuild_id,
                source_cycle_id=self._cycle_id,
                checkpoint_id=checkpoint.checkpoint_id,
                checkpoint_watermark=checkpoint.source_through_event_id,
                reason=reason,
                source_message_count=len(request.messages),
                turn_id=turn_id,
            )
        except BaseException:
            # A start append can have reached durable storage before failing.
            # Do not start another rebuild until resume validates the stream.
            self._requires_resume = True
            raise
        next_version = self._projection_version + 1
        try:
            projection = build_rebuild_projection(
                request,
                checkpoint=_redact_checkpoint_host_paths(
                    checkpoint,
                    workspace_root=self._workspace_root,
                ),
                budget=self._request_budget,
                estimator=self._token_estimator,
                projection_version=next_version,
            )
        except BaseException:
            # Preserve the original computation interruption.  A durable
            # failure terminal is useful when it succeeds, but a failed
            # append leaves the lifecycle uncertain and both cases require a
            # fresh resume before this process can mutate the session again.
            self._requires_resume = True
            try:
                self.session.append_rebuild_failed(
                    rebuild_id=rebuild_id,
                    error_code="rebuild_failed",
                    message="Context rebuild failed",
                    cycle_id=self._cycle_id,
                    turn_id=turn_id,
                )
            except BaseException:
                # Keep the computation exception as the caller-visible
                # failure while marking the append uncertainty explicitly.
                self._requires_resume = True
            raise
        next_cycle_id = self._cycle_id + 1
        try:
            self.session.append_rebuild_completed(
                rebuild_id=rebuild_id,
                projection_version=projection.version,
                strategy=projection.strategy,
                checkpoint_id=projection.checkpoint_id,
                checkpoint_watermark=projection.checkpoint_watermark,
                projected_message_count=len(projection.messages),
                estimated_tokens=projection.estimate.total_tokens,
                source_message_count=len(request.messages),
                cycle_id=next_cycle_id,
                turn_id=turn_id,
            )
        except BaseException:
            # Completion may have been written before an I/O failure.  Only
            # resume can determine whether the durable lifecycle is closed.
            self._requires_resume = True
            raise
        self._cycle_id = next_cycle_id
        self._projection_version = next_version
        self._passed_milestones = {
            milestone
            for milestone in self._config.context.checkpoint_milestones
            if milestone <= projection.utilization_ratio
        }
        if projection.omitted_message_count:
            self.session.append_context_pruned(
                projection_version=projection.version,
                source_message_count=len(request.messages),
                projected_message_count=len(projection.messages),
                omitted_message_count=projection.omitted_message_count,
                cycle_id=self._cycle_id,
                turn_id=turn_id,
            )
        return projection

    def _record_projection(self, projection: ContextProjection, *, turn_id: str) -> None:
        self.session.append_context_estimated(
            projection_version=projection.version,
            strategy=projection.strategy,
            input_limit=projection.input_limit,
            estimated_tokens=projection.estimate.total_tokens,
            utilization_ratio=projection.utilization_ratio,
            checkpoint_id=projection.checkpoint_id,
            cycle_id=self._cycle_id,
            turn_id=turn_id,
        )

    async def _checkpoint_passed_milestone(
        self,
        projection: ContextProjection,
        *,
        call_budget: _ProviderCallBudget,
    ) -> None:
        passed = [
            milestone
            for milestone in self._config.context.checkpoint_milestones
            if milestone <= projection.utilization_ratio
            and milestone not in self._passed_milestones
        ]
        if not passed:
            return
        self._passed_milestones.update(passed)
        await self._checkpoint(
            through_event_id=self.session.events[-1].id,
            model_call_budget=call_budget,
            reserved_model_call_count=1,
        )

    async def _complete_with_retry(
        self,
        request: ModelRequest,
        *,
        turn_id: str,
        call_budget: _ProviderCallBudget,
        maximum_tool_call_count: int = MAX_TOOL_CALLS_PER_TURN,
    ) -> ModelResponse:
        request = normalize_model_request_host_paths(
            request,
            workspace_root=self._workspace_root,
        )
        maximum_attempts = self._config.runtime.provider_retry_count + 1
        terminal_error: ProviderError | None = None

        for attempt in range(1, maximum_attempts + 1):
            call_budget.consume()
            try:
                self.session.append_model_request_started(
                    turn_id=turn_id,
                    message_count=len(request.messages),
                    attempt=attempt,
                )
            except BaseException:
                # A failed append can still have reached durable storage.  A
                # restart validates and closes any resulting active request.
                self._requires_resume = True
                raise
            try:
                response = await self._provider.complete(request)
                _validate_model_response_tool_call_count(
                    response,
                    maximum_count=maximum_tool_call_count,
                )
            except ProviderError as error:
                try:
                    safe_error = safe_provider_error(error)
                    self.session.append_model_request_failed(
                        safe_error,
                        attempt=attempt,
                        turn_id=turn_id,
                    )
                except BaseException:
                    # We cannot prove that this request reached a durable
                    # terminal.  Preserve the append/redaction failure and
                    # require session recovery before any new request.
                    self._requires_resume = True
                    raise
                if (
                    not safe_error.is_retryable
                    or attempt == maximum_attempts
                    or not call_budget.has_capacity
                ):
                    terminal_error = safe_error
                    break
                await asyncio.sleep(self._config.runtime.retry_backoff_seconds * attempt)
                continue
            except BaseException:
                # Providers are required to use ProviderError, but a buggy
                # adapter, cancellation, or KeyboardInterrupt still occurs
                # after request_started.  Keep the original exception and
                # leave recovery to append the stable interrupted terminal.
                self._requires_resume = True
                raise

            return response

        if terminal_error is not None:
            # Raise after leaving the provider exception handler so callers
            # cannot inspect the original exception through __context__.
            raise terminal_error
        raise RuntimeError("Provider retry loop ended without a result")

    def _execute_tool_calls(
        self,
        response: ModelResponse,
        *,
        durable_tool_calls: tuple[ToolCall, ...],
        turn_id: str,
        transcript_before_event_id: int,
    ) -> None:
        for tool_call, durable_tool_call in zip(
            response.tool_calls,
            durable_tool_calls,
            strict=True,
        ):
            try:
                self._execute_tool_call(
                    tool_call,
                    durable_tool_call=durable_tool_call,
                    turn_id=turn_id,
                    transcript_before_event_id=transcript_before_event_id,
                )
            except BaseException:
                self._requires_resume = True
                raise

    def _execute_tool_call(
        self,
        tool_call: ToolCall,
        *,
        durable_tool_call: ToolCall,
        turn_id: str,
        transcript_before_event_id: int,
    ) -> None:
        try:
            tool = self._tools.get(tool_call.name)
        except ToolError as error:
            self.session.append_tool_interrupted(
                tool_call,
                recovery_status=ToolRecoveryStatus.INTERRUPTED,
                reason=normalize_host_path_text(
                    str(error),
                    workspace_root=self._workspace_root,
                ),
                turn_id=turn_id,
            )
            return

        self.session.append_tool_requested(
            durable_tool_call,
            is_read_only=tool.definition.is_read_only,
            turn_id=turn_id,
        )
        preflight = getattr(tool, "preflight", None)
        if callable(preflight):
            try:
                preflight(tool_call.arguments_json)
            except ToolError as error:
                self.session.append_tool_failed(
                    tool_call,
                    error_code=_safe_tool_error_code(error.code),
                    message=normalize_host_path_text(
                        str(error),
                        workspace_root=self._workspace_root,
                    ),
                    before_start=True,
                    turn_id=turn_id,
                )
                return
        try:
            arguments_normalized = _tool_call_arguments_changed(
                tool_call.arguments_json,
                durable_tool_call.arguments_json,
            )
            built_in_patch_grant = (
                built_in_apply_patch_session_grant(tool, tool_call.arguments_json)
                if not arguments_normalized
                else None
            )
            approval_request = self._permissions.request_for(
                tool.definition,
                durable_tool_call,
                built_in_apply_patch_grant=built_in_patch_grant,
            )
        except (ApprovalArgumentsTooLargeError, ValueError) as error:
            self.session.append_tool_failed(
                tool_call,
                error_code=(
                    "approval_scope_too_large"
                    if isinstance(error, ApprovalArgumentsTooLargeError)
                    else "invalid_approval_scope"
                ),
                message=normalize_host_path_text(
                    str(error),
                    workspace_root=self._workspace_root,
                ),
                before_start=True,
                turn_id=turn_id,
            )
            return
        if approval_request is not None:
            self.session.append_approval_requested(
                tool_call=tool_call,
                redacted_arguments=approval_request.redacted_arguments,
                scope_descriptor=approval_request.scope_descriptor,
                scope_fingerprint=approval_request.scope_fingerprint,
                can_allow_session=approval_request.can_allow_session,
                turn_id=turn_id,
            )
            decision = self._permissions.decide(approval_request)
            self.session.append_approval_resolved(
                tool_call=tool_call,
                scope_fingerprint=approval_request.scope_fingerprint,
                decision=decision,
                turn_id=turn_id,
            )
            self._permissions.record(approval_request, decision)
            if decision is ApprovalDecision.DENY:
                self.session.append_tool_failed(
                    tool_call,
                    error_code="permission_denied",
                    message="User denied this tool call",
                    before_start=True,
                    turn_id=turn_id,
                )
                return
        started_event = self.session.append_tool_started(tool_call, turn_id=turn_id)
        try:
            if isinstance(tool, TranscriptBoundTool):
                result = tool.execute_with_transcript_bound(
                    tool_call.arguments_json,
                    before_event_id=transcript_before_event_id,
                )
            else:
                result = tool.execute(tool_call.arguments_json)
        except ToolError as error:
            if not tool.definition.is_read_only and error.code in {
                "command_timeout",
                "write_status_unknown",
            }:
                self._revoke_unknown_scope(approval_request)
                self.session.append_tool_interrupted(
                    tool_call,
                    recovery_status=ToolRecoveryStatus.UNKNOWN,
                    reason="Tool stopped after starting; side effects may be incomplete",
                    turn_id=turn_id,
                )
                raise RuntimeError(
                    "Tool status is unknown after execution; review before continuing"
                ) from None
            self.session.append_tool_failed(
                tool_call,
                error_code=_safe_tool_error_code(error.code),
                message=normalize_host_path_text(
                    str(error),
                    workspace_root=self._workspace_root,
                ),
                turn_id=turn_id,
            )
            return
        result = _redact_tool_result_at_boundary(
            result,
            workspace_root=self._workspace_root,
        )
        output, artifact_id, source_redaction_summary = self._persist_large_tool_result(
            result,
            source_event_id=started_event.id,
            turn_id=turn_id,
        )
        self.session.append_tool_completed(
            tool_call,
            output,
            artifact_id=artifact_id,
            source_redaction_summary=source_redaction_summary,
            facts=result.facts,
            turn_id=turn_id,
        )

    def _revoke_unknown_scope(self, approval_request: ApprovalRequest | None) -> None:
        if approval_request is not None:
            self._permissions.revoke(approval_request.scope_fingerprint)

    def _persist_large_tool_result(
        self,
        result: ToolResult,
        *,
        source_event_id: int,
        turn_id: str,
    ) -> tuple[str, str | None, RedactionSummary]:
        maximum_chars = self._config.runtime.inline_tool_result_max_chars
        if len(result.content) <= maximum_chars:
            return result.render_for_context(), None, result.source_redaction_summary
        if self._artifacts is None:
            return (
                result.preview_for_context(maximum_chars),
                None,
                result.source_redaction_summary,
            )

        record = self._artifacts.create(
            result.content,
            source_event_id=source_event_id,
            source_redaction_summary=result.source_redaction_summary,
            truncated_at_start=result.is_truncated,
            truncated_at_end=result.is_truncated,
        )
        self.session.append_artifact_created(record, turn_id=turn_id)
        reference = record.context_reference(preview_chars=maximum_chars)
        # The artifact registration already carries the source redaction total.
        # Keep the terminal tool event focused on the safe artifact reference.
        return result.render_for_context(reference), record.artifact_id, RedactionSummary()


def _redact_tool_result_at_boundary(
    result: ToolResult,
    *,
    workspace_root: Path,
) -> ToolResult:
    """Redact once more while the tool's real truncation boundary is known."""

    path_redaction = redact_host_paths(
        result.content,
        workspace_root=workspace_root,
        truncated_at_start=result.is_truncated,
        truncated_at_end=result.is_truncated,
    )
    redaction = redact_text(
        path_redaction.text,
        truncated_at_end=result.is_truncated,
    )
    safe_modified_paths: list[str] = []
    modified_path_match_count = 0
    for path in result.facts.modified_paths:
        path_result = redact_host_paths(path, workspace_root=workspace_root)
        safe_modified_paths.append(path_result.text)
        modified_path_match_count += path_result.match_count
    source_summary = result.source_redaction_summary
    combined_summary = RedactionSummary(
        match_count=(
            source_summary.match_count
            + path_redaction.match_count
            + modified_path_match_count
            + redaction.match_count
        ),
        kinds=tuple(
            sorted(
                {
                    *source_summary.kinds,
                    *path_redaction.matched_kinds,
                    *(
                        (RedactionKind.HOST_PATH,)
                        if modified_path_match_count
                        else ()
                    ),
                    *redaction.matched_kinds,
                },
                key=lambda kind: kind.value,
            )
        ),
    )
    return result.model_copy(
        update={
            "content": redaction.text,
            "source_redaction_summary": combined_summary,
            "facts": result.facts.model_copy(
                update={"modified_paths": tuple(safe_modified_paths)}
            ),
        }
    )


def _redact_model_response_at_boundary(
    response: ModelResponse,
    *,
    workspace_root: Path,
) -> ModelResponse:
    if response.content is None:
        return response
    content = response.content
    path_redaction = redact_host_paths(
        content,
        workspace_root=workspace_root,
        truncated_at_end=response.finish_reason in {
            FinishReason.LENGTH,
            FinishReason.OTHER,
        },
    )
    redaction = redact_text(
        path_redaction.text,
        truncated_at_end=response.finish_reason in {
            FinishReason.LENGTH,
            FinishReason.OTHER,
        },
    )
    return response.model_copy(update={"content": redaction.text})


def _validate_model_response_tool_call_count(
    response: ModelResponse,
    *,
    maximum_count: int,
) -> None:
    """Defend the agent boundary when an adapter bypasses Pydantic validation."""

    try:
        tool_call_count = len(response.tool_calls)
    except TypeError as error:
        raise ProviderError(
            "invalid_provider_response",
            "Model provider returned invalid tool calls",
            is_retryable=False,
        ) from error
    if (
        tool_call_count > MAX_TOOL_CALLS_PER_RESPONSE
        or tool_call_count > maximum_count
    ):
        raise ProviderError(
            "invalid_provider_response",
            "Model provider returned too many tool calls",
            is_retryable=False,
        )


def _tool_call_without_session_grant(
    tool_call: ToolCall,
    *,
    arguments_normalized: bool,
) -> ToolCall:
    """Persist a durable marker so redacted host paths cannot receive a grant."""

    if not arguments_normalized:
        return tool_call
    arguments_json = tool_call.arguments_json.replace(
        "<workspace-root>",
        "<workspace-root>[REDACTED]",
    ).replace(
        "<home>",
        "<home>[REDACTED]",
    )
    return tool_call.model_copy(update={"arguments_json": arguments_json})


def _durable_tool_call(
    tool_call: ToolCall,
    *,
    arguments_normalized: bool,
) -> ToolCall:
    """Create the safe audit representation while preserving raw execution input."""

    durable_tool_call = _tool_call_without_session_grant(
        tool_call,
        arguments_normalized=arguments_normalized,
    )
    if durable_tool_call.name != "exec_command":
        return durable_tool_call
    safe_arguments = sanitize_exec_command_arguments(durable_tool_call.arguments_json)
    return durable_tool_call.model_copy(update={"arguments_json": safe_arguments.arguments_json})


def _tool_call_arguments_changed(raw_arguments: str, safe_arguments: str) -> bool:
    """Ignore JSON formatting changes but retain security-relevant mutations."""

    try:
        return json.loads(raw_arguments) != json.loads(safe_arguments)
    except json.JSONDecodeError:
        # Public ToolCall validation rules out this path. Treat a malformed
        # model-constructed value conservatively if a caller bypasses it.
        return raw_arguments != safe_arguments


def _checkpoint_coordinator(
    *,
    config: MiniAgentConfig,
    provider: ModelProvider,
    session: AgentSession,
    store: CheckpointStore,
    request_budget: RequestBudget,
) -> CheckpointCoordinator:
    deterministic = DeterministicCheckpointExtractor(
        maximum_bytes=config.context.checkpoint_max_bytes,
    )
    extractor = deterministic
    if config.context.checkpoint_model is not None:
        from mini_agent.checkpoint_model import ModelCheckpointExtractor

        primary = ModelCheckpointExtractor(
            provider=provider,
            model=config.context.checkpoint_model,
            maximum_output_tokens=config.context.checkpoint_max_tokens,
            maximum_checkpoint_bytes=config.context.checkpoint_max_bytes,
            maximum_input_tokens=request_budget.rebuild_seed_tokens,
            workspace_root=Path(session.metadata.workspace),
        )
        extractor = FallbackCheckpointExtractor(primary=primary, fallback=deterministic)
    return CheckpointCoordinator(
        session=session,
        store=store,
        extractor=extractor,
    )


def _restore_context_projection(
    session: AgentSession,
    checkpoints: CheckpointStore,
    budget: RequestBudget,
    tools: tuple[ToolDefinition, ...],
    system_context: ConversationMessage,
    *,
    workspace_root: Path,
) -> ContextProjection | None:
    from mini_agent.events import RebuildCompletedData

    completed = next(
        (
            event.data
            for event in reversed(session.events)
            if isinstance(event.data, RebuildCompletedData)
        ),
        None,
    )
    if completed is None:
        return None
    checkpoint = next(
        (
            item
            for item in reversed(checkpoints.checkpoints)
            if item.checkpoint_id == completed.checkpoint_id
        ),
        None,
    )
    if checkpoint is None:
        return None
    request = normalize_model_request_host_paths(
        ModelRequest(
            model=session.metadata.model,
            messages=_session_request_messages(
                session,
                system_context,
                workspace_root=workspace_root,
            )[: completed.source_message_count],
            tools=tools,
            max_output_tokens=budget.desired_output_tokens,
        ),
        workspace_root=workspace_root,
    )
    return build_rebuild_projection(
        request,
        checkpoint=_redact_checkpoint_host_paths(
            checkpoint,
            workspace_root=workspace_root,
        ),
        budget=budget,
        estimator=Utf8TokenEstimator(),
        projection_version=completed.projection_version,
    )


def _restored_projection_source_message_count(session: AgentSession) -> int:
    from mini_agent.events import RebuildCompletedData

    return next(
        (
            event.data.source_message_count
            for event in reversed(session.events)
            if isinstance(event.data, RebuildCompletedData)
        ),
        0,
    )


def _session_request_messages(
    session: AgentSession,
    system_context: ConversationMessage,
    *,
    workspace_root: Path,
) -> tuple[ConversationMessage, ...]:
    return tuple(
        normalize_conversation_message_host_paths(message, workspace_root=workspace_root)
        for message in (system_context, *session.conversation_messages())
    )


_TOOL_ERROR_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _safe_tool_error_code(value: str) -> str:
    if _TOOL_ERROR_CODE_PATTERN.fullmatch(value) is None:
        return "invalid_tool_error_code"
    if redact_text(value).match_count:
        return "invalid_tool_error_code"
    return value


def _redact_checkpoint_host_paths(
    checkpoint: Checkpoint,
    *,
    workspace_root: Path,
) -> Checkpoint:
    safe_json = redact_host_paths(
        checkpoint.model_dump_json(),
        workspace_root=workspace_root,
    ).text
    return Checkpoint.model_validate_json(safe_json)
