"""One-turn, non-interactive agent runs for scripting and benchmarking.

The headless mode runs a single task through the same durable session,
workspace, approval, and redaction boundaries as the interactive CLI, then
reports mechanical facts (turn outcome, tool activity, provider-reported token
usage) for the caller to judge.  Task success itself is not evaluated here:
that belongs to the caller's verification step against the workspace.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mini_agent.agent import MiniAgent
from mini_agent.artifacts import ArtifactStore
from mini_agent.config import MiniAgentConfig
from mini_agent.events import (
    AssistantMessageData,
    StoredEvent,
    ToolFailedData,
    ToolRequestedData,
)
from mini_agent.goal import GoalStatus
from mini_agent.messages import TokenUsage
from mini_agent.permissions import (
    AllowOncePrompt,
    PermissionController,
)
from mini_agent.provider import ModelProvider
from mini_agent.session import AgentSession
from mini_agent.tools.registry import ToolRegistry


class HeadlessResult(BaseModel):
    """Machine-readable summary of one headless agent turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    model: str
    stop_reason: Literal["completed", "user_interrupt"]
    duration_seconds: float = Field(ge=0)
    response_text: str | None = None
    tool_calls: int = Field(ge=0)
    tool_failures: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    goal_status: GoalStatus | None = None


async def run_headless_task(
    *,
    config: MiniAgentConfig,
    provider: ModelProvider,
    session: AgentSession,
    tools: ToolRegistry,
    task: str,
    auto_approve: bool = False,
    artifacts: ArtifactStore | None = None,
    response_sanitizer: Callable[[str], str] | None = None,
) -> HeadlessResult:
    """Run one task in one turn and report mechanical facts about the run.

    The turn loop, goal creation, checkpointing, and crash recovery behave
    exactly as in the interactive CLI.  Without ``auto_approve`` a non-read-only
    tool action is denied, because there is no interactive approver; read-only
    tools still run without approval, exactly as in the interactive CLI.
    """

    stripped_task = task.strip()
    if not stripped_task:
        raise ValueError("Headless task cannot be empty")

    agent = MiniAgent(
        config=config,
        provider=provider,
        session=session,
        tools=tools,
        artifacts=artifacts,
        permissions=PermissionController(
            prompt=AllowOncePrompt() if auto_approve else None,
            session_grant_fingerprints=session.session_grant_fingerprints,
        ),
    )
    started_at = time.perf_counter()
    previous_event_count = len(session.events)
    stop_reason, response_text = await _run_single_turn(agent, stripped_task)
    return _build_result(
        session=session,
        events=session.events[previous_event_count:],
        model=config.model.model,
        stop_reason=stop_reason,
        response_text=response_text,
        duration_seconds=time.perf_counter() - started_at,
        response_sanitizer=response_sanitizer,
    )


async def _run_single_turn(
    agent: MiniAgent, task: str
) -> tuple[Literal["completed", "user_interrupt"], str | None]:
    """Run one turn; Ctrl-C becomes a clean interrupt instead of an error."""

    try:
        response = await agent.run_turn(task)
    except asyncio.CancelledError:
        # asyncio delivers Ctrl-C inside the coroutine as CancelledError.  The
        # session stays finalizable by the caller, mirroring the interactive
        # CLI's interrupted-turn handling.
        return "user_interrupt", None
    except KeyboardInterrupt:
        return "user_interrupt", None
    return "completed", response.content


def _build_result(
    *,
    session: AgentSession,
    events: tuple[StoredEvent, ...],
    model: str,
    stop_reason: Literal["completed", "user_interrupt"],
    response_text: str | None,
    duration_seconds: float,
    response_sanitizer: Callable[[str], str] | None,
) -> HeadlessResult:
    tool_calls = sum(
        1 for event in events if isinstance(event.data, ToolRequestedData)
    )
    tool_failures = sum(
        1 for event in events if isinstance(event.data, ToolFailedData)
    )
    usage = _sum_usage(events)
    response_text = (
        response_sanitizer(response_text)
        if response_text is not None and response_sanitizer is not None
        else response_text
    )
    return HeadlessResult(
        session_id=session.metadata.session_id,
        model=model,
        stop_reason=stop_reason,
        duration_seconds=round(duration_seconds, 3),
        response_text=response_text,
        tool_calls=tool_calls,
        tool_failures=tool_failures,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        goal_status=session.goal.status if session.goal is not None else None,
    )


def _sum_usage(events: tuple[StoredEvent, ...]) -> TokenUsage:
    usages = [
        event.data.usage or TokenUsage()
        for event in events
        if isinstance(event.data, AssistantMessageData)
    ]
    return TokenUsage(
        prompt_tokens=_sum_reported_counts(usage.prompt_tokens for usage in usages),
        completion_tokens=_sum_reported_counts(usage.completion_tokens for usage in usages),
        total_tokens=_sum_reported_counts(usage.total_tokens for usage in usages),
    )


def _sum_reported_counts(counts: Iterable[int | None]) -> int | None:
    """Incomplete accounting must stay unknown rather than look like a total."""

    total: int | None = None
    for count in counts:
        if count is None:
            return None
        total = (total or 0) + count
    return total
