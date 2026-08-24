"""Restricted model-backed extraction of typed checkpoint state."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mini_agent.checkpoint import Checkpoint, CheckpointItem
from mini_agent.checkpoint_writer import ModelCallBudget
from mini_agent.context import Utf8TokenEstimator
from mini_agent.events import StoredEvent
from mini_agent.host_path_redaction import redact_host_paths
from mini_agent.messages import ConversationMessage, FinishReason, MessageRole, ModelRequest
from mini_agent.provider import ModelProvider, ProviderError, safe_provider_error
from mini_agent.redaction import redact_json_text


class CheckpointExtractionError(RuntimeError):
    """Safe failure raised when a checkpoint model cannot produce valid state."""


class CheckpointDraft(BaseModel):
    """Model-owned fields; identity and watermarks remain runtime-owned."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    current_intent: str = Field(min_length=1, max_length=8_000)
    acceptance_criteria: tuple[CheckpointItem, ...] = ()
    constraints_and_preferences: tuple[CheckpointItem, ...] = ()
    task_tree: tuple[CheckpointItem, ...] = ()
    completed: tuple[CheckpointItem, ...] = ()
    active_work: tuple[CheckpointItem, ...] = ()
    blocked: tuple[CheckpointItem, ...] = ()
    next_actions: tuple[CheckpointItem, ...] = ()
    relevant_files: tuple[CheckpointItem, ...] = ()
    cross_task_findings: tuple[CheckpointItem, ...] = ()
    errors_and_fixes: tuple[CheckpointItem, ...] = ()
    runtime_state: tuple[CheckpointItem, ...] = ()
    key_decisions: tuple[CheckpointItem, ...] = ()
    artifact_references: tuple[CheckpointItem, ...] = ()
    miscellaneous_notes: tuple[CheckpointItem, ...] = ()


class ModelCheckpointExtractor:
    """Calls one provider without tools and validates a bounded JSON response."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        model: str,
        maximum_output_tokens: int,
        maximum_checkpoint_bytes: int,
        maximum_input_tokens: int,
        workspace_root: Path,
    ) -> None:
        if (
            maximum_output_tokens < 1_024
            or maximum_checkpoint_bytes < 4_096
            or maximum_input_tokens < 1
        ):
            raise ValueError("Checkpoint model budgets are too small")
        self._provider = provider
        self._model = model
        self._maximum_output_tokens = maximum_output_tokens
        self._maximum_checkpoint_bytes = maximum_checkpoint_bytes
        self._maximum_input_tokens = maximum_input_tokens
        self._workspace_root = workspace_root
        self._token_estimator = Utf8TokenEstimator()

    async def extract(
        self,
        *,
        previous: Checkpoint | None,
        events: tuple[StoredEvent, ...],
        source_from_event_id: int,
        source_through_event_id: int,
        checkpoint_id: str,
        version: int,
        focus: str | None,
        model_call_budget: ModelCallBudget | None = None,
        reserved_model_call_count: int = 0,
    ) -> Checkpoint:
        user_content = _writer_input(
            previous=previous,
            events=events,
            focus=focus,
            maximum_input_tokens=self._maximum_input_tokens,
            model=self._model,
            maximum_output_tokens=self._maximum_output_tokens,
            workspace_root=self._workspace_root,
            estimator=self._token_estimator,
        )
        request = _checkpoint_request(
            model=self._model,
            user_content=user_content,
            maximum_output_tokens=self._maximum_output_tokens,
        )
        if model_call_budget is not None and not model_call_budget.try_consume(
            reserved_count=reserved_model_call_count
        ):
            raise CheckpointExtractionError(
                "Checkpoint provider call was skipped to preserve the turn model-call budget"
            )
        try:
            response = await self._provider.complete(request)
        except ProviderError as error:
            safe_error = safe_provider_error(error)
            raise CheckpointExtractionError(
                f"Checkpoint provider failed [{safe_error.code}]"
            ) from None
        if (
            response.content is None
            or response.tool_calls
            or response.finish_reason is not FinishReason.STOP
        ):
            raise CheckpointExtractionError("Checkpoint provider returned incomplete output")
        if len(response.content.encode("utf-8")) > self._maximum_checkpoint_bytes:
            raise CheckpointExtractionError(
                "Checkpoint provider output exceeds its storage budget"
            )
        try:
            safe_json = redact_json_text(response.content).text
            draft = CheckpointDraft.model_validate_json(safe_json)
            checkpoint = Checkpoint(
                checkpoint_id=checkpoint_id,
                session_id=events[0].session_id,
                cycle_id=events[-1].cycle_id,
                version=version,
                source_from_event_id=source_from_event_id,
                source_through_event_id=source_through_event_id,
                writer_model=self._model,
                created_at=datetime.now(UTC),
                **draft.model_dump(),
            )
        except (ValidationError, ValueError):
            raise CheckpointExtractionError(
                "Checkpoint provider returned invalid structured state"
            ) from None
        encoded_checkpoint = checkpoint.model_dump_json().encode("utf-8")
        if len(encoded_checkpoint) > self._maximum_checkpoint_bytes:
            raise CheckpointExtractionError(
                "Checkpoint structured state exceeds its storage budget"
            )
        return checkpoint


_SYSTEM_PROMPT = """You are a restricted checkpoint writer. Return exactly one JSON object.
Do not call tools. Preserve current facts, constraints, failures, decisions, file paths, and next
actions. Do not mark plans or failed attempts completed. Every item must cite source_event_ids.
When uncertain, say unknown or needs_verification. Never include credentials. The JSON fields are:
current_intent and the arrays acceptance_criteria, constraints_and_preferences, task_tree,
completed, active_work, blocked, next_actions, relevant_files, cross_task_findings,
errors_and_fixes, runtime_state, key_decisions, artifact_references, miscellaneous_notes.
Each array item is {"text": string, "source_event_ids": [positive integers]}."""


def _writer_input(
    *,
    previous: Checkpoint | None,
    events: tuple[StoredEvent, ...],
    focus: str | None,
    maximum_input_tokens: int,
    model: str,
    maximum_output_tokens: int,
    workspace_root: Path,
    estimator: Utf8TokenEstimator,
) -> str:
    if not events:
        raise CheckpointExtractionError("Checkpoint writer requires source events")
    header = {
        "focus": focus,
        "previous_checkpoint": previous.model_dump(mode="json") if previous else None,
        "source_from_event_id": events[0].id,
        "source_through_event_id": events[-1].id,
    }
    event_strings: list[str] = []
    for event in reversed(events):
        summary = _bounded_event_summary(
            event,
            workspace_root=workspace_root,
        )
        candidate_events = [summary, *event_strings]
        candidate = _safe_writer_payload(
            header=header,
            event_strings=candidate_events,
            workspace_root=workspace_root,
        )
        if not _checkpoint_input_fits(
            candidate,
            model=model,
            maximum_output_tokens=maximum_output_tokens,
            maximum_input_tokens=maximum_input_tokens,
            estimator=estimator,
        ):
            if not event_strings:
                fitted_summary = _fit_single_event_summary(
                    summary,
                    header=header,
                    model=model,
                    maximum_output_tokens=maximum_output_tokens,
                    maximum_input_tokens=maximum_input_tokens,
                    workspace_root=workspace_root,
                    estimator=estimator,
                )
                if fitted_summary is not None:
                    event_strings.append(fitted_summary)
            break
        event_strings.insert(0, summary)
    if not event_strings:
        raise CheckpointExtractionError(
            "Checkpoint writer input cannot fit a source event"
        )
    payload = _safe_writer_payload(
        header=header,
        event_strings=event_strings,
        workspace_root=workspace_root,
    )
    if not _checkpoint_input_fits(
        payload,
        model=model,
        maximum_output_tokens=maximum_output_tokens,
        maximum_input_tokens=maximum_input_tokens,
        estimator=estimator,
    ):
        raise CheckpointExtractionError("Checkpoint writer input cannot fit its budget")
    return payload


_MAXIMUM_EVENT_SUMMARY_CHARS = 2_000
_MINIMUM_EVENT_SUMMARY_CHARS = 256
_EVENT_TRUNCATED_NOTICE = "...[event truncated to fit checkpoint input]"


def _bounded_event_summary(event: StoredEvent, *, workspace_root: Path) -> str:
    path_safe_summary = redact_host_paths(
        event.model_dump_json(),
        workspace_root=workspace_root,
    ).text
    summary = redact_json_text(path_safe_summary).text
    if len(summary) <= _MAXIMUM_EVENT_SUMMARY_CHARS:
        return summary
    retained_chars = _MAXIMUM_EVENT_SUMMARY_CHARS - len(_EVENT_TRUNCATED_NOTICE)
    return f"{summary[:retained_chars]}{_EVENT_TRUNCATED_NOTICE}"


def _safe_writer_payload(
    *,
    header: Mapping[str, object],
    event_strings: list[str],
    workspace_root: Path,
) -> str:
    payload = json.dumps(
        {**header, "events_oldest_first": event_strings},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    path_safe_payload = redact_host_paths(
        payload,
        workspace_root=workspace_root,
    ).text
    return redact_json_text(path_safe_payload).text


def _fit_single_event_summary(
    summary: str,
    *,
    header: Mapping[str, object],
    model: str,
    maximum_output_tokens: int,
    maximum_input_tokens: int,
    workspace_root: Path,
    estimator: Utf8TokenEstimator,
) -> str | None:
    maximum_prefix_chars = max(0, len(summary) - len(_EVENT_TRUNCATED_NOTICE))
    if maximum_prefix_chars < _MINIMUM_EVENT_SUMMARY_CHARS:
        return None
    lower = _MINIMUM_EVENT_SUMMARY_CHARS
    upper = maximum_prefix_chars
    best: str | None = None
    while lower <= upper:
        midpoint = (lower + upper) // 2
        candidate_summary = f"{summary[:midpoint]}{_EVENT_TRUNCATED_NOTICE}"
        payload = _safe_writer_payload(
            header=header,
            event_strings=[candidate_summary],
            workspace_root=workspace_root,
        )
        if _checkpoint_input_fits(
            payload,
            model=model,
            maximum_output_tokens=maximum_output_tokens,
            maximum_input_tokens=maximum_input_tokens,
            estimator=estimator,
        ):
            best = candidate_summary
            lower = midpoint + 1
        else:
            upper = midpoint - 1
    return best


def _checkpoint_input_fits(
    user_content: str,
    *,
    model: str,
    maximum_output_tokens: int,
    maximum_input_tokens: int,
    estimator: Utf8TokenEstimator,
) -> bool:
    request = _checkpoint_request(
        model=model,
        user_content=user_content,
        maximum_output_tokens=maximum_output_tokens,
    )
    return estimator.estimate_request(request).total_tokens <= maximum_input_tokens


def _checkpoint_request(
    *,
    model: str,
    user_content: str,
    maximum_output_tokens: int,
) -> ModelRequest:
    return ModelRequest(
        model=model,
        messages=(
            ConversationMessage(role=MessageRole.SYSTEM, content=_SYSTEM_PROMPT),
            ConversationMessage(role=MessageRole.USER, content=user_content),
        ),
        tools=(),
        max_output_tokens=maximum_output_tokens,
    )
