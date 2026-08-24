"""Single-writer coordination for incremental checkpoint candidates."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import Literal, Protocol, cast

from mini_agent.checkpoint import (
    Checkpoint,
    CheckpointItem,
    CheckpointRecord,
    CheckpointStore,
)
from mini_agent.events import (
    ArtifactCreatedData,
    AssistantMessageData,
    CheckpointCommittedData,
    GoalCreatedData,
    GoalStatusChangedData,
    GoalUpdatedData,
    StoredEvent,
    ToolCompletedData,
    ToolFailedData,
    ToolInterruptedData,
    UserMessageData,
)
from mini_agent.session import AgentSession


class CheckpointExtractor(Protocol):
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
    ) -> Checkpoint: ...


class ModelCallBudget(Protocol):
    """Narrow budget consumed only when an extractor calls a provider."""

    def try_consume(self, *, reserved_count: int = 0) -> bool: ...


class CheckpointRecoveryRequiredError(RuntimeError):
    """Raised after a checkpoint may have committed but was not activated locally."""

    def __init__(self) -> None:
        super().__init__("Checkpoint recovery requires resuming the session")


class CheckpointCoordinator:
    """Serializes checkpoint commits and coalesces concurrent watermarks."""

    def __init__(
        self,
        *,
        session: AgentSession,
        store: CheckpointStore,
        extractor: CheckpointExtractor,
    ) -> None:
        self._session = session
        self._store = store
        self._extractor = extractor
        self._lock = asyncio.Lock()
        self._requested_watermark = 0
        self._is_poisoned = False
        self._checkpoint_attempt_started: ContextVar[bool] = ContextVar(
            "checkpoint_attempt_started",
            default=False,
        )

    @property
    def last_checkpoint_attempt_started(self) -> bool:
        """Whether this task's latest call crossed the checkpoint start boundary."""

        return self._checkpoint_attempt_started.get()

    async def checkpoint(
        self,
        *,
        through_event_id: int,
        focus: str | None = None,
        model_call_budget: ModelCallBudget | None = None,
        reserved_model_call_count: int = 0,
    ) -> Checkpoint:
        self._checkpoint_attempt_started.set(False)
        self._raise_if_poisoned()
        if through_event_id < 1:
            raise ValueError("Checkpoint watermark must be positive")
        self._requested_watermark = max(self._requested_watermark, through_event_id)

        async with self._lock:
            self._raise_if_poisoned()
            while True:
                current = self._store.current
                target = self._requested_watermark
                if current is not None and target <= current.source_through_event_id:
                    return current
                checkpoint = await self._write_one(
                    through_event_id=target,
                    focus=focus,
                    model_call_budget=model_call_budget,
                    reserved_model_call_count=reserved_model_call_count,
                )
                if self._requested_watermark <= checkpoint.source_through_event_id:
                    return checkpoint

    async def _write_one(
        self,
        *,
        through_event_id: int,
        focus: str | None,
        model_call_budget: ModelCallBudget | None,
        reserved_model_call_count: int,
    ) -> Checkpoint:
        previous = self._store.current
        source_from_event_id = (
            previous.source_through_event_id + 1 if previous is not None else 1
        )
        checkpoint_id = CheckpointStore.new_checkpoint_id()
        version = previous.version + 1 if previous is not None else 1
        source_events = tuple(
            event
            for event in self._session.events
            if source_from_event_id <= event.id <= through_event_id
        )
        if not source_events or source_events[-1].id != through_event_id:
            raise ValueError("Checkpoint watermark is outside the durable transcript")

        self._checkpoint_attempt_started.set(True)
        try:
            self._session.append_checkpoint_started(
                checkpoint_id=checkpoint_id,
                source_from_event_id=source_from_event_id,
                source_through_event_id=through_event_id,
            )
        except BaseException:
            # An append error may have left a durable start event behind.
            self._is_poisoned = True
            raise
        try:
            checkpoint = await self._extractor.extract(
                previous=previous,
                events=source_events,
                source_from_event_id=source_from_event_id,
                source_through_event_id=through_event_id,
                checkpoint_id=checkpoint_id,
                version=version,
                focus=focus,
                model_call_budget=model_call_budget,
                reserved_model_call_count=reserved_model_call_count,
            )
            _validate_extracted_checkpoint(
                checkpoint,
                session_id=self._session.metadata.session_id,
                checkpoint_id=checkpoint_id,
                version=version,
                source_from_event_id=source_from_event_id,
                source_through_event_id=through_event_id,
            )
        except BaseException:
            self._record_precommit_failure(checkpoint_id)
            raise

        try:
            record = self._store.stage(checkpoint)
        except BaseException:
            # Stage can have written content or an index before its failure.
            self._is_poisoned = True
            raise

        try:
            registration = self._commit(record)
            return self._store.activate(registration)
        except BaseException as error:
            # The commit append can fail after the event has reached durable storage.
            # Do not add a failed marker or attempt another version in this process.
            self._is_poisoned = True
            if not isinstance(error, Exception):
                raise
            raise CheckpointRecoveryRequiredError() from error

    def _record_precommit_failure(self, checkpoint_id: str) -> None:
        """Record a known-safe pre-commit terminal marker, or fail-stop on ambiguity."""

        try:
            self._session.append_checkpoint_failed(
                checkpoint_id=checkpoint_id,
                error_code="checkpoint_generation_failed",
                message="Checkpoint generation or validation failed",
            )
        except BaseException:
            # A failed terminal append may itself have reached durable storage.
            self._is_poisoned = True

    def _raise_if_poisoned(self) -> None:
        if self._is_poisoned:
            raise CheckpointRecoveryRequiredError()

    def _commit(self, record: CheckpointRecord) -> CheckpointCommittedData:
        event = self._session.append_checkpoint_committed(
            checkpoint_id=record.checkpoint_id,
            version=record.version,
            source_through_event_id=record.source_through_event_id,
            content_hash=record.content_hash,
        )
        data = event.data
        if not isinstance(data, CheckpointCommittedData):
            raise AssertionError("Checkpoint commit appended the wrong event type")
        return data


class FallbackCheckpointExtractor:
    """Use deterministic extraction only for an expected model extraction failure."""

    def __init__(
        self,
        *,
        primary: CheckpointExtractor,
        fallback: CheckpointExtractor,
    ) -> None:
        self._primary = primary
        self._fallback = fallback

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
        from mini_agent.checkpoint_model import CheckpointExtractionError

        try:
            return await self._primary.extract(
                previous=previous,
                events=events,
                source_from_event_id=source_from_event_id,
                source_through_event_id=source_through_event_id,
                checkpoint_id=checkpoint_id,
                version=version,
                focus=focus,
                model_call_budget=model_call_budget,
                reserved_model_call_count=reserved_model_call_count,
            )
        except CheckpointExtractionError:
            return await self._fallback.extract(
                previous=previous,
                events=events,
                source_from_event_id=source_from_event_id,
                source_through_event_id=source_through_event_id,
                checkpoint_id=checkpoint_id,
                version=version,
                focus=focus,
                model_call_budget=model_call_budget,
                reserved_model_call_count=reserved_model_call_count,
            )


class DeterministicCheckpointExtractor:
    """Safe fallback extractor that never calls a model or workspace tool."""

    def __init__(
        self,
        writer_model: str = "deterministic-v1",
        maximum_bytes: int = 65_536,
    ) -> None:
        if maximum_bytes < 4_096:
            raise ValueError("Checkpoint storage budget must be at least 4096 bytes")
        self._writer_model = writer_model
        self._maximum_bytes = maximum_bytes

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
        user_messages = [
            event.data.content for event in events if isinstance(event.data, UserMessageData)
        ]
        intent = user_messages[-1] if user_messages else None
        current_intent = (
            focus
            or intent
            or (previous.current_intent if previous is not None else None)
            or "Continue the active session"
        )
        completed = list(previous.completed if previous is not None else ())
        active_work = list(previous.active_work if previous is not None else ())
        relevant_files = list(previous.relevant_files if previous is not None else ())
        errors = list(previous.errors_and_fixes if previous is not None else ())
        artifacts = list(previous.artifact_references if previous is not None else ())
        runtime_state = list(previous.runtime_state if previous is not None else ())
        notes = list(previous.miscellaneous_notes if previous is not None else ())
        acceptance_criteria = list(
            previous.acceptance_criteria if previous is not None else ()
        )
        # Model-owned state the deterministic extractor cannot derive from raw
        # events. Carry it forward from the previous checkpoint so a fallback
        # extraction does not silently drop constraints, blockers, the task
        # tree, planned next actions, cross-task findings, or key decisions.
        constraints = list(previous.constraints_and_preferences if previous is not None else ())
        task_tree = list(previous.task_tree if previous is not None else ())
        blocked = list(previous.blocked if previous is not None else ())
        next_actions = list(previous.next_actions if previous is not None else ())
        cross_task_findings = list(previous.cross_task_findings if previous is not None else ())
        key_decisions = list(previous.key_decisions if previous is not None else ())

        for event in events:
            data = event.data
            if isinstance(data, UserMessageData):
                active_work = [_item(data.content, event.id)]
            elif isinstance(data, GoalCreatedData | GoalUpdatedData):
                current_intent = data.objective
                acceptance_criteria = [
                    _item(criterion.description, event.id)
                    for criterion in data.acceptance_criteria
                ]
            elif isinstance(data, GoalStatusChangedData):
                runtime_state.append(_item(f"Goal status: {data.status.value}", event.id))
            elif isinstance(data, AssistantMessageData) and data.content:
                notes.append(_item(data.content, event.id))
            elif isinstance(data, ToolCompletedData):
                runtime_state.append(_item(f"Tool {data.tool_call_id} completed", event.id))
            elif isinstance(data, ToolFailedData | ToolInterruptedData):
                errors.append(_item(f"Tool {data.tool_call_id}: {data.kind}", event.id))
            elif isinstance(data, ArtifactCreatedData):
                artifacts.append(_item(f"Artifact {data.artifact_id}", event.id))

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            session_id=events[0].session_id,
            cycle_id=events[-1].cycle_id,
            version=version,
            source_from_event_id=source_from_event_id,
            source_through_event_id=source_through_event_id,
            current_intent=current_intent,
            acceptance_criteria=_bounded_items(acceptance_criteria),
            completed=_bounded_items(completed),
            active_work=_bounded_items(active_work),
            relevant_files=_bounded_items(relevant_files),
            errors_and_fixes=_bounded_items(errors),
            artifact_references=_bounded_items(artifacts),
            runtime_state=_bounded_items(runtime_state),
            miscellaneous_notes=_bounded_items(notes),
            constraints_and_preferences=_bounded_items(constraints),
            task_tree=_bounded_items(task_tree),
            blocked=_bounded_items(blocked),
            next_actions=_bounded_items(next_actions),
            cross_task_findings=_bounded_items(cross_task_findings),
            key_decisions=_bounded_items(key_decisions),
            writer_model=self._writer_model,
            created_at=events[-1].timestamp,
        )
        return _fit_checkpoint(checkpoint, self._maximum_bytes)


def _validate_extracted_checkpoint(
    checkpoint: Checkpoint,
    *,
    session_id: str,
    checkpoint_id: str,
    version: int,
    source_from_event_id: int,
    source_through_event_id: int,
) -> None:
    actual = (
        checkpoint.session_id,
        checkpoint.checkpoint_id,
        checkpoint.version,
        checkpoint.source_from_event_id,
        checkpoint.source_through_event_id,
    )
    expected = (
        session_id,
        checkpoint_id,
        version,
        source_from_event_id,
        source_through_event_id,
    )
    if actual != expected:
        raise ValueError("Checkpoint extractor returned mismatched identity or watermark")


def _item(text: str, event_id: int) -> CheckpointItem:
    bounded_text = text[:1_000].strip() or "Recorded event"
    return CheckpointItem(text=bounded_text, source_event_ids=(event_id,))


def _bounded_items(items: list[CheckpointItem]) -> tuple[CheckpointItem, ...]:
    return tuple(items[-5:])


CheckpointItemField = Literal[
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
]

_LOW_PRIORITY_ITEM_FIELDS: tuple[CheckpointItemField, ...] = (
    "miscellaneous_notes",
    "completed",
    "runtime_state",
    "artifact_references",
    "errors_and_fixes",
    "relevant_files",
    "cross_task_findings",
    "key_decisions",
    "next_actions",
    "task_tree",
    "constraints_and_preferences",
    "blocked",
)
_TEXT_COMPRESSION_ITEM_FIELDS: tuple[CheckpointItemField, ...] = (
    "acceptance_criteria",
    "active_work",
    *_LOW_PRIORITY_ITEM_FIELDS,
)
_LAST_RESORT_ITEM_FIELDS: tuple[CheckpointItemField, ...] = (
    *_LOW_PRIORITY_ITEM_FIELDS,
    "active_work",
    "acceptance_criteria",
)


def _fit_checkpoint(checkpoint: Checkpoint, maximum_bytes: int) -> Checkpoint:
    """Fit canonical JSON by bytes while preserving useful provenance when possible."""

    current = checkpoint
    while _checkpoint_size_bytes(current) > maximum_bytes:
        reduced = _drop_oldest_item(current, _LOW_PRIORITY_ITEM_FIELDS)
        if reduced is not None:
            current = reduced
            continue

        shortened = _shorten_checkpoint_text(current)
        if shortened is not None:
            current = shortened
            continue

        reduced = _drop_oldest_item(current, _LAST_RESORT_ITEM_FIELDS)
        if reduced is not None:
            current = reduced
            continue

        raise ValueError("Checkpoint cannot fit its configured output budget")
    return current


def _checkpoint_size_bytes(checkpoint: Checkpoint) -> int:
    return len(checkpoint.model_dump_json().encode("utf-8"))


def _drop_oldest_item(
    checkpoint: Checkpoint,
    fields: tuple[CheckpointItemField, ...],
) -> Checkpoint | None:
    for field in fields:
        items = cast(tuple[CheckpointItem, ...], getattr(checkpoint, field))
        if items:
            return checkpoint.model_copy(update={field: items[1:]})
    return None


def _shorten_checkpoint_text(checkpoint: Checkpoint) -> Checkpoint | None:
    """Shorten the largest remaining text atom without splitting UTF-8 characters."""

    candidates: list[tuple[int, int, str, int | None, str]] = [
        (
            len(checkpoint.current_intent.encode("utf-8")),
            0,
            "current_intent",
            None,
            checkpoint.current_intent,
        ),
        (
            len(checkpoint.writer_model.encode("utf-8")),
            1,
            "writer_model",
            None,
            checkpoint.writer_model,
        ),
    ]
    for field_priority, field in enumerate(_TEXT_COMPRESSION_ITEM_FIELDS, start=2):
        items = cast(tuple[CheckpointItem, ...], getattr(checkpoint, field))
        candidates.extend(
            (
                len(item.text.encode("utf-8")),
                field_priority,
                field,
                item_index,
                item.text,
            )
            for item_index, item in enumerate(items)
        )

    original_size = _checkpoint_size_bytes(checkpoint)
    for _byte_count, _priority, field, item_index, text in sorted(
        candidates,
        key=lambda candidate: (-candidate[0], candidate[1], candidate[3] or -1),
    ):
        shortened_text = _shorten_utf8_text(text)
        if shortened_text == text:
            continue
        if item_index is None:
            candidate = checkpoint.model_copy(update={field: shortened_text})
        else:
            items = list(cast(tuple[CheckpointItem, ...], getattr(checkpoint, field)))
            items[item_index] = items[item_index].model_copy(update={"text": shortened_text})
            candidate = checkpoint.model_copy(update={field: tuple(items)})
        if _checkpoint_size_bytes(candidate) < original_size:
            return candidate
    return None


def _shorten_utf8_text(text: str) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= 1:
        return "." if text != "." else text
    maximum_bytes = max(1, len(encoded) // 2)
    if maximum_bytes < 4:
        return "."
    prefix = encoded[: maximum_bytes - len("…".encode())].decode(
        "utf-8",
        errors="ignore",
    )
    if not prefix:
        return "."
    return f"{prefix}…"
