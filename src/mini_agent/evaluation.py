"""Reproducible, evidence-backed evaluation of one validated session."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mini_agent.checkpoint import Checkpoint, CheckpointStore
from mini_agent.events import (
    CheckpointCommittedData,
    CheckpointFailedData,
    ContextEstimatedData,
    ContextPrunedData,
    RebuildCompletedData,
    RebuildFailedData,
    StoredEvent,
    ToolInterruptedData,
    ToolRecoveryStatus,
)
from mini_agent.goal import GoalStatus
from mini_agent.session import AgentSession


class AtomCategory(StrEnum):
    CRITICAL_CONSTRAINT = "critical_constraint"
    EXACT_DETAIL = "exact_detail"
    COMPLETED_STATE = "completed_state"
    ACTIVE_STATE = "active_state"
    BLOCKED_STATE = "blocked_state"
    SUPERSEDED_DECISION = "superseded_decision"
    RETRIEVAL_TARGET = "retrieval_target"
    ACCEPTANCE_CRITERION = "acceptance_criterion"
    COMPLETION_EVIDENCE = "completion_evidence"


class InformationAtom(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    atom_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
    )
    category: AtomCategory
    text: str = Field(min_length=1, max_length=1_000)
    source_event_ids: tuple[int, ...] = Field(min_length=1, max_length=32)
    is_current: bool = True

    @model_validator(mode="after")
    def require_sorted_unique_source_event_ids(self) -> Self:
        if tuple(sorted(set(self.source_event_ids))) != self.source_event_ids:
            raise ValueError("Information atom source event ids must be sorted and unique")
        if any(event_id < 1 for event_id in self.source_event_ids):
            raise ValueError("Information atom source event ids must be positive")
        if self.category is AtomCategory.SUPERSEDED_DECISION and self.is_current:
            raise ValueError("A superseded decision cannot be current")
        return self


class EvaluationFixture(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    fixture_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
    )
    description: str = Field(min_length=1, max_length=1_000)
    atoms: tuple[InformationAtom, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def require_unique_atom_ids(self) -> Self:
        ids = [atom.atom_id for atom in self.atoms]
        if len(set(ids)) != len(ids):
            raise ValueError("Evaluation fixture atom ids must be unique")
        return self


class Availability(StrEnum):
    OBSERVED = "observed"
    UNAVAILABLE = "unavailable"


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    fixture_id: str
    session_id: str
    event_count: int = Field(ge=1)
    cycle_count: int = Field(ge=1)
    checkpoint_count: int = Field(ge=0)
    rebuild_count: int = Field(ge=0)
    prune_count: int = Field(ge=0)
    recovery_count: int = Field(ge=0)
    estimated_request_tokens: int = Field(ge=0)
    critical_constraint_exact_recall: float = Field(ge=0, le=1)
    state_precision: float = Field(ge=0, le=1)
    state_recall: float = Field(ge=0, le=1)
    state_micro_f1: float = Field(ge=0, le=1)
    stale_fact_rate: float = Field(ge=0, le=1)
    exact_detail_recall: float = Field(ge=0, le=1)
    retrieval_success: float = Field(ge=0, le=1)
    false_completion_rate: float = Field(ge=0, le=1)
    end_task_success: bool
    recovery_success: bool
    passed_acceptance_thresholds: bool
    missing_atom_ids: tuple[str, ...]
    stale_atom_ids: tuple[str, ...]
    history_recoverable_atom_ids: tuple[str, ...]
    provider_usage_status: Availability = Availability.UNAVAILABLE
    provider_reported_input_tokens: int | None = Field(default=None, ge=0)
    provider_reported_output_tokens: int | None = Field(default=None, ge=0)
    cost_status: Availability = Availability.UNAVAILABLE
    cost_usd: float | None = Field(default=None, ge=0)
    latency_status: Availability = Availability.UNAVAILABLE
    latency_seconds: float | None = Field(default=None, ge=0)
    human_evaluation_status: Availability = Availability.UNAVAILABLE
    human_score: float | None = Field(default=None, ge=0, le=1)


def load_evaluation_fixture(path: Path) -> EvaluationFixture:
    """Load a versioned fixture without accepting a directory or non-JSON shape."""

    try:
        return EvaluationFixture.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        raise ValueError("Evaluation fixture is unreadable or invalid") from error


def evaluate_session(
    session: AgentSession,
    checkpoints: CheckpointStore,
    fixture: EvaluationFixture,
) -> EvaluationReport:
    """Measure retention and recovery from durable state only."""

    checkpoint_text = _checkpoint_text(checkpoints.current)
    event_text = "\n".join(event.model_dump_json() for event in session.events)
    _validate_fixture_provenance(fixture, session.events)
    retention_categories = {
        AtomCategory.CRITICAL_CONSTRAINT,
        AtomCategory.EXACT_DETAIL,
        AtomCategory.COMPLETED_STATE,
        AtomCategory.ACTIVE_STATE,
        AtomCategory.BLOCKED_STATE,
        AtomCategory.ACCEPTANCE_CRITERION,
    }
    missing = tuple(
        atom.atom_id
        for atom in fixture.atoms
        if atom.is_current
        and atom.category in retention_categories
        and atom.text not in checkpoint_text
    )
    stale = tuple(
        atom.atom_id
        for atom in fixture.atoms
        if not atom.is_current and atom.text in checkpoint_text
    )
    history_recoverable = tuple(
        atom.atom_id
        for atom in fixture.atoms
        if atom.is_current and atom.text in event_text
    )

    critical = _atoms(fixture, AtomCategory.CRITICAL_CONSTRAINT, current=True)
    exact = _atoms(fixture, AtomCategory.EXACT_DETAIL, current=True)
    retrieval = _atoms(fixture, AtomCategory.RETRIEVAL_TARGET, current=True)
    expected_states = tuple(
        atom
        for atom in fixture.atoms
        if atom.is_current
        and atom.category in {
            AtomCategory.COMPLETED_STATE,
            AtomCategory.ACTIVE_STATE,
            AtomCategory.BLOCKED_STATE,
        }
    )
    predicted_state_ids = {
        atom.atom_id
        for atom in fixture.atoms
        if atom.category in {
            AtomCategory.COMPLETED_STATE,
            AtomCategory.ACTIVE_STATE,
            AtomCategory.BLOCKED_STATE,
        }
        and atom.text in checkpoint_text
    }
    expected_state_ids = {atom.atom_id for atom in expected_states}
    true_state_ids = predicted_state_ids & expected_state_ids
    state_precision = _ratio(len(true_state_ids), len(predicted_state_ids), empty=1.0)
    state_recall = _ratio(len(true_state_ids), len(expected_state_ids), empty=1.0)
    state_micro_f1 = _f1(state_precision, state_recall)
    goal = session.goal
    completed = goal is not None and goal.status is GoalStatus.COMPLETED
    completion_criteria = _atoms(fixture, AtomCategory.COMPLETION_EVIDENCE, current=True)
    false_completion_rate = 1.0 if completed and any(
        atom.text not in event_text for atom in completion_criteria
    ) else 0.0
    recovery_success = _recovery_succeeded(session.events)
    critical_recall = _retention_ratio(critical, checkpoint_text)
    exact_recall = _retention_ratio(exact, checkpoint_text)
    retrieval_success = _retention_ratio(retrieval, event_text)
    superseded = _atoms(fixture, AtomCategory.SUPERSEDED_DECISION, current=False)
    stale_fact_rate = _ratio(
        sum(atom.text in checkpoint_text for atom in superseded),
        len(superseded),
        empty=0.0,
    )
    estimated_request_tokens = sum(
        event.data.estimated_tokens
        for event in session.events
        if isinstance(event.data, ContextEstimatedData)
    )
    end_task_success = completed and false_completion_rate == 0.0
    passed = (
        session.current_cycle_id >= 4
        and critical_recall == 1.0
        and state_micro_f1 >= 0.95
        and stale_fact_rate == 0.0
        and false_completion_rate == 0.0
        and retrieval_success == 1.0
        and recovery_success
    )
    return EvaluationReport(
        fixture_id=fixture.fixture_id,
        session_id=session.metadata.session_id,
        event_count=len(session.events),
        cycle_count=session.current_cycle_id,
        checkpoint_count=sum(
            isinstance(event.data, CheckpointCommittedData) for event in session.events
        ),
        rebuild_count=sum(
            isinstance(event.data, RebuildCompletedData) for event in session.events
        ),
        prune_count=sum(isinstance(event.data, ContextPrunedData) for event in session.events),
        recovery_count=sum(
            isinstance(event.data, ToolInterruptedData) for event in session.events
        ),
        estimated_request_tokens=estimated_request_tokens,
        critical_constraint_exact_recall=critical_recall,
        state_precision=state_precision,
        state_recall=state_recall,
        state_micro_f1=state_micro_f1,
        stale_fact_rate=stale_fact_rate,
        exact_detail_recall=exact_recall,
        retrieval_success=retrieval_success,
        false_completion_rate=false_completion_rate,
        end_task_success=end_task_success,
        recovery_success=recovery_success,
        passed_acceptance_thresholds=passed,
        missing_atom_ids=missing,
        stale_atom_ids=stale,
        history_recoverable_atom_ids=history_recoverable,
    )


def write_evaluation_report(report: EvaluationReport, path: Path) -> None:
    """Write one deterministic report; caller chooses the task-owned location."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(f"{payload}\n", encoding="utf-8")


def _checkpoint_text(checkpoint: Checkpoint | None) -> str:
    if checkpoint is None:
        return ""
    values = [checkpoint.current_intent, *(item.text for item in checkpoint.items)]
    return "\n".join(values)


def _validate_fixture_provenance(
    fixture: EvaluationFixture,
    events: tuple[StoredEvent, ...],
) -> None:
    event_by_id = {event.id: event for event in events}
    for atom in fixture.atoms:
        if any(event_id not in event_by_id for event_id in atom.source_event_ids):
            raise ValueError(f"Information atom source is missing: {atom.atom_id}")


def _atoms(
    fixture: EvaluationFixture,
    category: AtomCategory,
    *,
    current: bool,
) -> tuple[InformationAtom, ...]:
    return tuple(
        atom
        for atom in fixture.atoms
        if atom.category is category and atom.is_current is current
    )


def _retention_ratio(atoms: tuple[InformationAtom, ...], text: str) -> float:
    return _ratio(sum(atom.text in text for atom in atoms), len(atoms), empty=1.0)


def _ratio(numerator: int, denominator: int, *, empty: float) -> float:
    return numerator / denominator if denominator else empty


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _recovery_succeeded(events: tuple[StoredEvent, ...]) -> bool:
    if any(isinstance(event.data, RebuildFailedData) for event in events):
        return False
    if any(isinstance(event.data, CheckpointFailedData) for event in events):
        return False
    unknown_ids = {
        event.data.tool_call_id
        for event in events
        if isinstance(event.data, ToolInterruptedData)
        and event.data.recovery_status is ToolRecoveryStatus.UNKNOWN
    }
    return all(
        not (
            event.data.kind == "tool_started"
            and getattr(event.data, "tool_call_id", None) in unknown_ids
            and any(
                later.id > event.id
                and later.data.kind == "tool_started"
                and getattr(later.data, "tool_call_id", None)
                == getattr(event.data, "tool_call_id", None)
                for later in events
            )
        )
        for event in events
    )
