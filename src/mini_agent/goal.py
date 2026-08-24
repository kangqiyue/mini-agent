"""Durable goal state and mechanical completion evidence checks."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from mini_agent.events import StoredEvent


class GoalStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class EvidenceKind(StrEnum):
    ASSISTANT_RESPONSE = "assistant_response"
    TOOL_COMPLETED = "tool_completed"
    COMMAND_SUCCEEDED = "command_succeeded"
    FILE_MODIFIED = "file_modified"
    ARTIFACT_CREATED = "artifact_created"


class AcceptanceCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
    )
    description: str = Field(min_length=1, max_length=512)
    evidence_kind: EvidenceKind


class CompletionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
    )
    event_ids: tuple[int, ...] = Field(min_length=1, max_length=32)
    note: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def require_sorted_unique_event_ids(self) -> Self:
        if tuple(sorted(set(self.event_ids))) != self.event_ids:
            raise ValueError("Completion evidence event ids must be sorted and unique")
        return self


class GoalState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    goal_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    objective: str = Field(min_length=1, max_length=4_096)
    acceptance_criteria: tuple[AcceptanceCriterion, ...] = Field(max_length=16)
    status: GoalStatus = GoalStatus.ACTIVE
    completion_evidence: tuple[CompletionEvidence, ...] = ()
    blocked_reason: str | None = Field(default=None, min_length=1, max_length=2_048)

    @model_validator(mode="after")
    def require_consistent_state(self) -> Self:
        criterion_ids = [item.criterion_id for item in self.acceptance_criteria]
        if len(set(criterion_ids)) != len(criterion_ids):
            raise ValueError("Goal acceptance criterion ids must be unique")
        evidence_ids = [item.criterion_id for item in self.completion_evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("Goal completion evidence must name each criterion once")
        if set(evidence_ids) - set(criterion_ids):
            raise ValueError("Goal completion evidence names an unknown criterion")
        if self.status is GoalStatus.COMPLETED and not self.acceptance_criteria:
            raise ValueError("A goal without acceptance criteria cannot be completed")
        if self.status is GoalStatus.BLOCKED and self.blocked_reason is None:
            raise ValueError("A blocked goal requires a reason")
        if self.status is not GoalStatus.BLOCKED and self.blocked_reason is not None:
            raise ValueError("Only a blocked goal can carry a blocked reason")
        if self.status is not GoalStatus.COMPLETED and self.completion_evidence:
            raise ValueError("Only a completed goal can carry completion evidence")
        return self


class CompletionGap(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=512)
    criterion_id: str | None = None


class CompletionGateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    is_satisfied: bool
    gaps: tuple[CompletionGap, ...]

    @model_validator(mode="after")
    def require_result_to_match_gaps(self) -> Self:
        if self.is_satisfied == bool(self.gaps):
            raise ValueError("Completion result and gaps disagree")
        return self


class GoalLifecycleError(ValueError):
    """Raised when persisted goal events cannot form one valid state."""


class GoalCompletionRejected(ValueError):
    def __init__(self, result: CompletionGateResult) -> None:
        self.result = result
        super().__init__("Goal completion evidence is incomplete")


GOAL_CONTEXT_OBJECTIVE_MAX_CHARS = 1_024


def validate_goal_lifecycle(events: tuple[StoredEvent, ...]) -> GoalState | None:
    """Reconstruct one goal and reject invalid transitions or completion claims."""

    from mini_agent.events import GoalCreatedData, GoalStatusChangedData, GoalUpdatedData

    state: GoalState | None = None
    for index, event in enumerate(events):
        data = event.data
        if isinstance(data, GoalCreatedData):
            if state is not None:
                raise GoalLifecycleError("Session contains more than one goal")
            state = _validated_goal_state(
                {
                    "goal_id": data.goal_id,
                    "objective": data.objective,
                    "acceptance_criteria": data.acceptance_criteria,
                }
            )
            continue
        if isinstance(data, GoalUpdatedData):
            state = _require_matching_active_goal(state, data.goal_id)
            state = _validated_goal_state(
                {
                    **state.model_dump(),
                    "objective": data.objective,
                    "acceptance_criteria": data.acceptance_criteria,
                }
            )
            continue
        if not isinstance(data, GoalStatusChangedData):
            continue
        state = _require_matching_goal(state, data.goal_id)
        if state.status in {GoalStatus.COMPLETED, GoalStatus.CANCELLED}:
            raise GoalLifecycleError("Terminal goal status cannot transition")
        if not _can_transition(state.status, data.status):
            raise GoalLifecycleError("Invalid goal status transition")
        candidate = _validated_goal_state(
            {
                **state.model_dump(),
                "status": data.status,
                "completion_evidence": data.completion_evidence,
                "blocked_reason": data.blocked_reason,
            }
        )
        if data.status is GoalStatus.COMPLETED:
            result = evaluate_completion_gate(candidate, events[:index])
            if not result.is_satisfied:
                raise GoalLifecycleError("Completed goal does not satisfy its evidence gate")
        state = candidate
    return state


def render_goal_context(goal: GoalState) -> str:
    """Render authoritative goal state for the main model, not for persistence."""

    objective = goal.objective
    if len(objective) > GOAL_CONTEXT_OBJECTIVE_MAX_CHARS:
        objective = (
            f"{objective[:GOAL_CONTEXT_OBJECTIVE_MAX_CHARS]}\n"
            "[objective view truncated; use history_read for the exact request]"
        )
    lines = [
        "Runtime-owned goal state:",
        f"Status: {goal.status.value}",
        f"Objective: {objective}",
    ]
    if goal.acceptance_criteria:
        lines.append("Acceptance criteria:")
        lines.extend(
            f"- {item.criterion_id} [{item.evidence_kind.value}]: {item.description}"
            for item in goal.acceptance_criteria
        )
    else:
        lines.append("Acceptance criteria: not defined; completion is not allowed.")
    lines.append(
        "Do not claim completion unless the runtime Completion Gate accepts durable evidence."
    )
    return "\n".join(lines)


def _validated_goal_state(values: object) -> GoalState:
    try:
        return GoalState.model_validate(values)
    except ValueError as error:
        raise GoalLifecycleError("Goal state violates its schema") from error


def _can_transition(current: GoalStatus, target: GoalStatus) -> bool:
    if current is GoalStatus.ACTIVE:
        return target in {
            GoalStatus.COMPLETED,
            GoalStatus.BLOCKED,
            GoalStatus.CANCELLED,
        }
    if current is GoalStatus.BLOCKED:
        return target in {GoalStatus.ACTIVE, GoalStatus.CANCELLED}
    return False


def evaluate_completion_gate(
    goal: GoalState,
    events: tuple[StoredEvent, ...],
    *,
    evidence: tuple[CompletionEvidence, ...] | None = None,
) -> CompletionGateResult:
    """Check explicit evidence against already-durable, mechanically typed events."""

    candidate_evidence = evidence if evidence is not None else goal.completion_evidence
    evidence_by_criterion = {item.criterion_id: item for item in candidate_evidence}
    gaps: list[CompletionGap] = []
    if not goal.acceptance_criteria:
        gaps.append(
            CompletionGap(
                code="acceptance_criteria_missing",
                message="Define at least one acceptance criterion before completion.",
            )
        )
    event_by_id = {event.id: event for event in events}
    tool_names = _tool_names_by_call_id(events)
    goal_definition_event_id = _goal_definition_event_id(events, goal.goal_id)
    for criterion in goal.acceptance_criteria:
        item = evidence_by_criterion.get(criterion.criterion_id)
        if item is None:
            gaps.append(
                CompletionGap(
                    code="evidence_missing",
                    criterion_id=criterion.criterion_id,
                    message="No completion evidence was supplied for this criterion.",
                )
            )
            continue
        observed_events = tuple(
            event_by_id[event_id]
            for event_id in item.event_ids
            if event_id in event_by_id
        )
        if len(observed_events) != len(item.event_ids):
            gaps.append(
                CompletionGap(
                    code="evidence_event_missing",
                    criterion_id=criterion.criterion_id,
                    message="At least one referenced evidence event does not exist.",
                )
            )
            continue
        if any(event.id <= goal_definition_event_id for event in observed_events):
            gaps.append(
                CompletionGap(
                    code="evidence_event_stale",
                    criterion_id=criterion.criterion_id,
                    message="Evidence must follow the current goal definition.",
                )
            )
            continue
        if not _matches_evidence_kind(criterion.evidence_kind, observed_events, tool_names):
            gaps.append(
                CompletionGap(
                    code="evidence_kind_mismatch",
                    criterion_id=criterion.criterion_id,
                    message=(
                        "Referenced events do not prove the required mechanical outcome: "
                        f"{criterion.evidence_kind.value}."
                    ),
                )
            )
    unknown_evidence = set(evidence_by_criterion) - {
        item.criterion_id for item in goal.acceptance_criteria
    }
    if unknown_evidence:
        gaps.append(
            CompletionGap(
                code="unknown_criterion_evidence",
                message="Completion evidence names a criterion that is not in the goal.",
            )
        )
    return CompletionGateResult(is_satisfied=not gaps, gaps=tuple(gaps))


def _require_matching_goal(state: GoalState | None, goal_id: str) -> GoalState:
    if state is None:
        raise GoalLifecycleError("Goal event appeared before goal_created")
    if state.goal_id != goal_id:
        raise GoalLifecycleError("Goal event references a different goal")
    return state


def _require_matching_active_goal(state: GoalState | None, goal_id: str) -> GoalState:
    state = _require_matching_goal(state, goal_id)
    if state.status is not GoalStatus.ACTIVE:
        raise GoalLifecycleError("Only an active goal can be updated")
    return state


def _tool_names_by_call_id(events: tuple[StoredEvent, ...]) -> dict[str, str]:
    from mini_agent.events import ToolRequestedData

    return {
        event.data.tool_call.id: event.data.tool_call.name
        for event in events
        if isinstance(event.data, ToolRequestedData)
    }


def _goal_definition_event_id(events: tuple[StoredEvent, ...], goal_id: str) -> int:
    from mini_agent.events import GoalCreatedData, GoalUpdatedData

    return max(
        (
            event.id
            for event in events
            if isinstance(event.data, GoalCreatedData | GoalUpdatedData)
            and event.data.goal_id == goal_id
        ),
        default=0,
    )


def _matches_evidence_kind(
    kind: EvidenceKind,
    events: tuple[StoredEvent, ...],
    tool_names: dict[str, str],
) -> bool:
    from mini_agent.events import ArtifactCreatedData, AssistantMessageData, ToolCompletedData

    if kind is EvidenceKind.ASSISTANT_RESPONSE:
        return any(
            isinstance(event.data, AssistantMessageData) and event.data.content
            for event in events
        )
    if kind is EvidenceKind.ARTIFACT_CREATED:
        return any(isinstance(event.data, ArtifactCreatedData) for event in events)
    completed = tuple(event.data for event in events if isinstance(event.data, ToolCompletedData))
    if kind is EvidenceKind.TOOL_COMPLETED:
        return bool(completed)
    if kind is EvidenceKind.COMMAND_SUCCEEDED:
        return any(
            tool_names.get(data.tool_call_id) == "exec_command"
            and (
                data.facts.command_exit_code == 0
                or (
                    data.legacy_facts_missing
                    and data.output.startswith("Exit code: 0\n")
                )
            )
            for data in completed
        )
    if kind is EvidenceKind.FILE_MODIFIED:
        return any(
            tool_names.get(data.tool_call_id) == "apply_patch"
            and (
                bool(data.facts.modified_paths)
                or (
                    data.legacy_facts_missing
                    and data.output.startswith("Modified files:\n")
                )
            )
            for data in completed
        )
    return False
