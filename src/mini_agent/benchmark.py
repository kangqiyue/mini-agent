"""Deterministic three-task acceptance benchmark using the real agent/tool loop."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from mini_agent.agent import MiniAgent
from mini_agent.checkpoint import CheckpointStore
from mini_agent.config import ContextConfig, MiniAgentConfig, ModelConfig, RuntimeConfig
from mini_agent.evaluation import (
    AtomCategory,
    Availability,
    EvaluationFixture,
    EvaluationReport,
    InformationAtom,
    evaluate_session,
)
from mini_agent.events import GoalCreatedData, GoalStatusChangedData, ToolCompletedData
from mini_agent.goal import AcceptanceCriterion, CompletionEvidence, EvidenceKind
from mini_agent.messages import FinishReason, ModelRequest, ModelResponse, ToolCall
from mini_agent.permissions import (
    ApprovalDecision,
    ApprovalPrompt,
    ApprovalRequest,
    PermissionController,
)
from mini_agent.session import AgentSession
from mini_agent.tools.apply_patch import ApplyPatchTool
from mini_agent.tools.exec_command import ExecCommandTool
from mini_agent.tools.read_file import ReadFileTool
from mini_agent.tools.registry import ToolRegistry
from mini_agent.workspace import Workspace


class BenchmarkTaskReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1, max_length=64)
    success: bool
    duration_seconds: float = Field(ge=0)
    evaluation: EvaluationReport


class BenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    benchmark_id: str = "v0_1_deterministic_coding"
    task_count: int = Field(ge=1)
    passed_task_count: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    tasks: tuple[BenchmarkTaskReport, ...]
    provider_usage_status: Availability = Availability.UNAVAILABLE
    cost_status: Availability = Availability.UNAVAILABLE
    human_evaluation_status: Availability = Availability.UNAVAILABLE
    limitations: tuple[str, ...] = (
        "The provider is deterministic; this report does not measure stochastic model quality.",
        "Provider token usage and price data are unavailable, so cost is not estimated.",
        "Human evaluation was not supplied.",
    )

    @model_validator(mode="after")
    def require_aggregate_to_match_tasks(self) -> Self:
        if self.task_count != len(self.tasks):
            raise ValueError("Benchmark task count does not match task reports")
        if self.passed_task_count != sum(item.success for item in self.tasks):
            raise ValueError("Benchmark pass count does not match task reports")
        expected_rate = self.passed_task_count / self.task_count
        if self.success_rate != expected_rate:
            raise ValueError("Benchmark success rate does not match task reports")
        return self


class _TaskDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    path: str
    before: str
    after: str
    verification_code: str
    constraint: str
    omitted_detail: str
    superseded_decision: str


_TASKS = (
    _TaskDefinition(
        task_id="fix_increment",
        path="counter.py",
        before="def increment(value: int) -> int:\n    return value - 1\n",
        after="def increment(value: int) -> int:\n    return value + 1\n",
        verification_code=(
            "from counter import increment; assert increment(2) == 3; "
            "print('required-suite: passed')"
        ),
        constraint="Preserve the public increment(value) signature",
        omitted_detail="counter-regression-7319",
        superseded_decision="Replace increment with a global constant",
    ),
    _TaskDefinition(
        task_id="fix_slugify",
        path="slug.py",
        before=(
            "def slugify(value: str) -> str:\n"
            "    return value.lower().replace(' ', '_')\n"
        ),
        after=(
            "def slugify(value: str) -> str:\n"
            "    return value.strip().lower().replace(' ', '-')\n"
        ),
        verification_code=(
            "from slug import slugify; assert slugify(' Hello World ') == 'hello-world'; "
            "print('required-suite: passed')"
        ),
        constraint="Keep slugify as a pure one-argument function",
        omitted_detail="slug-regression-4182",
        superseded_decision="Return underscores from slugify",
    ),
    _TaskDefinition(
        task_id="enable_feature_flag",
        path="flags.py",
        before="FEATURE_ENABLED = False\n",
        after="FEATURE_ENABLED = True\n",
        verification_code=(
            "from flags import FEATURE_ENABLED; assert FEATURE_ENABLED is True; "
            "print('required-suite: passed')"
        ),
        constraint="Do not rename FEATURE_ENABLED",
        omitted_detail="flag-regression-9024",
        superseded_decision="Delete the feature flag module",
    ),
)


async def run_reproducible_benchmark() -> BenchmarkReport:
    reports: list[BenchmarkTaskReport] = []
    for task in _TASKS:
        reports.append(await _run_task(task))
    passed_count = sum(report.success for report in reports)
    return BenchmarkReport(
        task_count=len(reports),
        passed_task_count=passed_count,
        success_rate=passed_count / len(reports),
        tasks=tuple(reports),
    )


async def run_single_benchmark_task(task_id: str) -> BenchmarkTaskReport:
    """Run one named task for isolated fault-injection and development tests."""

    task = next((item for item in _TASKS if item.task_id == task_id), None)
    if task is None:
        raise ValueError("Unknown benchmark task id")
    return await _run_task(task)


def write_benchmark_report(report: BenchmarkReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(f"{payload}\n", encoding="utf-8")


async def _run_task(task: _TaskDefinition) -> BenchmarkTaskReport:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"mini-agent-{task.task_id}-") as temporary:
        workspace_path = Path(temporary)
        (workspace_path / task.path).write_text(task.before, encoding="utf-8")
        data_dir = workspace_path / ".runtime"
        session = AgentSession.create(
            data_dir=data_dir,
            workspace=workspace_path,
            model="deterministic-benchmark",
        )
        try:
            goal = session.create_goal(
                objective=f"{task.constraint}; repair {task.path} and verify it",
                acceptance_criteria=(
                    AcceptanceCriterion(
                        criterion_id="modified",
                        description=f"Modify {task.path}",
                        evidence_kind=EvidenceKind.FILE_MODIFIED,
                    ),
                    AcceptanceCriterion(
                        criterion_id="verified",
                        description="Verification command exits zero",
                        evidence_kind=EvidenceKind.COMMAND_SUCCEEDED,
                    ),
                ),
            )
            old_event = session.append_user_message(
                f"Exact detail {task.omitted_detail}. Superseded: {task.superseded_decision}.",
                turn_id=session.new_turn_id(),
            )
            workspace = Workspace(workspace_path, excluded_roots=(data_dir,))
            agent = MiniAgent(
                config=_benchmark_config(data_dir),
                provider=_SequenceProvider(_responses(task)),
                session=session,
                tools=ToolRegistry(
                    (
                        ReadFileTool(workspace),
                        ApplyPatchTool(workspace),
                        ExecCommandTool(workspace),
                    )
                ),
                permissions=PermissionController(prompt=_AllowOncePrompt()),
            )
            await agent.run_turn(f"Repair {task.path} while following the active goal.")
            patch_event_id, command_event_id = _completion_event_ids(session)
            session.complete_goal(
                (
                    CompletionEvidence(
                        criterion_id="modified",
                        event_ids=(patch_event_id,),
                        note="apply_patch completed",
                    ),
                    CompletionEvidence(
                        criterion_id="verified",
                        event_ids=(command_event_id,),
                        note="verification exited zero",
                    ),
                )
            )
            completed_event = next(
                event
                for event in reversed(session.events)
                if isinstance(event.data, GoalStatusChangedData)
            )
            for cycle in range(2, 5):
                await agent.compact(focus=f"{task.constraint}; cycle {cycle}")
            checkpoints = CheckpointStore.open(
                session.paths.root,
                registrations=session.checkpoint_registrations,
                workspace_root=Path(session.metadata.workspace),
            )
            fixture = _fixture(
                task,
                goal_event_id=_goal_event_id(session, goal.goal_id),
                old_event_id=old_event.id,
                completed_event_id=completed_event.id,
                command_event_id=command_event_id,
            )
            evaluation = evaluate_session(session, checkpoints, fixture)
            duration = time.perf_counter() - started
            evaluation = evaluation.model_copy(
                update={
                    "latency_status": Availability.OBSERVED,
                    "latency_seconds": duration,
                }
            )
            success = (
                evaluation.passed_acceptance_thresholds
                and (workspace_path / task.path).read_text(encoding="utf-8") == task.after
            )
            return BenchmarkTaskReport(
                task_id=task.task_id,
                success=success,
                duration_seconds=duration,
                evaluation=evaluation,
            )
        finally:
            session.close()


def _responses(task: _TaskDefinition) -> list[ModelResponse]:
    read_call = ToolCall(
        id=f"read-{task.task_id}",
        name="read_file",
        arguments_json=json.dumps({"path": task.path}),
    )
    patch_call = ToolCall(
        id=f"patch-{task.task_id}",
        name="apply_patch",
        arguments_json=json.dumps(
            {
                "changes": [
                    {
                        "path": task.path,
                        "expected_content": task.before,
                        "replacement_content": task.after,
                    }
                ]
            }
        ),
    )
    command_call = ToolCall(
        id=f"verify-{task.task_id}",
        name="exec_command",
        arguments_json=json.dumps(
            {
                "argv": [str(Path(sys.executable).resolve()), "-c", task.verification_code],
                "cwd": ".",
            }
        ),
    )
    return [
        ModelResponse(tool_calls=(read_call,), finish_reason=FinishReason.TOOL_CALLS),
        ModelResponse(tool_calls=(patch_call,), finish_reason=FinishReason.TOOL_CALLS),
        ModelResponse(tool_calls=(command_call,), finish_reason=FinishReason.TOOL_CALLS),
        ModelResponse(content=f"Completed {task.task_id}."),
    ]


class _SequenceProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = iter(responses)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        return next(self._responses)


class _AllowOncePrompt(ApprovalPrompt):
    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        del request
        return ApprovalDecision.ALLOW_ONCE


def _benchmark_config(data_dir: Path) -> MiniAgentConfig:
    return MiniAgentConfig(
        model=ModelConfig(
            model="deterministic-benchmark",
            base_url=HttpUrl("https://example.test/v1"),
            context_window=131_072,
            max_output_tokens=4_096,
        ),
        context=ContextConfig(
            checkpoint_milestones=(0.70, 0.80, 0.90),
            rebuild_ratio=0.95,
            reserve_tokens=4_096,
            rebuild_seed_max_tokens=16_384,
            checkpoint_max_tokens=4_096,
            minimum_input_tokens=1_024,
        ),
        runtime=RuntimeConfig(
            data_dir=data_dir,
            provider_retry_count=0,
            retry_backoff_seconds=0,
        ),
    )


def _completion_event_ids(session: AgentSession) -> tuple[int, int]:
    tool_names = {
        event.data.tool_call.id: event.data.tool_call.name
        for event in session.events
        if event.data.kind == "tool_requested"
    }
    completed = {
        tool_names.get(event.data.tool_call_id): event.id
        for event in session.events
        if isinstance(event.data, ToolCompletedData)
    }
    patch_event_id = completed.get("apply_patch")
    command_event_id = completed.get("exec_command")
    if patch_event_id is None or command_event_id is None:
        raise RuntimeError("Benchmark task did not complete patch and command tools")
    return patch_event_id, command_event_id


def _goal_event_id(session: AgentSession, goal_id: str) -> int:
    return next(
        event.id
        for event in session.events
        if isinstance(event.data, GoalCreatedData) and event.data.goal_id == goal_id
    )


def _fixture(
    task: _TaskDefinition,
    *,
    goal_event_id: int,
    old_event_id: int,
    completed_event_id: int,
    command_event_id: int,
) -> EvaluationFixture:
    return EvaluationFixture(
        fixture_id=task.task_id,
        description=f"Actual read/patch/exec task for {task.path}",
        atoms=(
            InformationAtom(
                atom_id="constraint",
                category=AtomCategory.CRITICAL_CONSTRAINT,
                text=task.constraint,
                source_event_ids=(goal_event_id,),
            ),
            InformationAtom(
                atom_id="exact_detail",
                category=AtomCategory.EXACT_DETAIL,
                text=task.omitted_detail,
                source_event_ids=(old_event_id,),
            ),
            InformationAtom(
                atom_id="completed_state",
                category=AtomCategory.COMPLETED_STATE,
                text="Goal status: completed",
                source_event_ids=(completed_event_id,),
            ),
            InformationAtom(
                atom_id="superseded_decision",
                category=AtomCategory.SUPERSEDED_DECISION,
                text=task.superseded_decision,
                source_event_ids=(old_event_id,),
                is_current=False,
            ),
            InformationAtom(
                atom_id="retrieval_target",
                category=AtomCategory.RETRIEVAL_TARGET,
                text=task.omitted_detail,
                source_event_ids=(old_event_id,),
            ),
            InformationAtom(
                atom_id="completion_evidence",
                category=AtomCategory.COMPLETION_EVIDENCE,
                text="Exit code: 0",
                source_event_ids=(command_event_id,),
            ),
        ),
    )
