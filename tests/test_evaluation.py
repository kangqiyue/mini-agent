from datetime import UTC, datetime
from pathlib import Path

from mini_agent.checkpoint import Checkpoint, CheckpointItem, CheckpointStore
from mini_agent.evaluation import (
    Availability,
    evaluate_session,
    load_evaluation_fixture,
    write_evaluation_report,
)
from mini_agent.events import ApprovalDecision, CheckpointCommittedData
from mini_agent.goal import AcceptanceCriterion, CompletionEvidence, EvidenceKind
from mini_agent.messages import FinishReason, ToolCall
from mini_agent.permissions import build_approval_scope
from mini_agent.session import AgentSession
from mini_agent.tool_facts import ToolCompletionFacts
from mini_agent.tools.base import ToolDefinition

FIXTURE = Path(__file__).parent / "fixtures/evaluation/three_cycle_recovery.json"


def test_evaluation_reports_retention_recovery_and_unavailable_external_metrics(
    tmp_path: Path,
) -> None:
    session, checkpoints = _evaluated_session(tmp_path)
    fixture = load_evaluation_fixture(FIXTURE)

    read_only_checkpoints = CheckpointStore.open(
        session.paths.root,
        registrations=session.checkpoint_registrations,
        workspace_root=Path(session.metadata.workspace),
        writable=False,
    )

    report = evaluate_session(session, read_only_checkpoints, fixture)

    assert report.cycle_count == 4
    assert report.critical_constraint_exact_recall == 1.0
    assert report.state_micro_f1 == 1.0
    assert report.stale_fact_rate == 0.0
    assert report.exact_detail_recall == 0.0
    assert report.retrieval_success == 1.0
    assert report.false_completion_rate == 0.0
    assert report.end_task_success is True
    assert report.recovery_success is True
    assert report.passed_acceptance_thresholds is True
    assert report.missing_atom_ids == ("exact_regression_code",)
    assert "history_target" in report.history_recoverable_atom_ids
    assert report.provider_usage_status is Availability.UNAVAILABLE
    assert report.cost_status is Availability.UNAVAILABLE
    assert report.latency_status is Availability.UNAVAILABLE
    assert report.human_evaluation_status is Availability.UNAVAILABLE
    assert read_only_checkpoints.current == checkpoints.current

    report_path = tmp_path / "reports" / "report.json"
    write_evaluation_report(report, report_path)
    persisted = report_path.read_text(encoding="utf-8")
    assert persisted.endswith("\n")
    assert '"fixture_id": "three_cycle_recovery"' in persisted
    session.close()


def test_evaluation_fails_thresholds_for_stale_state_and_false_completion(
    tmp_path: Path,
) -> None:
    session, checkpoints = _evaluated_session(
        tmp_path,
        include_stale=True,
        successful_command=False,
    )

    report = evaluate_session(session, checkpoints, load_evaluation_fixture(FIXTURE))

    assert report.stale_fact_rate == 1.0
    assert report.false_completion_rate == 1.0
    assert report.end_task_success is False
    assert report.passed_acceptance_thresholds is False
    assert report.stale_atom_ids == ("old_strategy",)
    session.close()


def test_fixture_loader_rejects_unversioned_or_invalid_input(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"fixture_id":"bad"}', encoding="utf-8")

    try:
        load_evaluation_fixture(invalid)
    except ValueError as error:
        assert str(error) == "Evaluation fixture is unreadable or invalid"
    else:
        raise AssertionError("Invalid fixture was accepted")


def _evaluated_session(
    tmp_path: Path,
    *,
    include_stale: bool = False,
    successful_command: bool = True,
) -> tuple[AgentSession, CheckpointStore]:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    session.create_goal(
        objective="Ship the change",
        acceptance_criteria=(
            AcceptanceCriterion(
                criterion_id="tests",
                description="Tests pass",
                evidence_kind=EvidenceKind.COMMAND_SUCCEEDED,
            ),
        ),
    )
    constraint = session.append_user_message(
        "Never remove the compatibility shim",
        turn_id=session.new_turn_id(),
    )
    exact = session.append_user_message(
        "The exact error is rare-regression-code-7319",
        turn_id=session.new_turn_id(),
    )
    active = session.append_user_message(
        "Verify the final change",
        turn_id=session.new_turn_id(),
    )
    session.append_user_message(
        "Use the deleted legacy strategy",
        turn_id=session.new_turn_id(),
    )
    command = ToolCall(
        id="verify-command",
        name="exec_command",
        arguments_json='{"argv":["pytest"]}',
    )
    turn_id = session.new_turn_id()
    session.append_assistant_message(
        None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(command,),
        turn_id=turn_id,
    )
    session.append_tool_requested(command, is_read_only=False, turn_id=turn_id)
    scope = build_approval_scope(
        ToolDefinition(
            name="exec_command",
            description="Run verification",
            parameters={"type": "object"},
            is_read_only=False,
        ),
        command,
    )
    assert scope is not None
    session.append_approval_requested(
        tool_call=command,
        redacted_arguments=scope.redacted_arguments,
        scope_descriptor=scope.scope_descriptor,
        scope_fingerprint=scope.scope_fingerprint,
        can_allow_session=scope.can_allow_session,
        turn_id=turn_id,
    )
    session.append_approval_resolved(
        tool_call=command,
        scope_fingerprint=scope.scope_fingerprint,
        decision=ApprovalDecision.ALLOW_ONCE,
        turn_id=turn_id,
    )
    session.append_tool_started(command, turn_id=turn_id)
    output = (
        "Exit code: 0\nrequired-suite: passed"
        if successful_command
        else "Exit code: 0\nunrelated-suite: passed"
    )
    command_event = session.append_tool_completed(
        command,
        output,
        facts=ToolCompletionFacts(command_exit_code=0),
        turn_id=turn_id,
    )
    session.complete_goal(
        (
            CompletionEvidence(
                criterion_id="tests",
                event_ids=(command_event.id,),
                note="pytest exited zero",
            ),
        )
    )
    checkpoint = _commit_checkpoint(
        session,
        constraint_event_id=constraint.id,
        active_event_id=active.id,
        include_stale=include_stale,
    )
    for cycle in range(2, 5):
        rebuild_id = f"{cycle:032x}"
        source_count = len(session.conversation_messages())
        session.append_rebuild_started(
            rebuild_id=rebuild_id,
            source_cycle_id=cycle - 1,
            checkpoint_id=checkpoint.checkpoint_id,
            checkpoint_watermark=checkpoint.source_through_event_id,
            reason="fixture_cycle",
            source_message_count=source_count,
            turn_id=session.new_turn_id(),
        )
        session.append_rebuild_completed(
            rebuild_id=rebuild_id,
            projection_version=cycle,
            strategy="checkpoint_rebuild",
            checkpoint_id=checkpoint.checkpoint_id,
            checkpoint_watermark=checkpoint.source_through_event_id,
            projected_message_count=2,
            estimated_tokens=256,
            source_message_count=source_count,
            cycle_id=cycle,
            turn_id=session.new_turn_id(),
        )
    assert exact.id > 0
    return session, CheckpointStore.open(
        session.paths.root,
        registrations=session.checkpoint_registrations,
        workspace_root=Path(session.metadata.workspace),
    )


def _commit_checkpoint(
    session: AgentSession,
    *,
    constraint_event_id: int,
    active_event_id: int,
    include_stale: bool,
) -> Checkpoint:
    store = CheckpointStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    through = session.events[-1].id
    checkpoint = Checkpoint(
        checkpoint_id="a" * 32,
        session_id=session.metadata.session_id,
        cycle_id=1,
        version=1,
        source_from_event_id=1,
        source_through_event_id=through,
        current_intent="Verify the final change",
        constraints_and_preferences=(
            CheckpointItem(
                text="Never remove the compatibility shim",
                source_event_ids=(constraint_event_id,),
            ),
        ),
        active_work=(
            CheckpointItem(
                text="Verify the final change",
                source_event_ids=(active_event_id,),
            ),
        ),
        key_decisions=(
            (
                CheckpointItem(
                    text="Use the deleted legacy strategy",
                    source_event_ids=(through,),
                ),
            )
            if include_stale
            else ()
        ),
        writer_model="fixture",
        created_at=datetime.now(UTC),
    )
    record = store.stage(checkpoint)
    session.append_checkpoint_started(
        checkpoint_id=checkpoint.checkpoint_id,
        source_from_event_id=checkpoint.source_from_event_id,
        source_through_event_id=checkpoint.source_through_event_id,
    )
    registration_event = session.append_checkpoint_committed(
        checkpoint_id=record.checkpoint_id,
        version=record.version,
        source_through_event_id=record.source_through_event_id,
        content_hash=record.content_hash,
    )
    registration = registration_event.data
    assert isinstance(registration, CheckpointCommittedData)
    store.activate(registration)
    return checkpoint
