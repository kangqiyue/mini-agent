from pathlib import Path

import pytest
from pydantic import HttpUrl

from mini_agent.agent import MiniAgent
from mini_agent.config import MiniAgentConfig, ModelConfig
from mini_agent.event_store import EventStoreCorruptionError
from mini_agent.events import ApprovalDecision, GoalStatusChangedData
from mini_agent.goal import (
    AcceptanceCriterion,
    CompletionEvidence,
    EvidenceKind,
    GoalCompletionRejected,
    GoalStatus,
)
from mini_agent.messages import FinishReason, ModelRequest, ModelResponse, ToolCall
from mini_agent.permissions import build_approval_scope
from mini_agent.session import AgentSession
from mini_agent.tool_facts import ToolCompletionFacts
from mini_agent.tools.base import ToolDefinition


class _UnusedProvider:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError(request)


def _criterion(
    criterion_id: str,
    evidence_kind: EvidenceKind,
) -> AcceptanceCriterion:
    return AcceptanceCriterion(
        criterion_id=criterion_id,
        description=f"Prove {criterion_id}",
        evidence_kind=evidence_kind,
    )


def test_goal_round_trips_updates_and_blocked_status(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    session = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="m")
    goal = session.create_goal(objective="Ship the feature")
    goal = session.update_goal(
        objective="Ship and verify the feature",
        acceptance_criteria=(_criterion("tests", EvidenceKind.COMMAND_SUCCEEDED),),
    )
    assert goal.objective == "Ship and verify the feature"

    assert session.block_goal("Waiting for the test service").status is GoalStatus.BLOCKED
    assert session.activate_goal().status is GoalStatus.ACTIVE
    session_id = session.metadata.session_id
    session.close()

    loaded = AgentSession.load(data_dir=data_dir, session_id=session_id)
    assert loaded.goal is not None
    assert loaded.goal.objective == "Ship and verify the feature"
    assert loaded.goal.status is GoalStatus.ACTIVE
    loaded.close()


def test_completion_gate_rejects_missing_and_wrong_evidence(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    session.create_goal(
        objective="Modify and verify",
        acceptance_criteria=(
            _criterion("modified", EvidenceKind.FILE_MODIFIED),
            _criterion("tests", EvidenceKind.COMMAND_SUCCEEDED),
        ),
    )

    with pytest.raises(GoalCompletionRejected) as missing:
        session.complete_goal(())
    assert {gap.criterion_id for gap in missing.value.result.gaps} == {"modified", "tests"}

    turn_id = session.new_turn_id()
    call = ToolCall(id="call_read", name="read_file", arguments_json='{"path":"a"}')
    session.append_assistant_message(
        None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(call,),
        turn_id=turn_id,
    )
    session.append_tool_requested(call, is_read_only=True, turn_id=turn_id)
    session.append_tool_started(call, turn_id=turn_id)
    completed = session.append_tool_completed(call, "contents", turn_id=turn_id)

    evidence = (
        CompletionEvidence(
            criterion_id="modified",
            event_ids=(completed.id,),
            note="A tool completed",
        ),
        CompletionEvidence(
            criterion_id="tests",
            event_ids=(completed.id,),
            note="A tool completed",
        ),
    )
    with pytest.raises(GoalCompletionRejected) as wrong:
        session.complete_goal(evidence)
    assert {gap.code for gap in wrong.value.result.gaps} == {"evidence_kind_mismatch"}
    assert session.goal is not None
    assert session.goal.status is GoalStatus.ACTIVE
    session.close()


def test_completion_gate_accepts_file_and_command_events(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    session = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="m")
    session.create_goal(
        objective="Modify and verify",
        acceptance_criteria=(
            _criterion("modified", EvidenceKind.FILE_MODIFIED),
            _criterion("tests", EvidenceKind.COMMAND_SUCCEEDED),
        ),
    )
    patch_event = _complete_tool(
        session,
        ToolCall(id="call_patch", name="apply_patch", arguments_json='{"changes":[]}'),
        "Modified files:\n- target.txt",
        is_read_only=False,
    )
    command_event = _complete_tool(
        session,
        ToolCall(id="call_test", name="exec_command", arguments_json='{"argv":["pytest"]}'),
        "Exit code: 0\nDuration: 0.1 seconds\nStdout:\npassed\nStderr:\n",
        is_read_only=False,
    )
    evidence = (
        CompletionEvidence(
            criterion_id="modified",
            event_ids=(patch_event,),
            note="apply_patch completed",
        ),
        CompletionEvidence(
            criterion_id="tests",
            event_ids=(command_event,),
            note="tests exited successfully",
        ),
    )

    assert session.complete_goal(evidence).is_satisfied
    assert session.goal is not None
    assert session.goal.status is GoalStatus.COMPLETED
    session_id = session.metadata.session_id
    session.close()

    loaded = AgentSession.load(data_dir=data_dir, session_id=session_id)
    assert loaded.goal is not None
    assert loaded.goal.completion_evidence == evidence
    loaded.close()


def test_completion_gate_uses_command_facts_when_output_is_an_artifact_reference(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    criterion = _criterion("tests", EvidenceKind.COMMAND_SUCCEEDED)
    session.create_goal(objective="Verify", acceptance_criteria=(criterion,))
    command_event = _complete_tool(
        session,
        ToolCall(id="call_large", name="exec_command", arguments_json='{"argv":["pytest"]}'),
        "Artifact " + "a" * 32 + " contains the full captured result.",
        is_read_only=False,
    )

    result = session.complete_goal(
        (
            CompletionEvidence(
                criterion_id="tests",
                event_ids=(command_event,),
                note="command exited zero",
            ),
        )
    )

    assert result.is_satisfied
    session.close()


def test_completion_gate_rejects_evidence_from_before_the_latest_goal_update(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    criterion = _criterion("tests", EvidenceKind.COMMAND_SUCCEEDED)
    session.create_goal(objective="Initial goal", acceptance_criteria=(criterion,))
    command_event = _complete_tool(
        session,
        ToolCall(id="call_old", name="exec_command", arguments_json='{"argv":["pytest"]}'),
        "Exit code: 0",
        is_read_only=False,
    )
    session.update_goal(objective="Revised goal", acceptance_criteria=(criterion,))

    with pytest.raises(GoalCompletionRejected) as rejected:
        session.complete_goal(
            (
                CompletionEvidence(
                    criterion_id="tests",
                    event_ids=(command_event,),
                    note="stale command",
                ),
            )
        )

    assert {gap.code for gap in rejected.value.result.gaps} == {"evidence_event_stale"}
    session.close()


def test_load_rejects_completed_goal_without_mechanical_evidence(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    session = AgentSession.create(data_dir=data_dir, workspace=tmp_path, model="m")
    goal = session.create_goal(
        objective="Verify",
        acceptance_criteria=(_criterion("tests", EvidenceKind.COMMAND_SUCCEEDED),),
    )
    session.store.append(
        session_id=session.metadata.session_id,
        data=GoalStatusChangedData(
            goal_id=goal.goal_id,
            status=GoalStatus.COMPLETED,
            completion_evidence=(
                CompletionEvidence(
                    criterion_id="tests",
                    event_ids=(1,),
                    note="claimed",
                ),
            ),
        ),
    )
    session_id = session.metadata.session_id
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="evidence gate"):
        AgentSession.load(data_dir=data_dir, session_id=session_id)


@pytest.mark.asyncio
async def test_agent_will_not_continue_a_completed_goal(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    session.create_goal(
        objective="Deliver",
        acceptance_criteria=(
            _criterion("delivered", EvidenceKind.ASSISTANT_RESPONSE),
        ),
    )
    assistant_event = session.append_assistant_message(
        "delivered",
        finish_reason=FinishReason.STOP,
        turn_id=session.new_turn_id(),
    )
    session.complete_goal(
        (
            CompletionEvidence(
                criterion_id="delivered",
                event_ids=(assistant_event.id,),
                note="final response persisted",
            ),
        )
    )
    agent = MiniAgent(
        config=MiniAgentConfig(
            model=ModelConfig(model="m", base_url=HttpUrl("https://example.test/v1"))
        ),
        provider=_UnusedProvider(),
        session=session,
    )

    with pytest.raises(RuntimeError, match="Goal is terminal"):
        await agent.run_turn("do more")
    session.close()


@pytest.mark.asyncio
async def test_blocked_goal_must_be_explicitly_resumed_before_agent_continues(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    session.create_goal(objective="Wait and continue")
    session.block_goal("external dependency")
    event_count = len(session.events)
    agent = MiniAgent(
        config=MiniAgentConfig(
            model=ModelConfig(model="m", base_url=HttpUrl("https://example.test/v1"))
        ),
        provider=_UnusedProvider(),
        session=session,
    )

    with pytest.raises(RuntimeError, match="Goal is blocked"):
        await agent.run_turn("continue")
    assert len(session.events) == event_count
    session.activate_goal()
    assert session.goal is not None
    assert session.goal.status is GoalStatus.ACTIVE
    session.close()


def test_invalid_status_transition_does_not_append_an_event(tmp_path: Path) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    session.create_goal(objective="Wait")
    session.block_goal("dependency")
    event_count = len(session.events)

    with pytest.raises(ValueError, match="active goal can become blocked"):
        session.block_goal("again")

    assert len(session.events) == event_count
    assert session.goal is not None
    assert session.goal.status is GoalStatus.BLOCKED
    session.close()


def _complete_tool(
    session: AgentSession,
    call: ToolCall,
    output: str,
    *,
    is_read_only: bool,
) -> int:
    turn_id = session.new_turn_id()
    session.append_model_request_started(turn_id=turn_id, message_count=1, attempt=1)
    session.append_assistant_message(
        None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(call,),
        turn_id=turn_id,
    )
    session.append_tool_requested(call, is_read_only=is_read_only, turn_id=turn_id)
    if not is_read_only:
        scope = build_approval_scope(
            ToolDefinition(
                name=call.name,
                description="test tool",
                parameters={"type": "object"},
                is_read_only=False,
            ),
            call,
        )
        assert scope is not None
        session.append_approval_requested(
            tool_call=call,
            redacted_arguments=scope.redacted_arguments,
            scope_descriptor=scope.scope_descriptor,
            scope_fingerprint=scope.scope_fingerprint,
            can_allow_session=scope.can_allow_session,
            turn_id=turn_id,
        )
        session.append_approval_resolved(
            tool_call=call,
            scope_fingerprint=scope.scope_fingerprint,
            decision=ApprovalDecision.ALLOW_ONCE,
            turn_id=turn_id,
        )
    session.append_tool_started(call, turn_id=turn_id)
    if call.name == "exec_command":
        facts = ToolCompletionFacts(command_exit_code=0)
    elif call.name == "apply_patch":
        facts = ToolCompletionFacts(modified_paths=("target.txt",))
    else:
        facts = ToolCompletionFacts()
    return session.append_tool_completed(call, output, facts=facts, turn_id=turn_id).id
