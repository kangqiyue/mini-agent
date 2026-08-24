"""Approval event validation and crash-recovery invariants."""

import hashlib
import json
from pathlib import Path

import pytest

from mini_agent.event_store import EventStoreCorruptionError
from mini_agent.events import (
    ApprovalDecision,
    ApprovalRequestedData,
    ApprovalResolvedData,
    ToolFailedData,
    ToolInterruptedData,
    ToolRecoveryStatus,
)
from mini_agent.messages import FinishReason, ToolCall
from mini_agent.permissions import (
    ApprovalScope,
    PermissionController,
    built_in_apply_patch_session_grant,
    recompute_approval_scope,
)
from mini_agent.session import AgentSession
from mini_agent.tools.apply_patch import ApplyPatchTool
from mini_agent.workspace import Workspace


def test_load_rejects_orphan_approval_request(tmp_path: Path) -> None:
    session, tool_call, turn_id = _session_with_write_tool_call(tmp_path)
    session.store.append(
        session_id=session.metadata.session_id,
        turn_id=turn_id,
        correlation_id=tool_call.id,
        data=_approval_requested(tool_call),
    )
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="no write tool request"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_load_rejects_approval_event_with_mismatched_correlation_id(tmp_path: Path) -> None:
    session, tool_call, turn_id = _session_with_write_tool_call(tmp_path)
    session.append_tool_requested(tool_call, is_read_only=False, turn_id=turn_id)
    session.store.append(
        session_id=session.metadata.session_id,
        turn_id=turn_id,
        correlation_id="another-call",
        data=_approval_requested(tool_call),
    )
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="invalid correlation id"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_load_rejects_approval_event_on_wrong_turn(tmp_path: Path) -> None:
    session, tool_call, turn_id = _session_with_write_tool_call(tmp_path)
    session.append_tool_requested(tool_call, is_read_only=False, turn_id=turn_id)
    session.store.append(
        session_id=session.metadata.session_id,
        turn_id=session.new_turn_id(),
        correlation_id=tool_call.id,
        data=_approval_requested(tool_call),
    )
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="does not match tool call"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_load_rejects_approval_resolution_with_wrong_fingerprint(tmp_path: Path) -> None:
    session, tool_call, turn_id = _session_with_write_tool_call(tmp_path)
    session.append_tool_requested(tool_call, is_read_only=False, turn_id=turn_id)
    _append_approval_request(session, tool_call, turn_id=turn_id)
    session.store.append(
        session_id=session.metadata.session_id,
        turn_id=turn_id,
        correlation_id=tool_call.id,
        data=ApprovalResolvedData(
            tool_call_id=tool_call.id,
            scope_fingerprint="b" * 64,
            decision=ApprovalDecision.ALLOW_ONCE,
        ),
    )
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="Invalid approval decision"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_load_rejects_duplicate_approval_resolution(tmp_path: Path) -> None:
    session, tool_call, turn_id = _session_with_write_tool_call(tmp_path)
    session.append_tool_requested(tool_call, is_read_only=False, turn_id=turn_id)
    _append_allow_once(session, tool_call, turn_id=turn_id)
    session.store.append(
        session_id=session.metadata.session_id,
        turn_id=turn_id,
        correlation_id=tool_call.id,
        data=ApprovalResolvedData(
            tool_call_id=tool_call.id,
            scope_fingerprint=_scope(tool_call).scope_fingerprint,
            decision=ApprovalDecision.ALLOW_ONCE,
        ),
    )
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="Invalid approval decision"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_load_rejects_ineligible_session_approval_grant(tmp_path: Path) -> None:
    session, tool_call, turn_id = _session_with_write_tool_call(
        tmp_path, arguments_json='{"password":"fake-value"}'
    )
    session.append_tool_requested(tool_call, is_read_only=False, turn_id=turn_id)
    scope = _append_approval_request(session, tool_call, turn_id=turn_id)
    session.append_approval_resolved(
        tool_call=tool_call,
        scope_fingerprint=scope.scope_fingerprint,
        decision=ApprovalDecision.ALLOW_SESSION,
        turn_id=turn_id,
    )
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="Sensitive scope"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_resume_marks_allow_resolved_but_unstarted_tool_as_interrupted(tmp_path: Path) -> None:
    session, tool_call, turn_id = _session_with_write_tool_call(tmp_path)
    session.append_tool_requested(tool_call, is_read_only=False, turn_id=turn_id)
    _append_allow_once(session, tool_call, turn_id=turn_id)
    session.close()

    resumed = AgentSession.resume(
        data_dir=tmp_path / "data",
        session_id=session.metadata.session_id,
    )

    recovered = resumed.events[-2].data
    assert isinstance(recovered, ToolInterruptedData)
    assert recovered.recovery_status is ToolRecoveryStatus.INTERRUPTED
    assert "not confirmed as started" in recovered.reason
    resumed.close()


def test_resume_marks_denied_unstarted_tool_as_permission_failure(tmp_path: Path) -> None:
    session, tool_call, turn_id = _session_with_write_tool_call(tmp_path)
    session.append_tool_requested(tool_call, is_read_only=False, turn_id=turn_id)
    scope = _append_approval_request(session, tool_call, turn_id=turn_id)
    session.append_approval_resolved(
        tool_call=tool_call,
        scope_fingerprint=scope.scope_fingerprint,
        decision=ApprovalDecision.DENY,
        turn_id=turn_id,
    )
    session.close()

    resumed = AgentSession.resume(
        data_dir=tmp_path / "data",
        session_id=session.metadata.session_id,
    )

    recovered = resumed.events[-2].data
    assert isinstance(recovered, ToolFailedData)
    assert recovered.before_start is True
    assert recovered.error_code == "permission_denied"
    resumed.close()


def test_load_accepts_preflight_failure_before_any_approval(tmp_path: Path) -> None:
    session, tool_call, turn_id = _session_with_write_tool_call(tmp_path)
    session.append_tool_requested(tool_call, is_read_only=False, turn_id=turn_id)
    session.append_tool_failed(
        tool_call,
        error_code="sensitive_replacement_content",
        message="Replacement content resembles a credential",
        before_start=True,
        turn_id=turn_id,
    )
    session.close()

    loaded = AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)
    loaded.close()


@pytest.mark.parametrize("decision", (ApprovalDecision.ALLOW_ONCE, ApprovalDecision.DENY))
def test_load_rejects_preflight_failure_after_approval_request_or_resolution(
    tmp_path: Path, decision: ApprovalDecision
) -> None:
    session, tool_call, turn_id = _session_with_write_tool_call(tmp_path)
    session.append_tool_requested(tool_call, is_read_only=False, turn_id=turn_id)
    scope = _append_approval_request(session, tool_call, turn_id=turn_id)
    if decision is ApprovalDecision.ALLOW_ONCE:
        session.append_approval_resolved(
            tool_call=tool_call,
            scope_fingerprint=scope.scope_fingerprint,
            decision=decision,
            turn_id=turn_id,
        )
    session.append_tool_failed(
        tool_call,
        error_code="sensitive_replacement_content",
        message="Forged after approval phase",
        before_start=True,
        turn_id=turn_id,
    )
    session.close()

    with pytest.raises(
        EventStoreCorruptionError, match="Invalid pre-start failure|Tool did not start"
    ):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_load_rejects_preflight_failure_after_tool_started(tmp_path: Path) -> None:
    session, tool_call, turn_id = _session_with_write_tool_call(tmp_path)
    session.append_tool_requested(tool_call, is_read_only=False, turn_id=turn_id)
    _append_allow_once(session, tool_call, turn_id=turn_id)
    session.append_tool_started(tool_call, turn_id=turn_id)
    session.append_tool_failed(
        tool_call,
        error_code="sensitive_replacement_content",
        message="Forged after tool start",
        before_start=True,
        turn_id=turn_id,
    )
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="Invalid pre-start failure"):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_load_rejects_permission_denied_without_a_matching_deny(tmp_path: Path) -> None:
    session, tool_call, turn_id = _session_with_write_tool_call(tmp_path)
    session.append_tool_requested(tool_call, is_read_only=False, turn_id=turn_id)
    session.append_tool_failed(
        tool_call,
        error_code="permission_denied",
        message="Forged denial",
        before_start=True,
        turn_id=turn_id,
    )
    session.close()

    with pytest.raises(
        EventStoreCorruptionError, match="Invalid pre-start failure|Tool did not start"
    ):
        AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)


def test_legacy_generic_session_allow_event_loads_without_restoring_grant(tmp_path: Path) -> None:
    session, tool_call, turn_id = _session_with_write_tool_call(
        tmp_path,
        tool_name="write_file",
        arguments_json='{"path":"notes.txt"}',
    )
    session.append_tool_requested(tool_call, is_read_only=False, turn_id=turn_id)
    scope = _scope(tool_call)
    session.store.append(
        session_id=session.metadata.session_id,
        turn_id=turn_id,
        correlation_id=tool_call.id,
        data=ApprovalRequestedData(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            redacted_arguments=scope.redacted_arguments,
            scope_descriptor=scope.scope_descriptor,
            scope_fingerprint=scope.scope_fingerprint,
            # v0.1 pre-hardening shape: accepted for transcript recovery but
            # intentionally not reactivated as durable authority.
            can_allow_session=True,
        ),
    )
    session.append_approval_resolved(
        tool_call=tool_call,
        scope_fingerprint=scope.scope_fingerprint,
        decision=ApprovalDecision.ALLOW_SESSION,
        turn_id=turn_id,
    )
    session.append_tool_started(tool_call, turn_id=turn_id)
    session.append_tool_completed(tool_call, "applied", turn_id=turn_id)
    session.close()

    resumed = AgentSession.resume(
        data_dir=tmp_path / "data",
        session_id=session.metadata.session_id,
    )

    assert resumed.session_grant_fingerprints == frozenset()
    resumed.close()


def test_apply_patch_path_grant_survives_resume_and_allows_new_content(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("old", encoding="utf-8")
    session, tool_call, turn_id = _session_with_write_tool_call(
        tmp_path,
        arguments_json=_patch_arguments(
            path="notes.txt", expected="old", replacement="first replacement"
        ),
    )
    session.append_tool_requested(tool_call, is_read_only=False, turn_id=turn_id)
    source_tool = ApplyPatchTool(Workspace(tmp_path))
    source_request = PermissionController().request_for(
        source_tool.definition,
        tool_call,
        built_in_apply_patch_grant=built_in_apply_patch_session_grant(
            source_tool, tool_call.arguments_json
        ),
    )
    assert source_request is not None
    assert source_request.can_allow_session is True
    session.append_approval_requested(
        tool_call=tool_call,
        redacted_arguments=source_request.redacted_arguments,
        scope_descriptor=source_request.scope_descriptor,
        scope_fingerprint=source_request.scope_fingerprint,
        can_allow_session=source_request.can_allow_session,
        turn_id=turn_id,
    )
    session.append_approval_resolved(
        tool_call=tool_call,
        scope_fingerprint=source_request.scope_fingerprint,
        decision=ApprovalDecision.ALLOW_SESSION,
        turn_id=turn_id,
    )
    session.append_tool_started(tool_call, turn_id=turn_id)
    session.append_tool_completed(tool_call, "applied", turn_id=turn_id)
    session.close()

    resumed = AgentSession.resume(
        data_dir=tmp_path / "data",
        session_id=session.metadata.session_id,
    )
    controller = PermissionController(
        session_grant_fingerprints=resumed.session_grant_fingerprints
    )
    later_call = ToolCall(
        id="call-later",
        name="apply_patch",
        arguments_json=_patch_arguments(
            path="./notes.txt",
            expected="first replacement",
            replacement="second replacement",
        ),
    )
    tool = ApplyPatchTool(Workspace(tmp_path))
    request = controller.request_for(
        tool.definition,
        later_call,
        built_in_apply_patch_grant=built_in_apply_patch_session_grant(
            tool, later_call.arguments_json
        ),
    )

    assert request is not None
    assert request.scope_fingerprint == source_request.scope_fingerprint
    assert controller.decide(request) is ApprovalDecision.ALLOW_SESSION
    resumed.close()


def test_unknown_write_revokes_prior_exact_session_grant(tmp_path: Path) -> None:
    session, tool_call, turn_id = _session_with_write_tool_call(
        tmp_path,
        arguments_json=_patch_arguments(
            path="notes.txt", expected="old", replacement="new"
        ),
    )
    session.append_tool_requested(tool_call, is_read_only=False, turn_id=turn_id)
    scope = _append_approval_request(session, tool_call, turn_id=turn_id)
    session.append_approval_resolved(
        tool_call=tool_call,
        scope_fingerprint=scope.scope_fingerprint,
        decision=ApprovalDecision.ALLOW_SESSION,
        turn_id=turn_id,
    )
    session.append_tool_started(tool_call, turn_id=turn_id)
    session.close()

    resumed = AgentSession.resume(
        data_dir=tmp_path / "data",
        session_id=session.metadata.session_id,
    )

    assert resumed.session_grant_fingerprints == frozenset()
    recovered = resumed.events[-2].data
    assert isinstance(recovered, ToolInterruptedData)
    assert recovered.recovery_status is ToolRecoveryStatus.UNKNOWN
    resumed.close()


def test_load_rejects_tool_start_after_terminal_event(tmp_path: Path) -> None:
    session, tool_call, turn_id = _session_with_write_tool_call(tmp_path)
    session.append_tool_interrupted(
        tool_call,
        recovery_status=ToolRecoveryStatus.INTERRUPTED,
        reason="not started",
        turn_id=turn_id,
    )
    session.append_tool_requested(tool_call, is_read_only=False, turn_id=turn_id)
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="after terminal"):
        AgentSession.load(
            data_dir=tmp_path / "data",
            session_id=session.metadata.session_id,
        )


def test_load_rejects_interrupted_status_for_started_write(tmp_path: Path) -> None:
    session, tool_call, turn_id = _session_with_write_tool_call(tmp_path)
    session.append_tool_requested(tool_call, is_read_only=False, turn_id=turn_id)
    _append_allow_once(session, tool_call, turn_id=turn_id)
    session.append_tool_started(tool_call, turn_id=turn_id)
    session.append_tool_interrupted(
        tool_call,
        recovery_status=ToolRecoveryStatus.INTERRUPTED,
        reason="incorrectly marked safe",
        turn_id=turn_id,
    )
    session.close()

    with pytest.raises(EventStoreCorruptionError, match="recovery status"):
        AgentSession.load(
            data_dir=tmp_path / "data",
            session_id=session.metadata.session_id,
        )


def _session_with_write_tool_call(
    tmp_path: Path,
    *,
    tool_name: str = "apply_patch",
    arguments_json: str = '{"patch":"x"}',
) -> tuple[AgentSession, ToolCall, str]:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    tool_call = ToolCall(id="call-write", name=tool_name, arguments_json=arguments_json)
    turn_id = session.new_turn_id()
    session.append_model_request_started(turn_id=turn_id, message_count=1, attempt=1)
    session.append_assistant_message(
        None,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(tool_call,),
        turn_id=turn_id,
    )
    return session, tool_call, turn_id


def _approval_requested(tool_call: ToolCall) -> ApprovalRequestedData:
    scope = _scope(tool_call)
    return ApprovalRequestedData(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        redacted_arguments=scope.redacted_arguments,
        scope_descriptor=scope.scope_descriptor,
        scope_fingerprint=scope.scope_fingerprint,
        can_allow_session=scope.can_allow_session,
    )


def _append_allow_once(session: AgentSession, tool_call: ToolCall, *, turn_id: str) -> None:
    scope = _append_approval_request(session, tool_call, turn_id=turn_id)
    session.append_approval_resolved(
        tool_call=tool_call,
        scope_fingerprint=scope.scope_fingerprint,
        decision=ApprovalDecision.ALLOW_ONCE,
        turn_id=turn_id,
    )


def _append_approval_request(
    session: AgentSession, tool_call: ToolCall, *, turn_id: str
) -> ApprovalScope:
    scope = _scope(tool_call)
    session.append_approval_requested(
        tool_call=tool_call,
        redacted_arguments=scope.redacted_arguments,
        scope_descriptor=scope.scope_descriptor,
        scope_fingerprint=scope.scope_fingerprint,
        can_allow_session=scope.can_allow_session,
        turn_id=turn_id,
    )
    return scope


def _scope(tool_call: ToolCall) -> ApprovalScope:
    session_scope_descriptor = None
    if tool_call.name == "apply_patch":
        try:
            path = json.loads(tool_call.arguments_json)["changes"][0]["path"]
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):
            path = None
        if isinstance(path, str):
            session_scope_descriptor = _patch_scope(path)
    scope = recompute_approval_scope(
        tool_name=tool_call.name,
        is_read_only=False,
        arguments_json=tool_call.arguments_json,
        session_scope_descriptor=session_scope_descriptor,
    )
    assert scope is not None
    return scope


def _patch_arguments(*, path: str, expected: str, replacement: str) -> str:
    return (
        "{\"changes\":[{\"path\":\""
        + path
        + "\",\"expected_content\":\""
        + expected
        + "\",\"replacement_content\":\""
        + replacement
        + "\"}]}"
    )


def _patch_scope(path: str, *, workspace: str = "/workspace") -> str:
    scope_identity = f"{workspace}\0{path}"
    return "apply_patch:workspace-path:v1:" + hashlib.sha256(
        scope_identity.encode()
    ).hexdigest()
