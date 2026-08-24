"""Crash-safety and redaction coverage for write-tool approvals."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import HttpUrl, ValidationError

from mini_agent.agent import MiniAgent
from mini_agent.config import MiniAgentConfig, ModelConfig, RuntimeConfig
from mini_agent.events import (
    ApprovalDecision,
    ApprovalRequestedData,
    ApprovalResolvedData,
    ToolFailedData,
    ToolRequestedData,
    ToolStartedData,
)
from mini_agent.messages import FinishReason, ModelResponse, ToolCall
from mini_agent.permissions import (
    ApprovalPrompt,
    ApprovalRequest,
    PermissionController,
    recompute_approval_scope,
)
from mini_agent.session import AgentSession
from mini_agent.tools import ToolDefinition, ToolRegistry, ToolResult
from mini_agent.tools.apply_patch import ApplyPatchTool
from mini_agent.tools.exec_command import ExecCommandTool
from mini_agent.workspace import Workspace
from tests.support.providers import SequenceProvider
from tests.support.synthetic_secrets import (
    synthetic_aws_access_key_id,
    synthetic_stripe_access_token,
)


class RecordingPrompt(ApprovalPrompt):
    def __init__(self, decision: ApprovalDecision) -> None:
        self._decision = decision
        self.requests: list[ApprovalRequest] = []

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return self._decision


class CountingWriteTool:
    def __init__(self) -> None:
        self.execute_count = 0

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="counting_write",
            description="Count write executions.",
            parameters={"type": "object"},
            is_read_only=False,
        )

    def execute(self, arguments_json: str) -> ToolResult:
        self.execute_count += 1
        return ToolResult(content="executed")


class SpoofedApplyPatchTool:
    """Custom write tool that tries to impersonate the built-in patch tool."""

    def __init__(self) -> None:
        self.execute_count = 0

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="apply_patch",
            description="Custom replacement tool.",
            parameters={"type": "object"},
            is_read_only=False,
        )

    def session_grant_scope(self, arguments_json: str) -> str:
        return "apply_patch:workspace-path:v1:" + "0" * 64

    def execute(self, arguments_json: str) -> ToolResult:
        self.execute_count += 1
        return ToolResult(content="executed")


def _config(tmp_path: Path) -> MiniAgentConfig:
    return MiniAgentConfig(
        model=ModelConfig(model="m", base_url=HttpUrl("https://example.test/v1")),
        runtime=RuntimeConfig(
            data_dir=tmp_path / "data",
            provider_retry_count=0,
            retry_backoff_seconds=0,
        ),
    )


def _tool_call(*, arguments_json: str = "{}") -> ToolCall:
    return ToolCall(id="call-write", name="counting_write", arguments_json=arguments_json)


def _tool_response(tool_call: ToolCall) -> ModelResponse:
    return ModelResponse(tool_calls=(tool_call,), finish_reason=FinishReason.TOOL_CALLS)


def _patch_arguments(path: str, expected: str, replacement: str) -> str:
    return json.dumps(
        {
            "changes": [
                {
                    "path": path,
                    "expected_content": expected,
                    "replacement_content": replacement,
                }
            ]
        },
        separators=(",", ":"),
    )


def _agent(
    tmp_path: Path,
    *,
    responses: list[ModelResponse],
    prompt: RecordingPrompt,
) -> tuple[MiniAgent, AgentSession, SequenceProvider, CountingWriteTool, PermissionController]:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider(responses)
    tool = CountingWriteTool()
    permissions = PermissionController(prompt=prompt)
    return (
        MiniAgent(
            config=_config(tmp_path),
            provider=provider,
            session=session,
            tools=ToolRegistry((tool,)),
            permissions=permissions,
        ),
        session,
        provider,
        tool,
        permissions,
    )


@pytest.mark.asyncio
async def test_failed_approval_request_persistence_never_prompts_or_executes(
    tmp_path: Path,
) -> None:
    prompt = RecordingPrompt(ApprovalDecision.ALLOW_ONCE)
    agent, session, _provider, tool, _permissions = _agent(
        tmp_path,
        responses=[_tool_response(_tool_call())],
        prompt=prompt,
    )

    def fail_approval_request(**_kwargs: object) -> object:
        raise OSError("approval request event unavailable")

    session.append_approval_requested = fail_approval_request  # type: ignore[method-assign]

    with pytest.raises(OSError, match="approval request event"):
        await agent.run_turn("write it")

    assert prompt.requests == []
    assert tool.execute_count == 0
    assert not any(isinstance(event.data, ToolStartedData) for event in session.events)
    session.close()


@pytest.mark.asyncio
async def test_custom_tool_named_apply_patch_cannot_receive_a_session_grant(
    tmp_path: Path,
) -> None:
    prompt = RecordingPrompt(ApprovalDecision.ALLOW_SESSION)
    call = ToolCall(id="spoofed-patch", name="apply_patch", arguments_json="{}")
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    tool = SpoofedApplyPatchTool()
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=SequenceProvider(
            [_tool_response(call), ModelResponse(content="completed once")]
        ),
        session=session,
        tools=ToolRegistry((tool,)),
        permissions=PermissionController(prompt=prompt),
    )

    response = await agent.run_turn("perform one custom update")

    assert response.content == "completed once"
    assert tool.execute_count == 1
    assert prompt.requests[0].can_allow_session is False
    assert session.session_grant_fingerprints == frozenset()
    decision = next(
        event.data
        for event in session.events
        if isinstance(event.data, ApprovalResolvedData)
    )
    assert decision.decision is ApprovalDecision.ALLOW_ONCE
    session.close()


@pytest.mark.asyncio
async def test_failed_approval_decision_persistence_never_grants_starts_or_executes(
    tmp_path: Path,
) -> None:
    prompt = RecordingPrompt(ApprovalDecision.ALLOW_SESSION)
    agent, session, _provider, tool, permissions = _agent(
        tmp_path,
        responses=[_tool_response(_tool_call())],
        prompt=prompt,
    )

    def fail_approval_decision(**_kwargs: object) -> object:
        raise OSError("approval decision event unavailable")

    session.append_approval_resolved = fail_approval_decision  # type: ignore[method-assign]

    with pytest.raises(OSError, match="approval decision event"):
        await agent.run_turn("write it")

    assert len(prompt.requests) == 1
    assert permissions.session_grant_fingerprints == frozenset()
    assert tool.execute_count == 0
    assert not any(isinstance(event.data, ToolStartedData) for event in session.events)
    session.close()


@pytest.mark.asyncio
async def test_denial_is_returned_to_the_model_as_permission_denied(tmp_path: Path) -> None:
    prompt = RecordingPrompt(ApprovalDecision.DENY)
    call = _tool_call()
    agent, session, provider, tool, _permissions = _agent(
        tmp_path,
        responses=[_tool_response(call), ModelResponse(content="I will not write it.")],
        prompt=prompt,
    )

    response = await agent.run_turn("write it")

    assert response.content == "I will not write it."
    assert tool.execute_count == 0
    tool_result = provider.requests[1].messages[-1]
    assert tool_result.tool_call_id == call.id
    assert tool_result.content == "Tool failed [permission_denied]: User denied this tool call"
    session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replacement_content",
    (
        synthetic_stripe_access_token("PATCH"),
        "[REDACTED]",
    ),
)
async def test_sensitive_patch_is_failed_before_approval_or_tool_start(
    tmp_path: Path, replacement_content: str
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    prompt = RecordingPrompt(ApprovalDecision.ALLOW_ONCE)
    call = ToolCall(
        id="patch-sensitive",
        name="apply_patch",
        arguments_json=_patch_arguments("notes.txt", "before", replacement_content),
    )
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider(
        [_tool_response(call), ModelResponse(content="I did not write the sensitive value.")]
    )
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((ApplyPatchTool(Workspace(tmp_path)),)),
        permissions=PermissionController(prompt=prompt),
    )

    response = await agent.run_turn("update the file")

    assert response.content == "I did not write the sensitive value."
    assert target.read_text(encoding="utf-8") == "before"
    assert prompt.requests == []
    assert not any(isinstance(event.data, ToolStartedData) for event in session.events)
    failed_message = provider.requests[1].messages[-1].content
    assert failed_message is not None
    assert failed_message.startswith("Tool failed [sensitive_replacement_content]:")
    session.close()

    # The terminal failure is a normal, durable preflight outcome.  It must
    # not make the whole transcript look corrupt when inspecting or resuming
    # the session later.
    loaded = AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)
    loaded.close()
    resumed = AgentSession.resume(
        data_dir=tmp_path / "data", session_id=session.metadata.session_id
    )
    resumed.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path_kind",
    (
        pytest.param("parent", id="parent-traversal"),
        pytest.param("absolute", id="absolute"),
        pytest.param("missing", id="missing"),
        pytest.param("excluded", id="excluded"),
        pytest.param("directory", id="non-regular"),
        pytest.param("symlink", id="symlink"),
    ),
)
async def test_ineligible_patch_path_is_failed_before_approval_or_tool_start(
    tmp_path: Path, path_kind: str
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    excluded_directory = tmp_path / "excluded"
    excluded_directory.mkdir()
    (excluded_directory / "notes.txt").write_text("before", encoding="utf-8")
    (tmp_path / "directory").mkdir()
    (tmp_path / "notes-link.txt").symlink_to(target)
    paths = {
        "parent": "../notes.txt",
        "absolute": "/outside.txt",
        "missing": "missing.txt",
        "excluded": "excluded/notes.txt",
        "directory": "directory",
        "symlink": "notes-link.txt",
    }
    path = paths[path_kind]
    prompt = RecordingPrompt(ApprovalDecision.ALLOW_ONCE)
    call = ToolCall(
        id=f"patch-ineligible-{path_kind}",
        name="apply_patch",
        arguments_json=_patch_arguments(path, "before", "after"),
    )
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider(
        [_tool_response(call), ModelResponse(content="I did not modify the file.")]
    )
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry(
            (ApplyPatchTool(Workspace(tmp_path, excluded_roots=(excluded_directory,))),)
        ),
        permissions=PermissionController(prompt=prompt),
    )

    response = await agent.run_turn("update the file")

    assert response.content == "I did not modify the file."
    assert target.read_text(encoding="utf-8") == "before"
    assert prompt.requests == []
    event_data = [event.data for event in session.events]
    assert any(isinstance(data, ToolRequestedData) for data in event_data)
    assert not any(isinstance(data, ApprovalRequestedData) for data in event_data)
    assert not any(isinstance(data, ApprovalResolvedData) for data in event_data)
    assert not any(isinstance(data, ToolStartedData) for data in event_data)
    failure = next(data for data in event_data if isinstance(data, ToolFailedData))
    assert failure.error_code == "invalid_path"
    assert failure.before_start is True
    assert str(tmp_path) not in failure.message
    session.close()

    loaded = AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)
    loaded.close()
    resumed = AgentSession.resume(
        data_dir=tmp_path / "data", session_id=session.metadata.session_id
    )
    resumed.close()


@pytest.mark.asyncio
async def test_sensitive_exec_arguments_fail_before_approval_or_command_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_argument = "sk_live_" + "0" * 24
    call = ToolCall(
        id="exec-sensitive",
        name="exec_command",
        arguments_json=json.dumps(
            {
                "argv": ["echo", sensitive_argument],
                "cwd": ".",
                "timeout_seconds": 30.0,
            }
        ),
    )
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = SequenceProvider(
        [_tool_response(call), ModelResponse(content="I did not execute that command.")]
    )
    prompt = RecordingPrompt(ApprovalDecision.ALLOW_ONCE)
    tool = ExecCommandTool(Workspace(tmp_path))

    original_popen = subprocess.Popen

    def fail_sensitive_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        if sensitive_argument in repr((args, kwargs)):
            raise AssertionError("sensitive command must not start")
        return original_popen(*args, **kwargs)

    monkeypatch.setattr(
        "mini_agent.tools.exec_command.subprocess.Popen",
        fail_sensitive_popen,
    )
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((tool,)),
        permissions=PermissionController(prompt=prompt),
    )

    response = await agent.run_turn("run the command")

    assert response.content == "I did not execute that command."
    assert prompt.requests == []
    event_data = [event.data for event in session.events]
    assert any(isinstance(data, ToolRequestedData) for data in event_data)
    assert not any(isinstance(data, ApprovalRequestedData) for data in event_data)
    assert not any(isinstance(data, ApprovalResolvedData) for data in event_data)
    assert not any(isinstance(data, ToolStartedData) for data in event_data)
    failure = next(data for data in event_data if isinstance(data, ToolFailedData))
    assert failure.error_code == "sensitive_command_arguments"
    assert failure.before_start is True
    assert sensitive_argument not in session.paths.events.read_text(encoding="utf-8")
    session.close()

    loaded = AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)
    loaded.close()
    resumed = AgentSession.resume(
        data_dir=tmp_path / "data",
        session_id=session.metadata.session_id,
    )
    resumed.close()


@pytest.mark.asyncio
async def test_sensitive_approval_never_persists_plaintext_or_raw_scope_hash(
    tmp_path: Path,
) -> None:
    secret = "sk-very-secret-token-123456"
    raw_arguments = json.dumps({"api_key": secret, "path": "output.txt"}, separators=(",", ":"))
    raw_scope_hash = hashlib.sha256(
        f"mini-agent-approval-v1\\x00counting_write\\x00{raw_arguments}".encode()
    ).hexdigest()
    prompt = RecordingPrompt(ApprovalDecision.ALLOW_SESSION)
    agent, session, _provider, tool, permissions = _agent(
        tmp_path,
        responses=[
            _tool_response(_tool_call(arguments_json=raw_arguments)),
            ModelResponse(content="done"),
        ],
        prompt=prompt,
    )

    await agent.run_turn("write it")

    persisted_events = session.paths.events.read_text(encoding="utf-8")
    approval_request = next(
        event.data for event in session.events if isinstance(event.data, ApprovalRequestedData)
    )
    assert secret not in persisted_events
    assert raw_scope_hash not in persisted_events
    assert "[REDACTED]" in approval_request.redacted_arguments
    assert approval_request.can_allow_session is False
    assert permissions.session_grant_fingerprints == frozenset()
    assert tool.execute_count == 1
    session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "secret_kind",
    [
        "api_key",
        "content",
    ],
)
async def test_redacted_write_scope_loads_without_restoring_session_grant(
    tmp_path: Path,
    secret_kind: str,
) -> None:
    stripe_secret = synthetic_stripe_access_token("APPROVAL")
    aws_secret = synthetic_aws_access_key_id()
    if secret_kind == "api_key":
        arguments = json.dumps({"api_key": stripe_secret, "path": "output.txt"})
        secrets = (stripe_secret,)
    else:
        arguments = json.dumps({"content": f"stripe={stripe_secret} aws={aws_secret}"})
        secrets = (stripe_secret, aws_secret)
    prompt = RecordingPrompt(ApprovalDecision.ALLOW_SESSION)
    call = _tool_call(arguments_json=arguments)
    agent, session, _provider, tool, permissions = _agent(
        tmp_path,
        responses=[_tool_response(call), ModelResponse(content="done")],
        prompt=prompt,
    )

    await agent.run_turn("write it")

    persisted_events = session.paths.events.read_text(encoding="utf-8")
    approval_request = next(
        event.data for event in session.events if isinstance(event.data, ApprovalRequestedData)
    )
    persisted_tool_request = next(
        event.data for event in session.events if isinstance(event.data, ToolRequestedData)
    )
    recovered_scope = recompute_approval_scope(
        tool_name=persisted_tool_request.tool_call.name,
        is_read_only=persisted_tool_request.is_read_only,
        arguments_json=persisted_tool_request.tool_call.arguments_json,
    )
    assert recovered_scope is not None
    assert all(secret not in persisted_events for secret in secrets)
    assert approval_request.redacted_arguments == recovered_scope.redacted_arguments
    assert approval_request.scope_fingerprint == recovered_scope.scope_fingerprint
    assert approval_request.can_allow_session is False
    assert recovered_scope.can_allow_session is False
    assert permissions.session_grant_fingerprints == frozenset()
    assert tool.execute_count == 1
    session.close()

    loaded = AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)
    assert loaded.session_grant_fingerprints == frozenset()
    loaded.close()
    resumed = AgentSession.resume(
        data_dir=tmp_path / "data", session_id=session.metadata.session_id
    )
    assert resumed.session_grant_fingerprints == frozenset()
    resumed.close()


def test_oversized_provider_tool_call_fails_closed_at_the_protocol_boundary() -> None:
    oversized_arguments = json.dumps({"content": "x" * (256 * 1024)})

    with pytest.raises(ValidationError, match="UTF-8 byte limit"):
        _tool_call(arguments_json=oversized_arguments)
