"""End-to-end coverage for the built-in local coding tools."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import HttpUrl

from mini_agent.agent import MiniAgent
from mini_agent.config import MiniAgentConfig, ModelConfig, RuntimeConfig
from mini_agent.events import ApprovalDecision, ToolCompletedData, ToolFailedData
from mini_agent.messages import FinishReason, ModelResponse, ToolCall
from mini_agent.permissions import ApprovalRequest, PermissionController
from mini_agent.redaction_types import RedactionKind
from mini_agent.session import AgentSession
from mini_agent.tools.apply_patch import ApplyPatchTool
from mini_agent.tools.exec_command import ExecCommandTool
from mini_agent.tools.read_file import ReadFileTool
from mini_agent.tools.registry import ToolRegistry
from mini_agent.tools.search import SearchTool
from mini_agent.workspace import Workspace
from tests.support.providers import SequenceProvider


class AllowOncePrompt:
    def __init__(self) -> None:
        self.requests: list[ApprovalRequest] = []

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return ApprovalDecision.ALLOW_ONCE


class AllowSessionPrompt(AllowOncePrompt):
    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return ApprovalDecision.ALLOW_SESSION


def _config(tmp_path: Path) -> MiniAgentConfig:
    return MiniAgentConfig(
        model=ModelConfig(model="m", base_url=HttpUrl("https://example.test/v1")),
        runtime=RuntimeConfig(
            data_dir=tmp_path / "data",
            provider_retry_count=0,
            retry_backoff_seconds=0,
        ),
    )


@pytest.mark.asyncio
async def test_coding_tools_complete_read_patch_exec_lifecycle(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("before\n", encoding="utf-8")
    python_executable = str(Path(sys.executable).resolve())
    assert Path(python_executable).is_absolute()

    read_call = ToolCall(
        id="read-target",
        name="read_file",
        arguments_json=json.dumps({"path": target.name}),
    )
    patch_call = ToolCall(
        id="patch-target",
        name="apply_patch",
        arguments_json=json.dumps(
            {
                "changes": [
                    {
                        "path": target.name,
                        "expected_content": "before\n",
                        "replacement_content": "after\n",
                    }
                ]
            }
        ),
    )
    exec_call = ToolCall(
        id="verify-target",
        name="exec_command",
        arguments_json=json.dumps(
            {
                "argv": [
                    python_executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        "assert Path('target.txt').read_text(encoding='utf-8') == 'after\\n'; "
                        "print('verified')"
                    ),
                ],
                "cwd": ".",
            }
        ),
    )
    provider = SequenceProvider(
        [
            ModelResponse(tool_calls=(read_call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(tool_calls=(patch_call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(tool_calls=(exec_call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="Updated and verified target.txt."),
        ]
    )
    prompt = AllowOncePrompt()
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    workspace = Workspace(tmp_path)
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry(
            (
                ReadFileTool(workspace),
                ApplyPatchTool(workspace),
                ExecCommandTool(workspace),
            )
        ),
        permissions=PermissionController(prompt=prompt),
    )

    response = await agent.run_turn("Read target.txt, update it, and verify the change.")

    assert response.content == "Updated and verified target.txt."
    assert target.read_text(encoding="utf-8") == "after\n"
    assert [request.tool_call_id for request in prompt.requests] == [
        patch_call.id,
        exec_call.id,
    ]
    assert prompt.requests[0].can_allow_session is True
    assert prompt.requests[1].can_allow_session is False

    assert len(provider.requests) == 4
    for request, tool_call_id, expected_output in (
        (provider.requests[1], read_call.id, "before"),
        (provider.requests[2], patch_call.id, "Modified files:\n- target.txt"),
        (provider.requests[3], exec_call.id, "Exit code: 0"),
    ):
        tool_result = request.messages[-1]
        assert tool_result.tool_call_id == tool_call_id
        assert expected_output in (tool_result.content or "")

    event_kinds = [
        event.data.kind
        for event in session.events
        if event.data.kind != "context_estimated"
    ]
    assert event_kinds == [
        "session_started",
        "goal_created",
        "user_message",
        "model_request_started",
        "assistant_message",
        "tool_requested",
        "tool_started",
        "tool_completed",
        "model_request_started",
        "assistant_message",
        "tool_requested",
        "approval_requested",
        "approval_resolved",
        "tool_started",
        "tool_completed",
        "model_request_started",
        "assistant_message",
        "tool_requested",
        "approval_requested",
        "approval_resolved",
        "tool_started",
        "tool_completed",
        "model_request_started",
        "assistant_message",
    ]
    session_id = session.metadata.session_id
    session.close()

    reloaded = AgentSession.load(data_dir=tmp_path / "data", session_id=session_id)
    assert len(reloaded.events) == len(session.events)
    reloaded.close()


@pytest.mark.asyncio
async def test_exec_output_is_safe_for_the_next_model_request_and_keeps_source_summary(
    tmp_path: Path,
) -> None:
    secret_value = "synthetic-exec-secret"
    output_path = tmp_path / "exec-output.txt"
    output_path.write_text(f"api_key={secret_value}", encoding="utf-8")
    tool_call = ToolCall(
        id="run-secret-output",
        name="exec_command",
        arguments_json=json.dumps(
            {
                "argv": [
                    str(Path(sys.executable).resolve()),
                    "-c",
                    "import sys; from pathlib import Path; print(Path(sys.argv[1]).read_text())",
                    str(output_path),
                ]
            }
        ),
    )
    provider = SequenceProvider(
        [
            ModelResponse(tool_calls=(tool_call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="handled"),
        ]
    )
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((ExecCommandTool(Workspace(tmp_path)),)),
        permissions=PermissionController(prompt=AllowOncePrompt()),
    )

    response = await agent.run_turn("Run the command")

    assert response.content == "handled"
    next_request = provider.requests[1]
    assert secret_value not in (next_request.messages[-1].content or "")
    assert "[REDACTED]" in (next_request.messages[-1].content or "")
    completed_event = next(
        event for event in session.events if isinstance(event.data, ToolCompletedData)
    )
    assert isinstance(completed_event.data, ToolCompletedData)
    assert completed_event.redaction_summary.match_count == 0
    assert completed_event.data.source_redaction_summary.match_count == 1
    assert completed_event.data.source_redaction_summary.kinds == (
        RedactionKind.NAMED_SECRET,
    )
    session.close()


@pytest.mark.asyncio
async def test_sensitive_exec_long_option_never_reaches_durable_events_or_followup(
    tmp_path: Path,
) -> None:
    raw_value = "example-password-value"
    tool_call = ToolCall(
        id="reject-sensitive-option",
        name="exec_command",
        arguments_json=json.dumps(
            {"argv": ["echo", "--password", raw_value], "cwd": "."}
        ),
    )
    provider = SequenceProvider(
        [
            ModelResponse(tool_calls=(tool_call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="handled safely"),
        ]
    )
    prompt = AllowOncePrompt()
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((ExecCommandTool(Workspace(tmp_path)),)),
        permissions=PermissionController(prompt=prompt),
    )

    response = await agent.run_turn("Run a command")

    assert response.content == "handled safely"
    assert prompt.requests == []
    persisted_events = session.paths.events.read_text(encoding="utf-8")
    assert raw_value not in persisted_events
    assert "[REDACTED]" in persisted_events
    assert raw_value not in provider.requests[1].model_dump_json()
    failed = next(event.data for event in session.events if isinstance(event.data, ToolFailedData))
    assert failed.error_code == "sensitive_command_arguments"
    assert failed.before_start is True
    session.close()


@pytest.mark.asyncio
async def test_private_key_read_failure_does_not_persist_or_return_a_fragment(
    tmp_path: Path,
) -> None:
    source_fragment = "encoded-private-material"
    opening = "-----BEGIN " + "PRIVATE" + " KEY-----"
    closing = "-----END " + "PRIVATE" + " KEY-----"
    (tmp_path / "key.pem").write_text(
        "\n".join((opening, source_fragment, closing, "")), encoding="utf-8"
    )
    tool_call = ToolCall(
        id="read-private-key-body",
        name="read_file",
        arguments_json=json.dumps({"path": "key.pem", "start_line": 2, "end_line": 2}),
    )
    provider = SequenceProvider(
        [
            ModelResponse(tool_calls=(tool_call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="Private material was not read."),
        ]
    )
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((ReadFileTool(Workspace(tmp_path)),)),
        permissions=PermissionController(prompt=AllowOncePrompt()),
    )

    response = await agent.run_turn("Inspect only line two of key.pem.")

    assert response.content == "Private material was not read."
    failed_event = next(
        event for event in session.events if isinstance(event.data, ToolFailedData)
    )
    assert isinstance(failed_event.data, ToolFailedData)
    assert failed_event.data.error_code == "private_key_material"
    assert failed_event.data.message == "Refusing to read private key material"
    assert source_fragment not in failed_event.model_dump_json()
    assert source_fragment not in (provider.requests[1].messages[-1].content or "")
    assert source_fragment not in response.content
    assert session.artifact_registrations == ()
    session.close()


@pytest.mark.asyncio
async def test_private_key_search_failure_does_not_persist_or_return_key_body(
    tmp_path: Path,
) -> None:
    source_fragment = "private-body-never-egress"
    opening = "-----BEGIN " + "PRIVATE" + " KEY-----"
    closing = "-----END " + "PRIVATE" + " KEY-----"
    (tmp_path / "unusual-extension.data").write_text(
        "\n".join((opening, source_fragment, closing, "")), encoding="utf-8"
    )
    tool_call = ToolCall(
        id="search-private-key-header",
        name="search",
        arguments_json=json.dumps({"query": "PRIVATE"}),
    )
    provider = SequenceProvider(
        [
            ModelResponse(tool_calls=(tool_call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="Private material was not returned."),
        ]
    )
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((SearchTool(Workspace(tmp_path)),)),
        permissions=PermissionController(prompt=AllowOncePrompt()),
    )

    response = await agent.run_turn("Search the workspace")

    assert response.content == "Private material was not returned."
    failed_event = next(
        event for event in session.events if isinstance(event.data, ToolFailedData)
    )
    assert isinstance(failed_event.data, ToolFailedData)
    assert failed_event.data.error_code == "private_key_material"
    assert failed_event.data.message == "Refusing to return private key material"
    persisted_events = session.paths.events.read_text(encoding="utf-8")
    assert source_fragment not in persisted_events
    assert source_fragment not in provider.requests[1].model_dump_json()
    assert source_fragment not in response.content
    assert session.artifact_registrations == ()
    session.close()


@pytest.mark.asyncio
async def test_vcs_metadata_read_failure_does_not_egress_branch_reference(
    tmp_path: Path,
) -> None:
    branch_reference = "sensitive-" + "branch-" + "reference"
    git_head = tmp_path / ".git" / "HEAD"
    git_head.parent.mkdir()
    git_head.write_text(f"ref: refs/heads/{branch_reference}\n", encoding="utf-8")
    tool_call = ToolCall(
        id="read-vcs-head",
        name="read_file",
        arguments_json=json.dumps({"path": ".git/HEAD"}),
    )
    provider = SequenceProvider(
        [
            ModelResponse(tool_calls=(tool_call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="VCS metadata was not read."),
        ]
    )
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((ReadFileTool(Workspace(tmp_path)),)),
        permissions=PermissionController(prompt=AllowOncePrompt()),
    )

    response = await agent.run_turn("Read the workspace metadata")

    failed_event = next(
        event for event in session.events if isinstance(event.data, ToolFailedData)
    )
    assert isinstance(failed_event.data, ToolFailedData)
    assert failed_event.data.error_code == "sensitive_path"
    assert failed_event.data.message == "Sensitive workspace paths cannot be read"
    persisted_events = session.paths.events.read_text(encoding="utf-8")
    assert branch_reference not in persisted_events
    assert branch_reference not in provider.requests[1].model_dump_json()
    assert response.content is not None
    assert branch_reference not in response.content
    assert session.artifact_registrations == ()
    session.close()


@pytest.mark.asyncio
async def test_apply_patch_session_grant_reuses_canonical_file_scope(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("before", encoding="utf-8")
    first_call = ToolCall(
        id="first-patch",
        name="apply_patch",
        arguments_json=json.dumps(
            {
                "changes": [
                    {
                        "path": "target.txt",
                        "expected_content": "before",
                        "replacement_content": "first",
                    }
                ]
            }
        ),
    )
    second_call = ToolCall(
        id="second-patch",
        name="apply_patch",
        arguments_json=json.dumps(
            {
                "changes": [
                    {
                        "path": "./target.txt",
                        "expected_content": "first",
                        "replacement_content": "second",
                    }
                ]
            }
        ),
    )
    provider = SequenceProvider(
        [
            ModelResponse(tool_calls=(first_call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(tool_calls=(second_call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="done"),
        ]
    )
    prompt = AllowSessionPrompt()
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    workspace = Workspace(tmp_path)
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((ApplyPatchTool(workspace),)),
        permissions=PermissionController(prompt=prompt),
    )

    response = await agent.run_turn("Update target.txt twice.")

    assert response.content == "done"
    assert target.read_text(encoding="utf-8") == "second"
    assert [request.tool_call_id for request in prompt.requests] == [first_call.id]
    assert len(session.session_grant_fingerprints) == 1
    session.close()
