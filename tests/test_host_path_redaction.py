import json
import tempfile
from pathlib import Path

import pytest
from pydantic import HttpUrl

import mini_agent.host_path_redaction as host_path_redaction
from mini_agent.agent import MiniAgent
from mini_agent.artifacts import ArtifactStore
from mini_agent.config import MiniAgentConfig, ModelConfig, RuntimeConfig
from mini_agent.events import ModelRequestFailedData, ToolCompletedData, ToolFailedData
from mini_agent.history import SessionHistory
from mini_agent.host_path_redaction import (
    normalize_conversation_message_host_paths,
    normalize_model_request_host_paths,
    normalize_model_response_host_paths,
    redact_host_paths,
)
from mini_agent.messages import (
    ConversationMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from mini_agent.redaction_types import RedactionKind
from mini_agent.session import AgentSession
from mini_agent.tool_facts import ToolCompletionFacts
from mini_agent.tools import ToolDefinition, ToolError, ToolRegistry, ToolResult
from mini_agent.tools.read_file import ReadFileTool
from mini_agent.workspace import Workspace


class _RecordingProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = iter(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return next(self._responses)


class _FixedPathTool:
    def __init__(self, result: ToolResult, *, schema_path: str | None = None) -> None:
        self._result = result
        self._schema_path = schema_path

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="fixed_path",
            description=(
                f"Return fixed host path output from {self._schema_path}."
                if self._schema_path is not None
                else "Return fixed host path output."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "default": self._schema_path or "relative.txt",
                    }
                },
            },
            is_read_only=True,
        )

    def execute(self, arguments_json: str) -> ToolResult:
        del arguments_json
        return self._result


class _FailingPathTool:
    def __init__(self, message: str, *, code: str = "injected_failure") -> None:
        self._message = message
        self._code = code

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="failing_path",
            description="Fail with a fixed host path message.",
            parameters={"type": "object"},
            is_read_only=True,
        )

    def execute(self, arguments_json: str) -> ToolResult:
        del arguments_json
        raise ToolError(self._code, self._message)


class _SensitiveSchemaTool:
    @property
    def definition(self) -> ToolDefinition:
        # Deliberately bypass the extension publication boundary. This models a
        # malformed in-memory extension and proves outbound sanitization stays
        # active as defense in depth.
        return ToolDefinition.model_construct(
            name="sensitive_schema",
            description="Connect with password=not-for-provider.",
            parameters={
                "type": "object",
                "properties": {
                    "api_key": {"type": "string", "default": "not-for-provider"}
                },
            },
            is_read_only=True,
        )

    def execute(self, arguments_json: str) -> ToolResult:
        del arguments_json
        return ToolResult(content="unused")


def _config(tmp_path: Path, *, inline_chars: int = 12_000) -> MiniAgentConfig:
    return MiniAgentConfig(
        model=ModelConfig(model="m", base_url=HttpUrl("https://example.test/v1")),
        runtime=RuntimeConfig(
            data_dir=tmp_path / "data",
            inline_tool_result_max_chars=inline_chars,
        ),
    )


def _assert_host_path_absent(payload: str, forbidden_path: Path) -> None:
    if str(forbidden_path) in payload:
        pytest.fail("provider-visible or persisted content contains a host path")


def test_redact_host_paths_replaces_only_complete_prefixes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    home = tmp_path / "home"
    text = (
        f"cwd={workspace}\n"
        f'File "{workspace}/src/main.py"\n'
        f"cache={home}/.cache\n"
        f"wrapped=<{workspace}>\n"
        f"query=?root={workspace}#section\n"
        f"similar={workspace}ish"
    )

    result = redact_host_paths(
        text,
        workspace_root=workspace,
        home_directory=home,
    )

    assert "cwd=<workspace-root>" in result.text
    assert 'File "<workspace-root>/src/main.py"' in result.text
    assert "cache=<home>/.cache" in result.text
    assert "wrapped=<<workspace-root>>" in result.text
    assert "query=?root=<workspace-root>#section" in result.text
    assert f"similar={workspace}ish" in result.text
    assert result.match_count == 5
    assert result.matched_kinds == frozenset((RedactionKind.HOST_PATH,))


def test_redact_host_paths_handles_json_escaped_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / 'work"space'
    payload = json.dumps({"path": f"{workspace}/file.py"})

    result = redact_host_paths(payload, workspace_root=workspace)

    assert json.loads(result.text) == {"path": "<workspace-root>/file.py"}
    assert result.match_count == 1


def test_redact_host_paths_handles_verified_darwin_private_tmp_alias() -> None:
    canonical_tmp = Path("/private/tmp")
    if Path("/tmp").resolve() != canonical_tmp:
        pytest.skip("The stable Darwin /tmp alias is unavailable")

    with tempfile.TemporaryDirectory(dir="/tmp", prefix="mini-agent-redaction-") as directory:
        workspace = Path(directory).resolve()
        if not str(workspace).startswith("/private/tmp/"):
            pytest.skip("Temporary directory is outside the Darwin private tmp root")
        alias = Path("/tmp") / workspace.relative_to(canonical_tmp)

        result = redact_host_paths(
            f"path={alias}/result.txt similar={alias}ish",
            workspace_root=workspace,
        )

    assert result.text == "path=<workspace-root>/result.txt similar=" f"{alias}ish"
    assert result.match_count == 1


def test_redact_host_paths_handles_case_variant_on_case_insensitive_darwin_volume(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "CaseSensitiveWorkspace"
    workspace.mkdir()
    case_variant = workspace.with_name("casesensitiveworkspace")
    try:
        has_case_insensitive_volume = case_variant.samefile(workspace)
    except OSError:
        has_case_insensitive_volume = False
    if not has_case_insensitive_volume:
        pytest.skip("The test workspace volume is case-sensitive")

    result = redact_host_paths(
        f"path={case_variant}/result.txt",
        workspace_root=workspace,
    )

    assert result.text == "path=<workspace-root>/result.txt"
    assert result.match_count == 1


def test_redact_host_paths_keeps_linux_case_variants_as_ordinary_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "CaseSensitiveWorkspace"
    workspace.mkdir()
    case_variant = workspace.with_name("casesensitiveworkspace")
    monkeypatch.setattr(host_path_redaction.sys, "platform", "linux")

    result = redact_host_paths(
        f"path={case_variant}/result.txt",
        workspace_root=workspace,
        home_directory=tmp_path / "unrelated-home",
    )

    assert result.text == f"path={case_variant}/result.txt"
    assert result.match_count == 0


def test_redact_host_paths_handles_mixed_json_escaped_separators_in_prose(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "工作目录"
    home = tmp_path / "家"

    def mixed_json_separators(path: Path) -> str:
        pieces: list[str] = []
        slash_count = 0
        for character in str(path):
            if character != "/":
                pieces.append(character)
                continue
            pieces.append("\\/" if slash_count % 2 else "/")
            slash_count += 1
        return "".join(pieces)

    escaped_workspace = mixed_json_separators(workspace / "src" / "main.py")
    escaped_home = str(home / ".cache").replace("/", "\\/")
    result = redact_host_paths(
        f'多字节 JSON 片段 {{"workspace":"{escaped_workspace}","home":"{escaped_home}"}}',
        workspace_root=workspace,
        home_directory=home,
    )

    assert str(workspace) not in result.text
    assert str(home) not in result.text
    assert escaped_workspace not in result.text
    assert escaped_home not in result.text
    assert "<workspace-root>" in result.text
    assert "src\\/main.py" in result.text
    assert "<home>\\/.cache" in result.text
    assert redact_host_paths(
        result.text,
        workspace_root=workspace,
        home_directory=home,
    ).text == result.text


def test_model_request_normalizes_escaped_host_paths_in_all_content_roles(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    home_path = Path.home() / "mini-agent-provider-content"
    escaped_workspace = str(workspace / "src" / "main.py").replace("/", "\\/")
    escaped_home = str(home_path).replace("/", "\\/")
    tool_call = ToolCall(id="call-content", name="read_file", arguments_json="{}")
    request = ModelRequest(
        model="m",
        messages=(
            ConversationMessage(
                role=MessageRole.SYSTEM,
                content=f'系统上下文 {{"path":"{escaped_workspace}"}}',
            ),
            ConversationMessage(
                role=MessageRole.USER,
                content=f'用户内容 {{"path":"{escaped_home}"}}',
            ),
            ConversationMessage(role=MessageRole.ASSISTANT, tool_calls=(tool_call,)),
            ConversationMessage(
                role=MessageRole.TOOL,
                content=f'工具内容 {{"nested":{{"path":"{escaped_workspace}"}}}}',
                tool_call_id=tool_call.id,
            ),
        ),
    )

    safe_request = normalize_model_request_host_paths(request, workspace_root=workspace)

    for message in safe_request.messages:
        if message.content is None:
            continue
        _assert_host_path_absent(message.content, workspace)
        assert escaped_workspace not in message.content
        assert escaped_home not in message.content
    system_content = safe_request.messages[0].content
    user_content = safe_request.messages[1].content
    assert system_content is not None
    assert user_content is not None
    assert "<workspace-root>\\/src\\/main.py" in system_content
    assert "<home>\\/mini-agent-provider-content" in user_content


def test_redact_host_paths_protects_truncated_prefix_fragments(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace-private-name"
    path_text = str(workspace.resolve())
    prefix = path_text[:-3]
    suffix = path_text[5:]
    middle = path_text[5:-3]

    truncated_end = redact_host_paths(
        f"cwd={prefix}",
        workspace_root=workspace,
        truncated_at_end=True,
    )
    truncated_start = redact_host_paths(
        f"{suffix}/file.py",
        workspace_root=workspace,
        truncated_at_start=True,
    )
    truncated_both = redact_host_paths(
        middle,
        workspace_root=workspace,
        truncated_at_start=True,
        truncated_at_end=True,
    )

    assert truncated_end.text == "cwd=<workspace-root>"
    assert truncated_start.text == "<workspace-root>/file.py"
    assert truncated_both.text == "<workspace-root>"


def test_redact_host_paths_protects_json_escaped_truncated_fragments(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace-private-name"
    path_text = str(workspace.resolve())
    escaped_prefix = path_text[:-3].replace("/", "\\/")
    escaped_suffix = path_text[5:].replace("/", "\\/")
    escaped_middle = path_text[5:-3].replace("/", "\\/")

    truncated_end = redact_host_paths(
        f"cwd={escaped_prefix}",
        workspace_root=workspace,
        truncated_at_end=True,
    )
    truncated_start = redact_host_paths(
        f"{escaped_suffix}\\/file.py",
        workspace_root=workspace,
        truncated_at_start=True,
    )
    truncated_both = redact_host_paths(
        escaped_middle,
        workspace_root=workspace,
        truncated_at_start=True,
        truncated_at_end=True,
    )

    assert truncated_end.text == "cwd=<workspace-root>"
    assert truncated_start.text == "<workspace-root>\\/file.py"
    assert truncated_both.text == "<workspace-root>"


def test_shared_request_and_response_normalizers_preserve_typed_payload_shape(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    request = ModelRequest(
        model="m",
        messages=(
            ConversationMessage(
                role=MessageRole.USER,
                content=f"inspect {workspace}/src/main.py",
            ),
        ),
        tools=(
            ToolDefinition(
                name="read_file",
                description=f"Read {workspace}/src/main.py",
                parameters={"type": "object", "default": f"{workspace}/src/main.py"},
                is_read_only=True,
            ),
        ),
    )
    response = ModelResponse(
        tool_calls=(
            ToolCall(
                id="call-path",
                name="read_file",
                arguments_json=json.dumps({"path": f"{workspace}/src/main.py"}),
            ),
        ),
        finish_reason=FinishReason.TOOL_CALLS,
    )

    safe_request = normalize_model_request_host_paths(
        request,
        workspace_root=workspace,
    )
    safe_response = normalize_model_response_host_paths(
        response,
        workspace_root=workspace,
    )

    assert safe_request.messages[0].content == "inspect <workspace-root>/src/main.py"
    assert safe_request.tools[0].parameters["default"] == "<workspace-root>/src/main.py"
    assert json.loads(safe_response.tool_calls[0].arguments_json) == {
        "path": "<workspace-root>/src/main.py"
    }


def test_escaped_json_tool_arguments_redact_nested_paths_and_credentials(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    raw_arguments = json.dumps(
        {
            "nested": {
                "path": f"{workspace}/src/main.py",
                "note": "password=not-for-provider",
            },
            "api_key": "not-for-provider",
        }
    ).replace("/", "\\/")
    tool_call = ToolCall(
        id="call-path",
        name="read_file",
        arguments_json=raw_arguments,
    )
    conversation = ConversationMessage(
        role=MessageRole.ASSISTANT,
        tool_calls=(tool_call,),
    )
    response = ModelResponse(
        tool_calls=(tool_call,),
        finish_reason=FinishReason.TOOL_CALLS,
    )

    safe_conversation = normalize_conversation_message_host_paths(
        conversation,
        workspace_root=workspace,
    )
    safe_response = normalize_model_response_host_paths(
        response,
        workspace_root=workspace,
    )

    for safe_arguments in (
        safe_conversation.tool_calls[0].arguments_json,
        safe_response.tool_calls[0].arguments_json,
    ):
        _assert_host_path_absent(safe_arguments, workspace)
        assert json.loads(safe_arguments) == {
            "api_key": "[REDACTED]",
            "nested": {
                "note": "password=[REDACTED]",
                "path": "<workspace-root>/src/main.py",
            },
        }


def test_json_tool_argument_normalization_rejects_invalid_or_excessive_input(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    malformed_call = ToolCall.model_construct(
        id="call-invalid",
        name="read_file",
        arguments_json="not-json",
    )
    malformed_response = ModelResponse.model_construct(
        tool_calls=(malformed_call,),
        finish_reason=FinishReason.TOOL_CALLS,
    )

    with pytest.raises(ValueError, match="valid JSON object"):
        normalize_model_response_host_paths(
            malformed_response,
            workspace_root=workspace,
        )

    nested_arguments = "{}"
    for _ in range(17):
        nested_arguments = '{"nested":' + nested_arguments + "}"
    excessive_call = ToolCall.model_construct(
        id="call-deep",
        name="read_file",
        arguments_json=nested_arguments,
    )
    excessive_response = ModelResponse.model_construct(
        tool_calls=(excessive_call,),
        finish_reason=FinishReason.TOOL_CALLS,
    )

    with pytest.raises(ValueError, match="nesting limit"):
        normalize_model_response_host_paths(
            excessive_response,
            workspace_root=workspace,
        )


@pytest.mark.asyncio
async def test_provider_request_sanitizer_redacts_conversation_and_custom_schema(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    provider = _RecordingProvider([ModelResponse(content="done")])
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((_SensitiveSchemaTool(),)),
    )

    await agent.run_turn("password=not-for-provider")

    provider_payload = provider.requests[0].model_dump_json()
    assert "not-for-provider" not in provider_payload
    assert "[REDACTED]" in provider_payload
    session.close()


@pytest.mark.asyncio
async def test_response_path_key_collision_requires_resume_without_persisting_raw_arguments(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = AgentSession.create(
        data_dir=tmp_path / "data",
        workspace=workspace,
        model="m",
    )
    raw_arguments = json.dumps(
        {
            str(workspace / "result.txt"): "one",
            "<workspace-root>/result.txt": "two",
        }
    )
    malformed_call = ToolCall.model_construct(
        id="call-collision",
        name="fixed_path",
        arguments_json=raw_arguments,
    )
    malformed_response = ModelResponse.model_construct(
        tool_calls=(malformed_call,),
        finish_reason=FinishReason.TOOL_CALLS,
    )
    provider = _RecordingProvider([malformed_response])
    agent = MiniAgent(config=_config(tmp_path), provider=provider, session=session)

    with pytest.raises(ValueError, match="JSON key collision"):
        await agent.run_turn("inspect result")

    assert len(provider.requests) == 1
    persisted = session.paths.events.read_text(encoding="utf-8")
    assert "call-collision" not in persisted
    with pytest.raises(RuntimeError, match="must be resumed"):
        await agent.run_turn("another request")

    session_id = session.metadata.session_id
    session.close()
    resumed = AgentSession.resume(data_dir=tmp_path / "data", session_id=session_id)
    terminal = resumed.events[-2].data
    assert isinstance(terminal, ModelRequestFailedData)
    assert terminal.error_code == "model_request_interrupted"
    resumed.close()


@pytest.mark.asyncio
async def test_agent_normalizes_host_paths_before_persistence_and_provider(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = AgentSession.create(
        data_dir=tmp_path / "data",
        workspace=workspace,
        model="m",
    )
    call = ToolCall(id="call-path", name="fixed_path", arguments_json="{}")
    provider = _RecordingProvider(
        [
            ModelResponse(tool_calls=(call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="done"),
        ]
    )
    home = Path.home().resolve()
    result = ToolResult(
        content=(
            f"cwd={workspace.resolve()}\n"
            f'File "{workspace.resolve()}/src/main.py"\n'
            f"cache={home}/.cache/trace.txt"
        ),
        facts=ToolCompletionFacts(
            modified_paths=(f"{workspace.resolve()}/src/main.py",)
        ),
    )
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry(
            (_FixedPathTool(result, schema_path=f"{workspace.resolve()}/schema.json"),)
        ),
    )

    await agent.run_turn(f"inspect {workspace.resolve()}/src/main.py")

    first_user_message = provider.requests[0].messages[-1].content or ""
    tool_message = provider.requests[1].messages[-1].content or ""
    completed = next(
        event.data
        for event in session.events
        if isinstance(event.data, ToolCompletedData)
    )
    assert first_user_message == "inspect <workspace-root>/src/main.py"
    assert "<workspace-root>/src/main.py" in tool_message
    assert "<home>/.cache/trace.txt" in tool_message
    _assert_host_path_absent(tool_message, workspace.resolve())
    _assert_host_path_absent(tool_message, home)
    _assert_host_path_absent(completed.output, workspace.resolve())
    provider_tool = provider.requests[0].tools[0]
    assert "<workspace-root>/schema.json" in provider_tool.description
    assert provider_tool.parameters["properties"] == {
        "path": {"type": "string", "default": "<workspace-root>/schema.json"}
    }
    assert completed.facts.modified_paths == ("<workspace-root>/src/main.py",)
    assert completed.source_redaction_summary.kinds == (RedactionKind.HOST_PATH,)
    session.close()


@pytest.mark.asyncio
async def test_agent_replaces_unstable_tool_error_code(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = AgentSession.create(
        data_dir=tmp_path / "data",
        workspace=workspace,
        model="m",
    )
    call = ToolCall(id="call-code", name="failing_path", arguments_json="{}")
    provider = _RecordingProvider(
        [
            ModelResponse(tool_calls=(call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="done"),
        ]
    )
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry(
            (
                _FailingPathTool(
                    "failed",
                    code=f"invalid-{workspace.resolve()}",
                ),
            )
        ),
    )

    await agent.run_turn("run the failing tool")

    failure = next(
        event.data for event in session.events if isinstance(event.data, ToolFailedData)
    )
    assert failure.error_code == "invalid_tool_error_code"
    _assert_host_path_absent(
        provider.requests[1].messages[-1].content or "",
        workspace.resolve(),
    )
    session.close()


@pytest.mark.asyncio
async def test_agent_normalizes_tool_error_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = AgentSession.create(
        data_dir=tmp_path / "data",
        workspace=workspace,
        model="m",
    )
    call = ToolCall(id="call-error", name="failing_path", arguments_json="{}")
    provider = _RecordingProvider(
        [
            ModelResponse(tool_calls=(call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="done"),
        ]
    )
    raw_message = f'File "{workspace.resolve()}/src/main.py", under {Path.home()}/cache'
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((_FailingPathTool(raw_message),)),
    )

    await agent.run_turn("run the failing tool")

    failure = next(
        event.data for event in session.events if isinstance(event.data, ToolFailedData)
    )
    provider_error = provider.requests[1].messages[-1].content or ""
    assert failure.message == 'File "<workspace-root>/src/main.py", under <home>/cache'
    assert failure.message in provider_error
    _assert_host_path_absent(provider_error, workspace.resolve())
    _assert_host_path_absent(provider_error, Path.home().resolve())
    session.close()


@pytest.mark.asyncio
async def test_symlink_escape_failure_does_not_report_resolved_host_path(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside)
    session = AgentSession.create(
        data_dir=tmp_path / "data",
        workspace=workspace,
        model="m",
    )
    call = ToolCall(
        id="call-escape",
        name="read_file",
        arguments_json='{"path":"link.txt"}',
    )
    provider = _RecordingProvider(
        [
            ModelResponse(tool_calls=(call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="done"),
        ]
    )
    agent = MiniAgent(
        config=_config(tmp_path),
        provider=provider,
        session=session,
        tools=ToolRegistry((ReadFileTool(Workspace(workspace)),)),
    )

    await agent.run_turn("read the link")

    failure = next(
        event.data for event in session.events if isinstance(event.data, ToolFailedData)
    )
    provider_error = provider.requests[1].messages[-1].content or ""
    assert failure.message == "Path escapes workspace"
    _assert_host_path_absent(provider_error, outside.resolve())
    session.close()


@pytest.mark.asyncio
async def test_artifact_stores_normalized_tool_output(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = AgentSession.create(
        data_dir=tmp_path / "data",
        workspace=workspace,
        model="m",
    )
    call = ToolCall(id="call-artifact", name="fixed_path", arguments_json="{}")
    provider = _RecordingProvider(
        [
            ModelResponse(tool_calls=(call,), finish_reason=FinishReason.TOOL_CALLS),
            ModelResponse(content="done"),
        ]
    )
    artifacts = ArtifactStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    content = f"traceback: {workspace.resolve()}/src/main.py\n" + "x" * 100
    agent = MiniAgent(
        config=_config(tmp_path, inline_chars=20),
        provider=provider,
        session=session,
        tools=ToolRegistry((_FixedPathTool(ToolResult(content=content)),)),
        artifacts=artifacts,
    )

    await agent.run_turn("capture output")

    record = artifacts.records[0]
    stored = artifacts.read(record.artifact_id, offset=0, limit=record.char_count)
    assert "<workspace-root>/src/main.py" in stored
    _assert_host_path_absent(stored, workspace.resolve())
    assert record.redaction_kinds == (RedactionKind.HOST_PATH,)
    session.close()


def test_history_normalizes_full_event_before_pagination(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = AgentSession.create(
        data_dir=tmp_path / "data",
        workspace=workspace,
        model="m",
    )
    history = SessionHistory(session.events, workspace_root=workspace)
    snippets: list[str] = []
    start_offset = 0
    while True:
        page = history.read(
            from_event_id=1,
            to_event_id=1,
            max_events=1,
            max_chars=17,
            start_offset=start_offset,
        )
        snippets.extend(match.snippet for match in page.matches)
        if page.next_cursor is None:
            break
        start_offset = page.next_cursor.char_offset

    serialized_provider_view = "".join(snippets)
    assert "<workspace-root>" in serialized_provider_view
    _assert_host_path_absent(serialized_provider_view, workspace.resolve())
    session.close()
