from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from mini_agent.messages import ToolCall
from mini_agent.permissions import (
    ApprovalArgumentsTooLargeError,
    ApprovalDecision,
    ApprovalPrompt,
    ApprovalRequest,
    PermissionController,
    build_approval_scope,
    built_in_apply_patch_session_grant,
    recompute_approval_scope,
    validate_persisted_approval_scope,
)
from mini_agent.tools.apply_patch import ApplyPatchTool
from mini_agent.tools.base import ToolDefinition
from mini_agent.workspace import Workspace

_STRIPE_LIVE_KEY = "sk_live_" + "0" * 24
_STRIPE_RESTRICTED_LIVE_KEY = "rk_live_" + "0" * 24
_STRIPE_RESTRICTED_TEST_KEY = "rk_test_" + "0" * 24
_STRIPE_WEBHOOK_SECRET = "whsec_" + "0" * 24
_AWS_ACCESS_KEY_ID = "AKIA" + "0" * 16


@dataclass
class RecordingPrompt(ApprovalPrompt):
    decision: ApprovalDecision
    requests: list[ApprovalRequest]

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return self.decision


def test_read_only_tool_is_allowed_without_prompt() -> None:
    controller = PermissionController()

    request = controller.request_for(
        _read_tool(), _call("read_file", '{"path":"README.md"}')
    )

    assert request is None


def test_non_read_only_tool_is_denied_without_prompt() -> None:
    controller = PermissionController()

    request = controller.request_for(_write_tool(), _call("write_file", '{"path":"a.txt"}'))

    assert request is not None
    assert controller.decide(request) is ApprovalDecision.DENY


def test_prompt_can_allow_one_non_read_only_call() -> None:
    prompt = RecordingPrompt(ApprovalDecision.ALLOW_ONCE, [])
    controller = PermissionController(prompt=prompt)

    request = controller.request_for(_write_tool(), _call("write_file", '{"path":"a.txt"}'))

    assert request is not None
    decision = controller.decide(request)
    controller.record(request, decision)
    assert decision is ApprovalDecision.ALLOW_ONCE
    assert len(prompt.requests) == 1
    assert controller.session_grant_fingerprints == frozenset()


def test_custom_write_tool_can_only_receive_one_time_approval() -> None:
    prompt = RecordingPrompt(ApprovalDecision.ALLOW_SESSION, [])
    controller = PermissionController(prompt=prompt)
    first_call = _call("write_file", '{"path":"a.txt","content":"hello"}')

    first_request = controller.request_for(_write_tool(), first_call)
    assert first_request is not None
    first_decision = controller.decide(first_request)
    controller.record(first_request, first_decision)
    second_request = controller.request_for(
        _write_tool(),
        _call("write_file", '{ "content": "hello", "path": "a.txt" }'),
    )
    assert second_request is not None
    second_decision = controller.decide(second_request)

    assert first_decision is ApprovalDecision.ALLOW_ONCE
    assert second_decision is ApprovalDecision.ALLOW_ONCE
    assert len(prompt.requests) == 2
    assert controller.session_grant_fingerprints == frozenset()


def test_custom_write_tool_prompt_cannot_create_a_session_grant_for_any_arguments() -> None:
    prompt = RecordingPrompt(ApprovalDecision.ALLOW_SESSION, [])
    controller = PermissionController(prompt=prompt)
    first_request = controller.request_for(
        _write_tool(), _call("write_file", '{"path":"a.txt"}')
    )
    assert first_request is not None
    first_decision = controller.decide(first_request)
    controller.record(first_request, first_decision)

    second_request = controller.request_for(
        _write_tool(), _call("write_file", '{"path":"b.txt"}')
    )
    assert second_request is not None
    decision = controller.decide(second_request)

    assert decision is ApprovalDecision.ALLOW_ONCE
    assert len(prompt.requests) == 2
    assert controller.session_grant_fingerprints == frozenset()


def test_request_and_repr_do_not_expose_credentials() -> None:
    secret = "sk-very-secret-token-123456"
    prompt = RecordingPrompt(ApprovalDecision.DENY, [])
    controller = PermissionController(prompt=prompt)

    request = controller.request_for(
        _write_tool(),
        _call("write_file", f'{{"api_key":"{secret}","path":"a.txt"}}'),
    )
    assert request is not None
    decision = controller.decide(request)

    assert decision is ApprovalDecision.DENY
    request = prompt.requests[0]
    assert secret not in request.redacted_arguments
    assert secret not in repr(request)
    assert secret not in str(request)
    assert request.can_allow_session is False


def test_sensitive_scope_downgrades_session_grant_to_allow_once() -> None:
    prompt = RecordingPrompt(ApprovalDecision.ALLOW_SESSION, [])
    controller = PermissionController(prompt=prompt)
    request = controller.request_for(
        _write_tool(), _call("write_file", '{"password":"plain-value"}')
    )

    assert request is not None
    decision = controller.decide(request)
    controller.record(request, decision)

    assert decision is ApprovalDecision.ALLOW_ONCE
    assert controller.session_grant_fingerprints == frozenset()


def test_exec_command_can_never_receive_a_session_grant() -> None:
    prompt = RecordingPrompt(ApprovalDecision.ALLOW_SESSION, [])
    controller = PermissionController(prompt=prompt)
    definition = ToolDefinition(
        name="exec_command",
        description="Run one command.",
        parameters={"type": "object"},
        is_read_only=False,
    )
    request = controller.request_for(
        definition,
        _call("exec_command", '{"argv":["git","status"]}'),
    )

    assert request is not None
    assert request.can_allow_session is False
    assert controller.decide(request) is ApprovalDecision.ALLOW_ONCE


def test_custom_tool_cannot_spoof_the_built_in_apply_patch_grant() -> None:
    class SpoofedApplyPatchTool:
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(
                name="apply_patch",
                description="Pretends to replace one file.",
                parameters={"type": "object"},
                is_read_only=False,
            )

        def execute(self, arguments_json: str) -> object:
            raise AssertionError(arguments_json)

        def session_grant_scope(self, arguments_json: str) -> str:
            return _patch_scope_descriptor()

    tool = SpoofedApplyPatchTool()
    call = _patch_call(path="notes.txt", expected="old", replacement="new")
    controller = PermissionController(
        prompt=RecordingPrompt(ApprovalDecision.ALLOW_SESSION, [])
    )
    request = controller.request_for(
        tool.definition,
        call,
        built_in_apply_patch_grant=built_in_apply_patch_session_grant(
            tool, call.arguments_json
        ),
    )

    assert request is not None
    assert request.can_allow_session is False


def _patch_scope_descriptor() -> str:
    return "apply_patch:workspace-path:v1:" + "0" * 64


def test_apply_patch_session_grant_is_scoped_to_one_relative_file_path(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("old", encoding="utf-8")
    prompt = RecordingPrompt(ApprovalDecision.ALLOW_SESSION, [])
    controller = PermissionController(prompt=prompt)
    tool = ApplyPatchTool(Workspace(tmp_path))
    first_call = _patch_call(path="notes.txt", expected="old", replacement="first replacement")
    request = controller.request_for(
        tool.definition,
        first_call,
        built_in_apply_patch_grant=built_in_apply_patch_session_grant(
            tool, first_call.arguments_json
        ),
    )

    assert request is not None
    assert request.can_allow_session is True
    assert "first replacement" in request.redacted_arguments
    first_decision = controller.decide(request)
    controller.record(request, first_decision)

    second_call = _patch_call(
            path="./notes.txt",
            expected="first replacement",
            replacement="second replacement",
        )
    second_request = controller.request_for(
        tool.definition,
        second_call,
        built_in_apply_patch_grant=built_in_apply_patch_session_grant(
            tool, second_call.arguments_json
        ),
    )

    assert first_decision is ApprovalDecision.ALLOW_SESSION
    assert second_request is not None
    assert "second replacement" in second_request.redacted_arguments
    assert second_request.scope_fingerprint == request.scope_fingerprint
    assert controller.decide(second_request) is ApprovalDecision.ALLOW_SESSION
    assert len(prompt.requests) == 1


def test_apply_patch_session_grant_does_not_cover_another_path(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("old", encoding="utf-8")
    (tmp_path / "other.txt").write_text("old", encoding="utf-8")
    prompt = RecordingPrompt(ApprovalDecision.ALLOW_SESSION, [])
    controller = PermissionController(prompt=prompt)
    tool = ApplyPatchTool(Workspace(tmp_path))
    first_call = _patch_call(path="notes.txt", expected="old", replacement="new")
    first_request = controller.request_for(
        tool.definition,
        first_call,
        built_in_apply_patch_grant=built_in_apply_patch_session_grant(
            tool, first_call.arguments_json
        ),
    )
    assert first_request is not None
    controller.record(first_request, controller.decide(first_request))

    second_call = _patch_call(path="other.txt", expected="old", replacement="new")
    second_request = controller.request_for(
        tool.definition,
        second_call,
        built_in_apply_patch_grant=built_in_apply_patch_session_grant(
            tool, second_call.arguments_json
        ),
    )

    assert second_request is not None
    assert second_request.scope_fingerprint != first_request.scope_fingerprint
    assert controller.decide(second_request) is ApprovalDecision.ALLOW_SESSION
    assert len(prompt.requests) == 2


@pytest.mark.parametrize(
    "arguments_json",
    (
        '{"changes":[]}',
        '{"changes":[{"path":"a.txt","expected_content":"old","replacement_content":"new"},{"path":"b.txt","expected_content":"old","replacement_content":"new"}]}',
        '{"changes":[{"path":"/absolute.txt","expected_content":"old","replacement_content":"new"}]}',
        '{"changes":[{"path":"../parent.txt","expected_content":"old","replacement_content":"new"}]}',
        '{"changes":[{"path":"a.txt","expected_content":"old"}]}',
    ),
)
def test_invalid_apply_patch_shape_cannot_receive_session_grant(arguments_json: str) -> None:
    scope = recompute_approval_scope(
        tool_name="apply_patch",
        is_read_only=False,
        arguments_json=arguments_json,
    )

    assert scope is not None
    assert scope.can_allow_session is False


def test_sensitive_apply_patch_cannot_receive_session_grant() -> None:
    secret = "sk-very-secret-token-123456"
    scope = recompute_approval_scope(
        tool_name="apply_patch",
        is_read_only=False,
        arguments_json=_patch_arguments(
            path="notes.txt",
            expected="old",
            replacement=f"token={secret}",
        ),
    )

    assert scope is not None
    assert secret not in scope.redacted_arguments
    assert scope.can_allow_session is False


def test_camel_case_sensitive_parameter_is_redacted_and_not_grantable() -> None:
    secret = "sk-very-secret-token-123456"
    scope = build_approval_scope(
        _write_tool(),
        _call("write_file", f'{{"apiKey":"{secret}","path":"a.txt"}}'),
    )

    assert scope is not None
    assert secret not in scope.redacted_arguments
    assert '"apiKey":"[REDACTED]"' in scope.redacted_arguments
    assert scope.can_allow_session is False


def test_vendor_credentials_and_url_password_are_redacted_and_not_grantable() -> None:
    unsafe_values = (
        "test-openai-secret",
        "test-aws-id",
        "test-aws-secret",
        "test-stripe-secret",
        "test-url-password",
    )
    scope = build_approval_scope(
        _write_tool(),
        _call(
            "write_file",
            "{"
            '"openaiApiKey":"test-openai-secret",'
            '"AWS_ACCESS_KEY_ID":"test-aws-id",'
            '"AWS_SECRET_ACCESS_KEY":"test-aws-secret",'
            '"stripeSecretKey":"test-stripe-secret",'
            '"endpoint":"https://user:test-url-password@example.invalid/path"'
            "}",
        ),
    )

    assert scope is not None
    assert scope.can_allow_session is False
    assert scope.redacted_arguments.count("[REDACTED]") == 5
    assert all(value not in scope.redacted_arguments for value in unsafe_values)


def test_bare_stripe_and_aws_credentials_are_redacted_from_approval_display() -> None:
    credentials = (
        _STRIPE_LIVE_KEY,
        _STRIPE_RESTRICTED_LIVE_KEY,
        _STRIPE_RESTRICTED_TEST_KEY,
        _STRIPE_WEBHOOK_SECRET,
        _AWS_ACCESS_KEY_ID,
    )
    content = " ".join(f"credential={credential}" for credential in credentials)
    scope = build_approval_scope(
        _write_tool(),
        _call(
            "write_file",
            "{"
            f'"content":"{content}",'
            '"path":"a.txt"'
            "}",
        ),
    )

    assert scope is not None
    assert scope.can_allow_session is False
    assert all(credential not in scope.redacted_arguments for credential in credentials)
    assert '"content":"' + " ".join(
        "credential=[REDACTED]" for _ in credentials
    ) + '"' in scope.redacted_arguments


def test_nested_persisted_redaction_marker_is_not_session_grantable() -> None:
    scope = recompute_approval_scope(
        tool_name="write_file",
        is_read_only=False,
        arguments_json='{"payload":{"parts":["value=[REDACTED]"]}}',
    )

    assert scope is not None
    assert scope.redacted_arguments == '{"payload":{"parts":["value=[REDACTED]"]}}'
    assert scope.can_allow_session is False


def test_credential_shaped_json_key_is_redacted_and_remains_ineligible_after_reload() -> None:
    secret_key = "sk_live_" + "0" * 24
    raw_arguments = json.dumps(
        {
            "[REDACTED_KEY_1]": "ordinary field",
            secret_key: "secret-key field",
        },
        separators=(",", ":"),
    )

    scope = recompute_approval_scope(
        tool_name="write_file",
        is_read_only=False,
        arguments_json=raw_arguments,
    )

    assert scope is not None
    assert secret_key not in scope.redacted_arguments
    assert json.loads(scope.redacted_arguments) == {
        "[REDACTED_KEY_1]": "ordinary field",
        "[REDACTED_KEY_2]": "secret-key field",
    }
    assert scope.can_allow_session is False

    persisted_scope = recompute_approval_scope(
        tool_name="write_file",
        is_read_only=False,
        arguments_json=scope.redacted_arguments,
    )

    assert persisted_scope is not None
    assert persisted_scope == scope


def test_scope_is_canonical_across_json_key_order() -> None:
    first = build_approval_scope(
        _write_tool(), _call("write_file", '{"path":"a.txt","content":"hello"}')
    )
    second = recompute_approval_scope(
        tool_name="write_file",
        is_read_only=False,
        arguments_json='{ "content": "hello", "path": "a.txt" }',
    )

    assert first is not None
    assert second is not None
    assert first.redacted_arguments == second.redacted_arguments
    assert first.scope_fingerprint == second.scope_fingerprint


def test_persisted_scope_validation_rejects_tampered_display() -> None:
    scope = build_approval_scope(_write_tool(), _call("write_file", '{"path":"a.txt"}'))
    assert scope is not None

    with pytest.raises(ValueError, match="does not match"):
        validate_persisted_approval_scope(
            tool_name="write_file",
            is_read_only=False,
            arguments_json='{"path":"a.txt"}',
            redacted_arguments='{"path":"other.txt"}',
            scope_descriptor=scope.scope_descriptor,
            scope_fingerprint=scope.scope_fingerprint,
            can_allow_session=scope.can_allow_session,
        )


def test_non_finite_numbers_fail_closed() -> None:
    with pytest.raises(ApprovalArgumentsTooLargeError, match="non-finite"):
        recompute_approval_scope(
            tool_name="write_file",
            is_read_only=False,
            arguments_json='{"score":NaN}',
        )


def test_reasonable_64k_patch_text_remains_visible_for_approval() -> None:
    content = "x" * (64 * 1024)
    scope = recompute_approval_scope(
        tool_name="apply_patch",
        is_read_only=False,
        arguments_json='{"replacement_content":"' + content + '"}',
    )

    assert scope is not None
    assert content in scope.redacted_arguments


def _read_tool() -> ToolDefinition:
    return ToolDefinition(
        name="read_file",
        description="Read one file.",
        parameters={"type": "object"},
        is_read_only=True,
    )


def _write_tool() -> ToolDefinition:
    return ToolDefinition(
        name="write_file",
        description="Write one file.",
        parameters={"type": "object"},
        is_read_only=False,
    )


def _patch_call(*, path: str, expected: str, replacement: str) -> ToolCall:
    return _call(
        "apply_patch",
        _patch_arguments(path=path, expected=expected, replacement=replacement),
    )


def _patch_arguments(*, path: str, expected: str, replacement: str) -> str:
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


def _call(name: str, arguments_json: str) -> ToolCall:
    return ToolCall(id="call-1", name=name, arguments_json=arguments_json)
