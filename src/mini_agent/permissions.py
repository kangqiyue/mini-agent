"""Explicit approval scopes for local tool execution.

The approval scope is deliberately a pure, reproducible value.  A session can
therefore validate an approval event after a restart without depending on a UI
implementation or on an in-memory permission cache.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from mini_agent.events import ApprovalDecision
from mini_agent.messages import ToolCall
from mini_agent.redaction import redact_json_text
from mini_agent.tools.base import ToolDefinition

APPROVAL_SCOPE_DOMAIN = "mini-agent.approval-scope"
APPROVAL_SCOPE_VERSION = 1
_APPLY_PATCH_SCOPE_PATTERN = re.compile(
    r"^apply_patch:workspace-path:v1:[0-9a-f]{64}$"
)

# These limits protect the approval boundary itself.  They are intentionally
# independent from an individual tool's input limits: a patch may legitimately
# contain tens of KiB of source text and must remain fully visible to the user.
MAX_APPROVAL_ARGUMENT_BYTES = 256 * 1024
MAX_APPROVAL_DISPLAY_CHARS = 256 * 1024
MAX_APPROVAL_ARGUMENT_DEPTH = 16
MAX_APPROVAL_ARGUMENT_ELEMENTS = 4_096

_FINGERPRINT_PATTERN = r"^[0-9a-f]{64}$"
_SENSITIVE_PARAMETER_NAMES = frozenset(
    {
        "apikey",
        "accesstoken",
        "authorization",
        "connectionstring",
        "cookie",
        "password",
        "privatekey",
        "secret",
        "token",
    }
)


_BUILTIN_APPLY_PATCH_GRANT_AUTHORITY = object()


@dataclass(frozen=True)
class _BuiltInApplyPatchSessionGrant:
    """Internal proof that the live built-in patch tool derived a scope.

    A descriptor string is deliberately not enough: extensions can choose an
    arbitrary tool name and implement arbitrary methods.  The factory below
    recognises the exact built-in implementation before minting this value.
    """

    scope_descriptor: str
    authority: object


def built_in_apply_patch_session_grant(
    tool: object, arguments_json: str
) -> _BuiltInApplyPatchSessionGrant | None:
    """Return a session-grant proof only for the exact built-in patch tool.

    This is intentionally a narrow capability rather than a general tool
    extension point.  Custom write tools, including one that calls itself
    ``apply_patch``, remain one-time approval only.
    """

    # Keep the concrete dependency here rather than in tools.base: this is a
    # policy decision, not a generic tool capability.  ``type`` rather than
    # ``isinstance`` prevents an extension subclass from replacing preflight
    # or descriptor semantics while inheriting the trusted name.
    from mini_agent.tools.apply_patch import ApplyPatchTool

    if type(tool) is not ApplyPatchTool:
        return None
    scope_descriptor = tool.session_grant_scope(arguments_json)
    if scope_descriptor is None or not _is_apply_patch_session_scope(scope_descriptor):
        return None
    return _BuiltInApplyPatchSessionGrant(
        scope_descriptor=scope_descriptor,
        authority=_BUILTIN_APPLY_PATCH_GRANT_AUTHORITY,
    )


class ApprovalArgumentsTooLargeError(ValueError):
    """Approval input exceeds a hard safety limit and must fail closed."""


class ApprovalScopeValidationError(ValueError):
    """Persisted approval metadata does not match its canonical tool scope."""


class ApprovalScope(BaseModel):
    """Canonical, redacted scope of one non-read-only tool call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str = Field(min_length=1, max_length=128)
    redacted_arguments: str = Field(min_length=2, max_length=MAX_APPROVAL_DISPLAY_CHARS)
    scope_descriptor: str = Field(min_length=1, max_length=256)
    scope_fingerprint: str = Field(pattern=_FINGERPRINT_PATTERN)
    can_allow_session: bool


class ApprovalRequest(BaseModel):
    """The safe, stable information presented to an approval UI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_call_id: str = Field(min_length=1, max_length=256)
    tool_name: str = Field(min_length=1, max_length=128)
    redacted_arguments: str = Field(min_length=2, max_length=MAX_APPROVAL_DISPLAY_CHARS)
    scope_descriptor: str = Field(min_length=1, max_length=256)
    scope_fingerprint: str = Field(pattern=_FINGERPRINT_PATTERN)
    can_allow_session: bool


class ApprovalPrompt(Protocol):
    """A UI boundary that asks the user to approve one tool scope."""

    def decide(self, request: ApprovalRequest) -> ApprovalDecision: ...


class PermissionController:
    """Apply read-only and user-granted permissions to a tool call."""

    def __init__(
        self,
        *,
        prompt: ApprovalPrompt | None = None,
        session_grant_fingerprints: Iterable[str] = (),
    ) -> None:
        self._prompt = prompt
        self._session_grant_fingerprints = _validate_grant_fingerprints(
            session_grant_fingerprints
        )

    @property
    def session_grant_fingerprints(self) -> frozenset[str]:
        """Exact grants suitable for explicit session persistence."""

        return frozenset(self._session_grant_fingerprints)

    def request_for(
        self,
        tool_definition: ToolDefinition,
        tool_call: ToolCall,
        *,
        built_in_apply_patch_grant: _BuiltInApplyPatchSessionGrant | None = None,
    ) -> ApprovalRequest | None:
        """Build a safe approval request for a write operation.

        Returning ``None`` means the tool is read-only. Every write returns a
        request, including a scope that already has a session grant, so the
        event stream records why that concrete call was allowed.
        """

        scope = build_approval_scope(
            tool_definition,
            tool_call,
            built_in_apply_patch_grant=built_in_apply_patch_grant,
        )
        if scope is None:
            return None
        return ApprovalRequest(tool_call_id=tool_call.id, **scope.model_dump())

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        """Resolve a persisted request without mutating durable grant state."""

        if (
            request.can_allow_session
            and request.scope_fingerprint in self._session_grant_fingerprints
        ):
            return ApprovalDecision.ALLOW_SESSION
        if self._prompt is None:
            return ApprovalDecision.DENY
        decision = self._prompt.decide(request)
        if decision is ApprovalDecision.ALLOW_SESSION and not request.can_allow_session:
            return ApprovalDecision.ALLOW_ONCE
        return decision

    def record(self, request: ApprovalRequest, decision: ApprovalDecision) -> None:
        """Apply a decision only after its approval event was persisted."""

        if decision is ApprovalDecision.ALLOW_SESSION:
            if not request.can_allow_session:
                raise ValueError("Sensitive approval scopes cannot be granted for a session")
            self._session_grant_fingerprints.add(request.scope_fingerprint)

    def revoke(self, scope_fingerprint: str) -> None:
        """Remove an exact grant after an operation reaches unknown state."""

        self._session_grant_fingerprints.discard(scope_fingerprint)


def build_approval_scope(
    tool_definition: ToolDefinition,
    tool_call: ToolCall,
    *,
    built_in_apply_patch_grant: _BuiltInApplyPatchSessionGrant | None = None,
) -> ApprovalScope | None:
    """Derive the one canonical approval scope from a definition and call.

    This is the normal public entry point used by execution.  It contains no
    I/O, prompting, or mutable state and is safe to call during recovery.
    """

    if tool_definition.name != tool_call.name:
        raise ValueError("Tool definition name does not match tool call name")
    session_scope_descriptor = None
    if built_in_apply_patch_grant is not None:
        if (
            built_in_apply_patch_grant.authority
            is not _BUILTIN_APPLY_PATCH_GRANT_AUTHORITY
        ):
            raise ValueError("Invalid built-in apply_patch session grant")
        session_scope_descriptor = built_in_apply_patch_grant.scope_descriptor
    return recompute_approval_scope(
        tool_name=tool_definition.name,
        is_read_only=tool_definition.is_read_only,
        arguments_json=tool_call.arguments_json,
        session_scope_descriptor=session_scope_descriptor,
        allow_builtin_apply_patch_session=built_in_apply_patch_grant is not None,
    )


def recompute_approval_scope(
    *,
    tool_name: str,
    is_read_only: bool,
    arguments_json: str,
    session_scope_descriptor: str | None = None,
    allow_builtin_apply_patch_session: bool = True,
) -> ApprovalScope | None:
    """Recompute an approval scope from persisted tool-request fields.

    ``ToolRequestedData`` contains exactly these fields, so recovery can use
    this function to validate an event stream without a live tool registry.
    """

    if is_read_only:
        return None
    parsed_arguments = _parse_approval_arguments(arguments_json)
    sensitive_parameters = _contains_sensitive_parameter(parsed_arguments)
    # Tool-request events are redacted before persistence. On recovery their
    # arguments may therefore contain this marker instead of the original
    # credential. Preserve the original no-session-grant boundary.
    contains_persisted_redaction = _contains_redaction_marker(parsed_arguments)
    # ``redact_json_text`` knows the common snake_case spellings.  Normalize
    # parameter names here as well so a provider cannot bypass it with e.g.
    # ``apiKey`` and make a credential appear in an approval prompt.
    safe_arguments = _redact_sensitive_parameters(parsed_arguments)
    redaction = redact_json_text(
        json.dumps(
            safe_arguments,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    )
    if len(redaction.text) > MAX_APPROVAL_DISPLAY_CHARS:
        raise ApprovalArgumentsTooLargeError(
            "Approval display exceeds its hard limit; split the tool call"
        )
    scope_descriptor = _scope_descriptor(
        tool_name=tool_name,
        redacted_arguments=redaction.text,
        session_scope_descriptor=session_scope_descriptor,
    )
    can_allow_session = (
        allow_builtin_apply_patch_session
        and tool_name == "apply_patch"
        and _is_apply_patch_session_scope(scope_descriptor)
        and redaction.match_count == 0
        and not sensitive_parameters
        and not contains_persisted_redaction
    )
    if scope_descriptor is None:
        scope_descriptor = _fallback_scope_descriptor(tool_name, redaction.text)
    return ApprovalScope(
        tool_name=tool_name,
        redacted_arguments=redaction.text,
        scope_descriptor=scope_descriptor,
        scope_fingerprint=_scope_fingerprint(tool_name, scope_descriptor),
        can_allow_session=can_allow_session,
    )


def validate_persisted_approval_scope(
    *,
    tool_name: str,
    is_read_only: bool,
    arguments_json: str,
    redacted_arguments: str,
    scope_fingerprint: str,
    can_allow_session: bool,
    scope_descriptor: str,
) -> ApprovalScope:
    """Validate persisted approval fields against a recomputed scope.

    The returned scope is useful to callers that need the canonical values
    after validation.  A mismatch is corruption rather than a recoverable UI
    error, because the user may otherwise have approved different arguments.
    """

    try:
        scope = recompute_approval_scope(
            tool_name=tool_name,
            is_read_only=is_read_only,
            arguments_json=arguments_json,
            session_scope_descriptor=scope_descriptor,
        )
    except ValueError as error:
        if can_allow_session:
            raise ApprovalScopeValidationError(
                "Persisted approval scope does not match tool call"
            ) from error
        scope = _recompute_persisted_one_time_scope(
            tool_name=tool_name,
            is_read_only=is_read_only,
            redacted_arguments=redacted_arguments,
            scope_descriptor=scope_descriptor,
        )
    if scope is None:
        raise ApprovalScopeValidationError("Read-only tool calls cannot have approval scopes")
    # Event sanitisation may canonicalise a non-persistent tool request (for
    # example, adding exec_command's default cwd) after the approval UI saw
    # it.  The durable approval display remains the authoritative one-time
    # scope in that case.  This fallback is intentionally unavailable to any
    # session-capable scope.
    if not can_allow_session and not _scope_fields_match(
        scope,
        redacted_arguments=redacted_arguments,
        scope_descriptor=scope_descriptor,
        scope_fingerprint=scope_fingerprint,
        can_allow_session=can_allow_session,
    ):
        scope = _recompute_persisted_one_time_scope(
            tool_name=tool_name,
            is_read_only=is_read_only,
            redacted_arguments=redacted_arguments,
            scope_descriptor=scope_descriptor,
        )
        if scope is None:
            raise ApprovalScopeValidationError("Read-only tool calls cannot have approval scopes")
    is_legacy_generic_grant = is_legacy_generic_session_grant(
        tool_name=tool_name,
        redacted_arguments=redacted_arguments,
        scope_descriptor=scope_descriptor,
        can_allow_session=can_allow_session,
    )
    if not _scope_fields_match(
        scope,
        redacted_arguments=redacted_arguments,
        scope_descriptor=scope_descriptor,
        scope_fingerprint=scope_fingerprint,
        can_allow_session=can_allow_session,
        allow_legacy_can_allow_session=is_legacy_generic_grant,
    ):
        raise ApprovalScopeValidationError("Persisted approval scope does not match tool call")
    return scope


def _recompute_persisted_one_time_scope(
    *,
    tool_name: str,
    is_read_only: bool,
    redacted_arguments: str,
    scope_descriptor: str,
) -> ApprovalScope | None:
    try:
        return recompute_approval_scope(
            tool_name=tool_name,
            is_read_only=is_read_only,
            arguments_json=redacted_arguments,
            session_scope_descriptor=scope_descriptor,
            allow_builtin_apply_patch_session=False,
        )
    except ValueError as error:
        raise ApprovalScopeValidationError(
            "Persisted approval scope does not match tool call"
        ) from error


def _scope_fields_match(
    scope: ApprovalScope,
    *,
    redacted_arguments: str,
    scope_descriptor: str,
    scope_fingerprint: str,
    can_allow_session: bool,
    allow_legacy_can_allow_session: bool = False,
) -> bool:
    return (
        scope.redacted_arguments == redacted_arguments
        and scope.scope_descriptor == scope_descriptor
        and scope.scope_fingerprint == scope_fingerprint
        and (
            scope.can_allow_session == can_allow_session
            or allow_legacy_can_allow_session
        )
    )


def is_persisted_apply_patch_session_grant(
    *, tool_name: str, scope_descriptor: str, can_allow_session: bool
) -> bool:
    """Return whether a durable event may restore a session grant.

    Only the versioned descriptor written for the built-in patch operation is
    durable authority.  Older generic allow-session events are deliberately
    inert after recovery.
    """

    return (
        can_allow_session
        and tool_name == "apply_patch"
        and _is_apply_patch_session_scope(scope_descriptor)
    )


def _parse_approval_arguments(arguments_json: str) -> Mapping[str, object]:
    if len(arguments_json.encode("utf-8")) > MAX_APPROVAL_ARGUMENT_BYTES:
        raise ApprovalArgumentsTooLargeError("Approval arguments exceed the hard input limit")
    try:
        parsed_value: object = json.loads(
            arguments_json, parse_constant=_reject_non_finite
        )
    except json.JSONDecodeError as error:
        raise ValueError("Tool arguments must be valid JSON") from error
    if not isinstance(parsed_value, dict):
        raise ValueError("Tool arguments must be a JSON object")
    parsed_arguments = cast(Mapping[str, object], parsed_value)
    _enforce_argument_shape_limits(parsed_arguments)
    canonical_arguments = json.dumps(
        parsed_arguments,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    if len(canonical_arguments.encode("utf-8")) > MAX_APPROVAL_ARGUMENT_BYTES:
        raise ApprovalArgumentsTooLargeError("Approval arguments exceed the hard canonical limit")
    return parsed_arguments


def _reject_non_finite(value: str) -> object:
    raise ApprovalArgumentsTooLargeError(
        f"Approval arguments cannot contain non-finite JSON number {value!r}"
    )


def _enforce_argument_shape_limits(arguments: Mapping[str, object]) -> None:
    element_count = 0
    pending: list[tuple[object, int]] = [(arguments, 1)]
    while pending:
        value, depth = pending.pop()
        if depth > MAX_APPROVAL_ARGUMENT_DEPTH:
            raise ApprovalArgumentsTooLargeError("Approval arguments exceed the nesting limit")
        if isinstance(value, Mapping):
            items = tuple(cast(Mapping[str, object], value).items())
            element_count += len(items)
            pending.extend((item, depth + 1) for _, item in items)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            items = cast(Sequence[object], value)
            element_count += len(items)
            pending.extend((item, depth + 1) for item in items)
        if element_count > MAX_APPROVAL_ARGUMENT_ELEMENTS:
            raise ApprovalArgumentsTooLargeError("Approval arguments exceed the element limit")


def _contains_sensitive_parameter(arguments: Mapping[str, object]) -> bool:
    pending: list[object] = [arguments]
    while pending:
        value = pending.pop()
        if isinstance(value, Mapping):
            mapping = cast(Mapping[str, object], value)
            if any(_normalise_parameter_name(key) in _SENSITIVE_PARAMETER_NAMES for key in mapping):
                return True
            pending.extend(mapping.values())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            pending.extend(cast(Sequence[object], value))
    return False


def _contains_redaction_marker(arguments: Mapping[str, object]) -> bool:
    """Return whether persisted arguments contain a nested redaction marker."""

    pending: list[object] = [arguments]
    while pending:
        value = pending.pop()
        if isinstance(value, str) and _is_redaction_marker(value):
            return True
        if isinstance(value, Mapping):
            mapping = cast(Mapping[str, object], value)
            if any(_is_redaction_marker(key) for key in mapping):
                return True
            pending.extend(mapping.values())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            pending.extend(cast(Sequence[object], value))
    return False


def _is_redaction_marker(value: str) -> bool:
    return "[REDACTED]" in value or "[REDACTED_KEY_" in value


def _redact_sensitive_parameters(value: object) -> object:
    if isinstance(value, Mapping):
        safe_mapping: dict[str, object] = {}
        for key, item in cast(Mapping[str, object], value).items():
            if _normalise_parameter_name(key) in _SENSITIVE_PARAMETER_NAMES:
                safe_mapping[key] = "[REDACTED]"
            else:
                safe_mapping[key] = _redact_sensitive_parameters(item)
        return safe_mapping
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_redact_sensitive_parameters(item) for item in cast(Sequence[object], value)]
    return value


def _normalise_parameter_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _scope_descriptor(
    *,
    tool_name: str,
    redacted_arguments: str,
    session_scope_descriptor: str | None,
) -> str | None:
    """Choose a validated durable scope descriptor, never raw patch content."""

    if tool_name == "apply_patch":
        if session_scope_descriptor is None:
            return None
        # The descriptor binds a canonical workspace root and target path, but
        # neither plaintext value belongs in the transcript. Recovery can only
        # verify the versioned, hashed descriptor shape; a live ApplyPatchTool
        # constructs it again for every new request, so grants cannot cross
        # workspace roots.
        fallback_descriptor = _fallback_scope_descriptor(tool_name, redacted_arguments)
        if (
            not _is_apply_patch_session_scope(session_scope_descriptor)
            and session_scope_descriptor != fallback_descriptor
        ):
            raise ValueError("Invalid apply_patch approval scope descriptor")
        return session_scope_descriptor
    expected_descriptor = _fallback_scope_descriptor(tool_name, redacted_arguments)
    if session_scope_descriptor is not None and session_scope_descriptor != expected_descriptor:
        raise ValueError("Unexpected approval scope descriptor for this tool")
    return session_scope_descriptor


def _is_apply_patch_session_scope(scope_descriptor: str | None) -> bool:
    return scope_descriptor is not None and _APPLY_PATCH_SCOPE_PATTERN.fullmatch(
        scope_descriptor
    ) is not None


def is_legacy_generic_session_grant(
    *,
    tool_name: str,
    redacted_arguments: str,
    scope_descriptor: str,
    can_allow_session: bool,
) -> bool:
    """Recognise historical generic grants so they load without restoring.

    Versions before the built-in-only rule allowed generic write scopes.  A
    session containing such an already-durable approval remains readable, but
    its grant is never made active again.  Current code never writes this
    shape with ``can_allow_session=True``.
    """

    try:
        persisted_arguments = _parse_approval_arguments(redacted_arguments)
    except (ApprovalArgumentsTooLargeError, ValueError):
        return False
    return (
        can_allow_session
        and not is_persisted_apply_patch_session_grant(
            tool_name=tool_name,
            scope_descriptor=scope_descriptor,
            can_allow_session=can_allow_session,
        )
        and scope_descriptor == _fallback_scope_descriptor(tool_name, redacted_arguments)
        and not _contains_sensitive_parameter(persisted_arguments)
        and not _contains_redaction_marker(persisted_arguments)
        and redact_json_text(redacted_arguments).match_count == 0
    )


def _fallback_scope_descriptor(tool_name: str, redacted_arguments: str) -> str:
    digest = hashlib.sha256(redacted_arguments.encode()).hexdigest()
    if tool_name != "apply_patch":
        return f"arguments:v1:{digest}"
    return f"apply_patch:arguments:v1:{digest}"


def _scope_fingerprint(tool_name: str, scope_value: str) -> str:
    fingerprint_input = (
        f"{APPROVAL_SCOPE_DOMAIN}:v{APPROVAL_SCOPE_VERSION}"
        f"\x00{tool_name}\x00{scope_value}"
    ).encode()
    return hashlib.sha256(fingerprint_input).hexdigest()


def _validate_grant_fingerprints(grants: Iterable[str]) -> set[str]:
    validated_grants: set[str] = set()
    for fingerprint in grants:
        if re.fullmatch(_FINGERPRINT_PATTERN, fingerprint) is None:
            raise ValueError("Persisted session grant has an invalid fingerprint")
        validated_grants.add(fingerprint)
    return validated_grants
