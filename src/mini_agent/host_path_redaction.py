"""Normalize known local host paths before data crosses a provider boundary."""

from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from mini_agent.messages import ConversationMessage, FinishReason, ModelRequest, ModelResponse
from mini_agent.redaction import RedactionResult, redact_json_text, redact_text
from mini_agent.redaction_types import RedactionKind
from mini_agent.tools.base import ToolDefinition

# Tool-call arguments and tool schemas are both bounded at their own external
# boundaries. Keep the defensive canonicalization boundary equally bounded so
# a malformed model-constructed value cannot make redaction recursive or
# unbounded before it reaches a provider or durable transcript.
_MAX_JSON_CANONICALIZATION_BYTES = 256 * 1024
_MAX_JSON_CANONICALIZATION_DEPTH = 16
_MAX_JSON_CANONICALIZATION_NODES = 4_096

# These are macOS-owned public aliases, rather than a general symlink walk.
# Each candidate is still resolved locally and retained only when it reaches
# the supplied canonical root.  That keeps host-path redaction predictable
# and avoids treating arbitrary symlink spellings as sensitive paths.
_DARWIN_PUBLIC_ROOT_ALIASES: tuple[tuple[str, str], ...] = (
    ("/private/tmp", "/tmp"),
    ("/private/var", "/var"),
)


@dataclass(frozen=True)
class _PathReplacement:
    path_text: str
    placeholder: str
    is_case_insensitive: bool


def redact_host_paths(
    text: str,
    *,
    workspace_root: Path,
    home_directory: Path | None = None,
    truncated_at_start: bool = False,
    truncated_at_end: bool = False,
) -> RedactionResult:
    """Replace exact workspace/home prefixes while preserving relative suffixes."""

    home = home_directory
    if home is None:
        try:
            home = Path.home()
        except (RuntimeError, OSError):
            home = None
    path_values = [(workspace_root, "<workspace-root>")]
    if home is not None:
        path_values.append((home, "<home>"))
    replacements = _ordered_replacements(tuple(path_values))
    safe_text = text
    match_count = 0
    for replacement in replacements:
        path_text = replacement.path_text
        placeholder = replacement.placeholder
        pattern = _path_prefix_pattern(
            path_text,
            is_case_insensitive=replacement.is_case_insensitive,
        )
        safe_text, replacement_count = pattern.subn(placeholder, safe_text)
        match_count += replacement_count
        if truncated_at_start:
            safe_text, start_count = _redact_truncated_start(
                safe_text,
                path_text=path_text,
                placeholder=placeholder,
                is_case_insensitive=replacement.is_case_insensitive,
            )
            match_count += start_count
        if truncated_at_end:
            safe_text, end_count = _redact_truncated_end(
                safe_text,
                path_text=path_text,
                placeholder=placeholder,
                is_case_insensitive=replacement.is_case_insensitive,
            )
            match_count += end_count
        if truncated_at_start and truncated_at_end:
            safe_text, middle_count = _redact_whole_truncated_fragment(
                safe_text,
                path_text=path_text,
                placeholder=placeholder,
                is_case_insensitive=replacement.is_case_insensitive,
            )
            match_count += middle_count

    return RedactionResult(
        text=safe_text,
        match_count=match_count,
        matched_kinds=(
            frozenset((RedactionKind.HOST_PATH,)) if match_count else frozenset()
        ),
    )


_MINIMUM_PATH_FRAGMENT_CHARS = 8


def _redact_truncated_start(
    text: str,
    *,
    path_text: str,
    placeholder: str,
    is_case_insensitive: bool,
) -> tuple[str, int]:
    for start in range(1, len(path_text) - _MINIMUM_PATH_FRAGMENT_CHARS + 1):
        suffix = path_text[start:]
        match = _path_text_pattern(
            suffix,
            is_case_insensitive=is_case_insensitive,
        ).match(text)
        if match is None:
            continue
        following = text[match.end() : match.end() + 1]
        if following and (following.isalnum() or following in "._~-"):
            continue
        return f"{placeholder}{text[match.end():]}", 1
    return text, 0


def _redact_truncated_end(
    text: str,
    *,
    path_text: str,
    placeholder: str,
    is_case_insensitive: bool,
) -> tuple[str, int]:
    maximum_length = min(len(text), len(path_text) - 1)
    for length in range(maximum_length, _MINIMUM_PATH_FRAGMENT_CHARS - 1, -1):
        prefix = path_text[:length]
        match = re.search(
            rf"{_path_text_pattern_source(prefix)}$",
            text,
            flags=_path_pattern_flags(is_case_insensitive),
        )
        if match is not None:
            return f"{text[:match.start()]}{placeholder}", 1
    return text, 0


def _redact_whole_truncated_fragment(
    text: str,
    *,
    path_text: str,
    placeholder: str,
    is_case_insensitive: bool,
) -> tuple[str, int]:
    canonical_fragment = text.replace("\\/", "/")
    normalized_fragment = _normalize_path_case(
        canonical_fragment,
        is_case_insensitive=is_case_insensitive,
    )
    normalized_path = _normalize_path_case(
        path_text,
        is_case_insensitive=is_case_insensitive,
    )
    if (
        len(canonical_fragment) < _MINIMUM_PATH_FRAGMENT_CHARS
        or normalized_fragment not in normalized_path
    ):
        return text, 0
    return placeholder, 1


def redact_host_paths_in_json_value(
    value: object,
    *,
    workspace_root: Path,
    home_directory: Path | None = None,
) -> object:
    """Recursively normalize strings in one bounded JSON-compatible value."""

    _validate_json_value_for_canonicalization(value)
    return _redact_host_paths_in_bounded_json_value(
        value,
        workspace_root=workspace_root,
        home_directory=home_directory,
    )


def _redact_host_paths_in_bounded_json_value(
    value: object,
    *,
    workspace_root: Path,
    home_directory: Path | None,
) -> object:
    """Normalize a value whose depth and nodes were already checked."""

    if isinstance(value, str):
        return redact_host_paths(
            value,
            workspace_root=workspace_root,
            home_directory=home_directory,
        ).text
    if isinstance(value, Mapping):
        safe_mapping: dict[str, object] = {}
        for key, item in cast(Mapping[str, object], value).items():
            safe_key = cast(
                str,
                _redact_host_paths_in_bounded_json_value(
                    key,
                    workspace_root=workspace_root,
                    home_directory=home_directory,
                ),
            )
            if safe_key in safe_mapping:
                raise ValueError("Host path normalization caused a JSON key collision")
            safe_mapping[safe_key] = _redact_host_paths_in_bounded_json_value(
                item,
                workspace_root=workspace_root,
                home_directory=home_directory,
            )
        return safe_mapping
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _redact_host_paths_in_bounded_json_value(
                item,
                workspace_root=workspace_root,
                home_directory=home_directory,
            )
            for item in cast(Sequence[object], value)
        ]
    return value


def _validate_json_value_for_canonicalization(value: object) -> None:
    """Reject values that cannot safely cross the JSON redaction boundary."""

    node_count = 0
    pending: list[tuple[object, int]] = [(value, 1)]
    while pending:
        current, depth = pending.pop()
        node_count += 1
        if node_count > _MAX_JSON_CANONICALIZATION_NODES:
            raise ValueError("JSON redaction value exceeds the node limit")
        if depth > _MAX_JSON_CANONICALIZATION_DEPTH:
            raise ValueError("JSON redaction value exceeds the nesting limit")

        if isinstance(current, Mapping):
            mapping = cast(Mapping[object, object], current)
            for key, item in mapping.items():
                if not isinstance(key, str):
                    raise ValueError("JSON redaction value has a non-string object key")
                pending.append((item, depth + 1))
            continue
        if isinstance(current, list):
            items = cast(list[object], current)
            pending.extend((item, depth + 1) for item in items)
            continue
        if current is None or isinstance(current, (bool, int, str)):
            continue
        if isinstance(current, float) and math.isfinite(current):
            continue
        raise ValueError("JSON redaction value is not JSON-compatible")


def normalize_host_path_text(text: str, *, workspace_root: Path) -> str:
    """Return text safe to include in a provider-facing payload."""

    return redact_host_paths(text, workspace_root=workspace_root).text


def redact_persisted_text(
    text: str,
    *,
    workspace_root: Path,
    truncated_at_start: bool = False,
    truncated_at_end: bool = False,
) -> RedactionResult:
    """Apply the durable-text policy once and report only new removals.

    Storage has a stricter boundary than provider requests: both the known
    local roots and credentials are removed.  The operation is idempotent, so
    callers can safely accept text already normalized by an upstream boundary
    without inflating its redaction summary.
    """

    path_result = redact_host_paths(
        text,
        workspace_root=workspace_root,
        truncated_at_start=truncated_at_start,
        truncated_at_end=truncated_at_end,
    )
    credential_result = redact_text(
        path_result.text,
        truncated_at_end=truncated_at_end,
    )
    return RedactionResult(
        text=credential_result.text,
        match_count=path_result.match_count + credential_result.match_count,
        matched_kinds=path_result.matched_kinds | credential_result.matched_kinds,
    )


def normalize_conversation_message_host_paths(
    message: ConversationMessage,
    *,
    workspace_root: Path,
) -> ConversationMessage:
    """Normalize all provider-visible text in one canonical message."""

    safe_content = (
        _sanitize_provider_text(message.content, workspace_root=workspace_root)
        if message.content is not None
        else None
    )
    safe_tool_calls = tuple(
        tool_call.model_copy(
            update={
                "arguments_json": _sanitize_provider_json_text(
                    tool_call.arguments_json,
                    workspace_root=workspace_root,
                )
            }
        )
        for tool_call in message.tool_calls
    )
    return message.model_copy(
        update={"content": safe_content, "tool_calls": safe_tool_calls}
    )


def normalize_tool_definition_host_paths(
    definition: ToolDefinition,
    *,
    workspace_root: Path,
) -> ToolDefinition:
    """Normalize provider-visible tool metadata without changing its contract."""

    return definition.model_copy(
        update={
            "description": _sanitize_provider_text(
                definition.description,
                workspace_root=workspace_root,
            ),
            "parameters": _sanitize_provider_json_value(
                definition.parameters,
                workspace_root=workspace_root,
            ),
        }
    )


def normalize_model_request_host_paths(
    request: ModelRequest,
    *,
    workspace_root: Path,
) -> ModelRequest:
    """Normalize every provider-facing field in a canonical model request."""

    return request.model_copy(
        update={
            "messages": tuple(
                normalize_conversation_message_host_paths(
                    message,
                    workspace_root=workspace_root,
                )
                for message in request.messages
            ),
            "tools": tuple(
                normalize_tool_definition_host_paths(
                    definition,
                    workspace_root=workspace_root,
                )
                for definition in request.tools
            ),
        }
    )


def normalize_model_response_host_paths(
    response: ModelResponse,
    *,
    workspace_root: Path,
) -> ModelResponse:
    """Normalize provider output before it becomes durable conversation state."""

    safe_content = response.content
    if safe_content is not None:
        safe_content = redact_host_paths(
            safe_content,
            workspace_root=workspace_root,
            truncated_at_end=response.finish_reason
            in {FinishReason.LENGTH, FinishReason.OTHER},
        ).text
    safe_tool_calls = tuple(
        tool_call.model_copy(
            update={
                "arguments_json": _sanitize_provider_json_text(
                    tool_call.arguments_json,
                    workspace_root=workspace_root,
                )
            }
        )
        for tool_call in response.tool_calls
    )
    return response.model_copy(
        update={"content": safe_content, "tool_calls": safe_tool_calls}
    )


def _sanitize_provider_text(text: str, *, workspace_root: Path) -> str:
    return redact_text(
        normalize_host_path_text(text, workspace_root=workspace_root)
    ).text


def _sanitize_provider_json_text(text: str, *, workspace_root: Path) -> str:
    """Return a fixed-point-safe JSON object for a provider-visible field.

    Path matching must happen *after* JSON parsing: legal JSON may escape a
    slash as ``\\/``, which is semantically a normal slash but does not match a
    raw-text path prefix. Each pass parses, normalizes every string key/value,
    then applies credential redaction. Two monotonic passes establish the
    fixed point while the hard bound makes malformed constructed models safe.
    """

    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("Provider JSON text must be valid UTF-8") from error
    if len(encoded) > _MAX_JSON_CANONICALIZATION_BYTES:
        raise ValueError("Provider JSON text exceeds the UTF-8 byte limit")

    current = _canonical_json_object(text)
    for _ in range(3):
        parsed = _parse_bounded_json_object(current)
        path_safe_value = redact_host_paths_in_json_value(
            parsed,
            workspace_root=workspace_root,
        )
        path_safe_json = _serialize_json_object(path_safe_value)
        next_text = redact_json_text(path_safe_json).text
        if next_text == current:
            return next_text
        current = next_text
    raise ValueError("Provider JSON text did not reach a redaction fixed point")


def _sanitize_provider_json_value(
    value: dict[str, object],
    *,
    workspace_root: Path,
) -> dict[str, object]:
    serialized = _serialize_json_object(value)
    redacted = _sanitize_provider_json_text(serialized, workspace_root=workspace_root)
    return _parse_bounded_json_object(redacted)


def _canonical_json_object(text: str) -> str:
    """Parse and serialize one JSON object before its first redaction pass."""

    return _serialize_json_object(_parse_bounded_json_object(text))


def _parse_bounded_json_object(text: str) -> dict[str, object]:
    try:
        value: object = json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ValueError("Provider JSON text must be a valid JSON object") from error
    if not isinstance(value, dict):
        raise ValueError("Provider JSON text must be a JSON object")
    object_value = cast(dict[str, object], value)
    _validate_json_value_for_canonicalization(object_value)
    return object_value


def _serialize_json_object(value: object) -> str:
    _validate_json_value_for_canonicalization(value)
    if not isinstance(value, Mapping):
        raise ValueError("Provider JSON text must be a JSON object")
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        encoded = serialized.encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError) as error:
        raise ValueError("Provider JSON text must be JSON-compatible") from error
    if len(encoded) > _MAX_JSON_CANONICALIZATION_BYTES:
        raise ValueError("Provider JSON text exceeds the UTF-8 byte limit")
    return serialized


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def _ordered_replacements(
    values: tuple[tuple[Path, str], ...],
) -> tuple[_PathReplacement, ...]:
    replacements: dict[str, _PathReplacement] = {}
    for path, placeholder in values:
        path_text = _canonical_path_text(path)
        if path_text is None:
            continue
        is_case_insensitive = _has_case_insensitive_darwin_root(path)
        for spelling in _known_path_spellings(path_text):
            replacement = _PathReplacement(
                path_text=spelling,
                placeholder=placeholder,
                is_case_insensitive=is_case_insensitive,
            )
            replacements.setdefault(spelling, replacement)
            # Preserve non-separator JSON escapes such as an otherwise valid
            # quote in a local directory name. Slash escaping itself is handled
            # by the mixed-separator pattern below.
            json_escaped_path = json.dumps(spelling, ensure_ascii=False)[1:-1]
            replacements.setdefault(
                json_escaped_path,
                _PathReplacement(
                    path_text=json_escaped_path,
                    placeholder=placeholder,
                    is_case_insensitive=is_case_insensitive,
                ),
            )
    return tuple(
        sorted(
            replacements.values(),
            key=lambda replacement: len(replacement.path_text),
            reverse=True,
        )
    )


def _path_prefix_pattern(
    path_text: str,
    *,
    is_case_insensitive: bool = False,
) -> re.Pattern[str]:
    """Match a path prefix whose forward slashes may be JSON escaped.

    This operates on ordinary text rather than parsing the complete message as
    JSON. It catches strings embedded in prose such as ``\"\\/private...\"``
    while retaining the normal prefix boundary rules.
    """

    return re.compile(
        rf"{_path_text_pattern_source(path_text)}(?=$|(?:/|\\/)|[^\w.~-])",
        flags=_path_pattern_flags(is_case_insensitive),
    )


def _path_text_pattern(
    path_text: str,
    *,
    is_case_insensitive: bool = False,
) -> re.Pattern[str]:
    return re.compile(
        _path_text_pattern_source(path_text),
        flags=_path_pattern_flags(is_case_insensitive),
    )


def _path_text_pattern_source(path_text: str) -> str:
    """Allow each POSIX separator to be literal or the JSON ``\\/`` spelling."""

    escaped_parts = (re.escape(part) for part in path_text.split("/"))
    return r"(?:/|\\/)".join(escaped_parts)


def _canonical_path_text(path: Path) -> str | None:
    try:
        resolved = path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None
    path_text = str(resolved)
    if path_text == resolved.anchor:
        return None
    return path_text.rstrip("/\\")


def _known_path_spellings(canonical_path_text: str) -> tuple[str, ...]:
    """Return verified spellings for one root without discovering symlinks."""

    spellings = [canonical_path_text]
    if sys.platform != "darwin":
        return tuple(spellings)

    for canonical_root, alias_root in _DARWIN_PUBLIC_ROOT_ALIASES:
        if canonical_path_text == canonical_root:
            suffix = ""
        elif canonical_path_text.startswith(f"{canonical_root}/"):
            suffix = canonical_path_text[len(canonical_root) :]
        else:
            continue
        alias = f"{alias_root}{suffix}"
        if _canonical_path_text(Path(alias)) == canonical_path_text:
            spellings.append(alias)
    return tuple(spellings)


def _has_case_insensitive_darwin_root(path: Path) -> bool:
    """Detect a case-insensitive Darwin volume without creating probe files."""

    if sys.platform != "darwin":
        return False
    try:
        current = path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False
    while not current.exists() and current != current.parent:
        current = current.parent
    while current != current.parent:
        alternate_name = _alternate_case_name(current.name)
        if alternate_name is not None:
            try:
                if (current.parent / alternate_name).samefile(current):
                    return True
            except OSError:
                pass
        current = current.parent
    return False


def _alternate_case_name(name: str) -> str | None:
    for index, character in enumerate(name):
        if character.islower():
            return f"{name[:index]}{character.upper()}{name[index + 1:]}"
        if character.isupper():
            return f"{name[:index]}{character.lower()}{name[index + 1:]}"
    return None


def _path_pattern_flags(is_case_insensitive: bool) -> re.RegexFlag:
    return re.IGNORECASE if is_case_insensitive else re.NOFLAG


def _normalize_path_case(text: str, *, is_case_insensitive: bool) -> str:
    return text.casefold() if is_case_insensitive else text
