"""Credential redaction applied before persistence or display."""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast
from urllib.parse import unquote_plus

from mini_agent.redaction_types import RedactionKind


@dataclass(frozen=True)
class RedactionResult:
    text: str
    match_count: int
    matched_kinds: frozenset[RedactionKind]


_SENSITIVE_KEY_COMPACTS = frozenset(
    {
        "accesskeyid",
        "apikey",
        "accesstoken",
        "authtoken",
        "authorization",
        "connectionstring",
        "cookie",
        "databaseuri",
        "databaseurl",
        "dsn",
        "idtoken",
        "password",
        "privatekey",
        "refreshtoken",
        "secret",
        "secretaccesskey",
        "secretbinary",
        "secretkey",
        "secretstring",
        "secretvalue",
        "token",
    }
)

_SENSITIVE_KEY_SUFFIXES = (
    "_access_key_id",
    "_access_token",
    "_api_key",
    "_auth_token",
    "_connection_string",
    "_cookie",
    "_database_uri",
    "_database_url",
    "_dsn",
    "_id_token",
    "_password",
    "_private_key",
    "_refresh_token",
    "_secret",
    "_secret_access_key",
    "_secret_binary",
    "_secret_key",
    "_secret_string",
    "_secret_value",
    "_token",
)


_PATTERNS = (
    (
        RedactionKind.PRIVATE_KEY,
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    (
        RedactionKind.AUTHORIZATION,
        re.compile(
            r"(?im)\bauthorization\s*[:=]\s*(?:bearer|basic)\s+[^\s,;\"']+"
        ),
    ),
    (
        RedactionKind.NAMED_SECRET,
        re.compile(r"(?im)(\bcookie\s*:\s*)[^\r\n]+"),
    ),
    (
        RedactionKind.NAMED_SECRET,
        re.compile(
            r"(?i)\b(?:postgres(?:ql)?|mysql(?:\+\w+)?|mongodb(?:\+\w+)?|"
            r"redis(?:s)?|amqp(?:s)?|mssql|oracle)://[^\s'\"]+"
        ),
    ),
    (
        RedactionKind.NAMED_SECRET,
        re.compile(r"(?i)\bhttps?://[^\s/:@]+:[^\s/@]+@[^\s'\"<>]+"),
    ),
    (
        RedactionKind.SECRET_PREFIX,
        re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{16,}\b"),
    ),
    (
        RedactionKind.SECRET_PREFIX,
        re.compile(r"\b(?:rk_(?:live|test)|whsec)_[A-Za-z0-9]{16,}\b"),
    ),
    (
        RedactionKind.SECRET_PREFIX,
        re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    ),
    (RedactionKind.SECRET_PREFIX, re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")),
    (
        RedactionKind.SECRET_PREFIX,
        re.compile(r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[^\s]+)"),
    ),
    (
        RedactionKind.SECRET_PREFIX,
        re.compile(
            r"\b(?:AIza[A-Za-z0-9_-]{35}|hf_[A-Za-z0-9]{20,}|"
            r"glpat-[A-Za-z0-9_-]{20,})\b"
        ),
    ),
    (
        RedactionKind.SECRET_PREFIX,
        re.compile(
            r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{16,}\."
            r"[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}(?![A-Za-z0-9_-])"
        ),
    ),
)

# A bounded tool result may end halfway through a credential. Full-token
# patterns above intentionally require realistic minimum lengths to avoid
# redacting ordinary prose, so apply this stricter *tail-only* pass afterwards.
# One payload character is enough to prove that a truncated token was emitted;
# a bare documented prefix such as ``sk_live_`` remains readable.
_PARTIAL_SECRET_AT_END_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"(?:"
    r"sk_(?:live|test)_[A-Za-z0-9_-]+|"
    r"rk_(?:live|test)_[A-Za-z0-9_-]+|"
    r"whsec_[A-Za-z0-9_-]+|"
    r"(?:AKIA|ASIA)[A-Za-z0-9_-]+|"
    r"sk-[A-Za-z0-9_-]+|"
    r"ghp_[A-Za-z0-9_-]+|"
    r"github_pat_[A-Za-z0-9_-]+|"
    r"xox[A-Za-z0-9_-]+"
    r")"
    r"(?P<line_ending>\r?\n?)\Z"
)
_UNTERMINATED_PRIVATE_KEY_AT_END_PATTERN = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*\Z"
)

_NAMED_VALUE_PATTERN = re.compile(
    r"(?P<key_fragment>(?:(?P<quote>[\"'])[A-Za-z_][A-Za-z0-9_-]*(?P=quote)|"
    r"\b[A-Za-z_][A-Za-z0-9_-]*\b))"
    r"(?P<separator>[ \t]*[:=][ \t]*)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;\"']+)",
    re.IGNORECASE,
)

# Process URLs before generic ``name=value`` matches.  The latter can consume
# an entire URL as the value of its ``https:`` prefix, skipping a sensitive
# query parameter nested inside it.
_HTTP_URL_PATTERN = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)


def redact_text(text: str, *, truncated_at_end: bool = False) -> RedactionResult:
    """Redact credentials, including partial trailing tokens when truncated.

    ``truncated_at_end`` must only be set by the component that applied the
    bound. A short prefix at the natural end of complete text can be ordinary
    documentation and is therefore left intact by default.
    """

    redacted_text = text
    match_count = 0
    matched_kinds: set[RedactionKind] = set()

    redacted_text, url_query_count = _redact_http_url_queries(redacted_text)
    match_count += url_query_count
    if url_query_count:
        matched_kinds.add(RedactionKind.NAMED_SECRET)

    for kind, pattern in _PATTERNS:
        redacted_text, pattern_count = pattern.subn("[REDACTED]", redacted_text)
        match_count += pattern_count
        if pattern_count:
            matched_kinds.add(kind)

    if truncated_at_end:
        redacted_text, partial_prefix_count = _PARTIAL_SECRET_AT_END_PATTERN.subn(
            lambda match: f"[REDACTED]{match.group('line_ending')}",
            redacted_text,
        )
        match_count += partial_prefix_count
        if partial_prefix_count:
            matched_kinds.add(RedactionKind.SECRET_PREFIX)

        redacted_text, unterminated_key_count = (
            _UNTERMINATED_PRIVATE_KEY_AT_END_PATTERN.subn(
                "[REDACTED]",
                redacted_text,
            )
        )
        match_count += unterminated_key_count
        if unterminated_key_count:
            matched_kinds.add(RedactionKind.PRIVATE_KEY)

    named_match_count = 0

    def redact_named_value(match: re.Match[str]) -> str:
        nonlocal named_match_count
        key_fragment = match.group("key_fragment")
        key = key_fragment.strip("\"'")
        if not is_sensitive_key(key):
            return match.group(0)
        if _is_redaction_marker(match.group("value")):
            return match.group(0)
        named_match_count += 1
        return f"{key_fragment}{match.group('separator')}[REDACTED]"

    redacted_text = _NAMED_VALUE_PATTERN.sub(redact_named_value, redacted_text)
    match_count += named_match_count
    if named_match_count:
        matched_kinds.add(RedactionKind.NAMED_SECRET)

    return RedactionResult(
        text=redacted_text,
        match_count=match_count,
        matched_kinds=frozenset(matched_kinds),
    )


def _redact_http_url_queries(text: str) -> tuple[str, int]:
    """Redact sensitive HTTP(S) parameter values without changing URL structure.

    Raw parameter key spelling is retained for diagnostics, including percent
    encoding. Classification uses decoded keys, so ``api%5Fkey`` receives the
    same protection as ``api_key``. Query strings and ``key=value`` fragments
    both carry credentials in common OAuth and callback URLs.
    """

    match_count = 0

    def redact_url(match: re.Match[str]) -> str:
        nonlocal match_count
        url = match.group(0)
        before_fragment, fragment_separator, fragment = url.partition("#")
        prefix, query_separator, query = before_fragment.partition("?")

        safe_query, query_count = _redact_url_parameters(query)
        safe_fragment, fragment_count = _redact_url_parameters(fragment)
        match_count += query_count + fragment_count
        return f"{prefix}{query_separator}{safe_query}{fragment_separator}{safe_fragment}"

    return _HTTP_URL_PATTERN.sub(redact_url, text), match_count


def _redact_url_parameters(parameters: str) -> tuple[str, int]:
    """Return a query- or fragment-like ``&`` parameter string safely."""

    if not parameters:
        return parameters, 0

    match_count = 0
    safe_parts: list[str] = []
    for parameter in parameters.split("&"):
        raw_key, separator, raw_value = parameter.partition("=")
        decoded_key = unquote_plus(raw_key)
        if not separator or not is_sensitive_key(decoded_key):
            safe_parts.append(parameter)
            continue
        if _is_redaction_marker(unquote_plus(raw_value)):
            safe_parts.append(parameter)
            continue
        safe_parts.append(f"{raw_key}{separator}[REDACTED]")
        match_count += 1

    return "&".join(safe_parts), match_count


def redact_json_text(text: str) -> RedactionResult:
    """Redact strings and sensitive keys in a JSON value, then canonicalize it."""

    try:
        parsed: object = json.loads(text)
    except json.JSONDecodeError:
        return redact_text(text)
    safe_value, match_count, matched_kinds = _redact_json_value(parsed)
    return RedactionResult(
        text=json.dumps(
            safe_value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        match_count=match_count,
        matched_kinds=frozenset(matched_kinds),
    )


def _redact_json_value(value: object) -> tuple[object, int, set[RedactionKind]]:
    if isinstance(value, str):
        result = redact_text(value)
        return result.text, result.match_count, set(result.matched_kinds)

    if isinstance(value, Mapping):
        safe_mapping: dict[str, object] = {}
        match_count = 0
        matched_kinds: set[RedactionKind] = set()
        mapping = cast(Mapping[str, object], value)
        safe_keys, key_count, key_kinds = _redact_json_mapping_keys(mapping)
        match_count += key_count
        matched_kinds.update(key_kinds)
        for key, item in mapping.items():
            safe_key = safe_keys[key]
            if is_sensitive_key(key):
                if item == "[REDACTED]":
                    safe_mapping[safe_key] = item
                    continue
                safe_mapping[safe_key] = "[REDACTED]"
                match_count += 1
                matched_kinds.add(RedactionKind.NAMED_SECRET)
                continue
            safe_item, item_count, item_kinds = _redact_json_value(item)
            safe_mapping[safe_key] = safe_item
            match_count += item_count
            matched_kinds.update(item_kinds)
        return safe_mapping, match_count, matched_kinds

    if isinstance(value, Sequence) and not isinstance(value, bytes):
        safe_items: list[object] = []
        match_count = 0
        matched_kinds: set[RedactionKind] = set()
        for item in cast(Sequence[object], value):
            safe_item, item_count, item_kinds = _redact_json_value(item)
            safe_items.append(safe_item)
            match_count += item_count
            matched_kinds.update(item_kinds)
        return safe_items, match_count, matched_kinds

    return value, 0, set()


def _redact_json_mapping_keys(
    mapping: Mapping[str, object],
) -> tuple[dict[str, str], int, set[RedactionKind]]:
    """Replace credential-shaped JSON keys without dropping colliding fields.

    JSON object keys are persisted just like values.  A provider can therefore
    put a credential in an otherwise ordinary field name.  The replacement
    indices are assigned from sorted source keys so canonical output is stable
    across input ordering.  Existing placeholder-looking *ordinary* keys are
    reserved first, which prevents a redacted key from silently overwriting a
    user field.
    """

    key_results = {key: redact_text(key) for key in mapping}
    redacted_keys = sorted(
        key for key, result in key_results.items() if result.match_count > 0
    )
    reserved_keys = {
        key for key, result in key_results.items() if result.match_count == 0
    }
    safe_keys = {key: key for key in mapping}
    next_index = 1

    for key in redacted_keys:
        while True:
            replacement = f"[REDACTED_KEY_{next_index}]"
            next_index += 1
            if replacement not in reserved_keys:
                break
        safe_keys[key] = replacement
        reserved_keys.add(replacement)

    match_count = sum(key_results[key].match_count for key in redacted_keys)
    matched_kinds: set[RedactionKind] = set()
    for key in redacted_keys:
        matched_kinds.update(key_results[key].matched_kinds)
    return safe_keys, match_count, matched_kinds


def _normalize_key(key: str) -> str:
    with_word_boundaries = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
    with_word_boundaries = re.sub(
        r"([a-z0-9])([A-Z])", r"\1_\2", with_word_boundaries
    )
    return re.sub(r"[^a-z0-9]+", "_", with_word_boundaries.lower()).strip("_")


def is_sensitive_key(key: str) -> bool:
    """Return whether a field name conventionally carries a credential.

    Exact names use a separator-free form so ``apiKey``, ``api_key``, and
    ``apikey`` have identical treatment. Vendor prefixes still use the
    normalized snake/camel-case suffix form, avoiding false positives such as
    ``token_count`` or ``api_key_hint``.
    """

    normalized_key = _normalize_key(key)
    compact_key = normalized_key.replace("_", "")
    return (
        compact_key in _SENSITIVE_KEY_COMPACTS
        or normalized_key.endswith(_SENSITIVE_KEY_SUFFIXES)
    )


def _is_redaction_marker(value: str) -> bool:
    return value.strip("\"'") == "[REDACTED]"
