"""Bounded safety handling for ``exec_command`` argument vectors.

The agent stores model tool calls before executing them.  Positional command
arguments therefore need their own credential boundary: generic JSON
redaction cannot infer that the value following ``--password`` is secret.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import cast

from mini_agent.redaction import is_sensitive_key, redact_text
from mini_agent.redaction_types import RedactionKind

_MAX_EXEC_COMMAND_ARGUMENT_BYTES = 256 * 1024
_SAFE_REJECTED_ARGUMENTS_JSON = '{"argv":["[REDACTED]"],"cwd":"[REDACTED]"}'
_REDACTION_MARKERS = ("[REDACTED]", "[REDACTED_KEY_")
_CURL_CREDENTIAL_SHORT_OPTIONS = frozenset(("u", "U"))
_CURL_CREDENTIAL_LONG_OPTIONS = frozenset(("user", "proxy-user"))
_PASSWORD_SHORT_OPTIONS_BY_COMMAND = {
    "mariadb": frozenset(("p",)),
    "mongosh": frozenset(("p",)),
    "mysql": frozenset(("p",)),
    "redis-cli": frozenset(("a",)),
    "sshpass": frozenset(("p",)),
}
_CREDENTIAL_LONG_OPTIONS_BY_COMMAND = {
    "redis-cli": frozenset(("pass",)),
}
_CONTAINER_CLI_GLOBAL_OPTIONS_WITH_VALUES = frozenset(
    (
        "--config",
        "--context",
        "--host",
        "--log-level",
        "--tlscacert",
        "--tlscert",
        "--tlskey",
        "-H",
        "-l",
    )
)
_ENV_FLAG_OPTIONS = frozenset(("-i", "--ignore-environment"))
_ENV_OPTIONS_WITH_VARIABLE_NAME = frozenset(("-u", "--unset"))
_CREDENTIAL_BEARING_COMMANDS = frozenset(
    (
        "curl",
        "docker",
        "mariadb",
        "mongosh",
        "mysql",
        "podman",
        "redis-cli",
        "sshpass",
    )
)
# Shells whose ``-c <string>`` form hides a command inside a single argv value,
# so a credential-bearing command inside that string bypasses argv-level
# option detection (e.g. ``sh -c "mysql -pSECRET"``).
_SHELL_COMMANDS = frozenset(("sh", "bash", "dash", "zsh", "ksh", "ash"))
_MAX_CREDENTIAL_COMMAND_CANDIDATES = 64


@dataclass(frozen=True)
class ExecCommandArgumentSanitization:
    """A bounded, JSON-safe representation of one command tool call."""

    arguments_json: str
    has_sensitive_content: bool
    redaction_match_count: int
    redaction_kinds: frozenset[RedactionKind]


@dataclass(frozen=True)
class _ParsedExecCommandArguments:
    argv: list[str]
    cwd: str
    timeout_seconds: int | float | None
    has_timeout_seconds: bool

    def as_json_object(self) -> dict[str, object]:
        value: dict[str, object] = {"argv": self.argv, "cwd": self.cwd}
        if self.has_timeout_seconds:
            value["timeout_seconds"] = self.timeout_seconds
        return value


@dataclass(frozen=True)
class _CommandInvocation:
    """The command whose option meanings apply to an argv vector."""

    name: str
    option_start_index: int
    sensitive_wrapper_indices: frozenset[int] = frozenset()


def sanitize_exec_command_arguments(arguments_json: str) -> ExecCommandArgumentSanitization:
    """Return a durable-safe argv JSON representation.

    Invalid command shapes become a fixed safe sentinel.  The original value
    remains available only to the immediate execution preflight, which rejects
    it; it must never be used as durable assistant or approval data.
    """

    parsed = _parse_bounded_arguments(arguments_json)
    if parsed is None:
        return ExecCommandArgumentSanitization(
            arguments_json=_SAFE_REJECTED_ARGUMENTS_JSON,
            has_sensitive_content=True,
            redaction_match_count=1,
            redaction_kinds=frozenset((RedactionKind.NAMED_SECRET,)),
        )

    argv = parsed.argv
    cwd = parsed.cwd
    sensitive_indices = _sensitive_argv_indices(argv)
    has_sensitive_cwd = _is_sensitive_or_redacted_text(cwd)
    if not sensitive_indices and not has_sensitive_cwd:
        return ExecCommandArgumentSanitization(
            arguments_json=_serialize_arguments(parsed.as_json_object()),
            has_sensitive_content=False,
            redaction_match_count=0,
            redaction_kinds=frozenset(),
        )

    safe_argv = [
        "[REDACTED]" if index in sensitive_indices else value
        for index, value in enumerate(argv)
    ]
    safe_arguments: dict[str, object] = {
        "argv": safe_argv,
        "cwd": "[REDACTED]" if has_sensitive_cwd else cwd,
    }
    if parsed.has_timeout_seconds:
        safe_arguments["timeout_seconds"] = parsed.timeout_seconds
    redaction_match_count = sum(
        original != replacement for original, replacement in zip(argv, safe_argv, strict=True)
    ) + int(cwd != safe_arguments["cwd"])
    return ExecCommandArgumentSanitization(
        arguments_json=_serialize_arguments(safe_arguments),
        has_sensitive_content=True,
        redaction_match_count=redaction_match_count,
        redaction_kinds=(
            frozenset((RedactionKind.NAMED_SECRET,)) if redaction_match_count else frozenset()
        ),
    )


def has_sensitive_exec_command_arguments(*, argv: tuple[str, ...], cwd: str) -> bool:
    """Return whether raw parsed command input must be rejected before launch."""

    return bool(_sensitive_argv_indices(list(argv))) or _is_sensitive_or_redacted_text(cwd)


def _parse_bounded_arguments(arguments_json: str) -> _ParsedExecCommandArguments | None:
    try:
        if len(arguments_json.encode("utf-8")) > _MAX_EXEC_COMMAND_ARGUMENT_BYTES:
            return None
        parsed: object = json.loads(arguments_json)
    except (UnicodeEncodeError, json.JSONDecodeError, RecursionError):
        return None
    if not isinstance(parsed, dict):
        return None
    value = cast(dict[str, object], parsed)
    if set(value) - {"argv", "cwd", "timeout_seconds"}:
        return None
    raw_argv = value.get("argv")
    raw_cwd = value.get("cwd", ".")
    timeout_seconds = value.get("timeout_seconds")
    argv_items = cast(list[object], raw_argv) if isinstance(raw_argv, list) else None
    if (
        argv_items is None
        or not argv_items
        or any(not isinstance(item, str) for item in argv_items)
        or not isinstance(raw_cwd, str)
    ):
        return None
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
    ):
        return None
    return _ParsedExecCommandArguments(
        argv=cast(list[str], argv_items),
        cwd=raw_cwd,
        timeout_seconds=timeout_seconds,
        has_timeout_seconds="timeout_seconds" in value,
    )


def _shell_wrapper_sensitive_command_string_indices(
    argv: list[str],
    *,
    invocation: _CommandInvocation,
) -> set[int]:
    """Redact a shell ``-c <string>`` that hides a credential-bearing command.

    ``sh -c "mysql -pSECRET ..."`` places a command inside a single argv value,
    so argv-level credential flag detection never runs on its options. The
    string's whitespace-split tokens are re-checked with the same per-command
    credential-flag detection used for a direct argv, so a credential-bearing
    command is redacted only when its credential flag (or an env assignment) is
    actually present -- a benign ``sh -c "curl -s http://localhost/health"`` is
    left intact, matching the unwrapped ``curl -s …`` behavior.

    The whole string is redacted because quoting makes per-value redaction
    unreliable. ``-c`` clustered with other shell options (e.g. ``bash -ic …``)
    is not detected; only the standalone ``-c`` form is.
    """
    if invocation.name not in _SHELL_COMMANDS:
        return set()
    command_string_index = _shell_command_string_index(argv, invocation)
    if command_string_index is None:
        return set()
    command_string = argv[command_string_index]
    sub_argv = command_string.split()
    if not sub_argv:
        return set()
    sub_invocation = _command_invocation(sub_argv)
    if sub_invocation is None:
        # An opaque inner form (e.g. another env wrapper) hides command text.
        return {command_string_index}
    if _command_specific_sensitive_argv_indices(sub_argv, invocation=sub_invocation):
        return {command_string_index}
    return set()


def _shell_command_string_index(
    argv: list[str],
    invocation: _CommandInvocation,
) -> int | None:
    """Return the command-string index for a shell ``-c`` form, or None."""
    for index in range(invocation.option_start_index, len(argv)):
        if argv[index] == "-c" and index + 1 < len(argv):
            return index + 1
    return None


def _sensitive_argv_indices(argv: list[str]) -> set[int]:
    invocation = _command_invocation(argv)
    if invocation is None:
        # Unsupported env forms may hide a command in a string or change the
        # command's execution context.  Fail closed without retaining input.
        return set(range(len(argv)))

    sensitive_indices = {
        index for index, value in enumerate(argv) if _is_sensitive_or_redacted_text(value)
    }
    sensitive_indices.update(
        _conservative_wrapper_sensitive_long_option_indices(argv)
    )
    sensitive_indices.update(
        _command_specific_sensitive_argv_indices(argv, invocation=invocation)
    )
    sensitive_indices.update(
        _shell_wrapper_sensitive_command_string_indices(argv, invocation=invocation)
    )
    return sensitive_indices


def _command_specific_sensitive_argv_indices(
    argv: list[str],
    *,
    invocation: _CommandInvocation,
) -> set[int]:
    """Identify opaque credential values that generic text redaction cannot infer.

    Short flags such as ``-p`` and ``-u`` are deliberately not treated as
    globally sensitive: widely used commands assign unrelated meanings to
    them.  Keep this list tied to commands whose documented option semantics
    accept credentials, so safe commands such as ``mkdir -p`` and ``docker
    run -p`` retain their ordinary argument vectors.
    """

    sensitive_indices = set(invocation.sensitive_wrapper_indices)
    command_indices = [
        index
        for index in range(len(argv))
        if _command_name_at(argv, index) in _CREDENTIAL_BEARING_COMMANDS
    ]
    if len(command_indices) > _MAX_CREDENTIAL_COMMAND_CANDIDATES:
        return set(range(len(argv)))

    for command_index in command_indices:
        command = _command_name_at(argv, command_index)
        option_start_index = command_index + 1
        if command == "curl":
            sensitive_indices.update(
                _credential_option_value_indices(
                    argv,
                    start_index=option_start_index,
                    short_options=_CURL_CREDENTIAL_SHORT_OPTIONS,
                    long_options=_CURL_CREDENTIAL_LONG_OPTIONS,
                )
            )
            continue

        short_options = _PASSWORD_SHORT_OPTIONS_BY_COMMAND.get(command)
        if short_options is not None:
            sensitive_indices.update(
                _credential_option_value_indices(
                    argv,
                    start_index=option_start_index,
                    short_options=short_options,
                    long_options=_CREDENTIAL_LONG_OPTIONS_BY_COMMAND.get(command, frozenset()),
                )
            )
            continue

        if command in {"docker", "podman"}:
            login_index = _container_login_subcommand_index(
                argv,
                start_index=option_start_index,
            )
            if login_index is not None:
                sensitive_indices.update(
                    _credential_option_value_indices(
                        argv,
                        start_index=login_index + 1,
                        short_options=frozenset(("p",)),
                        long_options=frozenset(),
                    )
                )
    return sensitive_indices


def _conservative_wrapper_sensitive_long_option_indices(argv: list[str]) -> set[int]:
    """Find generic sensitive long options without trusting an unknown wrapper.

    A wrapper's ``--`` can precede the target command, so globally stopping at
    the first delimiter would miss that target's options.  The first value
    after each delimiter remains literal to preserve the established direct
    command behavior.  Later long options are conservatively rejected because
    they can belong to a target hidden by an arbitrary wrapper.
    """

    sensitive_indices: set[int] = set()
    is_first_value_after_double_dash = False
    for index in range(1, len(argv)):
        value = argv[index]
        if value == "--":
            is_first_value_after_double_dash = True
            continue
        if is_first_value_after_double_dash:
            is_first_value_after_double_dash = False
            continue
        if not value.startswith("--") or len(value) == 2:
            continue
        option, separator, _ = value[2:].partition("=")
        if not option or not is_sensitive_key(option):
            continue
        sensitive_indices.add(index)
        if not separator and index + 1 < len(argv) and argv[index + 1] != "--":
            sensitive_indices.add(index + 1)
    return sensitive_indices


def _command_invocation(argv: list[str]) -> _CommandInvocation | None:
    if not argv:
        return None
    if _command_name_at(argv, 0) != "env":
        return _CommandInvocation(name=_command_name_at(argv, 0), option_start_index=1)
    return _env_command_invocation(argv)


def _env_command_invocation(argv: list[str]) -> _CommandInvocation | None:
    """Unwrap only the non-ambiguous subset of ``env`` invocation syntax.

    Environment assignments are rejected before launch and fully redacted from
    durable arguments: even an innocuous-looking variable name can carry an
    opaque credential value.  Options that can split a command string or
    change its working directory are deliberately unsupported.  ``--`` ends
    option processing, but not the assignment prefix accepted by ``env``.
    """

    index = 1
    has_ended_options = False
    sensitive_wrapper_indices: set[int] = set()
    while index < len(argv):
        value = argv[index]
        if not has_ended_options and value == "--":
            has_ended_options = True
            index += 1
            continue
        if not has_ended_options and value in _ENV_FLAG_OPTIONS:
            index += 1
            continue
        if not has_ended_options and value in _ENV_OPTIONS_WITH_VARIABLE_NAME:
            variable_index = index + 1
            if variable_index >= len(argv) or not _is_environment_variable_name(
                argv[variable_index]
            ):
                return None
            index += 2
            continue
        if not has_ended_options and value.startswith("--unset="):
            if not _is_environment_variable_name(value.removeprefix("--unset=")):
                return None
            index += 1
            continue
        if not has_ended_options and value.startswith("-"):
            return None
        if _is_environment_assignment(value):
            sensitive_wrapper_indices.add(index)
            index += 1
            continue
        break

    if index >= len(argv):
        return None
    if _command_name_at(argv, index) == "env":
        return None
    return _CommandInvocation(
        name=_command_name_at(argv, index),
        option_start_index=index + 1,
        sensitive_wrapper_indices=frozenset(sensitive_wrapper_indices),
    )


def _is_environment_assignment(value: str) -> bool:
    return "=" in value


def _is_environment_variable_name(value: str) -> bool:
    return bool(value) and (value[0].isalpha() or value[0] == "_") and all(
        character.isalnum() or character == "_" for character in value[1:]
    )


def _command_name_at(argv: list[str], index: int) -> str:
    return argv[index].rsplit("/", maxsplit=1)[-1]


def _container_login_subcommand_index(
    argv: list[str],
    *,
    start_index: int,
) -> int | None:
    """Return Docker/Podman ``login`` subcommand index before ``--`` if present."""

    index = start_index
    while index < len(argv):
        value = argv[index]
        if value == "--":
            return None
        if value in _CONTAINER_CLI_GLOBAL_OPTIONS_WITH_VALUES:
            if index + 1 >= len(argv) or argv[index + 1] == "--":
                return None
            index += 2
            continue
        if value.startswith("-"):
            index += 1
            continue
        return index if value == "login" else None
    return None


def _credential_option_value_indices(
    argv: list[str],
    *,
    start_index: int,
    short_options: frozenset[str],
    long_options: frozenset[str],
) -> set[int]:
    sensitive_indices: set[int] = set()
    for index in range(start_index, len(argv)):
        value = argv[index]
        if value == "--":
            break

        if value.startswith("--"):
            option, separator, _ = value[2:].partition("=")
            if option not in long_options:
                continue
            sensitive_indices.add(index)
            if not separator and index + 1 < len(argv) and argv[index + 1] != "--":
                sensitive_indices.add(index + 1)
            continue

        if not value.startswith("-") or value == "-":
            continue
        short_option_cluster = value[1:]
        sensitive_positions = [
            option_index
            for option_index, option in enumerate(short_option_cluster)
            if option in short_options
        ]
        if not sensitive_positions:
            continue
        sensitive_indices.add(index)
        # A sensitive option at the end of a cluster may consume the next
        # argv value.  We deliberately do not model every preceding option's
        # value grammar: a matching letter inside an attached value fails
        # closed rather than risking credential persistence.
        if (
            len(short_option_cluster) - 1 in sensitive_positions
            and index + 1 < len(argv)
            and argv[index + 1] != "--"
        ):
            sensitive_indices.add(index + 1)
    return sensitive_indices


def _is_sensitive_or_redacted_text(value: str) -> bool:
    return redact_text(value).match_count > 0 or any(
        marker in value for marker in _REDACTION_MARKERS
    )


def _serialize_arguments(arguments: dict[str, object]) -> str:
    return json.dumps(
        arguments,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
