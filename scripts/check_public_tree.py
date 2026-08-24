"""Reject common private files and local identity leaks before publication."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import unicodedata
from collections.abc import Iterable
from pathlib import PurePosixPath

_DENYLIST_ENVIRONMENT_VARIABLE = "MINI_AGENT_PUBLICATION_DENYLIST"
_MAXIMUM_FILE_BYTES = 10 * 1024 * 1024
_SENSITIVE_FILENAMES = frozenset(
    {
        ".env",
        ".envrc",
        ".git-credentials",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".vault-token",
        "agents.md",
        "auth.json",
        "client_secrets.json",
        "config.toml",
        "credentials.json",
        "handoff.md",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "service-account.json",
        "service_account.json",
    }
)
_SENSITIVE_SUFFIXES = (".key", ".p12", ".pem", ".pfx", ".tfstate", ".tfvars")
_PRIVATE_PATH_PATTERNS = (
    re.compile(r"/(?:Users|home)/[^/\s]+/"),
    re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\", re.IGNORECASE),
    re.compile(r"/(?:private/)?var/folders/[^/\s]+/[^/\s]+/"),
)


def _normalized(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _load_denylist(*, is_required: bool) -> tuple[str, ...]:
    raw_value = os.environ.get(_DENYLIST_ENVIRONMENT_VARIABLE, "")
    if not raw_value:
        if is_required:
            raise ValueError(f"{_DENYLIST_ENVIRONMENT_VARIABLE} is required")
        return ()

    terms = tuple(_normalized(term.strip()) for term in raw_value.splitlines() if term.strip())
    if not terms or any(len(term) < 4 for term in terms):
        raise ValueError("publication denylist contains an invalid term")
    return terms


def _git_output(*arguments: str) -> bytes:
    process = subprocess.run(
        ["git", *arguments],
        check=False,
        capture_output=True,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.decode("utf-8", errors="replace").strip())
    return process.stdout


def _nul_delimited_paths(output: bytes) -> tuple[str, ...]:
    if not output:
        return ()
    if not output.endswith(b"\0"):
        raise ValueError("Git returned an invalid path list")
    return tuple(os.fsdecode(path) for path in output[:-1].split(b"\0"))


def _staged_paths() -> tuple[str, ...]:
    return _nul_delimited_paths(
        _git_output("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    )


def _tracked_paths() -> tuple[str, ...]:
    return _nul_delimited_paths(_git_output("ls-files", "-z"))


def _index_content(path: str) -> bytes:
    return _git_output("show", f":{path}")


def _path_problem(path: str, denylist: tuple[str, ...]) -> str | None:
    normalized_path = path.replace("\\", "/")
    pure_path = PurePosixPath(normalized_path)
    lowered_name = pure_path.name.casefold()
    lowered_path = normalized_path.casefold()

    if pure_path.is_absolute() or ".." in pure_path.parts:
        return "unsafe repository path"
    if lowered_name in _SENSITIVE_FILENAMES:
        if lowered_path == ".mini-agent/config.example.toml":
            return None
        return "private or credential-related filename"
    if lowered_name.endswith(_SENSITIVE_SUFFIXES):
        return "credential-related filename suffix"
    if lowered_path.startswith(".mini-agent/data/"):
        return "runtime data must not be published"
    if any(term in _normalized(normalized_path) for term in denylist):
        return "matches the private publication denylist"
    return None


def _content_problem(content: bytes, denylist: tuple[str, ...]) -> str | None:
    if len(content) > _MAXIMUM_FILE_BYTES:
        return "file is too large for the privacy check"
    text = content.decode("utf-8", errors="replace")
    if any(pattern.search(text) for pattern in _PRIVATE_PATH_PATTERNS):
        return "contains a machine-specific absolute path"
    normalized_text = _normalized(text)
    if any(term in normalized_text for term in denylist):
        return "matches the private publication denylist"
    return None


def _scan_paths(paths: Iterable[str], denylist: tuple[str, ...]) -> list[str]:
    findings: list[str] = []
    for path in paths:
        path_problem = _path_problem(path, denylist)
        if path_problem is not None:
            findings.append(f"{path}: {path_problem}")
            continue
        content_problem = _content_problem(_index_content(path), denylist)
        if content_problem is not None:
            findings.append(f"{path}: {content_problem}")
    return findings


def _scan_commit_message(path: str, denylist: tuple[str, ...]) -> list[str]:
    with open(path, "rb") as message_file:
        problem = _content_problem(message_file.read(), denylist)
    return [f"commit message: {problem}"] if problem is not None else []


def _parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--staged", action="store_true", help="scan staged files")
    targets.add_argument("--tracked", action="store_true", help="scan all tracked files")
    targets.add_argument("--commit-message", metavar="PATH", help="scan a commit message")
    parser.add_argument(
        "--require-denylist",
        action="store_true",
        help=f"fail unless {_DENYLIST_ENVIRONMENT_VARIABLE} is set",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    options = _parse_arguments(sys.argv[1:] if arguments is None else arguments)
    try:
        denylist = _load_denylist(is_required=options.require_denylist)
        if options.commit_message is not None:
            findings = _scan_commit_message(options.commit_message, denylist)
        else:
            paths = _staged_paths() if options.staged else _tracked_paths()
            findings = _scan_paths(paths, denylist)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"privacy check failed: {error}", file=sys.stderr)
        return 2

    for finding in findings:
        print(f"privacy check: {finding}", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
