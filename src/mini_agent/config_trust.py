"""Trust gating for workspace-sourced configuration.

A workspace's ``.mini-agent/config.toml`` is committed alongside project files and
is therefore untrusted: it can redirect the agent's credential (``api_key_env``)
and provider endpoint (``base_url``) to an attacker-controlled host, exfiltrating
an existing host credential on the first provider call. Such a config is loaded
only after an explicit, recorded user decision -- never silently.

Explicit ``--config`` paths and the user-level config are trusted sources and are
not gated. Trust decisions are persisted under the user config directory so each
workspace is confirmed at most once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from platformdirs import user_config_path

_TRUST_RECORD_FILENAME = "trusted-workspaces.json"


class WorkspaceConfigUntrustedError(Exception):
    """Raised when an untrusted workspace config must not be loaded."""


def workspace_config_path(workspace: Path) -> Path:
    """Return the conventional workspace config path."""
    return workspace / ".mini-agent" / "config.toml"


def is_untrusted_workspace_config(
    resolved: Path,
    workspace: Path,
) -> bool:
    """Return whether the resolved config is the workspace's own config.

    The workspace's ``.mini-agent/config.toml`` is gated whether it is auto-loaded
    or reached via an explicit ``--config`` that resolves to it; user-chosen
    files elsewhere (and the user-level config) are trusted sources.
    """
    return _resolve(resolved) == _resolve(workspace_config_path(workspace))


def is_workspace_trusted(workspace: Path) -> bool:
    """Return whether the workspace was previously trusted by the user."""
    record = _trust_record_path()
    if not record.is_file():
        return False
    try:
        parsed: object = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(parsed, dict):
        return False
    data = cast(dict[str, object], parsed)
    workspaces_obj = data.get("workspaces")
    if not isinstance(workspaces_obj, list):
        return False
    workspaces = cast(list[object], workspaces_obj)
    canonical = _canonical(workspace)
    return any(isinstance(entry, str) and entry == canonical for entry in workspaces)


def trust_workspace(workspace: Path) -> None:
    """Record that the user trusts this workspace's config.

    Persistence is best-effort: a write failure (e.g. a read-only user config
    directory) must not block a load the user already authorized, so the decision
    then applies only to this run and re-prompts on the next.
    """
    canonical = _canonical(workspace)
    record = _trust_record_path()
    workspaces: list[str] = []
    if record.is_file():
        try:
            parsed: object = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            existing = cast(dict[str, object], parsed).get("workspaces")
            if isinstance(existing, list):
                workspaces = [
                    entry
                    for entry in cast(list[object], existing)
                    if isinstance(entry, str)
                ]
    if canonical not in workspaces:
        workspaces.append(canonical)
    try:
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(
            json.dumps({"workspaces": workspaces}, indent=2),
            encoding="utf-8",
        )
    except OSError:
        # Best-effort: a read-only user config dir should not void an explicit
        # opt-in. This run proceeds trusted; the next run re-prompts.
        pass


def _trust_record_path() -> Path:
    return user_config_path("mini-agent") / _TRUST_RECORD_FILENAME


def _canonical(workspace: Path) -> str:
    return str(_resolve(workspace))


def _resolve(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except OSError:
        return path.absolute()