"""Assemble one authoritative system message for every model request."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from mini_agent.config import SystemPromptConfig
from mini_agent.goal import GoalState, render_goal_context
from mini_agent.messages import ConversationMessage, MessageRole
from mini_agent.private_key_material import (
    FileScanLimitExceededError,
    contains_private_key_pem,
    read_bounded_open_file_bytes,
)
from mini_agent.redaction import redact_text
from mini_agent.tools.base import ToolDefinition
from mini_agent.workspace import SensitiveWorkspacePathError, Workspace, WorkspacePathError

_MAX_UTF8_BYTES_PER_CHARACTER = 4


class SystemPromptError(RuntimeError):
    """Raised when configured prompt material cannot be loaded safely."""


class GitContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    is_repository: bool
    branch: str | None = Field(default=None, max_length=256)
    change_count: int = Field(default=0, ge=0)
    is_available: bool = True
    # Keep manually supplied UI/test contexts compatible with the prior model.
    # Automatic collection always sets this false and never inspects the tree.
    is_worktree_status_available: bool = True

    @property
    def state(self) -> str:
        if not self.is_available:
            return "unavailable"
        if not self.is_repository:
            return "not a repository"
        if not self.is_worktree_status_available:
            return "not collected"
        if self.change_count:
            return f"dirty ({self.change_count} changes)"
        return "clean"


class RuntimeContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1)
    workspace: Path
    platform: str = Field(min_length=1)
    git: GitContext


class SystemPromptAssembler:
    """Compose identity, runtime, tools, instructions, and goal in one place."""

    def __init__(
        self,
        *,
        config: SystemPromptConfig,
        model: str,
        workspace: Path,
        tools: tuple[ToolDefinition, ...],
        excluded_roots: tuple[Path, ...] = (),
    ) -> None:
        self._config = config
        self._model = model
        self._workspace = workspace.expanduser().resolve(strict=True)
        self._workspace_paths = Workspace(self._workspace, excluded_roots=excluded_roots)
        self._tools = tools
        self._git_executable = _trusted_git_executable(self._workspace)

    def runtime_context(self) -> RuntimeContext:
        git = (
            _collect_git_context(self._workspace, self._git_executable)
            if self._config.include_git_status
            else GitContext(is_repository=False, is_available=False)
        )
        return RuntimeContext(
            model=self._model,
            workspace=self._workspace,
            platform=f"{platform.system()} {platform.machine()}",
            git=git,
        )

    def assemble(
        self,
        goal: GoalState | None,
        *,
        runtime: RuntimeContext | None = None,
    ) -> ConversationMessage:
        """Keep reusable instructions before live state for prefix caching."""

        current_runtime = runtime or self.runtime_context()
        stable_sections = (
            _identity_section(self._config.identity),
            _operating_rules_section(),
            self._project_instructions_section(),
            _tools_section(self._tools),
            _runtime_section(current_runtime),
        )
        dynamic_sections = (
            _live_context_section(current_runtime.git),
            _goal_section(goal),
        )
        return ConversationMessage(
            role=MessageRole.SYSTEM,
            content="\n\n".join(
                section
                for section in (*stable_sections, *dynamic_sections)
                if section is not None
            ),
        )

    def _project_instructions_section(self) -> str | None:
        configured_path = self._config.instructions_file
        if configured_path is None:
            return None
        instructions_relative_path = _resolve_instructions_path(
            self._workspace_paths,
            configured_path,
        )
        instructions = _read_bounded_instructions(
            self._workspace_paths,
            instructions_relative_path,
            maximum_chars=self._config.maximum_instructions_chars,
        )
        redacted = redact_text(instructions)
        source = configured_path.as_posix()
        summary = (
            f"\n\n[Credential-shaped content was redacted: {redacted.match_count} match(es).]"
            if redacted.match_count
            else ""
        )
        return f"## Project instructions\nSource: {source}\n\n{redacted.text}{summary}"


def _identity_section(identity: str) -> str:
    return f"## Identity\n{identity}"


def _runtime_section(runtime: RuntimeContext) -> str:
    return "\n".join(
        (
            "## Runtime",
            f"- Model: {runtime.model}",
            "- Workspace: <workspace-root>",
            f"- Platform: {runtime.platform}",
        )
    )


def _live_context_section(git: GitContext) -> str:
    return "\n".join(
        (
            "## Live context",
            f"- Git branch class: {_provider_branch_class(git)}",
            f"- Git state: {git.state}",
        )
    )


def _provider_branch_class(git: GitContext) -> str:
    """Keep potentially sensitive branch names out of remote model requests."""

    if not git.is_available or not git.is_repository:
        return "none"
    if git.branch == "detached":
        return "detached"
    if git.branch in {"main", "master", "trunk"}:
        return "primary"
    if git.branch is None:
        return "unknown"
    return "other"


def _tools_section(tools: tuple[ToolDefinition, ...]) -> str:
    names = ", ".join(tool.name for tool in tools) or "none"
    return f"## Available tools\n{names}"


def _operating_rules_section() -> str:
    return "\n".join(
        (
            "## Operating rules",
            "- Treat Runtime and Goal as authoritative.",
            "- Inspect before editing; use tools for workspace facts.",
            "- Respect approvals and require evidence for side effects and completion.",
            "- Retrieve omitted exact details through history or artifacts.",
            "- Never expose or persist credentials.",
        )
    )


def _goal_section(goal: GoalState | None) -> str:
    content = render_goal_context(goal) if goal is not None else "No active goal yet."
    return f"## Goal\n{content}"


def _resolve_instructions_path(workspace: Workspace, relative_path: Path) -> str:
    try:
        workspace.resolve_existing(relative_path.as_posix())
        return relative_path.as_posix()
    except (SensitiveWorkspacePathError, WorkspacePathError, FileNotFoundError) as error:
        raise SystemPromptError("Configured system instructions file is unavailable") from error


def _read_bounded_instructions(
    workspace: Workspace,
    relative_path: str,
    *,
    maximum_chars: int,
) -> str:
    try:
        with workspace.directory_anchor.open_existing_regular_file(
            relative_path
        ) as file_descriptor:
            file_bytes = read_bounded_open_file_bytes(
                file_descriptor,
                max_bytes=maximum_chars * _MAX_UTF8_BYTES_PER_CHARACTER,
            )
    except (FileScanLimitExceededError, OSError) as error:
        raise SystemPromptError("Configured system instructions file cannot be read") from error
    if contains_private_key_pem(file_bytes):
        raise SystemPromptError("Configured system instructions file is unavailable")
    try:
        content = file_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise SystemPromptError("Configured system instructions file cannot be read") from error
    if len(content) > maximum_chars:
        raise SystemPromptError("Configured system instructions exceed the character limit")
    if not content.strip():
        raise SystemPromptError("Configured system instructions file is empty")
    return content.strip()


def _trusted_git_executable(workspace: Path) -> Path | None:
    candidate = shutil.which("git")
    if candidate is None:
        return None
    try:
        resolved = Path(candidate).resolve(strict=True)
    except (OSError, ValueError):
        return None
    is_untrusted = (
        resolved.is_relative_to(workspace)
        or not resolved.is_file()
        or not os.access(resolved, os.X_OK)
    )
    if is_untrusted:
        return None
    return resolved


def _collect_git_context(workspace: Path, git_executable: Path | None) -> GitContext:
    if git_executable is None:
        return GitContext(is_repository=False, is_available=False)
    repository_probe = _run_git_ref_probe(
        workspace,
        git_executable,
        ("rev-parse", "--is-inside-work-tree"),
    )
    if repository_probe is None:
        return GitContext(is_repository=False, is_available=False)
    if repository_probe.returncode != 0 or repository_probe.stdout.strip() != "true":
        return GitContext(is_repository=False)

    branch_probe = _run_git_ref_probe(
        workspace,
        git_executable,
        ("symbolic-ref", "--quiet", "--short", "HEAD"),
    )
    if branch_probe is None:
        return GitContext(is_repository=True, is_available=False)
    branch = (
        "detached"
        if branch_probe.returncode != 0
        else _parse_branch(branch_probe.stdout)
    )
    return GitContext(
        is_repository=True,
        branch=branch,
        is_worktree_status_available=False,
    )


def _run_git_ref_probe(
    workspace: Path,
    git_executable: Path,
    arguments: tuple[str, ...],
) -> subprocess.CompletedProcess[str] | None:
    """Read Git repository metadata without traversing worktree content.

    The only callers use built-in ref commands. They never invoke a Git alias,
    hook, clean/smudge filter, fsmonitor, or worktree-status operation.
    """

    try:
        return subprocess.run(
            (str(git_executable), "--no-optional-locks", *arguments),
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1.0,
            check=False,
            env={
                "PATH": os.defpath,
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_OPTIONAL_LOCKS": "0",
            },
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _parse_branch(output: str) -> str:
    value = output.strip()
    redacted = redact_text(value)
    return redacted.text[:256] or "unknown"
