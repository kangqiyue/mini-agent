import os
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import mini_agent.workspace_directory_fd as workspace_directory_fd
from mini_agent.config import SystemPromptConfig
from mini_agent.goal import GoalState
from mini_agent.system_prompt import (
    GitContext,
    SystemPromptAssembler,
    SystemPromptError,
)
from mini_agent.tools.base import ToolDefinition
from tests.support.synthetic_secrets import synthetic_stripe_access_token


def test_assembler_builds_one_structured_system_message(tmp_path: Path) -> None:
    instructions = tmp_path / "AGENTS.md"
    instructions.write_text("Prefer small, verified changes.", encoding="utf-8")
    tool = ToolDefinition(
        name="read_file",
        description="Read a file.",
        parameters={"type": "object"},
        is_read_only=True,
    )
    goal = GoalState(
        goal_id="a" * 32,
        objective="Ship the requested change",
        acceptance_criteria=(),
    )
    assembler = SystemPromptAssembler(
        config=SystemPromptConfig(
            identity="You are the test coding agent.",
            instructions_file=Path("AGENTS.md"),
            include_git_status=False,
        ),
        model="test-model",
        workspace=tmp_path,
        tools=(tool,),
    )

    message = assembler.assemble(goal)

    assert message.role.value == "system"
    assert message.content is not None
    assert "## Identity\nYou are the test coding agent." in message.content
    assert "## Runtime" in message.content
    assert "- Model: test-model" in message.content
    assert "- Workspace: <workspace-root>" in message.content
    assert str(tmp_path) not in message.content
    assert "## Live context" in message.content
    assert "- Git branch class: none" in message.content
    assert "- Git state: unavailable" in message.content
    assert "## Available tools\nread_file" in message.content
    assert "## Operating rules" in message.content
    assert "## Project instructions" in message.content
    assert "Prefer small, verified changes." in message.content
    assert "## Goal" in message.content
    assert "Objective: Ship the requested change" in message.content
    assert message.content.index("## Project instructions") < message.content.index(
        "## Live context"
    )
    assert message.content.index("## Available tools") < message.content.index(
        "## Live context"
    )
    assert message.content.index("## Runtime") < message.content.index("## Live context")
    assert message.content.index("## Live context") < message.content.index("## Goal")


def test_live_state_changes_only_after_the_stable_prompt_prefix(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Keep the stable prefix reusable.", encoding="utf-8")
    assembler = SystemPromptAssembler(
        config=SystemPromptConfig(
            instructions_file=Path("AGENTS.md"),
            include_git_status=True,
        ),
        model="test-model",
        workspace=tmp_path,
        tools=(),
    )
    first_goal = GoalState(
        goal_id="a" * 32,
        objective="First objective",
        acceptance_criteria=(),
    )
    second_goal = first_goal.model_copy(update={"objective": "Second objective"})

    with patch(
        "mini_agent.system_prompt._collect_git_context",
        side_effect=(
            GitContext(
                is_repository=True,
                branch="main",
                is_worktree_status_available=True,
            ),
            GitContext(
                is_repository=True,
                branch="main",
                change_count=1,
                is_worktree_status_available=True,
            ),
        ),
    ):
        first = assembler.assemble(first_goal).content or ""
        second = assembler.assemble(second_goal).content or ""

    first_prefix, first_live = first.split("\n\n## Live context", maxsplit=1)
    second_prefix, second_live = second.split("\n\n## Live context", maxsplit=1)
    assert first_prefix == second_prefix
    assert "Keep the stable prefix reusable." in first_prefix
    assert "Git state" not in first_prefix
    assert first_live != second_live


def test_assembler_keeps_full_branch_and_workspace_out_of_provider_prompt(
    tmp_path: Path,
) -> None:
    private_branch = "feature/confidential-customer-ticket"
    assembler = SystemPromptAssembler(
        config=SystemPromptConfig(include_git_status=True),
        model="test-model",
        workspace=tmp_path,
        tools=(),
    )

    with patch(
        "mini_agent.system_prompt._collect_git_context",
        return_value=GitContext(is_repository=True, branch=private_branch),
    ):
        runtime = assembler.runtime_context()
        content = assembler.assemble(None, runtime=runtime).content or ""

    assert runtime.workspace == tmp_path.resolve()
    assert runtime.git.branch == private_branch
    assert str(tmp_path) not in content
    assert private_branch not in content
    assert "- Git branch class: other" in content


def test_runtime_context_does_not_run_repository_clean_filters(tmp_path: Path) -> None:
    _initialize_repository_with_clean_filter(tmp_path)
    marker = tmp_path / "filter-ran"
    assembler = SystemPromptAssembler(
        config=SystemPromptConfig(include_git_status=True),
        model="test-model",
        workspace=tmp_path,
        tools=(),
    )

    runtime = assembler.runtime_context()
    content = assembler.assemble(None, runtime=runtime).content or ""

    assert runtime.git.is_repository is True
    assert runtime.git.is_worktree_status_available is False
    assert runtime.git.state == "not collected"
    assert marker.exists() is False
    assert "- Git state: not collected" in content
    assert "test-branch" not in content
    assert "- Git branch class: other" in content


def test_git_status_fixture_runs_the_configured_clean_filter(tmp_path: Path) -> None:
    _initialize_repository_with_clean_filter(tmp_path)

    _run_git(tmp_path, "status", "--porcelain=v1")

    assert (tmp_path / "filter-ran").exists()


def test_assembler_redacts_credentials_from_project_instructions(tmp_path: Path) -> None:
    secret = synthetic_stripe_access_token("SYSTEMPROMPT")
    (tmp_path / "AGENTS.md").write_text(f"token={secret}", encoding="utf-8")
    assembler = SystemPromptAssembler(
        config=SystemPromptConfig(
            instructions_file=Path("AGENTS.md"),
            include_git_status=False,
        ),
        model="test-model",
        workspace=tmp_path,
        tools=(),
    )

    content = assembler.assemble(None).content or ""

    assert secret not in content
    assert "[REDACTED]" in content


@pytest.mark.parametrize(
    "relative_path",
    (
        ".netrc",
        ".mini-agent/config.toml",
        ".git/config",
    ),
)
def test_assembler_rejects_sensitive_instruction_paths_before_reading(
    tmp_path: Path,
    relative_path: str,
) -> None:
    sensitive_content = "must-not-enter-provider-message"
    instructions_path = tmp_path / relative_path
    instructions_path.parent.mkdir(parents=True, exist_ok=True)
    instructions_path.write_text(sensitive_content, encoding="utf-8")
    assembler = SystemPromptAssembler(
        config=SystemPromptConfig(
            instructions_file=Path(relative_path),
            include_git_status=False,
        ),
        model="test-model",
        workspace=tmp_path,
        tools=(),
    )

    with pytest.raises(SystemPromptError) as error_info:
        assembler.assemble(None)

    assert str(error_info.value) == "Configured system instructions file is unavailable"
    assert relative_path not in str(error_info.value)
    assert sensitive_content not in str(error_info.value)


def test_assembler_rejects_configured_runtime_store_before_reading(tmp_path: Path) -> None:
    runtime_store = tmp_path / "runtime-store"
    runtime_store.mkdir()
    sensitive_content = "runtime-data-must-not-enter-provider-message"
    (runtime_store / "instructions.md").write_text(sensitive_content, encoding="utf-8")
    assembler = SystemPromptAssembler(
        config=SystemPromptConfig(
            instructions_file=Path("runtime-store/instructions.md"),
            include_git_status=False,
        ),
        model="test-model",
        workspace=tmp_path,
        tools=(),
        excluded_roots=(runtime_store,),
    )

    with pytest.raises(SystemPromptError) as error_info:
        assembler.assemble(None)

    assert str(error_info.value) == "Configured system instructions file is unavailable"
    assert sensitive_content not in str(error_info.value)


def test_assembler_rejects_private_key_in_complete_instruction_file(tmp_path: Path) -> None:
    private_material = "private-material-must-not-enter-provider-message"
    (tmp_path / "instructions.pem").write_text(
        _private_key_pem(private_material),
        encoding="utf-8",
    )
    assembler = SystemPromptAssembler(
        config=SystemPromptConfig(
            instructions_file=Path("instructions.pem"),
            include_git_status=False,
        ),
        model="test-model",
        workspace=tmp_path,
        tools=(),
    )

    with pytest.raises(SystemPromptError) as error_info:
        assembler.assemble(None)

    assert str(error_info.value) == "Configured system instructions file is unavailable"
    assert private_material not in str(error_info.value)
    assert "instructions.pem" not in str(error_info.value)


def test_assembler_allows_safe_instructions_and_public_certificate(tmp_path: Path) -> None:
    (tmp_path / "public.pem").write_text(
        "-----BEGIN CERTIFICATE-----\ncertificate-body\n-----END CERTIFICATE-----\n",
        encoding="utf-8",
    )
    assembler = SystemPromptAssembler(
        config=SystemPromptConfig(
            instructions_file=Path("public.pem"),
            include_git_status=False,
        ),
        model="test-model",
        workspace=tmp_path,
        tools=(),
    )

    content = assembler.assemble(None).content or ""

    assert "certificate-body" in content


def test_assembler_keeps_held_instruction_parent_after_intermediate_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    (nested_directory / "instructions.md").write_text(
        "Use the held instructions.", encoding="utf-8"
    )
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    secret = "outside-instructions-must-not-enter-prompt"
    (outside_directory / "instructions.md").write_text(secret, encoding="utf-8")
    moved_directory = tmp_path / "nested-moved"
    original_open = os.open
    has_swapped = False

    def swap_parent_before_leaf_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        *args: int,
        **kwargs: int,
    ) -> int:
        nonlocal has_swapped
        if not has_swapped and path == "instructions.md" and "dir_fd" in kwargs:
            has_swapped = True
            nested_directory.rename(moved_directory)
            nested_directory.symlink_to(outside_directory, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(
        workspace_directory_fd.os, "open", swap_parent_before_leaf_open
    )
    assembler = SystemPromptAssembler(
        config=SystemPromptConfig(
            instructions_file=Path("nested/instructions.md"),
            include_git_status=False,
        ),
        model="test-model",
        workspace=tmp_path,
        tools=(),
    )

    content = assembler.assemble(None).content or ""

    assert has_swapped is True
    assert "Use the held instructions." in content
    assert secret not in content


def test_assembler_rejects_oversized_project_instructions(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("x" * 1_025, encoding="utf-8")
    assembler = SystemPromptAssembler(
        config=SystemPromptConfig(
            instructions_file=Path("AGENTS.md"),
            include_git_status=False,
            maximum_instructions_chars=1_024,
        ),
        model="test-model",
        workspace=tmp_path,
        tools=(),
    )

    with pytest.raises(SystemPromptError, match="character limit"):
        assembler.assemble(None)


def _initialize_repository_with_clean_filter(repository: Path) -> None:
    _run_git(repository, "init", "-q", "--initial-branch=test-branch")
    _run_git(repository, "config", "user.name", "Test User")
    _run_git(repository, "config", "user.email", "test@example.invalid")
    (repository / ".gitattributes").write_text(
        "tracked.txt filter=side-effect\n",
        encoding="utf-8",
    )
    (repository / "tracked.txt").write_text("original\n", encoding="utf-8")
    _run_git(repository, "add", ".gitattributes", "tracked.txt")
    _run_git(repository, "commit", "-qm", "Baseline")

    marker = repository / "filter-ran"
    filter_script = repository / "record_filter.py"
    filter_script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        f"Path({str(marker)!r}).touch()\n"
        "sys.stdout.buffer.write(sys.stdin.buffer.read())\n",
        encoding="utf-8",
    )
    clean_command = f"{shlex.quote(sys.executable)} {shlex.quote(str(filter_script))}"
    _run_git(repository, "config", "filter.side-effect.clean", clean_command)
    (repository / "tracked.txt").write_text("modified\n", encoding="utf-8")


def _private_key_pem(material: str) -> str:
    return "\n".join(
        (
            "-----BEGIN PRIVATE KEY-----",
            material,
            "-----END PRIVATE KEY-----",
            "",
        )
    )


def _run_git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
