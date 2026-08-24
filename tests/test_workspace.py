import os
from pathlib import Path

import pytest

import mini_agent.workspace as workspace_module
from mini_agent.workspace import Workspace, WorkspacePathError
from mini_agent.workspace_directory_fd import WorkspaceDirectoryFdError


def test_resolve_existing_accepts_a_file_inside_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    expected_path = workspace_root / "notes.txt"
    expected_path.write_text("hello", encoding="utf-8")

    workspace = Workspace(workspace_root)

    assert workspace.resolve_existing("notes.txt") == expected_path


def test_workspace_maps_directory_anchor_setup_failure_to_fixed_path_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_directory_anchor(root: Path) -> object:
        del root
        raise WorkspaceDirectoryFdError("unsupported directory fd primitive")

    monkeypatch.setattr(
        workspace_module, "WorkspaceDirectoryAnchor", fail_directory_anchor
    )

    with pytest.raises(WorkspacePathError) as error_info:
        Workspace(tmp_path)

    assert str(error_info.value) == "Could not inspect workspace safely"


def test_resolve_existing_rejects_absolute_path(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspacePathError, match="relative"):
        workspace.resolve_existing(str(tmp_path / "file.txt"))


def test_resolve_existing_rejects_parent_escape(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("private", encoding="utf-8")
    workspace = Workspace(workspace_root)

    with pytest.raises(WorkspacePathError, match="escapes"):
        workspace.resolve_existing("../outside.txt")


def test_resolve_existing_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("private", encoding="utf-8")
    (workspace_root / "link.txt").symlink_to(outside_file)
    workspace = Workspace(workspace_root)

    with pytest.raises(WorkspacePathError, match="escapes"):
        workspace.resolve_existing("link.txt")


def test_resolve_existing_rejects_directory_by_default(tmp_path: Path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspacePathError, match="directory"):
        workspace.resolve_existing("directory")


def test_resolve_existing_rejects_excluded_root_and_its_contents(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".mini-agent"
    runtime_dir.mkdir()
    internal_file = runtime_dir / "events.jsonl"
    internal_file.write_text("private", encoding="utf-8")
    workspace = Workspace(tmp_path, excluded_roots=(runtime_dir,))

    with pytest.raises(WorkspacePathError, match="excluded"):
        workspace.resolve_existing(".mini-agent")
    with pytest.raises(WorkspacePathError, match="excluded"):
        workspace.resolve_existing(".mini-agent/events.jsonl")


def test_resolve_existing_rejects_nul_and_too_long_paths(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspacePathError, match="NUL"):
        workspace.resolve_existing("bad\x00path")
    with pytest.raises(WorkspacePathError, match="too long"):
        workspace.resolve_existing("a" * 5_000)


def test_resolve_existing_rejects_fifo(tmp_path: Path) -> None:
    fifo = tmp_path / "pending-input"
    os.mkfifo(fifo)
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspacePathError, match="regular file"):
        workspace.resolve_existing("pending-input")


def test_resolve_existing_allows_directories_only_when_requested(tmp_path: Path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()
    workspace = Workspace(tmp_path)

    assert workspace.resolve_existing("directory", allow_directory=True) == directory


@pytest.mark.parametrize(
    "relative_path",
    (
        ".netrc",
        "nested/_netrc",
        ".env",
        ".env.local",
        ".env.production",
        "nested/.aws/credentials",
        "nested/.aws/config",
        "nested/.config/gh/hosts.yml",
        "nested/.docker/config.json",
        "nested/.kube/config",
        "nested/.mini-agent/config.toml",
        "nested/.git/HEAD",
        "nested/.hg/requires",
        "nested/.svn/entries",
        "credentials/id_ed25519",
        "credentials/private.key",
        "infrastructure/state.tfstate",
    ),
)
def test_resolve_existing_rejects_runtime_sensitive_paths(
    tmp_path: Path, relative_path: str
) -> None:
    target = tmp_path / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("placeholder\n", encoding="utf-8")
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspacePathError, match="Sensitive workspace paths"):
        workspace.resolve_existing(relative_path)


@pytest.mark.parametrize("relative_path", (".netrc", "nested/.git/HEAD"))
def test_resolve_existing_rejects_sensitive_lexical_symlink_name_before_resolution(
    tmp_path: Path, relative_path: str
) -> None:
    safe_target = tmp_path / "notes.txt"
    safe_target.write_text("safe\n", encoding="utf-8")
    sensitive_link = tmp_path / relative_path
    sensitive_link.parent.mkdir(parents=True, exist_ok=True)
    sensitive_link.symlink_to(safe_target)
    workspace = Workspace(tmp_path)

    with pytest.raises(WorkspacePathError, match="Sensitive workspace paths"):
        workspace.resolve_existing(relative_path)


@pytest.mark.parametrize(
    "relative_path",
    (
        ".env.example",
        ".env.sample",
        ".env.template",
        "certificates/public.pem",
        "certificates/public.crt",
        "notes/config.json",
    ),
)
def test_resolve_existing_allows_runtime_safe_templates_and_public_certificates(
    tmp_path: Path, relative_path: str
) -> None:
    target = tmp_path / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("placeholder\n", encoding="utf-8")
    workspace = Workspace(tmp_path)

    assert workspace.resolve_existing(relative_path) == target
