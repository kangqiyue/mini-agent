from __future__ import annotations

import os
import stat
import tomllib
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from mini_agent import config_template
from mini_agent.config import MiniAgentConfig, load_config
from mini_agent.config_template import (
    ConfigAlreadyExistsError,
    ConfigInitializationError,
    initialize_workspace_config,
    read_packaged_config_template,
)


def test_packaged_template_matches_the_public_example() -> None:
    public_example = Path(__file__).parents[1] / ".mini-agent" / "config.example.toml"

    assert read_packaged_config_template() == public_example.read_text(encoding="utf-8")


def test_demo_config_defaults_match_model_defaults() -> None:
    template_config = MiniAgentConfig.model_validate(tomllib.loads(read_packaged_config_template()))
    default_config = MiniAgentConfig.model_validate(
        {
            "model": {
                "model": template_config.model.model,
                "base_url": str(template_config.model.base_url),
            }
        }
    )

    assert template_config == default_config


def test_initialize_workspace_config_creates_a_complete_private_file(tmp_path: Path) -> None:
    with patch("mini_agent.config_template.os.fsync", wraps=os.fsync) as fsync:
        config_path = initialize_workspace_config(tmp_path)

    assert config_path == tmp_path / ".mini-agent" / "config.toml"
    assert config_path.read_text(encoding="utf-8") == read_packaged_config_template()
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    config = load_config(config_path)
    assert config.model.model == "your-model-name"
    assert fsync.call_count == 3


def test_initialize_workspace_config_preserves_an_existing_file(tmp_path: Path) -> None:
    config_directory = tmp_path / ".mini-agent"
    config_directory.mkdir()
    config_path = config_directory / "config.toml"
    config_path.write_text("preserve this file\n", encoding="utf-8")

    with pytest.raises(ConfigAlreadyExistsError, match="not overwritten"):
        initialize_workspace_config(tmp_path)

    assert config_path.read_text(encoding="utf-8") == "preserve this file\n"


def test_initialize_workspace_config_rejects_a_symlinked_directory(tmp_path: Path) -> None:
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    (tmp_path / ".mini-agent").symlink_to(outside_directory, target_is_directory=True)

    with pytest.raises(ConfigInitializationError, match="symbolic link"):
        initialize_workspace_config(tmp_path)

    assert not (outside_directory / "config.toml").exists()


def test_initialize_workspace_config_removes_an_incomplete_write(tmp_path: Path) -> None:
    (tmp_path / ".mini-agent").mkdir()
    with (
        patch("mini_agent.config_template.os.fsync", side_effect=OSError("synthetic failure")),
        pytest.raises(ConfigInitializationError, match="written durably"),
    ):
        initialize_workspace_config(tmp_path)

    assert (tmp_path / ".mini-agent").is_dir()
    assert not (tmp_path / ".mini-agent" / "config.toml").exists()


def test_initialize_workspace_config_removes_file_when_directory_sync_fails(
    tmp_path: Path,
) -> None:
    (tmp_path / ".mini-agent").mkdir()
    with (
        patch(
            "mini_agent.config_template.os.fsync",
            side_effect=(None, OSError("synthetic directory failure")),
        ),
        pytest.raises(ConfigInitializationError, match="written durably"),
    ):
        initialize_workspace_config(tmp_path)

    assert not (tmp_path / ".mini-agent" / "config.toml").exists()


def test_initialize_workspace_config_uses_a_resolved_workspace(tmp_path: Path) -> None:
    workspace_link = tmp_path / "workspace-link"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace_link.symlink_to(workspace, target_is_directory=True)

    config_path = initialize_workspace_config(workspace_link)

    assert config_path == workspace / ".mini-agent" / "config.toml"
    assert os.path.samefile(config_path.parent, workspace / ".mini-agent")


def test_initialize_workspace_config_fails_closed_when_workspace_root_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    displaced_workspace = tmp_path / "displaced-workspace"
    original_verify = cast(
        Callable[
            [config_template.WorkspaceDirectoryAnchor, config_template.OpenWorkspaceParent],
            None,
        ],
        vars(config_template)["_verify_current_parent_route"],
    )
    verification_count = 0

    def replace_root_then_verify(
        anchor: config_template.WorkspaceDirectoryAnchor,
        opened_parent: config_template.OpenWorkspaceParent,
    ) -> None:
        nonlocal verification_count
        verification_count += 1
        if verification_count == 2:
            workspace.rename(displaced_workspace)
            workspace.mkdir()
            (workspace / ".mini-agent").mkdir()
        original_verify(anchor, opened_parent)

    monkeypatch.setattr(
        config_template,
        "_verify_current_parent_route",
        replace_root_then_verify,
    )

    with pytest.raises(ConfigInitializationError, match="changed"):
        initialize_workspace_config(workspace)

    assert not (workspace / ".mini-agent" / "config.toml").exists()
    assert not (displaced_workspace / ".mini-agent" / "config.toml").exists()


def test_initialize_workspace_config_rejects_existing_directory_replaced_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_directory = tmp_path / ".mini-agent"
    config_directory.mkdir()
    displaced_config_directory = tmp_path / "displaced-mini-agent"

    original_open_parent = config_template.WorkspaceDirectoryAnchor.open_existing_parent
    has_replaced_directory = False

    @contextmanager
    def replace_directory_before_opening_parent(
        anchor: config_template.WorkspaceDirectoryAnchor, relative_path: str
    ) -> Generator[config_template.OpenWorkspaceParent, None, None]:
        nonlocal has_replaced_directory
        if not has_replaced_directory:
            config_directory.rename(displaced_config_directory)
            config_directory.mkdir()
            has_replaced_directory = True
        with original_open_parent(anchor, relative_path) as opened_parent:
            yield opened_parent

    monkeypatch.setattr(
        config_template.WorkspaceDirectoryAnchor,
        "open_existing_parent",
        replace_directory_before_opening_parent,
    )

    with pytest.raises(ConfigInitializationError, match="changed"):
        initialize_workspace_config(tmp_path)

    assert not (config_directory / "config.toml").exists()
    assert not (displaced_config_directory / "config.toml").exists()


def test_initialize_workspace_config_cleans_config_when_directory_is_replaced_after_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_directory = tmp_path / ".mini-agent"
    config_directory.mkdir()
    displaced_config_directory = tmp_path / "displaced-mini-agent"
    original_write = cast(
        Callable[[int, bytes], None],
        vars(config_template)["_write_new_config"],
    )

    def replace_directory_then_write(directory_descriptor: int, template: bytes) -> None:
        config_directory.rename(displaced_config_directory)
        config_directory.mkdir()
        original_write(directory_descriptor, template)

    monkeypatch.setattr(
        config_template,
        "_write_new_config",
        replace_directory_then_write,
    )

    with pytest.raises(ConfigInitializationError, match="changed"):
        initialize_workspace_config(tmp_path)

    assert not (config_directory / "config.toml").exists()
    assert not (displaced_config_directory / "config.toml").exists()


def test_initialize_workspace_config_reports_failed_route_change_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_directory = tmp_path / ".mini-agent"
    config_directory.mkdir()
    displaced_config_directory = tmp_path / "displaced-mini-agent"
    original_write = cast(
        Callable[[int, bytes], None],
        vars(config_template)["_write_new_config"],
    )

    def replace_directory_then_write(directory_descriptor: int, template: bytes) -> None:
        config_directory.rename(displaced_config_directory)
        config_directory.mkdir()
        original_write(directory_descriptor, template)

    def fail_cleanup_unlink(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic cleanup failure")

    monkeypatch.setattr(
        config_template,
        "_write_new_config",
        replace_directory_then_write,
    )
    monkeypatch.setattr(config_template.os, "unlink", fail_cleanup_unlink)

    with pytest.raises(ConfigInitializationError, match="cleanup could not be completed"):
        initialize_workspace_config(tmp_path)

    assert not (config_directory / "config.toml").exists()
    assert (displaced_config_directory / "config.toml").is_file()


def test_initialize_workspace_config_reports_failed_route_change_cleanup_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_directory = tmp_path / ".mini-agent"
    config_directory.mkdir()
    displaced_config_directory = tmp_path / "displaced-mini-agent"
    original_write = cast(
        Callable[[int, bytes], None],
        vars(config_template)["_write_new_config"],
    )
    original_fsync = os.fsync
    fsync_call_count = 0

    def replace_directory_then_write(directory_descriptor: int, template: bytes) -> None:
        config_directory.rename(displaced_config_directory)
        config_directory.mkdir()
        original_write(directory_descriptor, template)

    def fail_cleanup_sync(descriptor: int) -> None:
        nonlocal fsync_call_count
        fsync_call_count += 1
        if fsync_call_count == 3:
            raise OSError("synthetic cleanup sync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(
        config_template,
        "_write_new_config",
        replace_directory_then_write,
    )
    monkeypatch.setattr(config_template.os, "fsync", fail_cleanup_sync)

    with pytest.raises(ConfigInitializationError, match="cleanup could not be completed"):
        initialize_workspace_config(tmp_path)

    assert not (config_directory / "config.toml").exists()
    assert not (displaced_config_directory / "config.toml").exists()
