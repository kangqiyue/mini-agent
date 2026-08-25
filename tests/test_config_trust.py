from pathlib import Path
from unittest.mock import patch

import pytest
import typer

from mini_agent import cli as cli_module
from mini_agent.config import MiniAgentConfig
from mini_agent.config_trust import (
    is_untrusted_workspace_config,
    is_workspace_trusted,
    trust_workspace,
    workspace_config_path,
)


def _write_workspace_config(workspace: Path, *, data_dir: Path) -> None:
    (workspace / ".mini-agent").mkdir(parents=True, exist_ok=True)
    workspace_config_path(workspace).write_text(
        "\n".join(
            (
                "[model]",
                'model = "test-model"',
                'base_url = "https://example.test/v1"',
                "",
                "[runtime]",
                f'data_dir = "{data_dir}"',
                "",
            )
        ),
        encoding="utf-8",
    )


@pytest.fixture
def isolated_trust_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the trust record to a temp file so tests never touch real user config."""
    record = tmp_path / "trusted-workspaces.json"
    monkeypatch.setattr("mini_agent.config_trust._trust_record_path", lambda: record)
    return record


def test_is_untrusted_workspace_config_flags_only_the_workspace_config(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # The workspace's own config is gated, whether auto-loaded or via --config.
    assert is_untrusted_workspace_config(workspace_config_path(workspace), workspace)
    # A different resolved path (user-level or an explicit file elsewhere) is trusted.
    assert not is_untrusted_workspace_config(tmp_path / "elsewhere.toml", workspace)


def test_load_runtime_config_refuses_untrusted_workspace_config_non_interactively(
    tmp_path: Path, isolated_trust_record: Path
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_workspace_config(workspace, data_dir=tmp_path / "data")
    with (
        patch("mini_agent.cli._is_interactive_terminal", return_value=False),
        pytest.raises(typer.BadParameter),
    ):
        cli_module._load_runtime_config(workspace, None)  # pyright: ignore[reportPrivateUsage]
    assert not is_workspace_trusted(workspace)


def test_load_runtime_config_trust_flag_loads_and_records_trust(
    tmp_path: Path, isolated_trust_record: Path
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_workspace_config(workspace, data_dir=tmp_path / "data")
    config = cli_module._load_runtime_config(  # pyright: ignore[reportPrivateUsage]
        workspace, None, trust_workspace_config=True
    )
    assert isinstance(config, MiniAgentConfig)
    assert is_workspace_trusted(workspace)


def test_load_runtime_config_loads_after_prior_trust(
    tmp_path: Path, isolated_trust_record: Path
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_workspace_config(workspace, data_dir=tmp_path / "data")
    trust_workspace(workspace)
    config = cli_module._load_runtime_config(workspace, None)  # pyright: ignore[reportPrivateUsage]
    assert isinstance(config, MiniAgentConfig)


def test_load_runtime_config_explicit_config_skips_gate(
    tmp_path: Path, isolated_trust_record: Path
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    explicit = tmp_path / "config.toml"
    explicit.write_text(
        "\n".join(
            (
                "[model]",
                'model = "test-model"',
                'base_url = "https://example.test/v1"',
                "",
                "[runtime]",
                f'data_dir = "{tmp_path / "data"}"',
                "",
            )
        ),
        encoding="utf-8",
    )
    config = cli_module._load_runtime_config(workspace, explicit)  # pyright: ignore[reportPrivateUsage]
    assert isinstance(config, MiniAgentConfig)
    assert not is_workspace_trusted(workspace)


def test_load_runtime_config_interactive_prompt_yes_trusts(
    tmp_path: Path, isolated_trust_record: Path
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_workspace_config(workspace, data_dir=tmp_path / "data")
    with (
        patch("mini_agent.cli._is_interactive_terminal", return_value=True),
        patch("mini_agent.cli.typer.confirm", return_value=True),
    ):
        config = cli_module._load_runtime_config(workspace, None)  # pyright: ignore[reportPrivateUsage]
    assert isinstance(config, MiniAgentConfig)
    assert is_workspace_trusted(workspace)


def test_load_runtime_config_interactive_prompt_no_refuses(
    tmp_path: Path, isolated_trust_record: Path
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_workspace_config(workspace, data_dir=tmp_path / "data")
    with (
        patch("mini_agent.cli._is_interactive_terminal", return_value=True),
        patch("mini_agent.cli.typer.confirm", return_value=False),
        pytest.raises(typer.BadParameter),
    ):
        cli_module._load_runtime_config(workspace, None)  # pyright: ignore[reportPrivateUsage]
    assert not is_workspace_trusted(workspace)


def test_load_runtime_config_skips_gate_for_read_only_commands(
    tmp_path: Path, isolated_trust_record: Path
) -> None:
    # sessions/inspect/evaluate run no provider and must load an untrusted
    # workspace config without the gate (no flag, no prompt).
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_workspace_config(workspace, data_dir=tmp_path / "data")
    config = cli_module._load_runtime_config(  # pyright: ignore[reportPrivateUsage]
        workspace, None, enforce_workspace_trust=False
    )
    assert isinstance(config, MiniAgentConfig)
    assert not is_workspace_trusted(workspace)


def test_load_runtime_config_gates_explicit_config_pointing_at_workspace_config(
    tmp_path: Path, isolated_trust_record: Path
) -> None:
    # An explicit --config that resolves to the workspace's own config is still
    # gated, closing the exfil vector of pointing --config at the workspace file.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_workspace_config(workspace, data_dir=tmp_path / "data")
    workspace_cfg = workspace_config_path(workspace)
    with (
        patch("mini_agent.cli._is_interactive_terminal", return_value=False),
        pytest.raises(typer.BadParameter),
    ):
        cli_module._load_runtime_config(workspace, workspace_cfg)  # pyright: ignore[reportPrivateUsage]
    assert not is_workspace_trusted(workspace)