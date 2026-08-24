from pathlib import Path

import pytest
from pydantic import HttpUrl

from mini_agent.config import MiniAgentConfig, ModelConfig, RuntimeConfig, load_config
from mini_agent.storage_paths import find_config_path, resolve_data_dir


def _config(*, data_dir: Path | None = None) -> MiniAgentConfig:
    return MiniAgentConfig(
        model=ModelConfig(model="test-model", base_url=HttpUrl("https://example.test/v1")),
        runtime=RuntimeConfig(data_dir=data_dir),
    )


def test_find_config_path_uses_explicit_path_before_workspace_or_user_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace_config = workspace / ".mini-agent" / "config.toml"
    workspace_config.parent.mkdir(parents=True)
    workspace_config.write_text("workspace", encoding="utf-8")
    explicit_path = tmp_path / "explicit" / "config.toml"
    user_config_dir = tmp_path / "user-config"
    requested_app_names: list[str] = []

    def fake_user_config_path(app_name: str) -> Path:
        requested_app_names.append(app_name)
        return user_config_dir

    monkeypatch.setattr("mini_agent.storage_paths.user_config_path", fake_user_config_path)

    assert find_config_path(workspace, explicit_path) == explicit_path.resolve()
    assert requested_app_names == []


def test_find_config_path_uses_workspace_config_before_user_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace_config = workspace / ".mini-agent" / "config.toml"
    workspace_config.parent.mkdir(parents=True)
    workspace_config.write_text("workspace", encoding="utf-8")
    requested_app_names: list[str] = []

    def fake_user_config_path(app_name: str) -> Path:
        requested_app_names.append(app_name)
        return tmp_path / "user-config"

    monkeypatch.setattr("mini_agent.storage_paths.user_config_path", fake_user_config_path)

    assert find_config_path(workspace, None) == workspace_config
    assert requested_app_names == []


def test_find_config_path_falls_back_to_platform_user_config_without_reading_host_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    user_config_dir = tmp_path / "isolated-user-config"
    requested_app_names: list[str] = []

    def fake_user_config_path(app_name: str) -> Path:
        requested_app_names.append(app_name)
        return user_config_dir

    monkeypatch.setattr("mini_agent.storage_paths.user_config_path", fake_user_config_path)

    config_path = find_config_path(workspace, None)

    assert config_path == user_config_dir / "config.toml"
    assert requested_app_names == ["mini-agent"]
    assert not config_path.exists()


def test_find_config_path_returns_uncreated_explicit_path_with_stable_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    config_path = find_config_path(tmp_path / "workspace", Path("missing/config.toml"))

    assert config_path == tmp_path / "missing" / "config.toml"
    assert not config_path.exists()


def test_resolve_data_dir_resolves_explicit_absolute_and_relative_direct_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    absolute_data_dir = tmp_path / "absolute-data"
    assert resolve_data_dir(_config(data_dir=absolute_data_dir)) == absolute_data_dir
    assert resolve_data_dir(_config(data_dir=Path("relative-data"))) == tmp_path / "relative-data"


def test_resolve_data_dir_uses_platform_default_without_reading_host_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_data_dir = tmp_path / "isolated-user-data"
    requested_app_names: list[str] = []

    def fake_user_data_path(app_name: str) -> Path:
        requested_app_names.append(app_name)
        return user_data_dir

    monkeypatch.setattr("mini_agent.storage_paths.user_data_path", fake_user_data_path)

    data_dir = resolve_data_dir(_config())

    assert data_dir == user_data_dir
    assert requested_app_names == ["mini-agent"]
    assert not data_dir.exists()


def test_loaded_relative_data_dir_is_resolved_from_the_selected_config_file(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "nested" / "config.toml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
[model]
model = "test-model"
base_url = "https://example.test/v1"

[runtime]
data_dir = "runtime-data"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.runtime.data_dir == config_path.parent / "runtime-data"
    assert resolve_data_dir(config) == config_path.parent / "runtime-data"
