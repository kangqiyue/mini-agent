"""Resolve configuration and runtime storage paths."""

from pathlib import Path

from platformdirs import user_config_path, user_data_path

from mini_agent.config import MiniAgentConfig


def find_config_path(workspace: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()

    project_path = workspace / ".mini-agent" / "config.toml"
    if project_path.is_file():
        return project_path

    return user_config_path("mini-agent") / "config.toml"


def resolve_data_dir(config: MiniAgentConfig) -> Path:
    configured_path = config.runtime.data_dir
    if configured_path is not None:
        return configured_path.expanduser().resolve()
    return user_data_path("mini-agent")
