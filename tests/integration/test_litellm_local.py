"""Optional smoke coverage for a local LiteLLM-compatible deployment."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Never
from uuid import uuid4

import pytest
from pydantic import HttpUrl
from typer.testing import CliRunner

from mini_agent.cli import app
from mini_agent.config import ContextConfig, MiniAgentConfig, ModelConfig, RuntimeConfig
from mini_agent.messages import MessageRole
from mini_agent.session import AgentSession, list_sessions

_RUN_ENVIRONMENT = "MINI_AGENT_RUN_LOCAL_INTEGRATION"
_BASE_URL_ENVIRONMENT = "MINI_AGENT_LITELLM_BASE_URL"
_MODEL_ENVIRONMENT = "MINI_AGENT_LITELLM_MODEL"
_API_KEY_ENVIRONMENT = "MINI_AGENT_LITELLM_API_KEY"
_API_KEY_NAME_ENVIRONMENT = "MINI_AGENT_LITELLM_API_KEY_ENV"
_DEFAULT_BASE_URL = "http://127.0.0.1:4000/v1"
_ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")

_runner = CliRunner()


def _require_explicit_local_integration() -> None:
    """Keep real provider calls out of the normal test suite."""

    if os.environ.get(_RUN_ENVIRONMENT) != "1":
        pytest.skip("Set MINI_AGENT_RUN_LOCAL_INTEGRATION=1 to run local provider coverage.")


@pytest.mark.local_integration
def test_litellm_local_cli_round_trip_preserves_memory_after_resume(tmp_path: Path) -> None:
    """Exercise one CLI turn and one resumed turn without persisting a credential."""

    _require_explicit_local_integration()
    model = _local_model()
    api_key_environment = _api_key_environment_name()
    api_key = os.environ.get(api_key_environment)
    if api_key is None or len(api_key) < 12:
        _fail("The explicitly enabled local integration has no configured API credential.")

    config_path = tmp_path / "local-integration.toml"
    data_dir = tmp_path / "agent-data"
    config = _local_config(
        data_dir=data_dir,
        api_key_environment=api_key_environment,
        model=model,
    )
    _write_config(config_path, config=config)

    marker = f"local-integration-{uuid4().hex}"
    first_request = (
        f"Reply with this exact opaque marker and no other text: {marker}. "
        "Do not call any tools."
    )
    initial_result = _runner.invoke(
        app,
        ["chat", "--workspace", str(tmp_path), "--config", str(config_path)],
        input=f"{first_request}\n/exit\n",
    )
    if initial_result.exit_code != 0:
        _fail("The initial local CLI turn did not complete successfully.")

    session_id = _only_session_id(data_dir)
    _require_marker_in_last_assistant_message(
        data_dir=data_dir,
        session_id=session_id,
        marker=marker,
    )

    resumed_result = _runner.invoke(
        app,
        ["resume", session_id, "--workspace", str(tmp_path), "--config", str(config_path)],
        input=(
            "Reply with the opaque marker from the prior request exactly and no other text.\n"
            "/exit\n"
        ),
    )
    if resumed_result.exit_code != 0:
        _fail("The resumed local CLI turn did not complete successfully.")

    _require_marker_in_last_assistant_message(
        data_dir=data_dir,
        session_id=session_id,
        marker=marker,
    )
    _require_credential_not_persisted(
        paths=(config_path, data_dir),
        credential=api_key,
    )


def test_marker_missing_failure_rendering_hides_local_path_and_credential(tmp_path: Path) -> None:
    """Prove the marker helper hides its child test internals when it fails."""

    credential = f"credential-{uuid4().hex}"
    verification_file = tmp_path / "test_trace_suppression.py"
    source_path = Path(__file__).resolve()
    verification_file.write_text(
        "\n".join(
            (
                "import importlib.util",
                "from pathlib import Path",
                "from mini_agent.messages import FinishReason",
                "from mini_agent.session import AgentSession",
                f"source_path = Path({str(source_path)!r})",
                "specification = importlib.util.spec_from_file_location(",
                "    \"local_integration\", source_path",
                ")",
                "if specification is None or specification.loader is None:",
                "    raise RuntimeError(\"source module could not load\")",
                "module = importlib.util.module_from_spec(specification)",
                "specification.loader.exec_module(module)",
                "def test_marker_missing() -> None:",
                f"    credential = {credential!r}",
                "    workspace = Path.cwd()",
                "    session = AgentSession.create(",
                "        data_dir=workspace / \"data\", workspace=workspace, model=\"m\"",
                "    )",
                "    turn_id = session.new_turn_id()",
                "    session.append_user_message(\"request\", turn_id=turn_id)",
                "    session.append_model_request_started(",
                "        turn_id=turn_id, message_count=1, attempt=1",
                "    )",
                "    session.append_assistant_message(",
                "        \"response\", finish_reason=FinishReason.STOP, turn_id=turn_id",
                "    )",
                "    session.stop()",
                "    session.close()",
                "    module._require_marker_in_last_assistant_message(",
                "        data_dir=workspace / \"data\",",
                "        session_id=session.metadata.session_id,",
                "        marker=\"missing-marker\",",
                "    )",
                "",
            )
        ),
        encoding="utf-8",
    )
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short", verification_file.name],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _fail("The local failure-rendering verification timed out.")

    rendered_output = result.stdout + result.stderr
    if result.returncode == 0:
        _fail("The local failure-rendering verification did not trigger the fixed failure.")
    if "The model response did not contain the requested opaque marker." not in rendered_output:
        _fail("The local failure-rendering verification did not preserve the fixed message.")
    if str(tmp_path) in rendered_output or credential in rendered_output:
        _fail("The local failure-rendering verification exposed sensitive test details.")


def _api_key_environment_name() -> str:
    configured_name = os.environ.get(_API_KEY_NAME_ENVIRONMENT, _API_KEY_ENVIRONMENT)
    if _ENVIRONMENT_NAME.fullmatch(configured_name) is None:
        _fail("The local integration API credential environment name is invalid.")
    return configured_name


@pytest.mark.parametrize(
    ("configured_model", "expected_model"),
    (
        (None, None),
        (" \t", None),
        ("local-test-model", "local-test-model"),
    ),
)
def test_local_integration_model_environment_is_explicit_and_non_blank(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    configured_model: str | None,
    expected_model: str | None,
) -> None:
    """Validate the local model boundary without contacting a provider."""

    if configured_model is None:
        monkeypatch.delenv(_MODEL_ENVIRONMENT, raising=False)
    else:
        monkeypatch.setenv(_MODEL_ENVIRONMENT, configured_model)

    if expected_model is None:
        with pytest.raises(pytest.fail.Exception, match="requires MINI_AGENT_LITELLM_MODEL"):
            _local_model()
        return

    model = _local_model()
    assert model == expected_model
    config = _local_config(
        data_dir=tmp_path / "data",
        api_key_environment="TEST_LOCAL_API_KEY",
        model=model,
    )
    assert config.model.model == expected_model


def _local_model() -> str:
    """Require an explicit, non-secret model identifier for a real local run."""

    model = os.environ.get(_MODEL_ENVIRONMENT)
    if model is None or not model.strip():
        _fail("The explicitly enabled local integration requires MINI_AGENT_LITELLM_MODEL.")
    return model.strip()


def _local_config(
    *, data_dir: Path, api_key_environment: str, model: str
) -> MiniAgentConfig:
    """Validate non-secret provider settings before writing the temporary config."""

    base_url = os.environ.get(_BASE_URL_ENVIRONMENT, _DEFAULT_BASE_URL)
    try:
        return MiniAgentConfig(
            model=ModelConfig(
                model=model,
                base_url=HttpUrl(base_url),
                api_key_env=api_key_environment,
                context_window=128_000,
                max_output_tokens=256,
                timeout_seconds=60,
            ),
            context=ContextConfig(reserve_tokens=2_048, minimum_input_tokens=1_024),
            runtime=RuntimeConfig(
                data_dir=data_dir,
                provider_retry_count=0,
                max_model_calls_per_turn=3,
            ),
        )
    except ValueError:
        _fail("The local integration endpoint or model configuration is invalid.")


def _write_config(path: Path, *, config: MiniAgentConfig) -> None:
    """Persist only the credential variable name, never its value."""

    data_dir = config.runtime.data_dir
    if data_dir is None:
        raise AssertionError("Local integration config requires an explicit data directory")
    path.write_text(
        "\n".join(
            (
                "[model]",
                f"model = {json.dumps(config.model.model)}",
                f"base_url = {json.dumps(str(config.model.base_url))}",
                f"api_key_env = {json.dumps(config.model.api_key_env)}",
                f"context_window = {config.model.context_window}",
                f"max_output_tokens = {config.model.max_output_tokens}",
                f"timeout_seconds = {config.model.timeout_seconds}",
                "",
                "[context]",
                f"reserve_tokens = {config.context.reserve_tokens}",
                f"minimum_input_tokens = {config.context.minimum_input_tokens}",
                "",
                "[runtime]",
                f"data_dir = {json.dumps(str(data_dir))}",
                f"provider_retry_count = {config.runtime.provider_retry_count}",
                f"max_model_calls_per_turn = {config.runtime.max_model_calls_per_turn}",
                "",
            )
        ),
        encoding="utf-8",
    )


def _only_session_id(data_dir: Path) -> str:
    sessions = list_sessions(data_dir)
    if len(sessions) != 1:
        _fail("The local CLI turn did not create exactly one session.")
    return sessions[0].session_id


def _require_marker_in_last_assistant_message(
    *, data_dir: Path, session_id: str, marker: str
) -> None:
    session = AgentSession.load(data_dir=data_dir, session_id=session_id)
    try:
        assistant_messages = tuple(
            message.content or ""
            for message in session.conversation_messages()
            if message.role is MessageRole.ASSISTANT
        )
    finally:
        session.close()
    if not assistant_messages or marker not in assistant_messages[-1]:
        _fail("The model response did not contain the requested opaque marker.")


def _require_credential_not_persisted(*, paths: tuple[Path, ...], credential: str) -> None:
    """Inspect only regular temporary config/session files without revealing the key."""

    credential_bytes = credential.encode("utf-8")
    for path in paths:
        for regular_file in _regular_files_beneath(path):
            try:
                descriptor = os.open(regular_file, os.O_RDONLY | os.O_NOFOLLOW)
                with os.fdopen(descriptor, "rb") as persisted_file:
                    if credential_bytes in persisted_file.read():
                        _fail("The local integration credential was persisted unexpectedly.")
            except OSError:
                _fail("The local integration persistence check could not run safely.")


def _regular_files_beneath(path: Path) -> Iterator[Path]:
    """Yield only regular files and never follow symlinks below test-owned paths."""

    try:
        path_mode = path.stat(follow_symlinks=False).st_mode
    except OSError:
        _fail("The local integration persistence path could not be inspected safely.")
    if stat.S_ISREG(path_mode):
        yield path
        return
    if not stat.S_ISDIR(path_mode):
        _fail("The local integration persistence path is not a regular directory.")

    for current_root, directory_names, file_names in os.walk(path, followlinks=False):
        current_path = Path(current_root)
        directory_names[:] = [
            name
            for name in directory_names
            if _is_regular_directory(current_path / name)
        ]
        for file_name in file_names:
            candidate = current_path / file_name
            if _is_regular_file(candidate):
                yield candidate


def _is_regular_directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.stat(follow_symlinks=False).st_mode)
    except OSError:
        _fail("The local integration persistence path could not be inspected safely.")


def _is_regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.stat(follow_symlinks=False).st_mode)
    except OSError:
        _fail("The local integration persistence path could not be inspected safely.")


def _fail(message: str) -> Never:
    """Fail with an intentional public message and no traceback or local values."""

    pytest.fail(message, pytrace=False)
