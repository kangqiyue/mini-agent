from pathlib import Path

import pytest
from pydantic import HttpUrl, ValidationError

from mini_agent.config import (
    ConfigNotFoundError,
    ContextConfig,
    MiniAgentConfig,
    ModelConfig,
    RuntimeConfig,
    SystemPromptConfig,
    load_config,
)
from tests.support.synthetic_secrets import synthetic_stripe_access_token

_SYNTHETIC_SECRET = synthetic_stripe_access_token("CONFIG")


def test_example_config_parses_and_documents_every_supported_field() -> None:
    example_path = Path(__file__).parents[1] / ".mini-agent" / "config.example.toml"

    config = load_config(example_path)
    example_text = example_path.read_text(encoding="utf-8")

    assert config.model.model == "your-model-name"
    section_models = {
        "model": ModelConfig,
        "context": ContextConfig,
        "runtime": RuntimeConfig,
        "system_prompt": SystemPromptConfig,
    }
    assert set(MiniAgentConfig.model_fields) == set(section_models)
    for section_name, section_model in section_models.items():
        section_start = example_text.index(f"[{section_name}]")
        following_sections = [
            example_text.find(f"[{name}]", section_start + 1)
            for name in section_models
        ]
        section_ends = [position for position in following_sections if position >= 0]
        section_end = min(section_ends, default=len(example_text))
        section_text = example_text[section_start:section_end]
        for field_name in section_model.model_fields:
            section_lines = section_text.splitlines()
            field_line_index = next(
                (
                    index
                    for index, line in enumerate(section_lines)
                    if line.lstrip("# ").startswith(f"{field_name} =")
                ),
                None,
            )
            assert field_line_index is not None, (
                f"{section_name}.{field_name} is missing from config.example.toml"
            )
            nearby_comments: list[str] = []
            for line in reversed(section_lines[:field_line_index]):
                if not line.startswith("#"):
                    break
                nearby_comments.append(line.removeprefix("#").strip())
            assert any(nearby_comments), (
                f"{section_name}.{field_name} needs a nearby explanatory comment"
            )


def test_load_config_resolves_relative_data_dir(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[model]
model = "test-model"
base_url = "https://example.test/v1"

[runtime]
data_dir = "state"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.runtime.data_dir == tmp_path / "state"


def test_load_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[model]
model = "test-model"
base_url = "https://example.test/v1"
unexpected = true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="unexpected"):
        load_config(config_path)


def test_load_config_rejects_credential_shaped_model_identifier(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[model]
model = "api_key=synthetic-model-credential"
base_url = "https://example.test/v1"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="credential-shaped content"):
        load_config(config_path)


@pytest.mark.parametrize(
    "model_identifier",
    (
        "/private/models/example",
        r"C:\models\example",
        "~/models/example",
        "../models/example",
        "file:///private/models/example",
    ),
)
def test_model_config_rejects_local_path_identifiers_without_echoing_them(
    model_identifier: str,
) -> None:
    with pytest.raises(ValidationError, match="local path") as error:
        ModelConfig(
            model=model_identifier,
            base_url=HttpUrl("https://example.test/v1"),
        )

    assert model_identifier not in str(error.value)


def test_model_config_allows_provider_qualified_identifier() -> None:
    config = ModelConfig(
        model="provider/model-name",
        base_url=HttpUrl("https://example.test/v1"),
    )

    assert config.model == "provider/model-name"


@pytest.mark.parametrize(
    "base_url",
    (
        f"https://user:{_SYNTHETIC_SECRET}@example.test/v1",
        f"https://example.test/v1?api_key={_SYNTHETIC_SECRET}",
        f"https://example.test/v1?mode={_SYNTHETIC_SECRET}",
    ),
)
def test_model_config_rejects_credential_bearing_base_url_without_echoing_value(
    base_url: str,
) -> None:
    with pytest.raises(ValidationError) as error:
        ModelConfig(model="test-model", base_url=HttpUrl(base_url))

    assert _SYNTHETIC_SECRET not in str(error.value)


@pytest.mark.parametrize(
    "base_url",
    (
        f"https://example.test/v1/{_SYNTHETIC_SECRET}",
        f"https://example.test/v1#access_token={_SYNTHETIC_SECRET}",
        f"https://{_SYNTHETIC_SECRET}.example.test/v1",
    ),
)
def test_model_config_rejects_credential_shaped_content_anywhere_in_base_url(
    base_url: str,
) -> None:
    with pytest.raises(ValidationError) as error:
        ModelConfig(model="test-model", base_url=HttpUrl(base_url))

    assert _SYNTHETIC_SECRET not in str(error.value)


def test_model_config_allows_ordinary_query_parameters() -> None:
    config = ModelConfig(
        model="test-model",
        base_url=HttpUrl("https://example.test/v1?token_count=12&request_mode=fast"),
    )

    assert str(config.base_url).startswith("https://example.test/v1")


def test_model_config_rejects_fragment() -> None:
    with pytest.raises(ValidationError, match="fragment"):
        ModelConfig(
            model="test-model",
            base_url=HttpUrl("https://example.test/v1#ordinary-section"),
        )


@pytest.mark.parametrize(
    "base_url",
    (
        "http://127.0.0.1:4000/v1",
        "http://localhost:4000/v1",
        "http://[::1]:4000/v1",
    ),
)
def test_model_config_allows_plain_http_only_for_loopback(base_url: str) -> None:
    config = ModelConfig(model="test-model", base_url=HttpUrl(base_url))

    assert config.base_url.scheme == "http"


def test_model_config_rejects_plain_http_for_remote_provider() -> None:
    with pytest.raises(ValidationError, match="HTTPS"):
        ModelConfig(
            model="test-model",
            base_url=HttpUrl("http://provider.example/v1"),
        )


def test_model_config_rejects_credential_shaped_api_key_env_without_echoing_value() -> None:
    with pytest.raises(ValidationError) as error:
        ModelConfig(
            model="test-model",
            base_url=HttpUrl("https://example.test/v1"),
            api_key_env=_SYNTHETIC_SECRET,
        )

    assert _SYNTHETIC_SECRET not in str(error.value)


def test_runtime_config_rejects_credential_shaped_data_dir_without_echoing_value() -> None:
    with pytest.raises(ValidationError) as error:
        RuntimeConfig(data_dir=Path(f"state/{_SYNTHETIC_SECRET}"))

    assert _SYNTHETIC_SECRET not in str(error.value)


def test_context_config_requires_ordered_milestones_below_rebuild() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        ContextConfig(checkpoint_milestones=(0.20, 0.20), rebuild_ratio=0.85)
    with pytest.raises(ValidationError, match="below rebuild"):
        ContextConfig(checkpoint_milestones=(0.20, 0.90), rebuild_ratio=0.85)


def test_model_context_override_cannot_exceed_physical_window() -> None:
    with pytest.raises(ValidationError, match="provider context window"):
        ModelConfig(
            model="m",
            base_url=HttpUrl("https://example.test/v1"),
            context_window=8_192,
            max_context=16_384,
        )


def test_context_checkpoint_model_rejects_credential_shaped_identifier() -> None:
    with pytest.raises(ValidationError) as error:
        ContextConfig(checkpoint_model=_SYNTHETIC_SECRET)

    assert _SYNTHETIC_SECRET not in str(error.value)


@pytest.mark.parametrize(
    "model_identifier",
    ("/private/models/checkpoint", r"C:\models\checkpoint", "~/models/checkpoint"),
)
def test_context_checkpoint_model_rejects_local_path_identifier(
    model_identifier: str,
) -> None:
    with pytest.raises(ValidationError, match="local path") as error:
        ContextConfig(checkpoint_model=model_identifier)

    assert model_identifier not in str(error.value)


def test_load_config_rejects_credential_shaped_resolved_data_dir(
    tmp_path: Path,
) -> None:
    config_parent = tmp_path / _SYNTHETIC_SECRET
    config_parent.mkdir()
    config_path = config_parent / "config.toml"
    config_path.write_text(
        """
[model]
model = "test-model"
base_url = "https://example.test/v1"

[runtime]
data_dir = "state"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_config(config_path)

    assert _SYNTHETIC_SECRET not in str(error.value)


def test_load_config_reports_missing_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigNotFoundError):
        load_config(tmp_path / "missing.toml")


@pytest.mark.parametrize("instructions_file", (Path("/tmp/prompt.md"), Path("../prompt.md")))
def test_system_prompt_instructions_file_must_be_workspace_relative(
    instructions_file: Path,
) -> None:
    with pytest.raises(ValidationError, match="workspace-relative"):
        SystemPromptConfig(instructions_file=instructions_file)
