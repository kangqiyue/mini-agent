import math
from typing import cast

import pytest
from pydantic import ValidationError

from mini_agent.tools.base import (
    MAX_TOOL_DESCRIPTION_UTF8_BYTES,
    MAX_TOOL_SCHEMA_DEPTH,
    MAX_TOOL_SCHEMA_KEY_UTF8_BYTES,
    MAX_TOOL_SCHEMA_NODES,
    MAX_TOOL_SCHEMA_UTF8_BYTES,
    ToolDefinition,
)


def _definition(**overrides: object) -> ToolDefinition:
    values: dict[str, object] = {
        "name": "read_file",
        "description": "Read one workspace file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        "is_read_only": True,
    }
    values.update(overrides)
    return ToolDefinition.model_validate(values)


def _nested_schema(depth: int) -> dict[str, object]:
    schema: object = "value"
    for _ in range(depth):
        schema = {"nested": schema}
    assert isinstance(schema, dict)
    return cast(dict[str, object], schema)


def _validation_message(**overrides: object) -> str:
    with pytest.raises(ValidationError) as error_info:
        _definition(**overrides)
    return str(error_info.value)


@pytest.mark.parametrize(
    "parameters",
    (
        ["not", "an", "object"],
        {"tuple": ("not", "json")},
        {"set": {"not-json"}},
        {"object": object()},
        {1: "not-a-string-key"},
        {True: "not-a-string-key"},
        {"nan": math.nan},
        {"infinity": math.inf},
    ),
)
def test_tool_definition_rejects_non_json_schema_values(parameters: object) -> None:
    message = _validation_message(parameters=parameters)

    assert "Tool definition" in message


def test_tool_definition_rejects_excessive_schema_depth() -> None:
    message = _validation_message(parameters=_nested_schema(MAX_TOOL_SCHEMA_DEPTH + 1))

    assert "nesting" in message


def test_tool_definition_rejects_excessive_schema_nodes() -> None:
    message = _validation_message(
        parameters={"items": [0] * MAX_TOOL_SCHEMA_NODES},
    )

    assert "node" in message


def test_tool_definition_rejects_excessive_schema_bytes() -> None:
    message = _validation_message(
        parameters={"type": "object", "description": "x" * MAX_TOOL_SCHEMA_UTF8_BYTES},
    )

    assert "UTF-8 byte" in message


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("description", "x" * (MAX_TOOL_DESCRIPTION_UTF8_BYTES + 1)),
        ("parameters", {"x" * (MAX_TOOL_SCHEMA_KEY_UTF8_BYTES + 1): "value"}),
    ),
)
def test_tool_definition_rejects_excessive_text_field_bytes(
    field_name: str,
    value: object,
) -> None:
    message = _validation_message(**{field_name: value})

    assert "UTF-8 byte" in message


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("name", "sk_live_" + "0" * 24),
        ("description", "Bearer sk_live_" + "0" * 24),
        ("parameters", {"api_key": {"type": "string"}}),
        ("parameters", {"default": "sk_live_" + "0" * 24}),
    ),
)
def test_tool_definition_rejects_credential_shaped_content(
    field_name: str,
    value: object,
) -> None:
    message = _validation_message(**{field_name: value})

    assert "credential-shaped" in message
    assert str(value) not in message


def test_tool_definition_rejects_credential_shaped_text_without_echoing_it() -> None:
    secret = "sk_live_" + "0" * 24
    message = _validation_message(description=f"Use {secret}.")

    assert secret not in message
    assert "credential-shaped" in message


def test_tool_definition_allows_non_secret_documentation_labels() -> None:
    definition = _definition(
        description="Describe the API key environment variable without its value.",
        parameters={"type": "object", "description": "A secret label, not a secret."},
    )

    assert definition.description.startswith("Describe")


@pytest.mark.parametrize("value", (0, 1, "true", "false"))
def test_tool_definition_requires_a_strict_boolean_read_only_flag(value: object) -> None:
    message = _validation_message(is_read_only=value)

    assert "is_read_only" in message


def test_tool_definition_accepts_schema_at_each_structural_boundary() -> None:
    # The root object is depth one and the scalar leaf takes the final level.
    nested = _nested_schema(MAX_TOOL_SCHEMA_DEPTH - 2)
    definition = _definition(
        description="d" * MAX_TOOL_DESCRIPTION_UTF8_BYTES,
        parameters={"key": "k" * MAX_TOOL_SCHEMA_KEY_UTF8_BYTES, "nested": nested},
    )

    assert definition.parameters["key"] == "k" * MAX_TOOL_SCHEMA_KEY_UTF8_BYTES


def test_tool_definition_accepts_maximum_schema_depth_and_node_count() -> None:
    at_depth_limit = _definition(parameters=_nested_schema(MAX_TOOL_SCHEMA_DEPTH - 1))
    at_node_limit = _definition(
        parameters={"items": [0] * (MAX_TOOL_SCHEMA_NODES - 2)},
    )

    assert at_depth_limit.parameters
    items = cast(list[object], at_node_limit.parameters["items"])
    assert len(items) == MAX_TOOL_SCHEMA_NODES - 2


def test_tool_definition_accepts_exact_canonical_schema_byte_limit() -> None:
    serialized_overhead = len(b'{"value":""}')
    definition = _definition(
        parameters={"value": "x" * (MAX_TOOL_SCHEMA_UTF8_BYTES - serialized_overhead)},
    )

    value = definition.parameters["value"]
    assert isinstance(value, str)
    assert len(value) == (
        MAX_TOOL_SCHEMA_UTF8_BYTES - serialized_overhead
    )
