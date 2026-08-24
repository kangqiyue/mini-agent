from pathlib import Path

import pytest

from mini_agent.artifacts import ArtifactStore
from mini_agent.tools.artifact_read import ArtifactReadTool
from mini_agent.tools.base import ToolError


def test_artifact_read_returns_only_a_bounded_registered_window(tmp_path: Path) -> None:
    store = ArtifactStore.open(tmp_path, registrations=(), workspace_root=tmp_path)
    record = store.create("0123456789", source_event_id=1)
    tool = ArtifactReadTool(store)

    result = tool.execute(f'{{"artifact_id":"{record.artifact_id}","offset":3,"limit":4}}')

    assert result.content == "3456"
    assert result.is_truncated is True
    properties = tool.definition.parameters["properties"]
    assert isinstance(properties, dict)
    assert "path" not in properties


def test_artifact_read_rejects_unknown_or_unregistered_artifacts(tmp_path: Path) -> None:
    tool = ArtifactReadTool(
        ArtifactStore.open(tmp_path, registrations=(), workspace_root=tmp_path)
    )

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"artifact_id":"a"}')
    assert error_info.value.code == "invalid_arguments"

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"artifact_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}')
    assert error_info.value.code == "artifact_not_found"
