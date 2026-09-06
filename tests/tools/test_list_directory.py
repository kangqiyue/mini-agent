from pathlib import Path

import pytest

import mini_agent.tools.list_directory as listing_module
from mini_agent.tools.base import ToolError
from mini_agent.tools.list_directory import ListDirectoryArguments, ListDirectoryTool
from mini_agent.workspace import Workspace


def test_list_directory_renders_sorted_names_and_types(tmp_path: Path) -> None:
    (tmp_path / "a_directory").mkdir()
    (tmp_path / "b_file").write_text("content", encoding="utf-8")
    (tmp_path / "c_link").symlink_to(tmp_path / "b_file")
    tool = ListDirectoryTool(Workspace(tmp_path))

    result = tool.execute(ListDirectoryArguments().model_dump_json())

    assert result.content == "d a_directory\nf b_file\nl c_link"
    assert not result.is_truncated


@pytest.mark.parametrize("entry_count", [0, 2, 3, 20])
def test_list_directory_bounds_metadata_reads_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry_count: int
) -> None:
    for index in range(entry_count):
        (tmp_path / f"file_{index:02}").touch()
    original_entry_type = listing_module._entry_type  # pyright: ignore[reportPrivateUsage]
    inspected: list[str] = []

    def entry_type(descriptor: int, name: str) -> str:
        inspected.append(name)
        return original_entry_type(descriptor, name)

    monkeypatch.setattr(listing_module, "_entry_type", entry_type)
    result = ListDirectoryTool(Workspace(tmp_path), max_entries=2).execute(
        ListDirectoryArguments().model_dump_json()
    )

    assert len(inspected) <= 2
    assert result.is_truncated == (entry_count > 2)
    assert len([line for line in result.content.splitlines() if line.startswith("f ")]) == min(
        2, entry_count
    )


@pytest.mark.parametrize("path", [".git", "../outside", "missing"])
def test_list_directory_rejects_ineligible_paths(tmp_path: Path, path: str) -> None:
    with pytest.raises(ToolError):
        ListDirectoryTool(Workspace(tmp_path)).execute(
            ListDirectoryArguments(path=path).model_dump_json()
        )


def test_list_directory_does_not_follow_symlink_directory(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(ToolError):
        ListDirectoryTool(Workspace(tmp_path)).execute(
            ListDirectoryArguments(path="link").model_dump_json()
        )
