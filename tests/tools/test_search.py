import json
import os
import shlex
from pathlib import Path
from typing import Any, cast

import pytest
from pytest import MonkeyPatch

import mini_agent.workspace_directory_fd as workspace_directory_fd
import mini_agent.workspace_subprocess as workspace_subprocess
from mini_agent.tools.base import ToolError
from mini_agent.tools.search import SearchTool
from mini_agent.workspace import Workspace


def _write_executable(path: Path, output: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{output}'\n", encoding="utf-8")
    path.chmod(0o755)


def _json_match(*, text: str, relative_path: str = "source.txt") -> str:
    return json.dumps(
        {
            "type": "match",
            "data": {
                "path": {"text": relative_path},
                "lines": {"text": text + "\n"},
                "line_number": 1,
                "submatches": [{"start": 0}],
            },
        }
    )


def _private_key_pem() -> str:
    opening = "-----BEGIN " + "PRIVATE" + " KEY-----"
    closing = "-----END " + "PRIVATE" + " KEY-----"
    return "\n".join((opening, "encoded-private-material", closing, ""))


def test_search_returns_bounded_matches(tmp_path: Path) -> None:
    (tmp_path / "first.py").write_text("needle = 1\n", encoding="utf-8")
    (tmp_path / "second.txt").write_text("needle\nneedle\n", encoding="utf-8")
    tool = SearchTool(Workspace(tmp_path), max_matches=2)

    result = tool.execute('{"query":"needle"}')

    assert result.content.count("needle") == 2
    assert result.is_truncated is True


def test_search_applies_glob_filter(tmp_path: Path) -> None:
    (tmp_path / "first.py").write_text("needle = 1\n", encoding="utf-8")
    (tmp_path / "second.txt").write_text("needle\n", encoding="utf-8")
    tool = SearchTool(Workspace(tmp_path))

    result = tool.execute('{"query":"needle","globs":["*.py"]}')

    assert "first.py" in result.content
    assert "second.txt" not in result.content


def test_search_reports_no_matches(tmp_path: Path) -> None:
    (tmp_path / "first.py").write_text("haystack = 1\n", encoding="utf-8")
    tool = SearchTool(Workspace(tmp_path))

    result = tool.execute('{"query":"needle"}')

    assert result.content == "No matches found."
    assert result.is_truncated is False


@pytest.mark.parametrize(
    "relative_path",
    (
        ".netrc",
        "nested/.config/gh/hosts.yml",
        "nested/.mini-agent/config.toml",
        "nested/.git/HEAD",
    ),
)
def test_search_rejects_runtime_sensitive_search_roots_without_echoing_path_or_content(
    tmp_path: Path, relative_path: str
) -> None:
    secret = "unrecognized-" + "netrc-" + "secret"
    target = tmp_path / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(secret, encoding="utf-8")
    tool = SearchTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute(json.dumps({"query": "unrecognized", "path": relative_path}))

    assert error_info.value.code == "sensitive_path"
    assert str(error_info.value) == "Sensitive workspace paths cannot be searched"
    assert relative_path not in str(error_info.value)
    assert secret not in str(error_info.value)


def test_search_rejects_sensitive_path_emitted_by_ripgrep_before_returning_match(
    tmp_path: Path,
) -> None:
    secret = "unrecognized-" + "netrc-" + "secret"
    (tmp_path / ".netrc").write_text(secret, encoding="utf-8")
    trusted_executable = tmp_path.parent / "trusted-rg"
    _write_executable(
        trusted_executable,
        _json_match(text=secret, relative_path=".netrc"),
    )
    tool = SearchTool(Workspace(tmp_path), executable=str(trusted_executable))

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"query":"unrecognized"}')

    assert error_info.value.code == "sensitive_path"
    assert str(error_info.value) == "Sensitive workspace paths cannot be searched"
    assert secret not in str(error_info.value)


def test_search_rejects_sensitive_lexical_symlink_name_before_running_ripgrep(
    tmp_path: Path,
) -> None:
    secret = "unrecognized-" + "config-" + "secret"
    safe_target = tmp_path / "notes.txt"
    safe_target.write_text(secret, encoding="utf-8")
    sensitive_link = tmp_path / "nested" / ".mini-agent" / "config.toml"
    sensitive_link.parent.mkdir(parents=True)
    sensitive_link.symlink_to(safe_target)
    tool = SearchTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute(
            json.dumps({"query": "unrecognized", "path": "nested/.mini-agent/config.toml"})
        )

    assert error_info.value.code == "sensitive_path"
    assert str(error_info.value) == "Sensitive workspace paths cannot be searched"
    assert secret not in str(error_info.value)


def test_search_keeps_content_past_default_context_budget(tmp_path: Path) -> None:
    tail = "needle-after-12k"
    (tmp_path / "large.txt").write_text("needle " + "x" * 12_100 + tail + "\n", encoding="utf-8")
    tool = SearchTool(Workspace(tmp_path))

    result = tool.execute('{"query":"needle"}')

    assert tail in result.content
    assert result.is_truncated is False
    assert tail not in result.preview_for_context(12_000)


def test_search_marks_partial_json_at_byte_limit_as_truncated_not_failed(
    tmp_path: Path,
) -> None:
    (tmp_path / "match.txt").write_text("needle\n", encoding="utf-8")
    tool = SearchTool(Workspace(tmp_path), max_chars=16)

    result = tool.execute('{"query":"needle"}')

    assert result.content == "No matches found."
    assert result.is_truncated is True


def test_search_rejects_private_key_file_before_returning_matching_body(
    tmp_path: Path,
) -> None:
    (tmp_path / "unusual-extension.data").write_text(
        _private_key_pem(), encoding="utf-8"
    )
    tool = SearchTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"query":"encoded-private-material"}')

    assert error_info.value.code == "private_key_material"
    assert str(error_info.value) == "Refusing to return private key material"
    assert "unusual-extension.data" not in str(error_info.value)
    assert "encoded-private-material" not in str(error_info.value)


def test_search_fails_closed_when_matching_file_is_replaced_by_symlink_before_open(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    requested_path = tmp_path / "source.txt"
    requested_path.write_text("safe matching text\n", encoding="utf-8")
    secret = "do-not-return-this-private-content"
    replacement_target = tmp_path / "private-target.txt"
    replacement_target.write_text(secret, encoding="utf-8")
    trusted_executable = tmp_path.parent / "trusted-rg"
    _write_executable(
        trusted_executable,
        _json_match(text="matching text", relative_path="source.txt"),
    )
    original_open = os.open
    has_replaced = False

    def replace_final_path_before_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        *args: int,
        **kwargs: int,
    ) -> int:
        nonlocal has_replaced
        if not has_replaced and path == "source.txt" and "dir_fd" in kwargs:
            has_replaced = True
            requested_path.unlink()
            requested_path.symlink_to(replacement_target)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(
        workspace_directory_fd.os, "open", replace_final_path_before_open
    )
    tool = SearchTool(Workspace(tmp_path), executable=str(trusted_executable))

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"query":"matching"}')

    assert has_replaced is True
    assert error_info.value.code == "search_failed"
    assert str(error_info.value) == "Search result file could not be classified"
    assert secret not in str(error_info.value)


def test_search_keeps_held_parent_for_private_key_classification_after_swap(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    (nested_directory / "source.txt").write_text("safe matching text\n", encoding="utf-8")
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    secret = "outside-private-key-material-must-not-be-classified"
    (outside_directory / "source.txt").write_text(
        _private_key_pem() + secret,
        encoding="utf-8",
    )
    trusted_executable = tmp_path.parent / "trusted-rg"
    _write_executable(
        trusted_executable,
        _json_match(text="safe matching text", relative_path="nested/source.txt"),
    )
    moved_directory = tmp_path / "nested-moved"
    original_open = os.open
    has_swapped = False

    def swap_parent_before_leaf_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        *args: int,
        **kwargs: int,
    ) -> int:
        nonlocal has_swapped
        if not has_swapped and path == "source.txt" and "dir_fd" in kwargs:
            has_swapped = True
            nested_directory.rename(moved_directory)
            nested_directory.symlink_to(outside_directory, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(
        workspace_directory_fd.os, "open", swap_parent_before_leaf_open
    )
    tool = SearchTool(Workspace(tmp_path), executable=str(trusted_executable))

    result = tool.execute('{"query":"matching"}')

    assert has_swapped is True
    assert "safe matching text" in result.content
    assert secret not in result.content


def test_search_runs_ripgrep_from_held_root_after_root_path_swap(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "marker.txt").write_text("held-root-content", encoding="utf-8")
    (workspace_root / "source.txt").write_text("safe source\n", encoding="utf-8")
    observer = tmp_path / "observed-root-content"
    trusted_executable = tmp_path.parent / "trusted-rg"
    trusted_executable.write_text(
        "#!/bin/sh\n"
        f"cat marker.txt > {shlex.quote(str(observer))}\n"
        f"printf '%s\\n' '{_json_match(text='safe source')}'\n",
        encoding="utf-8",
    )
    trusted_executable.chmod(0o755)
    tool = SearchTool(Workspace(workspace_root), executable=str(trusted_executable))
    original_start: object = workspace_subprocess.WorkspaceSubprocessLauncher.start
    moved_root = tmp_path / "workspace-moved"
    secret = "replacement-root-content-must-not-be-read"

    def swap_root_before_bootstrap(
        launcher: workspace_subprocess.WorkspaceSubprocessLauncher,
        **kwargs: object,
    ) -> object:
        workspace_root.rename(moved_root)
        workspace_root.mkdir()
        (workspace_root / "marker.txt").write_text(secret, encoding="utf-8")
        (workspace_root / "source.txt").write_text(secret, encoding="utf-8")
        return cast(Any, original_start)(launcher, **kwargs)

    monkeypatch.setattr(
        workspace_subprocess.WorkspaceSubprocessLauncher,
        "start",
        swap_root_before_bootstrap,
    )

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"query":"safe"}')

    assert error_info.value.code == "search_failed"
    assert observer.read_text(encoding="utf-8") == "held-root-content"
    assert secret not in str(error_info.value)


def test_search_allows_certificates_public_keys_and_private_key_prose(
    tmp_path: Path,
) -> None:
    content = "\n".join(
        (
            "-----BEGIN CERTIFICATE-----",
            "certificate-body",
            "-----END CERTIFICATE-----",
            "-----BEGIN PUBLIC KEY-----",
            "public-key-body",
            "-----END PUBLIC KEY-----",
            "This prose mentions a PRIVATE KEY but is not PEM material.",
            "",
        )
    )
    (tmp_path / "public-material.txt").write_text(content, encoding="utf-8")
    tool = SearchTool(Workspace(tmp_path))

    result = tool.execute('{"query":"body|prose"}')

    assert "certificate-body" in result.content
    assert "public-key-body" in result.content
    assert "PRIVATE KEY but is not PEM" in result.content


def test_search_fails_closed_when_matching_files_exceed_total_classification_budget(
    tmp_path: Path,
) -> None:
    (tmp_path / "first.txt").write_text("needle-1111\n", encoding="utf-8")
    (tmp_path / "second.txt").write_text("needle-222\n", encoding="utf-8")
    tool = SearchTool(
        Workspace(tmp_path),
        max_scan_bytes=32,
        max_total_scan_bytes=20,
    )

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"query":"needle"}')

    assert error_info.value.code == "search_scan_limit_exceeded"
    assert str(error_info.value) == "Search result file classification exceeded its limit"


def test_search_deduplicates_file_classification_across_multiple_matching_lines(
    tmp_path: Path,
) -> None:
    (tmp_path / "matches.txt").write_text("needle-one\nneedle-two\n", encoding="utf-8")
    tool = SearchTool(
        Workspace(tmp_path),
        max_scan_bytes=32,
        max_total_scan_bytes=24,
    )

    result = tool.execute('{"query":"needle"}')

    assert "needle-one" in result.content
    assert "needle-two" in result.content


def test_search_from_workspace_root_excludes_runtime_data_dir_even_with_reinclude_globs(
    tmp_path: Path,
) -> None:
    (tmp_path / "source.txt").write_text("visible-needle\n", encoding="utf-8")
    runtime_dir = tmp_path / "agent-data"
    runtime_dir.mkdir()
    (runtime_dir / "events.txt").write_text("private-needle\n", encoding="utf-8")
    tool = SearchTool(Workspace(tmp_path, excluded_roots=(runtime_dir,)))

    result = tool.execute(
        '{"query":"needle","path":".","globs":["**","agent-data","agent-data/**"]}'
    )

    assert "visible-needle" in result.content
    assert "private-needle" not in result.content


def test_search_ignores_ripgrep_config_that_enables_symlink_following(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "private.txt").write_text("private-needle\n", encoding="utf-8")
    (workspace_root / "outside-link").symlink_to(outside_dir, target_is_directory=True)
    config_path = tmp_path / "ripgreprc"
    config_path.write_text("--follow\n", encoding="utf-8")
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(config_path))
    tool = SearchTool(Workspace(workspace_root))

    result = tool.execute('{"query":"needle","path":"."}')

    assert result.content == "No matches found."


def test_search_resolves_relative_path_from_launch_directory_not_workspace(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    launcher_root = tmp_path / "launcher"
    launcher_root.mkdir()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "source.txt").write_text("source\n", encoding="utf-8")
    _write_executable(launcher_root / "bin" / "rg", _json_match(text="trusted-launcher-rg"))
    _write_executable(workspace_root / "bin" / "rg", _json_match(text="untrusted-workspace-rg"))

    monkeypatch.chdir(launcher_root)
    monkeypatch.setenv("PATH", "bin")
    tool = SearchTool(Workspace(workspace_root))
    monkeypatch.chdir(workspace_root)

    result = tool.execute('{"query":"needle"}')

    assert result.content == "source.txt:1:1:trusted-launcher-rg\n"


def test_search_resolves_empty_path_entry_from_launch_directory_not_workspace(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    launcher_root = tmp_path / "launcher"
    launcher_root.mkdir()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "source.txt").write_text("source\n", encoding="utf-8")
    _write_executable(launcher_root / "rg", _json_match(text="trusted-launcher-rg"))
    _write_executable(workspace_root / "rg", _json_match(text="untrusted-workspace-rg"))

    monkeypatch.chdir(launcher_root)
    monkeypatch.setenv("PATH", "")
    tool = SearchTool(Workspace(workspace_root))
    monkeypatch.chdir(workspace_root)

    result = tool.execute('{"query":"needle"}')

    assert result.content == "source.txt:1:1:trusted-launcher-rg\n"


def test_search_rejects_workspace_executable_found_via_path(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _write_executable(workspace_root / "bin" / "rg", "untrusted-workspace-rg")
    monkeypatch.chdir(workspace_root)
    monkeypatch.setenv("PATH", "bin")
    tool = SearchTool(Workspace(workspace_root))

    with pytest.raises(ToolError, match="cannot be located inside the workspace") as error:
        tool.execute('{"query":"needle"}')

    assert error.value.code == "unsafe_dependency"


def test_search_runs_trusted_absolute_executable_even_if_path_has_workspace_binary(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    trusted_root = tmp_path / "trusted"
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    trusted_executable = trusted_root / "rg"
    (workspace_root / "source.txt").write_text("source\n", encoding="utf-8")
    _write_executable(trusted_executable, _json_match(text="trusted-absolute-rg"))
    _write_executable(workspace_root / "bin" / "rg", _json_match(text="untrusted-workspace-rg"))
    monkeypatch.chdir(workspace_root)
    monkeypatch.setenv("PATH", "bin")
    tool = SearchTool(Workspace(workspace_root), executable=str(trusted_executable))

    result = tool.execute('{"query":"needle"}')

    assert result.content == "source.txt:1:1:trusted-absolute-rg\n"
