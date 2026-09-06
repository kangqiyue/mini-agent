import json
import os
import stat
from pathlib import Path

import pytest

import mini_agent.tools.apply_patch as apply_patch_module
from mini_agent.messages import ToolCall
from mini_agent.permissions import (
    ApprovalDecision,
    PermissionController,
    built_in_apply_patch_session_grant,
)
from mini_agent.tools.apply_patch import (
    MAX_PATCH_CONTENT_BYTES,
    ApplyPatchArguments,
    ApplyPatchTool,
    FileReplacement,
)
from mini_agent.tools.base import ToolError
from mini_agent.workspace import Workspace
from tests.support.synthetic_secrets import synthetic_stripe_access_token


def test_apply_patch_replaces_existing_file_after_exact_match(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    tool = ApplyPatchTool(Workspace(tmp_path))

    result = tool.execute(_arguments("notes.txt", "before", "after"))

    assert target.read_text(encoding="utf-8") == "after"
    assert result.content == "Modified files:\n- notes.txt"
    assert result.is_truncated is False
    assert tool.definition.is_read_only is False


def _create_arguments(path: str, replacement_content: str) -> str:
    return json.dumps(
        {
            "changes": [
                {
                    "path": path,
                    "replacement_content": replacement_content,
                }
            ]
        }
    )


def test_apply_patch_creates_new_file_without_expected_content(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    tool = ApplyPatchTool(Workspace(tmp_path))

    result = tool.execute(_create_arguments("src/new_file.py", "print('hello')\n"))

    created = tmp_path / "src" / "new_file.py"
    assert created.read_text(encoding="utf-8") == "print('hello')\n"
    assert stat.S_IMODE(created.stat().st_mode) == 0o644
    assert result.content == "Modified files:\n- src/new_file.py"
    assert result.facts is not None
    assert result.facts.modified_paths == ("src/new_file.py",)
    # No temporary files remain next to the created file.
    assert [entry.name for entry in (tmp_path / "src").iterdir()] == ["new_file.py"]


def test_apply_patch_create_rejects_existing_target_without_clobbering(
    tmp_path: Path,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("existing", encoding="utf-8")
    tool = ApplyPatchTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute(_create_arguments("notes.txt", "attacker content"))

    assert error_info.value.code == "target_exists"
    assert target.read_text(encoding="utf-8") == "existing"
    assert [entry.name for entry in tmp_path.iterdir()] == ["notes.txt"]


def test_apply_patch_create_preflight_rejects_existing_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("existing", encoding="utf-8")
    tool = ApplyPatchTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.preflight(_create_arguments("notes.txt", "content"))

    assert error_info.value.code == "target_exists"


def test_apply_patch_create_rejects_missing_parent_directory(tmp_path: Path) -> None:
    tool = ApplyPatchTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute(_create_arguments("missing_dir/notes.txt", "content"))

    assert error_info.value.code == "invalid_path"


def test_apply_patch_create_rejects_sensitive_path(tmp_path: Path) -> None:
    tool = ApplyPatchTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute(_create_arguments(".git/HEAD", "content"))

    assert error_info.value.code == "sensitive_path"
    assert not (tmp_path / ".git").exists()


def test_apply_patch_create_rejects_sensitive_replacement_content(
    tmp_path: Path,
) -> None:
    tool = ApplyPatchTool(Workspace(tmp_path))
    secret = synthetic_stripe_access_token("CREATE")

    with pytest.raises(ToolError) as error_info:
        tool.execute(_create_arguments("notes.txt", f"api = '{secret}'"))

    assert error_info.value.code == "sensitive_replacement_content"
    assert not (tmp_path / "notes.txt").exists()


def test_apply_patch_create_mints_no_session_grant_scope(tmp_path: Path) -> None:
    tool = ApplyPatchTool(Workspace(tmp_path))

    # A create never yields a path-bound session grant: an approved creation
    # must not authorize a later replacement at the same path.
    assert tool.session_grant_scope(_create_arguments("notes.txt", "content")) is None


def test_apply_patch_create_rejects_symlink_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = tmp_path / "linked.txt"
    link.symlink_to(outside)
    tool = ApplyPatchTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute(_create_arguments("linked.txt", "content"))

    assert outside.read_text(encoding="utf-8") == "outside"
    assert link.is_symlink()
    assert error_info.value.code == "invalid_path"


def test_apply_patch_explicit_null_expected_content_selects_create(
    tmp_path: Path,
) -> None:
    tool = ApplyPatchTool(Workspace(tmp_path))
    arguments = json.dumps(
        {
            "changes": [
                {
                    "path": "notes.txt",
                    "expected_content": None,
                    "replacement_content": "created",
                }
            ]
        }
    )

    tool.execute(arguments)

    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "created"


def test_apply_patch_rejects_workspace_root_replaced_after_workspace_initialization(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    original_target = workspace_root / "notes.txt"
    original_target.write_text("before", encoding="utf-8")
    workspace = Workspace(workspace_root)
    moved_workspace = tmp_path / "original-workspace"
    workspace_root.rename(moved_workspace)
    workspace_root.mkdir()
    replacement_target = workspace_root / "notes.txt"
    replacement_target.write_text("attacker content", encoding="utf-8")
    tool = ApplyPatchTool(workspace)

    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments("notes.txt", "attacker content", "after"))

    assert error_info.value.code == "invalid_path"
    assert replacement_target.read_text(encoding="utf-8") == "attacker content"
    assert (moved_workspace / "notes.txt").read_text(encoding="utf-8") == "before"


@pytest.mark.parametrize(
    "replacement_content",
    (
        f"api = '{synthetic_stripe_access_token('PATCH')}'",
        "api = '[REDACTED]'",
        "api = '[REDACTED_KEY_1]'",
    ),
)
def test_apply_patch_refuses_sensitive_or_redacted_replacement_before_write(
    tmp_path: Path, replacement_content: str
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    tool = ApplyPatchTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments("notes.txt", "before", replacement_content))

    assert error_info.value.code == "sensitive_replacement_content"
    assert target.read_text(encoding="utf-8") == "before"
    assert tool.session_grant_scope(
        _arguments("notes.txt", "before", replacement_content)
    ) is None


def test_apply_patch_allows_expected_secret_to_be_removed(tmp_path: Path) -> None:
    secret = synthetic_stripe_access_token("REMOVE")
    target = tmp_path / "notes.txt"
    target.write_text(f"api = '{secret}'", encoding="utf-8")
    tool = ApplyPatchTool(Workspace(tmp_path))

    tool.execute(_arguments("notes.txt", f"api = '{secret}'", "api = ''"))

    assert target.read_text(encoding="utf-8") == "api = ''"


def test_apply_patch_rejects_unknown_arguments(tmp_path: Path) -> None:
    tool = ApplyPatchTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"changes": [], "unexpected": true}')

    assert error_info.value.code == "invalid_arguments"


def test_apply_patch_rejects_multiple_changes(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    tool = ApplyPatchTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute(
            _multi_arguments(
                ("first.txt", "one", "updated"),
                ("second.txt", "two", "updated"),
            )
        )

    assert error_info.value.code == "invalid_arguments"
    assert first.read_text(encoding="utf-8") == "one"
    assert second.read_text(encoding="utf-8") == "two"


@pytest.mark.parametrize("path", ("/tmp/notes.txt", "nested/../notes.txt"))
def test_apply_patch_rejects_absolute_and_parent_paths(tmp_path: Path, path: str) -> None:
    (tmp_path / "notes.txt").write_text("before", encoding="utf-8")
    tool = ApplyPatchTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments(path, "before", "after"))

    assert error_info.value.code == "invalid_path"


@pytest.mark.parametrize(
    "requested_path",
    (
        pytest.param("../notes.txt", id="parent-traversal"),
        pytest.param("/outside.txt", id="absolute"),
        pytest.param("missing.txt", id="missing"),
        pytest.param("excluded/notes.txt", id="excluded"),
        pytest.param("directory", id="non-regular"),
        pytest.param("file-link.txt", id="file-symlink"),
        pytest.param("directory-link/notes.txt", id="directory-symlink"),
    ),
)
def test_apply_patch_preflight_rejects_ineligible_paths_before_execution(
    tmp_path: Path, requested_path: str
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    excluded_directory = tmp_path / "excluded"
    excluded_directory.mkdir()
    (excluded_directory / "notes.txt").write_text("before", encoding="utf-8")
    (tmp_path / "directory").mkdir()
    (tmp_path / "file-link.txt").symlink_to(target)
    linked_directory = tmp_path / "linked-directory"
    linked_directory.mkdir()
    (linked_directory / "notes.txt").write_text("before", encoding="utf-8")
    (tmp_path / "directory-link").symlink_to(linked_directory, target_is_directory=True)
    tool = ApplyPatchTool(Workspace(tmp_path, excluded_roots=(excluded_directory,)))
    arguments = _arguments(requested_path, "before", "after")

    with pytest.raises(ToolError) as preflight_error:
        tool.preflight(arguments)
    with pytest.raises(ToolError) as execute_error:
        tool.execute(arguments)

    assert preflight_error.value.code == "invalid_path"
    assert execute_error.value.code == "invalid_path"
    assert str(tmp_path) not in str(preflight_error.value)
    assert target.read_text(encoding="utf-8") == "before"


def test_apply_patch_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(outside)
    tool = ApplyPatchTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments("link.txt", "outside", "after"))

    assert error_info.value.code == "invalid_path"
    assert outside.read_text(encoding="utf-8") == "outside"


@pytest.mark.parametrize("relative_path", (".git/HEAD", "nested/.svn/entries"))
def test_apply_patch_rejects_vcs_metadata_without_echoing_path(
    tmp_path: Path, relative_path: str
) -> None:
    target = tmp_path / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("before", encoding="utf-8")
    tool = ApplyPatchTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as preflight_error:
        tool.preflight(_arguments(relative_path, "before", "after"))
    with pytest.raises(ToolError) as execute_error:
        tool.execute(_arguments(relative_path, "before", "after"))

    assert preflight_error.value.code == "sensitive_path"
    assert execute_error.value.code == "sensitive_path"
    assert str(preflight_error.value) == "Sensitive workspace paths cannot be modified"
    assert relative_path not in str(preflight_error.value)
    assert target.read_text(encoding="utf-8") == "before"


def test_apply_patch_rejects_sensitive_lexical_symlink_name_before_target_resolution(
    tmp_path: Path,
) -> None:
    safe_target = tmp_path / "notes.txt"
    safe_target.write_text("before", encoding="utf-8")
    sensitive_link = tmp_path / ".git" / "HEAD"
    sensitive_link.parent.mkdir()
    sensitive_link.symlink_to(safe_target)
    tool = ApplyPatchTool(Workspace(tmp_path))
    arguments = _arguments(".git/HEAD", "before", "after")

    with pytest.raises(ToolError) as error_info:
        tool.preflight(arguments)

    assert error_info.value.code == "sensitive_path"
    assert str(error_info.value) == "Sensitive workspace paths cannot be modified"
    assert safe_target.read_text(encoding="utf-8") == "before"


def test_apply_patch_session_scope_rejects_symlink_aliases(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    (tmp_path / "alias.txt").symlink_to(target)
    tool = ApplyPatchTool(Workspace(tmp_path))

    direct_scope = tool.session_grant_scope(_arguments("target.txt", "old", "new"))
    alias_scope = tool.session_grant_scope(_arguments("alias.txt", "old", "new"))

    assert direct_scope is not None
    assert alias_scope is None

    directory = tmp_path / "directory"
    directory.mkdir()
    nested = directory / "nested.txt"
    nested.write_text("old", encoding="utf-8")
    (tmp_path / "directory-alias").symlink_to(directory, target_is_directory=True)

    assert tool.session_grant_scope(
        _arguments("directory-alias/nested.txt", "old", "new")
    ) is None


def test_apply_patch_session_scope_binds_canonical_workspace_and_target(tmp_path: Path) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir()
    workspace_b.mkdir()
    (workspace_a / "target.txt").write_text("old", encoding="utf-8")
    (workspace_b / "target.txt").write_text("old", encoding="utf-8")
    arguments = _arguments("target.txt", "old", "new")

    scope_a = ApplyPatchTool(Workspace(workspace_a)).session_grant_scope(arguments)
    alias_scope_a = ApplyPatchTool(Workspace(workspace_a / ".")).session_grant_scope(
        arguments
    )
    scope_b = ApplyPatchTool(Workspace(workspace_b)).session_grant_scope(arguments)

    assert scope_a is not None
    assert scope_a == alias_scope_a
    assert scope_b is not None
    assert scope_b != scope_a
    assert str(workspace_a) not in scope_a

    source_tool = ApplyPatchTool(Workspace(workspace_a))
    source_call = ToolCall(id="source", name="apply_patch", arguments_json=arguments)
    source_request = PermissionController().request_for(
        source_tool.definition,
        source_call,
        built_in_apply_patch_grant=built_in_apply_patch_session_grant(
            source_tool, arguments
        ),
    )
    assert source_request is not None
    restored = PermissionController(
        session_grant_fingerprints=(source_request.scope_fingerprint,)
    )
    destination_tool = ApplyPatchTool(Workspace(workspace_b))
    destination_call = ToolCall(id="destination", name="apply_patch", arguments_json=arguments)
    destination_request = restored.request_for(
        destination_tool.definition,
        destination_call,
        built_in_apply_patch_grant=built_in_apply_patch_session_grant(
            destination_tool, arguments
        ),
    )

    assert destination_request is not None
    assert restored.decide(destination_request) is ApprovalDecision.DENY


def test_apply_patch_rejects_target_retargeted_to_symlink_after_scope_resolution(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    outside = tmp_path.parent / "outside-after-scope.txt"
    outside.write_text("outside", encoding="utf-8")
    tool = ApplyPatchTool(Workspace(tmp_path))
    arguments = _arguments("target.txt", "old", "new")

    assert tool.session_grant_scope(arguments) is not None
    target.unlink()
    target.symlink_to(outside)

    with pytest.raises(ToolError) as error_info:
        tool.execute(arguments)

    assert error_info.value.code == "invalid_path"
    assert str(tmp_path) not in str(error_info.value)
    assert outside.read_text(encoding="utf-8") == "outside"


def test_apply_patch_rejects_missing_file(tmp_path: Path) -> None:
    tool = ApplyPatchTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments("missing.txt", "", "after"))

    assert error_info.value.code == "invalid_path"


def test_apply_patch_mismatch_does_not_modify_file(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("current", encoding="utf-8")
    tool = ApplyPatchTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments("notes.txt", "stale", "updated"))

    assert error_info.value.code == "expected_content_mismatch"
    assert target.read_text(encoding="utf-8") == "current"


def test_apply_patch_rejects_oversized_target_without_modification(
    tmp_path: Path,
) -> None:
    target = tmp_path / "large.txt"
    original_content = b"x" * (MAX_PATCH_CONTENT_BYTES + 1)
    target.write_bytes(original_content)
    tool = ApplyPatchTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments("large.txt", "not read", "replacement"))

    assert error_info.value.code == "file_too_large"
    assert target.read_bytes() == original_content


def test_apply_patch_accepts_exact_multibyte_boundary(tmp_path: Path) -> None:
    target = tmp_path / "multibyte.txt"
    character = "界"
    character_bytes = len(character.encode("utf-8"))
    expected_content = character * (MAX_PATCH_CONTENT_BYTES // character_bytes)
    expected_content += "a" * (
        MAX_PATCH_CONTENT_BYTES - len(expected_content.encode("utf-8"))
    )
    target.write_text(expected_content, encoding="utf-8")
    tool = ApplyPatchTool(Workspace(tmp_path))

    tool.execute(_arguments("multibyte.txt", expected_content, "after"))

    assert target.read_text(encoding="utf-8") == "after"


@pytest.mark.parametrize("oversized_field", ("expected_content", "replacement_content"))
def test_apply_patch_counts_content_limits_in_utf8_bytes(
    tmp_path: Path, oversized_field: str
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    oversized_content = "界" * (MAX_PATCH_CONTENT_BYTES // 3 + 1)
    expected_content = oversized_content if oversized_field == "expected_content" else "before"
    replacement_content = (
        oversized_content if oversized_field == "replacement_content" else "after"
    )
    tool = ApplyPatchTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments("notes.txt", expected_content, replacement_content))

    assert error_info.value.code == "content_too_large"
    assert target.read_text(encoding="utf-8") == "before"


def test_apply_patch_preserves_target_file_mode(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    target.chmod(0o755)
    tool = ApplyPatchTool(Workspace(tmp_path))

    tool.execute(_arguments("notes.txt", "before", "after"))

    assert stat.S_IMODE(target.stat().st_mode) == 0o755


def test_apply_patch_fails_closed_when_target_changes_before_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    tool = ApplyPatchTool(Workspace(tmp_path))
    original_create_temporary_file = apply_patch_module.create_replacement_temporary_file

    def change_before_temporary_file(
        parent_descriptor: int, target_name: str
    ) -> tuple[str, int]:
        target.write_text("concurrent edit", encoding="utf-8")
        return original_create_temporary_file(parent_descriptor, target_name)

    monkeypatch.setattr(
        apply_patch_module,
        "create_replacement_temporary_file",
        change_before_temporary_file,
    )

    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments("notes.txt", "before", "after"))

    assert error_info.value.code == "target_changed"
    assert target.read_text(encoding="utf-8") == "concurrent edit"


def test_apply_patch_rejects_intermediate_directory_swapped_to_outside_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    target = nested_directory / "target.txt"
    target.write_text("before", encoding="utf-8")
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    outside_target = outside_directory / "target.txt"
    outside_target.write_text("outside", encoding="utf-8")
    moved_directory = tmp_path / "moved-inside-workspace"
    tool = ApplyPatchTool(Workspace(tmp_path))
    original_create_temporary_file = apply_patch_module.create_replacement_temporary_file

    def swap_directory_before_replace(
        parent_descriptor: int, target_name: str
    ) -> tuple[str, int]:
        nested_directory.rename(moved_directory)
        nested_directory.symlink_to(outside_directory, target_is_directory=True)
        return original_create_temporary_file(parent_descriptor, target_name)

    monkeypatch.setattr(
        apply_patch_module,
        "create_replacement_temporary_file",
        swap_directory_before_replace,
    )

    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments("nested/target.txt", "before", "after"))

    assert error_info.value.code == "target_changed"
    assert outside_target.read_text(encoding="utf-8") == "outside"
    assert (moved_directory / "target.txt").read_text(encoding="utf-8") == "before"


def test_apply_patch_reports_unknown_status_when_replace_raises_before_replacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    tool = ApplyPatchTool(Workspace(tmp_path))

    def fail_replace(
        source: object,
        destination: object,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("mini_agent.tools.apply_patch.os.replace", fail_replace)

    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments("notes.txt", "before", "after"))

    assert error_info.value.code == "write_status_unknown"
    assert target.read_text(encoding="utf-8") == "before"


def test_apply_patch_reports_write_failure_before_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    tool = ApplyPatchTool(Workspace(tmp_path))

    def fail_create_temporary_file(*_args: object, **_kwargs: object) -> tuple[str, int]:
        raise OSError("simulated temporary-file failure")

    monkeypatch.setattr(
        apply_patch_module,
        "create_replacement_temporary_file",
        fail_create_temporary_file,
    )

    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments("notes.txt", "before", "after"))

    assert error_info.value.code == "write_failed"
    assert target.read_text(encoding="utf-8") == "before"


def test_apply_patch_reports_unknown_status_when_replace_applies_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    tool = ApplyPatchTool(Workspace(tmp_path))
    original_replace = os.replace

    def replace_then_raise(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
    ) -> None:
        original_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        raise OSError("simulated post-replace failure")

    monkeypatch.setattr("mini_agent.tools.apply_patch.os.replace", replace_then_raise)

    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments("notes.txt", "before", "after"))

    assert error_info.value.code == "write_status_unknown"
    assert target.read_text(encoding="utf-8") == "after"


def test_apply_patch_reports_unknown_status_after_replace_when_directory_sync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    tool = ApplyPatchTool(Workspace(tmp_path))
    original_fsync = os.fsync
    fsync_calls = 0

    def fail_directory_sync(file_descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("simulated directory sync failure")
        original_fsync(file_descriptor)

    monkeypatch.setattr("mini_agent.tools.apply_patch.os.fsync", fail_directory_sync)

    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments("notes.txt", "before", "after"))

    assert error_info.value.code == "write_status_unknown"
    assert target.read_text(encoding="utf-8") == "after"


@pytest.mark.parametrize("operation", ["preflight", "execute"])
def test_apply_patch_create_rejects_an_excluded_leaf(
    tmp_path: Path, operation: str
) -> None:
    target = tmp_path / "runtime-store"
    tool = ApplyPatchTool(Workspace(tmp_path, excluded_roots=(target,)))
    arguments = ApplyPatchArguments(changes=(FileReplacement(
        path=target.name, replacement_content="new file",
    ),)).model_dump_json()

    with pytest.raises(ToolError, match="workspace"):
        if operation == "preflight":
            tool.preflight(arguments)
        else:
            tool.execute(arguments)
    assert not target.exists()


def test_apply_patch_create_reports_unknown_when_link_lands_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_link = os.link

    def link_then_raise(
        source: str, destination: str, *, src_dir_fd: int, dst_dir_fd: int,
        follow_symlinks: bool = True,
    ) -> None:
        original_link(source, destination, src_dir_fd=src_dir_fd,
                      dst_dir_fd=dst_dir_fd, follow_symlinks=follow_symlinks)
        raise OSError("simulated post-link failure")

    monkeypatch.setattr(apply_patch_module.os, "link", link_then_raise)
    tool = ApplyPatchTool(Workspace(tmp_path))
    with pytest.raises(ToolError) as error_info:
        tool.execute(_create_arguments("notes.txt", "created"))

    assert error_info.value.code == "write_status_unknown"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "created"
    assert [entry.name for entry in tmp_path.iterdir()] == ["notes.txt"]


def test_apply_patch_create_never_overwrites_a_concurrent_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_link = os.link

    def create_before_link(
        source: str, destination: str, *, src_dir_fd: int, dst_dir_fd: int,
        follow_symlinks: bool = True,
    ) -> None:
        (tmp_path / destination).write_text("concurrent", encoding="utf-8")
        original_link(source, destination, src_dir_fd=src_dir_fd,
                      dst_dir_fd=dst_dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(apply_patch_module.os, "link", create_before_link)
    tool = ApplyPatchTool(Workspace(tmp_path))
    with pytest.raises(ToolError) as error_info:
        tool.execute(_create_arguments("notes.txt", "created"))

    assert error_info.value.code == "target_exists"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "concurrent"


def test_apply_patch_closes_open_target_when_it_grows_past_the_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("before", encoding="utf-8")
    tool = ApplyPatchTool(Workspace(tmp_path))
    original_open = apply_patch_module.open_regular_at
    descriptors: list[int] = []

    def open_then_grow(parent_descriptor: int, leaf_name: str) -> int:
        descriptor = original_open(parent_descriptor, leaf_name)
        descriptors.append(descriptor)
        if len(descriptors) == 2:
            target.write_bytes(b"x" * (MAX_PATCH_CONTENT_BYTES + 1))
        return descriptor

    monkeypatch.setattr(apply_patch_module, "open_regular_at", open_then_grow)
    with pytest.raises(ToolError) as error_info:
        tool.execute(_arguments("notes.txt", "before", "after"))
    assert error_info.value.code == "file_too_large"
    assert len(descriptors) == 2
    try:
        with pytest.raises(OSError):
            os.fstat(descriptors[-1])
    finally:
        # Keep the failing regression run from itself retaining the leaked fd.
        try:
            os.fstat(descriptors[-1])
        except OSError:
            pass
        else:
            os.close(descriptors[-1])


def _arguments(path: str, expected_content: str, replacement_content: str) -> str:
    return _multi_arguments((path, expected_content, replacement_content))


def _multi_arguments(*changes: tuple[str, str, str]) -> str:
    return json.dumps(
        {
            "changes": [
                {
                    "path": path,
                    "expected_content": expected_content,
                    "replacement_content": replacement_content,
                }
                for path, expected_content, replacement_content in changes
            ]
        }
    )
