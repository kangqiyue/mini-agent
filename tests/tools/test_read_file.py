import os
from pathlib import Path

import pytest

import mini_agent.private_key_material as private_key_material
import mini_agent.workspace_directory_fd as workspace_directory_fd
from mini_agent.tools.base import ToolError
from mini_agent.tools.read_file import ReadFileTool
from mini_agent.workspace import Workspace


def _private_key_pem(*, kind: str = "", include_end: bool = True) -> str:
    kind_prefix = f"{kind} " if kind else ""
    opening = "-----BEGIN " + kind_prefix + "PRIVATE" + " KEY-----"
    closing = "-----END " + kind_prefix + "PRIVATE" + " KEY-----"
    lines = [opening, "encoded-private-material"]
    if include_end:
        lines.append(closing)
    return "\n".join(lines) + "\n"


def _netrc_with_unrecognized_whitespace_secret() -> tuple[str, str]:
    secret = "unrecognized-" + "netrc-" + "secret"
    return f"machine example.test   login user\tpassword {secret}\n", secret


def test_read_file_returns_numbered_line_range(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    tool = ReadFileTool(Workspace(tmp_path))

    result = tool.execute('{"path":"notes.txt","start_line":2,"end_line":3}')

    assert result.content == "     2 | two\n     3 | three\n"
    assert result.is_truncated is False


def test_read_file_marks_result_truncated_at_line_limit(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    tool = ReadFileTool(Workspace(tmp_path), max_lines=2)

    result = tool.execute('{"path":"notes.txt"}')

    assert result.is_truncated is True
    assert "three" not in result.content


def test_read_file_keeps_content_past_default_context_budget(tmp_path: Path) -> None:
    tail = "end-after-12k"
    (tmp_path / "notes.txt").write_text("x" * 12_100 + "\n" + tail + "\n", encoding="utf-8")
    tool = ReadFileTool(Workspace(tmp_path))

    result = tool.execute('{"path":"notes.txt"}')

    assert tail in result.content
    assert result.is_truncated is False
    assert result.preview_for_context(12_000) != result.content
    assert tail not in result.preview_for_context(12_000)


def test_read_file_rejects_unknown_arguments(tmp_path: Path) -> None:
    tool = ReadFileTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"path":"notes.txt","unexpected":true}')

    assert error_info.value.code == "invalid_arguments"


def test_read_file_rejects_workspace_runtime_files(tmp_path: Path) -> None:
    runtime_dir = tmp_path / ".mini-agent"
    runtime_dir.mkdir()
    (runtime_dir / "events.jsonl").write_text("private", encoding="utf-8")
    tool = ReadFileTool(Workspace(tmp_path, excluded_roots=(runtime_dir,)))

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"path":".mini-agent/events.jsonl"}')

    assert error_info.value.code == "invalid_path"


def test_read_file_rejects_netrc_before_nonstandard_whitespace_secret_can_egress(
    tmp_path: Path,
) -> None:
    content, secret = _netrc_with_unrecognized_whitespace_secret()
    (tmp_path / ".netrc").write_text(content, encoding="utf-8")
    tool = ReadFileTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"path":".netrc"}')

    assert error_info.value.code == "sensitive_path"
    assert str(error_info.value) == "Sensitive workspace paths cannot be read"
    assert secret not in str(error_info.value)
    assert ".netrc" not in str(error_info.value)


def test_read_file_rejects_environment_file_with_nonstandard_assignment_syntax(
    tmp_path: Path,
) -> None:
    secret = "unrecognized-" + "environment-" + "secret"
    (tmp_path / ".env.production").write_text(
        f"unusual setting : {secret}\n", encoding="utf-8"
    )
    tool = ReadFileTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"path":".env.production"}')

    assert error_info.value.code == "sensitive_path"
    assert str(error_info.value) == "Sensitive workspace paths cannot be read"
    assert secret not in str(error_info.value)


def test_read_file_rejects_sensitive_lexical_symlink_name_before_reading_target(
    tmp_path: Path,
) -> None:
    secret = "unrecognized-" + "netrc-" + "secret"
    safe_target = tmp_path / "notes.txt"
    safe_target.write_text(secret, encoding="utf-8")
    (tmp_path / ".netrc").symlink_to(safe_target)
    tool = ReadFileTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"path":".netrc"}')

    assert error_info.value.code == "sensitive_path"
    assert str(error_info.value) == "Sensitive workspace paths cannot be read"
    assert secret not in str(error_info.value)


def test_read_file_fails_closed_when_final_file_is_replaced_by_symlink_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fd open, rather than an earlier path check, rejects a symlink race."""

    requested_path = tmp_path / "notes.txt"
    requested_path.write_text("safe text\n", encoding="utf-8")
    secret = "do-not-return-this-private-content"
    replacement_target = tmp_path / "private-target.txt"
    replacement_target.write_text(secret, encoding="utf-8")
    original_open = os.open
    has_replaced = False

    def replace_final_path_before_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        *args: int,
        **kwargs: int,
    ) -> int:
        nonlocal has_replaced
        if not has_replaced and path == "notes.txt" and "dir_fd" in kwargs:
            has_replaced = True
            requested_path.unlink()
            requested_path.symlink_to(replacement_target)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(
        workspace_directory_fd.os, "open", replace_final_path_before_open
    )
    tool = ReadFileTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"path":"notes.txt"}')

    assert has_replaced is True
    assert error_info.value.code == "read_failed"
    assert str(error_info.value) == "Could not read file: notes.txt"
    assert secret not in str(error_info.value)


def test_read_file_keeps_held_parent_when_its_lexical_path_is_swapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An intermediate replacement after descent cannot redirect the final read."""

    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    (nested_directory / "notes.txt").write_text("safe held content\n", encoding="utf-8")
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    secret = "outside-private-content-must-not-be-read"
    (outside_directory / "notes.txt").write_text(secret, encoding="utf-8")
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
        if not has_swapped and path == "notes.txt" and "dir_fd" in kwargs:
            has_swapped = True
            nested_directory.rename(moved_directory)
            nested_directory.symlink_to(outside_directory, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(
        workspace_directory_fd.os, "open", swap_parent_before_leaf_open
    )
    result = ReadFileTool(Workspace(tmp_path)).execute('{"path":"nested/notes.txt"}')

    assert has_swapped is True
    assert "safe held content" in result.content
    assert secret not in result.content


def test_read_file_rejects_intermediate_symlink_swap_before_directory_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement before the next open fails instead of following outside."""

    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    (nested_directory / "notes.txt").write_text("safe content\n", encoding="utf-8")
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    secret = "outside-private-content-must-not-be-read"
    (outside_directory / "notes.txt").write_text(secret, encoding="utf-8")
    moved_directory = tmp_path / "nested-moved"
    original_open = os.open
    has_swapped = False

    def swap_parent_before_directory_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        *args: int,
        **kwargs: int,
    ) -> int:
        nonlocal has_swapped
        if not has_swapped and path == "nested" and "dir_fd" in kwargs:
            has_swapped = True
            nested_directory.rename(moved_directory)
            nested_directory.symlink_to(outside_directory, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(
        workspace_directory_fd.os, "open", swap_parent_before_directory_open
    )
    tool = ReadFileTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"path":"nested/notes.txt"}')

    assert has_swapped is True
    assert error_info.value.code == "read_failed"
    assert secret not in str(error_info.value)


def test_read_file_fails_closed_when_workspace_root_is_replaced(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "notes.txt").write_text("original content\n", encoding="utf-8")
    tool = ReadFileTool(Workspace(workspace_root))
    moved_root = tmp_path / "workspace-moved"
    workspace_root.rename(moved_root)
    workspace_root.mkdir()
    secret = "replacement-root-private-content"
    (workspace_root / "notes.txt").write_text(secret, encoding="utf-8")

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"path":"notes.txt"}')

    assert error_info.value.code == "read_failed"
    assert str(error_info.value) == "Could not read file: notes.txt"
    assert secret not in str(error_info.value)


def test_read_file_fails_closed_without_a_safe_no_follow_open_primitive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "notes.txt").write_text("safe text\n", encoding="utf-8")
    monkeypatch.delattr(private_key_material.os, "O_NOFOLLOW", raising=False)
    tool = ReadFileTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"path":"notes.txt"}')

    assert error_info.value.code == "read_failed"
    assert str(error_info.value) == "Could not read file: notes.txt"


def test_read_file_fails_closed_without_fd_relative_open_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "notes.txt").write_text("safe text\n", encoding="utf-8")
    monkeypatch.setattr(workspace_directory_fd, "_OPEN_SUPPORTS_DIR_FD", False)
    tool = ReadFileTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"path":"notes.txt"}')

    assert error_info.value.code == "read_failed"
    assert str(error_info.value) == "Could not read file: notes.txt"


def test_read_file_allows_environment_template_and_public_certificate(tmp_path: Path) -> None:
    (tmp_path / ".env.example").write_text("SETTING=example\n", encoding="utf-8")
    (tmp_path / "public.pem").write_text(
        "-----BEGIN CERTIFICATE-----\ncertificate-body\n-----END CERTIFICATE-----\n",
        encoding="utf-8",
    )
    tool = ReadFileTool(Workspace(tmp_path))

    environment_template = tool.execute('{"path":".env.example"}')
    public_certificate = tool.execute('{"path":"public.pem"}')

    assert "SETTING=example" in environment_template.content
    assert "certificate-body" in public_certificate.content


def test_read_file_truncates_a_very_large_single_line_without_unbounded_output(
    tmp_path: Path,
) -> None:
    (tmp_path / "large-line.txt").write_text("x" * 1_000_000, encoding="utf-8")
    tool = ReadFileTool(Workspace(tmp_path), max_chars=32)

    result = tool.execute('{"path":"large-line.txt"}')

    assert len(result.content) == 32
    assert result.is_truncated is True


def test_read_file_rejects_start_line_beyond_hard_limit(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("one\n", encoding="utf-8")
    tool = ReadFileTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"path":"notes.txt","start_line":100001}')

    assert error_info.value.code == "invalid_arguments"


def test_read_file_rejects_file_one_byte_past_scan_limit(tmp_path: Path) -> None:
    (tmp_path / "large-file.txt").write_text("x" * 33, encoding="utf-8")
    tool = ReadFileTool(Workspace(tmp_path), max_scan_bytes=32)

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"path":"large-file.txt"}')

    assert error_info.value.code == "read_limit_exceeded"


def test_read_file_accepts_file_exactly_at_scan_limit(tmp_path: Path) -> None:
    (tmp_path / "exact-limit.txt").write_text("x" * 32, encoding="utf-8")
    tool = ReadFileTool(Workspace(tmp_path), max_scan_bytes=32)

    result = tool.execute('{"path":"exact-limit.txt"}')

    assert result.content == f"     1 | {'x' * 32}\n"
    assert result.is_truncated is False


@pytest.mark.parametrize(
    ("kind", "arguments_json"),
    [
        ("", '{"path":"key.pem"}'),
        ("RSA", '{"path":"key.pem","start_line":1,"end_line":1}'),
        ("EC", '{"path":"key.pem","start_line":2,"end_line":2}'),
        ("OPENSSH", '{"path":"key.pem","start_line":3,"end_line":3}'),
        ("ENCRYPTED", '{"path":"key.pem","start_line":2,"end_line":2}'),
    ],
)
def test_read_file_rejects_private_key_pem_before_rendering_any_line_range(
    tmp_path: Path,
    kind: str,
    arguments_json: str,
) -> None:
    (tmp_path / "key.pem").write_text(_private_key_pem(kind=kind), encoding="utf-8")
    tool = ReadFileTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute(arguments_json)

    assert error_info.value.code == "private_key_material"
    assert str(error_info.value) == "Refusing to read private key material"
    assert "key.pem" not in str(error_info.value)
    assert "encoded-private-material" not in str(error_info.value)


def test_read_file_rejects_unterminated_private_key_pem(tmp_path: Path) -> None:
    (tmp_path / "key.pem").write_text(
        _private_key_pem(kind="DSA", include_end=False), encoding="utf-8"
    )
    tool = ReadFileTool(Workspace(tmp_path))

    with pytest.raises(ToolError) as error_info:
        tool.execute('{"path":"key.pem","start_line":2,"end_line":2}')

    assert error_info.value.code == "private_key_material"
    assert str(error_info.value) == "Refusing to read private key material"


def test_read_file_allows_certificates_public_keys_and_private_key_prose(
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
    tool = ReadFileTool(Workspace(tmp_path))

    result = tool.execute('{"path":"public-material.txt"}')

    assert "certificate-body" in result.content
    assert "public-key-body" in result.content
    assert "PRIVATE KEY but is not PEM" in result.content
