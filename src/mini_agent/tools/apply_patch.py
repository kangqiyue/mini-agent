"""Atomic, expected-content file replacements inside a workspace."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mini_agent.redaction import redact_text
from mini_agent.tool_facts import ToolCompletionFacts
from mini_agent.tools.base import ToolDefinition, ToolError, ToolResult
from mini_agent.workspace import (
    SensitiveWorkspacePathError,
    Workspace,
    WorkspacePathError,
)
from mini_agent.workspace_directory_fd import (
    OpenWorkspaceParent,
    WorkspaceDirectoryFdError,
    open_regular_at,
)

# Keep direct tool execution within the same hard byte budget as approval input.
MAX_PATCH_CONTENT_BYTES = 256 * 1024


class FileReplacement(BaseModel):
    """A full-file replacement guarded by its current content, or a creation.

    ``expected_content`` omitted (null) selects creation: the target must not
    exist and ``replacement_content`` becomes the new file's content.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    expected_content: str | None = None
    replacement_content: str

    @property
    def creates_file(self) -> bool:
        return self.expected_content is None


class ApplyPatchArguments(BaseModel):
    """Validated inputs for one atomic, per-file change."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    changes: tuple[FileReplacement, ...] = Field(min_length=1, max_length=1)


@dataclass(frozen=True)
class _FileIdentity:
    """Stable attributes used to detect a target changed before replacement."""

    device: int
    inode: int
    modified_at_ns: int
    size: int
    mode: int

    @classmethod
    def from_stat_result(cls, result: os.stat_result) -> _FileIdentity:
        return cls(
            device=result.st_dev,
            inode=result.st_ino,
            modified_at_ns=result.st_mtime_ns,
            size=result.st_size,
            mode=stat.S_IMODE(result.st_mode),
        )


@dataclass
class _OpenedWorkspaceTarget:
    """Verified target and its parent directory held open by file descriptor."""

    parent_context: AbstractContextManager[OpenWorkspaceParent]
    parent: OpenWorkspaceParent
    target_descriptor: int
    target_identity: _FileIdentity
    relative_path: str

    def close(self) -> None:
        try:
            os.close(self.target_descriptor)
        finally:
            self.parent_context.__exit__(None, None, None)


@dataclass(frozen=True)
class _ValidatedReplacement:
    relative_path: str
    expected_content: str
    replacement_content: str
    original_identity: _FileIdentity
    opened_target: _OpenedWorkspaceTarget


@dataclass
class _ValidatedCreation:
    """A new-file target whose parent directory is held open by descriptor."""

    relative_path: str
    replacement_content: str
    parent_context: AbstractContextManager[OpenWorkspaceParent]
    parent: OpenWorkspaceParent

    def close(self) -> None:
        self.parent_context.__exit__(None, None, None)


class ApplyPatchTool:
    """Replace existing workspace files, or create new ones, atomically."""

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace
        self._directory_anchor = workspace.directory_anchor

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="apply_patch",
            description=(
                "Atomically change one workspace file. Omit expected_content to "
                "create a new file (it must not already exist); provide it to "
                "replace an existing file when its current content matches "
                "expected_content exactly."
            ),
            parameters=ApplyPatchArguments.model_json_schema(),
            is_read_only=False,
        )

    def execute(self, arguments_json: str) -> ToolResult:
        replacement: _ValidatedReplacement | None = None
        creation: _ValidatedCreation | None = None
        try:
            self.preflight(arguments_json)
            arguments = self._parse_preflight_arguments(arguments_json)
            change = arguments.changes[0]
            if change.creates_file:
                creation = self._validate_creation(change)
            else:
                replacement = self._validate_all_changes(arguments.changes)[0]
        except ValidationError as error:
            raise ToolError("invalid_arguments", "Invalid apply_patch arguments") from error
        except SensitiveWorkspacePathError as error:
            raise ToolError(
                "sensitive_path", "Sensitive workspace paths cannot be modified"
            ) from error
        except (OSError, WorkspacePathError) as error:
            raise self._invalid_path_error() from error

        changed_path: str
        if creation is not None:
            try:
                self._write_creation(creation)
            finally:
                creation.close()
            changed_path = creation.relative_path
        else:
            assert replacement is not None
            try:
                self._write_replacement(replacement)
            finally:
                replacement.opened_target.close()
            changed_path = replacement.relative_path
        return ToolResult(
            content=f"Modified files:\n- {changed_path}",
            facts=ToolCompletionFacts(modified_paths=(changed_path,)),
        )

    def preflight(self, arguments_json: str) -> None:
        """Reject unsafe change payloads before the agent starts the tool.

        This method has no write side effects.  It reads workspace metadata to
        reject paths that cannot safely identify their target: an existing
        regular file for replacement, or an existing parent directory with an
        absent leaf for creation.  The agent calls it after recording the
        requested tool call but before approval or ``tool_started``; direct
        callers receive the same protection through :meth:`execute`.
        """

        try:
            arguments = self._parse_preflight_arguments(arguments_json)
        except ValidationError as error:
            raise ToolError("invalid_arguments", "Invalid apply_patch arguments") from error
        change = arguments.changes[0]
        self._require_bounded_patch_content(change)
        self._reject_sensitive_replacement_content(change)
        try:
            opened = (
                self._open_create_target(change.path)
                if change.creates_file
                else None
            )
            if opened is None:
                self._validate_path_eligibility(change.path)
        except SensitiveWorkspacePathError as error:
            raise ToolError(
                "sensitive_path", "Sensitive workspace paths cannot be modified"
            ) from error
        except (FileNotFoundError, WorkspacePathError, OSError) as error:
            raise self._invalid_path_error() from error
        if opened is not None:
            opened.close()

    @staticmethod
    def _invalid_path_error() -> ToolError:
        """Return a stable path failure without echoing a caller-supplied path."""

        return ToolError(
            "invalid_path",
            "apply_patch requires a regular workspace file or a new file "
            "in an existing workspace directory",
        )

    def _validate_path_eligibility(self, relative_path: str) -> None:
        """Read-check a target without treating this preflight as a lock."""

        self._reject_parent_path(relative_path)
        opened_target = self._open_existing_target(relative_path)
        opened_target.close()

    def session_grant_scope(self, arguments_json: str) -> str | None:
        """Return a stable scope for one non-symlink canonical workspace file.

        Creation never mints a session grant: an approved create must not
        authorize a later replacement at the same path, so each creation is
        approved on its own.
        """

        try:
            self.preflight(arguments_json)
            arguments = self._parse_preflight_arguments(arguments_json)
            change = arguments.changes[0]
            if change.creates_file:
                return None
            self._reject_parent_path(change.path)
            if self._workspace.has_symlink_component(change.path):
                return None
            target_path = self._workspace.resolve_existing(change.path)
        except (ToolError, ValidationError, FileNotFoundError, WorkspacePathError):
            return None
        relative_target = self._workspace.relative(target_path)
        scope_identity = f"{self._workspace.root}\0{relative_target}"
        target_digest = hashlib.sha256(scope_identity.encode("utf-8")).hexdigest()
        return f"apply_patch:workspace-path:v1:{target_digest}"

    def _parse_preflight_arguments(self, arguments_json: str) -> ApplyPatchArguments:
        return ApplyPatchArguments.model_validate_json(arguments_json)

    def _validate_all_changes(
        self, changes: tuple[FileReplacement, ...]
    ) -> tuple[_ValidatedReplacement, ...]:
        change = changes[0]
        assert change.expected_content is not None
        self._reject_parent_path(change.path)
        opened_target = self._open_existing_target(change.path)
        original_identity = opened_target.target_identity
        try:
            self._require_bounded_target(original_identity)
            self._require_bounded_patch_content(change)
            current_content = self._read_current_content(opened_target.target_descriptor)
            confirmed_identity = self._file_identity_from_descriptor(
                opened_target.target_descriptor
            )
            if confirmed_identity != original_identity:
                raise ToolError("target_changed", "Target changed while being validated")
            if current_content != change.expected_content:
                raise ToolError("expected_content_mismatch", "Current content does not match")
            return (
                _ValidatedReplacement(
                    relative_path=opened_target.relative_path,
                    expected_content=change.expected_content,
                    replacement_content=change.replacement_content,
                    original_identity=original_identity,
                    opened_target=opened_target,
                ),
            )
        except BaseException:
            opened_target.close()
            raise

    def _reject_parent_path(self, relative_path: str) -> None:
        if ".." in Path(relative_path).parts:
            raise WorkspacePathError("Workspace paths cannot contain '..'")

    def _open_existing_target(self, relative_path: str) -> _OpenedWorkspaceTarget:
        """Open one existing target through no-follow directory descriptors.

        The path checks here provide user-facing policy errors, while the
        descriptor descent is the write boundary: every directory component
        and the final target is opened relative to a descriptor that remains
        open until the replacement completes.
        """

        self._reject_parent_path(relative_path)
        if self._workspace.has_symlink_component(relative_path):
            raise WorkspacePathError("apply_patch does not allow symlink paths")
        # Keep the workspace's excluded-root and runtime-sensitive-path policy
        # as an early user-facing guard. Descriptor traversal below is still
        # the authoritative boundary for every later read and write.
        self._workspace.resolve_existing(relative_path)
        path_parts = tuple(part for part in Path(relative_path).parts if part != ".")
        if not path_parts:
            raise WorkspacePathError("apply_patch requires a file path")
        parent_context = self._directory_anchor.open_existing_parent(relative_path)
        opened_parent: OpenWorkspaceParent | None = None
        target_descriptor = -1
        try:
            opened_parent = parent_context.__enter__()
            target_descriptor = open_regular_at(
                opened_parent.parent_descriptor, opened_parent.leaf_name
            )
            return _OpenedWorkspaceTarget(
                parent_context=parent_context,
                parent=opened_parent,
                target_descriptor=target_descriptor,
                target_identity=self._file_identity_from_descriptor(target_descriptor),
                relative_path=Path(*path_parts).as_posix(),
            )
        except BaseException as error:
            if target_descriptor >= 0:
                os.close(target_descriptor)
            if opened_parent is not None:
                parent_context.__exit__(None, None, None)
            if isinstance(error, (WorkspaceDirectoryFdError, OSError)):
                raise WorkspacePathError(
                    "apply_patch target cannot be opened safely"
                ) from error
            raise

    def _verify_parent_route_is_current(self, opened_target: _OpenedWorkspaceTarget) -> None:
        """Ensure the original root still reaches the held parent directory."""

        current_parent_descriptor = -1
        try:
            current_parent_descriptor = self._directory_anchor.reopen_parent(
                opened_target.parent
            )
        except (WorkspaceDirectoryFdError, OSError) as error:
            raise ToolError("target_changed", "Target parent changed before replacement") from error
        finally:
            if current_parent_descriptor >= 0:
                os.close(current_parent_descriptor)

    def _open_target_at_parent(self, opened_target: _OpenedWorkspaceTarget) -> int:
        try:
            return open_regular_at(
                opened_target.parent.parent_descriptor, opened_target.parent.leaf_name
            )
        except (WorkspaceDirectoryFdError, OSError) as error:
            raise ToolError("target_changed", "Target changed before replacement") from error

    def _read_current_content(self, file_descriptor: int) -> str:
        try:
            os.lseek(file_descriptor, 0, os.SEEK_SET)
            encoded_content = _read_descriptor_bounded(
                file_descriptor, maximum_bytes=MAX_PATCH_CONTENT_BYTES
            )
            return encoded_content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ToolError("invalid_encoding", "File is not valid UTF-8") from error
        except OSError as error:
            raise ToolError("read_failed", "Could not read target file") from error

    def _require_bounded_target(self, identity: _FileIdentity) -> None:
        if identity.size > MAX_PATCH_CONTENT_BYTES:
            raise ToolError("file_too_large", "Target exceeds the patch byte limit")

    def _require_bounded_patch_content(self, change: FileReplacement) -> None:
        expected = change.expected_content
        fields: tuple[tuple[str, str], ...] = (
            ("replacement_content", change.replacement_content),
        )
        if expected is not None:
            fields = (("expected_content", expected), *fields)
        for field_name, content in fields:
            try:
                content_size = len(content.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise ToolError(
                    "invalid_encoding", f"{field_name} is not valid UTF-8"
                ) from error
            if content_size > MAX_PATCH_CONTENT_BYTES:
                raise ToolError(
                    "content_too_large",
                    "Patch content exceeds the byte limit",
                )

    def _reject_sensitive_replacement_content(self, change: FileReplacement) -> None:
        """Keep credentials and redaction placeholders out of workspace writes.

        ``expected_content`` is intentionally excluded: matching it is needed
        to remove or rotate a secret already present in a file.  A replacement,
        by contrast, would persist new credential-like material, so reject it
        before any temporary file or target mutation can occur.
        """

        redaction = redact_text(change.replacement_content)
        has_redaction_marker = (
            "[REDACTED]" in change.replacement_content
            or "[REDACTED_KEY_" in change.replacement_content
        )
        if redaction.match_count or has_redaction_marker:
            raise ToolError(
                "sensitive_replacement_content",
                "replacement_content contains credential-shaped content or a "
                "redaction marker; refusing to write it",
            )

    @staticmethod
    def _file_identity_from_descriptor(file_descriptor: int) -> _FileIdentity:
        try:
            return _FileIdentity.from_stat_result(os.fstat(file_descriptor))
        except OSError as error:
            raise ToolError("read_failed", "Could not inspect target file") from error

    def _verify_replacement_is_current(self, replacement: _ValidatedReplacement) -> None:
        """Fail closed if the fd-relative target route changed before replace."""

        opened_target = replacement.opened_target
        self._verify_parent_route_is_current(opened_target)
        current_descriptor = self._open_target_at_parent(opened_target)
        try:
            current_identity = self._file_identity_from_descriptor(current_descriptor)
            self._require_bounded_target(current_identity)
            current_content = self._read_current_content(current_descriptor)
            confirmed_identity = self._file_identity_from_descriptor(current_descriptor)
        finally:
            os.close(current_descriptor)
        if (
            current_identity != replacement.original_identity
            or confirmed_identity != current_identity
            or current_content != replacement.expected_content
        ):
            raise ToolError("target_changed", "Target changed before replacement")

    def _validate_creation(self, change: FileReplacement) -> _ValidatedCreation:
        """Validate one new-file change while holding its parent directory."""

        creation = self._open_create_target(change.path)
        try:
            self._require_bounded_patch_content(change)
            self._reject_sensitive_replacement_content(change)
            creation.replacement_content = change.replacement_content
        except BaseException:
            creation.close()
            raise
        return creation

    def _open_create_target(self, relative_path: str) -> _ValidatedCreation:
        """Hold the target's parent open and require the leaf to be absent.

        The descriptor-held parent is the write boundary; the absence check is
        a friendly early error.  The later ``os.link`` with ``O_EXCL``
        semantics is the authoritative no-clobber decision.
        """

        self._reject_parent_path(relative_path)
        self._workspace.resolve_creatable_file_path(relative_path)
        path_parts = tuple(part for part in Path(relative_path).parts if part != ".")
        if not path_parts:
            raise WorkspacePathError("apply_patch requires a file path")
        parent_context = self._directory_anchor.open_existing_parent(relative_path)
        opened_parent: OpenWorkspaceParent | None = None
        try:
            opened_parent = parent_context.__enter__()
            try:
                os.stat(
                    opened_parent.leaf_name,
                    dir_fd=opened_parent.parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            except OSError as error:
                raise WorkspacePathError(
                    "apply_patch target cannot be inspected safely"
                ) from error
            else:
                raise ToolError("target_exists", "Target already exists")
            return _ValidatedCreation(
                relative_path=Path(*path_parts).as_posix(),
                replacement_content="",
                parent_context=parent_context,
                parent=opened_parent,
            )
        except BaseException:
            if opened_parent is not None:
                parent_context.__exit__(None, None, None)
            raise

    def _verify_create_route_is_current(self, creation: _ValidatedCreation) -> None:
        """Ensure the original root still reaches the held parent directory."""

        current_parent_descriptor = -1
        try:
            current_parent_descriptor = self._directory_anchor.reopen_parent(
                creation.parent
            )
        except (WorkspaceDirectoryFdError, OSError) as error:
            raise ToolError("target_changed", "Target parent changed before creation") from error
        finally:
            if current_parent_descriptor >= 0:
                os.close(current_parent_descriptor)

    def _write_creation(self, creation: _ValidatedCreation) -> None:
        """Create the new file atomically without ever clobbering a neighbor.

        The full content is written and fsynced into a temporary file, which is
        then hard-linked onto the target name.  ``os.link`` fails atomically
        when a file of that name appeared meanwhile, so the write can never
        overwrite an unanticipated file, and a successful link guarantees the
        target holds complete, durable content.
        """

        temporary_name: str | None = None
        is_creation_uncertain = False
        try:
            temporary_name, temporary_descriptor = create_replacement_temporary_file(
                creation.parent.parent_descriptor,
                creation.parent.leaf_name,
            )
            try:
                _write_all(
                    temporary_descriptor, creation.replacement_content.encode("utf-8")
                )
                os.fchmod(temporary_descriptor, 0o644)
                os.fsync(temporary_descriptor)
            finally:
                os.close(temporary_descriptor)

            self._verify_create_route_is_current(creation)
            try:
                # Once linking starts, an exception may arrive after the target
                # was published. Only an explicit collision proves no creation.
                is_creation_uncertain = True
                os.link(
                    temporary_name,
                    creation.parent.leaf_name,
                    src_dir_fd=creation.parent.parent_descriptor,
                    dst_dir_fd=creation.parent.parent_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError as error:
                # The link was refused atomically; the workspace is untouched.
                is_creation_uncertain = False
                raise ToolError(
                    "target_exists", "Target already exists"
                ) from error
            # The target name now exists; the temporary link below is dropped
            # in the finally block so exactly one name remains.
            self._sync_directory(creation.parent.parent_descriptor)
        except BaseException as error:
            if is_creation_uncertain:
                raise ToolError(
                    "write_status_unknown",
                    "File creation may have succeeded, but its durable status "
                    "could not be confirmed",
                ) from error
            if isinstance(error, ToolError):
                raise
            if isinstance(error, OSError):
                raise ToolError("write_failed", "Could not create target file") from error
            raise
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(
                        temporary_name,
                        dir_fd=creation.parent.parent_descriptor,
                    )
                except OSError as error:
                    if is_creation_uncertain:
                        raise ToolError(
                            "write_status_unknown",
                            "File creation may have succeeded, but its durable "
                            "status could not be confirmed",
                        ) from error
                    raise ToolError(
                        "write_failed", "Could not create target file"
                    ) from error

    def _write_replacement(self, replacement: _ValidatedReplacement) -> None:
        temporary_name: str | None = None
        replacement_outcome_uncertain = False
        try:
            temporary_name, temporary_descriptor = create_replacement_temporary_file(
                replacement.opened_target.parent.parent_descriptor,
                replacement.opened_target.parent.leaf_name,
            )
            try:
                _write_all(temporary_descriptor, replacement.replacement_content.encode("utf-8"))
                os.fchmod(temporary_descriptor, replacement.original_identity.mode)
                os.fsync(temporary_descriptor)
            finally:
                os.close(temporary_descriptor)

            self._verify_replacement_is_current(replacement)
            # A wrapper can perform the replacement and then report an error.  Once
            # this call begins, an exception cannot prove that the old file remains.
            replacement_outcome_uncertain = True
            os.replace(
                temporary_name,
                replacement.opened_target.parent.leaf_name,
                src_dir_fd=replacement.opened_target.parent.parent_descriptor,
                dst_dir_fd=replacement.opened_target.parent.parent_descriptor,
            )
            temporary_name = None
            self._sync_directory(replacement.opened_target.parent.parent_descriptor)
        except BaseException as error:
            if replacement_outcome_uncertain:
                raise ToolError(
                    "write_status_unknown",
                    "File replacement may have succeeded, but its durable status "
                    "could not be confirmed",
                ) from error
            if isinstance(error, ToolError):
                raise
            if isinstance(error, OSError):
                raise ToolError("write_failed", "Could not replace target file") from error
            raise
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(
                        temporary_name,
                        dir_fd=replacement.opened_target.parent.parent_descriptor,
                    )
                except BaseException as error:
                    if replacement_outcome_uncertain:
                        raise ToolError(
                            "write_status_unknown",
                            "File replacement may have succeeded, but its durable "
                            "status could not be confirmed",
                        ) from error
                    if isinstance(error, OSError):
                        raise ToolError("write_failed", "Could not replace target file") from error
                    raise

    @staticmethod
    def _sync_directory(directory_descriptor: int) -> None:
        os.fsync(directory_descriptor)


def _temporary_no_follow_flag() -> int:
    value = getattr(os, "O_NOFOLLOW", None)
    if value is None:
        raise WorkspaceDirectoryFdError(
            "Safe workspace directory descriptors are unavailable"
        )
    return value


def _read_descriptor_bounded(file_descriptor: int, *, maximum_bytes: int) -> bytes:
    output = bytearray()
    while len(output) <= maximum_bytes:
        chunk = os.read(file_descriptor, min(8_192, maximum_bytes - len(output) + 1))
        if not chunk:
            return bytes(output)
        output.extend(chunk)
    raise ToolError("file_too_large", "Target exceeds the patch byte limit")


def _write_all(file_descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written_count = os.write(file_descriptor, view)
        if written_count < 1:
            raise OSError("Could not write temporary replacement")
        view = view[written_count:]


def create_replacement_temporary_file(
    parent_descriptor: int, target_name: str
) -> tuple[str, int]:
    """Create a replacement temp file inside an already-open parent directory."""

    prefix = f".{target_name}."
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _temporary_no_follow_flag()
    for _ in range(128):
        name = f"{prefix}{secrets.token_hex(16)}.tmp"
        try:
            return name, os.open(name, flags, 0o600, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
    raise OSError("Could not allocate a unique temporary file")
