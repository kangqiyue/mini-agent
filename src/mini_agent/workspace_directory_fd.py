"""Descriptor-anchored workspace path traversal for POSIX local tools.

Path validation can nominate a workspace object, but it cannot bind that
object: an attacker with write access to a workspace can rename an
intermediate directory after validation and before a later pathname open.
This module keeps each directory descriptor open while it descends, always
uses ``O_NOFOLLOW`` for the next component, and exposes the held parent for
operations that must remain bound to the same directory object.
"""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd


class WorkspaceDirectoryFdError(OSError):
    """A workspace object cannot be reached through safe directory descriptors."""


@dataclass(frozen=True)
class DirectoryIdentity:
    """The stable identity of one opened directory."""

    device: int
    inode: int

    @classmethod
    def from_stat_result(cls, result: os.stat_result) -> DirectoryIdentity:
        return cls(device=result.st_dev, inode=result.st_ino)


@dataclass
class OpenWorkspaceParent:
    """A held root and parent directory for one existing relative leaf path."""

    root_descriptor: int
    parent_descriptor: int
    parent_parts: tuple[str, ...]
    parent_identity: DirectoryIdentity
    leaf_name: str

    def close(self) -> None:
        """Close owned descriptors exactly once."""

        try:
            os.close(self.parent_descriptor)
        finally:
            os.close(self.root_descriptor)


class WorkspaceDirectoryAnchor:
    """Open descendants of one workspace root without pathname re-traversal."""

    def __init__(self, root: Path) -> None:
        self._root = root
        try:
            root_status = root.stat()
        except OSError as error:
            raise WorkspaceDirectoryFdError(
                errno.ENOENT, "Workspace directory is unavailable"
            ) from error
        if not stat.S_ISDIR(root_status.st_mode):
            raise WorkspaceDirectoryFdError(
                errno.ENOTDIR, "Workspace directory is unavailable"
            )
        self._root_identity = DirectoryIdentity.from_stat_result(root_status)

    @property
    def root_identity(self) -> DirectoryIdentity:
        return self._root_identity

    @contextmanager
    def open_existing_parent(
        self, relative_path: str
    ) -> Generator[OpenWorkspaceParent, None, None]:
        """Hold the parent directory of one existing relative leaf path.

        The returned root descriptor remains open too, so callers can re-walk
        the parent route and compare identities immediately before a mutation.
        """

        parts = _relative_parts(relative_path, allow_root=False)
        root_descriptor = -1
        parent_descriptor = -1
        try:
            root_descriptor = self._open_verified_root()
            parent_descriptor = self._open_parent_from_root(
                root_descriptor, parts[:-1]
            )
            opened_parent = OpenWorkspaceParent(
                root_descriptor=root_descriptor,
                parent_descriptor=parent_descriptor,
                parent_parts=parts[:-1],
                parent_identity=DirectoryIdentity.from_stat_result(
                    os.fstat(parent_descriptor)
                ),
                leaf_name=parts[-1],
            )
        except BaseException:
            if parent_descriptor >= 0:
                os.close(parent_descriptor)
            if root_descriptor >= 0:
                os.close(root_descriptor)
            raise
        try:
            yield opened_parent
        finally:
            opened_parent.close()

    @contextmanager
    def open_existing_regular_file(
        self, relative_path: str
    ) -> Generator[int, None, None]:
        """Open one existing regular file through held no-follow directory fds."""

        with self.open_existing_parent(relative_path) as opened_parent:
            file_descriptor = -1
            try:
                file_descriptor = open_regular_at(
                    opened_parent.parent_descriptor, opened_parent.leaf_name
                )
                yield file_descriptor
            finally:
                if file_descriptor >= 0:
                    os.close(file_descriptor)

    @contextmanager
    def open_existing_directory(
        self, relative_path: str
    ) -> Generator[int, None, None]:
        """Open one existing workspace directory through held no-follow fds."""

        parts = _relative_parts(relative_path, allow_root=True)
        root_descriptor = -1
        directory_descriptor = -1
        try:
            root_descriptor = self._open_verified_root()
            directory_descriptor = self._open_parent_from_root(root_descriptor, parts)
        except BaseException:
            if directory_descriptor >= 0:
                os.close(directory_descriptor)
            if root_descriptor >= 0:
                os.close(root_descriptor)
            raise
        try:
            yield directory_descriptor
        finally:
            if directory_descriptor >= 0:
                os.close(directory_descriptor)
            if root_descriptor >= 0:
                os.close(root_descriptor)

    def reopen_parent(self, opened_parent: OpenWorkspaceParent) -> int:
        """Re-walk a held route from its root and require the same parent inode.

        The caller owns the returned descriptor and must close it.
        """

        current_parent = self._open_parent_from_root(
            opened_parent.root_descriptor, opened_parent.parent_parts
        )
        try:
            current_identity = DirectoryIdentity.from_stat_result(
                os.fstat(current_parent)
            )
            if current_identity != opened_parent.parent_identity:
                raise WorkspaceDirectoryFdError(
                    errno.ESTALE, "Workspace parent directory changed"
                )
            return current_parent
        except BaseException:
            os.close(current_parent)
            raise

    def _open_verified_root(self) -> int:
        root_descriptor = _open_directory(self._root)
        try:
            identity = DirectoryIdentity.from_stat_result(os.fstat(root_descriptor))
            if identity != self._root_identity:
                raise WorkspaceDirectoryFdError(
                    errno.ESTALE, "Workspace directory changed"
                )
            return root_descriptor
        except BaseException:
            os.close(root_descriptor)
            raise

    @staticmethod
    def _open_parent_from_root(
        root_descriptor: int, components: tuple[str, ...]
    ) -> int:
        current_descriptor = os.dup(root_descriptor)
        try:
            for component in components:
                next_descriptor = _open_directory(component, dir_fd=current_descriptor)
                os.close(current_descriptor)
                current_descriptor = next_descriptor
            return current_descriptor
        except BaseException:
            os.close(current_descriptor)
            raise


def open_regular_at(parent_descriptor: int, leaf_name: str) -> int:
    """Open one direct regular-file child without following a symlink."""

    _require_leaf_name(leaf_name)
    _require_open_dir_fd_support()
    try:
        file_descriptor = os.open(
            leaf_name, _file_open_flags(), dir_fd=parent_descriptor
        )
    except (NotImplementedError, TypeError) as error:
        raise _unsupported_directory_fd_error() from error
    try:
        if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
            raise WorkspaceDirectoryFdError(
                errno.EINVAL, "Workspace path is not a regular file"
            )
        return file_descriptor
    except BaseException:
        os.close(file_descriptor)
        raise


def _open_directory(path: str | Path, *, dir_fd: int | None = None) -> int:
    if dir_fd is not None:
        _require_open_dir_fd_support()
    try:
        if dir_fd is None:
            directory_descriptor = os.open(path, _directory_open_flags())
        else:
            directory_descriptor = os.open(
                path, _directory_open_flags(), dir_fd=dir_fd
            )
    except (NotImplementedError, TypeError) as error:
        raise _unsupported_directory_fd_error() from error
    try:
        if not stat.S_ISDIR(os.fstat(directory_descriptor).st_mode):
            raise WorkspaceDirectoryFdError(
                errno.ENOTDIR, "Workspace path is not a directory"
            )
        return directory_descriptor
    except BaseException:
        os.close(directory_descriptor)
        raise


def _relative_parts(relative_path: str, *, allow_root: bool) -> tuple[str, ...]:
    """Normalize dot components without ever traversing above the held root."""

    if not relative_path or "\0" in relative_path:
        raise WorkspaceDirectoryFdError(errno.EINVAL, "Workspace path is invalid")
    path = Path(relative_path)
    if path.is_absolute():
        raise WorkspaceDirectoryFdError(errno.EINVAL, "Workspace path is invalid")
    parts: list[str] = []
    for component in path.parts:
        if component == ".":
            continue
        if component == "..":
            if not parts:
                raise WorkspaceDirectoryFdError(
                    errno.EPERM, "Workspace path escapes its root"
                )
            parts.pop()
            continue
        parts.append(component)
    if not parts and not allow_root:
        raise WorkspaceDirectoryFdError(errno.EINVAL, "Workspace path is invalid")
    return tuple(parts)


def _require_leaf_name(value: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\0" in value:
        raise WorkspaceDirectoryFdError(errno.EINVAL, "Workspace leaf name is invalid")


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | _required_os_flag("O_DIRECTORY")
        | _required_os_flag("O_NOFOLLOW")
        | _close_on_exec_flag()
    )


def _file_open_flags() -> int:
    # A regular file can be replaced by a FIFO after path validation. Open
    # without waiting for a writer so fstat can reject non-regular objects.
    return (
        os.O_RDONLY
        | _required_os_flag("O_NOFOLLOW")
        | _required_os_flag("O_NONBLOCK")
        | _close_on_exec_flag()
    )


def _close_on_exec_flag() -> int:
    return getattr(os, "O_CLOEXEC", 0)


def _required_os_flag(name: str) -> int:
    value = getattr(os, name, None)
    if value is None:
        raise _unsupported_directory_fd_error()
    return value


def _require_open_dir_fd_support() -> None:
    if not _OPEN_SUPPORTS_DIR_FD:
        raise _unsupported_directory_fd_error()


def _unsupported_directory_fd_error() -> WorkspaceDirectoryFdError:
    return WorkspaceDirectoryFdError(
        errno.ENOTSUP, "Safe workspace directory descriptors are unavailable"
    )
