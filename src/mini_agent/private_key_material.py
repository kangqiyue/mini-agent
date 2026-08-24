"""Bounded classification for PEM private-key material in workspace files."""

from __future__ import annotations

import errno
import os
import re
import stat
from pathlib import Path

# Match PEM *boundaries*, rather than ordinary prose containing the words
# "private key". Callers must classify the complete bounded file before they
# return any portion of it, otherwise a selected line range can expose only a
# key body or closing delimiter.
PRIVATE_KEY_PEM_BOUNDARY_PATTERN = re.compile(
    rb"(?im)^[ \t]*-----BEGIN[ \t]+"
    rb"(?:(?:[A-Z0-9_-]+[ \t]+)*)PRIVATE[ \t]+KEY"
    rb"(?:[ \t]+BLOCK)?-----[ \t]*\r?$"
)


class FileScanLimitExceededError(RuntimeError):
    """A file could not be completely classified within its byte budget."""


def read_bounded_file_bytes(path: Path, *, max_bytes: int) -> bytes:
    """Read one regular, non-symlink file without reopening its pathname.

    A caller may have validated ``path`` before this function runs. That is not
    a property of the path: its final component can be replaced by a symlink
    between validation and opening. ``O_NOFOLLOW`` rejects that replacement,
    and every check and read below uses the same descriptor.
    """

    if max_bytes < 1:
        raise ValueError("File scan byte limit must be positive")

    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        # Opening and checking afterwards would follow a terminal symlink, so
        # unsupported platforms must fail closed rather than reopen the path.
        raise OSError(errno.ENOTSUP, "Safe non-symlink file opening is unavailable")

    flags = os.O_RDONLY | no_follow
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC

    file_descriptor = os.open(path, flags)
    try:
        return read_bounded_open_file_bytes(file_descriptor, max_bytes=max_bytes)
    finally:
        os.close(file_descriptor)


def read_bounded_open_file_bytes(file_descriptor: int, *, max_bytes: int) -> bytes:
    """Classify bytes from one already-open regular file descriptor.

    The caller retains ownership of ``file_descriptor``. Reading from this
    descriptor, rather than reopening a validated pathname, keeps a private
    key scan bound to the same filesystem object selected by a dirfd walk.
    """

    if max_bytes < 1:
        raise ValueError("File scan byte limit must be positive")
    file_status = os.fstat(file_descriptor)
    if not stat.S_ISREG(file_status.st_mode):
        raise OSError(errno.EINVAL, "Expected a regular file")
    if file_status.st_size > max_bytes:
        raise FileScanLimitExceededError("File exceeds the private-key scan byte limit")
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    file_bytes = _read_at_most(file_descriptor, max_bytes + 1)
    if len(file_bytes) > max_bytes:
        raise FileScanLimitExceededError("Private-key scan exceeded the byte limit")
    return file_bytes


def _read_at_most(file_descriptor: int, maximum_bytes: int) -> bytes:
    """Read no more than ``maximum_bytes`` from one validated descriptor."""

    chunks: list[bytes] = []
    remaining_bytes = maximum_bytes
    while remaining_bytes:
        chunk = os.read(file_descriptor, remaining_bytes)
        if not chunk:
            break
        chunks.append(chunk)
        remaining_bytes -= len(chunk)
    return b"".join(chunks)


def contains_private_key_pem(file_bytes: bytes) -> bool:
    """Return whether complete file bytes contain a PEM private-key boundary."""

    return PRIVATE_KEY_PEM_BOUNDARY_PATTERN.search(file_bytes) is not None
