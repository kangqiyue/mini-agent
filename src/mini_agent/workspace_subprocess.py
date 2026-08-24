"""Launch trusted subprocesses after changing to a held workspace directory fd."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


class WorkspaceSubprocessLauncherError(RuntimeError):
    """A safe descriptor-anchored subprocess launcher is unavailable."""


DIRECTORY_FD_BOOTSTRAP = """import os
import sys

descriptor = int(sys.argv[1])
executable = sys.argv[2]
argv = sys.argv[2:]
try:
    os.fchdir(descriptor)
    os.close(descriptor)
    os.execve(executable, argv, os.environ)
except OSError:
    os._exit(127)
"""


class WorkspaceSubprocessLauncher:
    """Start a process only after its child changes directory by trusted fd.

    ``subprocess.Popen`` has no file-descriptor ``cwd`` parameter. On macOS,
    ``/dev/fd/<n>`` cannot reliably be used as a directory cwd, while Linux's
    ``/proc/self/fd`` is not portable. The active trusted Python interpreter
    instead executes a fixed isolated bootstrap. The bootstrap receives a
    directory descriptor as a separate argv item, calls ``fchdir``, closes the
    descriptor, and immediately executes the requested executable.
    """

    def __init__(self, workspace_root: Path) -> None:
        if not hasattr(os, "fchdir"):
            raise WorkspaceSubprocessLauncherError(
                "Safe workspace subprocess launching is unavailable"
            )
        self._launcher_executable = _trusted_launcher_executable(workspace_root)

    def start(
        self,
        *,
        executable: str,
        argv: Sequence[str],
        directory_descriptor: int,
        environment: dict[str, str],
        stdin: int | None,
        stdout: int | None,
        stderr: int | None,
        start_new_session: bool = False,
    ) -> subprocess.Popen[bytes]:
        """Run ``argv`` after a no-pathname cwd change in the child process."""

        if not argv or argv[0] != executable:
            raise ValueError("Workspace subprocess argv must begin with its executable")
        return subprocess.Popen(
            [
                self._launcher_executable,
                "-I",
                "-P",
                "-c",
                DIRECTORY_FD_BOOTSTRAP,
                str(directory_descriptor),
                *argv,
            ],
            shell=False,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            start_new_session=start_new_session,
            close_fds=True,
            pass_fds=(directory_descriptor,),
            env=environment,
        )


def _trusted_launcher_executable(workspace_root: Path) -> str:
    try:
        executable = Path(sys.executable).resolve(strict=True)
        mode = executable.stat().st_mode
    except OSError as error:
        raise WorkspaceSubprocessLauncherError(
            "Safe workspace subprocess launching is unavailable"
        ) from error
    if (
        executable.is_relative_to(workspace_root)
        or not stat.S_ISREG(mode)
        or not os.access(executable, os.X_OK)
    ):
        raise WorkspaceSubprocessLauncherError(
            "Safe workspace subprocess launching is unavailable"
        )
    return str(executable)
