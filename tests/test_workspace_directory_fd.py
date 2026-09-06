import os
import subprocess
import sys
from pathlib import Path


def test_regular_file_open_rejects_fifo_without_waiting_for_a_writer(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "pipe")
    # A subprocess timeout keeps a regression from hanging the test runner.
    code = """
import sys
from pathlib import Path
from mini_agent.workspace_directory_fd import WorkspaceDirectoryAnchor, WorkspaceDirectoryFdError

anchor = WorkspaceDirectoryAnchor(Path(sys.argv[1]))
try:
    with anchor.open_existing_regular_file("pipe"):
        raise AssertionError("FIFO accepted as a regular file")
except WorkspaceDirectoryFdError:
    pass
"""
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        check=True,
        capture_output=True,
        timeout=5,
    )
