"""Exercise the installed ``mini-agent`` entry point through a real PTY."""

from __future__ import annotations

import os
import pty
import select
import shutil
import time
from pathlib import Path

ENTRYPOINT = shutil.which("mini-agent")
CACHE = Path(__file__).resolve().parent / "cache"


def _run_chat(workspace: Path, config: Path) -> str:
    master_fd, slave_fd = pty.openpty()
    os.set_blocking(master_fd, False)
    process_id = os.fork()
    if process_id == 0:
        os.close(master_fd)
        os.environ["COLUMNS"] = "80"
        os.environ["LINES"] = "32"
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        os.close(slave_fd)
        os.execv(
            ENTRYPOINT,
            [ENTRYPOINT, "chat", "--workspace", str(workspace), "--config", str(config)],
        )
    os.close(slave_fd)
    transcript = bytearray()
    writes = ((0.15, b"/output detailed\n/permissions\n/exit\n"),)
    write_index = 0
    started_at = time.monotonic()
    try:
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            while (
                write_index < len(writes)
                and time.monotonic() - started_at >= writes[write_index][0]
            ):
                os.write(master_fd, writes[write_index][1])
                write_index += 1
            readable, _, _ = select.select([master_fd], [], [], 0.05)
            if readable:
                try:
                    transcript.extend(os.read(master_fd, 65_536))
                except OSError:
                    break
            complete_id, _ = os.waitpid(process_id, os.WNOHANG)
            if complete_id == process_id:
                break
        else:
            os.kill(process_id, 9)
            os.waitpid(process_id, 0)
            raise RuntimeError("installed CLI timed out")
    finally:
        os.close(master_fd)
    return transcript.decode("utf-8", errors="replace")


def main() -> None:
    if ENTRYPOINT is None:
        raise RuntimeError("The installed mini-agent entry point is unavailable")
    workspace = CACHE / "installed_entrypoint"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    config = workspace / "config.toml"
    config.write_text(
        """[model]
model = "installed-cli-no-request"
base_url = "http://127.0.0.1:9/v1"
context_window = 100000
max_output_tokens = 1024

[runtime]
data_dir = "data"
""",
        encoding="utf-8",
    )
    transcript = _run_chat(workspace, config)
    (CACHE / "installed_entrypoint.ansi").write_text(transcript, encoding="utf-8")
    required = (
        "Output: detailed.",
        "Workspace reads and search: automatic after path and content checks.",
        "session not saved",
    )
    missing = [item for item in required if item not in transcript]
    print("installed CLI: PASS" if not missing else "installed CLI: MISSING " + ", ".join(missing))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
