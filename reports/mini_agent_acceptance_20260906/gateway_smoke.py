"""Make one bounded, read-only request through the installed local gateway."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ENTRYPOINT = shutil.which("mini-agent")
CACHE = Path(__file__).resolve().parent / "cache"


def main() -> None:
    if ENTRYPOINT is None:
        raise RuntimeError("The installed mini-agent entry point is unavailable")
    workspace = CACHE / "gateway_smoke"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    (workspace / "fixture.txt").write_text("SMOKE_VALUE=river-stone\n", encoding="utf-8")
    config = workspace / "config.toml"
    config.write_text(
        """[model]
model = "cfuse/DeepSeek-V4-Flash-0731"
base_url = "http://127.0.0.1:4000/v1"
api_key_env = "LITELLM_API_KEY"
context_window = 128000
max_output_tokens = 512

[runtime]
data_dir = "data"
provider_retry_count = 0
retry_backoff_seconds = 0
max_model_calls_per_turn = 3
""",
        encoding="utf-8",
    )
    command = [
        ENTRYPOINT,
        "run",
        "Read fixture.txt with read_file. Do not call any other tool. Reply exactly: SMOKE_OK",
        "--workspace",
        str(workspace),
        "--config",
        str(config),
    ]
    environment = os.environ.copy()
    if not environment.get("LITELLM_API_KEY"):
        raise RuntimeError("LITELLM_API_KEY is unavailable")
    result = subprocess.run(
        command,
        cwd=workspace,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
    )
    (CACHE / "gateway_smoke.txt").write_text(result.stdout, encoding="utf-8")
    passed = result.returncode == 0 and "SMOKE_OK" in result.stdout
    print(f"gateway smoke: {'PASS' if passed else 'FAILED'} (exit={result.returncode})")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
