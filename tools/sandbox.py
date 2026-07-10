from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models import ExecutionResult


@dataclass
class SandboxConfig:
    timeout_seconds: int = 180


def run_python_script(script_path: str | Path, args: list[str] | None = None, cwd: str | Path | None = None, timeout_seconds: int = 180) -> ExecutionResult:
    start = time.time()
    script_path = Path(script_path)
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env={**os.environ},
    )
    duration = time.time() - start
    output_path: str | None = None
    for arg_index, arg in enumerate(cmd):
        if arg == "--output" and arg_index + 1 < len(cmd):
            output_path = cmd[arg_index + 1]
            break
    return ExecutionResult(
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        output_path=output_path,
        duration_seconds=duration,
    )

