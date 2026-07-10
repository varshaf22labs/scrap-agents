from __future__ import annotations

from pathlib import Path

from agents.context import AgentContext
from models import ExecutionResult
from tools.sandbox import run_python_script


class ExecutionAgent:
    def __init__(self, timeout_seconds: int = 180):
        self.timeout_seconds = timeout_seconds

    def run(self, context: AgentContext) -> ExecutionResult:
        if not context.generated_script_path:
            raise ValueError("Generated script path is required before execution")
        output_path = self._normalize_output_path(context.plan.output_path if context.plan else None)
        if output_path:
            Path(output_path).unlink(missing_ok=True)
        args = ["--output", output_path] if output_path else []
        result = run_python_script(context.generated_script_path, args=args, cwd=context.root_dir, timeout_seconds=self.timeout_seconds)
        context.output_path = Path(output_path) if output_path else None
        return result

    def _normalize_output_path(self, value: str | None) -> str | None:
        if not value:
            return None
        path = Path(value)
        if path.suffix.lower() != ".jsonl":
            path = path.with_suffix(".jsonl")
        return str(path)
