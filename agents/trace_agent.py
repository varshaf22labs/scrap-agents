from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_client import LLMClient
from models import to_jsonable
from trace_utils import TraceRecorder


class TraceAgent:
    def __init__(self, domain: str, trace_dir: Path, llm_client: LLMClient | None = None):
        self.recorder = TraceRecorder(domain=domain, trace_dir=trace_dir)
        self.llm_client = llm_client

    def record(self, actor: str, action: str, input_data: Any, output_data: Any, note: str | None = None) -> None:
        self.recorder.record(actor, action, input_data, output_data, note=note)

    def save(self) -> Path:
        summary: str | None = None
        if self.llm_client and self.recorder.events:
            summary = self._build_summary()
        trace_path = self.recorder.save()
        if summary:
            summary_path = trace_path.with_suffix(".md")
            summary_path.write_text(summary, encoding="utf-8")
        return trace_path

    def _build_summary(self) -> str:
        events = to_jsonable(self.recorder.events)
        prompt = (
            "Summarize this agent trace for a developer. Keep it concise, factual, and structured.\n"
            "Include the main decisions, tools used, and any repair loop outcomes.\n"
            f"Trace events: {json.dumps(events)[:12000]}"
        )
        try:
            response = self.llm_client.chat(
                [
                    {"role": "system", "content": "Return a short markdown summary."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            return response.strip()
        except Exception:
            return "# Trace Summary\nSummary generation failed.\n"
