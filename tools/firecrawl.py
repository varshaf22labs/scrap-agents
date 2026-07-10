from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from trace_utils import TraceRecorder


class FirecrawlToolError(RuntimeError):
    pass


@dataclass
class FirecrawlTool:
    api_key: str | None = None
    base_url: str = "https://api.firecrawl.dev/v1"
    timeout_seconds: int = 45
    trace_recorder: TraceRecorder | None = None

    def fetch(self, url: str, mode: str = "scrape") -> dict[str, Any]:
        if not self.api_key:
            raise FirecrawlToolError("FIRECRAWL_API_KEY is required")
        response = requests.post(
            f"{self.base_url.rstrip('/')}/{mode}",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"url": url, "formats": ["markdown", "html"]},
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise FirecrawlToolError(f"Firecrawl error: {response.status_code} {response.text[:500]}")
        data = response.json()
        if self.trace_recorder:
            self.trace_recorder.record(
                "firecrawl",
                "fetch",
                {"mode": mode, "url": url},
                {"raw_keys": sorted(data.keys()), "sample": data},
                note="Firecrawl fetch call",
            )
        return data
