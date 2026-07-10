from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from trace_utils import TraceRecorder


class SearchToolError(RuntimeError):
    pass


@dataclass
class SearchTool:
    provider: str
    api_key: str | None = None
    timeout_seconds: int = 30
    trace_recorder: TraceRecorder | None = None

    def search(self, query: str, num_results: int = 10) -> list[dict[str, Any]]:
        provider = self.provider.lower()
        if provider == "serper":
            results = self._search_serper(query, num_results)
        elif provider == "serpapi":
            results = self._search_serpapi(query, num_results)
        else:
            raise SearchToolError(f"Unsupported search provider: {self.provider}")
        if self.trace_recorder:
            self.trace_recorder.record(
                "search",
                "query",
                {"provider": self.provider, "query": query, "num_results": num_results},
                {"result_count": len(results), "results": results[:10]},
                note="Web search call",
            )
        return results

    def _search_serper(self, query: str, num_results: int) -> list[dict[str, Any]]:
        if not self.api_key:
            raise SearchToolError("SERPER_API_KEY is required for serper search")
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise SearchToolError(f"Serper error: {response.status_code} {response.text[:500]}")
        data = response.json()
        results = data.get("organic", []) or data.get("results", [])
        return [
            {
                "title": item.get("title"),
                "url": item.get("link") or item.get("url"),
                "snippet": item.get("snippet"),
                "source": "serper",
            }
            for item in results
        ]

    def _search_serpapi(self, query: str, num_results: int) -> list[dict[str, Any]]:
        if not self.api_key:
            raise SearchToolError("SERPAPI_API_KEY is required for serpapi search")
        response = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": "google", "q": query, "num": num_results, "api_key": self.api_key},
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise SearchToolError(f"SerpApi error: {response.status_code} {response.text[:500]}")
        data = response.json()
        results = data.get("organic_results", [])
        return [
            {
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                "source": "serpapi",
            }
            for item in results
        ]
