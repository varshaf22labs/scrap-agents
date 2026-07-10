from __future__ import annotations

from typing import Any
import warnings

import requests
from urllib3.exceptions import InsecureRequestWarning

from agents.context import AgentContext
from models import CandidatePage
from site_utils import build_common_urls, origin_from_domain, normalize_domain
from tools.firecrawl import FirecrawlTool, FirecrawlToolError
from tools.search import SearchTool, SearchToolError


class DiscoveryAgent:
    def __init__(
        self,
        search_tool: SearchTool | None = None,
        firecrawl_tool: FirecrawlTool | None = None,
        timeout_seconds: int = 30,
    ):
        self.search_tool = search_tool
        self.firecrawl_tool = firecrawl_tool
        self.timeout_seconds = timeout_seconds

    def run(self, context: AgentContext) -> list[CandidatePage]:
        candidates: list[CandidatePage] = []
        origin = origin_from_domain(context.domain)

        # Prefer Firecrawl when it is configured. It is the best signal we have
        # for actual page content, and it should not be blocked behind search
        # provider availability.
        candidates.extend(self._probe_candidates(context.domain))

        search_results = self._search_candidates(context.domain)
        for result in search_results:
            url = result.get("url")
            if url:
                candidates.append(
                    CandidatePage(
                        url=url,
                        source=result.get("source", "search"),
                        title=result.get("title"),
                        snippet=result.get("snippet"),
                        evidence={"search": result},
                    )
                )

        for url in [origin, *build_common_urls(context.domain)]:
            candidates.append(CandidatePage(url=url, source="heuristic"))

        deduped = self._dedupe_candidates(candidates)
        context.candidate_pages = deduped
        return deduped

    def _search_candidates(self, domain: str) -> list[dict[str, Any]]:
        if not self.search_tool:
            return []
        query = f"site:{normalize_domain(domain)} careers jobs India"
        try:
            return self.search_tool.search(query, num_results=8)
        except (SearchToolError, Exception):
            return []

    def _probe_candidates(self, domain: str) -> list[CandidatePage]:
        pages: list[CandidatePage] = []
        for url in build_common_urls(domain):
            if self.firecrawl_tool:
                page = self._probe_with_firecrawl(url)
                if page:
                    pages.append(page)
                    continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InsecureRequestWarning)
                    response = requests.get(
                        url,
                        timeout=(5, self.timeout_seconds),
                        allow_redirects=True,
                        verify=False,
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
                if response.status_code < 400:
                    pages.append(
                        CandidatePage(
                            url=url,
                            source="probe",
                            status_code=response.status_code,
                            final_url=response.url,
                            title=self._extract_title(response.text),
                            snippet=response.text[:300],
                            evidence={"status_code": response.status_code},
                        )
                    )
            except requests.RequestException:
                continue
        return pages

    def _probe_with_firecrawl(self, url: str) -> CandidatePage | None:
        try:
            data = self.firecrawl_tool.fetch(url)
        except FirecrawlToolError:
            return None
        except Exception:
            return None

        normalized_url = (
            data.get("url")
            or data.get("metadata", {}).get("sourceURL")
            or data.get("metadata", {}).get("sourceUrl")
            or url
        )
        title = (
            data.get("metadata", {}).get("title")
            or data.get("title")
            or data.get("metadata", {}).get("ogTitle")
            or self._extract_title(str(data.get("html") or data.get("markdown") or ""))
        )
        html = str(data.get("html") or "")
        markdown = str(data.get("markdown") or "")
        snippet_source = html or markdown
        return CandidatePage(
            url=url,
            source="firecrawl",
            status_code=200,
            final_url=normalized_url,
            title=title,
            snippet=snippet_source[:300],
            evidence={"firecrawl": {"raw_keys": sorted(data.keys())[:20]}},
        )

    def _dedupe_candidates(self, pages: list[CandidatePage]) -> list[CandidatePage]:
        seen: set[str] = set()
        deduped: list[CandidatePage] = []
        for page in pages:
            key = page.final_url or page.url
            if key in seen:
                continue
            seen.add(key)
            deduped.append(page)
        return deduped

    def _extract_title(self, html: str) -> str | None:
        lower = html.lower()
        start = lower.find("<title>")
        end = lower.find("</title>")
        if start == -1 or end == -1 or end <= start:
            return None
        return html[start + 7 : end].strip()
