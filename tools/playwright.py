from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

from tools.api_detector import detect_api_candidates
from tools.html_parser import extract_links
from tools.json_detector import detect_embedded_json
from trace_utils import TraceRecorder


@dataclass
class PageSnapshot:
    url: str
    final_url: str
    status_code: int | None
    title: str | None
    html: str
    text: str | None = None
    signals: dict[str, Any] | None = None


class PlaywrightToolError(RuntimeError):
    pass


def _looks_like_api_url(url: str) -> bool:
    lowered = url.lower()
    return any(token in lowered for token in ["/api/", "graphql", "jobs", "career", "recruit", "position", "vacanc"])


def _unique_dicts(items: list[dict[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for item in items:
        key = tuple(item.get(field) for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _css_attr_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", " ")


def _pagination_hints(html: str, base_url: str) -> list[dict[str, Any]]:
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("beautifulsoup4 is required for pagination discovery") from exc

    soup = BeautifulSoup(html, "html.parser")
    hints: list[dict[str, Any]] = []
    for anchor in soup.find_all(["a", "button"]):
        text = " ".join(anchor.get_text(" ", strip=True).split()).lower()
        href = anchor.get("href")
        rel = anchor.get("rel") or []
        selector = None
        classes = anchor.get("class") or []
        if href:
            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)
            query_keys = sorted(k for k in ["page", "offset", "cursor", "start"] if k in parsed.query.lower())
            if query_keys or any(token in absolute.lower() for token in ["page=", "offset=", "cursor=", "next", "more"]):
                hints.append(
                    {
                        "type": "link",
                        "text": text[:120] or None,
                        "href": absolute,
                        "rel": list(rel) if isinstance(rel, (list, tuple)) else [str(rel)] if rel else [],
                        "selector": selector,
                    }
                )
        if any(token in text for token in ["next", "more", "older", "load more", "show more", "page 2"]):
            selector = anchor.name
            if classes:
                selector += "".join(f'[class~="{_css_attr_value(str(part))}"]' for part in classes[:2] if part)
            hints.append(
                {
                    "type": "button",
                    "text": text[:120] or None,
                    "selector": selector,
                    "href": urljoin(base_url, href) if href else None,
                }
            )
    return _unique_dicts(hints, ("type", "text", "href", "selector"))


@dataclass
class PlaywrightTool:
    timeout_seconds: int = 30
    trace_recorder: TraceRecorder | None = None

    def fetch(self, url: str, wait_for: str | None = None, click_selector: str | None = None) -> PageSnapshot:
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # pragma: no cover - optional dependency
            raise PlaywrightToolError(
                "playwright is not installed. Add it to requirements and install browsers."
            ) from exc

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                network_requests: list[dict[str, Any]] = []
                network_responses: list[dict[str, Any]] = []

                def on_request(request: Any) -> None:
                    if len(network_requests) >= 100:
                        return
                    network_requests.append(
                        {
                            "url": request.url,
                            "method": request.method,
                            "resource_type": request.resource_type,
                        }
                    )

                def on_response(response: Any) -> None:
                    if len(network_responses) >= 100:
                        return
                    headers = response.headers or {}
                    content_type = str(headers.get("content-type") or headers.get("Content-Type") or "").lower()
                    network_responses.append(
                        {
                            "url": response.url,
                            "status": response.status,
                            "content_type": content_type or None,
                        }
                    )

                page.on("request", on_request)
                page.on("response", on_response)
                response = page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_seconds * 1000)
                if click_selector:
                    page.click(click_selector)
                if wait_for:
                    page.wait_for_selector(wait_for, timeout=self.timeout_seconds * 1000)
                page.wait_for_load_state("networkidle", timeout=self.timeout_seconds * 1000)
                html = page.content()
                embedded_json = detect_embedded_json(html)
                api_candidates = _unique_dicts(
                    [
                        *detect_api_candidates(html, page.url),
                        *[
                            {"kind": "network", "url": item["url"], "status": item["status"]}
                            for item in network_responses
                            if (
                                (item.get("content_type") and "json" in str(item["content_type"]).lower())
                                or _looks_like_api_url(str(item.get("url") or ""))
                            )
                        ],
                    ],
                    ("kind", "url", "status", "token"),
                )
                pagination_hints = _pagination_hints(html, page.url)
                snapshot = PageSnapshot(
                    url=url,
                    final_url=page.url,
                    status_code=response.status if response else None,
                    title=page.title(),
                    html=html,
                    text=page.locator("body").inner_text() if page.locator("body").count() else None,
                    signals={
                        "network_requests": network_requests,
                        "network_responses": network_responses,
                        "embedded_json": embedded_json,
                        "api_candidates": api_candidates,
                        "pagination_hints": pagination_hints,
                        "sample_links": extract_links(html, page.url)[:50],
                    },
                )
                if self.trace_recorder:
                    self.trace_recorder.record(
                        "playwright",
                        "fetch",
                        {"url": url, "wait_for": wait_for, "click_selector": click_selector},
                        {
                            "final_url": snapshot.final_url,
                            "status_code": snapshot.status_code,
                            "title": snapshot.title,
                            "signals": snapshot.signals,
                        },
                        note="Browser fetch call",
                    )
                return snapshot
            finally:
                browser.close()
